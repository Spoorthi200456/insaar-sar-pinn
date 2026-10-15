"""
Composite PINN objective (Module 1):

    L = w_PDE * L_PDE + w_data * L_data + w_BC * L_BC + w_reg * L_reg

Every term is masked so that padded timesteps (from `ARSDataset` left-
padding) and low-coherence/interpolated observations never contribute
gradient signal from data they were never truly observed at, and every
sum-of-squares term is computed with a numerically stable mean-reduction
over the *valid* element count (never a raw sum that can explode with
batch/sequence size, and never a division by zero when a mask is empty).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

_EPS = 1e-8


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean squared error over `mask == True` elements only; 0 if mask is empty."""
    diff_sq = (pred - target) ** 2
    if mask is None:
        return diff_sq.mean()
    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(_EPS)
    return (diff_sq * mask_f).sum() / denom


def _weighted_laplacian_divergence(
    conductivity: torch.Tensor, head: torch.Tensor, adjacency: torch.Tensor
) -> torch.Tensor:
    """
    Discrete graph approximation of ∇·(K∇h) at each node/timestep:

        sum_j a_ij * (K_i + K_j)/2 * (h_j - h_i)

    conductivity: (B,)      K_i, shared across time (steady material property)
    head:         (B, T)    h_i(t)   (here: predicted head-change series)
    adjacency:    (B, B)    a_ij connectivity weights
    returns:      (B, T)    per-node, per-timestep divergence estimate
    """
    b, t = head.shape
    k_i = conductivity.view(b, 1)              # (B, 1)
    k_avg = 0.5 * (k_i + k_i.T)                 # (B, B), (K_i + K_j) / 2
    weighted_adj = adjacency * k_avg            # (B, B)

    # sum_j a_ij*(K_i+K_j)/2 * h_j(t)  -  h_i(t) * sum_j a_ij*(K_i+K_j)/2
    neighbor_term = torch.einsum("ij,jt->it", weighted_adj, head)      # (B, T)
    self_term = head * weighted_adj.sum(dim=1, keepdim=True)           # (B, T)
    return neighbor_term - self_term


@dataclass
class LossWeights:
    pde: float = 1.0
    data: float = 1.0
    boundary: float = 0.5
    design_capacity: float = 2.0
    non_negativity: float = 5.0
    mass_balance: float = 1.0
    conductivity_smoothness: float = 0.1


@dataclass
class LossBreakdown:
    total: torch.Tensor
    pde: torch.Tensor
    data: torch.Tensor
    boundary: torch.Tensor
    design_capacity: torch.Tensor
    non_negativity: torch.Tensor
    mass_balance: torch.Tensor
    conductivity_smoothness: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "pde_residual": float(self.pde.detach().cpu()),
            "data_misfit": float(self.data.detach().cpu()),
            "boundary_condition": float(self.boundary.detach().cpu()),
            "design_capacity_penalty": float(self.design_capacity.detach().cpu()),
            "non_negativity_penalty": float(self.non_negativity.detach().cpu()),
            "mass_balance_penalty": float(self.mass_balance.detach().cpu()),
            "conductivity_smoothness": float(self.conductivity_smoothness.detach().cpu()),
        }


class CompositeARSLoss(nn.Module):
    """
    Full physics-informed composite loss described in Module 1, expanded
    into individually-testable sub-terms and recombined with configurable
    weights. All sub-terms are non-negative by construction (squared
    residuals or hinge penalties), so `total` is always >= 0.
    """

    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()

    def forward(
        self,
        predicted_displacement_cm: torch.Tensor,   # (B, T)  d_pred_i(t)
        observed_displacement_cm: torch.Tensor,     # (B, T)  d_obs_i(t)
        predicted_head_change: torch.Tensor,        # (B, T)  Δh_i(t)
        hydraulic_conductivity: torch.Tensor,        # (B,)    K_i
        recharge_rate: torch.Tensor,                 # (B, T)  Q_i(t)
        design_recharge_capacity: torch.Tensor,      # (B,)    Q_i,design
        rainfall_mm: torch.Tensor,                    # (B, T)  R_i(t) (mass-balance forcing)
        adjacency: torch.Tensor,                      # (B, B)
        observation_mask: torch.Tensor | None = None,      # (B, T) bool, True == reliable
        boundary_head: torch.Tensor | None = None,          # (B, T) known head, if any
        boundary_mask: torch.Tensor | None = None,           # (B, T) bool, True where boundary_head is known
        initial_head: torch.Tensor | None = None,             # (B,)  h_i(0) if known
    ) -> LossBreakdown:
        w = self.weights

        # --- PDE residual loss ---------------------------------------- #
        # storage term: finite difference of predicted head-change series.
        storage_term = torch.zeros_like(predicted_head_change)
        storage_term[:, 1:] = predicted_head_change[:, 1:] - predicted_head_change[:, :-1]

        divergence = _weighted_laplacian_divergence(
            hydraulic_conductivity, predicted_head_change, adjacency
        )
        pde_residual = divergence - storage_term - recharge_rate
        pde_residual = torch.clamp(pde_residual, min=-1e4, max=1e4)  # gradient-explosion guard
        pde_loss = _masked_mse(pde_residual, torch.zeros_like(pde_residual), observation_mask)

        # --- Data misfit loss ------------------------------------------ #
        data_loss = _masked_mse(predicted_displacement_cm, observed_displacement_cm, observation_mask)

        # --- Boundary / initial condition loss -------------------------- #
        boundary_loss = torch.zeros((), device=predicted_displacement_cm.device)
        if boundary_head is not None and boundary_mask is not None:
            boundary_loss = boundary_loss + _masked_mse(
                predicted_head_change, boundary_head, boundary_mask
            )
        if initial_head is not None:
            boundary_loss = boundary_loss + ((predicted_head_change[:, 0] - initial_head) ** 2).mean()

        # --- Domain-specific regularizers -------------------------------- #
        design_capacity_penalty = torch.clamp(
            recharge_rate - design_recharge_capacity.unsqueeze(-1), min=0.0
        ) ** 2
        design_capacity_loss = design_capacity_penalty.mean()

        non_negativity_penalty = torch.clamp(-recharge_rate, min=0.0) ** 2
        non_negativity_loss = non_negativity_penalty.mean()

        cumulative_recharge = recharge_rate.sum(dim=1)
        cumulative_rainfall = rainfall_mm.sum(dim=1)
        mass_balance_loss = (
            torch.clamp(cumulative_recharge - cumulative_rainfall, min=0.0) ** 2
        ).mean()

        # Spatial smoothness of K via the same graph Laplacian, applied to
        # the (per-node, time-invariant) conductivity field.
        k_diff = hydraulic_conductivity.unsqueeze(0) - hydraulic_conductivity.unsqueeze(1)  # (B,B)
        smoothness_loss = (adjacency * k_diff ** 2).sum() / adjacency.sum().clamp_min(_EPS)

        total = (
            w.pde * pde_loss
            + w.data * data_loss
            + w.boundary * boundary_loss
            + w.design_capacity * design_capacity_loss
            + w.non_negativity * non_negativity_loss
            + w.mass_balance * mass_balance_loss
            + w.conductivity_smoothness * smoothness_loss
        )

        return LossBreakdown(
            total=total,
            pde=pde_loss,
            data=data_loss,
            boundary=boundary_loss,
            design_capacity=design_capacity_loss,
            non_negativity=non_negativity_loss,
            mass_balance=mass_balance_loss,
            conductivity_smoothness=smoothness_loss,
        )
