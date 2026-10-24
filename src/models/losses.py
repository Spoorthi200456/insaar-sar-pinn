"""
Composite PINN objective (Module 1):

    L = w_PDE * L_PDE + w_data * L_data + w_BC * L_BC + w_reg * L_reg

Masking (docs/math/masking.md): the data term uses only reliable
observations (coherence, quality flag, not interpolated, not padding); the
PDE term uses every real step (Δt_k > 0 and Δt_{k-1} > 0), whatever the
coherence (review finding 8); only seed nodes of a graph batch contribute
(finding 4). Sum-of-squares terms are means over the valid element count
(no raw sums, no division by zero when a mask is empty).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

_EPS = 1e-8

# Fraction of rain on the catchment that runs off into the structure
# (PROVISIONAL, docs/open_questions.md OQ10). Rational-method runoff
# coefficients for Indian semi-arid catchments are typically 0.1-0.4.
RUNOFF_COEFFICIENT = 0.3
MM_PER_M = 1000.0
# Volume floor for the mass-balance ratio: 1 mm of water over the recharge
# area, so a dry series (no inflow) still gives a finite, dimensionless ratio.
_MASS_BALANCE_FLOOR_M = 1e-3


def _masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
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
    Discrete graph approximation of ∇·(T∇h) at each node/timestep:

        sum_j a_ij * (T_i + T_j)/2 * (h_j - h_i)

    conductivity: (B,)      T_i [m²/day] transmissivity (form (b)), shared across time
    head:         (B, T)    h_i(t) [m]   (here: predicted head-change series)
    adjacency:    (B, B)    a_ij connectivity weights. For the result to be in
                            m/day, a_ij must carry 1/m²; the current
                            exp(-d/r) weights are dimensionless (open, review
                            finding 4).
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


def pde_residual(
    head_m: torch.Tensor,
    time_delta_days: torch.Tensor,
    storativity: torch.Tensor,
    transmissivity: torch.Tensor,
    recharge_m_per_day: torch.Tensor,
    adjacency: torch.Tensor,
    leakance_per_day: torch.Tensor | float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Residual of the depth-integrated flow equation with Hantush leakage
    (form (b), provisional; review finding 11, docs/math/flow_equation_leakage.md):

        S ∂h/∂t = ∇·(T∇h) − L (h − h_ref) + R,   h_ref = 0
        r = S ∂h/∂t − ∇·(T∇h) + L h − R          every term in m/day

    head_m:             (B, T)  h [m]
    time_delta_days:    (B, T)  Δt_k = t_k − t_{k−1} [day]; 0 on padded steps
    storativity:        scalar or (B,)  S [dimensionless]
    transmissivity:     (B,)    T [m²/day]
    recharge_m_per_day: (B, T)  R [m/day], R > 0 = water INTO the aquifer
    adjacency:          (B, B)  a_ij [1/m²]
    leakance_per_day:   scalar or (B,)  L = K'/b' [1/day]; L·h [m/day] is the
                        drainage to the regional aquifer. 0 = no sink (old
                        equation); ARSNet always supplies its learned L.

    ∂h/∂t uses the backward difference (h_k − h_{k−1}) / Δt_k, which is only
    defined when both k and k−1 are real acquisitions (Δt > 0). Step 0, padded
    steps and the first real step after padding are returned as invalid.

    returns: (residual (B, T), valid (B, T) bool)
    """
    dt = time_delta_days
    valid = torch.zeros_like(head_m, dtype=torch.bool)
    valid[:, 1:] = (dt[:, 1:] > 0) & (dt[:, :-1] > 0)
    dt_safe = torch.where(valid, dt, torch.ones_like(dt))  # never divide by 0, even where masked

    dh_dt = torch.zeros_like(head_m)
    dh_dt[:, 1:] = (head_m[:, 1:] - head_m[:, :-1]) / dt_safe[:, 1:]
    dh_dt = torch.where(valid, dh_dt, torch.zeros_like(dh_dt))

    s = storativity.reshape(-1, 1) if storativity.dim() > 0 else storativity
    storage_term = s * dh_dt
    divergence = _weighted_laplacian_divergence(transmissivity, head_m, adjacency)
    leak = torch.as_tensor(leakance_per_day, dtype=head_m.dtype, device=head_m.device)
    leak = leak.reshape(-1, 1) if leak.dim() > 0 else leak
    leakage_term = leak * head_m  # implicit in h_k, consistent with the backward difference
    return storage_term - divergence + leakage_term - recharge_m_per_day, valid


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

    def __init__(
        self, weights: LossWeights | None = None, runoff_coefficient: float = RUNOFF_COEFFICIENT
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.runoff_coefficient = runoff_coefficient

    def forward(
        self,
        predicted_displacement_cm: torch.Tensor,   # (B, T)  d_pred_i(t)
        observed_displacement_cm: torch.Tensor,     # (B, T)  d_obs_i(t)
        predicted_head_change: torch.Tensor,        # (B, T)  h_i(t) [m]
        hydraulic_conductivity: torch.Tensor,        # (B,)    T_i [m²/day] (form (b))
        recharge_rate: torch.Tensor,                 # (B, T)  R_i(t) [m/day]
        design_recharge_capacity: torch.Tensor,      # (B,)    Q_i,design [m³/day]
        rainfall_mm: torch.Tensor,                    # (B, T)  rain per InSAR interval [mm]
        adjacency: torch.Tensor,                      # (B, B)
        time_delta_days: torch.Tensor,                # (B, T)  Δt_k [day], 0 on padding
        storativity: torch.Tensor,                    # scalar  S [dimensionless]
        recharge_area_m2: torch.Tensor,               # (B,)    A, structure recharge area [m²]
        catchment_area_m2: torch.Tensor,              # (B,)    A_catch [m²]
        leakance: torch.Tensor | float = 0.0,         # scalar  L [1/day] (finding 11)
        observation_mask: torch.Tensor | None = None,      # (B, T) bool, True == reliable
        boundary_head: torch.Tensor | None = None,          # (B, T) known head, if any
        boundary_mask: torch.Tensor | None = None,           # (B, T) bool, True where known
        initial_head: torch.Tensor | None = None,             # (B,)  h_i(0) if known
        node_mask: torch.Tensor | None = None,  # (B,) bool, True == seed node (finding 4)
        valid_mask: torch.Tensor | None = None,  # (B, T) bool, True == real acquisition
    ) -> LossBreakdown:
        w = self.weights
        b, t = predicted_displacement_cm.shape
        # Only seed nodes contribute (finding 4): neighbour-only nodes are in the
        # batch to give seeds exact neighbourhoods; their own are truncated.
        if node_mask is None:
            node_mask = torch.ones(b, dtype=torch.bool, device=predicted_displacement_cm.device)
        node_rows = node_mask.unsqueeze(-1).expand(b, t)
        data_mask = node_rows if observation_mask is None else observation_mask & node_rows
        node_f = node_mask.float()
        n_nodes = node_f.sum().clamp_min(1.0)
        # Real acquisitions, independent of coherence (finding 7). Padded steps
        # have Δt = 0; pass `valid_mask` explicitly for e.g. 1-step series.
        if valid_mask is None:
            valid_mask = time_delta_days > 0
        valid_f = valid_mask.float()
        n_valid_steps = valid_f.sum(dim=1).clamp_min(1.0)  # (B,)

        def _seed_mean_over_valid_steps(x: torch.Tensor) -> torch.Tensor:
            per_node = (x * valid_f).sum(dim=1) / n_valid_steps
            return (per_node * node_f).sum() / n_nodes

        # --- PDE residual loss ---------------------------------------- #
        # r = S ∂h/∂t − ∇·(T∇h) + L h − R  [m/day]; see `pde_residual`.
        residual, pde_valid = pde_residual(
            predicted_head_change,
            time_delta_days,
            storativity,
            hydraulic_conductivity,
            recharge_rate,
            adjacency,
            leakance,
        )
        # Review finding 8: physics applies at every real step of a seed node,
        # including low-coherence / interpolated InSAR steps (where it is the
        # only constraint), so the mask is validity only, never coherence. No
        # hard clamp: it zeroed the gradient for |r| > 1e4; explosions are
        # guarded by gradient-norm clipping in the training loop instead.
        pde_mask = pde_valid & node_rows
        pde_loss = _masked_mse(residual, torch.zeros_like(residual), pde_mask)

        # --- Data misfit loss ------------------------------------------ #
        data_loss = _masked_mse(predicted_displacement_cm, observed_displacement_cm, data_mask)

        # --- Boundary / initial condition loss -------------------------- #
        boundary_loss = torch.zeros((), device=predicted_displacement_cm.device)
        if boundary_head is not None and boundary_mask is not None:
            boundary_loss = boundary_loss + _masked_mse(
                predicted_head_change, boundary_head, boundary_mask & node_rows
            )
        if initial_head is not None:
            init_sq = (predicted_head_change[:, 0] - initial_head) ** 2
            boundary_loss = boundary_loss + (init_sq * node_f).sum() / n_nodes

        # --- Domain-specific regularizers -------------------------------- #
        # Units (review finding 6, docs/math/recharge_units.md): R [m/day] is a
        # flux per unit area. R·A [m³/day] is the structure's recharge volume
        # rate, comparable with Q_design [m³/day]. Both hinges are ratios, so
        # dimensionless.
        area = recharge_area_m2.unsqueeze(-1)                      # (B, 1) m²
        volume_rate = recharge_rate * area                         # (B, T) m³/day
        capacity = design_recharge_capacity.unsqueeze(-1)          # (B, 1) m³/day
        design_capacity_penalty = torch.clamp(volume_rate / capacity - 1.0, min=0.0) ** 2
        design_capacity_loss = _seed_mean_over_valid_steps(design_capacity_penalty)

        # Inert while R comes from a softplus (R >= 0 by construction); kept as a
        # guard for decoders without that constraint.
        non_negativity_penalty = torch.clamp(-recharge_rate, min=0.0) ** 2
        non_negativity_loss = _seed_mean_over_valid_steps(non_negativity_penalty)

        # Mass balance: recharge volume Σ R·A·Δt [m³] cannot exceed the rain
        # volume that can reach the structure, Σ P/1000·(A + c_r·A_catch) [m³]
        # (direct rain on the footprint + runoff from the catchment).
        recharge_volume = (volume_rate * time_delta_days * valid_f).sum(dim=1)  # (B,) m³
        contributing_area = recharge_area_m2 + self.runoff_coefficient * catchment_area_m2
        rain_m = (rainfall_mm * valid_f).sum(dim=1) / MM_PER_M
        inflow_volume = rain_m * contributing_area                               # (B,) m³
        volume_scale = inflow_volume + _MASS_BALANCE_FLOOR_M * recharge_area_m2
        mass_balance_per_node = (
            torch.clamp((recharge_volume - inflow_volume) / volume_scale, min=0.0) ** 2
        )
        mass_balance_loss = (mass_balance_per_node * node_f).sum() / n_nodes

        # Spatial smoothness of T on graph edges (finding 7): mean over edges out
        # of seed nodes of (ln T_i − ln T_j)². Dimensionless, so independent of
        # the units of T, and every edge counts once (the 1/m² stencil weight
        # sizes flow, not similarity).
        log_t = torch.log(hydraulic_conductivity.clamp_min(_EPS))
        log_diff_sq = (log_t.unsqueeze(0) - log_t.unsqueeze(1)) ** 2              # (B, B)
        edges = (adjacency > 0).float() * node_f.unsqueeze(-1)
        smoothness_loss = (edges * log_diff_sq).sum() / edges.sum().clamp_min(1.0)

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
