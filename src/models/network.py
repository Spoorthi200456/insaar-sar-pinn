"""
ARSNet: the complete multi-branch architecture from Module 2.

    v_i^(0) = [f_cv_i ; f_des_i]                (spatial node features)
    v_i^(L) = HydrologicGATStack(v^(0), A)       (aquifer connectivity)
    z_i     = TemporalAlignmentTransformer(d_i(t), R_i(t), Δt_t)
    u_i     = [v_i^(L) ; z_i]                    (fused representation)
    K_i     = exp(w_K^T u_i + b_K)               (positivity-constrained)
    Q_i(t)  = RechargeDecoder(u_i, R_i(t))        (non-negative via softplus) [m/day]
    v_i(t)  = V_SCALE * HeadRateDecoder(u_i, R_i(t))                         [m/day]
    h_i,k   = h_i,k-1 + Δt_k v_i,k,  h = 0 at the first real acquisition     [m]
    d_pred_i(t) = 100 * cos(theta_i) * S_k * (h_i(t) - h_i(t_ref))     [cm]

The last line is the poroelastic coupling of review finding 2
(`src/models/coupling.py`, derivation in docs/math/deformation_coupling.md):
a head rise lowers effective stress, the skeleton expands, the ground rises,
and uplift is projected onto the radar line of sight (positive = toward the
satellite, PROVISIONAL). S_k is the skeletal storage coefficient
(dimensionless, one global learnable value); theta_i is the per-structure
incidence angle. h is referenced to the first real acquisition because InSAR
displacement is relative to it (finding-2 plan, Q6).

The head is a separate network output tied to recharge only through the PDE
residual r = S ∂h/∂t − ∇·(T∇h) − R in the loss (a soft physics constraint, as
in a standard PINN). Its integrator uses the real Δt, so h is in metres by
construction (docs/math/head_integrator.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.coupling import head_to_los_cm
from src.models.layers import (
    CVEncoder,
    HydrologicGATStack,
    RechargeDecoder,
    TemporalAlignmentTransformer,
)


@dataclass
class ARSNetOutput:
    hydraulic_conductivity: torch.Tensor  # (B,)  K_i
    recharge_rate: torch.Tensor  # (B, T) Q_i(t)
    predicted_head_change: torch.Tensor  # (B, T) h_i(t) - h_i(t_first) [m]
    predicted_displacement_cm: torch.Tensor  # (B, T) d_pred_i(t)
    node_embedding: torch.Tensor  # (B, H) v_i^(L), for explainability
    temporal_embedding: torch.Tensor  # (B, D) z_i, for explainability
    storativity: torch.Tensor  # scalar S [dimensionless], in (0, 1)
    skeletal_storage: torch.Tensor  # scalar S_k [dimensionless], in (1e-6, 1e-2)
    head_rate: torch.Tensor  # (B, T) v_i(t) = ∂h/∂t estimate [m/day]
    leakance: torch.Tensor  # scalar L [1/day], in (1e-6, 1e-1)


# Characteristic head rate used to non-dimensionalise the head-rate decoder
# output (v = V_SCALE * network output): 1 cm/day, PROVISIONAL (OQ12). Sets
# the scale at initialisation only; units stay m/day.
HEAD_RATE_SCALE_M_PER_DAY = 0.01
# Characteristic recharge flux through a structure bed, same role for R
# (review finding 20): R = R_SCALE * softplus(.), so R starts near
# 0.01 * ln 2 ~ 7 mm/day instead of ~700 mm/day. PROVISIONAL (OQ16). Units m/day.
RECHARGE_SCALE_M_PER_DAY = 0.01


class HeadChangeIntegrator(nn.Module):
    """
    Explicit time integration with the REAL time gap (review-2 fix of the
    integrator that ignored Δt):

        h_k = h_{k-1} + Δt_k · v_k     for k with steps k and k-1 both real
        increment = 0                  at padding (Δt = 0) and the first real step

    so h = 0 at the first real acquisition, and the backward difference
    (h_k − h_{k-1})/Δt_k used by the PDE residual equals v_k exactly.
    Units: day · m/day = m. No learnable parameters.
    """

    def forward(self, head_rate: torch.Tensor, time_delta_days: torch.Tensor) -> torch.Tensor:
        """
        head_rate:       (B, T) v [m/day]
        time_delta_days: (B, T) Δt [day], 0 on padding
        returns:         (B, T) h − h(first real acquisition) [m]
        """
        dt = time_delta_days
        real = dt > 0
        step = torch.cat([torch.zeros_like(real[:, :1]), real[:, 1:] & real[:, :-1]], dim=1)
        increment = torch.where(step, dt * head_rate, torch.zeros_like(head_rate))
        return torch.cumsum(increment, dim=1)


class ARSNet(nn.Module):
    def __init__(
        self,
        design_feature_dim: int = 3,
        cv_in_channels: int | None = None,
        cv_embed_dim: int = 64,
        gat_hidden_dim: int = 32,
        gat_num_layers: int = 2,
        gat_num_heads: int = 4,
        temporal_embed_dim: int = 64,
        temporal_num_layers: int = 3,
        temporal_num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_cv_branch = cv_in_channels is not None
        if self.use_cv_branch:
            self.cv_encoder = CVEncoder(in_channels=cv_in_channels, embed_dim=cv_embed_dim)
            node_in_dim = cv_embed_dim + design_feature_dim
        else:
            self.cv_encoder = None
            node_in_dim = design_feature_dim

        self.gat_stack = HydrologicGATStack(
            in_dim=node_in_dim,
            hidden_dim=gat_hidden_dim,
            num_layers=gat_num_layers,
            num_heads=gat_num_heads,
            dropout=dropout,
        )

        self.temporal_encoder = TemporalAlignmentTransformer(
            input_dim=3,
            embed_dim=temporal_embed_dim,
            num_layers=temporal_num_layers,
            num_heads=temporal_num_heads,
            dropout=dropout,
        )

        fused_dim = self.gat_stack.output_dim + self.temporal_encoder.output_dim

        # Hydraulic Conductivity Head: K_i = exp(w_K^T u_i + b_K) for positivity.
        self.k_head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Linear(fused_dim // 2, 1),
        )

        self.recharge_decoder = RechargeDecoder(fused_dim=fused_dim, hidden_dim=gat_hidden_dim * 2)
        self.head_rate_decoder = RechargeDecoder(
            fused_dim=fused_dim, hidden_dim=gat_hidden_dim * 2, non_negative=False
        )
        self.head_integrator = HeadChangeIntegrator()

        # Storativity S (form (b), provisional): one global value, assumes a
        # single aquifer type. Sigmoid keeps 0 < S < 1 (S_y < porosity < 1).
        # Init 0.02 ~ specific yield of weathered hard rock (CGWB GEC-2015).
        # Not identifiable from the PDE term alone: (S, T, R) -> (cS, cT, cR)
        # leaves the residual unchanged, so report it as a fitted coefficient.
        self.storativity_logit = nn.Parameter(torch.logit(torch.tensor(0.02)))

        # Skeletal storage coefficient S_k (finding 2, PROVISIONAL, open question
        # OQ7): one global value, log10 S_k = -6 + 4*sigmoid(raw), so
        # 1e-6 < S_k < 1e-2 (hard rock ... compliant alluvium), init 1e-4 (raw = 0).
        # Global because per-structure S_k is not identifiable without wells.
        self.skeletal_storage_raw = nn.Parameter(torch.tensor(0.0))

        # Leakance L = K'/b' [1/day] of the Hantush sink −L·h (finding 11,
        # PROVISIONAL, OQ13): one global value, log10 L = −6 + 5*sigmoid(raw),
        # so 1e-6 < L < 1e-1 1/day (recession time S/L from days to millennia
        # for S ~ 0.02), init 1e-3 (raw = logit(0.6)).
        self.leakance_raw = nn.Parameter(torch.logit(torch.tensor(0.6)))

        self.fused_dim = fused_dim

    def forward(
        self,
        design_features: torch.Tensor,  # (B, design_feature_dim)
        displacement_cm: torch.Tensor,  # (B, T)
        rainfall_mm: torch.Tensor,  # (B, T)
        time_delta_days: torch.Tensor,  # (B, T)
        adjacency: torch.Tensor,  # (B, B) or (1, B, B)
        incidence_angle_deg: torch.Tensor,  # (B,) radar incidence angle [deg]
        coherence_mask: torch.Tensor | None = None,  # (B, T) bool, True == reliable
        cv_patch: torch.Tensor | None = None,  # (B, C, H, W)
    ) -> ARSNetOutput:
        if self.use_cv_branch:
            if cv_patch is None:
                raise ValueError("cv_in_channels was configured but no cv_patch was provided")
            f_cv = self.cv_encoder(cv_patch)
            v0 = torch.cat([f_cv, design_features], dim=-1)
        else:
            v0 = design_features

        # GAT expects a batch dimension over graphs; here the *batch itself*
        # is the single graph of B co-observed structures (per Module 2).
        adjacency_b = adjacency if adjacency.dim() == 3 else adjacency.unsqueeze(0)
        v0_b = v0.unsqueeze(0)  # (1, B, node_in_dim)
        v_l = self.gat_stack(v0_b, adjacency_b).squeeze(0)  # (B, gat_out_dim)

        key_padding_mask = None
        if coherence_mask is not None:
            # transformer expects True == "ignore this position"
            key_padding_mask = ~coherence_mask
            # guard against a fully-masked row (transformer NaNs on all-True rows)
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, -1] = False

        z = self.temporal_encoder(
            displacement_cm, rainfall_mm, time_delta_days, key_padding_mask=key_padding_mask
        )  # (B, temporal_embed_dim)

        u = torch.cat([v_l, z], dim=-1)  # (B, fused_dim)

        log_k = self.k_head(u).squeeze(-1)
        k = torch.exp(torch.clamp(log_k, min=-10.0, max=10.0))  # numerical stability guard

        q = RECHARGE_SCALE_M_PER_DAY * self.recharge_decoder(u, rainfall_mm)  # (B, T) m/day, >= 0

        v = HEAD_RATE_SCALE_M_PER_DAY * self.head_rate_decoder(u, rainfall_mm)  # (B, T) m/day
        delta_h = self.head_integrator(v, time_delta_days)  # (B, T) m

        # Reference head at the first real acquisition (first step with Δt > 0;
        # padded steps have Δt = 0). Index 0 if a row has no real step.
        first_real = (time_delta_days > 0).int().argmax(dim=1, keepdim=True)  # (B, 1)
        h_ref = torch.gather(delta_h, 1, first_real)
        s_k = self.skeletal_storage()
        d_pred = head_to_los_cm(delta_h - h_ref, s_k, incidence_angle_deg)  # (B, T) [cm]

        return ARSNetOutput(
            hydraulic_conductivity=k,
            recharge_rate=q,
            predicted_head_change=delta_h,
            predicted_displacement_cm=d_pred,
            node_embedding=v_l,
            temporal_embedding=z,
            storativity=torch.sigmoid(self.storativity_logit),
            skeletal_storage=s_k,
            head_rate=v,
            leakance=self.leakance(),
        )

    def leakance(self) -> torch.Tensor:
        """L [1/day], 1e-6 < L < 1e-1."""
        return torch.pow(10.0, -6.0 + 5.0 * torch.sigmoid(self.leakance_raw))

    def skeletal_storage(self) -> torch.Tensor:
        """S_k [dimensionless], 1e-6 < S_k < 1e-2."""
        return torch.pow(10.0, -6.0 + 4.0 * torch.sigmoid(self.skeletal_storage_raw))
