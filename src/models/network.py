"""
ARSNet: the complete multi-branch architecture from Module 2.

    v_i^(0) = [f_cv_i ; f_des_i]                (spatial node features)
    v_i^(L) = HydrologicGATStack(v^(0), A)       (aquifer connectivity)
    z_i     = TemporalAlignmentTransformer(d_i(t), R_i(t), Δt_t)
    u_i     = [v_i^(L) ; z_i]                    (fused representation)
    K_i     = exp(w_K^T u_i + b_K)               (positivity-constrained)
    Q_i(t)  = RechargeDecoder(u_i, R_i(t))        (non-negative via softplus)
    d_pred_i(t) = -alpha_i * cumulative_head_proxy(Q_i(t), K_i)

The last line operationalizes the poroelastic coupling
`Δd_i(t) ≈ -alpha * Δh_i(t)` from Module 1: instead of requiring a full PDE
solve inside the forward pass, we let the recharge decoder implicitly model
head *changes* and pass them through a learned, K-modulated integrator to a
predicted head-change series, then couple to displacement via `alpha`. This
keeps `d_pred` differentiable end-to-end w.r.t. both `K` and `Q`, which is
exactly what `losses.py::DataMisfitLoss` and `losses.py::PDEResidualLoss`
need to backpropagate through.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.layers import (
    CVEncoder,
    HydrologicGATStack,
    RechargeDecoder,
    TemporalAlignmentTransformer,
)


@dataclass
class ARSNetOutput:
    hydraulic_conductivity: torch.Tensor      # (B,)  K_i
    recharge_rate: torch.Tensor               # (B, T) Q_i(t)
    predicted_head_change: torch.Tensor       # (B, T) Δh_i(t) proxy
    predicted_displacement_cm: torch.Tensor   # (B, T) d_pred_i(t)
    node_embedding: torch.Tensor              # (B, H) v_i^(L), for explainability
    temporal_embedding: torch.Tensor          # (B, D) z_i, for explainability


class HeadChangeIntegrator(nn.Module):
    """
    Learned, K-modulated causal integrator mapping recharge Q_i(t) to a
    predicted head-change series Δh_i(t): higher conductivity K discharges
    accumulated head faster (a learned analogue of specific storage S_s in
    the discretized PDE, `S_s ∂h/∂t = ∇·(K∇h) + Q`). Implemented as a
    depthwise causal 1D convolution over log(1+K)-scaled cumulative Q, kept
    lightweight and fully differentiable so gradients reach both K and Q
    from the displacement data-misfit term.
    """

    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.decay_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid -> (0, 1) decay rate

    def forward(self, recharge: torch.Tensor, conductivity: torch.Tensor) -> torch.Tensor:
        """
        recharge:     (B, T) non-negative Q_i(t)
        conductivity: (B,)   positive K_i
        returns:      (B, T) Δh_i(t)
        """
        decay = torch.sigmoid(self.decay_logit)  # scalar in (0, 1)
        k_scale = 1.0 / (1.0 + conductivity)     # higher K -> faster dissipation -> smaller head buildup
        b, t = recharge.shape
        head = torch.zeros(b, device=recharge.device, dtype=recharge.dtype)
        outputs = []
        for step in range(t):
            head = decay * head + k_scale * recharge[:, step]
            outputs.append(head.unsqueeze(1))
        return torch.cat(outputs, dim=1)


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
        self.head_integrator = HeadChangeIntegrator()

        self.fused_dim = fused_dim

    def forward(
        self,
        design_features: torch.Tensor,       # (B, design_feature_dim)
        displacement_cm: torch.Tensor,       # (B, T)
        rainfall_mm: torch.Tensor,           # (B, T)
        time_delta_days: torch.Tensor,       # (B, T)
        adjacency: torch.Tensor,             # (B, B) or (1, B, B)
        biot_alpha: torch.Tensor,            # (B,)
        coherence_mask: torch.Tensor | None = None,  # (B, T) bool, True == reliable
        cv_patch: torch.Tensor | None = None,         # (B, C, H, W)
    ) -> ARSNetOutput:
        b = design_features.shape[0]

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

        q = self.recharge_decoder(u, rainfall_mm)  # (B, T), non-negative

        delta_h = self.head_integrator(q, k)  # (B, T)
        d_pred = -biot_alpha.unsqueeze(-1) * delta_h  # (B, T)

        return ARSNetOutput(
            hydraulic_conductivity=k,
            recharge_rate=q,
            predicted_head_change=delta_h,
            predicted_displacement_cm=d_pred,
            node_embedding=v_l,
            temporal_embedding=z,
        )
