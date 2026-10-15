"""
Custom sub-layers for ARSNet (Module 2 of the blueprint):

  * `CVEncoder`            -- footprint mask + displacement-map encoder.
  * `HydrologicGATLayer`   -- multi-head graph-attention layer implementing
                               v_i^{l+1} = ReLU( sum_j alpha_ij W v_j ),
                               with alpha_ij learned attention over the
                               hydraulic-connectivity graph.
  * `TimeDeltaPositionalEncoding` -- irregular-sampling-aware positional
                               encoding using the observed inter-acquisition
                               gaps (Δt) instead of fixed integer positions.
  * `TemporalAlignmentTransformer` -- fuses rainfall + displacement series.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# CV Encoder (footprint segmentation feature extractor)
# --------------------------------------------------------------------------- #

class CVEncoder(nn.Module):
    """
    Lightweight CNN encoder over (mask, displacement-map) patches.

    Input:  x of shape (B, C, H, W) where C = 1 (binary footprint mask) +
            T_patch (a short window of the dense displacement map D_i(x,y,t)).
    Output: spatial embedding f_cv of shape (B, embed_dim).

    A full production deployment would substitute a fine-tuned
    Segment-Anything / UNet backbone for footprint extraction upstream and
    feed its pooled features here; this encoder consumes whatever CV
    front-end tensor is produced and refines it into the embedding used by
    the fusion stage.
    """

    def __init__(self, in_channels: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.net(x).flatten(1)
        return self.proj(feats)


# --------------------------------------------------------------------------- #
# Hydrologic Graph Attention Network layer
# --------------------------------------------------------------------------- #

class HydrologicGATLayer(nn.Module):
    """
    Single multi-head graph-attention layer over the ARS connectivity graph.

    For node i, with neighbor set N(i) (edges weighted by prior hydraulic
    connectivity a_ij, e.g. geographic proximity or shared aquifer units):

        e_ij   = LeakyReLU( a^T [W v_i || W v_j] )              (per head)
        alpha_ij = softmax_j( e_ij ) masked by adjacency support a_ij > 0,
                   and additionally scaled by the prior a_ij itself so that
                   *known* hydraulic connectivity informs, not just gates,
                   the learned attention.
        v_i^{l+1} = ReLU( sum_j alpha_ij W v_j )

    Multi-head outputs are averaged (final layer) or concatenated
    (intermediate layers), matching standard GAT practice.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        concat_heads: bool = True,
        dropout: float = 0.1,
        leaky_relu_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.out_dim = out_dim

        self.W = nn.Parameter(torch.empty(num_heads, in_dim, out_dim))
        self.attn_src = nn.Parameter(torch.empty(num_heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(num_heads, out_dim))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        v:         (B, N, in_dim)  node features
        adjacency: (B, N, N)       non-negative connectivity weights a_ij
                                    (0 where no edge / self excluded)
        returns:   (B, N, out_dim * num_heads) if concat_heads else (B, N, out_dim)
        """
        b, n, _ = v.shape
        h = torch.einsum("bnd,hde->bhne", v, self.W)  # (B, H, N, out_dim)

        src_score = torch.einsum("bhne,he->bhn", h, self.attn_src)  # (B, H, N)
        dst_score = torch.einsum("bhne,he->bhn", h, self.attn_dst)  # (B, H, N)
        e = self.leaky_relu(src_score.unsqueeze(-1) + dst_score.unsqueeze(-2))  # (B,H,N,N)

        support = (adjacency > 0).unsqueeze(1)  # (B, 1, N, N)
        e = e.masked_fill(~support, float("-inf"))

        alpha = F.softmax(e, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)  # rows with no neighbors -> all -inf -> softmax nan
        alpha = alpha * adjacency.unsqueeze(1)    # fold the hydraulic-connectivity prior in
        alpha = self.dropout(alpha)

        out = torch.einsum("bhij,bhje->bhie", alpha, h)  # (B, H, N, out_dim)
        out = F.relu(out)

        if self.concat_heads:
            return out.permute(0, 2, 1, 3).reshape(b, n, self.num_heads * self.out_dim)
        return out.mean(dim=1)


class HydrologicGATStack(nn.Module):
    """L-layer stack of `HydrologicGATLayer`, matching v_i^{(L)} in the blueprint."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        dim = in_dim
        for layer_idx in range(num_layers):
            is_last = layer_idx == num_layers - 1
            layers.append(
                HydrologicGATLayer(
                    in_dim=dim,
                    out_dim=hidden_dim,
                    num_heads=num_heads,
                    concat_heads=not is_last,
                    dropout=dropout,
                )
            )
            dim = hidden_dim * num_heads if not is_last else hidden_dim
        self.layers = nn.ModuleList(layers)
        self.output_dim = dim

    def forward(self, v: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            v = layer(v, adjacency)
        return v


# --------------------------------------------------------------------------- #
# Time-delta-aware positional encoding + Temporal Alignment Transformer
# --------------------------------------------------------------------------- #

class TimeDeltaPositionalEncoding(nn.Module):
    """
    Encodes the *actual* elapsed time (days) since the previous acquisition,
    rather than a fixed integer index, so irregular InSAR revisit gaps and
    rainfall-alignment gaps are represented faithfully.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even for sin/cos encoding"
        self.embed_dim = embed_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, delta_t_days: torch.Tensor) -> torch.Tensor:
        """delta_t_days: (B, T) -> (B, T, embed_dim)"""
        cumulative_t = torch.cumsum(delta_t_days, dim=1)  # elapsed time, not raw index
        args = cumulative_t.unsqueeze(-1) * self.inv_freq  # (B, T, embed_dim/2)
        pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return pe


class AttentionPool1D(nn.Module):
    """
    Single learned query attending over a sequence, manually implemented
    (rather than `nn.MultiheadAttention`) because that module's internal
    `chunk`-based in-projection does not lower cleanly to ONNX under the
    dynamic batch/sequence axes this model requires (`unsafe_chunk, unknown
    dimension size`). This version uses only ops with well-defined ONNX
    symbolics (`matmul`, `softmax`, `masked_fill`) and is numerically
    equivalent to standard scaled dot-product multi-head attention with a
    single query token.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,               # (B, 1, E)
        key_value: torch.Tensor,           # (B, T, E)
        key_padding_mask: torch.Tensor | None = None,  # (B, T) bool, True == ignore
    ) -> torch.Tensor:
        b, t, e = key_value.shape
        q = self.q_proj(query).view(b, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, 1, T)
        if key_padding_mask is not None:
            mask = key_padding_mask.view(b, 1, 1, t)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, 1, head_dim)
        out = out.transpose(1, 2).reshape(b, 1, e)
        return self.out_proj(out).squeeze(1)


class TemporalAlignmentTransformer(nn.Module):
    """
    Transformer encoder over x_{i,t} = [d_i(t) ; R_i(t) ; Δt_t], producing a
    fixed-size temporal embedding z_i that summarizes how rainfall and
    InSAR-observed deformation co-vary for structure i.
    """

    def __init__(
        self,
        input_dim: int = 3,
        embed_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = TimeDeltaPositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_attn = AttentionPool1D(embed_dim, num_heads, dropout=dropout)
        self.output_dim = embed_dim

    def forward(
        self,
        displacement: torch.Tensor,
        rainfall: torch.Tensor,
        delta_t_days: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        displacement, rainfall, delta_t_days: (B, T)
        key_padding_mask: (B, T) bool, True where the position should be
            *ignored* (e.g. left-padding from `ARSDataset`).
        returns: (B, embed_dim)
        """
        x = torch.stack([displacement, rainfall, delta_t_days], dim=-1)  # (B, T, 3)
        h = self.input_proj(x) + self.pos_encoding(delta_t_days)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        b = h.shape[0]
        query = self.pool_query.expand(b, -1, -1)
        pooled = self.pool_attn(query, h, key_padding_mask=key_padding_mask)
        return pooled


# --------------------------------------------------------------------------- #
# Recharge-rate temporal decoder head
# --------------------------------------------------------------------------- #

class RechargeDecoder(nn.Module):
    """
    Autoregressive-free sequence decoder: unrolls an LSTM initialized from
    the fused node embedding u_i and conditioned at each step on the
    rainfall forcing, producing Q_i(t) for t = 1..T. Non-negativity is
    enforced with a softplus output activation (softer than a hard clamp,
    keeps gradients flowing near zero).
    """

    def __init__(self, fused_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.init_hidden = nn.Linear(fused_dim, hidden_dim)
        self.init_cell = nn.Linear(fused_dim, hidden_dim)
        self.cell = nn.LSTMCell(input_size=1, hidden_size=hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.hidden_dim = hidden_dim

    def forward(self, fused: torch.Tensor, rainfall: torch.Tensor) -> torch.Tensor:
        """
        fused:    (B, fused_dim)
        rainfall: (B, T) forcing sequence used to condition each decode step
        returns:  (B, T) non-negative recharge-rate estimate Q_i(t)
        """
        b, t = rainfall.shape
        h = torch.tanh(self.init_hidden(fused))
        c = torch.tanh(self.init_cell(fused))
        outputs = []
        for step in range(t):
            inp = rainfall[:, step].unsqueeze(-1)
            h, c = self.cell(inp, (h, c))
            outputs.append(self.output_head(h))
        q = torch.cat(outputs, dim=1)
        return F.softplus(q)
