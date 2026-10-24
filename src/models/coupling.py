"""
Head change -> InSAR line-of-sight displacement (review finding 2).

    d_LOS [cm] = CM_PER_M * cos(theta) * S_k * delta_h [m]

Derivation (docs/math/deformation_coupling.md): Terzaghi effective stress
(sigma = sigma' + p), constant total stress, p = rho_w g h, 1-D elastic skeleton
strain db/b = -alpha_c d(sigma') give db = S_k * dh with S_k = rho_w g alpha_c b
(Freeze & Cherry 1979 s.2.10; Hoffmann et al. 2001). Ground motion u_z = db
(nothing below deforms), projected onto the radar line of sight with
horizontal motion neglected: d_LOS = u_z cos(theta) (Hanssen 2001 ch.2).

Sign: positive = motion toward the satellite (PROVISIONAL, docs/open_questions.md
OQ2), so a head rise (recharge) gives uplift and d_LOS > 0.

This is the single implementation used by the model and by the Python
synthetic generator; `frontend/lib/synthetic.ts` mirrors it by hand.
"""

from __future__ import annotations

import math

import torch

CM_PER_M = 100.0  # unit conversion m -> cm, not a tuning knob
RAD_PER_DEG = math.pi / 180.0  # explicit (torch.deg2rad has no ONNX export symbolic)


def head_to_los_cm(
    delta_h_m: torch.Tensor,
    skeletal_storage: torch.Tensor,
    incidence_deg: torch.Tensor,
) -> torch.Tensor:
    """
    delta_h_m:        (B, T) head change since the reference acquisition [m]
    skeletal_storage: scalar or (B,) skeletal storage coefficient S_k [dimensionless]
    incidence_deg:    (B,)   radar incidence angle theta at the structure [deg]
    returns:          (B, T) LOS displacement [cm], positive = toward the satellite
    """
    cos_theta = torch.cos(incidence_deg * RAD_PER_DEG).unsqueeze(-1)  # (B, 1)
    s_k = skeletal_storage.reshape(-1, 1) if skeletal_storage.dim() > 0 else skeletal_storage
    return CM_PER_M * cos_theta * s_k * delta_h_m
