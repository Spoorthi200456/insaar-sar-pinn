"""
Review finding 23. `latent_drift_score` used to be ||e - mean(e)|| over the
components of ONE embedding: a spread, not a drift. Drift now means: how far a
structure's node embedding lies from the TRAINING embeddings, as the RMS
z-score sqrt(mean_j ((e_j - mu_j) / sigma_j)^2) against the per-component mean
mu and std sigma recorded on the training split. Without a training reference
the score is None. See docs/math/latent_drift.md.
"""

from __future__ import annotations

import pytest
import torch

from src.backend.inference_engine import InferenceEngine
from src.training.train import make_synthetic_records


def test_drift_is_rms_z_score_against_training_reference():
    from src.models.network import latent_drift

    mu, sigma = torch.tensor([1.0, -2.0, 0.5]), torch.tensor([0.5, 2.0, 1.0])
    assert latent_drift(mu.clone(), mu, sigma) == pytest.approx(0.0)
    assert latent_drift(mu + 3 * sigma, mu, sigma) == pytest.approx(3.0)
    assert latent_drift(mu + torch.tensor([1.5, 0.0, 0.0]), mu, sigma) == pytest.approx(
        (3.0**2 / 3) ** 0.5
    )


def test_untrained_model_reports_no_drift():
    recs = make_synthetic_records(n_structures=3, n_timesteps=8, seed=1)
    eng = InferenceEngine(onnx_path=None, torch_checkpoint_path=None, sequence_length=8)
    result = eng.explain(recs, recs[0].design.structure_id)
    assert result["latent_drift_score"] is None


def test_drift_uses_the_stored_reference():
    recs = make_synthetic_records(n_structures=3, n_timesteps=8, seed=1)
    eng = InferenceEngine(onnx_path=None, torch_checkpoint_path=None, sequence_length=8)
    model = eng._torch_model
    batch = eng._build_batch_tensors(recs, [])
    with torch.no_grad():
        emb = model(
            design_features=batch["design_features"],
            displacement_cm=batch["displacement_cm"],
            rainfall_mm=batch["rainfall_mm"],
            time_delta_days=batch["time_delta_days"],
            adjacency=batch["adjacency"],
            incidence_angle_deg=batch["incidence_angle_deg"],
            coherence_mask=batch["coherence_mask"],
        ).node_embedding
    model.set_embedding_reference(emb)
    got = eng.explain(recs, recs[1].design.structure_id)["latent_drift_score"]
    from src.models.network import latent_drift

    expected = latent_drift(emb[1], emb.mean(dim=0), emb.std(dim=0, unbiased=False).clamp_min(1e-6))
    assert got == pytest.approx(expected, rel=1e-4)


def test_training_records_the_reference(tmp_path):
    from src.training.train import train

    recs = make_synthetic_records(n_structures=12, n_timesteps=8, seed=1)
    train(
        recs,
        epochs=1,
        batch_size=4,
        sequence_length=8,
        checkpoint_path=tmp_path / "c.pt",
        device="cpu",
        num_workers=0,
    )
    state = torch.load(tmp_path / "c.pt", weights_only=False)["model_state_dict"]
    assert state["embedding_reference_count"].item() > 0
    assert torch.isfinite(state["embedding_mean"]).all()
    assert (state["embedding_std"] > 0).all()
