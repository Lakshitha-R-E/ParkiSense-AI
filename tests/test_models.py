"""
ParkiSense AI - Unit Tests
==========================
Tests for:
  1. SafeFallbackGate – verifies filler substitution logic
  2. ParkiSenseFusionNet – verifies output tensor shapes
"""

import unittest
import torch
import numpy as np
import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    SafeFallbackGate,
    ParkiSenseFusionNet,
    ModalityEncoder,
    EMBED_DIM,
    N_MODALITIES,
    N_CLASSES,
)
from src.utils import updrs_to_label, run_inference


class TestSafeFallbackGate(unittest.TestCase):
    """Test the SafeFallbackGate's embedding substitution behaviour."""

    def setUp(self):
        self.batch_size = 4
        self.embed_dim = EMBED_DIM
        self.n_mod = N_MODALITIES
        self.gate = SafeFallbackGate(embed_dim=self.embed_dim, n_modalities=self.n_mod)
        self.gate.eval()

        # Create deterministic embeddings so values are predictable
        torch.manual_seed(0)
        self.embeddings = torch.randn(self.batch_size, self.n_mod, self.embed_dim)

    def test_all_quality_ok_passes_true_embeddings(self):
        """When all flags are True, every output token must equal the input embedding."""
        flags = torch.ones(self.batch_size, self.n_mod, dtype=torch.bool)
        output = self.gate(self.embeddings, flags)
        self.assertEqual(output.shape, (self.batch_size, self.n_mod, self.embed_dim))
        self.assertTrue(
            torch.allclose(output, self.embeddings, atol=1e-5),
            "When all quality flags are True, embeddings should pass through unchanged."
        )

    def test_all_quality_bad_replaces_with_filler(self):
        """When all flags are False, every output token must equal the learned filler."""
        flags = torch.zeros(self.batch_size, self.n_mod, dtype=torch.bool)
        output = self.gate(self.embeddings, flags)
        self.assertEqual(output.shape, (self.batch_size, self.n_mod, self.embed_dim))

        # Expected: filler repeated across batch
        expected = self.gate.filler.unsqueeze(0).expand(self.batch_size, -1, -1)
        self.assertTrue(
            torch.allclose(output, expected, atol=1e-5),
            "When all quality flags are False, outputs should match the learned filler."
        )

    def test_mixed_flags_correct_substitution(self):
        """
        Mixed quality: modalities with True flag keep true embedding;
        modalities with False flag receive filler.
        """
        # flags: [[True, False, True], [False, True, False], ...]
        flags = torch.tensor([
            [True, False, True],
            [False, True, False],
            [True, True, False],
            [False, False, True],
        ], dtype=torch.bool)

        output = self.gate(self.embeddings, flags)
        filler = self.gate.filler  # (n_mod, embed_dim)

        for b in range(self.batch_size):
            for m in range(self.n_mod):
                if flags[b, m]:
                    # Should match the true embedding
                    self.assertTrue(
                        torch.allclose(output[b, m], self.embeddings[b, m], atol=1e-5),
                        f"Batch {b}, Modality {m}: expected true embedding for quality_ok=True.",
                    )
                else:
                    # Should match the filler
                    self.assertTrue(
                        torch.allclose(output[b, m], filler[m], atol=1e-5),
                        f"Batch {b}, Modality {m}: expected filler for quality_ok=False.",
                    )

    def test_filler_is_learnable_parameter(self):
        """filler must be an nn.Parameter (learnable)."""
        self.assertIsInstance(self.gate.filler, torch.nn.Parameter)
        self.assertTrue(self.gate.filler.requires_grad)

    def test_output_shape(self):
        """Output shape must always be (batch, n_modalities, embed_dim)."""
        flags = torch.randint(0, 2, (self.batch_size, self.n_mod)).bool()
        output = self.gate(self.embeddings, flags)
        self.assertEqual(output.shape, torch.Size([self.batch_size, self.n_mod, self.embed_dim]))


class TestParkiSenseFusionNet(unittest.TestCase):
    """Test forward-pass output shapes of ParkiSenseFusionNet."""

    def setUp(self):
        self.voice_dim = 52
        self.eeg_dim = 64
        self.gait_dim = 42
        self.batch_size = 8
        self.model = ParkiSenseFusionNet(
            voice_dim=self.voice_dim,
            eeg_dim=self.eeg_dim,
            gait_dim=self.gait_dim,
        )
        self.model.eval()
        torch.manual_seed(99)

    def _make_inputs(self, batch_size=None):
        bs = batch_size or self.batch_size
        voice = torch.randn(bs, self.voice_dim)
        eeg = torch.randn(bs, self.eeg_dim)
        gait = torch.randn(bs, self.gait_dim)
        flags = torch.ones(bs, N_MODALITIES, dtype=torch.bool)
        return voice, eeg, gait, flags

    def test_regression_output_shape(self):
        """Regression head must output a scalar per sample: (batch_size,)."""
        voice, eeg, gait, flags = self._make_inputs()
        with torch.no_grad():
            updrs_scores, _ = self.model(voice, eeg, gait, flags)
        self.assertEqual(
            updrs_scores.shape,
            torch.Size([self.batch_size]),
            f"Expected regression output shape ({self.batch_size},), got {updrs_scores.shape}.",
        )

    def test_classification_output_shape(self):
        """Classification head must output logits per class per sample: (batch_size, 3)."""
        voice, eeg, gait, flags = self._make_inputs()
        with torch.no_grad():
            _, cls_logits = self.model(voice, eeg, gait, flags)
        self.assertEqual(
            cls_logits.shape,
            torch.Size([self.batch_size, N_CLASSES]),
            f"Expected classification output shape ({self.batch_size}, {N_CLASSES}), got {cls_logits.shape}.",
        )

    def test_updrs_score_range(self):
        """UPDRS score must lie within [0, 100] (from Sigmoid × 100)."""
        voice, eeg, gait, flags = self._make_inputs()
        with torch.no_grad():
            updrs_scores, _ = self.model(voice, eeg, gait, flags)
        self.assertTrue(
            (updrs_scores >= 0).all() and (updrs_scores <= 100).all(),
            f"UPDRS scores out of range [0, 100]: min={updrs_scores.min():.2f}, max={updrs_scores.max():.2f}",
        )

    def test_batch_size_one(self):
        """Model must handle batch size of 1 without errors."""
        voice, eeg, gait, flags = self._make_inputs(batch_size=1)
        with torch.no_grad():
            updrs, logits = self.model(voice, eeg, gait, flags)
        self.assertEqual(updrs.shape, torch.Size([1]))
        self.assertEqual(logits.shape, torch.Size([1, N_CLASSES]))

    def test_with_fallback_flags(self):
        """Model must produce valid outputs when some modality quality flags are False."""
        voice, eeg, gait, _ = self._make_inputs()
        # Set EEG flag to False (bad quality)
        mixed_flags = torch.tensor(
            [[True, False, True]] * self.batch_size, dtype=torch.bool
        )
        with torch.no_grad():
            updrs, logits = self.model(voice, eeg, gait, mixed_flags)
        self.assertEqual(updrs.shape, torch.Size([self.batch_size]))
        self.assertEqual(logits.shape, torch.Size([self.batch_size, N_CLASSES]))
        self.assertFalse(torch.isnan(updrs).any(), "UPDRS scores contain NaN with fallback flags.")
        self.assertFalse(torch.isnan(logits).any(), "Logits contain NaN with fallback flags.")

    def test_classification_logits_are_unbounded(self):
        """Classification logits should be raw (not softmax-normalised)."""
        voice, eeg, gait, flags = self._make_inputs()
        with torch.no_grad():
            _, cls_logits = self.model(voice, eeg, gait, flags)
        # Softmax-normalised outputs would sum to ~1; raw logits generally don't per-row
        row_sums = cls_logits.sum(dim=-1)
        # At least some rows should deviate significantly from 1.0
        not_all_summing_to_one = not torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.01)
        self.assertTrue(not_all_summing_to_one, "Logits appear to already be softmax-normalised.")

    def test_no_nan_in_outputs(self):
        """No NaN values should appear in either output for standard all-good inputs."""
        voice, eeg, gait, flags = self._make_inputs()
        with torch.no_grad():
            updrs, logits = self.model(voice, eeg, gait, flags)
        self.assertFalse(torch.isnan(updrs).any())
        self.assertFalse(torch.isnan(logits).any())


class TestModalityEncoder(unittest.TestCase):
    """Basic shape tests for the standalone ModalityEncoder."""

    def test_output_shape(self):
        for in_dim in [52, 64, 42, 128]:
            with self.subTest(in_dim=in_dim):
                encoder = ModalityEncoder(in_dim=in_dim, embed_dim=EMBED_DIM)
                x = torch.randn(6, in_dim)
                out = encoder(x)
                self.assertEqual(out.shape, torch.Size([6, EMBED_DIM]))


class TestSeverityMapping(unittest.TestCase):
    """Test clinical severity tier thresholds and inference consistency."""

    def test_updrs_to_label_thresholds(self):
        # 53.5 must map to Moderate
        lbl_idx, lbl_str = updrs_to_label(53.5)
        self.assertEqual(lbl_str, "Moderate")
        self.assertEqual(lbl_idx, 1)

        # Mild range (<= 32.4)
        self.assertEqual(updrs_to_label(15.0)[1], "Mild")
        self.assertEqual(updrs_to_label(30.0)[1], "Mild")

        # Moderate range (32.4 to 64.8)
        self.assertEqual(updrs_to_label(35.0)[1], "Moderate")
        self.assertEqual(updrs_to_label(60.0)[1], "Moderate")

        # Severe range (> 64.8)
        self.assertEqual(updrs_to_label(70.0)[1], "Severe")
        self.assertEqual(updrs_to_label(85.0)[1], "Severe")

    def test_run_inference_severity_alignment(self):
        model = ParkiSenseFusionNet(voice_dim=52, eeg_dim=64, gait_dim=42)
        v = np.zeros(52, dtype=np.float32)
        e = np.zeros(64, dtype=np.float32)
        g = np.zeros(42, dtype=np.float32)
        flags = (True, True, True)

        res = run_inference(model, v, e, g, flags)
        # Check expected keys exist
        self.assertIn("updrs_score", res)
        self.assertIn("severity_name", res)
        self.assertIn("cls_probs", res)
        # Ensure severity_name aligns with updrs_score
        expected_lbl, expected_name = updrs_to_label(res["updrs_score"])
        self.assertEqual(res["severity_name"], expected_name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
