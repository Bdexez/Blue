from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

from blue import BlueConfig
from blue.cli import main
from blue.verifier import verify_architecture


class ArchitectureVerifierTests(unittest.TestCase):
    def test_verify_full_8modality_wave11_architecture_and_cli(self):
        cfg = replace(
            BlueConfig.preset("tiny"),
            ngram_vocab_size=256,
            kan_degree=3,
            moe_null_experts=2,
            moe_router_geometry="hyperbolic",
            differential_attention=True,
            cross_differential_attention=True,
            multimodal_adaln=True,
            kv_lora_rank=16,
            cross_kv_lora_rank=16,
            attention_sinks=2,
            value_residual=True,
            logit_softcap=30.0,
            attn_logit_softcap=50.0,
            nsa_block_size=4,
            nsa_top_k=2,
            ssm_state_dim=8,
            ttt_state_dim=8,
            residual_streams=2,
            latent_reasoning_steps=2,
            latent_reasoning_adaptive=True,
            memory_slots=4,
            vision_rope_2d=True,
            temporal_spectral_mixing=True,
            temporal_wavelet_mixing=True,
            forecast_tail_power=True,
            audio_enabled=True, audio_layers=1, max_audio_len=64,
            graph_enabled=True, graph_layers=1, max_graph_nodes=8, graph_rwse_steps=4,
            pointcloud_enabled=True, pointcloud_features=2, pointcloud_layers=1, max_pointcloud_points=10,
            tabular_enabled=True, tabular_num_features=4, tabular_cat_features=2,
            tabular_cat_cardinality=8, tabular_layers=1, tabular_plr_frequencies=4,
            video_enabled=True, video_channels=3, video_size=16, video_patch_size=8,
            video_tubelet_size=2, max_video_frames=4, video_layers=1,
            register_latents=2,
            hierarchical_resampler=True,
            latent_fsq_levels=(5, 5),
            latent_diversity_weight=0.02,
            jepa_weight=0.1,
            adaptive_loss_weighting=True,
            alignment_geometry="hyperbolic",
        )
        report = verify_architecture(cfg, seed=11)
        self.assertTrue(report["passed"], report)
        self.assertTrue(report["checks"]["causality_passed"])
        self.assertTrue(report["checks"]["cache_passed"])
        self.assertTrue(report["checks"]["series_padding_passed"])
        self.assertTrue(report["checks"]["quantile_monotonic_passed"])
        self.assertTrue(report["checks"]["graph_permutation_passed"])
        self.assertTrue(report["checks"]["pointcloud_se3_passed"])
        self.assertTrue(report["checks"]["loss_finite"])
        self.assertTrue(report["checks"]["gradients_finite"])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "verify.json"
            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["--threads", "2", "verify", "--preset", "tiny", "--output", str(out)])
            saved = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(saved["passed"])


if __name__ == "__main__":
    unittest.main()
