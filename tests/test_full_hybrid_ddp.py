from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from blue import BlueConfig


class FullHybridDistributedTests(unittest.TestCase):
    def test_two_rank_full_hybrid_6modality_muon_ema_exact_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "full_hybrid.json"
            replace(
                BlueConfig.preset("tiny"),
                max_seq_len=128,
                ngram_vocab_size=256,
                rope_scaling_type="yarn",
                rope_scaling_factor=2.0,
                differential_attention=True,
                cross_differential_attention=True,
                multimodal_adaln=True,
                kv_lora_rank=16,
                cross_kv_lora_rank=16,
                attention_sinks=2,
                value_residual=True,
                logit_softcap=30.0,
                attn_logit_softcap=50.0,
                dynamic_attn_scale=True,
                nsa_block_size=4,
                nsa_top_k=2,
                ssm_state_dim=8,
                ssm_heads=4,
                ssm_chunk_size=16,
                ssm_every=1,
                ttt_state_dim=8,
                ttt_heads=4,
                ttt_every=2,
                kan_degree=3,
                moe_null_experts=2,
                moe_shared_latent_dim=8,
                moe_router_geometry="hyperbolic",
                shared_expert_gate=True,
                router_bias_update_rate=0.01,
                depth_routing_capacity=0.75,
                residual_streams=2,
                latent_reasoning_steps=2,
                latent_reasoning_adaptive=True,
                memory_slots=4,
                vision_rope_2d=True,
                temporal_statistics=True,
                temporal_statistics_relative=True,
                temporal_spectral_mixing=True,
                temporal_wavelet_mixing=True,
                forecast_tail_power=True,
                audio_enabled=True,
                audio_patch_size=8,
                audio_layers=1,
                graph_enabled=True,
                graph_node_features=6,
                graph_edge_features=3,
                graph_layers=1,
                graph_rwse_steps=4,
                pointcloud_enabled=True,
                pointcloud_features=2,
                pointcloud_layers=1,
                tabular_enabled=True,
                tabular_num_features=6,
                tabular_cat_features=2,
                tabular_cat_cardinality=8,
                tabular_layers=1,
                tabular_plr_frequencies=4,
                video_enabled=True,
                video_channels=3,
                video_size=16,
                video_patch_size=8,
                video_tubelet_size=2,
                max_video_frames=4,
                video_layers=1,
                register_latents=2,
                cross_modal_fusion_layers=1,
                hierarchical_resampler=True,
                latent_fsq_levels=(5, 5),
                latent_diversity_weight=0.02,
                evidential_classification=True,
                mtp_kl_weight=0.05,
                jepa_weight=0.1,
                adaptive_loss_weighting=True,
                alignment_geometry="hyperbolic",
                alignment_scope="global",
                stable_rms_norm=True,
                gradient_checkpointing=True,
                checkpoint_every=2,
            ).save(config_path)

            common = [
                sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
                "-m", "blue", "--threads", "1", "train", "--config", str(config_path),
                "--steps", "4", "--batch-size", "2", "--accumulation", "2",
                "--optimizer", "muon", "--lr-schedule", "wsd", "--ema-decay", "0.95",
                "--eval-every", "2", "--save-every", "2", "--seed", "73",
            ]
            env = dict(os.environ, OMP_NUM_THREADS="1")
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                env.pop(key, None)

            def run(*options):
                res = subprocess.run([*common, *map(str, options)], env=env, text=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
                self.assertEqual(res.returncode, 0, res.stdout[-6000:])
                return res.stdout

            full, split = root / "full", root / "split"
            run("--output", full)
            run("--output", split, "--stop-after", "2")
            run("--output", split, "--resume", split / "latest.pt")

            expected = torch.load(full / "latest.pt", weights_only=True)
            actual = torch.load(split / "latest.pt", weights_only=True)
            self.assertEqual(expected["step"], 4)
            self.assertEqual(actual["step"], 4)
            self.assertEqual(actual["trainer"]["optimizer"], "muon")
            self.assertEqual(actual["trainer"]["ema_decay"], 0.95)
            self.assertEqual(actual["ema"]["updates"], 4)
            # Live weights, router biases, and EMA shadow weights match bit-for-bit across DDP resume.
            for name, tensor in expected["model"].items():
                torch.testing.assert_close(tensor, actual["model"][name], rtol=0, atol=0, msg=name)
            for name, tensor in expected["ema"]["shadow"].items():
                torch.testing.assert_close(tensor, actual["ema"]["shadow"][name], rtol=0, atol=0, msg=name)


if __name__ == "__main__":
    unittest.main()
