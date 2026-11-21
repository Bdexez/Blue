import unittest
import torch

from blue.config import BlueConfig
from blue.model import Blue
from blue.optimizer import build_optimizer
from blue.verifier import verify_architecture


class DynamicAttnAndSharedSubspaceTests(unittest.TestCase):
    def test_dynamic_attn_scale_and_shared_subspace_across_all_attention_modes(self):
        for kv_rank, diff in ((0, False), (0, True), (16, False), (16, True)):
            config = BlueConfig(
                d_model=32,
                n_heads=4,
                n_kv_heads=2,
                n_layers=2,
                hidden_dim=64,
                n_experts=4,
                top_k=2,
                moe_null_experts=1,
                moe_shared_latent_dim=16,
                moe_router_geometry="hyperbolic",
                dynamic_attn_scale=True,
                kv_lora_rank=kv_rank,
                differential_attention=diff,
                n_latents=4,
                resampler_layers=1,
                vision_layers=1,
                temporal_layers=1,
                image_size=16,
                patch_size=8,
                max_seq_len=32,
                max_series_len=16,
                forecast_horizon=4,
                forecast_context=4,
                series_features=2,
                conv_kernel=3,
                sliding_window=8,
                global_every=1,
                dropout=0.0,
            )
            model = Blue(config)
            # Non-zero pos_temp_alpha to test active position-dependent logit scaling
            with torch.no_grad():
                for block in model.blocks:
                    block.attn.pos_temp_alpha.fill_(0.8)

            # Verify strict causality & cache equivalence via verify_architecture
            model.eval()
            tokens = torch.randint(0, 260, (1, 10))
            full_logits = model(tokens).logits
            out = model(tokens[:, :1], use_cache=True)
            step_logits = [out.logits]
            cache = out.cache
            for idx in range(1, tokens.size(1)):
                out = model(tokens[:, idx:idx + 1], cache=cache, use_cache=True)
                step_logits.append(out.logits)
                cache = out.cache
            cat_logits = torch.cat(step_logits, dim=1)
            self.assertTrue(
                torch.allclose(full_logits, cat_logits, atol=1e-4),
                f"Cache mismatch for kv_rank={kv_rank}, diff={diff}",
            )

            # Verify gradients and Muon optimizer compatibility
            model.train()
            opt = build_optimizer(model, name="muon", lr=1e-3, weight_decay=0.01)
            out_train = model(tokens, labels=tokens)
            out_train.loss.backward()
            for block in model.blocks:
                self.assertIsNotNone(block.attn.pos_temp_alpha.grad)
                self.assertGreater(block.attn.pos_temp_alpha.grad.abs().sum().item(), 0.0)
                self.assertIsNotNone(block.moe.latent_down.weight.grad)
                self.assertIsNotNone(block.moe.latent_expert_mod.grad)
                self.assertIsNotNone(block.moe.latent_up.weight.grad)
                self.assertGreater(block.moe.latent_down.weight.grad.abs().sum().item(), 0.0)
                self.assertGreater(block.moe.latent_expert_mod.grad.abs().sum().item(), 0.0)
                self.assertGreater(block.moe.latent_up.weight.grad.abs().sum().item(), 0.0)
            opt.step()

        # Also run full verify_architecture
        rep = verify_architecture(config)
        self.assertTrue(rep["passed"], rep)


if __name__ == "__main__":
    unittest.main()
