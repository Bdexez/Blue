import unittest
import torch

from blue.config import BlueConfig
from blue.model import Blue
from blue.verifier import verify_architecture


class Wave20FeaturesTests(unittest.TestCase):
    def test_talking_heads_router_orthogonality_and_multimodal_cfg_decoding(self):
        config = BlueConfig(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            hidden_dim=64,
            n_experts=4,
            top_k=2,
            talking_heads=True,
            moe_router_orthogonality_weight=0.05,
            kv_lora_rank=16,
            differential_attention=True,
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
        rep = verify_architecture(config, seed=41)
        self.assertTrue(rep["passed"], rep)

        model = Blue(config).train()
        tokens = torch.randint(0, 260, (2, 10))
        images = torch.randn(2, 3, 16, 16)
        out = model(tokens, images=images, labels=tokens)
        self.assertIn("router_orthogonality", out.losses)
        self.assertGreaterEqual(out.losses["router_orthogonality"].item(), 0.0)
        out.loss.backward()
        for block in model.blocks:
            self.assertIsNotNone(block.attn.head_mix.weight.grad)
            self.assertGreater(block.attn.head_mix.weight.grad.abs().sum().item(), 0.0)
            self.assertIsNotNone(block.moe.router.weight.grad)
            self.assertGreater(block.moe.router.weight.grad.abs().sum().item(), 0.0)

        # Verify Multimodal CFG decoding (scale=1.0 matches standard conditional decoding,
        # scale=0.0 matches unconditional decoding!)
        model.eval()
        prompt = torch.tensor([[1, 66, 108, 117]])
        img1 = images[:1]
        cond = model.generate(prompt, max_new_tokens=8, temperature=0, eos_id=None, images=img1)
        cfg1 = model.generate(
            prompt, max_new_tokens=8, temperature=0, eos_id=None,
            images=img1, context_guidance_scale=1.0,
        )
        uncond = model.generate(prompt, max_new_tokens=8, temperature=0, eos_id=None)
        cfg0 = model.generate(
            prompt, max_new_tokens=8, temperature=0, eos_id=None,
            images=img1, context_guidance_scale=0.0,
        )
        cfg2 = model.generate(
            prompt, max_new_tokens=8, temperature=0, eos_id=None,
            images=img1, context_guidance_scale=2.5,
        )
        self.assertTrue(torch.equal(cond, cfg1))
        self.assertTrue(torch.equal(uncond, cfg0))
        self.assertEqual(cfg2.shape, (1, 12))


if __name__ == "__main__":
    unittest.main()
