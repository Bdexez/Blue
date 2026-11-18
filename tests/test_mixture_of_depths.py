from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import inspect_config
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class MixtureOfDepthsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_validation(self):
        base = BlueConfig.preset("tiny")
        for bad in (
            dict(depth_routing_capacity=0.0),
            dict(depth_routing_capacity=1.5),
            dict(depth_routing_capacity=float("nan")),
            dict(depth_budget_weight=-0.1),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(base, **bad)

    def test_causal_mod_skips_tokens_preserves_causality_cache_padding_and_gradients(self):
        torch.manual_seed(15001)
        config = replace(BlueConfig.preset("tiny"), max_seq_len=128,
                         depth_routing_capacity=0.5, depth_budget_weight=0.05,
                         residual_streams=2, attention_sinks=2,
                         gradient_checkpointing=True)
        info = inspect_config(config)
        self.assertEqual(info["architecture_features"]["depth_routing_capacity"], 0.5)
        model = Blue(config).eval()
        batch = synthetic_batch(config, batch_size=2, seed=15002)
        ids = torch.randint(4, 260, (2, 40))

        # Verify MoE actually skips tokens when router scores fall below 0.5.
        routed_counts = []
        hook = model.blocks[0].moe.register_forward_pre_hook(
            lambda module, args: routed_counts.append(int(args[1].sum().item()))
        )
        with torch.no_grad():
            full = model(ids, images=batch["images"], series=batch["series"]).logits
        hook.remove()
        self.assertLess(routed_counts[0], ids.numel())

        # 1. Strict causality (changing future tokens cannot affect past routing or logits).
        changed = ids.clone()
        changed[:, 12:] = torch.randint(4, 260, changed[:, 12:].shape)
        with torch.no_grad():
            alt = model(changed, images=batch["images"], series=batch["series"]).logits
        torch.testing.assert_close(full[:, :12], alt[:, :12], rtol=1e-5, atol=1e-6)

        # 2. Exact sliding-window cache equivalence.
        with torch.no_grad():
            p1 = model(ids[:, :34], images=batch["images"], series=batch["series"], use_cache=True)
            p2 = model(ids[:, 34:38], cache=p1.cache, use_cache=True)
            p3 = model(ids[:, 38:], cache=p2.cache, use_cache=True)
        torch.testing.assert_close(torch.cat((p1.logits, p2.logits, p3.logits), 1), full, rtol=3e-5, atol=3e-6)

        # 3. Exact padding invariance.
        with torch.no_grad():
            base_out = model(**batch)
            padded = batch.copy()
            padded["input_ids"] = torch.nn.functional.pad(padded["input_ids"], (0, 6), value=0)
            padded["labels"] = padded["input_ids"].clone()
            padded["attention_mask"] = torch.nn.functional.pad(padded["attention_mask"], (0, 6), value=False)
            pad_out = model(**padded)
        torch.testing.assert_close(base_out.loss, pad_out.loss, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(base_out.losses["depth_budget"], pad_out.losses["depth_budget"], rtol=1e-5, atol=1e-6)

        # 4. Gradients reach depth_router weight and bias under bf16 + checkpointing.
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertIn("depth_budget", out.losses)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        for block in model.blocks:
            self.assertGreater(block.depth_router.weight.grad.abs().sum().item(), 0.0)
            self.assertGreater(block.depth_router.bias.grad.abs().sum().item(), 0.0)

        # 5. Checkpoint roundtrip.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mod.pt"
            save_checkpoint(path, model.eval())
            restored, _ = load_checkpoint(path)
            self.assertEqual(restored.config.depth_routing_capacity, 0.5)
            with torch.no_grad():
                torch.testing.assert_close(model(ids).logits, restored(ids).logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
