from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import main
from blue.data import synthetic_batch
from blue.training import load_checkpoint


class MoEEnhancementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_validation(self):
        base = BlueConfig.preset("tiny")
        with self.assertRaises(ValueError):
            replace(base, shared_expert_gate=1)
        with self.assertRaises(ValueError):
            replace(base, router_bias_update_rate=-0.01)
        with self.assertRaises(ValueError):
            replace(base, router_bias_update_rate=float("nan"))

    def test_shared_expert_gate_gradients_and_checkpointing(self):
        torch.manual_seed(5001)
        config = replace(BlueConfig.preset("tiny"), shared_expert_gate=True,
                         router_bias_update_rate=0.01, balance_weight=0.0)
        batch = synthetic_batch(config, batch_size=2, seed=5002)
        regular = Blue(config)
        checkpointed = Blue(replace(config, gradient_checkpointing=True))
        checkpointed.load_state_dict(regular.state_dict())
        regular(**batch).loss.backward()
        checkpointed(**batch).loss.backward()
        for block in regular.blocks:
            grad = block.moe.shared_gate.weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0.0)
        for (name, p1), (_, p2) in zip(regular.named_parameters(), checkpointed.named_parameters()):
            if p1.grad is not None:
                torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-4, atol=1e-5, msg=name)

    def test_loss_free_bias_balancing_corrects_skewed_routing(self):
        torch.manual_seed(5003)
        config = replace(BlueConfig.preset("tiny"), n_experts=4, top_k=1,
                         router_bias_update_rate=0.005, balance_weight=0.0)
        model = Blue(config).eval()
        # Artificially skew routing strongly toward expert 0 via an initial bias,
        # then verify bias updates push it back toward balanced utilization with zero mean drift.
        with torch.no_grad():
            model.blocks[0].moe.expert_bias.copy_(torch.tensor([0.3, -0.1, -0.1, -0.1]))
        batch = synthetic_batch(config, batch_size=4, seed=5004)
        inputs = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            initial_usage = model(**inputs).expert_usage[0]
        self.assertGreater(initial_usage[0].item(), 0.9)
        for _ in range(80):
            with torch.no_grad():
                out = model(**inputs)
            model.update_router_biases(out.expert_usage)
        with torch.no_grad():
            final_usage = model(**inputs).expert_usage[0]
        self.assertAlmostEqual(model.blocks[0].moe.expert_bias.mean().item(), 0.0, places=6)
        self.assertLess(final_usage.max().item(), 0.4)

    def test_exact_resume_preserves_router_biases(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            cfg = root / "cfg.json"
            replace(BlueConfig.preset("tiny"), shared_expert_gate=True,
                    router_bias_update_rate=0.01, balance_weight=0.0).save(cfg)
            common = ["--threads", "2", "train", "--config", str(cfg), "--steps", "6",
                      "--batch-size", "2", "--eval-every", "6", "--seed", "77"]
            full, split = root / "full", root / "split"
            main(common + ["--output", str(full)])
            main(common + ["--output", str(split), "--stop-after", "3"])
            main(common + ["--output", str(split), "--resume", str(split / "latest.pt")])
            a, _ = load_checkpoint(full / "latest.pt")
            b, _ = load_checkpoint(split / "latest.pt")
            self.assertGreater(a.blocks[0].moe.expert_bias.abs().sum().item(), 0.0)
            for name, tensor in a.state_dict().items():
                torch.testing.assert_close(tensor, b.state_dict()[name], rtol=0, atol=0, msg=name)


if __name__ == "__main__":
    unittest.main()
