import unittest

import torch

from blue.config import BlueConfig
from blue.model import Blue


class Wave26FeaturesTest(unittest.TestCase):
    def test_ssm_d_skip_chunkwise_and_cache_equivalence(self):
        cfg = BlueConfig.preset("tiny")
        cfg.ssm_state_dim = 16
        cfg.ssm_heads = 4
        cfg.ssm_chunk_size = 4
        cfg.ssm_d_skip = True
        torch.manual_seed(2601)
        model = Blue(cfg).eval()
        for block in model.blocks:
            self.assertIsNotNone(block.ssm.d_skip)
            with torch.no_grad():
                block.ssm.d_skip.normal_(mean=0.2, std=0.05)

        x = torch.randn(2, 11, cfg.d_model)
        valid = torch.ones((2, 11), dtype=torch.bool)
        valid[1, -2:] = False
        with torch.no_grad():
            out_wy, state_wy = model.blocks[0].ssm(x, valid, use_chunkwise=True)
            out_seq, state_seq = model.blocks[0].ssm(x, valid, use_chunkwise=False)
        self.assertTrue(torch.allclose(out_wy, out_seq, atol=1e-5))
        self.assertTrue(torch.allclose(state_wy, state_seq, atol=1e-5))

        ids = torch.tensor([[1, 12, 23, 34, 45, 56]])
        model.train()
        out = model(ids, labels=ids)
        out.loss.backward()
        self.assertIsNotNone(model.blocks[0].ssm.d_skip.grad)
        self.assertGreater(model.blocks[0].ssm.d_skip.grad.abs().sum().item(), 0.0)

    def test_ttt_weight_decay_towards_meta_prior_and_cache_equivalence(self):
        cfg = BlueConfig.preset("tiny")
        cfg.ttt_state_dim = 16
        cfg.ttt_heads = 4
        cfg.ttt_weight_decay = True
        torch.manual_seed(2602)
        model = Blue(cfg).eval()
        for block in model.blocks:
            self.assertIsNotNone(block.ttt.decay_proj)
            self.assertTrue(torch.allclose(block.ttt.decay_proj.bias, torch.full_like(block.ttt.decay_proj.bias, 4.0)))
            with torch.no_grad():
                block.ttt.decay_proj.weight.normal_(std=0.05)

        ids = torch.tensor([[1, 14, 28, 42, 56, 70]])
        with torch.no_grad():
            full_logits = model(ids).logits
            prefill = model(ids[:, :3], use_cache=True)
            cache = prefill.cache
            pieces = [prefill.logits]
            for t in range(3, ids.size(1)):
                step_out = model(ids[:, t:t + 1], cache=cache, use_cache=True)
                cache = step_out.cache
                pieces.append(step_out.logits)
            cached_logits = torch.cat(pieces, dim=1)
        self.assertTrue(torch.allclose(full_logits, cached_logits, atol=1e-5))

        model.train()
        out = model(ids, labels=ids)
        out.loss.backward()
        self.assertIsNotNone(model.blocks[0].ttt.decay_proj.weight.grad)
        self.assertGreater(model.blocks[0].ttt.decay_proj.weight.grad.abs().sum().item(), 0.0)

    def test_entropy_dynamic_temperature_sampling(self):
        # Low-entropy confident distribution: EDT reduces effective temperature -> sharpens top-1 probability
        scores = torch.full((1, 260), -10.0)
        scores[0, 10] = 2.0
        scores[0, 11] = 1.0
        scores[0, 12] = 0.0
        p_fixed = Blue._filtered_probs(scores, temperature=1.0, top_k=0, top_p=1.0, edt_theta=0.0)
        p_edt = Blue._filtered_probs(scores, temperature=1.0, top_k=0, top_p=1.0, edt_theta=1.0, edt_n=0.8)
        self.assertGreater(p_edt[0, 10].item(), p_fixed[0, 10].item())

        cfg = BlueConfig.preset("tiny")
        model = Blue(cfg).eval()
        prompt = torch.tensor([[1, 17, 29]])
        gen = model.generate(prompt, max_new_tokens=5, temperature=0.9, edt_theta=0.8, edt_n=0.8, eos_id=None)
        self.assertEqual(gen.shape, (1, 8))


if __name__ == "__main__":
    unittest.main()
