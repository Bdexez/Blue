from dataclasses import replace
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch


class SoftCapTests(unittest.TestCase):
    def test_config_validation_and_bounds(self):
        base = BlueConfig.preset("tiny")
        for bad in (
            dict(logit_softcap=-1.0),
            dict(logit_softcap=float("nan")),
            dict(attn_logit_softcap=-0.5),
            dict(attn_logit_softcap=float("inf")),
        ):
            with self.assertRaises(ValueError):
                replace(base, **bad)

        torch.manual_seed(350)
        cfg = replace(
            base,
            differential_attention=True, kv_lora_rank=16, attention_sinks=2,
            value_residual=True, logit_softcap=5.0, attn_logit_softcap=15.0,
        )
        model = Blue(cfg).eval()
        # Amplify embedding weights to push raw logits well beyond softcap!
        with torch.no_grad():
            model.embedding.weight.mul_(25.0)
        ids = torch.randint(4, 260, (2, 16))
        with torch.no_grad():
            out = model(ids)
            p1 = model(ids[:, :10], use_cache=True)
            p2 = model(ids[:, 10:], cache=p1.cache, use_cache=True)
        self.assertLessEqual(out.logits.abs().max().item(), 5.0 + 1e-5)
        torch.testing.assert_close(torch.cat((p1.logits, p2.logits), 1), out.logits, rtol=1e-4, atol=1e-4)

        # Verify gradients flow under BF16 and speculative greedy decoding matches standard greedy.
        model.train()
        batch = synthetic_batch(cfg, batch_size=2, seed=351)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            tr = model(**batch)
        self.assertTrue(torch.isfinite(tr.loss))
        tr.loss.backward()

        model.eval()
        with torch.no_grad():
            g_std = model.generate(ids[:1, :5], max_new_tokens=6, temperature=0, speculative=False)
            g_spec = model.generate(ids[:1, :5], max_new_tokens=6, temperature=0, speculative=True)
        self.assertTrue(torch.equal(g_std, g_spec))


if __name__ == "__main__":
    unittest.main()
