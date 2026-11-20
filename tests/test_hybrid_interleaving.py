from dataclasses import replace
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch
from blue.profiler import profile_architecture


class HybridInterleavingTests(unittest.TestCase):
    def test_interleaved_ssm_and_ttt_layers_cache_and_profiling(self):
        base = BlueConfig.preset("tiny")
        for bad in (dict(ssm_every=0), dict(ssm_every=True), dict(ttt_every=0), dict(ttt_every=-1)):
            with self.assertRaises(ValueError):
                replace(base, **bad)

        torch.manual_seed(420)
        cfg_dense = replace(base, n_layers=4, ssm_state_dim=8, ttt_state_dim=8, ssm_every=1, ttt_every=1)
        cfg_inter = replace(base, n_layers=4, ssm_state_dim=8, ttt_state_dim=8, ssm_every=2, ttt_every=4)
        model = Blue(cfg_inter).eval()

        self.assertEqual([b.ssm is not None for b in model.blocks], [True, False, True, False])
        self.assertEqual([b.ttt is not None for b in model.blocks], [True, False, False, False])

        # Profiling reflects exact state memory reduction (2x less SSM, 4x less TTT)!
        p_dense = profile_architecture(cfg_dense, seq_len=64)
        p_inter = profile_architecture(cfg_inter, seq_len=64)
        self.assertAlmostEqual(
            p_inter["cache_profile"]["ssm_state_mib_fp32"] * 2,
            p_dense["cache_profile"]["ssm_state_mib_fp32"],
            places=3,
        )
        self.assertAlmostEqual(
            p_inter["cache_profile"]["ttt_state_mib_fp32"] * 4,
            p_dense["cache_profile"]["ttt_state_mib_fp32"],
            places=3,
        )

        # Exact cache equivalence + strict state slot validation!
        ids = torch.randint(4, 260, (2, 18))
        with torch.no_grad():
            full = model(ids).logits
            p1 = model(ids[:, :10], use_cache=True)
            p2 = model(ids[:, 10:], cache=p1.cache, use_cache=True)
        self.assertEqual([l.ssm_state is not None for l in p1.cache.layers], [True, False, True, False])
        self.assertEqual([l.ttt_state is not None for l in p1.cache.layers], [True, False, False, False])
        torch.testing.assert_close(torch.cat((p1.logits, p2.logits), 1), full, rtol=1e-5, atol=1e-5)

        # Training backward works cleanly.
        model.train()
        batch = synthetic_batch(cfg_inter, batch_size=2, seed=421)
        out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()


if __name__ == "__main__":
    unittest.main()
