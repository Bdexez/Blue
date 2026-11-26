import unittest
import torch

from blue.config import BlueConfig
from blue.model import Blue
from blue.verifier import verify_architecture


class Wave18FeaturesTests(unittest.TestCase):
    def test_qk_norm_temperature_chained_future_heads_and_beam_search(self):
        # Graine fixe : sans elle, l'initialisation dépend des tests exécutés avant dans le même processus.
        torch.manual_seed(1808)
        config = BlueConfig(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            hidden_dim=64,
            n_experts=4,
            top_k=2,
            qk_norm_temperature=True,
            chained_future_heads=True,
            prediction_steps=4,
            mtp_kl_weight=0.05,
            ngram_vocab_size=64,
            kv_lora_rank=16,
            differential_attention=True,
            ssm_state_dim=8,
            ttt_state_dim=8,
            memory_slots=4,
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
        rep = verify_architecture(config, seed=31)
        self.assertTrue(rep["passed"], rep)

        model = Blue(config).train()
        self.assertIsNotNone(model.mtp_chain_gates)
        self.assertEqual(model.mtp_chain_gates.shape, (2, 32))
        tokens = torch.randint(0, 260, (2, 12))
        out = model(tokens, labels=tokens)
        out.loss.backward()
        self.assertIsNotNone(model.mtp_chain_gates.grad)
        self.assertGreater(model.mtp_chain_gates.grad.abs().sum().item(), 0.0)
        for block in model.blocks:
            self.assertIsNotNone(block.attn.qk_log_scale.grad)
            self.assertGreater(block.attn.qk_log_scale.grad.abs().sum().item(), 0.0)

        # Verify Beam Search with beam_width=1 matches greedy decoding bit-for-bit,
        # and, for this fixed model, beam_width=3 reaches a cumulative log-probability >= greedy
        # (not guaranteed in general: beam search may prune the greedy continuation).
        model.eval()
        prompt = torch.tensor([[1, 66, 108, 117]])
        greedy = model.generate(prompt, max_new_tokens=8, temperature=0, eos_id=None)
        beam1 = model._generate_beam_search(
            prompt, max_new_tokens=8, beam_width=1, length_penalty=0.0,
            eos_id=None, reasoning_steps=None,
        )
        self.assertTrue(torch.equal(greedy, beam1))

        beam3 = model.generate(
            prompt, max_new_tokens=8, temperature=0, eos_id=None,
            beam_width=3, length_penalty=0.0,
        )
        self.assertEqual(beam3.shape, (1, 12))

        def seq_log_prob(seq):
            with torch.no_grad():
                logits = model(seq.clone()[:, :-1]).logits[0].float()
                logits[:, [0, 1, 3]] = -torch.inf
                lp = logits.log_softmax(dim=-1)
                p_len = prompt.size(1)
                gen_toks = seq[0, p_len:]
                step_lps = lp[p_len - 1:, :].gather(-1, gen_toks[:, None]).squeeze(-1)
                return step_lps.sum().item()

        self.assertGreaterEqual(seq_log_prob(beam3) + 1e-4, seq_log_prob(greedy))


if __name__ == "__main__":
    unittest.main()
