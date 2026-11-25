"""Régressions trouvées par l'audit : état résiduel, décodage et quantification."""

import copy
import unittest
from dataclasses import replace

import torch
from torch import nn

from blue.config import BlueConfig
from blue.data import synthetic_batch
from blue.model import Blue
from blue.quantize import Int4Linear


def tiny(**changes):
    torch.manual_seed(5)
    return Blue(replace(BlueConfig.preset("tiny"), **changes))


class ResidualStateTests(unittest.TestCase):
    def test_model_can_be_copied_after_a_training_step(self):
        for changes in ({}, {"qk_residual": True, "value_residual": True},
                        {"moe_null_experts": 2}, {"gradient_checkpointing": True, "qk_residual": True}):
            with self.subTest(changes):
                model = tiny(**changes)
                batch = synthetic_batch(model.config, batch_size=2)
                model(**batch).loss.backward()
                twin = copy.deepcopy(model).eval()
                model.eval()
                with torch.no_grad():
                    self.assertTrue(torch.equal(twin(**batch).logits, model(**batch).logits))

    def test_modules_keep_no_autograd_graph_between_passes(self):
        for checkpointing in (False, True):
            model = tiny(qk_residual=True, moe_entropy_weight=0.1, gradient_checkpointing=checkpointing)
            output = model(**synthetic_batch(model.config, batch_size=2))
            for backward in (False, True):
                if backward:
                    output.loss.backward()  # Le recalcul des checkpoints ne doit rien conserver non plus.
                for module in model.modules():
                    for name, value in vars(module).items():
                        if isinstance(value, torch.Tensor):
                            self.assertFalse(value.requires_grad, f"{type(module).__name__}.{name}")


class DecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny().eval()
        cls.ids = torch.randint(4, 200, (3, 8), generator=torch.Generator().manual_seed(1))

    def test_batched_beam_search_matches_each_row(self):
        model, ids = self.model, self.ids
        series = torch.randn(3, 8, model.config.series_features, generator=torch.Generator().manual_seed(2))
        batched = model.generate(ids, max_new_tokens=6, beam_width=3, eos_id=None, series=series)
        for row in range(3):
            single = model.generate(ids[row:row + 1], max_new_tokens=6, beam_width=3, eos_id=None,
                                    series=series[row:row + 1])
            self.assertTrue(torch.equal(batched[row:row + 1], single))

    def test_beam_rows_of_different_lengths_are_padded_with_eos(self):
        free = self.model.generate(self.ids, max_new_tokens=6, beam_width=2, eos_id=None)
        eos = int(free[0, 8])  # Premier token choisi par la première ligne.
        result = self.model.generate(self.ids, max_new_tokens=6, beam_width=2, eos_id=eos, length_penalty=0.0)
        self.assertEqual(result.size(0), 3)
        for row in result[:, 8:]:
            positions = (row == eos).nonzero()
            if positions.numel():
                self.assertTrue((row[positions[0].item():] == eos).all())

    def test_beam_may_be_wider_than_the_vocabulary(self):
        wide = self.model.config.vocab_size + 40
        result = self.model.generate(self.ids[:1], max_new_tokens=3, beam_width=wide, eos_id=None)
        self.assertEqual(result.shape, (1, 11))
        self.assertTrue((result[0, 8:] >= 2).all())

    def test_unsupported_options_are_refused_instead_of_ignored(self):
        image = torch.rand(1, 3, *(self.model.config.image_size,) * 2)
        for mode in ({"beam_width": 2}, {"speculative": True}, {"jacobi": True, "temperature": 0}):
            for option in ({"context_guidance_scale": 2.0, "images": image}, {"mirostat_tau": 3.0}):
                with self.subTest(mode=mode, option=list(option)), self.assertRaisesRegex(ValueError, "incompatible"):
                    self.model.generate(self.ids[:1], max_new_tokens=2, **mode, **option)
        with self.assertRaisesRegex(ValueError, "mutuellement exclusifs"):
            self.model.generate(self.ids[:1], max_new_tokens=2, beam_width=2, speculative=True)

    def test_missing_optional_context_is_accepted(self):
        for mode in ({}, {"beam_width": 2}):
            explicit = self.model.generate(self.ids[:1], max_new_tokens=3, temperature=0,
                                           attention_mask=None, images=None, **mode)
            implicit = self.model.generate(self.ids[:1], max_new_tokens=3, temperature=0, **mode)
            self.assertTrue(torch.equal(explicit, implicit))


class DispatchAndCacheTests(unittest.TestCase):
    def test_rotary_tables_follow_reloaded_frequencies_and_devices(self):
        from blue.modules import RotaryEmbedding
        rope = RotaryEmbedding(8, 10000.0)
        x = torch.randn(1, 2, 5, 8)
        first = rope(x, offset=3)
        rope.inv_freq = rope.frequencies() * 2  # Comme load_checkpoint, qui réassigne le tampon.
        second = rope(x, offset=3)
        self.assertFalse(torch.allclose(first, second))
        fresh = RotaryEmbedding(8, 10000.0)
        fresh.inv_freq = fresh.frequencies() * 2
        self.assertTrue(torch.equal(second, fresh(x, offset=3)))
        self.assertEqual(rope(x.double(), offset=3).dtype, torch.float64)

    def test_rotary_tables_built_under_inference_mode_can_train(self):
        model = tiny()
        batch = synthetic_batch(model.config, batch_size=2)
        with torch.inference_mode():
            model.eval()(**batch)
        model.train()(**batch).loss.backward()

    def test_dense_hint_matches_masked_path(self):
        model = tiny(moe_null_experts=1).eval()
        ids = torch.randint(4, 200, (2, 9))
        with torch.no_grad():
            dense = model(ids).logits
            masked = model(ids, attention_mask=torch.ones_like(ids)).logits
            prefix = model(ids[:, :5], use_cache=True)
            step = model(ids[:, 5:], cache=prefix.cache, use_cache=True).logits
        self.assertTrue(prefix.cache.dense)
        torch.testing.assert_close(dense, masked)
        torch.testing.assert_close(step, dense[:, 5:], rtol=1e-4, atol=1e-4)


class MuonGroupTests(unittest.TestCase):
    def test_muon_only_orthogonalises_linear_weights(self):
        from blue.optimizer import build_optimizer
        model = tiny(ttt_state_dim=8, residual_streams=2, tabular_enabled=True,
                     temporal_spectral_mixing=True, ngram_vocab_size=64)
        optimizer = build_optimizer(model, name="muon", lr=1e-3, weight_decay=0.0)
        linear = {id(module.weight) for module in model.modules() if isinstance(module, nn.Linear)}
        muon = [p for group in optimizer.param_groups if group["use_muon"] for p in group["params"]]
        self.assertTrue(muon)
        self.assertTrue(all(id(p) in linear for p in muon))
        names = {id(p): name for name, p in model.named_parameters()}
        adam = {names[id(p)] for group in optimizer.param_groups if not group["use_muon"] for p in group["params"]}
        for expected in ("modality", "blocks.0.ttt.ln_weight", "blocks.0.hc_attn.res_bias", "tabular.plr_freq"):
            self.assertIn(expected, adam)


class ObservedObjectiveTests(unittest.TestCase):
    CONFIGURATIONS = (
        {}, {"prediction_steps": 4, "mtp_decay": 0.5}, {"adaptive_loss_weighting": True},
        {"adaptive_loss_weighting": True, "prediction_steps": 3, "jepa_weight": 0.1},
        {"flow_matching_dim": 4}, {"moe_entropy_weight": 0.1, "lm_z_loss_weight": 0.01, "mtp_kl_weight": 0.1},
    )

    def test_single_microbatch_reproduces_the_model_loss(self):
        from blue.observed_objective import observed_objective, target_counts
        for changes in self.CONFIGURATIONS:
            with self.subTest(changes):
                model = tiny(**changes).eval()
                batch = synthetic_batch(model.config, batch_size=3)
                if changes.get("flow_matching_dim"):
                    batch["flow_targets"] = torch.randn(3, 4)
                with torch.no_grad():
                    output = model(**batch)
                    counts = target_counts(batch, model.config.prediction_steps)
                    observed = observed_objective(output, counts, counts, model.config, world=1, accumulation=1,
                                                  weight_task=model._weight_task_loss)
                torch.testing.assert_close(observed, output.loss, rtol=1e-6, atol=1e-6)

    def test_uncertainty_weights_are_trained_with_observed_normalization(self):
        from blue.observed_objective import observed_objective, target_counts
        model = tiny(adaptive_loss_weighting=True)
        batch = synthetic_batch(model.config, batch_size=3)
        counts = target_counts(batch, model.config.prediction_steps)
        observed_objective(model(**batch), counts, counts * 2, model.config, world=1, accumulation=2,
                           weight_task=model._weight_task_loss).backward()
        self.assertGreater(model.task_log_vars.grad.abs().sum().item(), 0)


class EmbeddingSearchCacheTests(unittest.TestCase):
    def write(self, path, count, generator):
        import json
        digest = "a" * 64
        with open(path, "w", encoding="utf-8") as stream:
            for index in range(count):
                stream.write(json.dumps(dict(
                    index=index, format_version=1, checkpoint_sha256=digest, dataset_sha256=digest,
                    dataset_images_sha256=None, alignment_text_view="joint", amp="none", device="cpu",
                    batch_size=4, torch_version="x",
                    vectors={"text": torch.randn(8, generator=generator).tolist()})) + "\n")

    def test_cached_and_streamed_candidates_give_the_same_neighbours(self):
        import tempfile
        from pathlib import Path
        from blue.embedding_search import search_embeddings
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.jsonl"
            self.write(path, 37, torch.Generator().manual_seed(0))
            common = dict(top_k=4, query_batch_size=5, candidate_batch_size=6, exclude_self=True)
            cached = list(search_embeddings(path, path, "text", "text", **common))
            streamed = list(search_embeddings(path, path, "text", "text", candidate_cache_bytes=0, **common))
            self.assertEqual(cached, streamed)
            self.assertEqual(len(cached), 37)

    def test_candidates_modified_after_caching_are_detected(self):
        import os
        import tempfile
        from pathlib import Path
        from blue.embedding_search import search_embeddings
        with tempfile.TemporaryDirectory() as directory:
            queries, candidates = Path(directory) / "q.jsonl", Path(directory) / "c.jsonl"
            self.write(queries, 6, torch.Generator().manual_seed(1))
            self.write(candidates, 6, torch.Generator().manual_seed(2))
            search = search_embeddings(queries, candidates, "text", "text", top_k=2, query_batch_size=2)
            next(search)
            stat = candidates.stat()
            os.utime(candidates, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
            with self.assertRaisesRegex(ValueError, "modifié"):
                list(search)


class QuantizationRangeTests(unittest.TestCase):
    def test_int4_scales_stay_positive_and_finite(self):
        for magnitude in (0.0, 1e-12, 1e-7, 1e-3, 1.0, 1e6):
            with self.subTest(magnitude):
                layer = nn.Linear(32, 4, bias=False)
                nn.init.constant_(layer.weight, magnitude)
                packed = Int4Linear.from_linear(layer)
                self.assertTrue((packed.weight_scale > 0).all())
                self.assertTrue(torch.isfinite(packed.weight_scale).all())
                self.assertTrue(torch.isfinite(packed.dequantized_weight()).all())


if __name__ == "__main__":
    unittest.main()
