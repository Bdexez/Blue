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
