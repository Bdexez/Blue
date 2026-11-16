from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class RelativeTemporalStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_translation_invariance_padding_missing_features_and_backward(self):
        torch.manual_seed(7041)
        config = replace(BlueConfig.preset("tiny"), temporal_statistics=True, temporal_statistics_relative=True,
                         gradient_checkpointing=True)
        model = Blue(config).eval()
        torch.nn.init.normal_(model.temporal.statistics_proj.weight, std=0.02)
        batch = synthetic_batch(config, batch_size=2, seed=7042)
        series = batch["series"][:, -16:].clone()
        series[0, :, 1] = float("nan")
        series[1, -2:, 0] = float("nan")
        valid = torch.ones(series.shape[:2], dtype=torch.bool)
        valid[1, 3] = False
        captured = []
        hook = model.temporal.statistics_proj.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        with torch.inference_mode():
            before = model(batch["input_ids"], series=series, series_mask=valid)
            after = model(batch["input_ids"], series=series + 20, series_mask=valid)
            padded = model(batch["input_ids"], series=torch.nn.functional.pad(series, (0, 0, 3, 4), value=999),
                           series_mask=torch.nn.functional.pad(valid, (3, 4), value=False))
            model.temporal(series[1, valid[1]][None], torch.ones(1, valid[1].sum(), dtype=torch.bool))
        hook.remove()
        torch.testing.assert_close(captured[0], captured[1], rtol=2e-4, atol=1e-5)
        torch.testing.assert_close(captured[0][1], captured[3][0], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(before.classification_logits, after.classification_logits, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(before.logits, after.logits, rtol=1e-4, atol=1e-5)
        observed_feature = (torch.isfinite(series) & valid.unsqueeze(-1)).any(1)
        torch.testing.assert_close(before.forecast + 20 * observed_feature[:, None, :, None], after.forecast,
                                   rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(before.forecast, padded.forecast, rtol=2e-5, atol=2e-6)
        self.assertTrue(torch.isfinite(captured[0]).all())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model.train()(series=series, series_mask=valid, class_labels=batch["class_labels"],
                                 forecast_targets=batch["forecast_targets"]).loss
        loss.backward()
        gradient = model.temporal.statistics_proj.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            replace(config, temporal_statistics_relative=1)

    def test_feature_mode_transfer_resets_only_statistics_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            model = Blue(replace(BlueConfig.preset("tiny"), temporal_statistics=True))
            torch.nn.init.normal_(model.temporal.statistics_proj.weight, std=0.01)
            save_checkpoint(source, model)
            relative, _ = load_checkpoint(source, overrides={"temporal_statistics_relative": True})
            self.assertEqual(relative.temporal.statistics_proj.weight.count_nonzero().item(), 0)
            for name, value in model.state_dict().items():
                if name != "temporal.statistics_proj.weight":
                    torch.testing.assert_close(relative.state_dict()[name], value, rtol=0, atol=0)
            torch.nn.init.normal_(relative.temporal.statistics_proj.weight, std=0.01)
            destination = Path(directory) / "relative.pt"
            save_checkpoint(destination, relative)
            restored, _ = load_checkpoint(destination)
            self.assertTrue(restored.temporal.statistics_relative)
            torch.testing.assert_close(restored.temporal.statistics_proj.weight, relative.temporal.statistics_proj.weight)
            payload = torch.load(source, weights_only=True)
            payload["model"].pop("temporal.statistics_proj.weight")
            torch.save(payload, Path(directory) / "broken.pt")
            with self.assertRaises(KeyError):
                load_checkpoint(Path(directory) / "broken.pt", overrides={"temporal_statistics_relative": True})


if __name__ == "__main__":
    unittest.main()
