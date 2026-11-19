from dataclasses import replace
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch
from blue.quantize import Int8Linear, quantize_model_int8


class Int8QuantizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_int8_quantization_reduces_memory_and_preserves_outputs_on_full_hybrid(self):
        torch.manual_seed(21001)
        config = replace(
            BlueConfig.preset("tiny"),
            max_seq_len=128,
            differential_attention=True,
            kv_lora_rank=16,
            attention_sinks=2,
            ssm_state_dim=8,
            residual_streams=2,
            latent_reasoning_steps=2,
            memory_slots=4,
            audio_enabled=True,
            audio_patch_size=8,
            audio_layers=1,
            graph_enabled=True,
            graph_node_features=6,
            graph_edge_features=3,
            graph_layers=1,
            pointcloud_enabled=True,
            pointcloud_features=2,
            pointcloud_layers=1,
            hierarchical_resampler=True,
            flow_matching_dim=12,
            flow_matching_layers=1,
            stable_rms_norm=True,
        )
        model = Blue(config).eval()
        batch = synthetic_batch(config, batch_size=2, seed=21002)
        inputs = {k: v for k, v in batch.items() if k not in ("labels", "class_labels", "forecast_targets")}
        with torch.no_grad():
            ref = model(**inputs)
            ref_flow = model.sample_flow(steps=4, solver="euler",
                                         initial_noise=torch.zeros(2, 12), **inputs)

        report = quantize_model_int8(model, min_numel=1024)
        self.assertGreater(report["quantized_modules"], 10)
        self.assertGreater(report["compression_ratio"], 3.5)
        self.assertLess(report["max_abs_weight_error"], 1e-3)
        self.assertTrue(any(isinstance(m, Int8Linear) for m in model.modules()))

        with torch.no_grad():
            quant = model(**inputs)
            quant_flow = model.sample_flow(steps=4, solver="euler",
                                           initial_noise=torch.zeros(2, 12), **inputs)
        # Quantized outputs stay close to FP32 reference across every head.
        torch.testing.assert_close(quant.logits, ref.logits, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(quant.classification_logits, ref.classification_logits, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(quant.forecast, ref.forecast, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(quant_flow, ref_flow, rtol=5e-2, atol=5e-2)

        # Cached generation still works cleanly after INT8 quantization.
        gen = model.generate(inputs["input_ids"][:1, :6], max_new_tokens=5, temperature=0)
        self.assertEqual(gen.shape, (1, 11))


if __name__ == "__main__":
    unittest.main()
