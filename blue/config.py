from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass
class BlueConfig:
    vocab_size: int = 260
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    hidden_dim: int = 768
    n_experts: int = 8
    top_k: int = 2
    shared_experts: int = 1
    shared_expert_gate: bool = False
    router_bias_update_rate: float = 0.0
    max_seq_len: int = 2048
    sliding_window: int = 256
    global_every: int = 4
    conv_kernel: int = 5
    rope_theta: float = 10000.0
    rope_scaling_type: str = "none"
    rope_scaling_factor: float = 1.0
    rope_original_max_seq_len: int | None = None
    differential_attention: bool = False
    kv_lora_rank: int = 0
    ssm_state_dim: int = 0
    ssm_heads: int | None = None
    residual_streams: int = 1
    latent_reasoning_steps: int = 1
    dropout: float = 0.0
    norm_eps: float = 1e-6
    stable_rms_norm: bool = False
    image_size: int = 64
    image_channels: int = 3
    patch_size: int = 8
    vision_layers: int = 2
    vision_rope_2d: bool = False
    series_features: int = 3
    max_series_len: int = 256
    temporal_layers: int = 2
    temporal_statistics: bool = False
    temporal_statistics_relative: bool = False
    temporal_spectral_mixing: bool = False
    audio_enabled: bool = False
    audio_channels: int = 1
    audio_patch_size: int = 16
    max_audio_len: int = 1024
    audio_layers: int = 2
    n_latents: int = 16
    resampler_layers: int = 2
    num_classes: int = 6
    forecast_horizon: int = 12
    forecast_context: int = 32
    forecast_adapter_context: int = 0
    forecast_adapter_relative: bool = False
    forecast_adapter_missing_gate: bool = False
    forecast_adapter_separate_missing: bool = False
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    prediction_steps: int = 2
    balance_weight: float = 0.01
    router_z_weight: float = 0.001
    mtp_weight: float = 0.1
    alignment_weight: float = 0.05
    alignment_mode: str = "paired"
    alignment_scope: str = "local"
    alignment_text_view: str = "joint"
    alignment_temperature: float = 0.1
    text_weight: float = 1.0
    classification_weight: float = 1.0
    forecast_weight: float = 1.0
    gradient_checkpointing: bool = False
    top1_gate: str = "normalized"

    def __post_init__(self):
        integers = (
            "vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads",
            "hidden_dim", "n_experts", "top_k", "shared_experts", "max_seq_len",
            "sliding_window", "global_every", "conv_kernel", "image_size",
            "image_channels", "patch_size", "vision_layers", "series_features",
            "max_series_len", "temporal_layers", "audio_channels", "audio_patch_size",
            "max_audio_len", "audio_layers", "n_latents", "resampler_layers",
            "num_classes", "forecast_horizon", "forecast_context", "prediction_steps",
        )
        for name in integers:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} doit être un entier strictement positif")
        if self.vocab_size < 260:
            raise ValueError("vocab_size doit être >= 260 pour le tokenizer UTF-8")
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("d_model / n_heads doit être un entier pair (RoPE)")
        if type(self.differential_attention) is not bool:
            raise ValueError("differential_attention doit être booléen")
        if self.differential_attention and (self.d_model // self.n_heads) % 4:
            raise ValueError("differential_attention nécessite d_model / n_heads divisible par 4")
        if type(self.kv_lora_rank) is not int or self.kv_lora_rank < 0:
            raise ValueError("kv_lora_rank doit être un entier positif ou nul")
        if self.kv_lora_rank > 0:
            head_dim = self.d_model // self.n_heads
            if head_dim % 4:
                raise ValueError("kv_lora_rank nécessite d_model / n_heads divisible par 4")
            if self.differential_attention and head_dim % 8:
                raise ValueError("kv_lora_rank + differential_attention nécessite d_model / n_heads divisible par 8")
        if type(self.ssm_state_dim) is not int or self.ssm_state_dim < 0:
            raise ValueError("ssm_state_dim doit être un entier positif ou nul")
        if self.ssm_heads is not None:
            if type(self.ssm_heads) is not int or self.ssm_heads < 1:
                raise ValueError("ssm_heads doit être un entier strictement positif")
            if self.ssm_state_dim == 0:
                raise ValueError("ssm_heads nécessite ssm_state_dim > 0")
            if self.d_model % self.ssm_heads:
                raise ValueError("d_model doit être divisible par ssm_heads")
        if type(self.residual_streams) is not int or self.residual_streams < 1:
            raise ValueError("residual_streams doit être un entier strictement positif")
        if type(self.latent_reasoning_steps) is not int or self.latent_reasoning_steps < 1:
            raise ValueError("latent_reasoning_steps doit être un entier strictement positif")
        if self.d_model % 4 or self.n_heads % self.n_kv_heads:
            raise ValueError("d_model doit être divisible par 4 et n_heads par n_kv_heads")
        if self.top_k > self.n_experts:
            raise ValueError("top_k ne peut pas dépasser n_experts")
        if self.top1_gate not in ("normalized", "probability"):
            raise ValueError("top1_gate doit être normalized ou probability")
        if self.top1_gate == "probability" and (self.top_k != 1 or self.n_experts < 2):
            raise ValueError("top1_gate probability nécessite top_k=1 et au moins deux experts")
        if type(self.shared_expert_gate) is not bool:
            raise ValueError("shared_expert_gate doit être booléen")
        if not math.isfinite(self.router_bias_update_rate) or self.router_bias_update_rate < 0:
            raise ValueError("router_bias_update_rate doit être fini et positif ou nul")
        if self.image_size % self.patch_size:
            raise ValueError("image_size doit être divisible par patch_size")
        if type(self.vision_rope_2d) is not bool:
            raise ValueError("vision_rope_2d doit être booléen")
        if self.vision_rope_2d:
            rope_dim = self.d_model // self.n_heads
            if self.kv_lora_rank > 0:
                rope_dim //= 2
            if self.differential_attention:
                rope_dim //= 2
            if rope_dim < 4 or rope_dim % 4:
                raise ValueError("vision_rope_2d nécessite une dimension rotative par tête divisible par 4")
        if (type(self.forecast_adapter_context) is not int
                or not 0 <= self.forecast_adapter_context <= self.forecast_context):
            raise ValueError("forecast_adapter_context doit être un entier entre 0 et forecast_context")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout doit appartenir à [0, 1[")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 1:
            raise ValueError("rope_theta doit être fini et > 1")
        if self.rope_scaling_type not in ("none", "linear", "ntk", "yarn"):
            raise ValueError("rope_scaling_type doit être none, linear, ntk ou yarn")
        if not math.isfinite(self.rope_scaling_factor) or self.rope_scaling_factor < 1.0:
            raise ValueError("rope_scaling_factor doit être fini et >= 1")
        if self.rope_scaling_type == "none" and self.rope_scaling_factor != 1.0:
            raise ValueError("rope_scaling_factor > 1 nécessite rope_scaling_type != none")
        if self.rope_scaling_type != "none" and self.rope_scaling_factor <= 1.0:
            raise ValueError("rope_scaling_type != none nécessite rope_scaling_factor > 1")
        if self.rope_original_max_seq_len is not None:
            if (type(self.rope_original_max_seq_len) is not int
                    or not 1 <= self.rope_original_max_seq_len <= self.max_seq_len):
                raise ValueError("rope_original_max_seq_len doit être un entier dans [1, max_seq_len]")
            if self.rope_scaling_type != "yarn":
                raise ValueError("rope_original_max_seq_len nécessite rope_scaling_type yarn")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps doit être fini et > 0")
        if type(self.stable_rms_norm) is not bool:
            raise ValueError("stable_rms_norm doit être booléen")
        if self.alignment_mode not in ("paired", "class"):
            raise ValueError("alignment_mode doit être paired ou class")
        if self.alignment_scope not in ("local", "global"):
            raise ValueError("alignment_scope doit être local ou global")
        if self.alignment_text_view not in ("joint", "isolated"):
            raise ValueError("alignment_text_view doit être joint ou isolated")
        if not math.isfinite(self.alignment_temperature) or self.alignment_temperature <= 0:
            raise ValueError("alignment_temperature doit être finie et > 0")
        self.quantiles = tuple(self.quantiles)
        if not self.quantiles or any(not 0 < q < 1 for q in self.quantiles):
            raise ValueError("Les quantiles doivent appartenir à ]0, 1[")
        if tuple(sorted(set(self.quantiles))) != self.quantiles:
            raise ValueError("Les quantiles doivent être strictement croissants")
        if 0.5 not in self.quantiles:
            raise ValueError("Le quantile médian 0.5 est requis")
        for name in ("balance_weight", "router_z_weight", "mtp_weight", "alignment_weight",
                     "text_weight", "classification_weight", "forecast_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} doit être fini et positif ou nul")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing doit être booléen")
        if type(self.temporal_statistics) is not bool:
            raise ValueError("temporal_statistics doit être booléen")
        if type(self.temporal_statistics_relative) is not bool:
            raise ValueError("temporal_statistics_relative doit être booléen")
        if type(self.temporal_spectral_mixing) is not bool:
            raise ValueError("temporal_spectral_mixing doit être booléen")
        if type(self.audio_enabled) is not bool:
            raise ValueError("audio_enabled doit être booléen")
        if type(self.forecast_adapter_relative) is not bool:
            raise ValueError("forecast_adapter_relative doit être booléen")
        if type(self.forecast_adapter_missing_gate) is not bool:
            raise ValueError("forecast_adapter_missing_gate doit être booléen")
        if type(self.forecast_adapter_separate_missing) is not bool:
            raise ValueError("forecast_adapter_separate_missing doit être booléen")
        if self.forecast_adapter_separate_missing and not (self.forecast_adapter_missing_gate
                                                           and self.forecast_adapter_context):
            raise ValueError("forecast_adapter_separate_missing nécessite l'adaptateur et sa porte de valeurs manquantes")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Clés de configuration inconnues : {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def preset(cls, name):
        presets = {
            "tiny": dict(d_model=64, n_layers=3, n_heads=4, n_kv_heads=2,
                         hidden_dim=128, n_experts=4, max_seq_len=256,
                         sliding_window=32, global_every=2, image_size=32,
                         patch_size=4, vision_layers=1, temporal_layers=1,
                         n_latents=8, resampler_layers=1, max_series_len=128),
            "small": {},
            "workstation": dict(d_model=512, n_layers=12, n_heads=8, n_kv_heads=2,
                                hidden_dim=1536, n_experts=16, n_latents=32,
                                vision_layers=3, temporal_layers=3, resampler_layers=2,
                                gradient_checkpointing=True),
            "base": dict(d_model=768, n_layers=16, n_heads=12, n_kv_heads=4,
                         hidden_dim=2048, n_experts=16, image_size=224,
                         patch_size=16, vision_layers=4, temporal_layers=4,
                         n_latents=32, resampler_layers=3, max_seq_len=4096),
            "large": dict(d_model=2048, n_layers=32, n_heads=16, n_kv_heads=4,
                          hidden_dim=5632, n_experts=32, image_size=224,
                          patch_size=16, vision_layers=8, temporal_layers=6,
                          n_latents=64, resampler_layers=4, max_seq_len=8192,
                          gradient_checkpointing=True),
            "ultra": dict(d_model=4096, n_layers=64, n_heads=32, n_kv_heads=8,
                          hidden_dim=11008, n_experts=64, top_k=4,
                          shared_experts=2, image_size=448, patch_size=16,
                          vision_layers=12, temporal_layers=8, n_latents=128,
                          resampler_layers=6, max_seq_len=32768,
                          sliding_window=1024, gradient_checkpointing=True),
        }
        if name not in presets:
            raise ValueError(f"Preset inconnu : {name}. Choix : {', '.join(presets)}")
        return cls(**presets[name])
