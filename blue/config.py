from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass
class BlueConfig:
    vocab_size: int = 260
    ngram_vocab_size: int = 0
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    hidden_dim: int = 768
    kan_degree: int = 0
    n_experts: int = 8
    top_k: int = 2
    shared_experts: int = 1
    moe_null_experts: int = 0
    moe_router_geometry: str = "euclidean"
    shared_expert_gate: bool = False
    router_bias_update_rate: float = 0.0
    depth_routing_capacity: float = 1.0
    depth_budget_weight: float = 0.01
    max_seq_len: int = 2048
    sliding_window: int = 256
    global_every: int = 4
    conv_kernel: int = 5
    rope_theta: float = 10000.0
    rope_scaling_type: str = "none"
    rope_scaling_factor: float = 1.0
    rope_original_max_seq_len: int | None = None
    differential_attention: bool = False
    cross_differential_attention: bool = False
    multimodal_adaln: bool = False
    kv_lora_rank: int = 0
    cross_kv_lora_rank: int = 0
    attention_sinks: int = 0
    value_residual: bool = False
    logit_softcap: float = 0.0
    attn_logit_softcap: float = 0.0
    nsa_block_size: int = 0
    nsa_top_k: int = 4
    ssm_state_dim: int = 0
    ssm_heads: int | None = None
    ssm_chunk_size: int = 32
    ssm_every: int = 1
    ttt_state_dim: int = 0
    ttt_heads: int | None = None
    ttt_base_lr: float = 0.5
    ttt_every: int = 1
    residual_streams: int = 1
    latent_reasoning_steps: int = 1
    latent_reasoning_adaptive: bool = False
    ponder_prior: float = 0.5
    ponder_kl_weight: float = 0.01
    memory_slots: int = 0
    dropout: float = 0.0
    norm_eps: float = 1e-6
    stable_rms_norm: bool = False
    dynamic_tanh_norm: bool = False
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
    temporal_wavelet_mixing: bool = False
    audio_enabled: bool = False
    audio_channels: int = 1
    audio_patch_size: int = 16
    max_audio_len: int = 1024
    audio_layers: int = 2
    graph_enabled: bool = False
    graph_node_features: int = 8
    graph_edge_features: int = 4
    max_graph_nodes: int = 64
    graph_layers: int = 2
    graph_rwse_steps: int = 8
    pointcloud_enabled: bool = False
    pointcloud_features: int = 0
    max_pointcloud_points: int = 128
    pointcloud_layers: int = 2
    tabular_enabled: bool = False
    tabular_num_features: int = 8
    tabular_cat_features: int = 4
    tabular_cat_cardinality: int = 32
    tabular_layers: int = 2
    tabular_plr_frequencies: int = 16
    video_enabled: bool = False
    video_channels: int = 3
    video_size: int = 32
    video_patch_size: int = 8
    video_tubelet_size: int = 2
    max_video_frames: int = 16
    video_layers: int = 2
    n_latents: int = 16
    register_latents: int = 0
    cross_modal_fusion_layers: int = 0
    resampler_layers: int = 2
    hierarchical_resampler: bool = False
    latent_fsq_levels: tuple[int, ...] = ()
    latent_diversity_weight: float = 0.0
    num_classes: int = 6
    forecast_horizon: int = 12
    forecast_context: int = 32
    forecast_adapter_context: int = 0
    forecast_adapter_relative: bool = False
    forecast_adapter_missing_gate: bool = False
    forecast_adapter_separate_missing: bool = False
    forecast_tail_power: bool = False
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
    alignment_geometry: str = "euclidean"
    alignment_curvature: float = 1.0
    text_weight: float = 1.0
    classification_weight: float = 1.0
    forecast_weight: float = 1.0
    flow_matching_dim: int = 0
    flow_matching_layers: int = 2
    flow_matching_weight: float = 0.25
    flow_cfg_dropout: float = 0.0
    jepa_weight: float = 0.0
    adaptive_loss_weighting: bool = False
    gradient_checkpointing: bool = False
    checkpoint_every: int = 1
    top1_gate: str = "normalized"

    def __post_init__(self):
        integers = (
            "vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads",
            "hidden_dim", "n_experts", "top_k", "shared_experts", "max_seq_len",
            "sliding_window", "global_every", "conv_kernel", "image_size",
            "image_channels", "patch_size", "vision_layers", "series_features",
            "max_series_len", "temporal_layers", "audio_channels", "audio_patch_size",
            "max_audio_len", "audio_layers", "graph_node_features", "graph_edge_features",
            "max_graph_nodes", "graph_layers", "graph_rwse_steps",
            "max_pointcloud_points", "pointcloud_layers",
            "tabular_num_features", "tabular_layers", "tabular_plr_frequencies",
            "video_channels", "video_size", "video_patch_size", "video_tubelet_size",
            "max_video_frames", "video_layers",
            "n_latents", "resampler_layers",
            "num_classes", "forecast_horizon", "forecast_context", "prediction_steps",
            "flow_matching_layers",
        )
        for name in integers:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} doit être un entier strictement positif")
        if self.vocab_size < 260:
            raise ValueError("vocab_size doit être >= 260 pour le tokenizer UTF-8")
        if type(self.ngram_vocab_size) is not int or (self.ngram_vocab_size != 0 and self.ngram_vocab_size < 16):
            raise ValueError("ngram_vocab_size doit être 0 ou un entier >= 16")
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("d_model / n_heads doit être un entier pair (RoPE)")
        if type(self.differential_attention) is not bool:
            raise ValueError("differential_attention doit être booléen")
        if self.differential_attention and (self.d_model // self.n_heads) % 4:
            raise ValueError("differential_attention nécessite d_model / n_heads divisible par 4")
        if type(self.cross_differential_attention) is not bool:
            raise ValueError("cross_differential_attention doit être booléen")
        if self.cross_differential_attention and (self.d_model // self.n_heads) % 4:
            raise ValueError("cross_differential_attention nécessite d_model / n_heads divisible par 4")
        if type(self.multimodal_adaln) is not bool:
            raise ValueError("multimodal_adaln doit être booléen")
        if type(self.kv_lora_rank) is not int or self.kv_lora_rank < 0:
            raise ValueError("kv_lora_rank doit être un entier positif ou nul")
        if type(self.cross_kv_lora_rank) is not int or self.cross_kv_lora_rank < 0:
            raise ValueError("cross_kv_lora_rank doit être un entier positif ou nul")
        if self.kv_lora_rank > 0:
            head_dim = self.d_model // self.n_heads
            if head_dim % 4:
                raise ValueError("kv_lora_rank nécessite d_model / n_heads divisible par 4")
            if self.differential_attention and head_dim % 8:
                raise ValueError("kv_lora_rank + differential_attention nécessite d_model / n_heads divisible par 8")
        if type(self.attention_sinks) is not int or self.attention_sinks < 0:
            raise ValueError("attention_sinks doit être un entier positif ou nul")
        if type(self.value_residual) is not bool:
            raise ValueError("value_residual doit être booléen")
        if not math.isfinite(self.logit_softcap) or self.logit_softcap < 0:
            raise ValueError("logit_softcap doit être fini et positif ou nul")
        if not math.isfinite(self.attn_logit_softcap) or self.attn_logit_softcap < 0:
            raise ValueError("attn_logit_softcap doit être fini et positif ou nul")
        if type(self.nsa_block_size) is not int or (self.nsa_block_size != 0 and self.nsa_block_size < 2):
            raise ValueError("nsa_block_size doit être 0 ou un entier >= 2")
        if type(self.nsa_top_k) is not int or self.nsa_top_k < 1:
            raise ValueError("nsa_top_k doit être un entier >= 1")
        if type(self.ssm_state_dim) is not int or self.ssm_state_dim < 0:
            raise ValueError("ssm_state_dim doit être un entier positif ou nul")
        if self.ssm_heads is not None:
            if type(self.ssm_heads) is not int or self.ssm_heads < 1:
                raise ValueError("ssm_heads doit être un entier strictement positif")
            if self.ssm_state_dim == 0:
                raise ValueError("ssm_heads nécessite ssm_state_dim > 0")
            if self.d_model % self.ssm_heads:
                raise ValueError("d_model doit être divisible par ssm_heads")
        if type(self.ssm_chunk_size) is not int or self.ssm_chunk_size < 1:
            raise ValueError("ssm_chunk_size doit être un entier strictement positif")
        if type(self.ssm_every) is not int or self.ssm_every < 1:
            raise ValueError("ssm_every doit être un entier strictement positif")
        if type(self.ttt_state_dim) is not int or self.ttt_state_dim < 0:
            raise ValueError("ttt_state_dim doit être un entier positif ou nul")
        if self.ttt_heads is not None:
            if type(self.ttt_heads) is not int or self.ttt_heads < 1:
                raise ValueError("ttt_heads doit être un entier strictement positif")
            if self.ttt_state_dim == 0:
                raise ValueError("ttt_heads nécessite ttt_state_dim > 0")
        if not math.isfinite(self.ttt_base_lr) or self.ttt_base_lr <= 0:
            raise ValueError("ttt_base_lr doit être fini et strictement positif")
        if type(self.ttt_every) is not int or self.ttt_every < 1:
            raise ValueError("ttt_every doit être un entier strictement positif")
        if type(self.residual_streams) is not int or self.residual_streams < 1:
            raise ValueError("residual_streams doit être un entier strictement positif")
        if type(self.latent_reasoning_steps) is not int or self.latent_reasoning_steps < 1:
            raise ValueError("latent_reasoning_steps doit être un entier strictement positif")
        if type(self.latent_reasoning_adaptive) is not bool:
            raise ValueError("latent_reasoning_adaptive doit être booléen")
        if self.latent_reasoning_adaptive and self.latent_reasoning_steps < 2:
            raise ValueError("latent_reasoning_adaptive nécessite latent_reasoning_steps >= 2")
        if not math.isfinite(self.ponder_prior) or not 0.0 < self.ponder_prior < 1.0:
            raise ValueError("ponder_prior doit être fini et strictement compris entre 0 et 1")
        if not math.isfinite(self.ponder_kl_weight) or self.ponder_kl_weight < 0:
            raise ValueError("ponder_kl_weight doit être fini et positif ou nul")
        if type(self.memory_slots) is not int or self.memory_slots < 0:
            raise ValueError("memory_slots doit être un entier positif ou nul")
        if self.d_model % 4 or self.n_heads % self.n_kv_heads:
            raise ValueError("d_model doit être divisible par 4 et n_heads par n_kv_heads")
        if self.top_k > self.n_experts:
            raise ValueError("top_k ne peut pas dépasser n_experts")
        if type(self.moe_null_experts) is not int or self.moe_null_experts < 0:
            raise ValueError("moe_null_experts doit être un entier positif ou nul")
        if self.moe_router_geometry not in ("euclidean", "hyperbolic"):
            raise ValueError("moe_router_geometry doit être euclidean ou hyperbolic")
        if self.top1_gate not in ("normalized", "probability"):
            raise ValueError("top1_gate doit être normalized ou probability")
        if self.top1_gate == "probability" and (self.top_k != 1 or self.n_experts < 2):
            raise ValueError("top1_gate probability nécessite top_k=1 et au moins deux experts")
        if type(self.shared_expert_gate) is not bool:
            raise ValueError("shared_expert_gate doit être booléen")
        if not math.isfinite(self.router_bias_update_rate) or self.router_bias_update_rate < 0:
            raise ValueError("router_bias_update_rate doit être fini et positif ou nul")
        if not math.isfinite(self.depth_routing_capacity) or not 0.0 < self.depth_routing_capacity <= 1.0:
            raise ValueError("depth_routing_capacity doit être fini et appartenir à ]0, 1]")
        if not math.isfinite(self.depth_budget_weight) or self.depth_budget_weight < 0:
            raise ValueError("depth_budget_weight doit être fini et positif ou nul")
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
        if type(self.dynamic_tanh_norm) is not bool:
            raise ValueError("dynamic_tanh_norm doit être booléen")
        if type(self.kan_degree) is not int or (self.kan_degree != 0 and not 2 <= self.kan_degree <= 8):
            raise ValueError("kan_degree doit être 0 ou un entier entre 2 et 8")
        if self.alignment_mode not in ("paired", "class"):
            raise ValueError("alignment_mode doit être paired ou class")
        if self.alignment_scope not in ("local", "global"):
            raise ValueError("alignment_scope doit être local ou global")
        if self.alignment_text_view not in ("joint", "isolated"):
            raise ValueError("alignment_text_view doit être joint ou isolated")
        if not math.isfinite(self.alignment_temperature) or self.alignment_temperature <= 0:
            raise ValueError("alignment_temperature doit être finie et > 0")
        if self.alignment_geometry not in ("euclidean", "hyperbolic"):
            raise ValueError("alignment_geometry doit être euclidean ou hyperbolic")
        if not math.isfinite(self.alignment_curvature) or self.alignment_curvature <= 0:
            raise ValueError("alignment_curvature doit être fini et > 0")
        self.quantiles = tuple(self.quantiles)
        if not self.quantiles or any(not 0 < q < 1 for q in self.quantiles):
            raise ValueError("Les quantiles doivent appartenir à ]0, 1[")
        if tuple(sorted(set(self.quantiles))) != self.quantiles:
            raise ValueError("Les quantiles doivent être strictement croissants")
        if 0.5 not in self.quantiles:
            raise ValueError("Le quantile médian 0.5 est requis")
        for name in ("balance_weight", "router_z_weight", "mtp_weight", "alignment_weight",
                     "text_weight", "classification_weight", "forecast_weight",
                     "flow_matching_weight", "jepa_weight", "latent_diversity_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} doit être fini et positif ou nul")
        if type(self.flow_matching_dim) is not int or self.flow_matching_dim < 0:
            raise ValueError("flow_matching_dim doit être un entier positif ou nul")
        if not math.isfinite(self.flow_cfg_dropout) or not 0.0 <= self.flow_cfg_dropout < 1.0:
            raise ValueError("flow_cfg_dropout doit être fini et appartenir à [0, 1[")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing doit être booléen")
        if type(self.checkpoint_every) is not int or self.checkpoint_every < 1:
            raise ValueError("checkpoint_every doit être un entier strictement positif")
        if type(self.adaptive_loss_weighting) is not bool:
            raise ValueError("adaptive_loss_weighting doit être booléen")
        if type(self.temporal_statistics) is not bool:
            raise ValueError("temporal_statistics doit être booléen")
        if type(self.temporal_statistics_relative) is not bool:
            raise ValueError("temporal_statistics_relative doit être booléen")
        if type(self.temporal_spectral_mixing) is not bool:
            raise ValueError("temporal_spectral_mixing doit être booléen")
        if type(self.temporal_wavelet_mixing) is not bool:
            raise ValueError("temporal_wavelet_mixing doit être booléen")
        if type(self.audio_enabled) is not bool:
            raise ValueError("audio_enabled doit être booléen")
        if type(self.graph_enabled) is not bool:
            raise ValueError("graph_enabled doit être booléen")
        if type(self.pointcloud_enabled) is not bool:
            raise ValueError("pointcloud_enabled doit être booléen")
        if type(self.pointcloud_features) is not int or self.pointcloud_features < 0:
            raise ValueError("pointcloud_features doit être un entier positif ou nul")
        if type(self.tabular_enabled) is not bool:
            raise ValueError("tabular_enabled doit être booléen")
        if type(self.tabular_cat_features) is not int or self.tabular_cat_features < 0:
            raise ValueError("tabular_cat_features doit être un entier positif ou nul")
        if type(self.tabular_cat_cardinality) is not int or self.tabular_cat_cardinality < 2:
            raise ValueError("tabular_cat_cardinality doit être un entier >= 2")
        if type(self.video_enabled) is not bool:
            raise ValueError("video_enabled doit être booléen")
        if self.video_size % self.video_patch_size:
            raise ValueError("video_size doit être divisible par video_patch_size")
        if type(self.hierarchical_resampler) is not bool:
            raise ValueError("hierarchical_resampler doit être booléen")
        if type(self.register_latents) is not int or self.register_latents < 0:
            raise ValueError("register_latents doit être un entier positif ou nul")
        if type(self.cross_modal_fusion_layers) is not int or self.cross_modal_fusion_layers < 0:
            raise ValueError("cross_modal_fusion_layers doit être un entier positif ou nul")
        self.latent_fsq_levels = tuple(self.latent_fsq_levels)
        if any(type(level) is not int or level < 2 for level in self.latent_fsq_levels):
            raise ValueError("latent_fsq_levels doit contenir des entiers >= 2")
        if type(self.forecast_adapter_relative) is not bool:
            raise ValueError("forecast_adapter_relative doit être booléen")
        if type(self.forecast_adapter_missing_gate) is not bool:
            raise ValueError("forecast_adapter_missing_gate doit être booléen")
        if type(self.forecast_adapter_separate_missing) is not bool:
            raise ValueError("forecast_adapter_separate_missing doit être booléen")
        if type(self.forecast_tail_power) is not bool:
            raise ValueError("forecast_tail_power doit être booléen")
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
            "hyper": dict(d_model=768, n_layers=16, n_heads=12, n_kv_heads=4,
                          ngram_vocab_size=8192,
                          hidden_dim=2048, kan_degree=3, n_experts=16, top_k=2,
                          shared_experts=1, moe_null_experts=2,
                          moe_router_geometry="hyperbolic", shared_expert_gate=True,
                          router_bias_update_rate=0.001,
                          depth_routing_capacity=0.75,
                          max_seq_len=8192, sliding_window=512,
                          rope_scaling_type="yarn", rope_scaling_factor=2.0,
                          differential_attention=True, cross_differential_attention=True,
                          multimodal_adaln=True,
                          kv_lora_rank=192, cross_kv_lora_rank=192,
                          attention_sinks=4, value_residual=True,
                          logit_softcap=30.0, attn_logit_softcap=50.0,
                          nsa_block_size=16, nsa_top_k=4,
                          ssm_state_dim=32, ssm_heads=12, ssm_chunk_size=32, ssm_every=1,
                          ttt_state_dim=16, ttt_heads=12, ttt_every=2,
                          residual_streams=4, latent_reasoning_steps=2,
                          latent_reasoning_adaptive=True,
                          memory_slots=32,
                          image_size=224, patch_size=16, vision_layers=4,
                          vision_rope_2d=True, temporal_layers=4,
                          temporal_statistics=True, temporal_statistics_relative=True,
                          temporal_spectral_mixing=True, temporal_wavelet_mixing=True,
                          forecast_tail_power=True,
                          audio_enabled=True, audio_channels=1, audio_patch_size=16,
                          max_audio_len=4096, audio_layers=4,
                          graph_enabled=True, graph_node_features=16, graph_edge_features=8,
                          max_graph_nodes=128, graph_layers=4, graph_rwse_steps=8,
                          pointcloud_enabled=True, pointcloud_features=4,
                          max_pointcloud_points=128, pointcloud_layers=4,
                          tabular_enabled=True, tabular_num_features=16,
                          tabular_cat_features=8, tabular_cat_cardinality=64,
                          tabular_layers=3, tabular_plr_frequencies=16,
                          video_enabled=True, video_channels=3, video_size=64,
                          video_patch_size=8, video_tubelet_size=2,
                          max_video_frames=32, video_layers=4,
                          flow_matching_dim=36, flow_matching_layers=3, flow_cfg_dropout=0.1,
                          jepa_weight=0.1, adaptive_loss_weighting=True,
                          n_latents=48, register_latents=8,
                          cross_modal_fusion_layers=2, resampler_layers=3,
                          hierarchical_resampler=True,
                          latent_fsq_levels=(8, 5, 5, 5),
                          latent_diversity_weight=0.02,
                          alignment_geometry="hyperbolic",
                          prediction_steps=4, stable_rms_norm=True,
                          gradient_checkpointing=True, checkpoint_every=2),
            "titan": dict(d_model=6144, n_layers=80, n_heads=48, n_kv_heads=8,
                          ngram_vocab_size=65536,
                          hidden_dim=16384, kan_degree=4, n_experts=64, top_k=4,
                          shared_experts=2, moe_null_experts=8,
                          moe_router_geometry="hyperbolic", shared_expert_gate=True,
                          router_bias_update_rate=0.001,
                          depth_routing_capacity=0.75,
                          max_seq_len=65536, sliding_window=2048,
                          rope_scaling_type="yarn", rope_scaling_factor=4.0,
                          differential_attention=True, cross_differential_attention=True,
                          multimodal_adaln=True,
                          kv_lora_rank=512, cross_kv_lora_rank=512,
                          attention_sinks=8, value_residual=True,
                          logit_softcap=30.0, attn_logit_softcap=50.0,
                          nsa_block_size=32, nsa_top_k=8,
                          ssm_state_dim=64, ssm_heads=48, ssm_chunk_size=64, ssm_every=1,
                          ttt_state_dim=32, ttt_heads=48, ttt_every=2,
                          residual_streams=4, latent_reasoning_steps=2,
                          latent_reasoning_adaptive=True,
                          memory_slots=64,
                          image_size=448, patch_size=16, vision_layers=16,
                          vision_rope_2d=True, temporal_layers=10,
                          temporal_statistics=True, temporal_statistics_relative=True,
                          temporal_spectral_mixing=True, temporal_wavelet_mixing=True,
                          forecast_tail_power=True,
                          audio_enabled=True, audio_channels=2, audio_patch_size=16,
                          max_audio_len=16384, audio_layers=8,
                          graph_enabled=True, graph_node_features=32, graph_edge_features=16,
                          max_graph_nodes=256, graph_layers=8, graph_rwse_steps=12,
                          pointcloud_enabled=True, pointcloud_features=8,
                          max_pointcloud_points=256, pointcloud_layers=8,
                          tabular_enabled=True, tabular_num_features=32,
                          tabular_cat_features=16, tabular_cat_cardinality=128,
                          tabular_layers=6, tabular_plr_frequencies=32,
                          video_enabled=True, video_channels=3, video_size=128,
                          video_patch_size=16, video_tubelet_size=2,
                          max_video_frames=64, video_layers=10,
                          flow_matching_dim=128, flow_matching_layers=6, flow_cfg_dropout=0.1,
                          jepa_weight=0.1, adaptive_loss_weighting=True,
                          n_latents=192, register_latents=32,
                          cross_modal_fusion_layers=4, resampler_layers=8,
                          hierarchical_resampler=True,
                          latent_fsq_levels=(8, 8, 5, 5, 5),
                          latent_diversity_weight=0.02,
                          alignment_geometry="hyperbolic",
                          prediction_steps=4, stable_rms_norm=True,
                          gradient_checkpointing=True, checkpoint_every=2),
        }
        if name not in presets:
            raise ValueError(f"Preset inconnu : {name}. Choix : {', '.join(presets)}")
        return cls(**presets[name])
