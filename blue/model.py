import dataclasses
from dataclasses import dataclass
import math
from typing import NamedTuple

import torch
from torch import nn
from torch import distributed as dist
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .audio import AudioEncoder
from .config import BlueConfig
from .distributed_alignment import distributed_alignment_loss
from .flow_matching import ConditionalFlowHead
from .graph import GraphEncoder
from .hyperbolic import poincare_exp0, poincare_pairwise_distance
from .jepa import MultimodalJEPAPredictor
from .losses import quantile_pinball_loss, symmetric_contrastive_loss
from .memory import EpisodicMemoryBank
from .forecast_adapter import ForecastAdapter, ForecastAdapterContext
from .modules import (
    CrossAttention, DecoderBlock, LayerCache, LatentResampler, QuantileForecastHead,
    RMSNorm, TemporalEncoder, VisionEncoder,
)
from .ngram import ByteNGramEmbedding
from .pointcloud import PointCloudEncoder
from .tabular import TabularEncoder
from .video import VideoEncoder


@dataclass
class ModelCache:
    layers: list[LayerCache]
    latents: torch.Tensor
    offset: int
    series_mean: torch.Tensor | None = None
    series_scale: torch.Tensor | None = None
    forecast_baseline: torch.Tensor | None = None
    forecast_adapter_context: ForecastAdapterContext | None = None
    episodic_memory: torch.Tensor | None = None
    ngram_tail: torch.Tensor | None = None
    ngram_tail_valid: torch.Tensor | None = None
    # Toutes les positions déjà mises en cache sont valides (None : inconnu).
    dense: bool | None = None


@dataclass
class BlueOutput:
    logits: torch.Tensor | None
    classification_logits: torch.Tensor
    forecast: torch.Tensor
    loss: torch.Tensor | None
    losses: dict[str, torch.Tensor]
    expert_usage: torch.Tensor
    cache: ModelCache | None = None
    auxiliary_text_losses: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
    routed_tokens: torch.Tensor | None = None
    last_hidden: torch.Tensor | None = None


class EncodedInput(NamedTuple):
    """Jetons d'une modalité d'entrée, leur validité et leur résumé pour l'alignement."""

    name: str
    alignment_index: int
    tokens: torch.Tensor
    valid: torch.Tensor
    pooled: torch.Tensor


def _masked_mean(tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return (tokens * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)


def masked_cross_entropy(logits, labels, label_smoothing: float = 0.0):
    # Un lot sans cible valide ne produit ni NaN ni apprentissage artificiel.
    values = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                             labels.reshape(-1), ignore_index=-100, reduction="none",
                             label_smoothing=float(label_smoothing))
    return values.sum() / (labels != -100).sum().clamp_min(1)


def evidential_dirichlet_loss(logits: torch.Tensor, labels: torch.Tensor, kl_reg: float = 0.05) -> torch.Tensor:
    """Perte évidentielle de Dirichlet (risque de Bayes digamma + régularisation KL de l'évidence erronée)."""
    valid = labels != -100
    if not bool(valid.any()):
        return logits.sum() * 0.0
    z = logits[valid].float()
    targets = labels[valid]
    num_classes = z.size(-1)
    evidence = F.softplus(z)
    alpha = evidence + 1.0
    strength = alpha.sum(dim=-1, keepdim=True)
    one_hot = F.one_hot(targets, num_classes).float()
    bayes_risk = (one_hot * (torch.digamma(strength) - torch.digamma(alpha))).sum(dim=-1)
    # Régulariser uniquement l'évidence attribuée aux classes non cibles vers Dir(1, ..., 1)
    alpha_tilde = one_hot + (1.0 - one_hot) * alpha
    strength_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
    kl = (
        torch.lgamma(strength_tilde.squeeze(-1))
        - math.lgamma(num_classes)
        - torch.lgamma(alpha_tilde).sum(dim=-1)
        + ((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(strength_tilde))).sum(dim=-1)
    )
    return (bayes_risk + kl_reg * kl).mean()


class Blue(nn.Module):
    """Réseau conjoint : génération causale, classification, prévision probabiliste."""

    def __init__(self, config: BlueConfig):
        super().__init__()
        self.config = config
        self._alignment_group = None
        self._alignment_world = None
        d = config.d_model
        self.embedding = nn.Embedding(config.vocab_size, d)
        self.ngram = ByteNGramEmbedding(config.ngram_vocab_size, d, norm_eps=config.norm_eps) if config.ngram_vocab_size else None
        self.vision = VisionEncoder(config)
        self.temporal = TemporalEncoder(config)
        self.audio = AudioEncoder(config) if config.audio_enabled else None
        self.graph = GraphEncoder(config) if config.graph_enabled else None
        self.pointcloud = PointCloudEncoder(config) if config.pointcloud_enabled else None
        self.tabular = TabularEncoder(config) if config.tabular_enabled else None
        self.video = VideoEncoder(config) if config.video_enabled else None
        self.modality = nn.Parameter(torch.empty(2, d))
        self.audio_modality = nn.Parameter(torch.empty(d)) if config.audio_enabled else None
        self.graph_modality = nn.Parameter(torch.empty(d)) if config.graph_enabled else None
        self.pointcloud_modality = nn.Parameter(torch.empty(d)) if config.pointcloud_enabled else None
        self.tabular_modality = nn.Parameter(torch.empty(d)) if config.tabular_enabled else None
        self.video_modality = nn.Parameter(torch.empty(d)) if config.video_enabled else None
        self.resampler = LatentResampler(config)
        self.task_queries = nn.Parameter(torch.empty(1, 2, d))
        self.blocks = nn.ModuleList([DecoderBlock(config, i) for i in range(config.n_layers)])
        if config.latent_reasoning_steps > 1:
            self.reasoning_step_embed = nn.Embedding(config.latent_reasoning_steps, d)
            self.reasoning_norm = RMSNorm(d, config.norm_eps)
            self.reasoning_adapter = nn.Linear(2 * d, d, bias=False)
            self.reasoning_gate = nn.Parameter(torch.full((d,), -1.0))
            self.ponder_halt = nn.Linear(d, 1, bias=True) if config.latent_reasoning_adaptive else None
        else:
            self.reasoning_step_embed = None
            self.reasoning_norm = None
            self.reasoning_adapter = None
            self.reasoning_gate = None
            self.ponder_halt = None
        self.norm = RMSNorm(d, config.norm_eps)
        if config.embedding_output_skip:
            self.embed_skip_norm = RMSNorm(d, config.norm_eps)
            self.embed_skip_gate = nn.Parameter(torch.full((d,), -3.0))
        else:
            self.embed_skip_norm = self.embed_skip_gate = None
        self.episodic_memory = (EpisodicMemoryBank(d, config.memory_slots, config.n_heads,
                                                   norm_eps=config.norm_eps,
                                                   memory_decay=config.memory_decay)
                                if config.memory_slots > 0 else None)
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.future_heads = nn.ModuleList([
            nn.Sequential(RMSNorm(d), nn.Linear(d, d, bias=False), nn.SiLU())
            for _ in range(config.prediction_steps - 1)
        ])
        self.mtp_chain_gates = (
            nn.Parameter(torch.zeros(config.prediction_steps - 2, d))
            if (config.chained_future_heads and config.prediction_steps > 2) else None
        )
        self.fusion = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), RMSNorm(d))
        if config.head_cross_pool:
            self.head_pool_norm = RMSNorm(d, config.norm_eps)
            self.head_pool_attn = CrossAttention(config)
            self.head_pool_gate = nn.Parameter(torch.full((2, 1, d), 0.1))
        else:
            self.head_pool_norm = self.head_pool_attn = self.head_pool_gate = None
        self.classifier = nn.Linear(d, config.num_classes)
        self.forecaster = QuantileForecastHead(config)
        self.flow_head = ConditionalFlowHead(config) if config.flow_matching_dim > 0 else None
        self.jepa = (
            MultimodalJEPAPredictor(d, norm_eps=config.norm_eps, covariance_weight=config.jepa_cov_weight)
            if config.jepa_weight > 0 else None
        )
        self.task_log_vars = nn.Parameter(torch.zeros(7)) if config.adaptive_loss_weighting else None
        self.alignment = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(3)])
        self.audio_alignment = nn.Linear(d, d, bias=False) if config.audio_enabled else None
        self.graph_alignment = nn.Linear(d, d, bias=False) if config.graph_enabled else None
        self.pointcloud_alignment = nn.Linear(d, d, bias=False) if config.pointcloud_enabled else None
        self.tabular_alignment = nn.Linear(d, d, bias=False) if config.tabular_enabled else None
        self.video_alignment = nn.Linear(d, d, bias=False) if config.video_enabled else None
        self.forecast_adapter = ForecastAdapter(config) if config.forecast_adapter_context else None
        self.apply(self._initialize)
        if config.stable_rms_norm:
            for module in self.modules():
                if isinstance(module, RMSNorm):
                    module.stable = True
        if config.dynamic_tanh_norm:
            for module in self.modules():
                if isinstance(module, RMSNorm):
                    module.enable_dyt(0.5)
        if config.kan_degree >= 2:
            from .modules import SwiGLU
            for module in self.modules():
                if isinstance(module, SwiGLU):
                    module.enable_kan(config.kan_degree)
        if config.conv_diff_branch:
            from .modules import GatedCausalConv
            for module in self.modules():
                if isinstance(module, GatedCausalConv):
                    module.enable_diff_branch()
        nn.init.zeros_(self.temporal.forecast_skip.weight)
        if self.temporal.statistics_proj is not None:
            nn.init.zeros_(self.temporal.statistics_proj.weight)
        if self.temporal.spectral is not None:
            self.temporal.spectral.reset_spectral_parameters()
        if self.temporal.wavelet is not None:
            self.temporal.wavelet.reset_wavelet_parameters()
        if self.temporal.freq_wavelet_gate is not None:
            nn.init.zeros_(self.temporal.freq_wavelet_gate)
        self.forecaster.reset_tail_parameters()
        if self.forecast_adapter is not None:
            nn.init.zeros_(self.forecast_adapter.proj.weight)
            nn.init.zeros_(self.forecast_adapter.proj.bias)
            if self.forecast_adapter.missing_proj is not None:
                nn.init.zeros_(self.forecast_adapter.missing_proj.weight)
                nn.init.zeros_(self.forecast_adapter.missing_proj.bias)
        for parameter in (self.modality, self.task_queries, self.resampler.latents,
                          self.resampler.null_context, self.vision.position):
            nn.init.normal_(parameter, std=0.02)
        if self.audio_modality is not None:
            nn.init.normal_(self.audio_modality, std=0.02)
        if self.graph_modality is not None:
            nn.init.normal_(self.graph_modality, std=0.02)
        if self.pointcloud_modality is not None:
            nn.init.normal_(self.pointcloud_modality, std=0.02)
        if self.tabular_modality is not None:
            nn.init.normal_(self.tabular_modality, std=0.02)
        if self.tabular is not None:
            self.tabular.reset_tabular_parameters()
        if self.video_modality is not None:
            nn.init.normal_(self.video_modality, std=0.02)
        if self.video is not None:
            self.video.reset_video_parameters()
        if self.ngram is not None:
            self.ngram.reset_ngram_parameters()
        if self.resampler.local_latents is not None:
            nn.init.normal_(self.resampler.local_latents, std=0.02)
        if self.resampler.registers is not None:
            nn.init.normal_(self.resampler.registers, std=0.02)
        if self.audio is not None and self.audio.spectral is not None:
            self.audio.spectral.reset_spectral_parameters()
        if self.reasoning_gate is not None:
            nn.init.constant_(self.reasoning_gate, -1.0)
        if self.ponder_halt is not None:
            p = self.config.ponder_prior
            nn.init.constant_(self.ponder_halt.bias, math.log(p / (1.0 - p)))
        if self.flow_head is not None:
            self.flow_head.reset_adaln_parameters()
        if self.jepa is not None:
            self.jepa.reset_jepa_parameters()
        if self.task_log_vars is not None:
            nn.init.zeros_(self.task_log_vars)
        if self.head_pool_gate is not None:
            nn.init.constant_(self.head_pool_gate, 0.1)
        if self.mtp_chain_gates is not None:
            nn.init.zeros_(self.mtp_chain_gates)
        if self.embed_skip_gate is not None:
            nn.init.constant_(self.embed_skip_gate, -3.0)
        if self.episodic_memory is not None:
            self.episodic_memory.reset_memory_parameters()
        for module in self.modules():
            if isinstance(module, CrossAttention):
                module.reset_cross_parameters()
        for block in self.blocks:
            block.attn.reset_sink_parameters()
            block.reset_depth_router()
            if block.ssm is not None:
                block.ssm.reset_gate_biases()
            if block.ttt is not None:
                block.ttt.reset_ttt_parameters()
            for hc in (block.hc_attn, block.hc_cross, block.hc_moe):
                if hc is not None:
                    hc.reset_manifold_parameters()
            for expert in [*block.moe.experts, block.moe.shared]:
                nn.init.normal_(expert.down.weight, std=0.02 / math.sqrt(2 * config.n_layers))
            block.moe.reset_moe_parameters(config.n_layers)

    def __getstate__(self):
        # Les groupes de communication sont propres au processus et ne doivent
        # pas être copiés ou sérialisés avec le modèle.
        state = super().__getstate__()
        state["_alignment_group"] = None
        state["_alignment_world"] = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._alignment_group = None
        self._alignment_world = None

    @torch.no_grad()
    def update_router_biases(self, expert_usage: torch.Tensor):
        """Mettre à jour les biais d'équilibrage MoE sans perte auxiliaire."""
        if self.config.router_bias_update_rate <= 0:
            return
        if expert_usage.shape != (self.config.n_layers, self.config.n_experts):
            raise ValueError("expert_usage doit avoir la forme [n_layers, n_experts]")
        for index, block in enumerate(self.blocks):
            block.moe.update_expert_bias(expert_usage[index])

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _validate(self, input_ids, images, series, attention_mask, series_mask, cache,
                  reasoning_steps=None, audio=None, audio_mask=None,
                  graph_nodes=None, graph_edges=None, graph_mask=None,
                  points=None, point_features=None, point_mask=None,
                  tabular_num=None, tabular_cat=None,
                  video=None, video_mask=None):
        c = self.config
        if reasoning_steps is not None:
            if type(reasoning_steps) is not int or reasoning_steps < 1:
                raise ValueError("reasoning_steps doit être un entier strictement positif")
            if c.latent_reasoning_steps == 1 and reasoning_steps != 1:
                raise ValueError("reasoning_steps > 1 nécessite latent_reasoning_steps > 1 dans la configuration")
        effective_steps = c.latent_reasoning_steps if reasoning_steps is None else reasoning_steps
        tensors = [x for x in (input_ids, images, series, audio, graph_nodes, points, tabular_num, video) if x is not None]
        if not tensors:
            raise ValueError("Au moins une modalité est requise : texte, images, séries, audio, graphe, nuage 3D, tabulaire ou vidéo")
        batch = tensors[0].size(0)
        if batch < 1 or any(x.size(0) != batch for x in tensors):
            raise ValueError("Les modalités doivent partager un batch non vide")
        device = self.embedding.weight.device
        if any(x.device != device for x in tensors):
            raise ValueError("Les entrées et le modèle doivent être sur le même device")
        if input_ids is not None:
            if input_ids.ndim != 2 or input_ids.dtype != torch.long or input_ids.size(1) < 1:
                raise ValueError("input_ids doit être un tenseur long non vide [B, T]")
            if (input_ids < 0).any() or (input_ids >= c.vocab_size).any():
                raise ValueError("Identifiant de token hors vocabulaire")
            offset = 0 if cache is None else cache.offset
            if input_ids.size(1) + offset > c.max_seq_len:
                raise ValueError("La longueur texte dépasse max_seq_len")
        elif attention_mask is not None or cache is not None:
            raise ValueError("attention_mask et cache nécessitent input_ids")
        if images is not None:
            if images.shape[1:] != (c.image_channels, c.image_size, c.image_size):
                raise ValueError("images doit avoir la forme [B, canaux, image_size, image_size]")
            if not images.is_floating_point() or not torch.isfinite(images).all():
                raise ValueError("images doit contenir des nombres flottants finis")
        if series is not None:
            if series.ndim != 3 or series.size(-1) != c.series_features:
                raise ValueError("series doit avoir la forme [B, T, series_features]")
            if not series.is_floating_point() or not 1 <= series.size(1) <= c.max_series_len:
                raise ValueError("series doit être flottant et respecter max_series_len")
        elif series_mask is not None:
            raise ValueError("series_mask nécessite series")
        if audio is not None:
            if self.audio is None:
                raise ValueError("audio nécessite audio_enabled=True dans la configuration")
            if audio.ndim != 3 or audio.size(1) != c.audio_channels:
                raise ValueError("audio doit avoir la forme [B, audio_channels, L]")
            if not audio.is_floating_point() or not 1 <= audio.size(2) <= c.max_audio_len:
                raise ValueError("audio doit être flottant et respecter max_audio_len")
            if not torch.isfinite(audio).all():
                raise ValueError("audio doit contenir des nombres flottants finis")
        elif audio_mask is not None:
            raise ValueError("audio_mask nécessite audio")
        if graph_nodes is not None:
            if self.graph is None:
                raise ValueError("graph_nodes nécessite graph_enabled=True dans la configuration")
            if graph_nodes.ndim != 3 or graph_nodes.size(-1) != c.graph_node_features:
                raise ValueError("graph_nodes doit avoir la forme [B, N, graph_node_features]")
            if not graph_nodes.is_floating_point() or not 1 <= graph_nodes.size(1) <= c.max_graph_nodes:
                raise ValueError("graph_nodes doit être flottant et respecter max_graph_nodes")
            if not torch.isfinite(graph_nodes).all():
                raise ValueError("graph_nodes doit contenir des nombres flottants finis")
            if graph_edges is not None:
                expected_edges = (batch, graph_nodes.size(1), graph_nodes.size(1), c.graph_edge_features)
                if graph_edges.shape != expected_edges or graph_edges.device != device:
                    raise ValueError("graph_edges doit avoir la forme [B, N, N, graph_edge_features] sur le même device")
                if not graph_edges.is_floating_point() or not torch.isfinite(graph_edges).all():
                    raise ValueError("graph_edges doit contenir des nombres flottants finis")
        elif graph_edges is not None or graph_mask is not None:
            raise ValueError("graph_edges et graph_mask nécessitent graph_nodes")
        if points is not None:
            if self.pointcloud is None:
                raise ValueError("points nécessite pointcloud_enabled=True dans la configuration")
            if points.ndim != 3 or points.size(-1) != 3:
                raise ValueError("points doit avoir la forme [B, P, 3]")
            if not points.is_floating_point() or not 1 <= points.size(1) <= c.max_pointcloud_points:
                raise ValueError("points doit être flottant et respecter max_pointcloud_points")
            if not torch.isfinite(points).all():
                raise ValueError("points doit contenir des nombres flottants finis")
            if point_features is not None:
                if c.pointcloud_features == 0:
                    raise ValueError("point_features nécessite pointcloud_features > 0")
                if (point_features.shape != (batch, points.size(1), c.pointcloud_features)
                        or point_features.device != device or not point_features.is_floating_point()
                        or not torch.isfinite(point_features).all()):
                    raise ValueError("point_features doit être flottant fini [B, P, pointcloud_features] sur le même device")
        elif point_features is not None or point_mask is not None:
            raise ValueError("point_features et point_mask nécessitent points")
        if tabular_num is not None:
            if self.tabular is None:
                raise ValueError("tabular_num nécessite tabular_enabled=True dans la configuration")
            if tabular_num.shape != (batch, c.tabular_num_features) or not tabular_num.is_floating_point():
                raise ValueError("tabular_num doit être flottant [B, tabular_num_features]")
            if torch.isinf(tabular_num).any():
                raise ValueError("tabular_num ne peut pas contenir de valeurs infinies (utiliser NaN pour manquant)")
            if tabular_cat is not None:
                if c.tabular_cat_features == 0:
                    raise ValueError("tabular_cat nécessite tabular_cat_features > 0")
                if (tabular_cat.shape != (batch, c.tabular_cat_features)
                        or tabular_cat.device != device or tabular_cat.dtype != torch.long):
                    raise ValueError("tabular_cat doit être long [B, tabular_cat_features] sur le même device")
                if (tabular_cat < -1).any() or (tabular_cat >= c.tabular_cat_cardinality).any():
                    raise ValueError("tabular_cat doit appartenir à [-1, tabular_cat_cardinality - 1]")
        elif tabular_cat is not None:
            raise ValueError("tabular_cat nécessite tabular_num")
        if video is not None:
            if self.video is None:
                raise ValueError("video nécessite video_enabled=True dans la configuration")
            if video.ndim != 5 or video.size(1) != c.video_channels or video.shape[3:] != (c.video_size, c.video_size):
                raise ValueError("video doit avoir la forme [B, video_channels, F, video_size, video_size]")
            if not video.is_floating_point() or not 1 <= video.size(2) <= c.max_video_frames:
                raise ValueError("video doit être flottant et respecter max_video_frames")
            if not torch.isfinite(video).all():
                raise ValueError("video doit contenir des nombres flottants finis")
        elif video_mask is not None:
            raise ValueError("video_mask nécessite video")
        for mask, name, expected_shape in (
            (attention_mask, "attention_mask", None if input_ids is None else input_ids.shape[:2]),
            (series_mask, "series_mask", None if series is None else series.shape[:2]),
            (audio_mask, "audio_mask", None if audio is None else (audio.size(0), audio.size(2))),
            (graph_mask, "graph_mask", None if graph_nodes is None else graph_nodes.shape[:2]),
            (point_mask, "point_mask", None if points is None else points.shape[:2]),
            (video_mask, "video_mask", None if video is None else (video.size(0), video.size(2))),
        ):
            if mask is not None:
                if mask.shape != expected_shape or mask.device != device:
                    raise ValueError(f"{name} doit partager la forme [B, T] et le device des données")
                if not ((mask == 0) | (mask == 1)).all():
                    raise ValueError(f"{name} doit être binaire")
                if not mask.bool().any(-1).all():
                    raise ValueError(f"{name} doit garder au moins une position par exemple")
        if cache is not None:
            if self.training:
                raise ValueError("Le cache est réservé à l'inférence en mode eval()")
            if any(m is not None for m in (images, series, audio, graph_nodes, points, tabular_num, video)):
                raise ValueError("Le contexte multimodal est déjà dans le cache ; ne pas le réinjecter")
            if len(cache.layers) != c.n_layers * effective_steps or cache.latents.size(0) != batch:
                raise ValueError("Cache incompatible avec ce modèle ou ce batch")
            ssm_heads = c.n_heads if c.ssm_heads is None else c.ssm_heads
            ssm_shape = (batch, ssm_heads, c.ssm_state_dim, c.d_model // ssm_heads) if c.ssm_state_dim else None
            ttt_heads = c.n_heads if c.ttt_heads is None else c.ttt_heads
            ttt_shape = (batch, ttt_heads, c.ttt_state_dim + 1, c.ttt_state_dim) if c.ttt_state_dim else None
            if (c.memory_slots > 0 and (cache.episodic_memory is None
                                        or cache.episodic_memory.shape != (batch, c.memory_slots, c.d_model))):
                raise ValueError("Mémoire épisodique incompatible dans le cache")
            if (c.ngram_vocab_size > 0 and (cache.ngram_tail is None or cache.ngram_tail.shape != (batch, 3)
                                            or cache.ngram_tail_valid is None or cache.ngram_tail_valid.shape != (batch, 3))):
                raise ValueError("État N-grammes incompatible dans le cache")
            if cache.offset < 1 or any(
                layer.start_pos < 0 or layer.start_pos + layer.key.size(-2) != cache.offset
                or (layer.key.shape != layer.value.shape if not c.kv_lora_rank
                    else layer.key.shape[:3] != layer.value.shape[:3])
                or layer.key.size(0) != batch
                or layer.valid.shape != (batch, layer.key.size(-2))
                or ((ssm_shape is not None and (slot % c.n_layers) % c.ssm_every == 0)
                    and (layer.ssm_state is None or layer.ssm_state.shape != ssm_shape))
                or ((ssm_shape is None or (slot % c.n_layers) % c.ssm_every != 0)
                    and layer.ssm_state is not None)
                or ((ttt_shape is not None and (slot % c.n_layers) % c.ttt_every == 0)
                    and (layer.ttt_state is None or layer.ttt_state.shape != ttt_shape))
                or ((ttt_shape is None or (slot % c.n_layers) % c.ttt_every != 0)
                    and layer.ttt_state is not None)
                for slot, layer in enumerate(cache.layers)
            ):
                raise ValueError("Positions incohérentes dans le cache")
        return batch

    def _project_alignment(self, index: int, value: torch.Tensor) -> torch.Tensor:
        if index == 3:
            proj = self.audio_alignment
        elif index == 4:
            proj = self.graph_alignment
        elif index == 5:
            proj = self.pointcloud_alignment
        elif index == 6:
            proj = self.tabular_alignment
        elif index == 7:
            proj = self.video_alignment
        else:
            proj = self.alignment[index]
        projected = proj(value).float()
        if self.config.alignment_geometry == "hyperbolic":
            return poincare_exp0(projected, self.config.alignment_curvature)
        return F.normalize(projected, dim=-1)

    def _default_graph_edges(self, graph_nodes: torch.Tensor) -> torch.Tensor:
        b, n, _ = graph_nodes.shape
        return graph_nodes.new_zeros((b, n, n, self.config.graph_edge_features))

    def encode_modalities(self, *, input_ids=None, images=None, series=None, audio=None,
                          graph_nodes=None, graph_edges=None, points=None, point_features=None,
                          tabular_num=None, tabular_cat=None, video=None,
                          attention_mask=None, series_mask=None, audio_mask=None,
                          graph_mask=None, point_mask=None, video_mask=None):
        """Représentations contrastives indépendantes de chaque modalité fournie.

        Le texte utilise toujours le contexte nul, même si l'alignement a été
        entraîné en mode joint. Le mode train/eval et les gradients restent ceux
        choisis par l'appelant.
        """
        self._validate(input_ids, images, series, attention_mask, series_mask, None,
                       audio=audio, audio_mask=audio_mask,
                       graph_nodes=graph_nodes, graph_edges=graph_edges, graph_mask=graph_mask,
                       points=points, point_features=point_features, point_mask=point_mask,
                       tabular_num=tabular_num, tabular_cat=tabular_cat,
                       video=video, video_mask=video_mask)
        representations = {}
        if input_ids is not None:
            valid = (torch.ones_like(input_ids, dtype=torch.bool)
                     if attention_mask is None else attention_mask.bool())
            representations['text'] = (2, self._isolated_text_representation(input_ids, valid))
        encoded, _ = self._encode_inputs(
            images=images, series=series, audio=audio, graph_nodes=graph_nodes, graph_edges=graph_edges,
            points=points, point_features=point_features, tabular_num=tabular_num, tabular_cat=tabular_cat,
            video=video, series_mask=series_mask, audio_mask=audio_mask, graph_mask=graph_mask,
            point_mask=point_mask, video_mask=video_mask)
        for item in encoded:
            representations[item.name] = (item.alignment_index, item.pooled)
        result = {}
        for name, (index, value) in representations.items():
            vector = self._project_alignment(index, value)
            if not torch.isfinite(vector).all():
                raise FloatingPointError("Les représentations contrastives doivent être finies")
            result[name] = vector
        return result

    def _encode_inputs(self, *, images=None, series=None, audio=None, graph_nodes=None, graph_edges=None,
                       points=None, point_features=None, tabular_num=None, tabular_cat=None, video=None,
                       series_mask=None, audio_mask=None, graph_mask=None, point_mask=None, video_mask=None):
        """Encoder les modalités non textuelles fournies, dans l'ordre du contexte multimodal.

        Renvoie les ``EncodedInput`` et, pour les séries, ``(valid, mean, scale, baseline)``
        de l'encodeur temporel (sinon ``None``).
        """
        def all_valid(data, length_dim=1):
            return torch.ones((data.size(0), data.size(length_dim)), device=data.device, dtype=torch.bool)

        encoded, temporal = [], None
        if images is not None:
            tokens, valid = self.vision(images)
            encoded.append(EncodedInput("image", 0, tokens, valid, tokens.mean(1)))
        if series is not None:
            mask = all_valid(series) if series_mask is None else series_mask.bool()
            tokens, valid, mean, scale, baseline = self.temporal(series, mask)
            temporal = valid, mean, scale, baseline
            encoded.append(EncodedInput("series", 1, tokens, valid, _masked_mean(tokens, valid)))
        if audio is not None:
            mask = all_valid(audio, 2) if audio_mask is None else audio_mask.bool()
            tokens, valid = self.audio(audio, mask)
            encoded.append(EncodedInput("audio", 3, tokens, valid, _masked_mean(tokens, valid)))
        if graph_nodes is not None:
            mask = all_valid(graph_nodes) if graph_mask is None else graph_mask.bool()
            edges = self._default_graph_edges(graph_nodes) if graph_edges is None else graph_edges
            tokens, valid = self.graph(graph_nodes, edges, mask)
            encoded.append(EncodedInput("graph", 4, tokens, valid, _masked_mean(tokens, valid)))
        if points is not None:
            mask = all_valid(points) if point_mask is None else point_mask.bool()
            tokens, valid = self.pointcloud(points, mask, point_features)
            encoded.append(EncodedInput("pointcloud", 5, tokens, valid, _masked_mean(tokens, valid)))
        if tabular_num is not None:
            tokens, valid = self.tabular(tabular_num, tabular_cat)
            encoded.append(EncodedInput("tabular", 6, tokens, valid, _masked_mean(tokens, valid)))
        if video is not None:
            mask = all_valid(video, 2) if video_mask is None else video_mask.bool()
            tokens, valid = self.video(video, mask)
            encoded.append(EncodedInput("video", 7, tokens, valid, _masked_mean(tokens, valid)))
        return encoded, temporal

    def _modality_embedding(self, name: str) -> torch.Tensor:
        return {"image": self.modality[0], "series": self.modality[1], "audio": self.audio_modality,
                "graph": self.graph_modality, "pointcloud": self.pointcloud_modality,
                "tabular": self.tabular_modality, "video": self.video_modality}[name]

    def _expand_streams(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.residual_streams == 1:
            return x
        return x.unsqueeze(-2).expand(-1, -1, self.config.residual_streams, -1).contiguous()

    def _collapse_streams(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.residual_streams == 1:
            return x
        return x.mean(dim=-2)

    def _prepare_reasoning_step(self, state: torch.Tensor, initial: torch.Tensor,
                                valid: torch.Tensor, step_index: int) -> torch.Tensor:
        if self.reasoning_step_embed is None:
            return state
        embed_index = min(step_index, self.config.latent_reasoning_steps - 1)
        step_vec = self.reasoning_step_embed.weight[embed_index]
        if state.ndim == 4:
            mask = valid[:, :, None, None]
            step_vec = step_vec.view(1, 1, 1, -1)
        else:
            mask = valid.unsqueeze(-1)
            step_vec = step_vec.view(1, 1, -1)
        if step_index == 0:
            return (state + step_vec) * mask
        fused = self.reasoning_adapter(torch.cat((self.reasoning_norm(state), initial), dim=-1))
        return (state + torch.sigmoid(self.reasoning_gate) * fused + step_vec) * mask

    def _run_decoder_stack(self, x_raw, valid, latents, *, offset=0, cache=None, use_cache=False,
                           reasoning_steps=None):
        steps = self.config.latent_reasoning_steps if reasoning_steps is None else reasoning_steps
        initial = self._expand_streams(x_raw)
        x = initial
        balance_losses, z_losses, depth_losses, entropy_losses, next_layers = [], [], [], [], []
        layer_usage = [None] * self.config.n_layers
        adaptive = self.ponder_halt is not None and steps > 1
        step_states = [] if adaptive else None
        halt_hazards = [] if adaptive else None
        # Une seule synchronisation par passe : toutes les positions (et le cache) sont-elles valides ?
        dense = bool(valid.all()) and (cache is None or cache.dense is True)
        for rec_step in range(steps):
            x = self._prepare_reasoning_step(x, initial, valid, rec_step)
            v0 = None
            q0 = None
            for index, block in enumerate(self.blocks):
                cache_slot = rec_step * self.config.n_layers + index
                use_ckpt = (
                    self.config.gradient_checkpointing
                    and self.training
                    and (index % self.config.checkpoint_every == 0)
                )
                if use_ckpt:
                    # Les sorties absentes (None) traversent checkpoint sans tenseur factice.
                    result = checkpoint(block, x, valid, latents, v0=v0, q0=q0, dense=dense, use_reentrant=False)
                else:
                    result = block(
                        x, valid, latents, offset=offset,
                        cache=None if cache is None else cache.layers[cache_slot],
                        use_cache=use_cache, v0=v0, q0=q0, dense=dense,
                    )
                    if use_cache:
                        next_layers.append(result.cache)
                x = result.hidden
                if index == 0:
                    v0, q0 = result.v_first, result.q_first
                balance_losses.append(result.balance)
                z_losses.append(result.z_loss)
                entropy_losses.append(result.router_entropy)
                if result.depth_loss is not None:
                    depth_losses.append(result.depth_loss)
                layer_usage[index] = (result.usage if layer_usage[index] is None
                                      else layer_usage[index] + result.usage)
            if adaptive:
                collapsed_step = self._collapse_streams(x)
                step_states.append(collapsed_step)
                with torch.autocast(device_type=collapsed_step.device.type, enabled=False):
                    normed = self.reasoning_norm(collapsed_step).float()
                    hazard = torch.sigmoid(F.linear(
                        normed, self.ponder_halt.weight.float(), self.ponder_halt.bias.float()
                    )).squeeze(-1)
                halt_hazards.append(hazard)
        usage = [value / steps for value in layer_usage]
        ponder_kl = None
        if adaptive:
            with torch.autocast(device_type=x.device.type, enabled=False):
                rem = torch.ones_like(halt_hazards[0])
                probs = []
                for idx in range(steps):
                    if idx < steps - 1:
                        p_step = rem * halt_hazards[idx]
                        rem = rem * (1.0 - halt_hazards[idx])
                    else:
                        p_step = rem
                    probs.append(p_step)
                halt_probs = torch.stack(probs, dim=-1)  # [B, T, R]
                prior_raw = torch.tensor(
                    [((1.0 - self.config.ponder_prior) ** idx) * self.config.ponder_prior for idx in range(steps)],
                    device=x.device, dtype=torch.float32,
                )
                prior = prior_raw / prior_raw.sum()
                kl_per_token = (halt_probs * (halt_probs.clamp_min(1e-8).log() - prior.clamp_min(1e-8).log())).sum(-1)
                ponder_kl = (kl_per_token * valid.float()).sum() / valid.sum().clamp_min(1)
            stacked_states = torch.stack(step_states, dim=-2)  # [B, T, R, D]
            collapsed = (stacked_states * halt_probs.to(stacked_states.dtype).unsqueeze(-1)).sum(dim=-2)
        else:
            collapsed = self._collapse_streams(x)
        next_memory = None
        if self.episodic_memory is not None:
            prev_mem = None if cache is None else cache.episodic_memory
            collapsed, next_memory = self.episodic_memory(collapsed, valid, prev_mem)
        out_hidden = self.norm(collapsed)
        if self.embed_skip_gate is not None:
            skip = torch.sigmoid(self.embed_skip_gate).to(out_hidden.dtype) * self.embed_skip_norm(x_raw)
            out_hidden = (out_hidden + skip) * valid.unsqueeze(-1)
        return out_hidden, balance_losses, z_losses, depth_losses, entropy_losses, ponder_kl, usage, next_layers, next_memory, dense

    def _embed_text(self, input_ids: torch.Tensor, valid: torch.Tensor, cache: ModelCache | None = None):
        x = self.embedding(input_ids)
        if self.ngram is None:
            return x, None, None
        prev_b = None if cache is None else cache.ngram_tail
        prev_v = None if cache is None else cache.ngram_tail_valid
        ng, next_b, next_v = self.ngram(input_ids, valid, prev_b, prev_v)
        return x + ng.to(x.dtype), next_b, next_v

    def _isolated_text_representation(self, input_ids, valid):
        """Encoder le texte avec le contexte nul, sans image ni série associée."""
        batch = input_ids.size(0)
        memory = self.resampler([], [], batch)
        x, _, _ = self._embed_text(input_ids, valid)
        hidden, *_ = self._run_decoder_stack(x, valid, memory)
        last = torch.arange(valid.size(1), device=hidden.device)[None, :].expand_as(valid)
        last = last.masked_fill(~valid, -1).max(-1).values
        return hidden[torch.arange(batch, device=hidden.device), last]

    def forward(
        self, input_ids=None, *, images=None, series=None, audio=None,
        graph_nodes=None, graph_edges=None, points=None, point_features=None,
        tabular_num=None, tabular_cat=None, video=None,
        attention_mask=None, series_mask=None, audio_mask=None, graph_mask=None,
        point_mask=None, video_mask=None,
        labels=None, class_labels=None, forecast_targets=None, flow_targets=None, alignment_ids=None,
        alignment_cross_rank_views=False, alignment_cross_class_views=False,
        cache: ModelCache | None = None, use_cache=False, reasoning_steps: int | None = None,
    ) -> BlueOutput:
        batch = self._validate(input_ids, images, series, attention_mask, series_mask, cache,
                               reasoning_steps, audio=audio, audio_mask=audio_mask,
                               graph_nodes=graph_nodes, graph_edges=graph_edges, graph_mask=graph_mask,
                               points=points, point_features=point_features, point_mask=point_mask,
                               tabular_num=tabular_num, tabular_cat=tabular_cat,
                               video=video, video_mask=video_mask)
        distributed_alignment = (self.config.alignment_scope == "global" and self.training and dist.is_available()
                                 and dist.is_initialized() and dist.get_world_size() > 1)
        if type(alignment_cross_rank_views) is not bool and not distributed_alignment:
            raise ValueError("alignment_cross_rank_views doit être booléen")
        if type(alignment_cross_class_views) is not bool and not distributed_alignment:
            raise ValueError("alignment_cross_class_views doit être booléen")
        if not distributed_alignment and alignment_cross_class_views and self.config.alignment_mode != "class":
            raise ValueError("alignment_cross_class_views nécessite le mode class")
        if use_cache and (self.training or input_ids is None):
            raise ValueError("use_cache nécessite eval() et des tokens texte")
        if cache is not None and not use_cache:
            raise ValueError("Passer use_cache=True pour continuer un cache")
        supervised = any(t is not None for t in (labels, class_labels, forecast_targets, flow_targets))
        if use_cache and supervised:
            raise ValueError("Le cache ne doit pas être utilisé pendant le calcul des pertes")
        if flow_targets is not None:
            if self.flow_head is None:
                raise ValueError("flow_targets nécessite flow_matching_dim > 0 dans la configuration")
            if flow_targets.shape != (batch, self.config.flow_matching_dim) or not flow_targets.is_floating_point():
                raise ValueError("flow_targets doit être flottant [B, flow_matching_dim]")
            if flow_targets.device != self.embedding.weight.device:
                raise ValueError("Les cibles doivent être sur le même device que le modèle")
        if labels is not None and (input_ids is None or labels.shape != input_ids.shape):
            raise ValueError("labels doit partager la forme [B, T] de input_ids")
        if class_labels is not None and (class_labels.shape != (batch,) or class_labels.dtype != torch.long):
            raise ValueError("class_labels doit être long [B]")
        if alignment_ids is not None and (alignment_ids.shape != (batch,) or alignment_ids.dtype != torch.long
                                          or alignment_ids.device != self.embedding.weight.device
                                          or (alignment_ids < -1).any()):
            raise ValueError("alignment_ids doit être long [B], avec -1 pour les identités inconnues")
        if forecast_targets is not None and forecast_targets.shape != (
            batch, self.config.forecast_horizon, self.config.series_features
        ):
            raise ValueError("forecast_targets doit avoir la forme [B, horizon, variables]")
        for target in (labels, class_labels, forecast_targets):
            if target is not None and target.device != self.embedding.weight.device:
                raise ValueError("Les cibles doivent être sur le même device que le modèle")
        if labels is not None and labels.dtype != torch.long:
            raise ValueError("labels doit contenir des entiers long ou -100")

        contexts, masks, align_vectors = [], [], []
        mean = scale = baseline = None
        adapter_context = None
        if cache is None:
            encoded, temporal = self._encode_inputs(
                images=images, series=series, audio=audio, graph_nodes=graph_nodes, graph_edges=graph_edges,
                points=points, point_features=point_features, tabular_num=tabular_num, tabular_cat=tabular_cat,
                video=video, series_mask=series_mask, audio_mask=audio_mask, graph_mask=graph_mask,
                point_mask=point_mask, video_mask=video_mask)
            for item in encoded:
                contexts.append(item.tokens + self._modality_embedding(item.name))
                masks.append(item.valid)
                align_vectors.append((item.alignment_index, item.pooled))
            if temporal is not None:
                temporal_valid, mean, scale, baseline = temporal
                if self.forecast_adapter is not None:
                    adapter_context = self.forecast_adapter.prepare(series, temporal_valid, mean, scale)
            latents = self.resampler(contexts, masks, batch)
        else:
            latents, mean, scale = cache.latents, cache.series_mean, cache.series_scale
            baseline = cache.forecast_baseline
            adapter_context = cache.forecast_adapter_context

        next_ngram_tail = next_ngram_valid = None
        if input_ids is None:
            x = self.task_queries.expand(batch, -1, -1)
            valid = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        else:
            valid = (torch.ones_like(input_ids, dtype=torch.bool)
                     if attention_mask is None else attention_mask.bool())
            x, next_ngram_tail, next_ngram_valid = self._embed_text(input_ids, valid, cache=cache)
        offset = 0 if cache is None else cache.offset
        hidden, balance_losses, z_losses, depth_losses, entropy_losses, ponder_kl, usage, next_layers, next_memory, dense = self._run_decoder_stack(
            x, valid, latents, offset=offset, cache=cache, use_cache=use_cache,
            reasoning_steps=reasoning_steps,
        )
        logits = self._lm_logits(hidden) if input_ids is not None else None
        if input_ids is None:
            class_pooled, forecast_pooled = hidden.unbind(1)
        else:
            last = torch.arange(valid.size(1), device=x.device)[None, :].expand_as(valid)
            last = last.masked_fill(~valid, -1).max(-1).values
            pooled = hidden[torch.arange(batch, device=x.device), last]
            class_pooled = forecast_pooled = pooled
            align_vectors.append((2, pooled))
        memory_pooled = latents.mean(1)
        if self.head_pool_attn is not None:
            task_q = torch.stack((class_pooled, forecast_pooled), dim=1)
            attended = self.head_pool_attn(self.head_pool_norm(task_q), self.head_pool_norm(latents))
            g_cls, g_fc = torch.tanh(self.head_pool_gate).unbind(0)
            cls_memory = memory_pooled + g_cls.to(memory_pooled.dtype) * attended[:, 0]
            fc_memory = memory_pooled + g_fc.to(memory_pooled.dtype) * attended[:, 1]
        else:
            cls_memory = fc_memory = memory_pooled
        class_pooled = self.fusion(torch.cat((class_pooled, cls_memory), -1))
        forecast_pooled = self.fusion(torch.cat((forecast_pooled, fc_memory), -1))
        class_logits = self.classifier(class_pooled)
        forecast = self.forecaster(forecast_pooled, mean, scale, baseline)
        if self.forecast_adapter is not None and adapter_context is not None:
            forecast = self.forecast_adapter(forecast, adapter_context)

        auxiliary_text_losses = {}
        losses = {
            "balance": torch.stack(balance_losses).mean(),
            "router_z": torch.stack(z_losses).mean(),
        }
        c = self.config
        total = c.balance_weight * losses["balance"] + c.router_z_weight * losses["router_z"]
        if c.moe_entropy_weight > 0.0 and entropy_losses:
            losses["router_entropy"] = torch.stack(entropy_losses).mean()
            total = total + c.moe_entropy_weight * losses["router_entropy"]
        if c.moe_router_orthogonality_weight > 0.0 and c.n_experts > 1:
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                eye_e = torch.eye(c.n_experts, device=hidden.device, dtype=torch.bool)
                ortho_terms = []
                for block in self.blocks:
                    w_exp = F.normalize(block.moe.router.weight[:c.n_experts].float(), p=2, dim=-1, eps=1e-6)
                    gram_e = torch.matmul(w_exp, w_exp.t())
                    ortho_terms.append(
                        gram_e.masked_fill(eye_e, 0.0).square().sum() / (c.n_experts * (c.n_experts - 1))
                    )
                losses["router_orthogonality"] = torch.stack(ortho_terms).mean()
            total = total + c.moe_router_orthogonality_weight * losses["router_orthogonality"]
        if depth_losses:
            losses["depth_budget"] = torch.stack(depth_losses).mean()
            total = total + c.depth_budget_weight * losses["depth_budget"]
        if ponder_kl is not None:
            losses["ponder_kl"] = ponder_kl
            total = total + c.ponder_kl_weight * losses["ponder_kl"]
        if c.latent_diversity_weight > 0.0 and latents.size(1) > 1:
            with torch.autocast(device_type=latents.device.type, enabled=False):
                z_norm = F.normalize(latents.float(), p=2, dim=-1, eps=1e-6)
                gram = torch.matmul(z_norm, z_norm.transpose(-1, -2))
                n_lat = latents.size(1)
                eye = torch.eye(n_lat, device=latents.device, dtype=torch.bool).unsqueeze(0)
                off_diag_sq = gram.masked_fill(eye, 0.0).square().sum(dim=(-1, -2)) / (n_lat * (n_lat - 1))
                losses["latent_diversity"] = off_diag_sq.mean()
            total = total + c.latent_diversity_weight * losses["latent_diversity"]
        if labels is not None:
            targets = labels.masked_fill(~valid, -100)
            shifted = targets[:, 1:].masked_fill(~valid[:, :-1], -100)
            losses["text"] = masked_cross_entropy(logits[:, :-1], shifted, label_smoothing=c.label_smoothing)
            total = total + self._weight_task_loss("text", c.text_weight, losses["text"])
            if c.lm_z_loss_weight > 0.0:
                with torch.autocast(device_type=logits.device.type, enabled=False):
                    z_mask = shifted != -100
                    log_z = logits[:, :-1].float().logsumexp(dim=-1)
                    losses["lm_z"] = (log_z.square() * z_mask.float()).sum() / z_mask.sum().clamp_min(1)
                total = total + c.lm_z_loss_weight * losses["lm_z"]
            mtp_losses, mtp_active = [], []
            mtp_kl_losses = []
            h_mtp_prev = None
            for distance, head in enumerate(self.future_heads, start=2):
                head_idx = distance - 2
                if self.mtp_chain_gates is not None and head_idx > 0 and h_mtp_prev is not None:
                    chain_gate = torch.tanh(self.mtp_chain_gates[head_idx - 1]).to(hidden.dtype)
                    h_mtp_in = hidden + chain_gate * h_mtp_prev
                else:
                    h_mtp_in = hidden
                h_mtp_prev = head(h_mtp_in)
                if hidden.size(1) > distance:
                    future_logits = self._lm_logits(h_mtp_prev[:, :-distance])
                    future_targets = targets[:, distance:].masked_fill(~valid[:, :-distance], -100)
                    head_loss = masked_cross_entropy(future_logits, future_targets, label_smoothing=c.label_smoothing)
                    head_count = (future_targets != -100).sum()
                    auxiliary_text_losses[distance] = (head_loss, head_count)
                    mtp_losses.append(head_loss)
                    mtp_active.append(head_count > 0)
                    if c.mtp_kl_weight > 0.0:
                        with torch.autocast(device_type=logits.device.type, enabled=False):
                            teacher_lp = F.log_softmax(logits[:, distance - 1:-1].detach().float(), dim=-1)
                            student_lp = F.log_softmax(future_logits.float(), dim=-1)
                            kl_tok = F.kl_div(student_lp, teacher_lp, log_target=True, reduction="none").sum(-1)
                            kl_mask = (future_targets != -100) & valid[:, distance - 1:-1]
                            kl_mean = (kl_tok * kl_mask.float()).sum() / kl_mask.sum().clamp_min(1)
                            mtp_kl_losses.append(kl_mean)
            if mtp_losses:
                # Les horizons sans cible ne doivent pas diluer les horizons actifs.
                active = torch.stack(mtp_active).to(dtype=mtp_losses[0].dtype)
                if c.mtp_decay != 1.0:
                    decay_w = torch.tensor(
                        [c.mtp_decay ** i for i in range(len(mtp_losses))],
                        device=active.device, dtype=active.dtype,
                    )
                    weights = active * decay_w
                else:
                    weights = active
                denom = weights.sum().clamp_min(1e-8)
                losses["multi_token"] = (torch.stack(mtp_losses) * weights).sum() / denom
                total = total + self._weight_task_loss("multi_token", c.mtp_weight, losses["multi_token"])
                if mtp_kl_losses:
                    losses["mtp_kl"] = (torch.stack(mtp_kl_losses) * weights).sum() / denom
                    total = total + c.mtp_kl_weight * losses["mtp_kl"]
        if class_labels is not None:
            if c.evidential_classification:
                losses["classification"] = evidential_dirichlet_loss(class_logits, class_labels)
            else:
                losses["classification"] = masked_cross_entropy(class_logits, class_labels, label_smoothing=c.label_smoothing)
            total = total + self._weight_task_loss("classification", c.classification_weight, losses["classification"])
        if forecast_targets is not None:
            losses["forecast"] = quantile_pinball_loss(forecast, forecast_targets, c.quantiles)
            total = total + self._weight_task_loss("forecast", c.forecast_weight, losses["forecast"])
            if c.forecast_smoothness_weight > 0.0 and forecast.size(1) >= 3:
                with torch.autocast(device_type=forecast.device.type, enabled=False):
                    f32 = forecast.float()
                    if scale is not None:
                        f32 = f32 / scale.float().clamp_min(1e-5)[:, None, :, None]
                    d2 = f32[:, 2:] - 2.0 * f32[:, 1:-1] + f32[:, :-2]
                    losses["forecast_smoothness"] = d2.square().mean()
                total = total + c.forecast_smoothness_weight * losses["forecast_smoothness"]
        if flow_targets is not None:
            losses["flow"] = self.flow_head.loss(flow_targets, forecast_pooled)
            total = total + self._weight_task_loss("flow", c.flow_matching_weight, losses["flow"])
        if self.jepa is not None and len(align_vectors) >= 2:
            losses["jepa"] = self.jepa(align_vectors)
            total = total + self._weight_task_loss("jepa", c.jepa_weight, losses["jepa"])
        # En entraînement distribué, toutes les paires effectivement présentes
        # peuvent partager leurs exemples négatifs entre rangs.
        if (c.alignment_text_view == "isolated" and input_ids is not None
                and (distributed_alignment or len(align_vectors) > 1 and batch > 1)):
            isolated_text = self._isolated_text_representation(input_ids, valid)
            align_vectors = [(index, isolated_text if index == 2 else vector) for index, vector in align_vectors]
        if distributed_alignment:
            if self._alignment_group is None or self._alignment_world is not dist.group.WORLD:
                self._alignment_group = dist.new_group()
                self._alignment_world = dist.group.WORLD
            vectors = {i: self._project_alignment(i, v) for i, v in align_vectors}
            n_mod = (8 if c.video_enabled else
                     (7 if c.tabular_enabled else
                      (6 if c.pointcloud_enabled else (5 if c.graph_enabled else (4 if c.audio_enabled else 3)))))
            losses["alignment"] = distributed_alignment_loss(vectors, class_labels, c.alignment_temperature,
                                                              c.alignment_mode, group=self._alignment_group,
                                                              text_view=c.alignment_text_view, sample_ids=alignment_ids,
                                                              cross_rank_views=alignment_cross_rank_views,
                                                              cross_class_views=alignment_cross_class_views,
                                                              num_modalities=n_mod,
                                                              geometry=c.alignment_geometry,
                                                              curvature=c.alignment_curvature,
                                                              margin=c.alignment_margin)
            total = total + self._weight_task_loss("alignment", c.alignment_weight, losses["alignment"])
        elif len(align_vectors) > 1 and batch > 1:
            vectors = [self._project_alignment(i, v) for i, v in align_vectors]
            contrastive = []
            margin_scaled = c.alignment_margin / c.alignment_temperature
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    with torch.autocast(device_type=x.device.type, enabled=False):
                        if c.alignment_geometry == "hyperbolic":
                            similarity = -poincare_pairwise_distance(vectors[i], vectors[j],
                                                                     c.alignment_curvature) / c.alignment_temperature
                        else:
                            similarity = vectors[i] @ vectors[j].T / c.alignment_temperature
                    contrastive.append(symmetric_contrastive_loss(
                        similarity, class_labels if c.alignment_mode == "class" else None,
                        sample_ids=alignment_ids, margin_scaled=margin_scaled,
                    ))
            losses["alignment"] = torch.stack(contrastive).mean()
            total = total + self._weight_task_loss("alignment", c.alignment_weight, losses["alignment"])
        next_cache = (ModelCache(next_layers, latents, offset + hidden.size(1), mean, scale, baseline,
                                 adapter_context, episodic_memory=next_memory,
                                 ngram_tail=next_ngram_tail, ngram_tail_valid=next_ngram_valid, dense=dense)
                      if use_cache else None)
        return BlueOutput(logits, class_logits, forecast, total if supervised else None,
                          losses, torch.stack(usage), next_cache, auxiliary_text_losses, valid.sum(),
                          last_hidden=hidden[:, -1:] if use_cache else None)

    @staticmethod
    def dirichlet_uncertainty(classification_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extraire paramètres de Dirichlet, probabilités, masses de croyance et incertitude épistémique."""
        z = classification_logits.float()
        num_classes = z.size(-1)
        evidence = F.softplus(z)
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1, keepdim=True)
        belief = evidence / strength
        uncertainty = (num_classes / strength).squeeze(-1)
        probabilities = alpha / strength
        return {
            "alpha": alpha,
            "probabilities": probabilities,
            "belief": belief,
            "uncertainty": uncertainty,
        }

    @torch.inference_mode()
    def sample_flow(self, *, steps: int = 20, solver: str = "heun", guidance_scale: float = 1.0,
                    time_grid: str = "linear",
                    generator: torch.Generator | None = None,
                    initial_noise: torch.Tensor | None = None, reasoning_steps: int | None = None,
                    **modalities) -> torch.Tensor:
        """Échantillonner des sorties continues conditionnées par les modalités fournies."""
        if self.flow_head is None:
            raise ValueError("sample_flow nécessite flow_matching_dim > 0 dans la configuration")
        was_training = self.training
        self.eval()
        try:
            captured = {}
            hook = self.forecaster.register_forward_pre_hook(
                lambda module, args: captured.update(condition=args[0])
            )
            try:
                self(reasoning_steps=reasoning_steps, **modalities)
            finally:
                hook.remove()
            return self.flow_head.sample(captured["condition"], steps=steps, solver=solver,
                                         guidance_scale=guidance_scale, time_grid=time_grid,
                                         generator=generator, initial_noise=initial_noise)
        finally:
            self.train(was_training)

    _TASK_LOG_VAR_INDEX = {
        "text": 0, "multi_token": 1, "classification": 2,
        "forecast": 3, "flow": 4, "alignment": 5, "jepa": 6,
    }

    def _weight_task_loss(self, name: str, base_weight: float, value: torch.Tensor) -> torch.Tensor:
        if self.task_log_vars is None or base_weight == 0.0:
            return base_weight * value
        idx = self._TASK_LOG_VAR_INDEX[name]
        with torch.autocast(device_type=value.device.type, enabled=False):
            s = self.task_log_vars[idx].float().clamp(-4.0, 4.0)
            weighted = base_weight * (torch.exp(-s) * value.float() + F.softplus(s))
        return weighted.to(value.dtype)

    def _lm_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden)
        cap = self.config.logit_softcap
        if cap > 0.0:
            logits = (cap * torch.tanh(logits.float() / cap)).to(logits.dtype)
        return logits

    @staticmethod
    def _apply_penalties(scores: torch.Tensor, history_ids: torch.Tensor | None, *,
                         repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
                         frequency_penalty: float = 0.0,
                         no_repeat_ngram_size: int = 0) -> torch.Tensor:
        scores = scores.float().clone()
        if history_ids is None or (
            repetition_penalty == 1.0 and presence_penalty == 0.0
            and frequency_penalty == 0.0 and no_repeat_ngram_size <= 0
        ):
            return scores
        if repetition_penalty != 1.0 or presence_penalty != 0.0 or frequency_penalty != 0.0:
            counts = torch.zeros_like(scores)
            ones = torch.ones_like(history_ids, dtype=scores.dtype)
            counts.scatter_add_(1, history_ids, ones)
            present = counts > 0
            if repetition_penalty != 1.0:
                scores = torch.where(present & (scores > 0), scores / repetition_penalty, scores)
                scores = torch.where(present & (scores < 0), scores * repetition_penalty, scores)
            if presence_penalty != 0.0:
                scores = scores - presence_penalty * present.to(scores.dtype)
            if frequency_penalty != 0.0:
                scores = scores - frequency_penalty * counts
        if no_repeat_ngram_size == 1:
            scores.scatter_(1, history_ids, -torch.inf)
        elif no_repeat_ngram_size >= 2 and history_ids.size(1) >= no_repeat_ngram_size:
            n = no_repeat_ngram_size
            windows = history_ids.unfold(1, n, 1)  # [B, L - n + 1, n]
            prefix = history_ids[:, -(n - 1):].unsqueeze(1)  # [B, 1, n - 1]
            matches = (windows[:, :, :-1] == prefix).all(dim=-1)  # [B, L - n + 1]
            # Compter, sans synchronisation ni boucle par ligne, les suites qui recréeraient un N-gramme vu.
            banned = torch.zeros_like(scores, dtype=torch.long).scatter_add_(1, windows[:, :, -1], matches.long())
            scores = scores.masked_fill(banned > 0, -torch.inf)
        return scores

    @staticmethod
    def _mirostat_v2_step(scores: torch.Tensor, mu: torch.Tensor, *,
                          temperature: float, tau: float, eta: float) -> tuple[torch.Tensor, torch.Tensor]:
        scores = scores.float()
        scores[:, [0, 1, 3]] = -torch.inf
        peak = scores.amax(-1, keepdim=True)
        if not torch.isfinite(peak).all():
            raise FloatingPointError("Les logits générables doivent avoir un maximum fini")
        probs = ((scores - peak) / temperature).softmax(-1)
        ordered_probs, order = probs.sort(dim=-1, descending=True)
        surprise = -torch.log2(ordered_probs.clamp_min(1e-12))
        remove = surprise > mu[:, None]
        remove[:, 0] = False
        filtered = ordered_probs.masked_fill(remove, 0.0)
        filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled_pos = torch.multinomial(filtered, 1)
        next_token = order.gather(-1, sampled_pos).squeeze(-1)
        obs_surprise = surprise.gather(-1, sampled_pos).squeeze(-1)
        next_mu = mu - eta * (obs_surprise - tau)
        return next_token, next_mu

    @staticmethod
    def _filtered_probs(scores: torch.Tensor, *, temperature: float, top_k: int, top_p: float,
                        min_p: float = 0.0, typical_p: float = 1.0,
                        eta_cutoff: float = 0.0, tfs_z: float = 1.0,
                        top_a: float = 0.0, edt_theta: float = 0.0,
                        edt_n: float = 0.8) -> torch.Tensor:
        scores = scores.float()
        scores[:, [0, 1, 3]] = -torch.inf
        if not torch.finfo(torch.float32).tiny <= temperature <= torch.finfo(torch.float32).max:
            scores = scores.double()
        peak = scores.amax(-1, keepdim=True)
        if not torch.isfinite(peak).all():
            raise FloatingPointError("Les logits générables doivent avoir un maximum fini")
        scores = scores - peak
        if edt_theta > 0.0:
            p_raw = scores.softmax(-1)
            log_p_raw = scores.log_softmax(-1)
            ent_raw = -(p_raw * torch.where(p_raw > 0, log_p_raw, torch.zeros_like(log_p_raw))).sum(-1, keepdim=True)
            valid_v = (scores > -torch.inf).sum(-1, keepdim=True).clamp_min(2).float()
            norm_ent = (ent_raw / valid_v.log()).clamp(0.0, 1.0)
            eff_temp = temperature * torch.exp(float(edt_theta) * (norm_ent.pow(float(edt_n)) - 1.0))
            scores = scores / eff_temp.clamp_min(1e-6)
        else:
            scores = scores / temperature
        if min_p > 0.0:
            raw_probs = scores.softmax(-1)
            max_prob = raw_probs.amax(-1, keepdim=True)
            scores = scores.masked_fill(raw_probs < min_p * max_prob, -torch.inf)
        if top_a > 0.0:
            p_topa = scores.softmax(-1)
            max_p_topa = p_topa.amax(-1, keepdim=True)
            thresh_topa = float(top_a) * max_p_topa.square()
            scores = scores.masked_fill((p_topa < thresh_topa) & (p_topa < max_p_topa), -torch.inf)
        if eta_cutoff > 0.0:
            p_eta = scores.softmax(-1)
            log_p_eta = scores.log_softmax(-1)
            ent_eta = -(p_eta * torch.where(p_eta > 0, log_p_eta, torch.zeros_like(log_p_eta))).sum(-1, keepdim=True)
            thresh_eta = torch.minimum(
                torch.full_like(ent_eta, float(eta_cutoff)),
                math.sqrt(float(eta_cutoff)) * torch.exp(-ent_eta),
            )
            max_p_eta = p_eta.amax(-1, keepdim=True)
            scores = scores.masked_fill((p_eta < thresh_eta) & (p_eta < max_p_eta), -torch.inf)
        if top_k:
            threshold = scores.topk(min(top_k, scores.size(-1)), dim=-1).values[:, -1:]
            scores = scores.masked_fill(scores < threshold, -torch.inf)
        if typical_p < 1.0:
            p_typ = scores.softmax(-1)
            log_p_typ = scores.log_softmax(-1)
            ent = -(p_typ * torch.where(p_typ > 0, log_p_typ, torch.zeros_like(log_p_typ))).sum(-1, keepdim=True)
            dev = torch.where(p_typ > 0, (-log_p_typ - ent).abs(), torch.full_like(p_typ, torch.inf))
            _, typ_order = dev.sort(dim=-1, descending=False)
            typ_probs = p_typ.gather(-1, typ_order)
            typ_cum = typ_probs.cumsum(-1)
            typ_remove = (typ_cum - typ_probs) >= typical_p
            typ_remove[:, 0] = False
            typ_keep = torch.ones_like(scores, dtype=torch.bool).scatter(-1, typ_order, ~typ_remove)
            scores = scores.masked_fill(~typ_keep, -torch.inf)
        if tfs_z < 1.0 and scores.size(-1) >= 3:
            tfs_ordered, tfs_order = scores.sort(descending=True)
            tfs_p = tfs_ordered.softmax(-1)
            d1 = tfs_p[:, :-1] - tfs_p[:, 1:]
            d2 = (d1[:, :-1] - d1[:, 1:]).abs()
            d2_norm = d2 / d2.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            d2_padded = F.pad(d2_norm, (0, 2), value=0.0)
            d2_cum = d2_padded.cumsum(-1)
            tfs_remove = (d2_cum - d2_padded) >= tfs_z
            tfs_remove[:, 0] = False
            tfs_keep = torch.ones_like(scores, dtype=torch.bool).scatter(-1, tfs_order, ~tfs_remove)
            scores = scores.masked_fill(~tfs_keep, -torch.inf)
        ordered, order = scores.sort(descending=True)
        ordered_probs = ordered.softmax(-1)
        cumulative = ordered_probs.cumsum(-1)
        remove = cumulative - ordered_probs >= top_p
        ordered = ordered.masked_fill(remove, -torch.inf)
        scores = torch.full_like(scores, -torch.inf).scatter(-1, order, ordered)
        return scores.softmax(-1)

    def _draft_future_tokens(self, last_hidden: torch.Tensor, count: int, *,
                             history_ids: torch.Tensor | None = None,
                             repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
                             frequency_penalty: float = 0.0,
                             no_repeat_ngram_size: int = 0) -> list[torch.Tensor]:
        drafts = []
        cur_hist = history_ids
        h_prev = None
        for head_idx, head in enumerate(self.future_heads[:count]):
            if self.mtp_chain_gates is not None and head_idx > 0 and h_prev is not None:
                h_in = last_hidden + torch.tanh(self.mtp_chain_gates[head_idx - 1]).to(last_hidden.dtype) * h_prev
            else:
                h_in = last_hidden
            h_prev = head(h_in)
            scores = self._apply_penalties(
                self._lm_logits(h_prev)[:, -1], cur_hist,
                repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
            )
            scores[:, [0, 1, 3]] = -torch.inf
            tok = scores.argmax(-1)
            drafts.append(tok)
            if cur_hist is not None:
                cur_hist = torch.cat((cur_hist, tok[:, None]), dim=-1)
        return drafts

    def _draft_future_stochastic(self, last_hidden: torch.Tensor, count: int, *,
                                 temperature: float, top_k: int, top_p: float, min_p: float = 0.0,
                                 typical_p: float = 1.0, eta_cutoff: float = 0.0, tfs_z: float = 1.0,
                                 top_a: float = 0.0, edt_theta: float = 0.0, edt_n: float = 0.8,
                                 history_ids: torch.Tensor | None = None,
                                 repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
                                 frequency_penalty: float = 0.0,
                                 no_repeat_ngram_size: int = 0):
        drafts, q_probs = [], []
        cur_hist = history_ids
        h_prev = None
        for head_idx, head in enumerate(self.future_heads[:count]):
            if self.mtp_chain_gates is not None and head_idx > 0 and h_prev is not None:
                h_in = last_hidden + torch.tanh(self.mtp_chain_gates[head_idx - 1]).to(last_hidden.dtype) * h_prev
            else:
                h_in = last_hidden
            h_prev = head(h_in)
            penalized = self._apply_penalties(
                self._lm_logits(h_prev)[:, -1], cur_hist,
                repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
            )
            probs = self._filtered_probs(
                penalized,
                temperature=temperature, top_k=top_k, top_p=top_p,
                min_p=min_p, typical_p=typical_p, eta_cutoff=eta_cutoff, tfs_z=tfs_z,
                top_a=top_a, edt_theta=edt_theta, edt_n=edt_n,
            )
            tok = torch.multinomial(probs, 1).squeeze(-1)
            drafts.append(tok)
            q_probs.append(probs)
            if cur_hist is not None:
                cur_hist = torch.cat((cur_hist, tok[:, None]), dim=-1)
        return drafts, q_probs

    @staticmethod
    def _reorder_cache(cache: ModelCache, beam_idx: torch.Tensor) -> ModelCache:
        """Sélectionner les lignes ``beam_idx`` de chaque tenseur du cache.

        Le parcours générique des champs (couches, contexte de l'adaptateur...)
        évite qu'un champ ajouté à l'une des structures soit oublié ici.
        """
        def select(value):
            if isinstance(value, torch.Tensor):
                return value.index_select(0, beam_idx)
            if isinstance(value, list):
                return [select(element) for element in value]
            if dataclasses.is_dataclass(value):
                return dataclasses.replace(value, **{field.name: select(getattr(value, field.name))
                                                     for field in dataclasses.fields(value)})
            return value  # Positions, indicateurs : identiques pour toutes les lignes.

        return select(cache)

    @torch.inference_mode()
    def generate(self, input_ids, *, max_new_tokens=64, temperature=0.8, top_k=40,
                 top_p=0.95, min_p: float = 0.0, typical_p: float = 1.0,
                 eta_cutoff: float = 0.0, tfs_z: float = 1.0, top_a: float = 0.0,
                 edt_theta: float = 0.0, edt_n: float = 0.8,
                 eos_id=2, reasoning_steps: int | None = None,
                 speculative: bool = False, jacobi: bool = False,
                 jacobi_window: int = 4, beam_width: int = 1, length_penalty: float = 1.0,
                 repetition_penalty: float = 1.0,
                 presence_penalty: float = 0.0, frequency_penalty: float = 0.0,
                 no_repeat_ngram_size: int = 0, context_guidance_scale: float = 1.0,
                 mirostat_tau: float = 0.0, mirostat_eta: float = 0.1,
                 **context):
        if input_ids.ndim != 2 or input_ids.size(1) < 1:
            raise ValueError("Le prompt doit avoir la forme [B, T] avec T > 0")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens doit être un entier positif ou nul")
        if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("temperature doit être >= 0 et top_p dans ]0, 1]")
        if not math.isfinite(min_p) or not 0.0 <= min_p < 1.0:
            raise ValueError("min_p doit appartenir à [0, 1[")
        if not math.isfinite(typical_p) or not 0.0 < typical_p <= 1.0:
            raise ValueError("typical_p doit appartenir à ]0, 1]")
        if not math.isfinite(eta_cutoff) or not 0.0 <= eta_cutoff < 1.0:
            raise ValueError("eta_cutoff doit appartenir à [0, 1[")
        if not math.isfinite(tfs_z) or not 0.0 < tfs_z <= 1.0:
            raise ValueError("tfs_z doit appartenir à ]0, 1]")
        if not math.isfinite(top_a) or not 0.0 <= top_a < 1.0:
            raise ValueError("top_a doit appartenir à [0, 1[")
        if not math.isfinite(edt_theta) or edt_theta < 0.0:
            raise ValueError("edt_theta doit être fini et >= 0")
        if not math.isfinite(edt_n) or edt_n <= 0.0:
            raise ValueError("edt_n doit être fini et > 0")
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k doit être un entier positif ou nul")
        if type(speculative) is not bool:
            raise ValueError("speculative doit être booléen")
        if type(jacobi) is not bool:
            raise ValueError("jacobi doit être booléen")
        if type(jacobi_window) is not int or jacobi_window < 1:
            raise ValueError("jacobi_window doit être un entier >= 1")
        if type(beam_width) is not int or beam_width < 1:
            raise ValueError("beam_width doit être un entier >= 1")
        if not math.isfinite(length_penalty) or length_penalty < 0:
            raise ValueError("length_penalty doit être fini et >= 0")
        if jacobi and temperature != 0:
            raise ValueError("Le décodage par point fixe de Jacobi nécessite temperature=0")
        if not math.isfinite(repetition_penalty) or repetition_penalty <= 0:
            raise ValueError("repetition_penalty doit être fini et > 0")
        if not math.isfinite(presence_penalty) or not math.isfinite(frequency_penalty):
            raise ValueError("presence_penalty et frequency_penalty doivent être finis")
        if type(no_repeat_ngram_size) is not int or no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size doit être un entier positif ou nul")
        if not math.isfinite(context_guidance_scale):
            raise ValueError("context_guidance_scale doit être fini")
        if not math.isfinite(mirostat_tau) or mirostat_tau < 0:
            raise ValueError("mirostat_tau doit être fini et >= 0")
        if not math.isfinite(mirostat_eta) or mirostat_eta <= 0:
            raise ValueError("mirostat_eta doit être fini et > 0")
        if input_ids.size(1) + max_new_tokens > self.config.max_seq_len:
            raise ValueError("Prompt + génération dépassent max_seq_len")
        if eos_id is not None and (type(eos_id) is not int or not 0 <= eos_id < self.config.vocab_size
                                   or eos_id in (0, 1, 3)):
            raise ValueError("eos_id doit être un entier générable du vocabulaire ou None")
        context = {name: value for name, value in context.items() if value is not None}
        if "attention_mask" in context:
            mask = context["attention_mask"]
            if mask.ndim != 2 or not mask[:, -1].bool().all():
                raise ValueError("La génération en batch nécessite du padding à gauche")
        if sum((beam_width > 1, jacobi, speculative)) > 1:
            raise ValueError("beam_width > 1, jacobi et speculative sont mutuellement exclusifs")
        if beam_width > 1 or jacobi or speculative:
            # Ces décodeurs n'implémentent ni le guidage ni Mirostat : refuser
            # plutôt que d'ignorer silencieusement une option demandée.
            mode = "beam search" if beam_width > 1 else "jacobi" if jacobi else "speculative"
            if context_guidance_scale != 1.0:
                raise ValueError(f"context_guidance_scale est incompatible avec le décodage {mode}")
            if mirostat_tau > 0.0:
                raise ValueError(f"mirostat_tau est incompatible avec le décodage {mode}")
        was_training = self.training
        self.eval()
        penalties = dict(
            repetition_penalty=float(repetition_penalty),
            presence_penalty=float(presence_penalty),
            frequency_penalty=float(frequency_penalty),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
        )
        try:
            if beam_width > 1 and max_new_tokens > 0:
                rows = [
                    self._generate_beam_search(
                        input_ids[row:row + 1], max_new_tokens=max_new_tokens, beam_width=beam_width,
                        length_penalty=length_penalty, eos_id=eos_id, reasoning_steps=reasoning_steps,
                        **penalties, **{name: value[row:row + 1] for name, value in context.items()},
                    )
                    for row in range(input_ids.size(0))
                ]
                # Les lignes achevées plus tôt sont complétées par eos_id, comme en glouton.
                width = max(row.size(1) for row in rows)
                return torch.cat([
                    row if row.size(1) == width else F.pad(row, (0, width - row.size(1)), value=eos_id)
                    for row in rows
                ])
            if jacobi and input_ids.size(0) == 1 and max_new_tokens > 0:
                return self._generate_jacobi_greedy(
                    input_ids, max_new_tokens=max_new_tokens, eos_id=eos_id,
                    reasoning_steps=reasoning_steps, jacobi_window=jacobi_window,
                    **penalties, **context,
                )
            if (speculative and input_ids.size(0) == 1
                    and max_new_tokens > 0 and len(self.future_heads) > 0):
                if temperature == 0:
                    return self._generate_speculative_greedy(
                        input_ids, max_new_tokens=max_new_tokens, eos_id=eos_id,
                        reasoning_steps=reasoning_steps, **penalties, **context,
                    )
                return self._generate_speculative_stochastic(
                    input_ids, max_new_tokens=max_new_tokens, temperature=temperature,
                    top_k=top_k, top_p=top_p, min_p=min_p, typical_p=typical_p,
                    eta_cutoff=eta_cutoff, tfs_z=tfs_z, top_a=top_a,
                    edt_theta=edt_theta, edt_n=edt_n, eos_id=eos_id,
                    reasoning_steps=reasoning_steps, **penalties, **context,
                )
            result = input_ids
            finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
            mirostat_mu = (
                torch.full((input_ids.size(0),), 2.0 * float(mirostat_tau), device=input_ids.device, dtype=torch.float32)
                if (mirostat_tau > 0.0 and temperature > 0.0) else None
            )
            cache = None
            uncond_cache = None
            has_mm_ctx = any(
                context.get(k) is not None
                for k in ("images", "series", "audio", "graph_nodes", "points", "tabular_num", "video")
            )
            use_cfg = context_guidance_scale != 1.0 and has_mm_ctx
            uncond_ctx = {"attention_mask": context["attention_mask"]} if "attention_mask" in context else {}
            for step in range(max_new_tokens):
                cur_in = input_ids if step == 0 else result[:, -1:]
                output = self(cur_in, cache=cache, use_cache=True, reasoning_steps=reasoning_steps,
                              **(context if step == 0 else {}))
                cache = output.cache
                raw_logits = output.logits[:, -1]
                if use_cfg:
                    uncond_out = self(cur_in, cache=uncond_cache, use_cache=True,
                                      reasoning_steps=reasoning_steps,
                                      **(uncond_ctx if step == 0 else {}))
                    uncond_cache = uncond_out.cache
                    u_logits = uncond_out.logits[:, -1].float()
                    raw_logits = u_logits + float(context_guidance_scale) * (raw_logits.float() - u_logits)
                scores = self._apply_penalties(raw_logits, result, **penalties)
                # Les tokens de contrôle réservés ne sont pas des caractères générables.
                scores[:, [0, 1, 3]] = -torch.inf
                if temperature == 0:
                    next_token = scores.argmax(-1)
                elif mirostat_mu is not None:
                    next_token, mirostat_mu = self._mirostat_v2_step(
                        scores, mirostat_mu,
                        temperature=temperature, tau=float(mirostat_tau), eta=float(mirostat_eta),
                    )
                else:
                    probs = self._filtered_probs(
                        scores,
                        temperature=temperature, top_k=top_k, top_p=top_p,
                        min_p=min_p, typical_p=typical_p, eta_cutoff=eta_cutoff, tfs_z=tfs_z,
                        top_a=top_a, edt_theta=edt_theta, edt_n=edt_n,
                    )
                    next_token = torch.multinomial(probs, 1).squeeze(-1)
                if eos_id is not None:
                    next_token = torch.where(finished, eos_id, next_token)
                    finished = finished | (next_token == eos_id)
                result = torch.cat((result, next_token[:, None]), -1)
                if eos_id is not None and finished.all():
                    break
            return result
        finally:
            self.train(was_training)

    def _penalized_verify_logits(self, verify_logits: torch.Tensor, result: torch.Tensor,
                                 candidate: torch.Tensor, **penalties) -> torch.Tensor:
        rows = []
        for j in range(verify_logits.size(1)):
            hist = result if j == 0 else torch.cat((result, candidate[:, 1:j + 1]), dim=-1)
            row = self._apply_penalties(verify_logits[:, j], hist, **penalties)
            row[:, [0, 1, 3]] = -torch.inf
            rows.append(row)
        return torch.cat(rows, dim=0)

    @staticmethod
    def _accepted_prefix(verified: torch.Tensor, drafts) -> int:
        """Nombre de brouillons consécutifs égaux aux tokens vérifiés."""
        accepted = 0
        for j, draft in enumerate(drafts):
            if verified[j].item() != draft.item():
                break
            accepted += 1
        return accepted

    def _parallel_decode(self, input_ids, *, max_new_tokens, eos_id, reasoning_steps, window,
                         first_token, propose_round, context):
        """Boucle commune des décodeurs auto-spéculatifs et de Jacobi (batch de 1).

        ``first_token(logits, history)`` choisit un token à partir de logits bruts ;
        ``propose_round(result, next_token, last_hidden, cache, n_draft)`` propose puis
        vérifie ``n_draft`` brouillons et renvoie ``(emitted, accepted, candidate, verify_out)``.
        Le cache n'est resynchronisé que sur les tokens réellement acceptés.
        """
        result = input_ids
        output = self(input_ids, cache=None, use_cache=True, reasoning_steps=reasoning_steps, **context)
        cache = output.cache
        next_token = first_token(output.logits[:, -1], result)
        result = torch.cat((result, next_token[:, None]), -1)
        generated = 1
        if eos_id is not None and next_token.item() == eos_id:
            return result
        last_hidden = output.last_hidden
        while generated < max_new_tokens:
            n_draft = min(window, max_new_tokens - generated - 1)
            if n_draft == 0:
                step_out = self(next_token[:, None], cache=cache, use_cache=True, reasoning_steps=reasoning_steps)
                next_token = first_token(step_out.logits[:, -1], result)
                result = torch.cat((result, next_token[:, None]), -1)
                break
            emitted, accepted, candidate, verify_out = propose_round(result, next_token, last_hidden, cache, n_draft)
            stop_now = False
            if eos_id is not None:
                eos_positions = (emitted == eos_id).nonzero(as_tuple=True)[0]
                if eos_positions.numel():
                    emitted = emitted[:eos_positions[0].item() + 1]
                    stop_now = True
            result = torch.cat((result, emitted[None, :]), -1)
            generated += emitted.numel()
            if stop_now or generated >= max_new_tokens:
                break
            next_token = emitted[-1:]
            if accepted == n_draft:
                # Tous les brouillons acceptés : le cache de vérification est déjà synchronisé.
                cache, last_hidden = verify_out.cache, verify_out.last_hidden
            else:
                sync_out = self(candidate[:, :accepted + 1], cache=cache, use_cache=True,
                                reasoning_steps=reasoning_steps)
                cache, last_hidden = sync_out.cache, sync_out.last_hidden
        return result

    def _greedy_token(self, logits, history, penalties):
        scores = self._apply_penalties(logits, history, **penalties)
        scores[:, [0, 1, 3]] = -torch.inf
        return scores.argmax(-1)

    def _generate_speculative_greedy(self, input_ids, *, max_new_tokens, eos_id, reasoning_steps,
                                     repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
                                     no_repeat_ngram_size=0,
                                     **context):
        """Décodage auto-spéculatif exact (greedy) exploitant les future_heads."""
        penalties = dict(
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        def propose_round(result, next_token, last_hidden, cache, n_draft):
            drafts = self._draft_future_tokens(last_hidden, n_draft, history_ids=result, **penalties)
            candidate = torch.stack([next_token, *drafts], dim=1)  # [1, 1 + n_draft]
            verify_out = self(candidate, cache=cache, use_cache=True, reasoning_steps=reasoning_steps)
            # verified[j] est le token exact qui suit candidate[0..j].
            verified = self._penalized_verify_logits(verify_out.logits, result, candidate, **penalties).argmax(-1)
            accepted = self._accepted_prefix(verified, drafts)
            return verified[:accepted + 1], accepted, candidate, verify_out

        return self._parallel_decode(
            input_ids, max_new_tokens=max_new_tokens, eos_id=eos_id, reasoning_steps=reasoning_steps,
            window=len(self.future_heads), first_token=lambda logits, history: self._greedy_token(logits, history, penalties),
            propose_round=propose_round, context=context)

    def _generate_speculative_stochastic(self, input_ids, *, max_new_tokens, temperature,
                                         top_k, top_p, min_p=0.0, typical_p=1.0, eta_cutoff=0.0,
                                         tfs_z=1.0, top_a=0.0, edt_theta=0.0, edt_n=0.8,
                                         eos_id, reasoning_steps,
                                         repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
                                         no_repeat_ngram_size=0,
                                         **context):
        """Décodage auto-spéculatif stochastique exact (échantillonnage par rejet de Leviathan et al.)."""
        penalties = dict(
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        filters = dict(temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p, typical_p=typical_p,
                       eta_cutoff=eta_cutoff, tfs_z=tfs_z, top_a=top_a, edt_theta=edt_theta, edt_n=edt_n)

        def target_probs(logits, history):
            return self._filtered_probs(self._apply_penalties(logits, history, **penalties), **filters)

        def sample(logits, history):
            return torch.multinomial(target_probs(logits, history), 1).squeeze(-1)

        def propose_round(result, next_token, last_hidden, cache, n_draft):
            drafts, q_probs = self._draft_future_stochastic(last_hidden, n_draft, **filters,
                                                            history_ids=result, **penalties)
            candidate = torch.stack([next_token, *drafts], dim=1)  # [1, 1 + n_draft]
            verify_out = self(candidate, cache=cache, use_cache=True, reasoning_steps=reasoning_steps)
            accepted = 0
            emitted = []
            for j in range(n_draft):
                history = result if j == 0 else torch.cat((result, candidate[:, 1:j + 1]), dim=-1)
                p_j = target_probs(verify_out.logits[:, j], history)
                token = drafts[j].item()
                p_token = p_j[0, token].item()
                q_token = q_probs[j][0, token].item()
                u = torch.rand((), device=input_ids.device).item()
                if q_token > 0.0 and u <= min(1.0, p_token / q_token):
                    emitted.append(drafts[j])
                    accepted += 1
                    continue
                # Rejet : rééchantillonner selon le résidu normalisé max(p - q, 0).
                residual = (p_j - q_probs[j]).clamp_min(0.0)
                residual_sum = residual.sum().item()
                corrected = torch.multinomial(residual / residual_sum if residual_sum > 0.0 else p_j, 1)
                emitted.append(corrected.squeeze(-1))
                break
            else:
                # Tous acceptés : un token bonus tiré de la distribution cible suivante.
                bonus_history = torch.cat((result, candidate[:, 1:n_draft + 1]), dim=-1)
                emitted.append(sample(verify_out.logits[:, n_draft], bonus_history))
            return torch.cat(emitted, dim=0), accepted, candidate, verify_out

        return self._parallel_decode(
            input_ids, max_new_tokens=max_new_tokens, eos_id=eos_id, reasoning_steps=reasoning_steps,
            window=len(self.future_heads), first_token=sample, propose_round=propose_round, context=context)

    def _generate_jacobi_greedy(self, input_ids, *, max_new_tokens, eos_id,
                                reasoning_steps, jacobi_window,
                                repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
                                no_repeat_ngram_size=0,
                                **context):
        """Décodage parallèle par itération de point fixe de Jacobi (lookahead exact sans future_heads requis)."""
        penalties = dict(
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        carry: list[torch.Tensor] = []

        def verify(result, next_token, cache, drafts):
            candidate = torch.stack([next_token, *drafts], dim=1)
            verify_out = self(candidate, cache=cache, use_cache=True, reasoning_steps=reasoning_steps)
            verified = self._penalized_verify_logits(verify_out.logits, result, candidate, **penalties).argmax(-1)
            return candidate, verify_out, verified, self._accepted_prefix(verified, drafts)

        def propose_round(result, next_token, last_hidden, cache, n_draft):
            nonlocal carry
            drafts = list(carry[:n_draft])
            if len(drafts) < n_draft and len(self.future_heads) > 0 and last_hidden is not None:
                head_drafts = self._draft_future_tokens(last_hidden, min(n_draft, len(self.future_heads)),
                                                        history_ids=result, **penalties)
                drafts.extend(head_drafts[len(drafts):n_draft])
            while len(drafts) < n_draft:
                drafts.append(drafts[-1] if drafts else next_token)
            candidate, verify_out, verified, accepted = verify(result, next_token, cache, drafts)
            if accepted == 0:
                # Raffinement de point fixe sur la trajectoire mise à jour : drafts[0] = verified[0]
                # est désormais exact, ce qui garantit accepted >= 1.
                drafts = [verified[j].unsqueeze(0) for j in range(n_draft)]
                candidate, verify_out, verified, accepted = verify(result, next_token, cache, drafts)
            carry = [verified[j].unsqueeze(0) for j in range(accepted + 1, n_draft + 1)]
            return verified[:accepted + 1], accepted, candidate, verify_out

        return self._parallel_decode(
            input_ids, max_new_tokens=max_new_tokens, eos_id=eos_id, reasoning_steps=reasoning_steps,
            window=jacobi_window, first_token=lambda logits, history: self._greedy_token(logits, history, penalties),
            propose_round=propose_round, context=context)

    def _generate_beam_search(self, input_ids, *, max_new_tokens, beam_width, length_penalty,
                              eos_id, reasoning_steps, repetition_penalty=1.0,
                              presence_penalty=0.0, frequency_penalty=0.0,
                              no_repeat_ngram_size=0, **context):
        """Recherche de faisceau exacte exploitant le cache KV/SSM/TTT/Mémoire/NGram réordonné à chaque pas."""
        penalties = dict(
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        device = input_ids.device
        vocab_size = self.config.vocab_size
        output = self(input_ids, cache=None, use_cache=True, reasoning_steps=reasoning_steps, **context)
        scores = self._apply_penalties(output.logits[:, -1], input_ids, **penalties)
        scores[:, [0, 1, 3]] = -torch.inf
        log_probs = scores.log_softmax(-1)
        # Le premier pas n'offre que vocab_size candidats ; les suivants élargissent le faisceau.
        width = min(beam_width, vocab_size)
        top_lp, top_tok = log_probs[0].topk(width, dim=-1)
        seqs = torch.cat((input_ids.expand(width, -1), top_tok[:, None]), dim=-1)
        beam_scores = top_lp
        beam_lengths = torch.ones(width, dtype=torch.long, device=device)
        finished = (
            (top_tok == eos_id) if eos_id is not None
            else torch.zeros(width, dtype=torch.bool, device=device)
        )
        cache = self._reorder_cache(
            output.cache, torch.zeros(width, dtype=torch.long, device=device),
        )
        for _ in range(1, max_new_tokens):
            if eos_id is not None and bool(finished.all()):
                break
            out = self(seqs[:, -1:], cache=cache, use_cache=True, reasoning_steps=reasoning_steps)
            next_cache = out.cache
            step_scores = self._apply_penalties(out.logits[:, -1], seqs, **penalties)
            step_scores[:, [0, 1, 3]] = -torch.inf
            step_lp = step_scores.log_softmax(-1)
            if eos_id is not None and bool(finished.any()):
                eos_row = torch.full_like(step_lp[0], -torch.inf)
                eos_row[eos_id] = 0.0
                step_lp = torch.where(finished[:, None], eos_row[None, :], step_lp)
            cand_scores = beam_scores[:, None] + step_lp
            flat_scores = cand_scores.view(-1)
            top_scores, flat_idx = flat_scores.topk(min(beam_width, flat_scores.numel()), dim=-1)
            parent_beams = flat_idx // vocab_size
            next_tokens = flat_idx % vocab_size
            prev_fin = finished.index_select(0, parent_beams)
            seqs = torch.cat((seqs.index_select(0, parent_beams), next_tokens[:, None]), dim=-1)
            beam_scores = top_scores
            beam_lengths = beam_lengths.index_select(0, parent_beams) + (~prev_fin).long()
            finished = prev_fin if eos_id is None else prev_fin | (next_tokens == eos_id)
            cache = self._reorder_cache(next_cache, parent_beams)
        norm_factor = ((5.0 + beam_lengths.float()) / 6.0) ** length_penalty
        best_beam = int((beam_scores / norm_factor).argmax(dim=0).item())
        # Ne pas renvoyer les eos ajoutés pendant que les autres faisceaux se terminaient.
        length = input_ids.size(1) + int(beam_lengths[best_beam])
        return seqs[best_beam:best_beam + 1, :length]
