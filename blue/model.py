from dataclasses import dataclass
import math

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
    DecoderBlock, LayerCache, LatentResampler, QuantileForecastHead,
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


def masked_cross_entropy(logits, labels):
    # Un lot sans cible valide ne produit ni NaN ni apprentissage artificiel.
    values = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                             labels.reshape(-1), ignore_index=-100, reduction="none")
    return values.sum() / (labels != -100).sum().clamp_min(1)


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
        self.episodic_memory = (EpisodicMemoryBank(d, config.memory_slots, config.n_heads, norm_eps=config.norm_eps)
                                if config.memory_slots > 0 else None)
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.future_heads = nn.ModuleList([
            nn.Sequential(RMSNorm(d), nn.Linear(d, d, bias=False), nn.SiLU())
            for _ in range(config.prediction_steps - 1)
        ])
        self.fusion = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), RMSNorm(d))
        self.classifier = nn.Linear(d, config.num_classes)
        self.forecaster = QuantileForecastHead(config)
        self.flow_head = ConditionalFlowHead(config) if config.flow_matching_dim > 0 else None
        self.jepa = MultimodalJEPAPredictor(d, norm_eps=config.norm_eps) if config.jepa_weight > 0 else None
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
        nn.init.zeros_(self.temporal.forecast_skip.weight)
        if self.temporal.statistics_proj is not None:
            nn.init.zeros_(self.temporal.statistics_proj.weight)
        if self.temporal.spectral is not None:
            self.temporal.spectral.reset_spectral_parameters()
        if self.temporal.wavelet is not None:
            self.temporal.wavelet.reset_wavelet_parameters()
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
        if self.episodic_memory is not None:
            self.episodic_memory.reset_memory_parameters()
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
        for mask, data, name, expected_shape in (
            (attention_mask, input_ids, "attention_mask", None if input_ids is None else input_ids.shape[:2]),
            (series_mask, series, "series_mask", None if series is None else series.shape[:2]),
            (audio_mask, audio, "audio_mask", None if audio is None else (audio.size(0), audio.size(2))),
            (graph_mask, graph_nodes, "graph_mask", None if graph_nodes is None else graph_nodes.shape[:2]),
            (point_mask, points, "point_mask", None if points is None else points.shape[:2]),
            (video_mask, video, "video_mask", None if video is None else (video.size(0), video.size(2))),
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
        if images is not None:
            tokens, _ = self.vision(images)
            representations['image'] = (0, tokens.mean(1))
        if series is not None:
            valid = (torch.ones(series.shape[:2], device=series.device, dtype=torch.bool)
                     if series_mask is None else series_mask.bool())
            tokens, mask, *_ = self.temporal(series, valid)
            pooled = (tokens * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            representations['series'] = (1, pooled)
        if audio is not None:
            valid = (torch.ones((audio.size(0), audio.size(2)), device=audio.device, dtype=torch.bool)
                     if audio_mask is None else audio_mask.bool())
            tokens, mask = self.audio(audio, valid)
            pooled = (tokens * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            representations['audio'] = (3, pooled)
        if graph_nodes is not None:
            valid = (torch.ones(graph_nodes.shape[:2], device=graph_nodes.device, dtype=torch.bool)
                     if graph_mask is None else graph_mask.bool())
            edges = self._default_graph_edges(graph_nodes) if graph_edges is None else graph_edges
            tokens, mask = self.graph(graph_nodes, edges, valid)
            pooled = (tokens * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            representations['graph'] = (4, pooled)
        if points is not None:
            valid = (torch.ones(points.shape[:2], device=points.device, dtype=torch.bool)
                     if point_mask is None else point_mask.bool())
            tokens, mask = self.pointcloud(points, valid, point_features)
            pooled = (tokens * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            representations['pointcloud'] = (5, pooled)
        if tabular_num is not None:
            tokens, mask = self.tabular(tabular_num, tabular_cat)
            pooled = (tokens * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            representations['tabular'] = (6, pooled)
        if video is not None:
            valid = (torch.ones((video.size(0), video.size(2)), device=video.device, dtype=torch.bool)
                     if video_mask is None else video_mask.bool())
            tokens, mask = self.video(video, valid)
            pooled = (tokens * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            representations['video'] = (7, pooled)
        result = {}
        for name, (index, value) in representations.items():
            vector = self._project_alignment(index, value)
            if not torch.isfinite(vector).all():
                raise FloatingPointError("Les représentations contrastives doivent être finies")
            result[name] = vector
        return result

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
        balance_losses, z_losses, depth_losses, next_layers = [], [], [], []
        layer_usage = [None] * self.config.n_layers
        adaptive = self.ponder_halt is not None and steps > 1
        step_states = [] if adaptive else None
        halt_hazards = [] if adaptive else None
        for rec_step in range(steps):
            x = self._prepare_reasoning_step(x, initial, valid, rec_step)
            v0 = None
            for index, block in enumerate(self.blocks):
                cache_slot = rec_step * self.config.n_layers + index
                use_ckpt = (
                    self.config.gradient_checkpointing
                    and self.training
                    and (index % self.config.checkpoint_every == 0)
                )
                if use_ckpt:
                    if self.config.value_residual and index == 0:
                        def run0(hidden, memory, current=block):
                            result = current(hidden, valid, memory)
                            d_loss = result[5] if result[5] is not None else hidden.sum() * 0
                            return result[0], result[1], result[2], result[3], d_loss, result[6]
                        x, balance, z_loss, expert_usage, depth_loss, v0 = checkpoint(run0, x, latents, use_reentrant=False)
                    elif self.config.value_residual and v0 is not None:
                        def run_v(hidden, memory, v_first, current=block):
                            result = current(hidden, valid, memory, v0=v_first)
                            d_loss = result[5] if result[5] is not None else hidden.sum() * 0
                            return result[0], result[1], result[2], result[3], d_loss
                        x, balance, z_loss, expert_usage, depth_loss = checkpoint(run_v, x, latents, v0, use_reentrant=False)
                    else:
                        def run(hidden, memory, current=block):
                            result = current(hidden, valid, memory)
                            d_loss = result[5] if result[5] is not None else hidden.sum() * 0
                            return result[0], result[1], result[2], result[3], d_loss
                        x, balance, z_loss, expert_usage, depth_loss = checkpoint(run, x, latents, use_reentrant=False)
                    if block.depth_router is None:
                        depth_loss = None
                else:
                    result = block(
                        x, valid, latents, offset=offset,
                        cache=None if cache is None else cache.layers[cache_slot],
                        use_cache=use_cache, v0=v0,
                    )
                    if self.config.value_residual and index == 0:
                        x, balance, z_loss, expert_usage, layer_cache, depth_loss, v0 = result
                    else:
                        x, balance, z_loss, expert_usage, layer_cache, depth_loss = result
                    if use_cache:
                        next_layers.append(layer_cache)
                balance_losses.append(balance)
                z_losses.append(z_loss)
                if depth_loss is not None:
                    depth_losses.append(depth_loss)
                layer_usage[index] = expert_usage if layer_usage[index] is None else layer_usage[index] + expert_usage
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
        return self.norm(collapsed), balance_losses, z_losses, depth_losses, ponder_kl, usage, next_layers, next_memory

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
            if images is not None:
                image_tokens, image_valid = self.vision(images)
                contexts.append(image_tokens + self.modality[0])
                masks.append(image_valid)
                align_vectors.append((0, image_tokens.mean(1)))
            if series is not None:
                series_mask = (torch.ones(series.shape[:2], device=series.device, dtype=torch.bool)
                               if series_mask is None else series_mask.bool())
                series_tokens, temporal_valid, mean, scale, baseline = self.temporal(series, series_mask)
                if self.forecast_adapter is not None:
                    adapter_context = self.forecast_adapter.prepare(series, temporal_valid, mean, scale)
                contexts.append(series_tokens + self.modality[1])
                masks.append(temporal_valid)
                pooled_series = (series_tokens * temporal_valid.unsqueeze(-1)).sum(1)
                pooled_series = pooled_series / temporal_valid.sum(1, keepdim=True).clamp_min(1)
                align_vectors.append((1, pooled_series))
            if audio is not None:
                audio_mask = (torch.ones((audio.size(0), audio.size(2)), device=audio.device, dtype=torch.bool)
                              if audio_mask is None else audio_mask.bool())
                audio_tokens, audio_valid = self.audio(audio, audio_mask)
                contexts.append(audio_tokens + self.audio_modality)
                masks.append(audio_valid)
                pooled_audio = (audio_tokens * audio_valid.unsqueeze(-1)).sum(1)
                pooled_audio = pooled_audio / audio_valid.sum(1, keepdim=True).clamp_min(1)
                align_vectors.append((3, pooled_audio))
            if graph_nodes is not None:
                graph_valid = (torch.ones(graph_nodes.shape[:2], device=graph_nodes.device, dtype=torch.bool)
                               if graph_mask is None else graph_mask.bool())
                edges = self._default_graph_edges(graph_nodes) if graph_edges is None else graph_edges
                graph_tokens, graph_valid = self.graph(graph_nodes, edges, graph_valid)
                contexts.append(graph_tokens + self.graph_modality)
                masks.append(graph_valid)
                pooled_graph = (graph_tokens * graph_valid.unsqueeze(-1)).sum(1)
                pooled_graph = pooled_graph / graph_valid.sum(1, keepdim=True).clamp_min(1)
                align_vectors.append((4, pooled_graph))
            if points is not None:
                point_valid = (torch.ones(points.shape[:2], device=points.device, dtype=torch.bool)
                               if point_mask is None else point_mask.bool())
                point_tokens, point_valid = self.pointcloud(points, point_valid, point_features)
                contexts.append(point_tokens + self.pointcloud_modality)
                masks.append(point_valid)
                pooled_pc = (point_tokens * point_valid.unsqueeze(-1)).sum(1)
                pooled_pc = pooled_pc / point_valid.sum(1, keepdim=True).clamp_min(1)
                align_vectors.append((5, pooled_pc))
            if tabular_num is not None:
                tab_tokens, tab_valid = self.tabular(tabular_num, tabular_cat)
                contexts.append(tab_tokens + self.tabular_modality)
                masks.append(tab_valid)
                pooled_tab = (tab_tokens * tab_valid.unsqueeze(-1)).sum(1)
                pooled_tab = pooled_tab / tab_valid.sum(1, keepdim=True).clamp_min(1)
                align_vectors.append((6, pooled_tab))
            if video is not None:
                vid_frames_valid = (torch.ones((video.size(0), video.size(2)), device=video.device, dtype=torch.bool)
                                    if video_mask is None else video_mask.bool())
                vid_tokens, vid_valid = self.video(video, vid_frames_valid)
                contexts.append(vid_tokens + self.video_modality)
                masks.append(vid_valid)
                pooled_vid = (vid_tokens * vid_valid.unsqueeze(-1)).sum(1)
                pooled_vid = pooled_vid / vid_valid.sum(1, keepdim=True).clamp_min(1)
                align_vectors.append((7, pooled_vid))
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
        hidden, balance_losses, z_losses, depth_losses, ponder_kl, usage, next_layers, next_memory = self._run_decoder_stack(
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
        class_pooled = self.fusion(torch.cat((class_pooled, memory_pooled), -1))
        forecast_pooled = self.fusion(torch.cat((forecast_pooled, memory_pooled), -1))
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
            losses["text"] = masked_cross_entropy(logits[:, :-1], shifted)
            total = total + self._weight_task_loss("text", c.text_weight, losses["text"])
            mtp_losses, mtp_active = [], []
            for distance, head in enumerate(self.future_heads, start=2):
                if hidden.size(1) > distance:
                    future_logits = self._lm_logits(head(hidden[:, :-distance]))
                    future_targets = targets[:, distance:].masked_fill(~valid[:, :-distance], -100)
                    head_loss = masked_cross_entropy(future_logits, future_targets)
                    head_count = (future_targets != -100).sum()
                    auxiliary_text_losses[distance] = (head_loss, head_count)
                    mtp_losses.append(head_loss)
                    mtp_active.append(head_count > 0)
            if mtp_losses:
                # Les horizons sans cible ne doivent pas diluer les horizons actifs.
                active = torch.stack(mtp_active).to(dtype=mtp_losses[0].dtype)
                losses["multi_token"] = (torch.stack(mtp_losses) * active).sum() / active.sum().clamp_min(1)
                total = total + self._weight_task_loss("multi_token", c.mtp_weight, losses["multi_token"])
        if class_labels is not None:
            losses["classification"] = masked_cross_entropy(class_logits, class_labels)
            total = total + self._weight_task_loss("classification", c.classification_weight, losses["classification"])
        if forecast_targets is not None:
            losses["forecast"] = quantile_pinball_loss(forecast, forecast_targets, c.quantiles)
            total = total + self._weight_task_loss("forecast", c.forecast_weight, losses["forecast"])
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
                                                              curvature=c.alignment_curvature)
            total = total + self._weight_task_loss("alignment", c.alignment_weight, losses["alignment"])
        elif len(align_vectors) > 1 and batch > 1:
            vectors = [self._project_alignment(i, v) for i, v in align_vectors]
            contrastive = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    with torch.autocast(device_type=x.device.type, enabled=False):
                        if c.alignment_geometry == "hyperbolic":
                            similarity = -poincare_pairwise_distance(vectors[i], vectors[j],
                                                                     c.alignment_curvature) / c.alignment_temperature
                        else:
                            similarity = vectors[i] @ vectors[j].T / c.alignment_temperature
                    contrastive.append(symmetric_contrastive_loss(
                        similarity, class_labels if c.alignment_mode == "class" else None, sample_ids=alignment_ids,
                    ))
            losses["alignment"] = torch.stack(contrastive).mean()
            total = total + self._weight_task_loss("alignment", c.alignment_weight, losses["alignment"])
        next_cache = (ModelCache(next_layers, latents, offset + hidden.size(1), mean, scale, baseline,
                                 adapter_context, episodic_memory=next_memory,
                                 ngram_tail=next_ngram_tail, ngram_tail_valid=next_ngram_valid)
                      if use_cache else None)
        return BlueOutput(logits, class_logits, forecast, total if supervised else None,
                          losses, torch.stack(usage), next_cache, auxiliary_text_losses, valid.sum(),
                          last_hidden=hidden[:, -1:] if use_cache else None)

    @torch.inference_mode()
    def sample_flow(self, *, steps: int = 20, solver: str = "heun", guidance_scale: float = 1.0,
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
                                         guidance_scale=guidance_scale,
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
    def _filtered_probs(scores: torch.Tensor, *, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
        scores = scores.float()
        scores[:, [0, 1, 3]] = -torch.inf
        if not torch.finfo(torch.float32).tiny <= temperature <= torch.finfo(torch.float32).max:
            scores = scores.double()
        peak = scores.amax(-1, keepdim=True)
        if not torch.isfinite(peak).all():
            raise FloatingPointError("Les logits générables doivent avoir un maximum fini")
        scores = (scores - peak) / temperature
        if top_k:
            threshold = scores.topk(min(top_k, scores.size(-1)), dim=-1).values[:, -1:]
            scores = scores.masked_fill(scores < threshold, -torch.inf)
        ordered, order = scores.sort(descending=True)
        ordered_probs = ordered.softmax(-1)
        cumulative = ordered_probs.cumsum(-1)
        remove = cumulative - ordered_probs >= top_p
        ordered = ordered.masked_fill(remove, -torch.inf)
        scores = torch.full_like(scores, -torch.inf).scatter(-1, order, ordered)
        return scores.softmax(-1)

    def _draft_future_tokens(self, last_hidden: torch.Tensor, count: int) -> list[torch.Tensor]:
        drafts = []
        for head in self.future_heads[:count]:
            scores = self._lm_logits(head(last_hidden))[:, -1].float()
            scores[:, [0, 1, 3]] = -torch.inf
            drafts.append(scores.argmax(-1))
        return drafts

    def _draft_future_stochastic(self, last_hidden: torch.Tensor, count: int, *,
                                 temperature: float, top_k: int, top_p: float):
        drafts, q_probs = [], []
        for head in self.future_heads[:count]:
            probs = self._filtered_probs(
                self._lm_logits(head(last_hidden))[:, -1],
                temperature=temperature, top_k=top_k, top_p=top_p,
            )
            drafts.append(torch.multinomial(probs, 1).squeeze(-1))
            q_probs.append(probs)
        return drafts, q_probs

    @torch.inference_mode()
    def generate(self, input_ids, *, max_new_tokens=64, temperature=0.8, top_k=40,
                 top_p=0.95, eos_id=2, reasoning_steps: int | None = None,
                 speculative: bool = False, jacobi: bool = False,
                 jacobi_window: int = 4, **context):
        if input_ids.ndim != 2 or input_ids.size(1) < 1:
            raise ValueError("Le prompt doit avoir la forme [B, T] avec T > 0")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens doit être un entier positif ou nul")
        if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("temperature doit être >= 0 et top_p dans ]0, 1]")
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k doit être un entier positif ou nul")
        if type(speculative) is not bool:
            raise ValueError("speculative doit être booléen")
        if type(jacobi) is not bool:
            raise ValueError("jacobi doit être booléen")
        if type(jacobi_window) is not int or jacobi_window < 1:
            raise ValueError("jacobi_window doit être un entier >= 1")
        if jacobi and temperature != 0:
            raise ValueError("Le décodage par point fixe de Jacobi nécessite temperature=0")
        if input_ids.size(1) + max_new_tokens > self.config.max_seq_len:
            raise ValueError("Prompt + génération dépassent max_seq_len")
        if eos_id is not None and (type(eos_id) is not int or not 0 <= eos_id < self.config.vocab_size
                                   or eos_id in (0, 1, 3)):
            raise ValueError("eos_id doit être un entier générable du vocabulaire ou None")
        if "attention_mask" in context:
            mask = context["attention_mask"]
            if not mask[:, -1].bool().all():
                raise ValueError("La génération en batch nécessite du padding à gauche")
        was_training = self.training
        self.eval()
        try:
            if jacobi and input_ids.size(0) == 1 and max_new_tokens > 0:
                return self._generate_jacobi_greedy(
                    input_ids, max_new_tokens=max_new_tokens, eos_id=eos_id,
                    reasoning_steps=reasoning_steps, jacobi_window=jacobi_window, **context,
                )
            if (speculative and input_ids.size(0) == 1
                    and max_new_tokens > 0 and len(self.future_heads) > 0):
                if temperature == 0:
                    return self._generate_speculative_greedy(
                        input_ids, max_new_tokens=max_new_tokens, eos_id=eos_id,
                        reasoning_steps=reasoning_steps, **context,
                    )
                return self._generate_speculative_stochastic(
                    input_ids, max_new_tokens=max_new_tokens, temperature=temperature,
                    top_k=top_k, top_p=top_p, eos_id=eos_id,
                    reasoning_steps=reasoning_steps, **context,
                )
            result = input_ids
            finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
            cache = None
            for step in range(max_new_tokens):
                output = self(input_ids if step == 0 else result[:, -1:],
                              cache=cache, use_cache=True, reasoning_steps=reasoning_steps,
                              **(context if step == 0 else {}))
                cache = output.cache
                scores = output.logits[:, -1].float()
                # Les tokens de contrôle réservés ne sont pas des caractères générables.
                scores[:, [0, 1, 3]] = -torch.inf
                if temperature == 0:
                    next_token = scores.argmax(-1)
                else:
                    probs = self._filtered_probs(output.logits[:, -1],
                                                 temperature=temperature, top_k=top_k, top_p=top_p)
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

    def _generate_speculative_greedy(self, input_ids, *, max_new_tokens, eos_id, reasoning_steps, **context):
        """Décodage auto-spéculatif exact (greedy) exploitant les future_heads."""
        result = input_ids
        output = self(input_ids, cache=None, use_cache=True, reasoning_steps=reasoning_steps, **context)
        cache = output.cache
        scores = output.logits[:, -1].float()
        scores[:, [0, 1, 3]] = -torch.inf
        next_token = scores.argmax(-1)
        result = torch.cat((result, next_token[:, None]), -1)
        generated = 1
        if eos_id is not None and next_token.item() == eos_id:
            return result
        last_hidden = output.last_hidden
        while generated < max_new_tokens:
            remaining = max_new_tokens - generated
            n_draft = min(len(self.future_heads), remaining - 1)
            if n_draft == 0:
                step_out = self(next_token[:, None], cache=cache, use_cache=True, reasoning_steps=reasoning_steps)
                step_scores = step_out.logits[:, -1].float()
                step_scores[:, [0, 1, 3]] = -torch.inf
                next_token = step_scores.argmax(-1)
                result = torch.cat((result, next_token[:, None]), -1)
                break
            drafts = self._draft_future_tokens(last_hidden, n_draft)
            candidate = torch.stack([next_token, *drafts], dim=1)  # [1, 1 + n_draft]
            verify_out = self(candidate, cache=cache, use_cache=True, reasoning_steps=reasoning_steps)
            v_scores = verify_out.logits[0].float()  # [1 + n_draft, V]
            v_scores[:, [0, 1, 3]] = -torch.inf
            verified = v_scores.argmax(-1)  # verified[j] is true token after candidate[0..j]
            accepted = 0
            for j in range(n_draft):
                if verified[j].item() == drafts[j].item():
                    accepted += 1
                else:
                    break
            # Les tokens exacts produits sont verified[0..accepted].
            emitted = verified[:accepted + 1]
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
                # Tous les brouillons acceptés : le cache de verify_out est déjà synchronisé !
                cache = verify_out.cache
                last_hidden = verify_out.last_hidden
            else:
                # Resynchroniser exactement le cache sur les seuls tokens validés.
                sync_tokens = candidate[:, :accepted + 1]
                sync_out = self(sync_tokens, cache=cache, use_cache=True, reasoning_steps=reasoning_steps)
                cache = sync_out.cache
                last_hidden = sync_out.last_hidden
        return result

    def _generate_speculative_stochastic(self, input_ids, *, max_new_tokens, temperature,
                                         top_k, top_p, eos_id, reasoning_steps, **context):
        """Décodage auto-spéculatif stochastique exact (échantillonnage par rejet de Leviathan et al.)."""
        result = input_ids
        output = self(input_ids, cache=None, use_cache=True, reasoning_steps=reasoning_steps, **context)
        cache = output.cache
        p0 = self._filtered_probs(output.logits[:, -1], temperature=temperature, top_k=top_k, top_p=top_p)
        next_token = torch.multinomial(p0, 1).squeeze(-1)
        result = torch.cat((result, next_token[:, None]), -1)
        generated = 1
        if eos_id is not None and next_token.item() == eos_id:
            return result
        last_hidden = output.last_hidden
        while generated < max_new_tokens:
            remaining = max_new_tokens - generated
            n_draft = min(len(self.future_heads), remaining - 1)
            if n_draft == 0:
                step_out = self(next_token[:, None], cache=cache, use_cache=True, reasoning_steps=reasoning_steps)
                p_last = self._filtered_probs(step_out.logits[:, -1],
                                              temperature=temperature, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(p_last, 1).squeeze(-1)
                result = torch.cat((result, next_token[:, None]), -1)
                break
            drafts, q_probs = self._draft_future_stochastic(
                last_hidden, n_draft, temperature=temperature, top_k=top_k, top_p=top_p,
            )
            candidate = torch.stack([next_token, *drafts], dim=1)  # [1, 1 + n_draft]
            verify_out = self(candidate, cache=cache, use_cache=True, reasoning_steps=reasoning_steps)
            accepted = 0
            emitted_list = []
            rejected = False
            for j in range(n_draft):
                p_j = self._filtered_probs(verify_out.logits[:, j],
                                           temperature=temperature, top_k=top_k, top_p=top_p)
                tok = drafts[j].item()
                p_tok = p_j[0, tok].item()
                q_tok = q_probs[j][0, tok].item()
                u = torch.rand((), device=input_ids.device).item()
                if q_tok > 0.0 and u <= min(1.0, p_tok / q_tok):
                    emitted_list.append(drafts[j])
                    accepted += 1
                else:
                    residual = (p_j - q_probs[j]).clamp_min(0.0)
                    res_sum = residual.sum().item()
                    if res_sum > 0.0:
                        corrected = torch.multinomial(residual / res_sum, 1).squeeze(-1)
                    else:
                        corrected = torch.multinomial(p_j, 1).squeeze(-1)
                    emitted_list.append(corrected)
                    rejected = True
                    break
            if not rejected:
                p_bonus = self._filtered_probs(verify_out.logits[:, n_draft],
                                               temperature=temperature, top_k=top_k, top_p=top_p)
                emitted_list.append(torch.multinomial(p_bonus, 1).squeeze(-1))
            emitted = torch.cat(emitted_list, dim=0)
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
                cache = verify_out.cache
                last_hidden = verify_out.last_hidden
            else:
                sync_tokens = candidate[:, :accepted + 1]
                sync_out = self(sync_tokens, cache=cache, use_cache=True, reasoning_steps=reasoning_steps)
                cache = sync_out.cache
                last_hidden = sync_out.last_hidden
        return result

    def _generate_jacobi_greedy(self, input_ids, *, max_new_tokens, eos_id,
                                reasoning_steps, jacobi_window, **context):
        """Décodage parallèle par itération de point fixe de Jacobi (lookahead exact sans future_heads requis)."""
        result = input_ids
        output = self(input_ids, cache=None, use_cache=True, reasoning_steps=reasoning_steps, **context)
        cache = output.cache
        scores = output.logits[:, -1].float()
        scores[:, [0, 1, 3]] = -torch.inf
        next_token = scores.argmax(-1)
        result = torch.cat((result, next_token[:, None]), -1)
        generated = 1
        if eos_id is not None and next_token.item() == eos_id:
            return result
        last_hidden = output.last_hidden
        carry: list[torch.Tensor] = []
        while generated < max_new_tokens:
            remaining = max_new_tokens - generated
            n_draft = min(jacobi_window, remaining - 1)
            if n_draft == 0:
                step_out = self(next_token[:, None], cache=cache, use_cache=True, reasoning_steps=reasoning_steps)
                step_scores = step_out.logits[:, -1].float()
                step_scores[:, [0, 1, 3]] = -torch.inf
                next_token = step_scores.argmax(-1)
                result = torch.cat((result, next_token[:, None]), -1)
                break
            drafts = list(carry[:n_draft])
            if len(drafts) < n_draft and len(self.future_heads) > 0 and last_hidden is not None:
                fh_drafts = self._draft_future_tokens(last_hidden, min(n_draft, len(self.future_heads)))
                for j in range(len(drafts), min(n_draft, len(fh_drafts))):
                    drafts.append(fh_drafts[j])
            while len(drafts) < n_draft:
                fallback = drafts[-1] if drafts else next_token
                drafts.append(fallback)
            candidate = torch.stack([next_token, *drafts], dim=1)
            verify_out = self(candidate, cache=cache, use_cache=True, reasoning_steps=reasoning_steps)
            v_scores = verify_out.logits[0].float()
            v_scores[:, [0, 1, 3]] = -torch.inf
            verified = v_scores.argmax(-1)
            accepted = 0
            for j in range(n_draft):
                if verified[j].item() == drafts[j].item():
                    accepted += 1
                else:
                    break
            if accepted == 0:
                # Raffinement de point fixe de Jacobi sur la trajectoire mise à jour :
                # drafts[0] = verified[0] est désormais exact, garantissant accepted >= 1.
                drafts = [verified[j].unsqueeze(0) for j in range(n_draft)]
                candidate = torch.stack([next_token, *drafts], dim=1)
                verify_out = self(candidate, cache=cache, use_cache=True, reasoning_steps=reasoning_steps)
                v_scores = verify_out.logits[0].float()
                v_scores[:, [0, 1, 3]] = -torch.inf
                verified = v_scores.argmax(-1)
                accepted = 0
                for j in range(n_draft):
                    if verified[j].item() == drafts[j].item():
                        accepted += 1
                    else:
                        break
            emitted = verified[:accepted + 1]
            carry = [verified[j].unsqueeze(0) for j in range(accepted + 1, n_draft + 1)]
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
                cache = verify_out.cache
                last_hidden = verify_out.last_hidden
            else:
                sync_tokens = candidate[:, :accepted + 1]
                sync_out = self(sync_tokens, cache=cache, use_cache=True, reasoning_steps=reasoning_steps)
                cache = sync_out.cache
                last_hidden = sync_out.last_hidden
        return result
