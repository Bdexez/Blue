"""Entraînement conjoint, AMP, accumulation, DDP et sauvegarde atomique."""

from contextlib import nullcontext
import itertools
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
import warnings

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, Sampler

from .config import BlueConfig
from .augmentation import SeriesAffineAugmentation, SeriesAugmentation
from .data import JointDataset, collate_records, synthetic_batch
from .model import Blue
from .modalities import INPUT_MODALITIES, balanced_modalities, has_observed_targets, keep_modalities
from .modality_batch_sampler import ModalityBatchSampler
from .observed_objective import observed_objective, target_counts
from .routing_metrics import summarize_expert_usage
from .evaluation import evaluate
from .ema import ModelEMA
from .gradient_clipping import clip_grad_norm
from .lora import DoRALinear, inject_dora
from .modules import RotaryEmbedding
from .optimizer import build_optimizer
from .quantize import quantize_model_int4, quantize_model_int8


class SkippedBatches(Sampler):
    """Les lots d'un échantillonneur de lots, sans les ``skip`` premiers."""

    def __init__(self, batches, skip):
        self.batches, self.skip = batches, skip

    def __len__(self):
        return max(len(self.batches) - self.skip, 0)

    def __iter__(self):
        return itertools.islice(iter(self.batches), self.skip, None)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state(device):
    return dict(torch=torch.get_rng_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def collect_rng_states(device, world, rank):
    """Tous les rangs participent ; seul le rang 0 reçoit la liste ordonnée."""
    state = capture_rng_state(device)
    if world == 1:
        return None
    states = [None] * world if rank == 0 else None
    dist.gather_object(state, states, dst=0)
    return states


def save_checkpoint(path, model, *, optimizer=None, scaler=None, step=0, trainer=None, rank_rng=None, ema=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1, "config": model.config.to_dict(), "model": model.state_dict(),
        "step": step, "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "trainer": trainer or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if rank_rng is not None:
        payload["rank_rng"] = rank_rng
    if ema is not None:
        payload["ema"] = ema.state_dict()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path, device="cpu", *, mmap=True, overrides=None):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    if payload.get("format_version") != 1:
        raise ValueError("Version de checkpoint inconnue")
    original_config = BlueConfig.from_dict(payload["config"])
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("Les changements de configuration doivent être un objet JSON")
    config = BlueConfig.from_dict(original_config.to_dict() | (overrides or {}))
    with torch.device("meta"):
        model = Blue(config)
    state = dict(payload["model"])
    statistics_key = "temporal.statistics_proj.weight"
    if config.temporal_statistics and not original_config.temporal_statistics:
        if statistics_key in state:
            raise ValueError("La projection statistique contredit la configuration du checkpoint")
        # Ce seul nouveau paramètre est initialisé à zéro ; toute autre
        # incompatibilité du checkpoint reste une erreur de chargement strict.
        state[statistics_key] = torch.zeros(model.temporal.statistics_proj.weight.shape,
                                            dtype=state["embedding.weight"].dtype, device="cpu")
    elif original_config.temporal_statistics and not config.temporal_statistics:
        state.pop(statistics_key)
    elif (config.temporal_statistics
          and config.temporal_statistics_relative != original_config.temporal_statistics_relative):
        # Même forme, autre sens des caractéristiques : ne pas réinterpréter
        # les coefficients appris sur le niveau et l'échelle absolus.
        previous_statistics = state.pop(statistics_key)
        state[statistics_key] = torch.zeros_like(previous_statistics)
    adapter_keys = ("forecast_adapter.proj.weight", "forecast_adapter.proj.bias")
    missing_keys = ("forecast_adapter.missing_proj.weight", "forecast_adapter.missing_proj.bias")
    if (config.forecast_adapter_context != original_config.forecast_adapter_context
            or config.forecast_adapter_relative != original_config.forecast_adapter_relative
            or config.forecast_adapter_missing_gate != original_config.forecast_adapter_missing_gate
            or config.forecast_adapter_separate_missing != original_config.forecast_adapter_separate_missing):
        same_length_branch = (config.forecast_adapter_context == original_config.forecast_adapter_context > 0
                              and config.forecast_adapter_relative == original_config.forecast_adapter_relative
                              and (not original_config.forecast_adapter_missing_gate
                                   or original_config.forecast_adapter_separate_missing)
                              and (not config.forecast_adapter_missing_gate or config.forecast_adapter_separate_missing))
        old_keys = (*adapter_keys, *(missing_keys if original_config.forecast_adapter_separate_missing else ()))
        new_keys = (*adapter_keys, *(missing_keys if config.forecast_adapter_separate_missing else ()))
        if original_config.forecast_adapter_context:
            if not original_config.forecast_adapter_separate_missing and any(key in state for key in missing_keys):
                raise ValueError("L'adaptateur contredit la configuration du checkpoint")
            for key in old_keys:
                if not same_length_branch or key in missing_keys:
                    state.pop(key)  # Refuser aussi un ancien adaptateur incomplet.
        elif any(key in state for key in (*adapter_keys, *missing_keys)):
            raise ValueError("L'adaptateur contredit la configuration du checkpoint")
        if config.forecast_adapter_context:
            expected = model.forecast_adapter.state_dict()
            for key in new_keys:
                if not same_length_branch or key in missing_keys:
                    relative = key.removeprefix("forecast_adapter.")
                    state[key] = torch.zeros(expected[relative].shape, dtype=state["embedding.weight"].dtype, device="cpu")
    trainer_meta = payload.get("trainer") or {}
    dora_meta = trainer_meta.get("dora")
    int8_meta = trainer_meta.get("quantized_int8")
    int4_meta = trainer_meta.get("quantized_int4")
    if dora_meta is not None:
        inject_dora(model, rank=int(dora_meta["rank"]), alpha=float(dora_meta["alpha"]),
                    use_dora=bool(dora_meta.get("use_dora", True)))
    if int8_meta is not None:
        quantize_model_int8(model, min_numel=int(int8_meta.get("min_numel", 1024)))
    if int4_meta is not None:
        quantize_model_int4(model, min_numel=int(int4_meta.get("min_numel", 1024)),
                            group_size=int(int4_meta.get("group_size", 32)))
    model.load_state_dict(state, strict=True, assign=True)
    # assign remplace les Parameters ; rétablir explicitement le poids partagé et requires_grad.
    model.lm_head.weight = model.embedding.weight
    if dora_meta is not None:
        for param in model.parameters():
            param.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, DoRALinear):
                module.lora_a.requires_grad_(True)
                module.lora_b.requires_grad_(True)
                if module.magnitude is not None:
                    module.magnitude.requires_grad_(True)
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            module.inv_freq = module.frequencies(device="cpu")
    return model.to(device), payload


def learning_rate(step, total_steps, peak, warmup, *, schedule="cosine", wsd_decay_fraction=0.2):
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    if schedule == "wsd":
        post_warmup = max(total_steps - warmup, 1)
        decay_steps = int(round(post_warmup * wsd_decay_fraction))
        stable_end = max(warmup, total_steps - decay_steps)
        if step < stable_end or decay_steps <= 0:
            return peak
        progress = (step - stable_end + 1) / max(decay_steps, 1)
        return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))
    progress = (step - warmup) / max(total_steps - warmup - 1, 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1))))


def drop_modalities(batch, probability, *, prediction_steps=1):
    """Apprendre aussi avec des modalités absentes, sans supprimer toutes les entrées."""
    names = [name for name in INPUT_MODALITIES if name in batch]
    kept = [name for name in names if random.random() >= probability]
    if not kept:
        kept = [random.choice(names)]
    result = keep_modalities(batch, kept)
    # Les champs présents peuvent ne contenir que des cibles ignorées.
    if not has_observed_targets(result, prediction_steps=prediction_steps):
        return batch
    return result


def step_optimizer(model, optimizer, scaler, grad_clip):
    """Laisser GradScaler ignorer un overflow, au lieu d'arrêter l'entraînement."""
    scaler.unscale_(optimizer)
    norm = clip_grad_norm(model.parameters(), grad_clip, allow_nonfinite=scaler.is_enabled())
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.is_enabled() and scaler.get_scale() < previous_scale
    return float(norm) if torch.isfinite(norm) else None, skipped


class DistributedTrainingModel(torch.nn.Module):
    """Ne présenter à DDP que la perte participant au backward."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *, local_counts=None, total_counts=None, accumulation=1, world=1, **batch):
        output = self.model(**batch)
        loss = (output.loss if total_counts is None else
                observed_objective(output, local_counts, total_counts, self.model.config,
                                   world=world, accumulation=accumulation,
                                   weight_task=self.model._weight_task_loss))
        return (loss, {name: value.detach() for name, value in output.losses.items()},
                output.expert_usage.detach(), output.routed_tokens.detach())


def train(args, config):
    if min(args.steps, args.batch_size, args.accumulation, args.save_every, args.log_every, args.eval_every) < 1:
        raise ValueError("Les nombres d'étapes, de lots et les fréquences doivent être > 0")
    for name in ("lr", "grad_clip", "weight_decay", "min_delta"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name in ("lr", "grad_clip") and value == 0):
            raise ValueError(f"Valeur invalide pour {name}")
    if args.warmup < 0 or args.patience < 0 or args.workers < 0:
        raise ValueError("warmup, patience et workers doivent être >= 0")
    if not 0 <= args.modality_dropout < 1:
        raise ValueError("modality-dropout doit appartenir à [0, 1[")
    if args.patience and not args.val_data:
        raise ValueError("L'arrêt anticipé nécessite un jeu indépendant fourni par --val-data")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(args.device)
    if args.amp == "fp16" and device.type != "cuda":
        raise ValueError("fp16 nécessite CUDA ; utiliser bf16 ou none sur CPU")
    if world > 1:
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    seed_everything(args.seed)
    initialization = None
    if args.resume or args.init_from:
        overrides = json.loads(Path(args.model_overrides).read_text(encoding="utf-8")) if args.model_overrides else None
        model, source = load_checkpoint(args.resume or args.init_from, device, overrides=overrides)
        config = model.config
        if args.resume:
            if "optimizer" not in source:
                raise ValueError("Une reprise nécessite le checkpoint complet latest.pt avec son optimiseur")
            payload, previous, start = source, source.get("trainer", {}), source["step"]
        else:
            initialization = dict(checkpoint=str(Path(args.init_from).resolve()), step=source["step"],
                                  seed=source.get("trainer", {}).get("seed"), overrides=overrides or {})
            payload, previous, start = {}, {}, 0
        del source
    else:
        model, payload, previous, start = Blue(config).to(device), {}, {}, 0
    def resumed_option(name, default, *, valid=None, invalid_message=None, convert=None):
        """Option demandée, sinon celle du checkpoint ; une reprise refuse toute différence."""
        saved = previous.get(name, default)
        requested = getattr(args, name, None)
        value = saved if requested is None else (convert(requested) if convert else requested)
        if valid is not None and not valid(value):
            raise ValueError(invalid_message)
        if args.resume and value != saved:
            raise ValueError(f"Reprise : {name} diffère du checkpoint ({saved!r})")
        setattr(args, name, value)

    resumed_option("modality_policy", "dropout")
    resumed_option("monitor", "loss")
    resumed_option("optimizer", "adamw", valid=lambda value: value in ("adamw", "muon"),
                   invalid_message="optimizer doit être adamw ou muon")
    resumed_option("lr_schedule", "cosine", valid=lambda value: value in ("cosine", "wsd"),
                   invalid_message="lr_schedule doit être cosine ou wsd")
    resumed_option("wsd_decay_fraction", 0.2, convert=float,
                   valid=lambda value: math.isfinite(value) and 0.0 <= value <= 1.0,
                   invalid_message="wsd_decay_fraction doit être fini et appartenir à [0, 1]")
    resumed_option("ema_decay", 0.0, valid=lambda value: math.isfinite(value) and 0.0 <= value < 1.0,
                   invalid_message="ema_decay doit être fini et appartenir à [0, 1[")
    previous_dora = previous.get("dora")
    req_dora_rank = getattr(args, "dora_rank", None)
    req_dora_alpha = getattr(args, "dora_alpha", None)
    req_dora_mode = getattr(args, "dora_mode", None)
    if args.resume:
        dora_spec = previous_dora
        if req_dora_rank is not None:
            if dora_spec is None or req_dora_rank != dora_spec["rank"]:
                raise ValueError(f"Reprise : dora_rank diffère du checkpoint ({dora_spec!r})")
        if req_dora_alpha is not None:
            if dora_spec is None or float(req_dora_alpha) != float(dora_spec["alpha"]):
                raise ValueError(f"Reprise : dora_alpha diffère du checkpoint ({dora_spec!r})")
        if req_dora_mode is not None:
            if dora_spec is None or (req_dora_mode == "dora") != bool(dora_spec["use_dora"]):
                raise ValueError(f"Reprise : dora_mode diffère du checkpoint ({dora_spec!r})")
    elif req_dora_rank is not None:
        if any(isinstance(m, DoRALinear) for m in model.modules()):
            raise ValueError("Le checkpoint source contient déjà un adaptateur DoRA ; fusionner d'abord avec export --merge-dora")
        if type(req_dora_rank) is not int or req_dora_rank < 1:
            raise ValueError("dora_rank doit être un entier strictement positif")
        alpha = float(2 * req_dora_rank) if req_dora_alpha is None else float(req_dora_alpha)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("dora_alpha doit être fini et strictement positif")
        mode = "dora" if req_dora_mode is None else req_dora_mode
        if mode not in ("dora", "lora"):
            raise ValueError("dora_mode doit être dora ou lora")
        use_dora = mode == "dora"
        dora_stats = inject_dora(model, rank=req_dora_rank, alpha=alpha, use_dora=use_dora)
        dora_spec = {"rank": req_dora_rank, "alpha": alpha, "use_dora": use_dora, **dora_stats}
    else:
        if req_dora_alpha is not None or req_dora_mode is not None:
            raise ValueError("dora_alpha et dora_mode nécessitent --dora-rank")
        dora_spec = None
    maximize = args.monitor == "accuracy"
    resumed_option("loss_normalization", "batch", valid=lambda value: value in ("batch", "observed"),
                   invalid_message="loss_normalization doit être batch ou observed")
    previous_mixed = previous.get("mixed_modalities", False)
    requested_mixed = getattr(args, "mixed_modalities", None)
    if args.resume and requested_mixed is not None and requested_mixed != previous_mixed:
        raise ValueError(f"Reprise : mixed_modalities diffère du checkpoint ({previous_mixed!r})")
    args.mixed_modalities = previous_mixed if requested_mixed is None else requested_mixed
    previous_duplicates = previous.get("alignment_duplicate_positives", False)
    requested_duplicates = getattr(args, "alignment_duplicate_positives", None)
    if args.resume:
        if requested_duplicates is not None and requested_duplicates != previous_duplicates:
            raise ValueError(f"Reprise : alignment_duplicate_positives diffère du checkpoint ({previous_duplicates!r})")
        args.alignment_duplicate_positives = previous_duplicates
    else:
        args.alignment_duplicate_positives = (args.mixed_modalities if requested_duplicates is None
                                               else requested_duplicates)
    previous_cross_views = previous.get("alignment_cross_rank_views", False)
    requested_cross_views = getattr(args, "alignment_cross_rank_views", None)
    if args.resume:
        if requested_cross_views is not None and requested_cross_views != previous_cross_views:
            raise ValueError(f"Reprise : alignment_cross_rank_views diffère du checkpoint ({previous_cross_views!r})")
        args.alignment_cross_rank_views = previous_cross_views
    else:
        args.alignment_cross_rank_views = (args.mixed_modalities and args.alignment_duplicate_positives
                                           if requested_cross_views is None else requested_cross_views)
    if args.alignment_cross_rank_views and not args.alignment_duplicate_positives:
        raise ValueError("alignment_cross_rank_views nécessite alignment_duplicate_positives")
    previous_cross_classes = previous.get("alignment_cross_class_views", False)
    requested_cross_classes = getattr(args, "alignment_cross_class_views", None)
    if args.resume and requested_cross_classes is not None and requested_cross_classes != previous_cross_classes:
        raise ValueError(f"Reprise : alignment_cross_class_views diffère du checkpoint ({previous_cross_classes!r})")
    args.alignment_cross_class_views = (previous_cross_classes if requested_cross_classes is None
                                        else requested_cross_classes)
    if args.alignment_cross_class_views and config.alignment_mode != "class":
        raise ValueError("alignment_cross_class_views nécessite le mode class")
    previous_shared_batches = previous.get("shared_modality_batches", False)
    requested_shared_batches = getattr(args, "shared_modality_batches", None)
    if args.resume:
        if requested_shared_batches is not None and requested_shared_batches != previous_shared_batches:
            raise ValueError(f"Reprise : shared_modality_batches diffère du checkpoint ({previous_shared_batches!r})")
        args.shared_modality_batches = previous_shared_batches
    else:
        args.shared_modality_batches = False if requested_shared_batches is None else requested_shared_batches
    if args.shared_modality_batches and (not args.mixed_modalities or not args.data):
        raise ValueError("shared_modality_batches nécessite mixed_modalities et un jeu data")
    previous_shared_interval = previous.get("shared_modality_batch_interval", 1)
    requested_shared_interval = getattr(args, "shared_modality_batch_interval", None)
    if args.resume:
        if requested_shared_interval is not None and requested_shared_interval != previous_shared_interval:
            raise ValueError(f"Reprise : shared_modality_batch_interval diffère du checkpoint ({previous_shared_interval!r})")
        args.shared_modality_batch_interval = previous_shared_interval
    else:
        args.shared_modality_batch_interval = 1 if requested_shared_interval is None else requested_shared_interval
    if type(args.shared_modality_batch_interval) is not int or args.shared_modality_batch_interval < 1:
        raise ValueError("shared_modality_batch_interval doit être un entier >= 1")
    if not args.shared_modality_batches and args.shared_modality_batch_interval != 1:
        raise ValueError("shared_modality_batch_interval nécessite shared_modality_batches")
    augmentation_options = {}
    previous_augmentation = previous.get("series_augmentation", {})
    for name, default in SeriesAugmentation().to_dict().items():
        saved = previous_augmentation.get(name, default)
        option = f"series_{name}"
        value = getattr(args, option)
        value = saved if value is None else value
        if args.resume and value != saved:
            raise ValueError(f"Reprise : {option} diffère du checkpoint ({saved!r})")
        augmentation_options[name] = value
    augment_series = SeriesAugmentation(**augmentation_options)
    affine_options = {}
    previous_affine = previous.get("series_affine_augmentation", {})
    for name, default in SeriesAffineAugmentation().to_dict().items():
        saved = previous_affine.get(name, default)
        option = f"series_affine_{name}"
        value = getattr(args, option, None)
        value = saved if value is None else value
        if args.resume and value != saved:
            raise ValueError(f"Reprise : {option} diffère du checkpoint ({saved!r})")
        affine_options[name] = value
    augment_affine = SeriesAffineAugmentation(**affine_options)
    args.shuffle = (previous.get("shuffle", False) if args.resume else True) if args.shuffle is None else args.shuffle
    keys = ("steps", "batch_size", "accumulation", "seed", "lr", "warmup", "weight_decay",
            "amp", "data", "modality_dropout", "modality_policy", "shuffle", "val_data", "eval_every", "patience", "min_delta", "monitor", "loss_normalization")
    trainer = {key: getattr(args, key) for key in keys}
    trainer["optimizer"] = args.optimizer
    trainer["lr_schedule"] = args.lr_schedule
    trainer["wsd_decay_fraction"] = args.wsd_decay_fraction
    trainer["ema_decay"] = args.ema_decay
    if dora_spec is not None:
        trainer["dora"] = dora_spec
    trainer["mixed_modalities"] = args.mixed_modalities
    trainer["alignment_duplicate_positives"] = args.alignment_duplicate_positives
    trainer["alignment_cross_rank_views"] = args.alignment_cross_rank_views
    trainer["alignment_cross_class_views"] = args.alignment_cross_class_views
    trainer["shared_modality_batches"] = args.shared_modality_batches
    trainer["shared_modality_batch_interval"] = args.shared_modality_batch_interval
    trainer["initialization"] = previous.get("initialization", initialization)
    trainer["series_augmentation"] = augment_series.to_dict()
    trainer["series_affine_augmentation"] = augment_affine.to_dict()
    trainer["world_size"] = world
    dataset = (JointDataset(args.data, config, allow_mixed=args.mixed_modalities,
                            include_alignment_ids=args.alignment_duplicate_positives) if args.data else None)
    validation = JointDataset(args.val_data, config, allow_mixed=args.mixed_modalities) if args.val_data else None
    trainer["data_fingerprint"] = dataset.fingerprint if dataset is not None else None
    trainer["validation_fingerprint"] = validation.fingerprint if validation is not None else None
    trainer["data_images_fingerprint"] = dataset.image_fingerprint if dataset is not None else None
    trainer["validation_images_fingerprint"] = validation.image_fingerprint if validation is not None else None
    if args.resume:
        for name, current in trainer.items():
            if name in previous and previous[name] != current:
                raise ValueError(f"Reprise : {name} diffère du checkpoint ({previous[name]!r})")
            if (name in ("data_images_fingerprint", "validation_images_fingerprint") and name not in previous
                    and current is not None and rank == 0):
                warnings.warn(f"Checkpoint ancien : {name} absent, le contenu initial des images est invérifiable.",
                              RuntimeWarning)
    initial_best = float("-inf") if maximize else float("inf")
    state = dict(previous.get("validation_state", {"best_loss": initial_best, "bad_evaluations": 0, "best_step": None}))
    trainer["validation_state"] = state
    end_step = min(args.steps, args.stop_after) if args.stop_after is not None else args.steps
    if end_step < start or end_step < 1:
        raise ValueError("stop-after doit être >= à l'étape reprise et > 0")
    optimizer = build_optimizer(model, name=args.optimizer, lr=args.lr, weight_decay=args.weight_decay)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
    if "optimizer" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except ValueError as exc:
            raise ValueError("L'état de l'optimiseur ne correspond pas à ses groupes de paramètres "
                             "(checkpoint Muon antérieur à la restriction de Muon aux poids linéaires ?) ; "
                             "reprendre avec --init-from pour repartir d'un optimiseur neuf") from exc
    if ema is not None and "ema" in payload:
        ema.load_state_dict(payload["ema"])
    if "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    if args.resume:
        if world == 1:
            torch.set_rng_state(payload["torch_rng"])
            random.setstate(payload["python_rng"])
            if device.type == "cuda" and payload["cuda_rng"]:
                torch.cuda.set_rng_state_all(payload["cuda_rng"])
        elif "rank_rng" in payload:
            states = payload["rank_rng"]
            if not isinstance(states, list) or len(states) != world:
                raise ValueError("États aléatoires distribués incompatibles avec le nombre de processus")
            state_for_rank = states[rank]
            torch.set_rng_state(state_for_rank["torch"])
            random.setstate(state_for_rank["python"])
            if device.type == "cuda":
                if state_for_rank["cuda"] is None:
                    if rank == 0:
                        warnings.warn("Checkpoint sans état aléatoire CUDA : la reprise ne sera pas exacte.", RuntimeWarning)
                else:
                    torch.cuda.set_rng_state(state_for_rank["cuda"], device)
            elif state_for_rank["cuda"] is not None and rank == 0:
                warnings.warn("Checkpoint CUDA repris sur CPU : la reprise ne sera pas exacte.", RuntimeWarning)
        elif rank == 0:
            warnings.warn("Checkpoint distribué ancien : les états aléatoires par rang ne sont pas disponibles ; "
                          "la reprise ne sera pas exacte.", RuntimeWarning)
    del payload
    wrapped = (DistributedDataParallel(DistributedTrainingModel(model), device_ids=[local_rank] if device.type == "cuda" else None,
                                      find_unused_parameters=True) if world > 1 else model)
    sampler = loader = iterator = None
    epoch = 0
    if dataset is not None:
        if args.mixed_modalities:
            sampler = ModalityBatchSampler(dataset, args.batch_size, num_replicas=world, rank=rank,
                                           seed=args.seed, shuffle=args.shuffle,
                                           shared_across_ranks=args.shared_modality_batches,
                                           shared_interval=args.shared_modality_batch_interval)
            loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_records,
                                num_workers=args.workers, generator=torch.Generator().manual_seed(args.seed))
        else:
            sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=args.seed, shuffle=args.shuffle)
            loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_records,
                                sampler=sampler, num_workers=args.workers,
                                generator=torch.Generator().manual_seed(args.seed))
        epoch, skip = divmod(start * args.accumulation, len(loader))
        sampler.set_epoch(epoch)
        if skip:
            # Reprendre au milieu d'une époque sans lire ni décoder les lots déjà vus :
            # les indices sont ceux qu'aurait produits le chargeur complet.
            batches = sampler if args.mixed_modalities else BatchSampler(sampler, args.batch_size, drop_last=False)
            iterator = iter(DataLoader(dataset, batch_sampler=SkippedBatches(batches, skip),
                                       collate_fn=collate_records, num_workers=args.workers,
                                       generator=torch.Generator().manual_seed(args.seed)))
        else:
            iterator = iter(loader)
    output_dir = Path(args.output)
    val = None
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        config.save(output_dir / "config.json")
        if validation is None:
            val = [synthetic_batch(config, batch_size=args.batch_size, seed=1000000 + args.seed)]
        elif args.mixed_modalities:
            validation_sampler = ModalityBatchSampler(validation, args.batch_size, shuffle=False)
            val = DataLoader(validation, batch_sampler=validation_sampler, collate_fn=collate_records,
                             num_workers=args.workers, generator=torch.Generator().manual_seed(args.seed + 1))
        else:
            val = DataLoader(validation, batch_size=args.batch_size, collate_fn=collate_records,
                             num_workers=args.workers, generator=torch.Generator().manual_seed(args.seed + 1))
    autocast = (lambda: nullcontext()) if args.amp == "none" else (
        lambda: torch.autocast(device.type, dtype=torch.bfloat16 if args.amp == "bf16" else torch.float16)
    )
    model.train()
    started, last_step, last_evaluated = time.monotonic(), start, None
    stopped_early = bool(args.patience and state["bad_evaluations"] >= args.patience)
    if stopped_early:
        end_step = start
    report = None
    for step in range(start, end_step):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(
                step, args.steps, args.lr, args.warmup,
                schedule=args.lr_schedule, wsd_decay_fraction=args.wsd_decay_fraction,
            ) * group["lr_scale"]
        metrics, modality_batches, series_statistics = {}, {}, {}
        expert_totals = torch.zeros(config.n_layers, config.n_experts, device=device, dtype=torch.float64)
        expert_tokens = torch.zeros((), device=device, dtype=torch.long)
        def prepare_microbatch(micro):
            nonlocal epoch, iterator
            if iterator is None:
                batch = synthetic_batch(config, batch_size=args.batch_size,
                                        seed=args.seed + (step * args.accumulation + micro) * world + rank,
                                        device=device)
            else:
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    sampler.set_epoch(epoch)
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = {key: value.to(device) for key, value in batch.items()}
            index = (step * args.accumulation + micro) * world + rank
            batch = augment_series(batch, index=index, seed=args.seed)
            supervised = has_observed_targets(batch, prediction_steps=config.prediction_steps)
            if not supervised and args.loss_normalization == "batch":
                raise ValueError("Le batch ne contient aucune cible supervisée observée")
            if supervised and args.modality_policy == "balanced":
                batch = balanced_modalities(batch, index, args.seed, prediction_steps=config.prediction_steps)
            elif supervised and args.modality_dropout:
                batch = drop_modalities(batch, args.modality_dropout, prediction_steps=config.prediction_steps)
            # Le futur ne doit changer d'unités que si l'historique permettant
            # de connaître ces unités est encore présent dans les entrées.
            original_series = batch.get("series")
            batch = augment_affine(batch, index=index, seed=args.seed + 2000000)
            signature = "+".join(name for name in INPUT_MODALITIES if name in batch)
            modality_batches[signature] = modality_batches.get(signature, 0) + 1
            if rank == 0 and "series" in batch:
                series = batch["series"]
                valid = batch.get("series_mask", torch.ones(series.shape[:2], device=series.device, dtype=torch.bool)).bool()
                valid_steps = valid.sum().item()
                statistics = dict(examples=series.size(0), valid_steps=valid_steps,
                                  possible_values=valid_steps * series.size(-1),
                                  observed_values=(torch.isfinite(series) & valid.unsqueeze(-1)).sum().item(),
                                  affine_changed_examples=(torch.isfinite(original_series) & valid.unsqueeze(-1)
                                                           & (series != original_series)).any(dim=(1, 2)).sum().item())
                for name, value in statistics.items():
                    series_statistics[name] = series_statistics.get(name, 0) + value
            if args.alignment_cross_rank_views:
                batch["alignment_cross_rank_views"] = True
            if args.alignment_cross_class_views:
                batch["alignment_cross_class_views"] = True
            return batch

        batches = (prepare_microbatch(micro) for micro in range(args.accumulation))
        total_counts = None
        if args.loss_normalization == "observed":
            batches = list(batches)
            local_counts = [target_counts(batch, config.prediction_steps) for batch in batches]
            total_counts = torch.stack(local_counts).sum(0)
            if world > 1:
                dist.all_reduce(total_counts, op=dist.ReduceOp.SUM)
            if not total_counts.any().item():
                raise ValueError("L'étape ne contient aucune cible supervisée observée")
        else:
            local_counts = [None] * args.accumulation
        for micro, (batch, counts) in enumerate(zip(batches, local_counts)):
            sync = wrapped.no_sync() if world > 1 and micro + 1 < args.accumulation else nullcontext()
            with sync, autocast():
                if world > 1 and total_counts is not None:
                    result = wrapped(local_counts=counts, total_counts=total_counts,
                                     accumulation=args.accumulation, world=world, **batch)
                else:
                    result = wrapped(**batch)
                if world > 1:
                    batch_loss, batch_losses, batch_usage, batch_routed = result
                else:
                    batch_loss, batch_losses, batch_usage, batch_routed = (
                        result.loss, result.losses, result.expert_usage, result.routed_tokens)
                if batch_loss is None and total_counts is None:
                    raise ValueError("Aucune cible d'entraînement dans le batch")
                if total_counts is None:
                    loss = batch_loss / args.accumulation
                elif world > 1:
                    loss = batch_loss
                else:
                    loss = observed_objective(result, counts, total_counts, config,
                                              world=world, accumulation=args.accumulation,
                                              weight_task=model._weight_task_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Perte non finie à l'étape {step + 1}")
            scaler.scale(loss).backward()
            logged_loss = batch_loss.item() / args.accumulation if total_counts is None else loss.item()
            metrics["loss"] = metrics.get("loss", 0) + logged_loss
            for key, value in batch_losses.items():
                metrics[key] = metrics.get(key, 0) + value.detach().item() / args.accumulation
            expert_totals += batch_usage.detach().double() * batch_routed
            expert_tokens += batch_routed
        norm, skipped = step_optimizer(model, optimizer, scaler, args.grad_clip)
        if ema is not None and not skipped:
            ema.update(model)
        last_step = step + 1
        routing = torch.cat((expert_totals.flatten(), expert_tokens.reshape(1).double()))
        if world > 1:
            if config.router_bias_update_rate > 0:
                dist.all_reduce(routing, op=dist.ReduceOp.SUM)
            else:
                dist.reduce(routing, dst=0, op=dist.ReduceOp.SUM)
        if config.router_bias_update_rate > 0 and not skipped and routing[-1].item() > 0:
            step_usage = routing[:-1].view(config.n_layers, config.n_experts) / routing[-1]
            step_usage = step_usage / step_usage.sum(-1, keepdim=True).clamp_min(1e-12)
            model.update_router_biases(step_usage)
        if rank == 0:
            routing_summary = summarize_expert_usage(routing[:-1].view(config.n_layers, config.n_experts),
                                                      int(routing[-1].item()))
            row = dict(step=last_step, **metrics, grad_norm=norm, optimizer_step_skipped=skipped,
                       lr=optimizer.param_groups[0]["lr"], elapsed_seconds=round(time.monotonic() - started, 3),
                       **routing_summary, modality_batches=modality_batches,
                       series_statistics=series_statistics)
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if step == start or last_step % args.log_every == 0 or last_step == end_step:
                print(json.dumps({key: value for key, value in row.items() if key != "expert_usage"}), flush=True)
        # Ne pas ajouter une validation de contrôle à un simple --stop-after :
        # sa reprise doit conserver le même calendrier d'arrêt anticipé.
        validation_event = last_step % args.eval_every == 0 or last_step == args.steps
        if validation_event:
            improved = False
            if rank == 0:
                eval_context = ema.apply_to(model) if ema is not None else nullcontext()
                with eval_context:
                    report = evaluate(model, val, args.amp)
                observed = sum(report.get(name, 0) for name in
                               ("text_tokens", "multi_token_targets", "classified_examples", "forecast_observations"))
                if not observed:
                    raise ValueError("Le jeu de validation ne contient aucune cible supervisée observée")
                score = report.get(args.monitor)
                if score is None or not math.isfinite(score):
                    raise ValueError(f"La mesure de validation {args.monitor} est absente ou non finie")
                last_evaluated = last_step
                reference = state.get("reference_loss", state["best_loss"])
                improved = score > state["best_loss"] if maximize else score < state["best_loss"]
                significant = score > reference + args.min_delta if maximize else score < reference - args.min_delta
                if improved:
                    state.update(best_loss=score, best_step=last_step)
                if significant:
                    state.update(reference_loss=score, bad_evaluations=0)
                else:
                    state["bad_evaluations"] += 1
                stopped_early = bool(args.patience and state["bad_evaluations"] >= args.patience)
                with (output_dir / "validation.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(dict(step=last_step, improved=improved, monitor=args.monitor,
                                                 monitor_value=score, **report), allow_nan=False) + "\n")
                print(json.dumps({"validation_step": last_step, "loss": report.get("loss"),
                                  "monitor": args.monitor, "monitor_value": score,
                                  "best_step": state["best_step"]}), flush=True)
            if world > 1:
                decision = torch.tensor([int(stopped_early), int(improved)], device=device)
                dist.broadcast(decision, src=0)
                stopped_early, improved = map(bool, decision.tolist())
            if improved:
                rank_rng = collect_rng_states(device, world, rank)
                if rank == 0:
                    save_checkpoint(output_dir / "best.pt", model, optimizer=optimizer, scaler=scaler,
                                    step=last_step, trainer=trainer, rank_rng=rank_rng, ema=ema)
        if last_step % args.save_every == 0 or last_step == end_step or stopped_early:
            rank_rng = collect_rng_states(device, world, rank)
            if rank == 0:
                save_checkpoint(output_dir / "latest.pt", model, optimizer=optimizer, scaler=scaler,
                                step=last_step, trainer=trainer, rank_rng=rank_rng, ema=ema)
        if stopped_early:
            break
    if rank == 0:
        if last_evaluated != last_step:
            eval_context = ema.apply_to(model) if ema is not None else nullcontext()
            with eval_context:
                report = evaluate(model, val, args.amp)
        report = dict(report, step=last_step, best_step=state["best_step"], stopped_early=stopped_early,
                      monitor=args.monitor, best_monitor_value=state["best_loss"] if state["best_step"] is not None else None)
        (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"evaluation": report}), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
