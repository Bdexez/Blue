"""Moyenne mobile exponentielle (EMA) des poids avec reprise exacte."""

from contextlib import contextmanager
import math
import torch


class ModelEMA:
    """Maintenir une copie lissée des paramètres entraînables du modèle."""

    def __init__(self, model: torch.nn.Module, decay: float):
        if not math.isfinite(decay) or not 0.0 < decay < 1.0:
            raise ValueError("ema_decay doit être fini et strictement compris entre 0 et 1")
        self.decay = float(decay)
        self.updates = 0
        self.shadow: dict[str, torch.Tensor] = {}
        seen_ids: set[int] = set()
        for name, param in model.named_parameters():
            if not param.is_floating_point() or id(param) in seen_ids:
                continue
            seen_ids.add(id(param))
            self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.updates += 1
        effective_decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        params = dict(model.named_parameters())
        for name, shadow in self.shadow.items():
            target = params[name].detach()
            if shadow.device != target.device:
                shadow = shadow.to(target.device)
                self.shadow[name] = shadow
            shadow.lerp_(target, 1.0 - effective_decay)

    @contextmanager
    def apply_to(self, model: torch.nn.Module):
        """Remplacer temporairement les poids du modèle par leur moyenne EMA."""
        params = dict(model.named_parameters())
        backup = {name: params[name].detach().clone() for name in self.shadow}
        try:
            with torch.no_grad():
                for name, shadow in self.shadow.items():
                    params[name].copy_(shadow.to(device=params[name].device, dtype=params[name].dtype))
                if hasattr(model, "lm_head") and hasattr(model, "embedding"):
                    model.lm_head.weight = model.embedding.weight
            yield model
        finally:
            with torch.no_grad():
                for name, saved in backup.items():
                    params[name].copy_(saved)
                if hasattr(model, "lm_head") and hasattr(model, "embedding"):
                    model.lm_head.weight = model.embedding.weight

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "updates": self.updates,
            "shadow": {name: tensor.detach().cpu() for name, tensor in self.shadow.items()},
        }

    def load_state_dict(self, state: dict):
        if not isinstance(state, dict) or "shadow" not in state or "updates" not in state:
            raise ValueError("État EMA invalide dans le checkpoint")
        if abs(float(state.get("decay", self.decay)) - self.decay) > 1e-12:
            raise ValueError("ema_decay diffère de l'état EMA du checkpoint")
        saved_shadow = state["shadow"]
        if set(saved_shadow) != set(self.shadow):
            raise ValueError("Les paramètres EMA du checkpoint ne correspondent pas au modèle")
        self.updates = int(state["updates"])
        for name, tensor in saved_shadow.items():
            self.shadow[name] = tensor.to(device=self.shadow[name].device, dtype=self.shadow[name].dtype).clone()
