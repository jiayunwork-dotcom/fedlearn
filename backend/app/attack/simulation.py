import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional


class DataPoisoningAttack:
    def __init__(self, poison_ratio: float = 0.5):
        self.poison_ratio = poison_ratio

    def poison_labels(self, labels: List[int], num_classes: int = 10) -> List[int]:
        poisoned = []
        for label in labels:
            if np.random.random() < self.poison_ratio:
                new_label = np.random.randint(0, num_classes)
                while new_label == label:
                    new_label = np.random.randint(0, num_classes)
                poisoned.append(int(new_label))
            else:
                poisoned.append(label)
        return poisoned


class ModelPoisoningAttack:
    def __init__(self, attack_scale: float = -10.0):
        self.attack_scale = attack_scale

    def poison_update(self, model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        poisoned = {}
        for key, value in model_state.items():
            poisoned[key] = value * self.attack_scale
        return poisoned


class BackdoorAttack:
    def __init__(self, trigger_label: int = 0, target_label: int = 1, poison_ratio: float = 0.3):
        self.trigger_label = trigger_label
        self.target_label = target_label
        self.poison_ratio = poison_ratio

    def inject_backdoor(self, images: torch.Tensor, labels: torch.Tensor) -> tuple:
        poisoned_images = images.clone()
        poisoned_labels = labels.clone()

        for i in range(len(labels)):
            if np.random.random() < self.poison_ratio:
                poisoned_images[i, :, 0:3, 0:3] = 1.0
                poisoned_labels[i] = self.target_label

        return poisoned_images, poisoned_labels


class KrumDefense:
    def __init__(self, num_byzantine: int = 0):
        self.num_byzantine = num_byzantine

    def _flatten(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([v.flatten().float() for v in state.values()])

    def _pairwise_distance(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        n = len(vectors)
        dists = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                d = torch.norm(vectors[i] - vectors[j]).item()
                dists[i][j] = d
                dists[j][i] = d
        return dists

    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: List[Dict],
        client_weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        n = len(client_updates)
        flat_updates = [self._flatten(u["model_state"]) for u in client_updates]
        dists = self._pairwise_distance(flat_updates)

        closest = n - self.num_byzantine - 2
        scores = []
        for i in range(n):
            sorted_dists = torch.sort(dists[i])[0]
            scores.append(sorted_dists[1 : closest + 1].sum().item())

        best_idx = np.argmin(scores)
        return client_updates[best_idx]["model_state"]

    def get_name(self) -> str:
        return "Krum"


class TrimmedMeanDefense:
    def __init__(self, beta: float = 0.2):
        self.beta = beta

    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: List[Dict],
        client_weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        global_state = global_model.state_dict()
        new_state = {}
        n = len(client_updates)
        k = max(1, int(n * self.beta))

        for key in global_state:
            stacked = torch.stack([u["model_state"][key].float() for u in client_updates], dim=0)
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[k : n - k]
            new_state[key] = trimmed.mean(dim=0)

        return new_state

    def get_name(self) -> str:
        return f"TrimmedMean(beta={self.beta})"


class MedianDefense:
    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: List[Dict],
        client_weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        global_state = global_model.state_dict()
        new_state = {}

        for key in global_state:
            stacked = torch.stack([u["model_state"][key].float() for u in client_updates], dim=0)
            new_state[key], _ = torch.median(stacked, dim=0)

        return new_state

    def get_name(self) -> str:
        return "Median"


def get_defense(defense_name: str, **kwargs):
    if defense_name == "krum":
        return KrumDefense(num_byzantine=kwargs.get("num_byzantine", 0))
    elif defense_name == "trimmed_mean":
        return TrimmedMeanDefense(beta=kwargs.get("beta", 0.2))
    elif defense_name == "median":
        return MedianDefense()
    elif defense_name == "none":
        return None
    else:
        raise ValueError(f"Unknown defense: {defense_name}")
