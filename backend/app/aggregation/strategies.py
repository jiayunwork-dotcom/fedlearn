import copy
import numpy as np
from typing import Dict, List, Optional
import torch
import torch.nn as nn


class BaseAggregator:
    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: List[Dict],
        client_weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError


class FedAvgAggregator(BaseAggregator):
    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: List[Dict],
        client_weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        global_state = global_model.state_dict()
        new_state = {}
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        for key in global_state:
            new_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
            for i, update in enumerate(client_updates):
                new_state[key] += normalized_weights[i] * update["model_state"][key].float()
        return new_state

    def get_name(self) -> str:
        return "FedAvg"


class FedProxAggregator(BaseAggregator):
    def __init__(self, mu: float = 0.01):
        self.mu = mu

    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: List[Dict],
        client_weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        global_state = global_model.state_dict()
        new_state = {}
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        for key in global_state:
            new_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
            for i, update in enumerate(client_updates):
                prox_update = update["model_state"][key].float()
                if "local_steps" in update:
                    prox_update = prox_update - (self.mu / max(1, update["local_steps"])) * (
                        update["model_state"][key].float() - global_state[key].float()
                    )
                new_state[key] += normalized_weights[i] * prox_update
        return new_state

    def get_name(self) -> str:
        return f"FedProx(mu={self.mu})"


class FedNovaAggregator(BaseAggregator):
    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: List[Dict],
        client_weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        global_state = global_model.state_dict()
        new_state = {}

        total_tau_eff = 0.0
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        for i, update in enumerate(client_updates):
            local_steps = update.get("local_steps", 1)
            total_tau_eff += normalized_weights[i] * local_steps

        for key in global_state:
            new_state[key] = global_state[key].float().clone()
            for i, update in enumerate(client_updates):
                local_steps = update.get("local_steps", 1)
                pseudo_gradient = global_state[key].float() - update["model_state"][key].float()
                normalized_pseudo_gradient = pseudo_gradient / local_steps
                new_state[key] -= normalized_weights[i] * total_tau_eff * normalized_pseudo_gradient

        return new_state

    def get_name(self) -> str:
        return "FedNova"


class ScaffoldAggregator(BaseAggregator):
    def __init__(self):
        self.global_control = None

    def initialize_control(self, model: nn.Module):
        self.global_control = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in model.state_dict().items()}

    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: List[Dict],
        client_weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        if self.global_control is None:
            self.initialize_control(global_model)

        global_state = global_model.state_dict()
        new_state = {}
        new_global_control = {}
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        num_selected = len(client_updates)

        for key in global_state:
            new_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
            delta_c_sum = torch.zeros_like(global_state[key], dtype=torch.float32)

            for i, update in enumerate(client_updates):
                new_state[key] += normalized_weights[i] * update["model_state"][key].float()
                client_control = update.get("control_variate", {})
                if key in client_control:
                    delta_c_sum += (client_control[key].float() - self.global_control[key].float())

            new_global_control[key] = self.global_control[key].float() + delta_c_sum / num_selected

        self.global_control = new_global_control
        return new_state

    def get_name(self) -> str:
        return "Scaffold"


def get_aggregator(strategy: str, **kwargs) -> BaseAggregator:
    if strategy == "fedavg":
        return FedAvgAggregator()
    elif strategy == "fedprox":
        return FedProxAggregator(mu=kwargs.get("mu", 0.01))
    elif strategy == "fednova":
        return FedNovaAggregator()
    elif strategy == "scaffold":
        return ScaffoldAggregator()
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")
