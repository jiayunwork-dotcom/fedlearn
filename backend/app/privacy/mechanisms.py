from __future__ import annotations

import math
import numpy as np
import torch
from typing import Tuple, List, Dict, Optional


class RDPAccountant:
    def __init__(self):
        self.orders = [2 ** i for i in range(1, 12)] + [2 ** 11 + 1]
        self.rdp = [0.0] * len(self.orders)

    def _compute_rdp_single(self, q: float, sigma: float, order: float) -> float:
        if sigma <= 0:
            return float('inf')
        if q == 0:
            return 0.0

        term1 = order * math.log(1 + q * q * (math.exp(1.0 / (sigma * sigma)) - 1))
        term2 = math.log(1 + q * (math.exp(1.0 / (sigma * sigma)) - 1))

        if order * term2 > term1:
            return float('inf')

        return term1 / (order - 1) if order > 1 else term2

    def accumulate(self, q: float, sigma: float, steps: int):
        for i, order in enumerate(self.orders):
            rdp_step = self._compute_rdp_single(q, sigma, order)
            self.rdp[i] += steps * rdp_step

    def get_epsilon(self, delta: float) -> float:
        epsilons = []
        for i, order in enumerate(self.orders):
            eps = self.rdp[i] + math.log(delta) / (1 - order) if order > 1 else float('inf')
            if not math.isinf(eps):
                epsilons.append(eps)
        return min(epsilons) if epsilons else float('inf')

    def reset(self):
        self.rdp = [0.0] * len(self.orders)


class DPSGD:
    def __init__(self, clip_bound: float, noise_multiplier: float, delta: float = 1e-5):
        self.clip_bound = clip_bound
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        self.accountant = RDPAccountant()
        self.total_steps = 0

    def clip_gradients(self, model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        clip_coef = self.clip_bound / max(total_norm, self.clip_bound)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
        return total_norm

    def add_noise(self, model):
        for p in model.parameters():
            if p.grad is not None:
                noise = torch.normal(
                    0,
                    self.noise_multiplier * self.clip_bound,
                    size=p.grad.shape,
                    device=p.grad.device,
                )
                p.grad.data.add_(noise)

    def accumulate_privacy_cost(self, sampling_rate: float, steps: int):
        self.accountant.accumulate(sampling_rate, self.noise_multiplier, steps)
        self.total_steps += steps

    def get_epsilon(self) -> float:
        return self.accountant.get_epsilon(self.delta)

    def reset(self):
        self.accountant.reset()
        self.total_steps = 0


class AdditiveSecretSharing:
    def __init__(self, n: int):
        self.n = n

    def split(self, secret: float) -> List[Tuple[int, float]]:
        shares = []
        remaining = secret
        for i in range(self.n - 1):
            scale = max(1e-6, abs(secret) * 0.1)
            s = np.random.uniform(-scale, scale)
            shares.append((i + 1, s))
            remaining -= s
        shares.append((self.n, remaining))
        return shares

    def reconstruct(self, shares: List[Tuple[int, float]]) -> float:
        if len(shares) < self.n:
            raise ValueError(f"Need all {self.n} shares, got {len(shares)}")
        return float(sum(s[1] for s in shares))


class SecureAggregation:
    def __init__(self, num_clients: int, threshold: int = None):
        self.num_clients = num_clients
        self.threshold = num_clients
        self.shamir = AdditiveSecretSharing(num_clients)
        self._split_operations = 0
        self._aggregate_operations = 0

    def client_split_update(self, client_id: int, update_params: Dict[str, torch.Tensor], all_client_ids: List[int]) -> dict:
        shares_dict: Dict[str, List] = {}
        for key, value in update_params.items():
            flat = value.cpu().numpy().flatten()
            param_shares: List[List[Tuple[int, float]]] = []
            for v in flat:
                s = self.shamir.split(float(v))
                param_shares.append(s)
            shares_dict[key] = param_shares
        self._split_operations += 1
        return shares_dict

    def aggregate_shares(self, all_client_shares: List[Dict], surviving_ids: List[int]) -> Dict[str, np.ndarray]:
        if len(surviving_ids) < self.threshold:
            raise ValueError(f"Not enough surviving clients: {len(surviving_ids)} < {self.threshold}")

        aggregated: Dict[str, np.ndarray] = {}
        num_surviving = len(surviving_ids)

        first_shares = all_client_shares[0]
        for key in first_shares:
            param_shares_list: List[List] = []
            for client_shares in all_client_shares:
                param_shares_list.append(client_shares[key])

            num_params = len(param_shares_list[0])
            result = np.zeros(num_params, dtype=np.float64)

            for p_idx in range(num_params):
                total = 0.0
                for c_idx in range(num_surviving):
                    share_points = param_shares_list[c_idx][p_idx]
                    client_secret = self.shamir.reconstruct(share_points)
                    total += client_secret
                result[p_idx] = total

            aggregated[key] = result

        self._aggregate_operations += 1
        return aggregated

    def get_stats(self) -> Dict[str, int]:
        return {
            "split_operations": self._split_operations,
            "aggregate_operations": self._aggregate_operations,
            "num_clients": self.num_clients,
            "threshold": self.threshold,
        }
