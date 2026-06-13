import math
import numpy as np
import torch
from typing import Tuple, List


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


class ShamirSecretSharing:
    def __init__(self, n: int, t: int, prime: int = 2 ** 31 - 1):
        self.n = n
        self.t = t
        self.prime = prime

    def _eval_poly(self, coeffs, x):
        result = 0
        for i, c in enumerate(coeffs):
            result = (result + c * pow(x, i, self.prime)) % self.prime
        return result

    def split(self, secret: float) -> List[Tuple[int, float]]:
        coeffs = [secret % self.prime]
        for _ in range(self.t - 1):
            coeffs.append(np.random.randint(0, self.prime))

        shares = []
        for i in range(1, self.n + 1):
            shares.append((i, self._eval_poly(coeffs, i)))
        return shares

    def reconstruct(self, shares: List[Tuple[int, float]]) -> float:
        if len(shares) < self.t:
            raise ValueError(f"Need at least {self.t} shares, got {len(shares)}")

        used = shares[: self.t]
        secret = 0.0
        for i, (xi, yi) in enumerate(used):
            numerator = 1.0
            denominator = 1.0
            for j, (xj, _) in enumerate(used):
                if i != j:
                    numerator *= -xj
                    denominator *= (xi - xj)
            lagrange = numerator / denominator
            secret += yi * lagrange
        return secret


class SecureAggregation:
    def __init__(self, num_clients: int, threshold: int = None):
        self.num_clients = num_clients
        self.threshold = threshold or (num_clients // 2 + 1)
        self.shamir = ShamirSecretSharing(num_clients, self.threshold)

    def client_split_update(self, client_id: int, update_params: dict, all_client_ids: list) -> dict:
        shares_dict = {}
        for key, value in update_params.items():
            flat = value.flatten().tolist()
            shares = []
            for v in flat:
                s = self.shamir.split(v)
                shares.append(s)
            shares_dict[key] = shares
        return shares_dict

    def aggregate_shares(self, all_client_shares: list, surviving_ids: list) -> dict:
        if len(surviving_ids) < self.threshold:
            raise ValueError(f"Not enough surviving clients: {len(surviving_ids)} < {self.threshold}")

        aggregated = {}
        num_surviving = len(surviving_ids)

        first_shares = all_client_shares[0]
        for key in first_shares:
            param_shares_list = []
            for client_shares in all_client_shares:
                param_shares_list.append(client_shares[key])

            num_params = len(param_shares_list[0])
            result = []
            for p_idx in range(num_params):
                surviving_shares = []
                for c_idx in range(num_surviving):
                    share_points = param_shares_list[c_idx][p_idx]
                    surviving_shares.append(share_points[c_idx % len(share_points)])

                reconstructed = self.shamir.reconstruct(surviving_shares)
                result.append(reconstructed / num_surviving)

            sample_shape = None
            for client_shares in all_client_shares:
                if key in client_shares:
                    break
            aggregated[key] = result

        return aggregated
