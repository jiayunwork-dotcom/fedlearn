import numpy as np
import random
from typing import Dict, List, Optional


class BaseSelector:
    def select(self, client_ids: List[int], fraction: float, round_num: int, **kwargs) -> List[int]:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError


class RandomSelector(BaseSelector):
    def select(self, client_ids: List[int], fraction: float, round_num: int, **kwargs) -> List[int]:
        k = max(1, int(len(client_ids) * fraction))
        return random.sample(client_ids, k)

    def get_name(self) -> str:
        return "random"


class ResourceSelector(BaseSelector):
    def __init__(self, client_resources: Optional[Dict[int, float]] = None):
        self.client_resources = client_resources or {}

    def select(self, client_ids: List[int], fraction: float, round_num: int, **kwargs) -> List[int]:
        k = max(1, int(len(client_ids) * fraction))

        if not self.client_resources:
            available = {cid: np.random.uniform(0.1, 1.0) for cid in client_ids}
        else:
            available = {cid: self.client_resources.get(cid, 0.5) for cid in client_ids}

        sorted_clients = sorted(available.keys(), key=lambda x: available[x], reverse=True)
        return sorted_clients[:k]

    def get_name(self) -> str:
        return "resource"


class ContributionSelector(BaseSelector):
    def __init__(self):
        self.contributions: Dict[int, float] = {}

    def update_contributions(self, client_id: int, accuracy_improvement: float):
        self.contributions[client_id] = self.contributions.get(client_id, 0.0) + max(0, accuracy_improvement)

    def select(self, client_ids: List[int], fraction: float, round_num: int, **kwargs) -> List[int]:
        k = max(1, int(len(client_ids) * fraction))

        if not self.contributions:
            return random.sample(client_ids, k)

        weights = []
        for cid in client_ids:
            w = self.contributions.get(cid, 0.0) + 1e-6
            weights.append(w)

        total = sum(weights)
        probs = [w / total for w in weights]

        selected = np.random.choice(client_ids, size=k, replace=False, p=probs).tolist()
        return selected

    def get_name(self) -> str:
        return "contribution"


def get_selector(strategy: str, **kwargs) -> BaseSelector:
    if strategy == "random":
        return RandomSelector()
    elif strategy == "resource":
        return ResourceSelector(kwargs.get("client_resources"))
    elif strategy == "contribution":
        return ContributionSelector()
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")
