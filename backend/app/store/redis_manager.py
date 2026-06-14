from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING

import redis.asyncio as redis

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class RedisManager:
    def __init__(self, url: str = "redis://redis:6379/0"):
        self.url = url
        self.client: Optional[redis.Redis] = None

    async def connect(self):
        self.client = redis.from_url(self.url, decode_responses=True)
        logger.info(f"Connected to Redis at {self.url}")

    async def disconnect(self):
        if self.client:
            await self.client.close()

    async def set_experiment(self, experiment_id: str, config: dict):
        key = f"experiment:{experiment_id}:config"
        await self.client.set(key, json.dumps(config))

    async def get_experiment(self, experiment_id: str) -> Optional[dict]:
        key = f"experiment:{experiment_id}:config"
        data = await self.client.get(key)
        return json.loads(data) if data else None

    async def list_experiments(self) -> List[dict]:
        keys = await self.client.keys("experiment:*:config")
        experiments = []
        for key in keys:
            data = await self.client.get(key)
            if data:
                experiments.append(json.loads(data))
        return experiments

    async def set_state(self, experiment_id: str, state: dict):
        key = f"experiment:{experiment_id}:state"
        await self.client.set(key, json.dumps(state))

    async def get_state(self, experiment_id: str) -> Optional[dict]:
        key = f"experiment:{experiment_id}:state"
        data = await self.client.get(key)
        return json.loads(data) if data else None

    async def append_metrics(self, experiment_id: str, metrics: dict):
        key = f"experiment:{experiment_id}:history"
        await self.client.rpush(key, json.dumps(metrics))

    async def get_history(self, experiment_id: str) -> List[dict]:
        key = f"experiment:{experiment_id}:history"
        data = await self.client.lrange(key, 0, -1)
        return [json.loads(item) for item in data]

    async def set_label_distribution(self, experiment_id: str, distribution: list):
        key = f"experiment:{experiment_id}:label_dist"
        await self.client.set(key, json.dumps(distribution))

    async def get_label_distribution(self, experiment_id: str) -> Optional[list]:
        key = f"experiment:{experiment_id}:label_dist"
        data = await self.client.get(key)
        return json.loads(data) if data else None

    async def set_contributions(self, experiment_id: str, contributions: dict):
        key = f"experiment:{experiment_id}:contributions"
        await self.client.set(key, json.dumps(contributions))

    async def get_contributions(self, experiment_id: str) -> Optional[dict]:
        key = f"experiment:{experiment_id}:contributions"
        data = await self.client.get(key)
        return json.loads(data) if data else None

    async def set_attack_log(self, experiment_id: str, log: list):
        key = f"experiment:{experiment_id}:attack_log"
        await self.client.set(key, json.dumps(log))

    async def get_attack_log(self, experiment_id: str) -> Optional[list]:
        key = f"experiment:{experiment_id}:attack_log"
        data = await self.client.get(key)
        return json.loads(data) if data else None

    async def delete_experiment(self, experiment_id: str):
        pattern = f"experiment:{experiment_id}:*"
        keys = await self.client.keys(pattern)
        if keys:
            await self.client.delete(*keys)

    async def set_interpretability_result(
        self,
        experiment_id: str,
        method: str,
        num_samples: int,
        result: dict
    ):
        key = f"experiment:{experiment_id}:interpretability:{method}:{num_samples}"
        await self.client.set(key, json.dumps(result))

    async def get_interpretability_result(
        self,
        experiment_id: str,
        method: str,
        num_samples: int
    ) -> Optional[dict]:
        key = f"experiment:{experiment_id}:interpretability:{method}:{num_samples}"
        data = await self.client.get(key)
        return json.loads(data) if data else None

    async def delete_interpretability_result(
        self,
        experiment_id: str,
        method: str,
        num_samples: int
    ):
        key = f"experiment:{experiment_id}:interpretability:{method}:{num_samples}"
        await self.client.delete(key)

    async def list_interpretability_results(self, experiment_id: str) -> List[List[str]]:
        pattern = f"experiment:{experiment_id}:interpretability:*"
        keys = await self.client.keys(pattern)
        results = []
        for key in keys:
            parts = key.split(":")
            if len(parts) >= 5:
                results.append([parts[-2], parts[-1]])
        return results

    async def set_interpretability_meta(
        self,
        experiment_id: str,
        method: str,
        num_samples: int,
        meta: dict
    ):
        key = f"experiment:{experiment_id}:interpretability:{method}:{num_samples}:meta"
        await self.client.set(key, json.dumps(meta))

    async def get_interpretability_meta(
        self,
        experiment_id: str,
        method: str,
        num_samples: int
    ) -> Optional[dict]:
        key = f"experiment:{experiment_id}:interpretability:{method}:{num_samples}:meta"
        data = await self.client.get(key)
        return json.loads(data) if data else None

    async def delete_interpretability_meta(
        self,
        experiment_id: str,
        method: str,
        num_samples: int
    ):
        key = f"experiment:{experiment_id}:interpretability:{method}:{num_samples}:meta"
        await self.client.delete(key)

    async def set_interpretability_running_state(
        self,
        experiment_id: str,
        method: str,
        num_samples: int,
        state: dict
    ):
        key = f"experiment:{experiment_id}:interpretability:{method}:{num_samples}:running"
        await self.client.set(key, json.dumps(state))

    async def get_interpretability_running_state(
        self,
        experiment_id: str,
        method: str,
        num_samples: int
    ) -> Optional[dict]:
        key = f"experiment:{experiment_id}:interpretability:{method}:{num_samples}:running"
        data = await self.client.get(key)
        return json.loads(data) if data else None

    async def delete_interpretability_running_state(
        self,
        experiment_id: str,
        method: str,
        num_samples: int
    ):
        key = f"experiment:{experiment_id}:interpretability:{method}:{num_samples}:running"
        await self.client.delete(key)

    async def append_interpretability_log(
        self,
        experiment_id: str,
        method: str,
        num_samples: int,
        log_entry: dict
    ):
        key = f"experiment:{experiment_id}:interpretability:{method}:{num_samples}:logs"
        await self.client.rpush(key, json.dumps(log_entry))

    async def get_interpretability_logs(
        self,
        experiment_id: str,
        method: str,
        num_samples: int
    ) -> List[dict]:
        key = f"experiment:{experiment_id}:interpretability:{method}:{num_samples}:logs"
        data = await self.client.lrange(key, 0, -1)
        return [json.loads(item) for item in data]

    async def delete_interpretability_logs(
        self,
        experiment_id: str,
        method: str,
        num_samples: int
    ):
        key = f"experiment:{experiment_id}:interpretability:{method}:{num_samples}:logs"
        await self.client.delete(key)

    async def list_running_interpretability(self, experiment_id: str) -> List[List[str]]:
        pattern = f"experiment:{experiment_id}:interpretability:*:*:running"
        keys = await self.client.keys(pattern)
        results = []
        for key in keys:
            parts = key.split(":")
            if len(parts) >= 6:
                results.append([parts[-4], parts[-3]])
        return results

    async def set_global_model_state(self, experiment_id: str, state_dict: Dict):
        key = f"experiment:{experiment_id}:global_model_state"
        serializable = {k: v.tolist() for k, v in state_dict.items()}
        await self.client.set(key, json.dumps(serializable))

    async def get_global_model_state(self, experiment_id: str) -> Optional[Dict[str, torch.Tensor]]:
        key = f"experiment:{experiment_id}:global_model_state"
        data = await self.client.get(key)
        if data is None:
            return None
        try:
            import torch
            serializable = json.loads(data)
            return {k: torch.tensor(v) for k, v in serializable.items()}
        except Exception as e:
            logger.error(f"Failed to deserialize global model state: {e}")
            return None
