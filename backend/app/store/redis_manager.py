import json
import logging
from typing import Dict, List, Optional, Any

import redis.asyncio as redis

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
