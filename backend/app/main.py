import uuid
import asyncio
import logging
import os
from typing import Dict, Optional

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core.trainer import FederatedTrainer, ExperimentConfig
from app.store.redis_manager import RedisManager

logger = logging.getLogger(__name__)

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_mgr = RedisManager(url=redis_url)
trainers: Dict[str, FederatedTrainer] = {}

app = FastAPI(title="Federated Learning Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BatchCompareRequest(BaseModel):
    experiment_ids: list[str] = Field(..., min_length=2, max_length=4)


class CreateExperimentRequest(BaseModel):
    dataset: str = Field(default="mnist", description="mnist | cifar10 | fashion_mnist")
    num_clients: int = Field(default=10, ge=5, le=100)
    client_fraction: float = Field(default=0.4, ge=0.1, le=1.0)
    local_epochs: int = Field(default=5, ge=1)
    global_rounds: int = Field(default=200, ge=1)
    early_stop_patience: int = Field(default=10, ge=1)
    aggregation_strategy: str = Field(default="fedavg", description="fedavg | fedprox | fednova | scaffold")
    fedprox_mu: float = Field(default=0.01, ge=0.0)
    non_iid_type: str = Field(default="label_skew", description="label_skew | dirichlet | feature_skew")
    non_iid_param: float = Field(default=2.0)
    learning_rate: float = Field(default=0.01, ge=0.0001)
    batch_size: int = Field(default=32, ge=8)
    client_selection: str = Field(default="random", description="random | resource | contribution")
    dp_enabled: bool = Field(default=False)
    dp_clip_bound: float = Field(default=1.0, ge=0.01)
    dp_noise_multiplier: float = Field(default=1.0, ge=0.01)
    dp_delta: float = Field(default=1e-5)
    dp_epsilon_budget: float = Field(default=10.0, ge=0.1)
    secure_aggregation: bool = Field(default=False)
    secure_agg_threshold: Optional[int] = Field(default=None)
    drop_rate: float = Field(default=0.0, ge=0.0, le=0.8)
    attack_type: str = Field(default="none", description="none | data_poisoning | model_poisoning | backdoor")
    attack_ratio: float = Field(default=0.0, ge=0.0, le=0.5)
    defense_type: str = Field(default="none", description="none | krum | trimmed_mean | median")
    defense_beta: float = Field(default=0.2, ge=0.0, le=0.5)


@app.on_event("startup")
async def startup():
    await redis_mgr.connect()
    logger.info("Application started, Redis connected")


@app.on_event("shutdown")
async def shutdown():
    await redis_mgr.disconnect()
    logger.info("Application shutdown")


@app.post("/api/experiments")
async def create_experiment(req: CreateExperimentRequest):
    experiment_id = str(uuid.uuid4())[:8]

    config = ExperimentConfig(
        experiment_id=experiment_id,
        **req.model_dump(),
    )

    trainer = FederatedTrainer(config, redis_client=redis_mgr.client)
    trainers[experiment_id] = trainer

    config_dict = req.model_dump()
    config_dict["experiment_id"] = experiment_id
    config_dict["status"] = "created"
    await redis_mgr.set_experiment(experiment_id, config_dict)
    await redis_mgr.set_state(experiment_id, {
        "current_round": 0,
        "global_rounds": config.global_rounds,
        "best_accuracy": 0.0,
        "is_running": False,
        "epsilon": 0.0,
    })

    return {"experiment_id": experiment_id, "config": config_dict}


@app.post("/api/experiments/{experiment_id}/start")
async def start_training(experiment_id: str):
    if experiment_id not in trainers:
        raise HTTPException(status_code=404, detail="Experiment not found")

    trainer = trainers[experiment_id]
    if trainer.is_running:
        raise HTTPException(status_code=400, detail="Training already running")

    config = await redis_mgr.get_experiment(experiment_id)
    if config:
        config["status"] = "running"
        await redis_mgr.set_experiment(experiment_id, config)

    asyncio.create_task(trainer.run_training())

    return {"experiment_id": experiment_id, "status": "running"}


@app.get("/api/experiments/{experiment_id}/status")
async def get_status(experiment_id: str):
    if experiment_id not in trainers:
        cached = await redis_mgr.get_state(experiment_id)
        if cached:
            return {"experiment_id": experiment_id, **cached}
        raise HTTPException(status_code=404, detail="Experiment not found")

    trainer = trainers[experiment_id]
    status = trainer.get_status()

    history = trainer.get_all_metrics()
    if history:
        latest = history[-1]
        status["current_accuracy"] = latest["global_accuracy"]
        status["current_loss"] = latest["global_loss"]
        status["epsilon"] = latest["epsilon"]
        status["client_similarity"] = latest["client_similarity"]
        total_rounds = status["global_rounds"]
        if total_rounds > 0 and len(history) > 1:
            avg_round_time = sum(h["round_seconds"] for h in history) / len(history)
            remaining = total_rounds - status["current_round"]
            status["estimated_remaining_seconds"] = avg_round_time * remaining

    if not trainer.is_running:
        config = await redis_mgr.get_experiment(experiment_id)
        if config and config.get("status") == "running":
            config["status"] = "completed"
            await redis_mgr.set_experiment(experiment_id, config)

    return status


@app.get("/api/experiments/{experiment_id}/metrics")
async def get_metrics(experiment_id: str):
    if experiment_id not in trainers:
        history = await redis_mgr.get_history(experiment_id)
        if history is not None:
            return {"experiment_id": experiment_id, "metrics": history}
        raise HTTPException(status_code=404, detail="Experiment not found")

    trainer = trainers[experiment_id]
    return {"experiment_id": experiment_id, "metrics": trainer.get_all_metrics()}


@app.post("/api/experiments/{experiment_id}/stop")
async def stop_training(experiment_id: str):
    if experiment_id not in trainers:
        raise HTTPException(status_code=404, detail="Experiment not found")

    trainer = trainers[experiment_id]
    trainer.stop()

    config = await redis_mgr.get_experiment(experiment_id)
    if config:
        config["status"] = "stopped"
        await redis_mgr.set_experiment(experiment_id, config)

    return {"experiment_id": experiment_id, "status": "stopping"}


@app.get("/api/experiments")
async def list_experiments():
    experiments = await redis_mgr.list_experiments()
    return {"experiments": experiments}


@app.get("/api/experiments/{experiment_id}/contributions")
async def get_contributions(experiment_id: str):
    if experiment_id not in trainers:
        data = await redis_mgr.get_contributions(experiment_id)
        if data is not None:
            return {"experiment_id": experiment_id, "contributions": data}
        raise HTTPException(status_code=404, detail="Experiment not found")

    trainer = trainers[experiment_id]
    return {"experiment_id": experiment_id, "contributions": trainer.get_contribution_ranking()}


@app.get("/api/experiments/{experiment_id}/label-distribution")
async def get_label_distribution(experiment_id: str):
    data = await redis_mgr.get_label_distribution(experiment_id)
    if data is not None:
        return {"experiment_id": experiment_id, "distribution": data}

    if experiment_id not in trainers:
        raise HTTPException(status_code=404, detail="Experiment not found")

    trainer = trainers[experiment_id]
    return {"experiment_id": experiment_id, "distribution": trainer.label_distribution}


@app.get("/api/experiments/{experiment_id}/attack-log")
async def get_attack_log(experiment_id: str):
    if experiment_id not in trainers:
        data = await redis_mgr.get_attack_log(experiment_id)
        if data is not None:
            return {"experiment_id": experiment_id, "attack_log": data}
        raise HTTPException(status_code=404, detail="Experiment not found")

    trainer = trainers[experiment_id]
    return {"experiment_id": experiment_id, "attack_log": trainer.get_attack_log()}


@app.get("/api/experiments/{experiment_id}/compare")
async def compare_experiments(experiment_id: str, other_id: str):
    metrics1 = None
    metrics2 = None

    if experiment_id in trainers:
        metrics1 = trainers[experiment_id].get_all_metrics()
    else:
        h = await redis_mgr.get_history(experiment_id)
        if h:
            metrics1 = h

    if other_id in trainers:
        metrics2 = trainers[other_id].get_all_metrics()
    else:
        h = await redis_mgr.get_history(other_id)
        if h:
            metrics2 = h

    if metrics1 is None or metrics2 is None:
        raise HTTPException(status_code=404, detail="One or both experiments not found")

    return {
        "experiment_1": {"id": experiment_id, "metrics": metrics1},
        "experiment_2": {"id": other_id, "metrics": metrics2},
    }


@app.post("/api/experiments/batch-compare")
async def batch_compare_experiments(req: BatchCompareRequest):
    results = []
    for eid in req.experiment_ids:
        config = await redis_mgr.get_experiment(eid)
        if config is None:
            raise HTTPException(status_code=404, detail=f"Experiment {eid} not found")

        metrics = None
        if eid in trainers:
            metrics = trainers[eid].get_all_metrics()
        else:
            metrics = await redis_mgr.get_history(eid)
        if metrics is None:
            metrics = []

        contributions = None
        if eid in trainers:
            contributions = trainers[eid].get_contribution_ranking()
        else:
            contributions = await redis_mgr.get_contributions(eid)
        if contributions is None:
            contributions = {}

        state = await redis_mgr.get_state(eid)
        total_elapsed = sum(m.get("elapsed_seconds", 0) for m in metrics)
        avg_round_time = 0.0
        if metrics:
            avg_round_time = sum(m.get("round_seconds", 0) for m in metrics) / len(metrics)

        final_accuracy = 0.0
        final_epsilon = 0.0
        round_to_90 = None
        if metrics:
            final_accuracy = metrics[-1].get("global_accuracy", 0.0)
            final_epsilon = metrics[-1].get("epsilon", 0.0)
            for m in metrics:
                if m.get("global_accuracy", 0) >= 0.9:
                    round_to_90 = m["round"]
                    break

        summary = {
            "final_accuracy": final_accuracy,
            "round_to_90_percent": round_to_90,
            "final_epsilon": final_epsilon,
            "total_elapsed_seconds": total_elapsed,
            "avg_round_seconds": avg_round_time,
            "best_accuracy": state.get("best_accuracy", 0.0) if state else 0.0,
            "current_round": state.get("current_round", 0) if state else 0,
            "global_rounds": state.get("global_rounds", 0) if state else 0,
        }

        results.append({
            "experiment_id": eid,
            "config": config,
            "metrics": metrics,
            "contributions": contributions,
            "summary": summary,
        })

    return {"comparisons": results}
