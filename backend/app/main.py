import uuid
import asyncio
import logging
import os
import json
from typing import Dict, Optional, List

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import torch

from app.core.trainer import FederatedTrainer, ExperimentConfig
from app.store.redis_manager import RedisManager
from app.interpretability.service import InterpretabilityService, AnalysisMethod, AnalysisStatus

logger = logging.getLogger(__name__)

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_mgr = RedisManager(url=redis_url)
trainers: Dict[str, FederatedTrainer] = {}
interpretability_services: Dict[str, InterpretabilityService] = {}
active_ws_connections: Dict[str, List[WebSocket]] = {}

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
        total_elapsed = metrics[-1].get("elapsed_seconds", 0) if metrics else 0.0
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


class InterpretabilityRequest(BaseModel):
    method: AnalysisMethod = Field(..., description="Analysis method: gradient | permutation | shap")
    num_samples: int = Field(..., ge=10, le=500, description="Number of samples (10, 50, 100, 500)")


async def _get_global_model_state(experiment_id: str) -> Optional[Dict[str, torch.Tensor]]:
    if experiment_id in trainers:
        trainer = trainers[experiment_id]
        return {k: v.cpu() for k, v in trainer.global_model.state_dict().items()}
    
    logger.warning(f"Experiment {experiment_id} not in trainers, attempting to load from Redis")
    saved_state = await redis_mgr.get_global_model_state(experiment_id)
    if saved_state is not None:
        logger.info(f"Successfully loaded global model state from Redis for {experiment_id}")
        return {k: v.cpu() for k, v in saved_state.items()}
    
    logger.error(f"No global model state available for {experiment_id}")
    config = await redis_mgr.get_experiment(experiment_id)
    if config is None:
        return None
    
    from app.models.networks import get_model
    model = get_model(config.get("dataset", "mnist"))
    logger.warning(f"Falling back to untrained model for {experiment_id} - interpretability results will be meaningless")
    return {k: v.cpu() for k, v in model.state_dict().items()}


async def _broadcast_interpretability_progress(
    experiment_id: str,
    progress: float,
    current_sample: int,
    log_entry: Optional[Dict] = None
):
    ws_key = f"interpretability:{experiment_id}"
    connections = active_ws_connections.get(ws_key, [])
    
    message = {
        "type": "progress",
        "progress": progress,
        "current_sample": current_sample,
    }
    if log_entry:
        message["log"] = log_entry
    
    disconnected = []
    for ws in connections:
        try:
            await ws.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send progress to WebSocket: {e}")
            disconnected.append(ws)
    
    for ws in disconnected:
        connections.remove(ws)


@app.post("/api/experiments/{experiment_id}/interpretability/start")
async def start_interpretability_analysis(
    experiment_id: str,
    req: InterpretabilityRequest
):
    config = await redis_mgr.get_experiment(experiment_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if config.get("status") not in ("completed", "stopped"):
        raise HTTPException(
            status_code=400,
            detail="Interpretability analysis is only available for completed experiments"
        )
    
    cached_result = await redis_mgr.get_interpretability_result(
        experiment_id, req.method.value, req.num_samples
    )
    if cached_result is not None:
        return {
            "experiment_id": experiment_id,
            "method": req.method.value,
            "num_samples": req.num_samples,
            "status": AnalysisStatus.COMPLETED.value,
            "cached": True,
            "result": cached_result
        }
    
    service_key = f"{experiment_id}:{req.method.value}:{req.num_samples}"
    if service_key in interpretability_services:
        existing = interpretability_services[service_key]
        status = existing.get_status()
        if status["status"] == AnalysisStatus.RUNNING.value:
            raise HTTPException(
                status_code=400,
                detail="Analysis is already running for this experiment with the same parameters"
            )
    
    try:
        num_classes = 10
        if req.num_samples < num_classes:
            raise ValueError(
                f"Sample count ({req.num_samples}) must be at least the number of classes ({num_classes})"
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    service = InterpretabilityService(experiment_id, config, redis_client=redis_mgr.client)
    
    global_state = await _get_global_model_state(experiment_id)
    if global_state is None:
        raise HTTPException(status_code=500, detail="Failed to load global model state")
    
    service.load_global_model(global_state)
    service.setup_data()
    
    def progress_callback(progress: float, current_sample: int, log_entry: Optional[Dict] = None):
        asyncio.create_task(
            _broadcast_interpretability_progress(experiment_id, progress, current_sample, log_entry)
        )
    
    service.add_progress_callback(progress_callback)
    
    interpretability_services[service_key] = service
    
    async def run_analysis():
        try:
            result = await service.analyze(req.method, req.num_samples)
            
            if result.get("status") == AnalysisStatus.COMPLETED.value:
                await redis_mgr.set_interpretability_result(
                    experiment_id, req.method.value, req.num_samples, result
                )
            
            ws_key = f"interpretability:{experiment_id}"
            connections = active_ws_connections.get(ws_key, [])
            message = {
                "type": "complete",
                "status": result.get("status"),
                "result": result
            }
            
            disconnected = []
            for ws in connections:
                try:
                    await ws.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send completion to WebSocket: {e}")
                    disconnected.append(ws)
            
            for ws in disconnected:
                connections.remove(ws)
            
            if service_key in interpretability_services:
                del interpretability_services[service_key]
            
        except Exception as e:
            logger.error(f"Analysis task failed: {e}", exc_info=True)
            if service_key in interpretability_services:
                del interpretability_services[service_key]
    
    asyncio.create_task(run_analysis())
    
    return {
        "experiment_id": experiment_id,
        "method": req.method.value,
        "num_samples": req.num_samples,
        "status": AnalysisStatus.RUNNING.value,
        "cached": False
    }


@app.post("/api/experiments/{experiment_id}/interpretability/cancel")
async def cancel_interpretability_analysis(
    experiment_id: str,
    method: Optional[AnalysisMethod] = None,
    num_samples: Optional[int] = None
):
    if method and num_samples:
        service_key = f"{experiment_id}:{method.value}:{num_samples}"
        if service_key in interpretability_services:
            interpretability_services[service_key].cancel()
            return {
                "experiment_id": experiment_id,
                "method": method.value,
                "num_samples": num_samples,
                "status": AnalysisStatus.CANCELLED.value
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="No running analysis found for the specified parameters"
            )
    else:
        cancelled = []
        for service_key, service in list(interpretability_services.items()):
            if service_key.startswith(f"{experiment_id}:"):
                service.cancel()
                parts = service_key.split(":")
                if len(parts) >= 3:
                    cancelled.append({
                        "method": parts[-2],
                        "num_samples": int(parts[-1])
                    })
                del interpretability_services[service_key]
        
        if not cancelled:
            raise HTTPException(
                status_code=404,
                detail="No running analysis found for this experiment"
            )
        
        return {
            "experiment_id": experiment_id,
            "status": AnalysisStatus.CANCELLED.value,
            "cancelled_tasks": cancelled
        }


@app.get("/api/experiments/{experiment_id}/interpretability/status")
async def get_interpretability_status(
    experiment_id: str,
    method: Optional[AnalysisMethod] = None,
    num_samples: Optional[int] = None
):
    if method and num_samples:
        service_key = f"{experiment_id}:{method.value}:{num_samples}"
        if service_key in interpretability_services:
            service = interpretability_services[service_key]
            return {
                "experiment_id": experiment_id,
                "method": method.value,
                "num_samples": num_samples,
                **service.get_status()
            }
        
        cached = await redis_mgr.get_interpretability_result(
            experiment_id, method.value, num_samples
        )
        if cached is not None:
            return {
                "experiment_id": experiment_id,
                "method": method.value,
                "num_samples": num_samples,
                "status": AnalysisStatus.COMPLETED.value,
                "progress": 100.0,
                "cached": True
            }
        
        return {
            "experiment_id": experiment_id,
            "method": method.value,
            "num_samples": num_samples,
            "status": AnalysisStatus.PENDING.value,
            "progress": 0.0
        }
    else:
        statuses = []
        for service_key, service in interpretability_services.items():
            if service_key.startswith(f"{experiment_id}:"):
                parts = service_key.split(":")
                if len(parts) >= 3:
                    statuses.append({
                        "method": parts[-2],
                        "num_samples": int(parts[-1]),
                        **service.get_status()
                    })
        
        cached_results = await redis_mgr.list_interpretability_results(experiment_id)
        for method_str, num_samples_str in cached_results:
            statuses.append({
                "method": method_str,
                "num_samples": int(num_samples_str),
                "status": AnalysisStatus.COMPLETED.value,
                "progress": 100.0,
                "cached": True
            })
        
        return {
            "experiment_id": experiment_id,
            "analyses": statuses
        }


@app.get("/api/experiments/{experiment_id}/interpretability/result")
async def get_interpretability_result(
    experiment_id: str,
    method: AnalysisMethod,
    num_samples: int
):
    cached = await redis_mgr.get_interpretability_result(
        experiment_id, method.value, num_samples
    )
    if cached is not None:
        return {
            "experiment_id": experiment_id,
            "method": method.value,
            "num_samples": num_samples,
            "status": AnalysisStatus.COMPLETED.value,
            "result": cached
        }
    
    service_key = f"{experiment_id}:{method.value}:{num_samples}"
    if service_key in interpretability_services:
        service = interpretability_services[service_key]
        status = service.get_status()
        return {
            "experiment_id": experiment_id,
            "method": method.value,
            "num_samples": num_samples,
            "status": status["status"],
            "progress": status["progress"]
        }
    
    raise HTTPException(
        status_code=404,
        detail="No result found for the specified analysis parameters"
    )


@app.websocket("/ws/experiments/{experiment_id}/interpretability")
async def interpretability_websocket(websocket: WebSocket, experiment_id: str):
    await websocket.accept()
    
    ws_key = f"interpretability:{experiment_id}"
    if ws_key not in active_ws_connections:
        active_ws_connections[ws_key] = []
    active_ws_connections[ws_key].append(websocket)
    
    logger.info(f"WebSocket connected for interpretability: {experiment_id}")
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for interpretability: {experiment_id}")
    except Exception as e:
        logger.error(f"WebSocket error for interpretability: {e}")
    finally:
        if ws_key in active_ws_connections:
            if websocket in active_ws_connections[ws_key]:
                active_ws_connections[ws_key].remove(websocket)
            if not active_ws_connections[ws_key]:
                del active_ws_connections[ws_key]
