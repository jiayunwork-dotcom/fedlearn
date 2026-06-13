import copy
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Optional, Any
import asyncio
import json

from app.models.networks import get_model
from app.data.partition import create_data_partitions
from app.aggregation.strategies import get_aggregator, ScaffoldAggregator
from app.privacy.mechanisms import DPSGD, SecureAggregation
from app.attack.simulation import (
    DataPoisoningAttack,
    ModelPoisoningAttack,
    BackdoorAttack,
    get_defense,
)
from app.selection.strategies import get_selector, ContributionSelector

logger = logging.getLogger(__name__)


class ExperimentConfig:
    def __init__(self, **kwargs):
        self.experiment_id = kwargs.get("experiment_id", "")
        self.dataset = kwargs.get("dataset", "mnist")
        self.num_clients = kwargs.get("num_clients", 10)
        self.client_fraction = kwargs.get("client_fraction", 0.4)
        self.local_epochs = kwargs.get("local_epochs", 5)
        self.global_rounds = kwargs.get("global_rounds", 200)
        self.early_stop_patience = kwargs.get("early_stop_patience", 10)
        self.aggregation_strategy = kwargs.get("aggregation_strategy", "fedavg")
        self.fedprox_mu = kwargs.get("fedprox_mu", 0.01)
        self.non_iid_type = kwargs.get("non_iid_type", "label_skew")
        self.non_iid_param = kwargs.get("non_iid_param", 2.0)
        self.learning_rate = kwargs.get("learning_rate", 0.01)
        self.batch_size = kwargs.get("batch_size", 32)
        self.client_selection = kwargs.get("client_selection", "random")
        self.dp_enabled = kwargs.get("dp_enabled", False)
        self.dp_clip_bound = kwargs.get("dp_clip_bound", 1.0)
        self.dp_noise_multiplier = kwargs.get("dp_noise_multiplier", 1.0)
        self.dp_delta = kwargs.get("dp_delta", 1e-5)
        self.dp_epsilon_budget = kwargs.get("dp_epsilon_budget", 10.0)
        self.secure_aggregation = kwargs.get("secure_aggregation", False)
        self.secure_agg_threshold = kwargs.get("secure_agg_threshold", None)
        self.drop_rate = kwargs.get("drop_rate", 0.0)
        self.attack_type = kwargs.get("attack_type", "none")
        self.attack_ratio = kwargs.get("attack_ratio", 0.0)
        self.defense_type = kwargs.get("defense_type", "none")
        self.defense_beta = kwargs.get("defense_beta", 0.2)


class FederatedTrainer:
    def __init__(self, config: ExperimentConfig, redis_client=None):
        self.config = config
        self.redis = redis_client
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.global_model = get_model(config.dataset).to(self.device)
        self.aggregator = get_aggregator(
            config.aggregation_strategy, mu=config.fedprox_mu
        )

        self.selector = get_selector(config.client_selection)

        self.dp_sgd = None
        if config.dp_enabled:
            self.dp_sgd = DPSGD(
                clip_bound=config.dp_clip_bound,
                noise_multiplier=config.dp_noise_multiplier,
                delta=config.dp_delta,
            )

        self.secure_agg = None
        if config.secure_aggregation:
            self.secure_agg = SecureAggregation(
                config.num_clients, config.secure_agg_threshold
            )

        self.client_datasets = {}
        self.test_dataset = None
        self.label_distribution = []

        self.current_round = 0
        self.best_accuracy = 0.0
        self.patience_counter = 0
        self.is_running = False
        self.is_stopped = False

        self.metrics_history: List[Dict] = []
        self.client_contributions: Dict[int, float] = {}
        self.attack_log: List[Dict] = []

        self.client_control_variates: Dict[int, Dict] = {}
        if config.aggregation_strategy == "scaffold":
            if isinstance(self.aggregator, ScaffoldAggregator):
                self.aggregator.initialize_control(self.global_model)

        self.byzyantine_clients: List[int] = []

    def setup_data(self):
        logger.info(f"Setting up {self.config.dataset} with {self.config.non_iid_type} partitioning...")
        self.client_datasets, self.test_dataset, self.label_distribution = create_data_partitions(
            dataset_name=self.config.dataset,
            num_clients=self.config.num_clients,
            non_iid_type=self.config.non_iid_type,
            non_iid_param=self.config.non_iid_param,
        )

        if self.config.attack_type != "none" and self.config.attack_ratio > 0:
            num_byzantine = max(1, int(self.config.num_clients * self.config.attack_ratio))
            self.byzyantine_clients = list(range(self.config.num_clients - num_byzantine, self.config.num_clients))
            logger.info(f"Byzantine clients: {self.byzyantine_clients}")

    def _get_client_dataloader(self, client_id: int) -> DataLoader:
        dataset = self.client_datasets.get(client_id)
        if dataset is None:
            return None
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

    def _evaluate_model(self, model: nn.Module, dataloader: DataLoader) -> tuple:
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss(reduction="sum")

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                total_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        accuracy = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)
        return accuracy, avg_loss

    def _train_client(
        self, client_id: int, global_state: Dict[str, torch.Tensor], round_num: int
    ) -> Dict:
        model = get_model(self.config.dataset).to(self.device)
        model.load_state_dict({k: v.to(self.device) for k, v in global_state.items()})

        dataloader = self._get_client_dataloader(client_id)
        if dataloader is None:
            return None

        is_byzantine = client_id in self.byzyantine_clients

        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        local_steps = 0

        for epoch in range(self.config.local_epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)

                if is_byzantine and self.config.attack_type == "backdoor":
                    attack = BackdoorAttack()
                    data, target = attack.inject_backdoor(data, target)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)

                if self.config.aggregation_strategy == "fedprox":
                    prox_term = 0.0
                    for name, param in model.named_parameters():
                        prox_term += ((param - global_state[name].to(self.device)) ** 2).sum()
                    loss += (self.config.fedprox_mu / 2) * prox_term

                loss.backward()

                if self.dp_sgd:
                    self.dp_sgd.clip_gradients(model)
                    self.dp_sgd.add_noise(model)

                optimizer.step()
                local_steps += 1

        model_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if is_byzantine:
            if self.config.attack_type == "model_poisoning":
                attack = ModelPoisoningAttack()
                model_state = attack.poison_update(model_state)

        result = {
            "client_id": client_id,
            "model_state": model_state,
            "local_steps": local_steps,
            "num_samples": len(dataloader.dataset),
        }

        if self.config.aggregation_strategy == "scaffold":
            if client_id not in self.client_control_variates:
                self.client_control_variates[client_id] = {
                    k: torch.zeros_like(v) for k, v in global_state.items()
                }
            result["control_variate"] = self.client_control_variates[client_id]

        if self.dp_sgd:
            sampling_rate = self.config.batch_size / max(len(dataloader.dataset), 1)
            self.dp_sgd.accumulate_privacy_cost(sampling_rate, local_steps)

        return result

    async def run_training(self):
        self.is_running = True
        self.setup_data()

        if self.redis:
            await self._save_label_distribution()

        test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False)

        start_time = time.time()

        for round_num in range(1, self.config.global_rounds + 1):
            if self.is_stopped:
                break

            self.current_round = round_num
            round_start = time.time()

            all_client_ids = list(range(self.config.num_clients))
            selected_ids = self.selector.select(
                all_client_ids, self.config.client_fraction, round_num
            )

            global_state = {k: v.cpu() for k, v in self.global_model.state_dict().items()}

            client_updates = []
            active_ids = list(selected_ids)

            if self.config.drop_rate > 0:
                num_drop = int(len(active_ids) * self.config.drop_rate)
                if num_drop > 0 and len(active_ids) > num_drop:
                    dropped = set(np.random.choice(active_ids, size=num_drop, replace=False))
                    active_ids = [cid for cid in active_ids if cid not in dropped]

            for client_id in active_ids:
                update = self._train_client(client_id, global_state, round_num)
                if update is not None:
                    client_updates.append(update)

            if not client_updates:
                logger.warning(f"Round {round_num}: No client updates received")
                continue

            client_weights = [u["num_samples"] for u in client_updates]

            if self.config.defense_type != "none":
                defense = get_defense(
                    self.config.defense_type,
                    beta=self.config.defense_beta,
                    num_byzantine=len(self.byzyantine_clients),
                )
                if defense is not None:
                    new_state = defense.aggregate(self.global_model, client_updates, client_weights)
                    self.global_model.load_state_dict(new_state)
                else:
                    new_state = self.aggregator.aggregate(self.global_model, client_updates, client_weights)
                    self.global_model.load_state_dict(new_state)
            else:
                new_state = self.aggregator.aggregate(self.global_model, client_updates, client_weights)
                self.global_model.load_state_dict(new_state)

            if self.config.secure_aggregation and self.secure_agg:
                pass

            accuracy, loss = self._evaluate_model(self.global_model, test_loader)

            client_accuracies = {}
            for update in client_updates:
                cid = update["client_id"]
                client_model = get_model(self.config.dataset).to(self.device)
                client_model.load_state_dict(
                    {k: v.to(self.device) for k, v in update["model_state"].items()}
                )
                client_loader = DataLoader(
                    self.client_datasets.get(cid, self.test_dataset),
                    batch_size=self.config.batch_size,
                    shuffle=False,
                )
                if client_loader.dataset is not None and len(client_loader.dataset) > 0:
                    acc, _ = self._evaluate_model(client_model, client_loader)
                    client_accuracies[cid] = acc

            similarity = self._compute_client_similarity(client_updates)

            epsilon = 0.0
            if self.dp_sgd:
                epsilon = self.dp_sgd.get_epsilon()
                if epsilon > self.config.dp_epsilon_budget:
                    logger.info(f"Privacy budget exhausted: epsilon={epsilon:.4f} > {self.config.dp_epsilon_budget}")
                    self.attack_log.append({
                        "round": round_num,
                        "type": "privacy_budget_exhausted",
                        "message": f"epsilon={epsilon:.4f} exceeded budget {self.config.dp_epsilon_budget}",
                    })
                    break

            if accuracy > self.best_accuracy:
                if isinstance(self.selector, ContributionSelector):
                    improvement = accuracy - self.best_accuracy
                    for update in client_updates:
                        self.selector.update_contributions(update["client_id"], improvement)
                self.best_accuracy = accuracy
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.early_stop_patience:
                logger.info(f"Early stopping at round {round_num}")
                break

            round_metrics = {
                "round": round_num,
                "global_accuracy": accuracy,
                "global_loss": loss,
                "client_accuracies": client_accuracies,
                "client_similarity": similarity,
                "epsilon": epsilon,
                "num_active_clients": len(active_ids),
                "elapsed_seconds": time.time() - start_time,
                "round_seconds": time.time() - round_start,
            }
            self.metrics_history.append(round_metrics)

            if self._detect_anomaly(client_updates, round_num):
                self.attack_log.append({
                    "round": round_num,
                    "type": "anomaly_detected",
                    "message": f"Anomalous update detected among {len(client_updates)} clients",
                })

            self.client_contributions = {}
            if isinstance(self.selector, ContributionSelector):
                self.client_contributions = dict(self.selector.contributions)
            else:
                for cid in range(self.config.num_clients):
                    self.client_contributions[cid] = self.client_contributions.get(cid, 0.0)

            if self.redis:
                await self._publish_metrics(round_metrics)

            logger.info(
                f"Round {round_num}: accuracy={accuracy:.4f}, loss={loss:.4f}, "
                f"epsilon={epsilon:.4f}, clients={len(active_ids)}"
            )

            await asyncio.sleep(0.01)

        self.is_running = False

    def _compute_client_similarity(self, client_updates: List[Dict]) -> float:
        if len(client_updates) < 2:
            return 1.0

        flat_updates = []
        for update in client_updates:
            flat = torch.cat([v.flatten().float() for v in update["model_state"].values()])
            flat_updates.append(flat)

        similarities = []
        for i in range(len(flat_updates)):
            for j in range(i + 1, len(flat_updates)):
                cos = torch.nn.functional.cosine_similarity(
                    flat_updates[i].unsqueeze(0), flat_updates[j].unsqueeze(0)
                ).item()
                similarities.append(cos)

        return np.mean(similarities) if similarities else 1.0

    def _detect_anomaly(self, client_updates: List[Dict], round_num: int) -> bool:
        if len(client_updates) < 3:
            return False

        flat_updates = []
        for update in client_updates:
            flat = torch.cat([v.flatten().float() for v in update["model_state"].values()])
            flat_updates.append(flat.norm().item())

        norms = np.array(flat_updates)
        mean_norm = norms.mean()
        std_norm = norms.std()

        if std_norm == 0:
            return False

        z_scores = np.abs((norms - mean_norm) / std_norm)
        return bool(np.any(z_scores > 3.0))

    async def _publish_metrics(self, metrics: Dict):
        try:
            channel = f"experiment:{self.config.experiment_id}:metrics"
            await self.redis.publish(channel, json.dumps(metrics))
            key = f"experiment:{self.config.experiment_id}:state"
            state = {
                "current_round": self.current_round,
                "global_rounds": self.config.global_rounds,
                "best_accuracy": self.best_accuracy,
                "is_running": self.is_running,
                "epsilon": metrics.get("epsilon", 0),
            }
            await self.redis.set(key, json.dumps(state))
            history_key = f"experiment:{self.config.experiment_id}:history"
            await self.redis.rpush(history_key, json.dumps(metrics))

            contrib_key = f"experiment:{self.config.experiment_id}:contributions"
            await self.redis.set(contrib_key, json.dumps(self.client_contributions))

            if self.attack_log:
                attack_key = f"experiment:{self.config.experiment_id}:attack_log"
                await self.redis.set(attack_key, json.dumps(self.attack_log))
        except Exception as e:
            logger.error(f"Redis publish error: {e}")

    async def _save_label_distribution(self):
        try:
            key = f"experiment:{self.config.experiment_id}:label_dist"
            await self.redis.set(key, json.dumps(self.label_distribution))
        except Exception as e:
            logger.error(f"Redis save label dist error: {e}")

    def stop(self):
        self.is_stopped = True

    def get_status(self) -> Dict:
        return {
            "experiment_id": self.config.experiment_id,
            "current_round": self.current_round,
            "global_rounds": self.config.global_rounds,
            "best_accuracy": self.best_accuracy,
            "is_running": self.is_running,
            "aggregation_strategy": self.config.aggregation_strategy,
            "dataset": self.config.dataset,
            "num_clients": self.config.num_clients,
        }

    def get_all_metrics(self) -> List[Dict]:
        return self.metrics_history

    def get_contribution_ranking(self) -> Dict[int, float]:
        return self.client_contributions

    def get_attack_log(self) -> List[Dict]:
        return self.attack_log
