import asyncio
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json

from app.models.networks import get_model
from app.data.partition import get_test_transforms, create_data_partitions

logger = logging.getLogger(__name__)


class AnalysisMethod(str, Enum):
    GRADIENT = "gradient"
    PERMUTATION = "permutation"
    SHAP = "shap"


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class InterpretabilityService:
    def __init__(self, experiment_id: str, config: dict, redis_client=None):
        self.experiment_id = experiment_id
        self.config = config
        self.redis = redis_client
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dataset_name = config.get("dataset", "mnist")
        self.num_clients = config.get("num_clients", 10)
        self.non_iid_type = config.get("non_iid_type", "label_skew")
        self.non_iid_param = config.get("non_iid_param", 2.0)
        
        self.global_model = get_model(self.dataset_name).to(self.device)
        self.client_datasets = {}
        self.test_dataset = None
        
        self._is_cancelled = False
        self._status = AnalysisStatus.PENDING
        self._progress = 0.0
        self._current_sample = 0
        self._logs: List[Dict] = []
        
        self._callbacks: List[Callable] = []

    def load_global_model(self, state_dict: Dict[str, torch.Tensor]):
        self.global_model.load_state_dict({k: v.to(self.device) for k, v in state_dict.items()})
        self.global_model.eval()

    def setup_data(self):
        logger.info(f"Setting up data for interpretability analysis...")
        self.client_datasets, self.test_dataset, _ = create_data_partitions(
            dataset_name=self.dataset_name,
            num_clients=self.num_clients,
            non_iid_type=self.non_iid_type,
            non_iid_param=self.non_iid_param,
            data_dir="./data",
        )

    def stratified_sample(self, num_samples: int) -> List[int]:
        if self.test_dataset is None:
            self.setup_data()
        
        targets = np.array([self.test_dataset[i][1] for i in range(len(self.test_dataset))])
        num_classes = len(np.unique(targets))
        
        if num_samples < num_classes:
            raise ValueError(
                f"Sample count ({num_samples}) must be at least the number of classes ({num_classes})"
            )
        
        indices_by_class = {c: np.where(targets == c)[0].tolist() for c in range(num_classes)}
        
        for c in indices_by_class:
            np.random.shuffle(indices_by_class[c])
        
        selected_indices = []
        per_class = num_samples // num_classes
        remainder = num_samples % num_classes
        
        for c in range(num_classes):
            count = per_class + (1 if c < remainder else 0)
            count = min(count, len(indices_by_class[c]))
            selected_indices.extend(indices_by_class[c][:count])
        
        if len(selected_indices) < num_samples:
            remaining_needed = num_samples - len(selected_indices)
            all_indices = list(range(len(self.test_dataset)))
            np.random.shuffle(all_indices)
            for idx in all_indices:
                if idx not in selected_indices and len(selected_indices) < num_samples:
                    selected_indices.append(idx)
                if len(selected_indices) >= num_samples:
                    break
        
        np.random.shuffle(selected_indices)
        return selected_indices

    def add_progress_callback(self, callback: Callable):
        self._callbacks.append(callback)

    def _notify_progress(self, progress: float, current_sample: int, log_entry: Optional[Dict] = None):
        self._progress = progress
        self._current_sample = current_sample
        if log_entry:
            self._logs.append(log_entry)
        
        for callback in self._callbacks:
            try:
                callback(progress, current_sample, log_entry)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def cancel(self):
        self._is_cancelled = True
        self._status = AnalysisStatus.CANCELLED

    def _check_cancelled(self) -> bool:
        return self._is_cancelled

    def _get_top_features(self, attribution: np.ndarray, top_k: int = 3) -> List[int]:
        flat = attribution.flatten()
        top_indices = np.argsort(np.abs(flat))[-top_k:][::-1]
        return top_indices.tolist()

    async def analyze(
        self,
        method: AnalysisMethod,
        num_samples: int,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        if self._check_cancelled():
            return {"status": AnalysisStatus.CANCELLED}
        
        self._status = AnalysisStatus.RUNNING
        self._progress = 0.0
        self._current_sample = 0
        self._logs = []
        
        try:
            sample_indices = self.stratified_sample(num_samples)
            logger.info(f"Starting {method.value} analysis with {num_samples} samples")
            
            if method == AnalysisMethod.GRADIENT:
                result = await self._gradient_attribution(sample_indices, batch_size)
            elif method == AnalysisMethod.PERMUTATION:
                result = await self._permutation_importance(sample_indices, batch_size)
            elif method == AnalysisMethod.SHAP:
                result = await self._shap_approximation(sample_indices, batch_size)
            else:
                raise ValueError(f"Unknown analysis method: {method}")
            
            if self._check_cancelled():
                return {"status": AnalysisStatus.CANCELLED}
            
            client_contributions = self._compute_client_contributions(sample_indices)
            result["client_contributions"] = client_contributions
            
            result["status"] = AnalysisStatus.COMPLETED
            self._status = AnalysisStatus.COMPLETED
            self._progress = 100.0
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            self._status = AnalysisStatus.FAILED
            return {"status": AnalysisStatus.FAILED, "error": str(e)}

    async def _gradient_attribution(
        self,
        sample_indices: List[int],
        batch_size: int
    ) -> Dict[str, Any]:
        if self.test_dataset is None:
            self.setup_data()
        
        num_samples = len(sample_indices)
        input_shape = (1, 28, 28) if self.dataset_name in ("mnist", "fashion_mnist") else (3, 32, 32)
        num_classes = 10
        
        overall_attribution = np.zeros(input_shape, dtype=np.float32)
        class_attributions = np.zeros((num_classes,) + input_shape, dtype=np.float32)
        class_sample_counts = np.zeros(num_classes, dtype=np.int32)
        
        total_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            if self._check_cancelled():
                return {"status": AnalysisStatus.CANCELLED}
            
            start = batch_idx * batch_size
            end = min(start + batch_size, num_samples)
            batch_sample_indices = sample_indices[start:end]
            actual_batch_size = len(batch_sample_indices)
            
            batch_start_time = time.time()
            
            batch_data = []
            batch_labels = []
            for idx in batch_sample_indices:
                data, label = self.test_dataset[idx]
                batch_data.append(data)
                batch_labels.append(label)
            
            batch_tensor = torch.stack(batch_data).to(self.device)
            batch_tensor.requires_grad_(True)
            labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=self.device)
            
            self.global_model.zero_grad()
            outputs = self.global_model(batch_tensor)
            
            target_scores = outputs.gather(1, labels_tensor.unsqueeze(1)).squeeze()
            target_scores.sum().backward()
            
            gradients = batch_tensor.grad.detach().cpu().numpy()
            
            batch_attributions = np.abs(gradients)
            
            for i in range(actual_batch_size):
                label = batch_labels[i]
                overall_attribution += batch_attributions[i]
                class_attributions[label] += batch_attributions[i]
                class_sample_counts[label] += 1
            
            non_zero_counts = class_sample_counts.copy()
            non_zero_counts[non_zero_counts == 0] = 1
            
            mean_overall = overall_attribution / max(end, 1)
            mean_class = class_attributions / non_zero_counts[:, np.newaxis, np.newaxis, np.newaxis]
            
            batch_top_features = self._get_top_features(mean_overall, 3)
            batch_time = time.time() - batch_start_time
            
            progress = (end / num_samples) * 100
            log_entry = {
                "batch": batch_idx + 1,
                "total_batches": total_batches,
                "sample_start": start + 1,
                "sample_end": end,
                "batch_time_ms": round(batch_time * 1000, 2),
                "top_features": batch_top_features,
                "timestamp": time.time()
            }
            
            self._notify_progress(progress, end, log_entry)
            await asyncio.sleep(0.01)
        
        overall_attribution = overall_attribution / max(num_samples, 1)
        for c in range(num_classes):
            if class_sample_counts[c] > 0:
                class_attributions[c] = class_attributions[c] / class_sample_counts[c]
        
        overall_normalized = self._normalize_to_percentage(overall_attribution)
        class_normalized = np.array([
            self._normalize_to_percentage(class_attributions[c]) 
            for c in range(num_classes)
        ])
        
        return {
            "method": AnalysisMethod.GRADIENT.value,
            "num_samples": num_samples,
            "overall_attribution": overall_normalized.tolist(),
            "class_attributions": class_normalized.tolist(),
            "class_sample_counts": class_sample_counts.tolist(),
            "logs": self._logs
        }

    async def _permutation_importance(
        self,
        sample_indices: List[int],
        batch_size: int
    ) -> Dict[str, Any]:
        if self.test_dataset is None:
            self.setup_data()
        
        num_samples = len(sample_indices)
        input_shape = (1, 28, 28) if self.dataset_name in ("mnist", "fashion_mnist") else (3, 32, 32)
        num_classes = 10
        num_features = input_shape[1] * input_shape[2]
        
        baseline_loader = DataLoader(
            Subset(self.test_dataset, sample_indices),
            batch_size=batch_size,
            shuffle=False
        )
        
        baseline_accuracy = self._evaluate_batch_accuracy(baseline_loader)
        
        overall_importance = np.zeros(input_shape, dtype=np.float32)
        class_importance = np.zeros((num_classes,) + input_shape, dtype=np.float32)
        class_sample_counts = np.zeros(num_classes, dtype=np.int32)
        
        flat_indices = list(range(num_features))
        np.random.shuffle(flat_indices)
        
        total_features = len(flat_indices)
        total_batches = total_features
        
        batch_idx = 0
        processed = 0
        
        for flat_idx in flat_indices:
            if self._check_cancelled():
                return {"status": AnalysisStatus.CANCELLED}
            
            batch_start_time = time.time()
            
            x = flat_idx // input_shape[2]
            y = flat_idx % input_shape[2]
            
            permuted_dataset = self._create_permuted_dataset(
                sample_indices, channel=0, x=x, y=y
            )
            permuted_loader = DataLoader(
                permuted_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            
            permuted_accuracy = self._evaluate_batch_accuracy(permuted_loader)
            accuracy_drop = baseline_accuracy - permuted_accuracy
            
            overall_importance[0, x, y] = accuracy_drop
            
            class_drops = self._evaluate_class_accuracy_drop(
                sample_indices, channel=0, x=x, y=y, num_classes=num_classes
            )
            
            for c in range(num_classes):
                class_importance[c, 0, x, y] = class_drops[c]
            
            class_sample_counts += self._get_class_counts(sample_indices, num_classes)
            
            batch_top_features = self._get_top_features(overall_importance, 3)
            batch_time = time.time() - batch_start_time
            
            processed += 1
            progress = (processed / total_features) * 100
            log_entry = {
                "batch": batch_idx + 1,
                "total_batches": total_batches,
                "feature_index": flat_idx,
                "feature_coord": [x, y],
                "accuracy_drop": round(accuracy_drop, 6),
                "batch_time_ms": round(batch_time * 1000, 2),
                "top_features": batch_top_features,
                "timestamp": time.time()
            }
            
            self._notify_progress(progress, processed, log_entry)
            batch_idx += 1
            await asyncio.sleep(0.01)
        
        overall_normalized = self._normalize_to_percentage(overall_importance)
        class_normalized = np.array([
            self._normalize_to_percentage(class_importance[c]) 
            for c in range(num_classes)
        ])
        
        return {
            "method": AnalysisMethod.PERMUTATION.value,
            "num_samples": num_samples,
            "baseline_accuracy": baseline_accuracy,
            "overall_attribution": overall_normalized.tolist(),
            "class_attributions": class_normalized.tolist(),
            "class_sample_counts": class_sample_counts.tolist(),
            "logs": self._logs
        }

    async def _shap_approximation(
        self,
        sample_indices: List[int],
        batch_size: int,
        num_perturbations: int = 50
    ) -> Dict[str, Any]:
        if self.test_dataset is None:
            self.setup_data()
        
        num_samples = len(sample_indices)
        input_shape = (1, 28, 28) if self.dataset_name in ("mnist", "fashion_mnist") else (3, 32, 32)
        num_classes = 10
        num_pixels = input_shape[1] * input_shape[2]
        
        overall_shap = np.zeros(input_shape, dtype=np.float32)
        class_shap = np.zeros((num_classes,) + input_shape, dtype=np.float32)
        class_sample_counts = np.zeros(num_classes, dtype=np.int32)
        
        total_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            if self._check_cancelled():
                return {"status": AnalysisStatus.CANCELLED}
            
            start = batch_idx * batch_size
            end = min(start + batch_size, num_samples)
            batch_sample_indices = sample_indices[start:end]
            actual_batch_size = len(batch_sample_indices)
            
            batch_start_time = time.time()
            
            for i, sample_idx in enumerate(batch_sample_indices):
                if self._check_cancelled():
                    return {"status": AnalysisStatus.CANCELLED}
                
                original_data, label = self.test_dataset[sample_idx]
                original_data = original_data.to(self.device)
                
                class_sample_counts[label] += 1
                
                with torch.no_grad():
                    original_output = self.global_model(original_data.unsqueeze(0))
                    original_pred = original_output.argmax(dim=1).item()
                    original_prob = torch.softmax(original_output, dim=1)[0, original_pred].item()
                
                for pixel_idx in range(num_pixels):
                    x = pixel_idx // input_shape[2]
                    y = pixel_idx % input_shape[2]
                    
                    total_contribution = 0.0
                    valid_perturbations = 0
                    
                    for _ in range(num_perturbations):
                        perm = np.random.permutation(num_pixels)
                        feature_pos = np.where(perm == pixel_idx)[0][0]
                        
                        mask = torch.zeros(input_shape, dtype=torch.bool, device=self.device)
                        for j in range(feature_pos):
                            px = perm[j] // input_shape[2]
                            py = perm[j] % input_shape[2]
                            mask[0, px, py] = True
                        
                        mask[0, x, y] = True
                        perturbed_with = original_data.clone()
                        mean_val = original_data.mean().item()
                        perturbed_with[~mask] = mean_val
                        
                        mask_without = mask.clone()
                        mask_without[0, x, y] = False
                        perturbed_without = original_data.clone()
                        perturbed_without[~mask_without] = mean_val
                        
                        with torch.no_grad():
                            output_with = self.global_model(perturbed_with.unsqueeze(0))
                            prob_with = torch.softmax(output_with, dim=1)[0, original_pred].item()
                            
                            output_without = self.global_model(perturbed_without.unsqueeze(0))
                            prob_without = torch.softmax(output_without, dim=1)[0, original_pred].item()
                        
                        contribution = prob_with - prob_without
                        total_contribution += contribution
                        valid_perturbations += 1
                    
                    if valid_perturbations > 0:
                        avg_contribution = total_contribution / valid_perturbations
                        overall_shap[0, x, y] += avg_contribution
                        class_shap[label, 0, x, y] += avg_contribution
            
            non_zero_counts = class_sample_counts.copy()
            non_zero_counts[non_zero_counts == 0] = 1
            
            mean_overall = overall_shap / max(end, 1)
            batch_top_features = self._get_top_features(mean_overall, 3)
            batch_time = time.time() - batch_start_time
            
            progress = (end / num_samples) * 100
            log_entry = {
                "batch": batch_idx + 1,
                "total_batches": total_batches,
                "sample_start": start + 1,
                "sample_end": end,
                "num_perturbations_per_feature": num_perturbations,
                "batch_time_ms": round(batch_time * 1000, 2),
                "top_features": batch_top_features,
                "timestamp": time.time()
            }
            
            self._notify_progress(progress, end, log_entry)
            await asyncio.sleep(0.01)
        
        overall_shap = overall_shap / max(num_samples, 1)
        for c in range(num_classes):
            if class_sample_counts[c] > 0:
                class_shap[c] = class_shap[c] / class_sample_counts[c]
        
        overall_normalized = self._normalize_to_percentage(np.abs(overall_shap))
        class_normalized = np.array([
            self._normalize_to_percentage(np.abs(class_shap[c])) 
            for c in range(num_classes)
        ])
        
        return {
            "method": AnalysisMethod.SHAP.value,
            "num_samples": num_samples,
            "num_perturbations": num_perturbations,
            "overall_attribution": overall_normalized.tolist(),
            "class_attributions": class_normalized.tolist(),
            "class_sample_counts": class_sample_counts.tolist(),
            "logs": self._logs
        }

    def _evaluate_batch_accuracy(self, loader: DataLoader) -> float:
        self.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return correct / max(total, 1)

    def _evaluate_class_accuracy_drop(
        self,
        sample_indices: List[int],
        channel: int,
        x: int,
        y: int,
        num_classes: int
    ) -> np.ndarray:
        class_correct = np.zeros(num_classes, dtype=np.int32)
        class_total = np.zeros(num_classes, dtype=np.int32)
        class_correct_permuted = np.zeros(num_classes, dtype=np.int32)
        
        for idx in sample_indices:
            data, label = self.test_dataset[idx]
            class_total[label] += 1
            
            with torch.no_grad():
                output = self.global_model(data.unsqueeze(0).to(self.device))
                pred = output.argmax(dim=1).item()
                if pred == label:
                    class_correct[label] += 1
                
                permuted_data = data.clone()
                permuted_data[channel, x, y] = torch.randn_like(permuted_data[channel, x, y])
                output_permuted = self.global_model(permuted_data.unsqueeze(0).to(self.device))
                pred_permuted = output_permuted.argmax(dim=1).item()
                if pred_permuted == label:
                    class_correct_permuted[label] += 1
        
        class_acc = class_correct / np.maximum(class_total, 1)
        class_acc_permuted = class_correct_permuted / np.maximum(class_total, 1)
        class_drops = class_acc - class_acc_permuted
        
        return class_drops

    def _create_permuted_dataset(
        self,
        sample_indices: List[int],
        channel: int,
        x: int,
        y: int
    ) -> Subset:
        class PermutedDataset(torch.utils.data.Dataset):
            def __init__(self, base_dataset, indices, channel, x, y):
                self.base_dataset = base_dataset
                self.indices = indices
                self.channel = channel
                self.x = x
                self.y = y
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                data, label = self.base_dataset[self.indices[idx]]
                data = data.clone()
                data[self.channel, self.x, self.y] = torch.randn_like(data[self.channel, self.x, self.y])
                return data, label
        
        return PermutedDataset(self.test_dataset, sample_indices, channel, x, y)

    def _get_class_counts(self, sample_indices: List[int], num_classes: int) -> np.ndarray:
        counts = np.zeros(num_classes, dtype=np.int32)
        for idx in sample_indices:
            _, label = self.test_dataset[idx]
            counts[label] += 1
        return counts

    def _normalize_to_percentage(self, arr: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(arr))
        if max_val > 0:
            normalized = arr / max_val * 100
            return np.clip(normalized, 0, 100)
        return np.zeros_like(arr)

    def _compute_client_contributions(self, sample_indices: List[int]) -> Dict[str, Any]:
        if self.test_dataset is None:
            self.setup_data()
        
        input_shape = (1, 28, 28) if self.dataset_name in ("mnist", "fashion_mnist") else (3, 32, 32)
        num_pixels = input_shape[1] * input_shape[2]
        num_clients = self.num_clients
        
        global_state = {k: v.cpu() for k, v in self.global_model.state_dict().items()}
        
        client_model_diffs = []
        client_names = []
        
        for client_id in range(num_clients):
            if client_id in self.client_datasets:
                client_model = get_model(self.dataset_name).to(self.device)
                client_model.load_state_dict({k: v.to(self.device) for k, v in global_state.items()})
                
                client_loader = DataLoader(
                    self.client_datasets[client_id],
                    batch_size=32,
                    shuffle=False
                )
                
                optimizer = torch.optim.SGD(client_model.parameters(), lr=0.01)
                criterion = nn.CrossEntropyLoss()
                
                client_model.train()
                for data, target in client_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = client_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    break
                
                client_state = {k: v.cpu() for k, v in client_model.state_dict().items()}
                
                diff = {}
                for key in global_state:
                    diff[key] = (client_state[key] - global_state[key]).abs().mean().item()
                
                avg_diff = np.mean(list(diff.values()))
                client_model_diffs.append(avg_diff)
                client_names.append(f"Client {client_id}")
        
        total_diff = sum(client_model_diffs) if client_model_diffs else 1
        client_weights = [d / total_diff for d in client_model_diffs] if total_diff > 0 else [1.0 / len(client_model_diffs)] * len(client_model_diffs)
        
        num_top_features = min(10, num_pixels)
        overall_importance = np.zeros(num_pixels, dtype=np.float32)
        
        sample_loader = DataLoader(
            Subset(self.test_dataset, sample_indices),
            batch_size=32,
            shuffle=False
        )
        
        for data, target in sample_loader:
            data = data.to(self.device)
            data.requires_grad_(True)
            self.global_model.zero_grad()
            output = self.global_model(data)
            output.sum().backward()
            gradients = data.grad.detach().cpu().numpy()
            overall_importance += np.abs(gradients).sum(axis=0).flatten()
        
        top_feature_indices = np.argsort(overall_importance)[-num_top_features:][::-1]
        top_feature_coords = [(i // input_shape[2], i % input_shape[2]) for i in top_feature_indices]
        
        feature_importance = overall_importance[top_feature_indices]
        feature_importance = feature_importance / feature_importance.sum() * 100
        
        client_feature_contributions = []
        for i, feat_idx in enumerate(top_feature_indices):
            x, y = top_feature_coords[i]
            contributions = []
            for j, client_id in enumerate(range(num_clients)):
                if client_id < len(client_weights):
                    weight = client_weights[client_id] * feature_importance[i]
                    contributions.append({
                        "client": f"Client {client_id}",
                        "value": round(weight, 4)
                    })
            client_feature_contributions.append({
                "feature_index": int(feat_idx),
                "feature_coord": [x, y],
                "importance": round(feature_importance[i], 4),
                "client_contributions": contributions
            })
        
        return {
            "client_names": client_names,
            "client_weights": [round(w, 6) for w in client_weights],
            "top_features": [
                {
                    "index": int(top_feature_indices[i]),
                    "coord": [top_feature_coords[i][0], top_feature_coords[i][1]],
                    "importance": round(feature_importance[i], 4),
                    "client_values": [
                        round(client_weights[j] * feature_importance[i], 4)
                        for j in range(len(client_weights))
                    ]
                }
                for i in range(num_top_features)
            ]
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": self._status.value,
            "progress": self._progress,
            "current_sample": self._current_sample,
            "logs": self._logs
        }
