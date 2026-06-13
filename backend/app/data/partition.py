import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import Dict, List, Tuple, Optional
import random


def get_transforms(dataset_name: str, feature_skew: bool = False, skew_level: float = 0.0):
    if dataset_name == "mnist":
        base = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    elif dataset_name == "fashion_mnist":
        base = [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    elif dataset_name == "cifar10":
        base = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if feature_skew and skew_level > 0:
        angle = skew_level * 30
        base.insert(0, transforms.RandomRotation(degrees=(angle, angle)))

    return transforms.Compose(base)


def get_test_transforms(dataset_name: str):
    if dataset_name == "mnist":
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    elif dataset_name == "fashion_mnist":
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    elif dataset_name == "cifar10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _download_dataset(dataset_name: str, data_dir: str = "./data"):
    if dataset_name == "mnist":
        return datasets.MNIST(data_dir, train=True, download=True), datasets.MNIST(data_dir, train=False, download=True)
    elif dataset_name == "fashion_mnist":
        return datasets.FashionMNIST(data_dir, train=True, download=True), datasets.FashionMNIST(data_dir, train=False, download=True)
    elif dataset_name == "cifar10":
        return datasets.CIFAR10(data_dir, train=True, download=True), datasets.CIFAR10(data_dir, train=False, download=True)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def partition_label_skew(train_dataset, num_clients: int, num_classes_per_client: int) -> Dict[int, List[int]]:
    targets = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    num_classes = len(np.unique(targets))
    indices_by_class = {c: np.where(targets == c)[0].tolist() for c in range(num_classes)}

    for c in indices_by_class:
        random.shuffle(indices_by_class[c])

    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    class_assignments = {}
    for client_id in range(num_clients):
        assigned = random.sample(range(num_classes), min(num_classes_per_client, num_classes))
        class_assignments[client_id] = assigned

    for client_id in range(num_clients):
        for c in class_assignments[client_id]:
            n_per_client_for_class = len(indices_by_class[c]) // num_clients
            start = (client_id % max(1, len(indices_by_class[c]) // max(1, n_per_client_for_class))) * n_per_client_for_class
            end = start + n_per_client_for_class
            client_indices[client_id].extend(indices_by_class[c][start:end])

    return client_indices


def partition_dirichlet(train_dataset, num_clients: int, alpha: float) -> Dict[int, List[int]]:
    targets = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    num_classes = len(np.unique(targets))
    indices_by_class = {c: np.where(targets == c)[0].tolist() for c in range(num_classes)}

    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        random.shuffle(indices_by_class[c])
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(indices_by_class[c])).astype(int)

        diff = len(indices_by_class[c]) - proportions.sum()
        for i in range(abs(diff)):
            proportions[i % num_clients] += 1 if diff > 0 else -1

        start = 0
        for client_id in range(num_clients):
            end = start + proportions[client_id]
            client_indices[client_id].extend(indices_by_class[c][start:end])
            start = end

    return client_indices


def partition_feature_skew(train_dataset, num_clients: int, skew_level: float) -> Dict[int, List[int]]:
    n = len(train_dataset)
    all_indices = list(range(n))
    random.shuffle(all_indices)
    per_client = n // num_clients
    return {i: all_indices[i * per_client:(i + 1) * per_client] for i in range(num_clients)}


class FeatureSkewDataset(Dataset):
    def __init__(self, base_dataset, indices, rotation_degrees: float = 0.0, noise_std: float = 0.0):
        self.base_dataset = base_dataset
        self.indices = indices
        self.rotation_degrees = rotation_degrees
        self.noise_std = noise_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base_dataset[self.indices[idx]]
        if self.rotation_degrees != 0:
            img = transforms.functional.rotate(img, self.rotation_degrees)
        if self.noise_std > 0:
            noise = torch.randn_like(img) * self.noise_std
            img = img + noise
            img = torch.clamp(img, 0.0, 1.0)
        return img, label


class ClientDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


def create_data_partitions(
    dataset_name: str,
    num_clients: int,
    non_iid_type: str,
    non_iid_param: float,
    data_dir: str = "./data",
) -> Tuple[Dict[int, Dataset], Dataset, List[int]]:
    train_dataset, test_dataset = _download_dataset(dataset_name, data_dir)

    if non_iid_type == "label_skew":
        client_indices = partition_label_skew(train_dataset, num_clients, int(non_iid_param))
    elif non_iid_type == "dirichlet":
        client_indices = partition_dirichlet(train_dataset, num_clients, non_iid_param)
    elif non_iid_type == "feature_skew":
        client_indices = partition_feature_skew(train_dataset, num_clients, non_iid_param)
    else:
        raise ValueError(f"Unknown Non-IID type: {non_iid_type}")

    test_transform = get_test_transforms(dataset_name)
    if dataset_name == "mnist":
        test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=test_transform)
    elif dataset_name == "fashion_mnist":
        test_ds = datasets.FashionMNIST(data_dir, train=False, download=True, transform=test_transform)
    elif dataset_name == "cifar10":
        test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    client_datasets: Dict[int, Dataset] = {}
    for client_id, indices in client_indices.items():
        if len(indices) == 0:
            continue
        if non_iid_type == "feature_skew":
            rotation = non_iid_param * 30 * (client_id / max(1, num_clients - 1) - 0.5) * 2
            noise = non_iid_param * 0.1 * (client_id / max(1, num_clients - 1))
            client_datasets[client_id] = FeatureSkewDataset(train_dataset, indices, rotation, noise)
        else:
            client_datasets[client_id] = ClientDataset(train_dataset, indices)

    label_distribution = []
    targets = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    for client_id in sorted(client_indices.keys()):
        client_targets = targets[client_indices[client_id]]
        label_counts = [int((client_targets == c).sum()) for c in range(10)]
        label_distribution.append(label_counts)

    return client_datasets, test_ds, label_distribution
