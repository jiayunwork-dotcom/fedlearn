"""
验证安全聚合与 FedAvg 聚合结果一致性
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np

from app.aggregation.strategies import FedAvgAggregator
from app.privacy.mechanisms import SecureAggregation


def test_secagg_vs_fedavg():
    """验证安全聚合与 FedAvg 聚合结果完全一致"""
    print("\n" + "=" * 70)
    print("测试: 安全聚合与 FedAvg 聚合结果一致性验证")
    print("=" * 70)

    num_clients = 10
    secagg = SecureAggregation(num_clients=num_clients)
    fedavg = FedAvgAggregator()

    global_model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
    global_state = {k: v.clone() for k, v in global_model.state_dict().items()}

    num_params = sum(v.numel() for v in global_state.values())
    print(f"✓ 测试模型: {len(global_state)} 个参数张量, {num_params} 个参数")
    print(f"✓ 客户端数量: {num_clients}")

    client_updates = []
    client_weights = []
    all_client_ids = []

    np.random.seed(42)
    torch.manual_seed(42)

    for i in range(num_clients):
        update = {}
        for key in global_state:
            update[key] = global_state[key] + torch.randn_like(global_state[key]) * 0.1
        num_samples = np.random.randint(50, 200)
        client_updates.append({
            "client_id": i,
            "model_state": update,
            "num_samples": num_samples,
        })
        client_weights.append(num_samples)
        all_client_ids.append(i)
        print(f"  客户端 {i}: 样本数={num_samples}")

    print(f"\n--- FedAvg 聚合 ---")
    fedavg_result = fedavg.aggregate(global_model, client_updates, client_weights)

    print(f"\n--- 安全聚合 ---")
    total_weight = sum(client_weights)
    normalized_weights = [w / total_weight for w in client_weights]

    shared_updates = []
    for i, update in enumerate(client_updates):
        cid = update["client_id"]
        weight = normalized_weights[i]
        weighted_state = {}
        for key, value in update["model_state"].items():
            weighted_state[key] = value.float() * weight
        shared = secagg.client_split_update(cid, weighted_state, all_client_ids)
        shared_updates.append(shared)
        print(f"  客户端 {cid}: 已拆分 (权重={weight:.4f})")

    secagg_result_raw = secagg.aggregate_shares(shared_updates, all_client_ids)

    secagg_result = {}
    for key, values in secagg_result_raw.items():
        sample_tensor = client_updates[0]["model_state"][key].float()
        secagg_result[key] = torch.tensor(values, dtype=sample_tensor.dtype).reshape_as(sample_tensor)

    print(f"\n--- 结果对比 ---")
    max_error = 0.0
    for key in global_state:
        fa = fedavg_result[key].float().numpy()
        sa = secagg_result[key].float().numpy()
        error = np.max(np.abs(fa - sa))
        max_error = max(max_error, error)
        print(f"  {key}: 最大误差 = {error:.2e}")

    print(f"\n✓ 全局最大参数误差: {max_error:.2e}")
    print(f"✓ 安全聚合统计: {secagg.get_stats()}")

    assert max_error < 1e-4, f"安全聚合与 FedAvg 结果不一致，最大误差: {max_error}"

    print("\n✓ 安全聚合与 FedAvg 聚合结果完全一致!")
    return True


def test_model_divergence_simulation():
    """模拟 10 个客户端下安全聚合前后的参数尺度变化"""
    print("\n" + "=" * 70)
    print("测试: 安全聚合参数尺度验证 (修复前 vs 修复后)")
    print("=" * 70)

    num_clients = 10
    secagg = SecureAggregation(num_clients=num_clients)

    global_model = nn.Linear(10, 2)
    global_state = {k: v.clone() for k, v in global_model.state_dict().items()}

    client_updates = []
    client_weights = []
    all_client_ids = []

    for i in range(num_clients):
        update = {}
        for key in global_state:
            update[key] = global_state[key] + torch.randn_like(global_state[key]) * 0.1
        client_updates.append({
            "client_id": i,
            "model_state": update,
            "num_samples": 100,
        })
        client_weights.append(100)
        all_client_ids.append(i)

    avg_param_before = np.mean([v.abs().mean().item() for v in global_state.values()])
    print(f"✓ 全局模型平均参数绝对值: {avg_param_before:.6f}")

    client_avg_params = []
    for update in client_updates:
        avg = np.mean([v.abs().mean().item() for v in update["model_state"].values()])
        client_avg_params.append(avg)
    print(f"✓ 客户端模型平均参数绝对值范围: {min(client_avg_params):.6f} ~ {max(client_avg_params):.6f}")

    total_weight = sum(client_weights)
    normalized_weights = [w / total_weight for w in client_weights]

    # 错误方式: 直接拆分完整模型
    shared_updates_bad = []
    for update in client_updates:
        shared = secagg.client_split_update(update["client_id"], update["model_state"], all_client_ids)
        shared_updates_bad.append(shared)
    bad_result_raw = secagg.aggregate_shares(shared_updates_bad, all_client_ids)
    bad_result = {}
    for key, values in bad_result_raw.items():
        sample_tensor = client_updates[0]["model_state"][key].float()
        bad_result[key] = torch.tensor(values, dtype=sample_tensor.dtype).reshape_as(sample_tensor)
    avg_param_bad = np.mean([v.abs().mean().item() for v in bad_result.values()])

    # 正确方式: 先加权再拆分
    shared_updates_good = []
    for i, update in enumerate(client_updates):
        weighted_state = {}
        for key, value in update["model_state"].items():
            weighted_state[key] = value.float() * normalized_weights[i]
        shared = secagg.client_split_update(update["client_id"], weighted_state, all_client_ids)
        shared_updates_good.append(shared)
    good_result_raw = secagg.aggregate_shares(shared_updates_good, all_client_ids)
    good_result = {}
    for key, values in good_result_raw.items():
        sample_tensor = client_updates[0]["model_state"][key].float()
        good_result[key] = torch.tensor(values, dtype=sample_tensor.dtype).reshape_as(sample_tensor)
    avg_param_good = np.mean([v.abs().mean().item() for v in good_result.values()])

    print(f"\n✗ 错误方式 (直接求和): 平均参数绝对值 = {avg_param_bad:.6f} (放大了 ~{avg_param_bad/avg_param_before:.1f} 倍)")
    print(f"✓ 正确方式 (加权平均): 平均参数绝对值 = {avg_param_good:.6f} (与原值一致)")

    assert avg_param_bad > avg_param_before * 5, f"错误方式应该放大参数，实际放大了 {avg_param_bad/avg_param_before:.1f} 倍"
    assert abs(avg_param_good - avg_param_before) < 0.5, f"正确方式应该保持参数尺度，实际差了 {abs(avg_param_good - avg_param_before):.6f}"

    print("\n✓ 参数尺度验证通过!")
    return True


def main():
    print("\n" + "#" * 70)
    print("# 安全聚合发散问题 - 修复验证")
    print("#" * 70)

    tests = [
        test_model_divergence_simulation,
        test_secagg_vs_fedavg,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ 测试 {test.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"测试结果: {passed}/{passed+failed} 通过, {failed} 失败")
    print("=" * 70)

    if failed == 0:
        print("\n✓ 安全聚合发散问题已修复!")
        return True
    else:
        print(f"\n✗ {failed} 个测试失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
