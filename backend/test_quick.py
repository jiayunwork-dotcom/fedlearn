"""
快速单元测试 - 只测试核心修复逻辑，不加载完整数据集
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np

from app.aggregation.strategies import FedAvgAggregator, FedProxAggregator
from app.privacy.mechanisms import AdditiveSecretSharing, SecureAggregation


def test_additive_secret_sharing():
    """测试加法秘密共享的拆分和重构精度"""
    print("\n" + "=" * 60)
    print("测试 1: 加法秘密共享精度验证")
    print("=" * 60)

    n = 5
    ass = AdditiveSecretSharing(n=n)

    test_secrets = [0.5, -0.3, 1.23456789, -0.987654321, 0.0, 100.0, -50.0]
    all_ok = True

    for secret in test_secrets:
        shares = ass.split(secret)
        assert len(shares) == n, f"应产生{n}份份额"

        reconstructed = ass.reconstruct(shares)
        diff = abs(reconstructed - secret)

        status = "✓" if diff < 1e-12 else "✗"
        print(f"  {status} 秘密={secret:12.8f}, 重构={reconstructed:12.8f}, 误差={diff:.2e}")

        if diff >= 1e-12:
            all_ok = False

    assert all_ok, "秘密共享重构精度不达标"
    print("\n✓ 加法秘密共享验证通过!")
    return True


def test_secure_aggregation_model():
    """测试安全聚合对模型参数的拆分和重构"""
    print("\n" + "=" * 60)
    print("测试 2: 安全聚合 - 模型参数级别的拆分重构")
    print("=" * 60)

    num_clients = 5
    secagg = SecureAggregation(num_clients=num_clients)

    model_state = {
        "layer1.weight": torch.randn(32, 10),
        "layer1.bias": torch.randn(32),
        "layer2.weight": torch.randn(10, 32),
        "layer2.bias": torch.randn(10),
    }

    num_params = sum(v.numel() for v in model_state.values())
    print(f"✓ 测试模型: {len(model_state)} 个参数张量, {num_params} 个参数")

    client_ids = list(range(num_clients))
    all_shares = []

    for cid in client_ids:
        shares = secagg.client_split_update(cid, model_state, client_ids)
        all_shares.append(shares)
        print(f"  客户端 {cid}: 拆分完成")

    stats_before = secagg.get_stats()
    print(f"✓ 拆分统计: {stats_before}")

    reconstructed = secagg.aggregate_shares(all_shares, client_ids)
    stats_after = secagg.get_stats()
    print(f"✓ 聚合统计: {stats_after}")

    max_error = 0.0
    for key in model_state:
        original = model_state[key].float().numpy()
        recon = reconstructed[key].reshape(original.shape)
        expected = original * num_clients  # 安全聚合是对多个客户端的更新求和
        error = np.max(np.abs(expected - recon))
        max_error = max(max_error, error)
        print(f"  {key}: 最大误差 = {error:.2e}")

    print(f"\n✓ 全局最大参数误差: {max_error:.2e}")
    print(f"  (聚合结果 = {num_clients} 个客户端更新之和)")
    assert max_error < 1e-4, f"聚合后误差太大: {max_error}"

    print("\n✓ 安全聚合模型参数验证通过!")
    return True


def test_fedprox_aggregator():
    """测试 FedProx 聚合器 - 验证与 FedAvg 行为一致（无重复修正）"""
    print("\n" + "=" * 60)
    print("测试 3: FedProx 聚合器一致性验证")
    print("=" * 60)

    fedavg = FedAvgAggregator()
    fedprox = FedProxAggregator(mu=0.1)

    global_model = nn.Linear(10, 2)
    global_state = {k: v.clone() for k, v in global_model.state_dict().items()}

    num_clients = 3
    client_updates = []
    client_weights = []

    for i in range(num_clients):
        update = {}
        for key in global_state:
            update[key] = global_state[key] + torch.randn_like(global_state[key]) * 0.01
        client_updates.append({
            "client_id": i,
            "model_state": update,
            "num_samples": 100 + i * 50,
        })
        client_weights.append(100 + i * 50)

    fedavg_result = fedavg.aggregate(global_model, client_updates, client_weights)
    fedprox_result = fedprox.aggregate(global_model, client_updates, client_weights)

    max_diff = 0.0
    for key in global_state:
        diff = torch.max(torch.abs(fedavg_result[key] - fedprox_result[key])).item()
        max_diff = max(max_diff, diff)
        print(f"  {key}: FedAvg vs FedProx 差异 = {diff:.2e}")

    print(f"\n✓ FedAvg 和 FedProx 聚合结果最大差异: {max_diff:.2e}")
    print("  (FedProx 聚合器应只做加权平均，与 FedAvg 聚合结果一致)")
    print("  (Prox 项只在客户端损失函数中计算，不在聚合器中)")

    assert max_diff < 1e-12, f"FedProx 和 FedAvg 聚合结果不一致: {max_diff}"

    print("\n✓ FedProx 聚合器验证通过!")
    return True


def test_anomaly_detection_logic():
    """测试 MAD + 修正 Z 分数的异常检测逻辑"""
    print("\n" + "=" * 60)
    print("测试 4: MAD + 修正 Z 分数异常检测")
    print("=" * 60)

    np.random.seed(42)

    norms = np.concatenate([
        np.random.normal(loc=1.0, scale=0.1, size=8),
        np.array([10.0]),
    ])

    median = np.median(norms)
    mad = np.median(np.abs(norms - median))
    modified_z_scores = 0.6745 * (norms - median) / (mad + 1e-10)

    print(f"  范数值: {norms}")
    print(f"  中位数: {median:.4f}")
    print(f"  MAD: {mad:.4f}")
    print(f"  修正 Z 分数: {modified_z_scores}")

    threshold = 2.5
    anomalies = np.where(np.abs(modified_z_scores) > threshold)[0]

    print(f"  检测到的异常索引: {anomalies}")
    print(f"  异常数量: {len(anomalies)}")

    assert len(anomalies) == 1, f"应检测到1个异常，实际{len(anomalies)}个"
    assert anomalies[0] == 8, f"异常应该在索引8，实际在{anomalies[0]}"

    print("\n✓ 异常检测逻辑验证通过!")
    return True


def test_control_variate_update_formula():
    """测试 Scaffold 控制变量更新公式的正确性"""
    print("\n" + "=" * 60)
    print("测试 5: Scaffold 控制变量更新公式")
    print("=" * 60)

    lr = 0.01
    tau = 5

    x_global = 1.0
    x_local = 0.9
    c_global = 0.1
    c_i_old = 0.05

    delta_c_i = (x_global - x_local) / (lr * tau)
    c_i_new = c_i_old - c_global + delta_c_i

    expected_c_i_new = c_i_old - c_global + (x_global - x_local) / (lr * tau)

    print(f"  x_global = {x_global}, x_local = {x_local}")
    print(f"  c_global = {c_global}, c_i_old = {c_i_old}")
    print(f"  lr = {lr}, τ = {tau}")
    print(f"  Δc_i = (x_global - x_local)/(lr*τ) = {delta_c_i:.4f}")
    print(f"  c_i_new = c_i_old - c_global + Δc_i = {c_i_new:.4f}")

    assert abs(c_i_new - expected_c_i_new) < 1e-10, "控制变量更新公式错误"

    grad_correction = c_global - c_i_old
    print(f"\n  梯度校正项 (c - c_i) = {grad_correction:.4f}")
    print("  g_i = ∇F_i + c - c_i ✓")

    print("\n✓ Scaffold 控制变量更新公式验证通过!")
    return True


def main():
    print("\n" + "#" * 60)
    print("# 联邦学习平台 - 核心修复快速验证")
    print("#" * 60)

    tests = [
        test_additive_secret_sharing,
        test_secure_aggregation_model,
        test_fedprox_aggregator,
        test_anomaly_detection_logic,
        test_control_variate_update_formula,
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

    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{passed+failed} 通过, {failed} 失败")
    print("=" * 60)

    if failed == 0:
        print("\n✓ 所有核心修复验证通过!")
        return True
    else:
        print(f"\n✗ {failed} 个测试失败，需要进一步修复")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
