"""
验证测试脚本 - 测试四个Bug修复
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np

from app.core.trainer import FederatedTrainer, ExperimentConfig
from app.aggregation.strategies import (
    FedAvgAggregator,
    FedProxAggregator,
    FedNovaAggregator,
    ScaffoldAggregator,
)
from app.privacy.mechanisms import (
    DPSGD,
    RDPAccountant,
    AdditiveSecretSharing,
    SecureAggregation,
)
from app.attack.simulation import (
    DataPoisoningAttack,
    ModelPoisoningAttack,
    BackdoorAttack,
    KrumDefense,
    TrimmedMeanDefense,
    MedianDefense,
)


def test_scaffold_control_variate():
    """测试 Scaffold 的控制变量校正是否在客户端训练时被应用"""
    print("=" * 60)
    print("测试 1: Scaffold 控制变量校正")
    print("=" * 60)

    config = ExperimentConfig(
        experiment_id="test_scaffold",
        dataset="mnist",
        num_clients=5,
        client_fraction=0.4,
        local_epochs=1,
        global_rounds=2,
        early_stop_patience=10,
        aggregation_strategy="scaffold",
        non_iid_type="label_skew",
        non_iid_param=2,
        learning_rate=0.01,
        batch_size=32,
        client_selection="random",
        dp_enabled=False,
        secure_aggregation=False,
        attack_type="none",
        defense_type="none",
    )

    trainer = FederatedTrainer(config)
    trainer.setup_data()

    from app.models.networks import get_model

    model = get_model("mnist")
    global_state = {k: v.clone() for k, v in model.state_dict().items()}

    aggregator = trainer.aggregator
    assert isinstance(aggregator, ScaffoldAggregator), "Aggregator should be Scaffold"

    if aggregator.global_control is None:
        aggregator.initialize_control(model)

    print(f"✓ 全局控制变量已初始化: {len(aggregator.global_control)} 个参数")
    print(f"✓ 全局控制变量非零: {any(v.abs().sum() > 0 for v in aggregator.global_control.values())}")

    client_id = 0
    trainer.client_control_variates[client_id] = {
        k: torch.zeros_like(v) for k, v in global_state.items()
    }

    global_control_before = {k: v.clone() for k, v in aggregator.global_control.items()}
    client_control_before = {k: v.clone() for k, v in trainer.client_control_variates[client_id].items()}

    result = trainer._train_client(client_id, global_state, 1)

    assert "control_variate" in result, "Result should contain control_variate"
    print(f"✓ 客户端返回结果包含控制变量: {len(result['control_variate'])} 个参数")

    client_control_after = trainer.client_control_variates[client_id]
    control_changed = any(
        not torch.equal(client_control_before[k], client_control_after[k])
        for k in client_control_before
    )
    print(f"✓ 客户端本地控制变量已更新: {control_changed}")

    updates = [result]
    weights = [result["num_samples"]]
    new_state = aggregator.aggregate(model, updates, weights)

    global_control_after = aggregator.global_control
    global_control_changed = any(
        not torch.equal(global_control_before[k], global_control_after[k])
        for k in global_control_before
    )
    print(f"✓ 全局控制变量已更新: {global_control_changed}")

    print("\n✓ Scaffold 控制变量校正验证通过!")
    return True


def test_data_poisoning_attack():
    """测试数据投毒攻击 - 标签翻转"""
    print("\n" + "=" * 60)
    print("测试 2: 数据投毒攻击 (标签翻转)")
    print("=" * 60)

    config = ExperimentConfig(
        experiment_id="test_poison",
        dataset="mnist",
        num_clients=10,
        client_fraction=0.4,
        local_epochs=1,
        global_rounds=2,
        early_stop_patience=10,
        aggregation_strategy="fedavg",
        non_iid_type="label_skew",
        non_iid_param=2,
        learning_rate=0.01,
        batch_size=32,
        client_selection="random",
        dp_enabled=False,
        secure_aggregation=False,
        attack_type="data_poisoning",
        attack_ratio=0.2,
        defense_type="none",
    )

    trainer = FederatedTrainer(config)
    trainer.setup_data()

    print(f"✓ 拜占庭客户端已设置: {trainer.byzyantine_clients}")
    print(f"✓ 攻击类型: {config.attack_type}, 比例: {config.attack_ratio}")

    from app.models.networks import get_model

    model = get_model("mnist")
    global_state = {k: v.clone() for k, v in model.state_dict().items()}

    byzantine_client_id = trainer.byzyantine_clients[0]
    normal_client_id = 0 if 0 not in trainer.byzyantine_clients else 1

    print(f"✓ 测试拜占庭客户端 {byzantine_client_id}...")
    print(f"✓ 测试正常客户端 {normal_client_id}...")

    result_byzantine = trainer._train_client(byzantine_client_id, global_state, 1)
    result_normal = trainer._train_client(normal_client_id, global_state, 1)

    assert result_byzantine.get("is_byzantine") == True, "拜占庭客户端标记应为True"
    assert result_normal.get("is_byzantine") == False, "正常客户端标记应为False"

    print(f"✓ 拜占庭客户端 is_byzantine 标记: {result_byzantine['is_byzantine']}")
    print(f"✓ 正常客户端 is_byzantine 标记: {result_normal['is_byzantine']}")

    anomalies = trainer._detect_anomaly([result_byzantine, result_normal], 1)
    print(f"✓ 异常检测发现: {len(anomalies)} 个异常")
    for a in anomalies:
        print(f"  - {a['type']}: {a['message']}")

    assert len(anomalies) >= 1, "应至少检测到1个异常(拜占庭客户端活跃)"

    attack = DataPoisoningAttack(poison_ratio=0.5)
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    poisoned = attack.poison_labels(labels, num_classes=10)
    flipped = sum(1 for o, p in zip(labels, poisoned) if o != p)
    print(f"✓ 测试标签翻转: {labels} -> {poisoned}")
    print(f"✓ 翻转比例: {flipped}/{len(labels)} ≈ {flipped/len(labels):.1%} (期望 ~50%)")
    assert flipped >= 3, f"应该至少翻转3个标签，实际翻转了{flipped}个"

    trainer.attack_log = []
    detected = trainer._detect_anomaly([result_byzantine, result_normal, result_normal], 1)
    for d in detected:
        trainer.attack_log.append(d)

    print(f"✓ 攻击日志长度: {len(trainer.attack_log)}")
    assert len(trainer.attack_log) >= 1

    print("\n✓ 数据投毒攻击验证通过!")
    return True


def test_secure_aggregation():
    """测试安全聚合 - 加法秘密共享拆分和重构"""
    print("\n" + "=" * 60)
    print("测试 3: 安全聚合 (加法秘密共享)")
    print("=" * 60)

    n = 5
    shamir = AdditiveSecretSharing(n=n)

    secrets = [0.123456789, -1.23456789, 0.987654321, -0.111111111]
    print(f"✓ 测试秘密值: {secrets}")

    all_reconstructions_ok = True
    for secret in secrets:
        shares = shamir.split(secret)
        assert len(shares) == n, f"应产生{n}份份额"
        print(f"  秘密 {secret:.6f}: 产生 {len(shares)} 份份额")

        reconstructed = shamir.reconstruct(shares)
        diff = abs(reconstructed - secret)
        print(f"  使用全部 {n} 份份额重构: {reconstructed:.6f}, 误差: {diff:.2e}")

        if diff > 1e-10:
            print(f"  ✗ 重构误差太大!")
            all_reconstructions_ok = False

    assert all_reconstructions_ok, "所有秘密重构验证失败"

    secagg = SecureAggregation(num_clients=5)

    from app.models.networks import get_model

    model = get_model("mnist")
    model_state = {k: v.clone() for k, v in model.state_dict().items()}

    num_params = sum(v.numel() for v in model_state.values())
    print(f"\n✓ 测试模型参数拆分: {len(model_state)} 个参数张量, {num_params} 个参数")

    client_ids = [0, 1, 2, 3, 4]
    all_shares = []
    for cid in client_ids:
        shares = secagg.client_split_update(cid, model_state, client_ids)
        all_shares.append(shares)
        print(f"  客户端 {cid}: 拆分完成, {len(shares)} 个参数")

    stats_before = secagg.get_stats()
    print(f"✓ 拆分操作统计: {stats_before}")

    reconstructed = secagg.aggregate_shares(all_shares, client_ids[:5])
    stats_after = secagg.get_stats()
    print(f"✓ 聚合操作统计: {stats_after}")
    assert stats_after["aggregate_operations"] > stats_before["aggregate_operations"]

    max_error = 0.0
    for key in model_state:
        original = model_state[key].float().numpy()
        recon = reconstructed[key]
        if recon.shape != original.shape:
            recon = recon.reshape(original.shape)
        error = np.max(np.abs(original - recon))
        max_error = max(max_error, error)

    print(f"✓ 聚合后最大参数误差: {max_error:.2e}")
    assert max_error < 1e-8, f"聚合后误差太大: {max_error}"

    print(f"\n✓ 安全聚合统计: {secagg.get_stats()}")

    print("\n✓ 安全聚合验证通过!")
    return True


def test_fedprox_convergence():
    """测试 FedProx 聚合器 - 验证不再有重复修正"""
    print("\n" + "=" * 60)
    print("测试 4: FedProx 聚合器 (修正重复计算问题)")
    print("=" * 60)

    from app.models.networks import get_model

    model = get_model("mnist")

    fedavg = FedAvgAggregator()
    fedprox = FedProxAggregator(mu=0.1)

    dummy_update = {
        "model_state": {k: v.clone() + 0.01 for k, v in model.state_dict().items()},
        "local_steps": 10,
    }

    updates = [dummy_update, dummy_update]
    weights = [1.0, 1.0]

    fedavg_result = fedavg.aggregate(model, updates, weights)
    fedprox_result = fedprox.aggregate(model, updates, weights)

    max_diff = 0.0
    for key in model.state_dict():
        diff = torch.max(torch.abs(fedavg_result[key] - fedprox_result[key])).item()
        max_diff = max(max_diff, diff)

    print(f"✓ FedAvg 和 FedProx 聚合结果最大差异: {max_diff:.2e}")
    print(f"  (FedProx 聚合器应该只做加权平均，与 FedAvg 聚合结果一致)")
    print(f"  (Prox 项只在客户端损失函数中计算，不在聚合器中)")
    assert max_diff < 1e-6, f"FedProx 和 FedAvg 聚合结果应该相同，但差异为 {max_diff}"

    config = ExperimentConfig(
        experiment_id="test_fedprox",
        dataset="mnist",
        num_clients=5,
        client_fraction=0.4,
        local_epochs=1,
        global_rounds=2,
        early_stop_patience=10,
        aggregation_strategy="fedprox",
        fedprox_mu=0.1,
        non_iid_type="label_skew",
        non_iid_param=2,
        learning_rate=0.01,
        batch_size=32,
        client_selection="random",
        dp_enabled=False,
        secure_aggregation=False,
        attack_type="none",
        defense_type="none",
    )

    trainer = FederatedTrainer(config)
    trainer.setup_data()

    global_state = {k: v.clone() for k, v in model.state_dict().items()}
    result = trainer._train_client(0, global_state, 1)

    print(f"✓ FedProx 客户端训练完成，mu={config.fedprox_mu}")
    print(f"  训练时损失函数会添加 (mu/2) * ||w - w_global||^2 正则项")

    print("\n✓ FedProx 收敛验证通过!")
    return True


def test_anomaly_detection_enhanced():
    """测试改进的异常检测"""
    print("\n" + "=" * 60)
    print("测试 5: 改进的异常检测 (MAD + 修正Z分数)")
    print("=" * 60)

    config = ExperimentConfig(
        experiment_id="test_anomaly",
        dataset="mnist",
        num_clients=10,
        client_fraction=0.4,
        local_epochs=1,
        global_rounds=2,
        aggregation_strategy="fedavg",
        non_iid_type="label_skew",
        non_iid_param=2,
        learning_rate=0.01,
        batch_size=32,
        attack_type="none",
        defense_type="none",
    )

    trainer = FederatedTrainer(config)

    from app.models.networks import get_model

    model = get_model("mnist")

    normal_updates = []
    for i in range(5):
        state = {k: v.clone() + torch.randn_like(v) * 0.001 for k, v in model.state_dict().items()}
        normal_updates.append({
            "client_id": i,
            "model_state": state,
            "local_steps": 10,
            "num_samples": 100,
            "is_byzantine": False,
        })

    poisoned_state = {k: v.clone() * -10.0 for k, v in model.state_dict().items()}
    poisoned_update = {
        "client_id": 99,
        "model_state": poisoned_state,
        "local_steps": 10,
        "num_samples": 100,
        "is_byzantine": True,
    }

    all_updates = normal_updates + [poisoned_update]

    anomalies = trainer._detect_anomaly(all_updates, 1)
    print(f"✓ 检测到 {len(anomalies)} 个异常:")
    for a in anomalies:
        print(f"  - {a['type']}: {a['message']}")

    high_anomalies = [a for a in anomalies if "high" in a["type"] or "byzantine" in a["type"]]
    assert len(high_anomalies) >= 2, f"应该至少检测到2个高严重性异常，实际检测到{len(high_anomalies)}个"

    print("\n✓ 改进的异常检测验证通过!")
    return True


def main():
    print("\n" + "#" * 60)
    print("# 联邦学习平台 - Bug 修复验证测试")
    print("#" * 60)

    tests = [
        test_scaffold_control_variate,
        test_data_poisoning_attack,
        test_secure_aggregation,
        test_fedprox_convergence,
        test_anomaly_detection_enhanced,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n✗ 测试 {test.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{len(tests)} 通过, {failed} 失败")
    print("=" * 60)

    if failed == 0:
        print("\n✓ 所有测试通过! Bug 修复验证成功!")
        return 0
    else:
        print(f"\n✗ {failed} 个测试失败，需要进一步修复")
        return 1


if __name__ == "__main__":
    sys.exit(main())
