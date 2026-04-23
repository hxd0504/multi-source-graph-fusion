# -*- coding: utf-8 -*-
"""
实验1：分段建模消融实验
======================
对比分段建模 vs 无分段建模的性能差异
这是论文的核心创新点

预期结果：分段建模显著优于无分段建模 (R²: ~0.25 → ~0.72)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gnn_models import FusionModel
from models.data_utils import load_real_data, create_spatial_graph, create_ecological_graph
from models.trainer import train_segmented_model, train_single_model, evaluate_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 配置
RANDOM_SEED = 42
N_RUNS = 10
EPOCHS = 300
HIDDEN_DIM = 64
DROPOUT = 0.3
LR = 0.005
WEIGHT_DECAY = 1e-4
K = 5
THRESHOLD = 6.0

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train_baseline_model(model_class, features, labels, edge_index, train_idx, test_idx,
                         device, is_fusion=False, edge_index_ekg=None, **model_kwargs):
    """训练无分段的基线模型"""
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    scaler_X.fit(features[train_idx])
    scaler_y.fit(labels[train_idx].reshape(-1, 1))

    features_scaled = scaler_X.transform(features)
    labels_scaled = scaler_y.transform(labels.reshape(-1, 1)).flatten()

    x = torch.tensor(features_scaled, dtype=torch.float32)
    y = torch.tensor(labels_scaled, dtype=torch.float32)

    model = model_class(**model_kwargs)
    model = model.to(device)

    x_dev = x.to(device)
    y_dev = y.to(device)
    edge_index_dev = edge_index.to(device)
    edge_index_ekg_dev = edge_index_ekg.to(device) if edge_index_ekg is not None else None

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        if is_fusion:
            pred, _ = model(x_dev, edge_index_dev, edge_index_ekg_dev)
        else:
            pred = model(x_dev, edge_index_dev)

        loss = criterion(pred[train_idx], y_dev[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # 评估
    model.eval()
    with torch.no_grad():
        if is_fusion:
            pred, _ = model(x_dev, edge_index_dev, edge_index_ekg_dev)
        else:
            pred = model(x_dev, edge_index_dev)
        pred_np = pred.cpu().numpy()

    pred_original = scaler_y.inverse_transform(pred_np.reshape(-1, 1)).flatten()
    test_pred = pred_original[test_idx]
    test_true = labels[test_idx]

    metrics = {
        'r2': r2_score(test_true, test_pred),
        'mae': mean_absolute_error(test_true, test_pred),
        'rmse': np.sqrt(mean_squared_error(test_true, test_pred)),
    }

    return metrics


if __name__ == '__main__':
    print("="*80)
    print("实验1：分段建模消融实验")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # 加载数据
    print("\n1. 加载真实数据...")
    features, labels, coords, metadata = load_real_data()
    print(f"   样本数: {len(labels)}")
    print(f"   特征维度: {features.shape[1]}")
    print(f"   生物量范围: [{labels.min():.2f}, {labels.max():.2f}] kg/m2")

    # 构建图
    print("\n2. 构建图结构...")
    edge_index_tkg = create_spatial_graph(coords, k=K)
    edge_index_ekg = create_ecological_graph(features, k=K)
    print(f"   TKG边数: {edge_index_tkg.shape[1]}")
    print(f"   EKG边数: {edge_index_ekg.shape[1]}")

    # 运行实验
    print(f"\n3. 运行 {N_RUNS} 次实验...")

    results = []

    for run in range(N_RUNS):
        seed = RANDOM_SEED + run
        set_seed(seed)

        indices = np.arange(len(labels))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)

        print(f"\n   Run {run+1}/{N_RUNS} (seed={seed})...")

        model_kwargs = {
            'input_dim': features.shape[1],
            'hidden_dim': HIDDEN_DIM,
            'heads': 2,
            'dropout': DROPOUT
        }

        # 无分段基线
        baseline_metrics = train_baseline_model(
            FusionModel, features, labels, edge_index_tkg, train_idx, test_idx,
            device, is_fusion=True, edge_index_ekg=edge_index_ekg, **model_kwargs
        )
        print(f"      无分段: R2={baseline_metrics['r2']:.4f}, MAE={baseline_metrics['mae']:.4f}")

        # 分段模型
        segmented_metrics, _ = train_segmented_model(
            FusionModel, features, labels, edge_index_tkg, train_idx, test_idx,
            device, threshold=THRESHOLD, is_fusion=True, edge_index_ekg=edge_index_ekg,
            epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, **model_kwargs
        )
        print(f"      分段:   R2={segmented_metrics['r2']:.4f}, MAE={segmented_metrics['mae']:.4f}")

        # 计算提升
        improvement = (segmented_metrics['r2'] - baseline_metrics['r2']) / max(baseline_metrics['r2'], 0.01) * 100

        results.append({
            'run': run + 1,
            'seed': seed,
            'baseline_r2': baseline_metrics['r2'],
            'baseline_mae': baseline_metrics['mae'],
            'baseline_rmse': baseline_metrics['rmse'],
            'segmented_r2': segmented_metrics['r2'],
            'segmented_mae': segmented_metrics['mae'],
            'segmented_rmse': segmented_metrics['rmse'],
            'improvement_pct': improvement
        })

    # 汇总结果
    print("\n" + "="*80)
    print("结果汇总")
    print("="*80)

    df = pd.DataFrame(results)

    print("\n【无分段基线】")
    print(f"   R2:   {df['baseline_r2'].mean():.4f} +/- {df['baseline_r2'].std():.4f}")
    print(f"   MAE:  {df['baseline_mae'].mean():.4f} +/- {df['baseline_mae'].std():.4f}")
    print(f"   RMSE: {df['baseline_rmse'].mean():.4f} +/- {df['baseline_rmse'].std():.4f}")

    print("\n[Segmented Model]")
    print(f"   R2:   {df['segmented_r2'].mean():.4f} +/- {df['segmented_r2'].std():.4f}")
    print(f"   MAE:  {df['segmented_mae'].mean():.4f} +/- {df['segmented_mae'].std():.4f}")
    print(f"   RMSE: {df['segmented_rmse'].mean():.4f} +/- {df['segmented_rmse'].std():.4f}")

    print("\n[Performance Improvement]")
    print(f"   R2 improvement:  +{df['improvement_pct'].mean():.1f}%")
    print(f"   MAE reduction: -{(1 - df['segmented_mae'].mean()/df['baseline_mae'].mean())*100:.1f}%")

    # 保存结果
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    df.to_csv(os.path.join(results_dir, 'exp1_segmentation_ablation.csv'), index=False)
    print(f"\n结果已保存: results/exp1_segmentation_ablation.csv")

    # 保存汇总
    summary = {
        'Baseline_R2_mean': df['baseline_r2'].mean(),
        'Baseline_R2_std': df['baseline_r2'].std(),
        'Segmented_R2_mean': df['segmented_r2'].mean(),
        'Segmented_R2_std': df['segmented_r2'].std(),
        'Improvement_pct': df['improvement_pct'].mean(),
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(results_dir, 'exp1_summary.csv'), index=False)

    print("\n" + "="*80)
    print("实验完成!")
    print("="*80)
