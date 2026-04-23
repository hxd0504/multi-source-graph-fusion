# -*- coding: utf-8 -*-
"""
实验3：边界区域稳健性测试
========================
在机制边界区域（τ附近，5-7 kg/m²）测试模型稳健性
这是证明Fusion价值的关键实验

预期结果：Fusion在边界区域的误差方差更小，更稳健
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gnn_models import TKGOnlyModel, EKGOnlyModel, FusionModel
from models.data_utils import load_real_data, create_spatial_graph, create_ecological_graph
from models.trainer import train_segmented_model, compute_boundary_metrics

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

# 边界区域定义
BOUNDARY_LOW = 5.0
BOUNDARY_HIGH = 7.0

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    print("="*80)
    print("实验3：边界区域稳健性测试")
    print(f"Boundary region: {BOUNDARY_LOW} - {BOUNDARY_HIGH} kg/m2")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # 加载数据
    print("\n1. 加载真实数据...")
    features, labels, coords, metadata = load_real_data()
    print(f"   样本数: {len(labels)}")

    # 边界区域统计
    boundary_mask = (labels >= BOUNDARY_LOW) & (labels <= BOUNDARY_HIGH)
    n_boundary = boundary_mask.sum()
    print(f"   边界区域样本: {n_boundary} ({n_boundary/len(labels)*100:.1f}%)")

    # 构建图
    print("\n2. 构建图结构...")
    edge_index_tkg = create_spatial_graph(coords, k=K)
    edge_index_ekg = create_ecological_graph(features, k=K)

    # 运行实验
    print(f"\n3. 运行 {N_RUNS} 次实验...")

    results = []

    for run in range(N_RUNS):
        seed = RANDOM_SEED + run
        set_seed(seed)

        indices = np.arange(len(labels))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)

        print(f"\n   Run {run+1}/{N_RUNS} (seed={seed})...")

        for model_name, model_class, is_fusion, edge_idx, model_kwargs in [
            ('TKG', TKGOnlyModel, False, edge_index_tkg,
             {'input_dim': features.shape[1], 'hidden_dim': HIDDEN_DIM, 'heads': 2, 'dropout': DROPOUT}),
            ('EKG', EKGOnlyModel, False, edge_index_ekg,
             {'input_dim': features.shape[1], 'hidden_dim': HIDDEN_DIM, 'dropout': DROPOUT}),
            ('Fusion', FusionModel, True, edge_index_tkg,
             {'input_dim': features.shape[1], 'hidden_dim': HIDDEN_DIM, 'heads': 2, 'dropout': DROPOUT})
        ]:
            edge_ekg = edge_index_ekg if is_fusion else None

            metrics, predictions = train_segmented_model(
                model_class, features, labels, edge_idx, train_idx, test_idx,
                device, threshold=THRESHOLD, is_fusion=is_fusion, edge_index_ekg=edge_ekg,
                epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, **model_kwargs
            )

            # 边界区域指标
            boundary_metrics = compute_boundary_metrics(
                predictions, labels, test_idx, BOUNDARY_LOW, BOUNDARY_HIGH
            )

            if boundary_metrics:
                results.append({
                    'run': run + 1,
                    'model': model_name,
                    'overall_r2': metrics['r2'],
                    'overall_mae': metrics['mae'],
                    'boundary_n': boundary_metrics['n_samples'],
                    'boundary_mae': boundary_metrics['mae'],
                    'boundary_rmse': boundary_metrics['rmse'],
                    'boundary_error_std': boundary_metrics['error_std'],
                })

                print(f"      {model_name}: R2={metrics['r2']:.4f}, BoundaryMAE={boundary_metrics['mae']:.4f}, ErrorStd={boundary_metrics['error_std']:.4f}")

    # 汇总结果
    print("\n" + "="*80)
    print("结果汇总")
    print("="*80)

    df = pd.DataFrame(results)

    print("\n【整体性能】")
    print("-"*60)
    for model in ['TKG', 'EKG', 'Fusion']:
        model_df = df[df['model'] == model]
        print(f"{model}: R2={model_df['overall_r2'].mean():.4f}+/-{model_df['overall_r2'].std():.4f}")

    print("\n[Boundary Robustness] (Key Metrics)")
    print("-"*60)
    print(f"{'Model':<10} {'BoundaryMAE':<15} {'BoundaryRMSE':<15} {'ErrorStd':<15}")
    print("-"*60)

    for model in ['TKG', 'EKG', 'Fusion']:
        model_df = df[df['model'] == model]
        mae = f"{model_df['boundary_mae'].mean():.4f}±{model_df['boundary_mae'].std():.4f}"
        rmse = f"{model_df['boundary_rmse'].mean():.4f}±{model_df['boundary_rmse'].std():.4f}"
        err_std = f"{model_df['boundary_error_std'].mean():.4f}±{model_df['boundary_error_std'].std():.4f}"
        print(f"{model:<10} {mae:<15} {rmse:<15} {err_std:<15}")

    # 统计检验
    print("\n【统计检验】")
    print("-"*60)

    tkg_err_std = df[df['model'] == 'TKG']['boundary_error_std'].values
    ekg_err_std = df[df['model'] == 'EKG']['boundary_error_std'].values
    fusion_err_std = df[df['model'] == 'Fusion']['boundary_error_std'].values

    # Fusion vs TKG
    t_stat, p_val = stats.ttest_rel(fusion_err_std, tkg_err_std)
    print(f"Fusion vs TKG (边界误差std): t={t_stat:.4f}, p={p_val:.4f}")
    if p_val < 0.05:
        winner = "Fusion更稳健" if fusion_err_std.mean() < tkg_err_std.mean() else "TKG更稳健"
        print(f"  → 显著差异 (p<0.05): {winner}")

    # Fusion vs EKG
    t_stat, p_val = stats.ttest_rel(fusion_err_std, ekg_err_std)
    print(f"Fusion vs EKG (边界误差std): t={t_stat:.4f}, p={p_val:.4f}")
    if p_val < 0.05:
        winner = "Fusion更稳健" if fusion_err_std.mean() < ekg_err_std.mean() else "EKG更稳健"
        print(f"  → 显著差异 (p<0.05): {winner}")

    # 保存结果
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    df.to_csv(os.path.join(results_dir, 'exp3_boundary_robustness.csv'), index=False)
    print(f"\n结果已保存: results/exp3_boundary_robustness.csv")

    # 保存汇总
    summary_data = []
    for model in ['TKG', 'EKG', 'Fusion']:
        model_df = df[df['model'] == model]
        summary_data.append({
            'Model': model,
            'Overall_R2_mean': model_df['overall_r2'].mean(),
            'Overall_R2_std': model_df['overall_r2'].std(),
            'Boundary_MAE_mean': model_df['boundary_mae'].mean(),
            'Boundary_MAE_std': model_df['boundary_mae'].std(),
            'Boundary_ErrorStd_mean': model_df['boundary_error_std'].mean(),
            'Boundary_ErrorStd_std': model_df['boundary_error_std'].std(),
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(results_dir, 'exp3_summary.csv'), index=False)

    print("\n" + "="*80)
    print("实验完成!")
    print("="*80)
