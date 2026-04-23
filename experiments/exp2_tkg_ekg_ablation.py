# -*- coding: utf-8 -*-
"""
实验2：TKG/EKG消融实验
=====================
对比 TKG-only, EKG-only, Fusion 三种模型的性能
所有模型都使用分段建模

预期结果：三者性能相近（~0.72），说明分段是关键
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gnn_models import TKGOnlyModel, EKGOnlyModel, FusionModel
from models.data_utils import load_real_data, create_spatial_graph, create_ecological_graph
from models.trainer import train_segmented_model

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


if __name__ == '__main__':
    print("="*80)
    print("实验2：TKG/EKG消融实验")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # 加载数据
    print("\n1. 加载真实数据...")
    features, labels, coords, metadata = load_real_data()
    print(f"   样本数: {len(labels)}")
    print(f"   特征维度: {features.shape[1]}")

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

        run_results = {'run': run + 1, 'seed': seed}

        # TKG-only
        tkg_metrics, _ = train_segmented_model(
            TKGOnlyModel, features, labels, edge_index_tkg, train_idx, test_idx,
            device, threshold=THRESHOLD, is_fusion=False,
            epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
            input_dim=features.shape[1], hidden_dim=HIDDEN_DIM, heads=2, dropout=DROPOUT
        )
        run_results['TKG_r2'] = tkg_metrics['r2']
        run_results['TKG_mae'] = tkg_metrics['mae']
        print(f"      TKG-only: R2={tkg_metrics['r2']:.4f}")

        # EKG-only
        ekg_metrics, _ = train_segmented_model(
            EKGOnlyModel, features, labels, edge_index_ekg, train_idx, test_idx,
            device, threshold=THRESHOLD, is_fusion=False,
            epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
            input_dim=features.shape[1], hidden_dim=HIDDEN_DIM, dropout=DROPOUT
        )
        run_results['EKG_r2'] = ekg_metrics['r2']
        run_results['EKG_mae'] = ekg_metrics['mae']
        print(f"      EKG-only: R2={ekg_metrics['r2']:.4f}")

        # Fusion
        fusion_metrics, _ = train_segmented_model(
            FusionModel, features, labels, edge_index_tkg, train_idx, test_idx,
            device, threshold=THRESHOLD, is_fusion=True, edge_index_ekg=edge_index_ekg,
            epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
            input_dim=features.shape[1], hidden_dim=HIDDEN_DIM, heads=2, dropout=DROPOUT
        )
        run_results['Fusion_r2'] = fusion_metrics['r2']
        run_results['Fusion_mae'] = fusion_metrics['mae']
        print(f"      Fusion:   R2={fusion_metrics['r2']:.4f}")

        results.append(run_results)

    # 汇总结果
    print("\n" + "="*80)
    print("结果汇总")
    print("="*80)

    df = pd.DataFrame(results)

    print("\n[Model Performance Comparison] (Segmented, tau=6)")
    print("-"*60)
    print(f"{'Model':<12} {'R2 Mean':<12} {'R2 Std':<12} {'MAE Mean':<12}")
    print("-"*60)

    for model in ['TKG', 'EKG', 'Fusion']:
        r2_mean = df[f'{model}_r2'].mean()
        r2_std = df[f'{model}_r2'].std()
        mae_mean = df[f'{model}_mae'].mean()
        print(f"{model:<12} {r2_mean:<12.4f} {r2_std:<12.4f} {mae_mean:<12.4f}")

    print("-"*60)

    # 计算Fusion增益
    best_single = max(df['TKG_r2'].mean(), df['EKG_r2'].mean())
    fusion_gain = (df['Fusion_r2'].mean() - best_single) / best_single * 100

    print(f"\nFusion相对最佳单图增益: {fusion_gain:+.2f}%")

    # 保存结果
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    df.to_csv(os.path.join(results_dir, 'exp2_tkg_ekg_ablation.csv'), index=False)
    print(f"\n结果已保存: results/exp2_tkg_ekg_ablation.csv")

    print("\n" + "="*80)
    print("实验完成!")
    print("="*80)
