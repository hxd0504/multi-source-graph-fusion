# -*- coding: utf-8 -*-
"""
实验5：图结构分析
================
分析TKG和EKG的结构差异，验证信息互补性

指标：
- Jaccard相似度（边重叠度）
- 度分布差异
- 邻居重叠度
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_utils import load_real_data, create_spatial_graph, create_ecological_graph

# 配置
K = 5


if __name__ == '__main__':
    print("="*80)
    print("实验5：图结构分析")
    print("="*80)

    # 加载数据
    print("\n1. 加载真实数据...")
    features, labels, coords, metadata = load_real_data()
    print(f"   样本数: {len(labels)}")

    # 构建图
    print("\n2. 构建图结构...")
    edge_index_tkg = create_spatial_graph(coords, k=K)
    edge_index_ekg = create_ecological_graph(features, k=K)

    print(f"   TKG边数: {edge_index_tkg.shape[1]}")
    print(f"   EKG边数: {edge_index_ekg.shape[1]}")

    # 转换为边集合
    tkg_edges = set(tuple(e) for e in edge_index_tkg.t().numpy().tolist())
    ekg_edges = set(tuple(e) for e in edge_index_ekg.t().numpy().tolist())

    # 计算结构差异
    print("\n3. 计算图结构差异...")

    # Jaccard相似度
    intersection = len(tkg_edges & ekg_edges)
    union = len(tkg_edges | ekg_edges)
    jaccard = intersection / union if union > 0 else 0

    print(f"\n【边重叠分析】")
    print(f"   TKG边数: {len(tkg_edges)}")
    print(f"   EKG边数: {len(ekg_edges)}")
    print(f"   共享边数: {intersection}")
    print(f"   Jaccard相似度: {jaccard:.4f} ({jaccard*100:.2f}%)")
    print(f"   结构差异度: {1-jaccard:.4f} ({(1-jaccard)*100:.2f}%)")

    # 度分布分析
    print("\n【度分布分析】")

    n_nodes = len(labels)
    tkg_degrees = np.zeros(n_nodes)
    ekg_degrees = np.zeros(n_nodes)

    for i, j in tkg_edges:
        tkg_degrees[i] += 1

    for i, j in ekg_edges:
        ekg_degrees[i] += 1

    print(f"   TKG度: mean={tkg_degrees.mean():.2f}, std={tkg_degrees.std():.2f}, range=[{tkg_degrees.min():.0f}, {tkg_degrees.max():.0f}]")
    print(f"   EKG度: mean={ekg_degrees.mean():.2f}, std={ekg_degrees.std():.2f}, range=[{ekg_degrees.min():.0f}, {ekg_degrees.max():.0f}]")

    # 度相关性
    degree_corr = np.corrcoef(tkg_degrees, ekg_degrees)[0, 1]
    print(f"   度相关性: {degree_corr:.4f}")

    # 邻居重叠分析
    print("\n【邻居重叠分析】")

    # 构建邻接表
    tkg_neighbors = {i: set() for i in range(n_nodes)}
    ekg_neighbors = {i: set() for i in range(n_nodes)}

    for i, j in tkg_edges:
        tkg_neighbors[i].add(j)

    for i, j in ekg_edges:
        ekg_neighbors[i].add(j)

    # 计算每个节点的邻居重叠度
    neighbor_overlaps = []
    for i in range(n_nodes):
        tkg_n = tkg_neighbors[i]
        ekg_n = ekg_neighbors[i]
        if len(tkg_n) > 0 and len(ekg_n) > 0:
            overlap = len(tkg_n & ekg_n) / len(tkg_n | ekg_n)
            neighbor_overlaps.append(overlap)

    neighbor_overlaps = np.array(neighbor_overlaps)
    print(f"   平均邻居重叠度: {neighbor_overlaps.mean():.4f} ({neighbor_overlaps.mean()*100:.2f}%)")
    print(f"   邻居重叠度std: {neighbor_overlaps.std():.4f}")
    print(f"   完全不重叠节点比例: {(neighbor_overlaps == 0).mean()*100:.1f}%")

    # 按生物量区间分析
    print("\n【按生物量区间的邻居重叠度】")

    ranges = [(0, 3, '低值(<3)'), (3, 6, '中低(3-6)'), (6, 9, '中值(6-9)'),
              (9, 12, '中高(9-12)'), (12, 20, '高值(>12)')]

    for low, high, name in ranges:
        mask = (labels >= low) & (labels < high)
        if mask.sum() < 10:
            continue

        range_overlaps = neighbor_overlaps[mask[:len(neighbor_overlaps)]]
        if len(range_overlaps) > 0:
            print(f"   {name}: 邻居重叠度={range_overlaps.mean():.4f} (n={mask.sum()})")

    # 保存结果
    print("\n4. 保存结果...")

    results = {
        'metric': [
            'TKG_edges', 'EKG_edges', 'Shared_edges',
            'Jaccard_similarity', 'Structure_difference',
            'TKG_degree_mean', 'EKG_degree_mean', 'Degree_correlation',
            'Neighbor_overlap_mean', 'Neighbor_overlap_std'
        ],
        'value': [
            len(tkg_edges), len(ekg_edges), intersection,
            jaccard, 1 - jaccard,
            tkg_degrees.mean(), ekg_degrees.mean(), degree_corr,
            neighbor_overlaps.mean(), neighbor_overlaps.std()
        ]
    }

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(results_dir, 'exp5_graph_structure.csv'), index=False)

    print(f"\n结果已保存: results/exp5_graph_structure.csv")

    # 总结
    print("\n" + "="*80)
    print("结论")
    print("="*80)
    print(f"""
TKG和EKG的结构差异度为 {(1-jaccard)*100:.1f}%，表明：
- 两个图捕获了几乎完全不同的关系
- TKG基于空间邻近性，EKG基于特征相似性
- 这种高度差异性为信息融合提供了理论基础

邻居重叠度仅为 {neighbor_overlaps.mean()*100:.1f}%，进一步证实：
- 同一节点在两个图中连接到不同的邻居
- 融合可以整合来自不同视角的信息
""")

    print("="*80)
    print("实验完成!")
    print("="*80)
