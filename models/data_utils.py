# -*- coding: utf-8 -*-
"""
数据加载和图构建工具
===================
确保所有实验使用相同的数据加载和图构建方式
"""

import os
import numpy as np
import pickle
import torch
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


# 数据路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def load_real_data():
    """
    加载真实数据
    返回: features, labels, coords, metadata
    """
    features = np.load(os.path.join(DATA_DIR, 'aligned_features_multiyear.npy'))
    labels = np.load(os.path.join(DATA_DIR, 'aligned_labels_multiyear.npy'))

    with open(os.path.join(DATA_DIR, 'aligned_metadata_multiyear.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    coords = metadata['coords']

    return features, labels, coords, metadata


def create_spatial_graph(coords, k=5):
    """
    创建空间拓扑图 (TKG)
    基于地理坐标的KNN图

    参数:
        coords: 坐标列表 [(lat, lon), ...]
        k: 每个节点连接的邻居数

    返回:
        edge_index: [2, num_edges] 的边索引张量
    """
    coords_array = np.array([[c[0], c[1]] for c in coords])
    distances = pairwise_distances(coords_array)

    edge_list = []
    for i in range(len(coords)):
        nearest = np.argsort(distances[i])[1:k+1]  # 排除自己
        for j in nearest:
            edge_list.append([i, j])
            edge_list.append([j, i])  # 无向图

    # 去重
    edge_set = set(tuple(e) for e in edge_list)
    edge_list = list(edge_set)

    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


def create_ecological_graph(features, k=5):
    """
    创建生态特征图 (EKG)
    基于特征相似度的KNN图

    参数:
        features: 特征矩阵 [n_samples, n_features]
        k: 每个节点连接的邻居数

    返回:
        edge_index: [2, num_edges] 的边索引张量
    """
    sim_matrix = cosine_similarity(features)

    edge_list = []
    for i in range(len(features)):
        # 选择相似度最高的k个（排除自己）
        similar = np.argsort(sim_matrix[i])[-k-1:-1]
        for j in similar:
            edge_list.append([i, j])
            edge_list.append([j, i])  # 无向图

    # 去重
    edge_set = set(tuple(e) for e in edge_list)
    edge_list = list(edge_set)

    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


def prepare_data_for_training(features, labels, train_idx, test_idx):
    """
    准备训练数据：标准化特征和标签

    返回:
        x: 标准化后的特征张量
        y: 标准化后的标签张量
        scaler_X: 特征标准化器
        scaler_y: 标签标准化器
    """
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # 仅在训练集上fit
    scaler_X.fit(features[train_idx])
    scaler_y.fit(labels[train_idx].reshape(-1, 1))

    # 变换所有数据
    features_scaled = scaler_X.transform(features)
    labels_scaled = scaler_y.transform(labels.reshape(-1, 1)).flatten()

    x = torch.tensor(features_scaled, dtype=torch.float32)
    y = torch.tensor(labels_scaled, dtype=torch.float32)

    return x, y, scaler_X, scaler_y


def get_data_statistics(features, labels, coords):
    """
    获取数据统计信息
    """
    stats = {
        'n_samples': len(labels),
        'n_features': features.shape[1],
        'label_min': labels.min(),
        'label_max': labels.max(),
        'label_mean': labels.mean(),
        'label_std': labels.std(),
        'label_median': np.median(labels),
        'lat_range': (min(c[0] for c in coords), max(c[0] for c in coords)),
        'lon_range': (min(c[1] for c in coords), max(c[1] for c in coords)),
    }

    # 各区间样本数
    stats['n_low'] = (labels < 6).sum()
    stats['n_high'] = (labels >= 6).sum()
    stats['n_boundary'] = ((labels >= 5) & (labels <= 7)).sum()

    return stats


if __name__ == '__main__':
    # 测试数据加载
    print("测试数据加载...")
    features, labels, coords, metadata = load_real_data()

    stats = get_data_statistics(features, labels, coords)
    print(f"\n数据统计:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n构建图结构...")
    edge_index_tkg = create_spatial_graph(coords, k=5)
    edge_index_ekg = create_ecological_graph(features, k=5)

    print(f"  TKG边数: {edge_index_tkg.shape[1]}")
    print(f"  EKG边数: {edge_index_ekg.shape[1]}")

    # 计算图重叠度
    tkg_edges = set(tuple(e) for e in edge_index_tkg.t().numpy().tolist())
    ekg_edges = set(tuple(e) for e in edge_index_ekg.t().numpy().tolist())
    overlap = len(tkg_edges & ekg_edges)
    jaccard = overlap / len(tkg_edges | ekg_edges)

    print(f"\n图结构差异:")
    print(f"  共享边数: {overlap}")
    print(f"  Jaccard相似度: {jaccard:.4f} ({jaccard*100:.2f}%)")
    print(f"  结构差异度: {1-jaccard:.4f} ({(1-jaccard)*100:.2f}%)")
