# -*- coding: utf-8 -*-
"""
训练工具
========
包含分段训练和普通训练两种模式
修复版本：测试集使用y_hat_init做regime assignment，避免标签泄露
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# 默认超参数
DEFAULT_CONFIG = {
    'epochs': 300,
    'lr': 0.005,
    'weight_decay': 1e-4,
    'hidden_dim': 64,
    'dropout': 0.3,
    'heads': 2,
    'threshold': 6.0,  # 分段阈值
}


def train_single_model(model, x, y, edge_index, train_idx, device,
                       epochs=300, lr=0.005, weight_decay=1e-4,
                       is_fusion=False, edge_index_ekg=None):
    """
    训练单个模型（不分段）
    """
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    if edge_index_ekg is not None:
        edge_index_ekg = edge_index_ekg.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        if is_fusion:
            pred, _ = model(x, edge_index, edge_index_ekg)
        else:
            pred = model(x, edge_index)

        loss = criterion(pred[train_idx], y[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return model


def evaluate_model(model, x, y, edge_index, test_idx, device, scaler_y,
                   is_fusion=False, edge_index_ekg=None):
    """
    评估模型性能
    """
    model.eval()
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    if edge_index_ekg is not None:
        edge_index_ekg = edge_index_ekg.to(device)

    with torch.no_grad():
        if is_fusion:
            pred, attn_weights = model(x, edge_index, edge_index_ekg)
        else:
            pred = model(x, edge_index)
            attn_weights = None

        pred_np = pred.cpu().numpy()
        y_np = y.cpu().numpy()

    # 逆变换到原始尺度
    pred_original = scaler_y.inverse_transform(pred_np.reshape(-1, 1)).flatten()
    y_original = scaler_y.inverse_transform(y_np.reshape(-1, 1)).flatten()

    # 计算测试集指标
    test_pred = pred_original[test_idx]
    test_true = y_original[test_idx]

    metrics = {
        'r2': r2_score(test_true, test_pred),
        'mae': mean_absolute_error(test_true, test_pred),
        'rmse': np.sqrt(mean_squared_error(test_true, test_pred)),
    }

    return metrics, pred_original, attn_weights


def train_segmented_model(model_class, features, labels, edge_index, train_idx, test_idx,
                          device, threshold=6.0, is_fusion=False, edge_index_ekg=None,
                          epochs=300, lr=0.005, weight_decay=1e-4, **model_kwargs):
    """
    分段训练模型（修复版）
    分别训练低值段(<threshold)和高值段(>=threshold)模型

    关键修复：
    - 训练集：用真实y做regime assignment（合理，因为训练时知道标签）
    - 测试集：用RF预测的y_hat_init做regime assignment（避免标签泄露）
    """
    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # 仅在训练集上fit
    scaler_X.fit(features[train_idx])
    scaler_y.fit(labels[train_idx].reshape(-1, 1))

    features_scaled = scaler_X.transform(features)
    labels_scaled = scaler_y.transform(labels.reshape(-1, 1)).flatten()

    x = torch.tensor(features_scaled, dtype=torch.float32)
    y = torch.tensor(labels_scaled, dtype=torch.float32)

    # ========== 关键修复：用non-segmented baseline预测y_hat_init ==========
    # 方案：先训练一个non-segmented模型，用其输出作为y_hat_init
    # 这比RF更合理，因为：1) 和论文描述一致 2) 利用了GNN的能力

    # 训练non-segmented baseline模型
    baseline_model = model_class(**model_kwargs)
    baseline_model = baseline_model.to(device)

    x_dev = x.to(device)
    y_dev = y.to(device)
    edge_index_dev = edge_index.to(device)
    edge_index_ekg_dev = edge_index_ekg.to(device) if edge_index_ekg is not None else None

    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # 训练baseline（用全部训练集，不分段）
    for epoch in range(epochs):
        baseline_model.train()
        optimizer.zero_grad()
        if is_fusion:
            pred, _ = baseline_model(x_dev, edge_index_dev, edge_index_ekg_dev)
        else:
            pred = baseline_model(x_dev, edge_index_dev)
        loss = criterion(pred[train_idx], y_dev[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), max_norm=1.0)
        optimizer.step()

    # 用baseline预测y_hat_init
    baseline_model.eval()
    with torch.no_grad():
        if is_fusion:
            pred_init, _ = baseline_model(x_dev, edge_index_dev, edge_index_ekg_dev)
        else:
            pred_init = baseline_model(x_dev, edge_index_dev)
        pred_init_np = pred_init.cpu().numpy()

    # 逆变换到原始尺度
    y_hat_init = scaler_y.inverse_transform(pred_init_np.reshape(-1, 1)).flatten()

    # 训练集：用真实y做regime assignment（合理）
    train_low = np.array([i for i in train_idx if labels[i] < threshold])
    train_high = np.array([i for i in train_idx if labels[i] >= threshold])

    # 测试集：用y_hat_init做regime assignment（避免泄露）
    test_low = np.array([i for i in test_idx if y_hat_init[i] < threshold])
    test_high = np.array([i for i in test_idx if y_hat_init[i] >= threshold])
    # ============================================================================

    # 存储预测
    all_predictions_scaled = np.zeros(len(labels))
    models = {}

    for regime_name, regime_train_idx in [('low', train_low), ('high', train_high)]:
        if len(regime_train_idx) < 50:
            continue

        # 创建新模型
        model = model_class(**model_kwargs)
        model = model.to(device)

        x_dev = x.to(device)
        y_dev = y.to(device)
        edge_index_dev = edge_index.to(device)
        edge_index_ekg_dev = edge_index_ekg.to(device) if edge_index_ekg is not None else None

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        # 训练（仅用该regime的训练样本）
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            if is_fusion:
                pred, _ = model(x_dev, edge_index_dev, edge_index_ekg_dev)
            else:
                pred = model(x_dev, edge_index_dev)

            loss = criterion(pred[regime_train_idx], y_dev[regime_train_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        models[regime_name] = model

    # 预测：根据y_hat_init分配到对应模型
    for regime_name, model in models.items():
        model.eval()
        with torch.no_grad():
            x_dev = x.to(device)
            edge_index_dev = edge_index.to(device)
            edge_index_ekg_dev = edge_index_ekg.to(device) if edge_index_ekg is not None else None

            if is_fusion:
                pred_all, _ = model(x_dev, edge_index_dev, edge_index_ekg_dev)
            else:
                pred_all = model(x_dev, edge_index_dev)
            pred_np = pred_all.cpu().numpy()

        # 训练集：用真实y分配
        if regime_name == 'low':
            for i in train_low:
                all_predictions_scaled[i] = pred_np[i]
        else:
            for i in train_high:
                all_predictions_scaled[i] = pred_np[i]

        # 测试集：用y_hat_init分配（关键修复！）
        if regime_name == 'low':
            for i in test_low:
                all_predictions_scaled[i] = pred_np[i]
        else:
            for i in test_high:
                all_predictions_scaled[i] = pred_np[i]

    # 逆变换
    all_predictions = scaler_y.inverse_transform(all_predictions_scaled.reshape(-1, 1)).flatten()

    # 计算测试集指标
    test_pred = all_predictions[test_idx]
    test_true = labels[test_idx]

    metrics = {
        'r2': r2_score(test_true, test_pred),
        'mae': mean_absolute_error(test_true, test_pred),
        'rmse': np.sqrt(mean_squared_error(test_true, test_pred)),
    }

    return metrics, all_predictions


def compute_boundary_metrics(predictions, labels, test_idx, boundary_low=5.0, boundary_high=7.0):
    """
    计算边界区域的指标
    """
    boundary_mask = (labels >= boundary_low) & (labels <= boundary_high)
    test_boundary_mask = np.array([boundary_mask[i] for i in test_idx])
    test_boundary_idx = test_idx[test_boundary_mask]

    if len(test_boundary_idx) < 5:
        return None

    boundary_pred = predictions[test_boundary_idx]
    boundary_true = labels[test_boundary_idx]

    metrics = {
        'n_samples': len(test_boundary_idx),
        'mae': mean_absolute_error(boundary_true, boundary_pred),
        'rmse': np.sqrt(mean_squared_error(boundary_true, boundary_pred)),
        'pred_std': np.std(boundary_pred),
        'error_std': np.std(np.abs(boundary_true - boundary_pred)),
    }

    return metrics
