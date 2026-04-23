# -*- coding: utf-8 -*-
"""
GNN模型定义
===========
包含：TKG-only, EKG-only, Fusion三种模型
所有模型使用相同的架构参数，确保公平对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class TKGOnlyModel(nn.Module):
    """
    TKG-only模型：仅使用空间拓扑图
    使用GAT (Graph Attention Network)
    """
    def __init__(self, input_dim, hidden_dim=64, heads=2, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        h = F.elu(self.conv2(h, edge_index))
        return self.fc(h).squeeze(-1)


class EKGOnlyModel(nn.Module):
    """
    EKG-only模型：仅使用生态特征图
    使用GCN (Graph Convolutional Network)
    """
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.fc(h).squeeze(-1)


class FusionModel(nn.Module):
    """
    双图融合模型：TKG + EKG + 交叉注意力
    """
    def __init__(self, input_dim, hidden_dim=64, heads=2, dropout=0.3):
        super().__init__()
        # TKG分支 (GAT)
        self.tkg_conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.tkg_conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)

        # EKG分支 (GCN)
        self.ekg_conv1 = GCNConv(input_dim, hidden_dim)
        self.ekg_conv2 = GCNConv(hidden_dim, hidden_dim)

        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=heads, dropout=dropout, batch_first=True)

        # 预测头
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index_tkg, edge_index_ekg):
        # TKG分支
        h_tkg = F.elu(self.tkg_conv1(x, edge_index_tkg))
        h_tkg = F.elu(self.tkg_conv2(h_tkg, edge_index_tkg))

        # EKG分支
        h_ekg = F.relu(self.ekg_conv1(x, edge_index_ekg))
        h_ekg = F.relu(self.ekg_conv2(h_ekg, edge_index_ekg))

        # 交叉注意力融合
        h_tkg_batch = h_tkg.unsqueeze(0)
        h_ekg_batch = h_ekg.unsqueeze(0)
        h_attn, attn_weights = self.cross_attn(h_tkg_batch, h_ekg_batch, h_ekg_batch)
        h_attn = h_attn.squeeze(0)

        # 拼接并预测
        h_fused = torch.cat([h_tkg, h_attn], dim=1)
        return self.fc(h_fused).squeeze(-1), attn_weights

class SoftGatedSegmentedModel(nn.Module):
    """
    Soft-Gated分段模型（Run-2方案）
    """
    def __init__(self, input_dim, hidden_dim=64, heads=2, dropout=0.3):
        super().__init__()

        # Low-biomass expert
        self.low_conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.low_conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.low_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # High-biomass expert
        self.high_conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.high_conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.high_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Gate network
        self.gate_conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.gate_conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        # Low expert
        h_low = F.elu(self.low_conv1(x, edge_index))
        h_low = F.elu(self.low_conv2(h_low, edge_index))
        y_low = self.low_fc(h_low).squeeze(-1)

        # High expert
        h_high = F.elu(self.high_conv1(x, edge_index))
        h_high = F.elu(self.high_conv2(h_high, edge_index))
        y_high = self.high_fc(h_high).squeeze(-1)

        # Gate
        h_gate = F.elu(self.gate_conv1(x, edge_index))
        h_gate = F.elu(self.gate_conv2(h_gate, edge_index))
        g = self.gate_fc(h_gate).squeeze(-1)

        # Soft combination
        y_pred = (1 - g) * y_low + g * y_high

        return y_pred, g
