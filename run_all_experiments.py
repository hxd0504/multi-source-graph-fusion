# -*- coding: utf-8 -*-
"""
运行所有实验
============
一键运行所有实验并生成汇总报告
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# 实验列表
EXPERIMENTS = [
    ('exp1_segmentation_ablation.py', '分段建模消融实验'),
    ('exp2_tkg_ekg_ablation.py', 'TKG/EKG消融实验'),
    ('exp3_boundary_robustness.py', '边界区域稳健性测试'),
    ('exp4_spatial_generalization.py', '空间泛化能力测试'),
    ('exp5_graph_structure_analysis.py', '图结构分析'),
]

def run_experiment(script_name, description):
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"运行: {description}")
    print(f"脚本: {script_name}")
    print('='*80)

    script_path = os.path.join(os.path.dirname(__file__), 'experiments', script_name)

    start_time = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True
    )
    elapsed = time.time() - start_time

    status = "成功" if result.returncode == 0 else "失败"
    print(f"\n状态: {status}, 耗时: {elapsed:.1f}秒")

    return result.returncode == 0, elapsed


def generate_summary_report():
    """生成汇总报告"""
    import pandas as pd

    results_dir = os.path.join(os.path.dirname(__file__), 'results')

    report = []
    report.append("="*80)
    report.append("实验结果汇总报告")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*80)

    # 实验1：分段建模
    try:
        df = pd.read_csv(os.path.join(results_dir, 'exp1_summary.csv'))
        report.append("\n【实验1：分段建模消融】")
        report.append(f"  无分段基线 R²: {df['Baseline_R2_mean'].values[0]:.4f} ± {df['Baseline_R2_std'].values[0]:.4f}")
        report.append(f"  分段模型 R²:   {df['Segmented_R2_mean'].values[0]:.4f} ± {df['Segmented_R2_std'].values[0]:.4f}")
        report.append(f"  性能提升:      +{df['Improvement_pct'].values[0]:.1f}%")
    except Exception as e:
        report.append(f"\n【实验1】读取失败: {e}")

    # 实验2：TKG/EKG消融
    try:
        df = pd.read_csv(os.path.join(results_dir, 'exp2_tkg_ekg_ablation.csv'))
        report.append("\n【实验2：TKG/EKG消融】")
        report.append(f"  TKG-only R²:  {df['TKG_r2'].mean():.4f} ± {df['TKG_r2'].std():.4f}")
        report.append(f"  EKG-only R²:  {df['EKG_r2'].mean():.4f} ± {df['EKG_r2'].std():.4f}")
        report.append(f"  Fusion R²:    {df['Fusion_r2'].mean():.4f} ± {df['Fusion_r2'].std():.4f}")
    except Exception as e:
        report.append(f"\n【实验2】读取失败: {e}")

    # 实验3：边界稳健性
    try:
        df = pd.read_csv(os.path.join(results_dir, 'exp3_summary.csv'))
        report.append("\n【实验3：边界区域稳健性】")
        for _, row in df.iterrows():
            report.append(f"  {row['Model']}: 边界MAE={row['Boundary_MAE_mean']:.4f}, 误差Std={row['Boundary_ErrorStd_mean']:.4f}")
    except Exception as e:
        report.append(f"\n【实验3】读取失败: {e}")

    # 实验4：空间泛化
    try:
        df = pd.read_csv(os.path.join(results_dir, 'exp4_summary.csv'))
        report.append("\n【实验4：空间泛化能力】")
        for _, row in df.iterrows():
            report.append(f"  {row['Model']}: R²={row['R2_mean']:.4f} ± {row['R2_std']:.4f}, CV={row['CV_pct']:.2f}%")
    except Exception as e:
        report.append(f"\n【实验4】读取失败: {e}")

    # 实验5：图结构
    try:
        df = pd.read_csv(os.path.join(results_dir, 'exp5_graph_structure.csv'))
        report.append("\n【实验5：图结构分析】")
        for _, row in df.iterrows():
            report.append(f"  {row['metric']}: {row['value']:.4f}")
    except Exception as e:
        report.append(f"\n【实验5】读取失败: {e}")

    report.append("\n" + "="*80)

    # 保存报告
    report_text = "\n".join(report)
    with open(os.path.join(results_dir, 'summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n报告已保存: results/summary_report.txt")


if __name__ == '__main__':
    print("="*80)
    print("开始运行所有实验")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    total_start = time.time()
    results = []

    for script, desc in EXPERIMENTS:
        success, elapsed = run_experiment(script, desc)
        results.append((desc, success, elapsed))

    total_elapsed = time.time() - total_start

    # 汇总
    print("\n" + "="*80)
    print("实验运行汇总")
    print("="*80)

    for desc, success, elapsed in results:
        status = "✓" if success else "✗"
        print(f"  {status} {desc}: {elapsed:.1f}秒")

    print(f"\n总耗时: {total_elapsed/60:.1f}分钟")

    # 生成报告
    print("\n生成汇总报告...")
    generate_summary_report()

    print("\n" + "="*80)
    print("所有实验完成!")
    print("="*80)
