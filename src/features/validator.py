#!/usr/bin/env python3
"""
特征验证和标准化模块
===================

本模块用于验证和修复特征值，确保训练和实时预测的分布一致。

主要功能:
1. validate_features: 检查特征值是否在合理范围内
2. fix_features: 修复超出范围的特征值
3. normalize_volume_features: 对成交量特征进行对数变换
4. check_feature_distribution: 检查特征分布统计

使用示例:
    from src.features.validator import validate_features, fix_features

    # 验证特征
    is_valid, issues = validate_features(df)
    if not is_valid:
        df = fix_features(df)

    # 标准化成交量特征
    df = normalize_volume_features(df)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# 定义特征值合理范围
FEATURE_RANGES = {
    # 技术指标 (0-100)
    'kdj_k': (0, 100),
    'kdj_d': (0, 100),
    'kdj_j': (-100, 200),
    'rsi_6': (0, 100),
    'rsi_14': (0, 100),
    'rsi_24': (0, 100),
    'rsi': (0, 100),
    # 威廉指标 (-100, 0)
    'williams_r': (-100, 0),
    # CCI (-300, 300)
    'cci': (-300, 300),
    # ADX (0, 100)
    'adx': (0, 100),
    # DI指标 (0, 100)
    'plus_di': (0, 100),
    'minus_di': (0, 100),
    # MACD相关（典型范围，但实际可能超出）
    'macd': (-50, 50),
    'macd_signal': (-50, 50),
    'macd_hist': (-30, 30),
    # 布林带位置 (0, 1)
    'bb_position': (-0.5, 1.5),  # 允许一定超出，因为价格可能突破布林带
}


def validate_features(df: pd.DataFrame, verbose: bool = True) -> Tuple[bool, List[str]]:
    """
    验证特征值是否在合理范围内

    Args:
        df: 特征数据框
        verbose: 是否打印详细信息

    Returns:
        (是否通过验证, 问题特征列表)
    """
    issues = []
    passed = True

    for col, (min_val, max_val) in FEATURE_RANGES.items():
        if col not in df.columns:
            continue

        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue

        actual_min = col_data.min()
        actual_max = col_data.max()
        actual_mean = col_data.mean()

        # 检查是否超出范围（允许一定容忍度）
        tolerance = 0.1
        if actual_min < min_val - tolerance or actual_max > max_val + tolerance:
            passed = False
            issues.append(f"{col}: 范围 [{actual_min:.2f}, {actual_max:.2f}] 超出 [{min_val}, {max_val}]")

            if verbose:
                print(f"⚠️  {col}:")
                print(f"    实际范围: [{actual_min:.2f}, {actual_max:.2f}]")
                print(f"    期望范围: [{min_val}, {max_val}]")
                print(f"    均值: {actual_mean:.2f}")

                # 找出异常值
                outliers_low = col_data[col_data < min_val]
                outliers_high = col_data[col_data > max_val]
                if len(outliers_low) > 0:
                    print(f"    低于最小值的样本数: {len(outliers_low)}")
                if len(outliers_high) > 0:
                    print(f"    高于最大值的样本数: {len(outliers_high)}")

    if verbose:
        if passed:
            print("✓ 所有特征值都在合理范围内")
        else:
            print(f"\n发现 {len(issues)} 个问题特征")

    return passed, issues


def fix_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    修复超出范围的特征值

    Args:
        df: 特征数据框

    Returns:
        修复后的数据框
    """
    df = df.copy()

    for col, (min_val, max_val) in FEATURE_RANGES.items():
        if col in df.columns:
            df[col] = df[col].clip(min_val, max_val)

    return df


def normalize_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    对成交量特征进行对数变换，减少极端值影响

    Args:
        df: 特征数据框

    Returns:
        处理后的数据框
    """
    df = df.copy()

    volume_cols = ['volume_ma5', 'volume_ma10', 'volume_ma20',
                   'turnover', 'turnover_ma5', 'volume']

    for col in volume_cols:
        if col in df.columns:
            # 对数变换: log(1 + x) 避免log(0)，并将大数值压缩
            df[col] = np.log1p(df[col].clip(lower=0))

    return df


def check_feature_distribution(df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """
    检查特征分布统计信息

    Args:
        df: 特征数据框
        feature_cols: 特征列名列表

    Returns:
        特征分布统计字典
    """
    stats = {}

    for col in feature_cols:
        if col not in df.columns:
            continue

        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue

        stats[col] = {
            'min': col_data.min(),
            'max': col_data.max(),
            'mean': col_data.mean(),
            'std': col_data.std(),
            'median': col_data.median(),
            'extreme_count': int((np.abs(col_data) > 1e6).sum()),
        }

    return stats


def print_feature_stats(stats: Dict, top_n: int = 10):
    """
    打印特征统计信息

    Args:
        stats: 特征统计字典
        top_n: 显示前N个异常特征
    """
    print("\n特征分布统计:")
    print("=" * 80)

    # 找出有极端值的特征
    extreme_features = [
        (col, info) for col, info in stats.items()
        if info.get('extreme_count', 0) > 0
    ]

    if extreme_features:
        print(f"\n⚠️  发现 {len(extreme_features)} 个特征有极端值:")
        for col, info in sorted(extreme_features, key=lambda x: x[1]['extreme_count'], reverse=True)[:top_n]:
            print(f"  {col}:")
            print(f"    范围: [{info['min']:.2e}, {info['max']:.2e}]")
            print(f"    均值: {info['mean']:.2e}, 标准差: {info['std']:.2e}")
            print(f"    极端值数量: {info['extreme_count']}")
    else:
        print("\n✓ 所有特征值分布正常")

    # 显示关键特征统计
    print("\n关键技术指标统计:")
    key_features = ['kdj_k', 'kdj_d', 'rsi', 'macd', 'bb_position']
    for col in key_features:
        if col in stats:
            info = stats[col]
            print(f"  {col}: [{info['min']:.2f}, {info['max']:.2f}], 均值={info['mean']:.2f}")


def validate_model_predictions(predictions: np.ndarray, threshold: float = 0.1) -> Tuple[bool, str]:
    """
    验证模型预测分布是否合理

    Args:
        predictions: 预测概率数组
        threshold: 异常阈值

    Returns:
        (是否通过验证, 诊断信息)
    """
    min_p = predictions.min()
    max_p = predictions.max()
    mean_p = predictions.mean()
    std_p = predictions.std()
    range_p = max_p - min_p

    issues = []

    # 检查概率范围
    if max_p < 0.5 + threshold:
        issues.append(f"最大预测概率 {max_p:.4f} 偏低")
    if min_p > 0.5 - threshold:
        issues.append(f"最小预测概率 {min_p:.4f} 偏高")
    if mean_p < 0.3 or mean_p > 0.7:
        issues.append(f"平均预测概率 {mean_p:.4f} 偏离 0.5 较远")
    if range_p < threshold:
        issues.append(f"预测概率范围 {range_p:.4f} 太小，模型区分度不足")
    if std_p < 0.05:
        issues.append(f"预测概率标准差 {std_p:.4f} 太小")

    if issues:
        return False, "; ".join(issues)

    return True, f"预测分布正常 (范围: [{min_p:.4f}, {max_p:.4f}], 均值: {mean_p:.4f}, 标准差: {std_p:.4f})"


if __name__ == "__main__":
    print("特征验证工具")
    print("=" * 80)
    print("用法：")
    print("  from src.features.validator import validate_features, fix_features")
    print("  from src.features.validator import normalize_volume_features")
    print("  from src.features.validator import check_feature_distribution, print_feature_stats")
    print()
    print("  # 验证和修复特征")
    print("  is_valid, issues = validate_features(df)")
    print("  if not is_valid:")
    print("      df = fix_features(df)")
    print()
    print("  # 标准化成交量特征")
    print("  df = normalize_volume_features(df)")
    print()
    print("  # 检查特征分布")
    print("  stats = check_feature_distribution(df, feature_cols)")
    print("  print_feature_stats(stats)")
    print("=" * 80)
