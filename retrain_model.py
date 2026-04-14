#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topaz 模型重新训练脚本
从历史数据生成特征和标签，重新训练集成模型
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ensemble_model import EnsembleModel
from feature_engineer import FeatureEngineer
from market_data import get_index_history
from quantpilot_data_api import get_history_data


def load_raw_data(data_path: str) -> pd.DataFrame:
    """加载原始数据"""
    print(f"📂 加载原始数据: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  原始数据: {len(df)} 行, {len(df.columns)} 列")
    return df


def generate_labels(df: pd.DataFrame, 
                    forward_days: int = 5,
                    return_threshold: float = 0.02) -> pd.DataFrame:
    """
    生成训练标签
    
    Args:
        df: 特征数据
        forward_days: 预测未来几天的收益
        return_threshold: 上涨阈值（超过此阈值视为上涨）
    
    Returns:
        添加了 target 列的数据
    """
    print(f"\n🏷️ 生成标签 (预测 {forward_days} 天收益，阈值 {return_threshold*100}%)...")
    
    # 确保按日期排序
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['code', 'date']).reset_index(drop=True)
    
    # 计算未来收益
    df['future_return'] = df.groupby('code')['close'].transform(
        lambda x: x.shift(-forward_days) / x - 1
    )
    
    # 生成标签
    # 1 = 上涨（收益超过阈值）
    # 0 = 下跌/持平
    df['target'] = (df['future_return'] > return_threshold).astype(int)
    
    # 统计标签分布
    total = len(df)
    valid = df['target'].notna().sum()
    up = df['target'].sum()
    down = valid - up
    
    print(f"  有效样本: {valid:,} / {total:,}")
    print(f"  上涨样本: {up:,} ({up/valid*100:.1f}%)")
    print(f"  下跌样本: {down:,} ({down/valid*100:.1f}%)")
    print(f"  不平衡比例: {down/up:.2f}:1" if up > 0 else "  ⚠️ 无上涨样本!")
    
    return df


def prepare_features(df: pd.DataFrame, fe: FeatureEngineer) -> Tuple[pd.DataFrame, List[str]]:
    """
    准备特征
    
    Args:
        df: 原始数据
        fe: 特征工程对象
    
    Returns:
        特征数据框和特征列名
    """
    print("\n🔧 生成特征...")
    
    # 检查是否已有特征列
    feature_keywords = ['ma5', 'volatility', 'rsi', 'macd', 'bb_', 'kdj', 'return_']
    has_features = any(kw in col for col in df.columns for kw in feature_keywords)
    
    if has_features:
        print("  数据已包含特征，跳过特征生成")
    else:
        print("  计算技术指标...")
        df = fe.generate_all_features(df)
    
    # 获取指数数据（如果需要）
    if 'index_close' not in df.columns:
        print("  添加指数因子...")
        try:
            index_history = get_index_history('000300.SH', days=500)
            if index_history is not None:
                df = fe.add_index_factors(df, index_history)
        except Exception as e:
            print(f"  ⚠️ 获取指数数据失败: {e}")
    
    # 填充缺失值
    df = df.fillna(0)
    
    # 定义特征列（原有45个 + 新增时间序列动量因子）
    feature_cols = [
        # 均线
        'ma5', 'ma10', 'ma20', 'ma60',
        'ma5_slope', 'ma10_slope', 'ma20_slope',
        'price_to_ma5', 'price_to_ma10', 'price_to_ma20',
        # 波动率（含 EWMA）
        'volatility_5', 'volatility_10', 'volatility_20', 'volatility_60',
        'vol_ewma', 'vol_regime', 'position_size',
        # 成交量
        'volume_ma5', 'volume_ratio',
        # 收益率
        'return_1d', 'return_5d', 'return_10d', 'return_20d',
        # 时间序列动量（新增）
        'tsmom_lb25', 'tsmom_lb60', 'tsmom_lb120',
        'ma_cross_5_20', 'ma_cross_10_50', 'trend_strength',
        'momentum_accel_5', 'momentum_accel_10',
        # 技术指标
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_position', 'kdj_k', 'kdj_d',
        # 均值回归（新增多周期）
        'mean_reversion_20', 'mean_reversion_60',
        'price_percentile_20', 'price_percentile_60',
        # 尾部风险（新增）
        'skewness_20', 'kurtosis_20', 'tail_risk', 'vol_spike',
        # 指数因子
        'index_close', 'index_return_1d', 'index_return_5d', 'index_return_20d',
        'index_ma_position', 'index_volatility',
        'relative_strength_1d', 'relative_strength_5d', 'relative_strength_20d',
        'beta',
        # 危机 Alpha（新增）
        'max_drawdown_20', 'dd_recovery', 'sharpe_proxy'
    ]
    
    # 检查特征是否存在
    available_features = [f for f in feature_cols if f in df.columns]
    missing_features = [f for f in feature_cols if f not in df.columns]
    
    if missing_features:
        print(f"  ⚠️ 缺失特征: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
        # 用 0 填充缺失特征
        for f in missing_features:
            df[f] = 0
        available_features = feature_cols
    
    print(f"  特征数量: {len(available_features)}")
    
    return df, available_features


def balance_samples(df: pd.DataFrame, method: str = 'undersample') -> pd.DataFrame:
    """
    平衡样本
    
    Args:
        df: 数据
        method: 平衡方法 ('undersample', 'oversample', 'none')
    
    Returns:
        平衡后的数据
    """
    if method == 'none':
        return df
    
    print(f"\n⚖️ 平衡样本 (方法: {method})...")
    
    # 分离正负样本
    up_samples = df[df['target'] == 1]
    down_samples = df[df['target'] == 0]
    
    print(f"  原始: 上涨 {len(up_samples):,}, 下跌 {len(down_samples):,}")
    
    if method == 'undersample':
        # 下采样：随机抽取多数类样本
        min_count = min(len(up_samples), len(down_samples))
        up_samples = up_samples.sample(n=min_count, random_state=42)
        down_samples = down_samples.sample(n=min_count, random_state=42)
    elif method == 'oversample':
        # 上采样：复制少数类样本
        max_count = max(len(up_samples), len(down_samples))
        if len(up_samples) < max_count:
            up_samples = up_samples.sample(n=max_count, replace=True, random_state=42)
        if len(down_samples) < max_count:
            down_samples = down_samples.sample(n=max_count, replace=True, random_state=42)
    
    balanced = pd.concat([up_samples, down_samples]).sample(frac=1, random_state=42)
    
    print(f"  平衡后: 上涨 {len(up_samples):,}, 下跌 {len(down_samples):,}")
    
    return balanced


def train_model(df: pd.DataFrame, feature_cols: List[str]) -> EnsembleModel:
    """
    训练模型
    
    Args:
        df: 训练数据
        feature_cols: 特征列
    
    Returns:
        训练好的模型
    """
    print("\n" + "="*60)
    print("🤖 训练集成模型")
    print("="*60)
    
    # 过滤有效样本
    df_clean = df.dropna(subset=feature_cols + ['target'])
    print(f"有效训练样本: {len(df_clean):,}")
    
    # 标签分布
    up = df_clean['target'].sum()
    down = len(df_clean) - up
    print(f"标签分布: 上涨 {up:,} ({up/len(df_clean)*100:.1f}%), 下跌 {down:,} ({down/len(df_clean)*100:.1f}%)")
    
    # 创建模型
    model = EnsembleModel(model_dir=os.path.dirname(os.path.abspath(__file__)))
    
    # 训练
    model.train(
        df=df_clean,
        feature_cols=feature_cols,
        target_col='target',
        test_size=0.2,
        n_folds=5
    )
    
    return model


def validate_model(model: EnsembleModel, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """
    验证模型
    
    Args:
        model: 训练好的模型
        df: 验证数据
        feature_cols: 特征列
    
    Returns:
        验证结果
    """
    print("\n📊 验证模型...")
    
    # 随机抽样测试
    sample = df.sample(n=min(100, len(df)), random_state=42)
    sample = sample.dropna(subset=feature_cols)
    
    X = sample[feature_cols].values
    
    predictions = model.predict(X)
    probabilities = predictions['probability']
    
    # 统计预测结果
    print(f"\n预测概率分布:")
    print(f"  min: {probabilities.min():.4f}")
    print(f"  max: {probabilities.max():.4f}")
    print(f"  mean: {probabilities.mean():.4f}")
    print(f"  median: {np.median(probabilities):.4f}")
    
    # 分位数
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        val = np.quantile(probabilities, q)
        print(f"  {q*100:.0f}%分位: {val:.4f}")
    
    return {
        'min': float(probabilities.min()),
        'max': float(probabilities.max()),
        'mean': float(probabilities.mean()),
        'median': float(np.median(probabilities))
    }


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='Topaz 模型重新训练')
    parser.add_argument('--data', type=str, default='training_data.csv', help='训练数据文件')
    parser.add_argument('--forward-days', type=int, default=5, help='预测未来几天收益')
    parser.add_argument('--threshold', type=float, default=0.02, help='上涨阈值')
    parser.add_argument('--balance', type=str, default='undersample', 
                        choices=['undersample', 'oversample', 'none'], help='样本平衡方法')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, args.data)
    
    print("="*60)
    print("Topaz 模型重新训练")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据文件: {data_path}")
    print(f"预测周期: {args.forward_days} 天")
    print(f"上涨阈值: {args.threshold*100}%")
    print(f"平衡方法: {args.balance}")
    
    # 1. 加载数据
    df = load_raw_data(data_path)
    
    # 2. 生成标签
    df = generate_labels(df, forward_days=args.forward_days, return_threshold=args.threshold)
    
    # 3. 准备特征
    fe = FeatureEngineer()
    df, feature_cols = prepare_features(df, fe)
    
    # 4. 平衡样本
    df = balance_samples(df, method=args.balance)
    
    # 5. 训练模型
    model = train_model(df, feature_cols)
    
    # 6. 验证模型
    validation = validate_model(model, df, feature_cols)
    
    # 7. 保存训练信息
    training_info = {
        'trained_at': datetime.now().isoformat(),
        'forward_days': args.forward_days,
        'return_threshold': args.threshold,
        'balance_method': args.balance,
        'feature_count': len(feature_cols),
        'validation': validation
    }
    
    info_path = os.path.join(base_dir, 'training_info.json')
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ 训练完成!")
    print("="*60)
    print(f"模型文件: {os.path.join(base_dir, 'ensemble_model.pkl')}")
    print(f"训练信息: {info_path}")


if __name__ == '__main__':
    main()