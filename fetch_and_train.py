#!/usr/bin/env python3
"""
数据获取和模型训练完整流程
=======================

本脚本执行以下步骤:
1. 获取沪深300历史数据（使用腾讯接口，支持更长历史）
2. 生成特征（使用修复后的特征工程）
3. 验证特征分布
4. 训练模型
5. 验证模型性能

使用方法:
    python fetch_and_train.py

注意:
    - 数据获取可能需要 5-10 分钟
    - 模型训练可能需要 10-20 分钟
    - 建议在非交易时间运行
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantpilot_data_api import get_qq_history
from feature_engineer import FeatureEngineer
from market_data import get_index_history
from ensemble_model import EnsembleModel
from feature_validator import validate_features, fix_features, check_feature_distribution, print_feature_stats, validate_model_predictions


def fetch_stock_data(symbol, days=500):
    """获取单只股票历史数据"""
    try:
        df = get_qq_history(symbol, days=days)
        if df is not None and len(df) >= 60:
            return symbol, df
    except Exception as e:
        print(f"    ✗ {symbol}: {e}")
    return symbol, None


def fetch_all_data(max_workers=8):
    """获取所有股票数据"""
    print("\n" + "=" * 80)
    print("步骤 1: 获取沪深300历史数据")
    print("=" * 80)

    # 加载股票列表
    import json
    with open('csi300_stocks.json', 'r', encoding='utf-8') as f:
        stocks = json.load(f)

    print(f"\n  股票总数: {len(stocks)}")
    print(f"  并行度: {max_workers}")
    print(f"  预计时间: {len(stocks) * 2 // max_workers} 秒")

    all_data = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_stock_data, stock['code'], 500): stock for stock in stocks}

        completed = 0
        for future in as_completed(futures):
            symbol, df = future.result()
            completed += 1

            if df is not None:
                # 添加股票代码和名称
                stock_info = futures[future]
                df['code'] = symbol
                df['name'] = stock_info.get('name', '')
                all_data.append(df)

            if completed % 10 == 0 or completed == len(stocks):
                print(f"  进度: {completed}/{len(stocks)} ({len(all_data)} 只成功)")

    if len(all_data) == 0:
        print("✗ 没有获取到任何数据")
        return None

    # 合并数据
    combined = pd.concat(all_data, ignore_index=True)
    combined['date'] = pd.to_datetime(combined.index if 'datetime' in combined.columns else combined['date'])

    print(f"\n  ✓ 获取完成: {len(combined):,} 条记录")
    print(f"  股票数: {combined['code'].nunique()}")
    print(f"  日期范围: {combined['date'].min()} 到 {combined['date'].max()}")

    # 保存原始数据
    output_file = 'csi300_full_history.csv'
    combined.to_csv(output_file, index=False)
    print(f"  ✓ 数据已保存: {output_file}")

    return combined


def generate_features(df):
    """生成特征"""
    print("\n" + "=" * 80)
    print("步骤 2: 生成特征")
    print("=" * 80)

    # 生成标签
    print("\n  生成标签...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date']).reset_index(drop=True)

    df['future_return'] = df.groupby('code')['close'].transform(
        lambda x: x.shift(-5) / x - 1
    )
    df['target'] = (df['future_return'] > 0.02).astype(int)

    valid = df['target'].notna().sum()
    up = df['target'].sum()
    down = valid - up
    print(f"    有效样本: {valid:,}")
    print(f"    上涨: {up:,} ({up/valid*100:.1f}%)")
    print(f"    下跌: {down:,} ({down/valid*100:.1f}%)")

    # 生成特征
    print("\n  生成技术指标...")
    fe = FeatureEngineer()
    df = df.groupby('code', group_keys=False).apply(fe._generate_features_for_stock)

    # 添加指数因子
    print("  添加指数因子...")
    try:
        index_history = get_index_history('000300.SH', days=500)
        if index_history is not None:
            df = fe.add_index_factors(df, index_history)
            print(f"    ✓ 指数数据已添加")
    except Exception as e:
        print(f"    ⚠️ 指数数据获取失败: {e}")

    df = df.fillna(0)

    print(f"\n  ✓ 特征生成完成: {len(df):,} 行")

    return df


def validate_and_fix_features(df):
    """验证和修复特征"""
    print("\n" + "=" * 80)
    print("步骤 3: 验证和修复特征")
    print("=" * 80)

    print("\n  验证特征值范围...")
    is_valid, issues = validate_features(df, verbose=True)

    if not is_valid:
        print(f"\n  修复 {len(issues)} 个问题特征...")
        df = fix_features(df)
        print("  ✓ 特征值已修复")
    else:
        print("  ✓ 所有特征值都在合理范围内")

    # 检查特征分布
    feature_cols = [
        'ma5', 'ma10', 'ma20', 'ma60',
        'ma5_slope', 'ma10_slope', 'ma20_slope',
        'price_to_ma5', 'price_to_ma10', 'price_to_ma20',
        'volatility_5', 'volatility_10', 'volatility_20', 'volatility_60',
        'vol_ewma', 'vol_regime', 'position_size',
        'volume_ma5', 'volume_ratio',
        'return_1d', 'return_5d', 'return_10d', 'return_20d',
        'tsmom_lb25', 'tsmom_lb60', 'tsmom_lb120',
        'ma_cross_5_20', 'ma_cross_10_50', 'trend_strength',
        'momentum_accel_5', 'momentum_accel_10',
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_position', 'kdj_k', 'kdj_d',
        'mean_reversion_20', 'mean_reversion_60',
        'price_percentile_20', 'price_percentile_60',
        'skewness_20', 'kurtosis_20', 'tail_risk', 'vol_spike',
        'index_close', 'index_return_1d', 'index_return_5d', 'index_return_20d',
        'index_ma_position', 'index_volatility',
        'relative_strength_1d', 'relative_strength_5d', 'relative_strength_20d',
        'beta',
        'max_drawdown_20', 'dd_recovery', 'sharpe_proxy'
    ]

    available_features = [f for f in feature_cols if f in df.columns]
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        print(f"  ⚠️ 缺失 {len(missing_features)} 个特征，用0填充")
        for f in missing_features:
            df[f] = 0

    print(f"\n  使用 {len(available_features)} 个特征")

    stats = check_feature_distribution(df, available_features)
    print_feature_stats(stats)

    return df, available_features


def train_model(df, feature_cols):
    """训练模型"""
    print("\n" + "=" * 80)
    print("步骤 4: 训练模型")
    print("=" * 80)

    # 平衡样本
    print("\n  平衡样本...")
    df_up = df[df['target'] == 1]
    df_down = df[df['target'] == 0]
    min_count = min(len(df_up), len(df_down))
    df_balanced = pd.concat([
        df_up.sample(n=min_count, random_state=42),
        df_down.sample(n=min_count, random_state=42)
    ]).sample(frac=1, random_state=42)
    print(f"    平衡后: {len(df_balanced):,} 样本")

    # 过滤有效样本
    df_clean = df_balanced.dropna(subset=feature_cols + ['target'])
    print(f"    有效训练样本: {len(df_clean):,}")

    # 创建和训练模型
    print("\n  训练集成模型...")
    model = EnsembleModel(model_dir='.')

    success = model.train(
        df=df_clean,
        feature_cols=feature_cols,
        target_col='target',
        test_size=0.2,
        n_folds=5
    )

    if not success:
        print("\n✗ 模型训练失败")
        return None

    return model


def validate_and_save(model, df, feature_cols):
    """验证和保存模型"""
    print("\n" + "=" * 80)
    print("步骤 5: 验证和保存模型")
    print("=" * 80)

    # 验证预测分布
    print("\n  验证预测分布...")
    sample = df.sample(n=min(1000, len(df)), random_state=42)
    sample = sample.dropna(subset=feature_cols)

    X = sample[feature_cols].values
    result = model.predict(X)
    probabilities = result['probability']

    print(f"    样本数: {len(probabilities)}")
    print(f"    min: {probabilities.min():.4f}")
    print(f"    max: {probabilities.max():.4f}")
    print(f"    mean: {probabilities.mean():.4f}")
    print(f"    median: {np.median(probabilities):.4f}")
    print(f"    std: {probabilities.std():.4f}")

    is_valid, message = validate_model_predictions(probabilities)
    if is_valid:
        print(f"    ✓ {message}")
    else:
        print(f"    ⚠️ {message}")

    # 保存模型
    print("\n  模型已保存至 ensemble_model.pkl")

    # 保存训练信息
    training_info = {
        'trained_at': datetime.now().isoformat(),
        'feature_count': len(feature_cols),
        'features_fixed': True,
        'data_fresh': True,
        'note': '使用最新特征工程生成的数据和模型'
    }

    import json
    with open('training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    print("    ✓ 训练信息已保存")

    return True


def main():
    print("=" * 80)
    print("Topaz 数据获取和模型训练")
    print("=" * 80)
    print("\n本脚本将:")
    print("  1. 获取沪深300历史数据（腾讯接口）")
    print("  2. 生成技术指标特征（使用修复后的特征工程）")
    print("  3. 验证特征分布")
    print("  4. 训练集成模型")
    print("  5. 验证和保存模型")
    print()

    # 步骤1: 获取数据
    df = fetch_all_data(max_workers=8)
    if df is None:
        print("\n✗ 数据获取失败")
        return

    # 步骤2: 生成特征
    df = generate_features(df)

    # 步骤3: 验证和修复特征
    df, feature_cols = validate_and_fix_features(df)

    # 步骤4: 训练模型
    model = train_model(df, feature_cols)
    if model is None:
        print("\n✗ 模型训练失败")
        return

    # 步骤5: 验证和保存
    validate_and_save(model, df, feature_cols)

    print("\n" + "=" * 80)
    print("✓ 全部完成!")
    print("=" * 80)
    print("\n后续操作:")
    print("  1. 运行测试: python daily_decision.py")
    print("  2. 检查预测分布是否正常")
    print("  3. 配置定时任务（可选）")


if __name__ == "__main__":
    main()
