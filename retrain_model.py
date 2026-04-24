#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Topaz 模型重新训练脚本
================================================================================

【模块说明】
本脚本用于从历史股票数据重新训练 Topaz 选股模型的集成学习模型。
模型采用多算法集成（XGBoost + LightGBM + CatBoost + Random Forest），通过技术指标
和指数因子预测股票未来收益是否超过指定阈值。

【训练流程】
1. 加载原始数据 -> 从 CSV 文件读取股票历史数据
2. 生成训练标签 -> 根据未来收益率计算正负样本
3. 准备特征数据 -> 计算技术指标、指数因子等 65+ 个特征
4. 平衡样本 -> 处理上涨/下跌样本不平衡问题
5. 训练模型 -> 使用 5 折交叉验证训练集成模型
6. 验证模型 -> 检查模型预测概率分布是否正常
7. 保存结果 -> 模型文件和训练信息持久化

【数据格式要求】
输入 CSV 必须包含以下列：
  - code: 股票代码（如 600000.SH）
  - date: 交易日期（YYYY-MM-DD 格式）
  - open: 开盘价
  - high: 最高价
  - low: 最低价
  - close: 收盘价
  - volume: 成交量

【训练注意事项】
- 数据量建议：至少 50000 条记录（约 100 只股票 × 2 年数据）
- 训练时间：约 5-15 分钟（取决于数据量和硬件配置）
- 样本平衡：默认使用下采样，避免模型偏向多数类
- 内存需求：建议 8GB 以上内存处理大规模数据

【使用示例】
# 基本用法
python retrain_model.py --data training_data.csv

# 自定义参数
python retrain_model.py --data training_data.csv --forward-days 10 --threshold 0.03 --balance oversample

【输出文件】
- ensemble_model.pkl: 训练好的集成模型文件
- training_info.json: 训练配置和验证结果

作者: Topaz Team
版本: 3.0
================================================================================
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
    """
    加载原始训练数据
    
    【数据格式要求】
    输入 CSV 文件必须包含以下必需列：
      - code: 股票代码，格式如 '600000.SH'（上交所）或 '000001.SZ'（深交所）
      - date: 交易日期，格式为 'YYYY-MM-DD' 或 'YYYY/MM/DD'
      - open: 开盘价（元）
      - high: 最高价（元）
      - low: 最低价（元）
      - close: 收盘价（元）
      - volume: 成交量（手或股均可，保持一致性即可）
    
    【可选列】
    如果数据已包含特征列（如 ma5, rsi, macd 等），将跳过特征计算步骤。
    
    【数据质量要求】
    - 数据需按 (code, date) 排序，如未排序会自动处理
    - 缺失值会被填充为 0，建议事先清洗数据
    - 建议至少 50000 条记录以获得稳定的训练效果
    
    Args:
        data_path: 训练数据 CSV 文件的完整路径
    
    Returns:
        pd.DataFrame: 加载的原始数据框
    
    Raises:
        FileNotFoundError: 数据文件不存在
        pd.errors.EmptyDataError: 数据文件为空
    """
    print(f"📂 加载原始数据: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  原始数据: {len(df)} 行, {len(df.columns)} 列")
    return df


def generate_labels(df: pd.DataFrame, 
                    forward_days: int = 5,
                    return_threshold: float = 0.02) -> pd.DataFrame:
    """
    生成训练标签（目标变量）
    
    【标签生成逻辑】
    1. 计算每只股票在未来 N 天的收益率
    2. 收益率超过阈值 -> 标签 = 1（上涨样本）
    3. 收益率未超过阈值 -> 标签 = 0（下跌/持平样本）
    
    【参数说明】
    forward_days: 预测周期
        - 默认 5 天，适合短线交易策略
        - 1-3 天：超短线，噪音较大
        - 5-10 天：短线，推荐范围
        - 10-20 天：中线，趋势更明显但信号滞后
    
    return_threshold: 上涨阈值
        - 默认 2%，表示收益超过 2% 才视为上涨
        - 1-2%：保守设置，正样本较多
        - 2-5%：标准设置，平衡正负样本
        - 5%+：激进设置，正样本较少但质量高
    
    【调优建议】
    - 如果正样本比例 < 20%，考虑降低阈值或使用上采样
    - 如果正样本比例 > 50%，考虑提高阈值以增加区分度
    - 不平衡比例建议控制在 1:1 到 3:1 之间
    
    Args:
        df: 包含 'code', 'date', 'close' 列的特征数据
        forward_days: 预测未来几天的收益，默认 5 天
        return_threshold: 上涨阈值，默认 0.02 (2%)
    
    Returns:
        pd.DataFrame: 添加了 target 和 future_return 列的数据
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
    准备模型训练特征
    
    【特征分类说明】
    本函数生成/整理 65+ 个技术指标和因子，分为以下类别：
    
    1. 均线类特征（10个）
       - ma5, ma10, ma20, ma60: 不同周期移动平均线
       - ma5_slope, ma10_slope, ma20_slope: 均线斜率，判断趋势方向
       - price_to_ma5/10/20: 价格相对均线位置，判断超买超卖
    
    2. 波动率特征（7个）
       - volatility_5/10/20/60: 不同周期历史波动率
       - vol_ewma: 指数加权移动平均波动率
       - vol_regime: 波动率状态（高/低波动）
       - position_size: 基于波动率的仓位建议
    
    3. 成交量特征（2个）
       - volume_ma5: 5日成交量均线
       - volume_ratio: 当日成交量/均量，判断放量缩量
    
    4. 收益率特征（4个）
       - return_1d/5d/10d/20d: 不同周期收益率
    
    5. 时间序列动量特征（新增，8个）
       - tsmom_lb25/60/120: 多周期动量因子
       - ma_cross_5_20, ma_cross_10_50: 均线交叉信号
       - trend_strength: 趋势强度
       - momentum_accel_5/10: 动量加速度
    
    6. 技术指标特征（7个）
       - rsi: 相对强弱指标
       - macd, macd_signal, macd_hist: MACD 指标组
       - bb_position: 布林带位置
       - kdj_k, kdj_d: KDJ 指标
    
    7. 均值回归特征（新增，4个）
       - mean_reversion_20/60: 多周期均值回归
       - price_percentile_20/60: 价格分位数
    
    8. 尾部风险特征（新增，4个）
       - skewness_20: 偏度
       - kurtosis_20: 峰度
       - tail_risk: 尾部风险指标
       - vol_spike: 波动率突变
    
    9. 指数因子特征（9个）
       - index_close/return: 指数价格和收益
       - index_ma_position/volatility: 指数技术指标
       - relative_strength_1d/5d/20d: 相对强弱
       - beta: 个股 Beta 值
    
    10. 危机 Alpha 特征（新增，3个）
        - max_drawdown_20: 最大回撤
        - dd_recovery: 回撤恢复
        - sharpe_proxy: 夏普比率代理
    
    【特征工程注意事项】
    - 缺失特征会自动填充为 0，建议检查数据完整性
    - 指数数据需要联网获取，失败时会使用默认值
    - 特征计算依赖历史数据，数据开头部分可能有 NaN
    
    Args:
        df: 包含 OHLCV 数据的原始数据框
        fe: FeatureEngineer 特征工程对象
    
    Returns:
        Tuple[pd.DataFrame, List[str]]: 
            - 添加特征后的数据框
            - 特征列名列表（共 65 个特征）
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

    # 验证特征值范围（修复后的关键步骤）
    print("  验证特征值范围...")
    from validate_features import validate_features, fix_features
    is_valid, issues = validate_features(df, verbose=True)
    if not is_valid:
        print(f"  发现 {len(issues)} 个问题特征，正在修复...")
        df = fix_features(df)
        print("  ✓ 特征值已修复")

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
    平衡训练样本的正负比例
    
    【样本不平衡问题】
    股票数据通常上涨样本少于下跌样本，比例可能达到 1:3 甚至 1:5。
    不平衡数据会导致模型偏向预测多数类（下跌），降低上涨识别率。
    
    【平衡方法说明】
    1. undersample（下采样，默认推荐）
       - 从多数类随机抽取与少数类等量的样本
       - 优点：训练速度快，减少过拟合风险
       - 缺点：丢弃部分数据，可能损失信息
       - 适用：数据量充足（>50000 条）时推荐使用
    
    2. oversample（上采样）
       - 复制少数类样本直到与多数类等量
       - 优点：保留所有数据，不丢失信息
       - 缺点：训练时间增加，可能导致过拟合
       - 适用：数据量较少（<30000 条）时推荐使用
    
    3. none（不处理）
       - 保持原始样本比例
       - 适用：样本比例接近 1:1 时可使用
       - 注意：需要调整分类阈值或使用 class_weight
    
    【调优建议】
    - 下采样后如果样本数 < 10000，考虑改用上采样
    - 样本比例在 1:1 到 1:2 之间时可以不处理
    - 可以尝试 SMOTE 等高级采样方法（需额外实现）
    
    Args:
        df: 包含 'target' 列的训练数据
        method: 平衡方法
            - 'undersample': 下采样（默认）
            - 'oversample': 上采样
            - 'none': 不处理
    
    Returns:
        pd.DataFrame: 平衡后的数据框
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
    训练集成学习模型
    
    【模型架构】
    使用 EnsembleModel 集成以下 4 种算法：
    1. XGBoost: 梯度提升树，擅长处理非线性关系
    2. LightGBM: 轻量级梯度提升，训练速度快
    3. CatBoost: 类别特征处理能力强，减少过拟合
    4. RandomForest: 随机森林，提供稳定基线
    
    集成方式：软投票（加权平均各模型的预测概率）
    
    【参数说明】
    test_size: 0.2
        - 测试集比例，20% 用于最终验证
        - 建议范围：0.15-0.25
        - 数据量小时可降低至 0.1
    
    n_folds: 5
        - 交叉验证折数
        - 建议范围：3-10
        - 折数越多训练时间越长，但评估更稳定
    
    【训练时间预估】
    - 10000 样本: 约 1-2 分钟
    - 50000 样本: 约 5-8 分钟
    - 100000 样本: 约 10-15 分钟
    - 500000 样本: 约 30-60 分钟
    
    【调优建议】
    1. 如果验证集 AUC < 0.55：
       - 检查特征质量，可能需要更多特征
       - 检查标签定义，阈值可能不合适
       - 增加训练数据量
    
    2. 如果验证集 AUC > 0.7 但实盘表现差：
       - 可能存在前视偏差（使用了未来数据）
       - 检查特征是否包含未来信息
       - 增加正则化，减少过拟合
    
    3. 如果训练时间过长：
       - 减少交叉验证折数（n_folds）
       - 使用下采样减少样本量
       - 在 EnsembleModel 中调低各基模型的 n_estimators
    
    【内存使用】
    - 建议内存：8GB 以上
    - 大数据集（>100万样本）建议：16GB+
    
    Args:
        df: 训练数据，必须包含 feature_cols 中的特征和 'target' 标签列
        feature_cols: 特征列名列表
    
    Returns:
        EnsembleModel: 训练好的集成模型对象
    """
    print("\n" + "="*60)
    print("🤖 训练集成模型")
    print("="*60)
    
    # 过滤有效样本
    df_clean = df.dropna(subset=feature_cols + ['target'])
    print(f"有效训练样本: {len(df_clean):,}")

    # 训练前再次验证特征分布
    print("\n  检查特征分布...")
    feature_stats = df_clean[feature_cols].describe()
    extreme_features = []
    for col in feature_cols:
        col_max = feature_stats.loc['max', col]
        col_min = feature_stats.loc['min', col]
        # 检查是否有极端异常值（超过1e6或小于-1e6）
        if abs(col_max) > 1e6 or abs(col_min) > 1e6:
            extreme_features.append(f"{col}: [{col_min:.2e}, {col_max:.2e}]")

    if extreme_features:
        print(f"  ⚠️ 发现 {len(extreme_features)} 个特征有极端值:")
        for feat in extreme_features[:5]:
            print(f"    - {feat}")
        if len(extreme_features) > 5:
            print(f"    ... 还有 {len(extreme_features)-5} 个")
        print("  建议使用 validate_features.fix_features() 进行修复")
    
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
    验证训练好的模型
    
    【验证方法】
    从数据中随机抽样 100 条记录，使用模型预测并分析概率分布。
    
    【概率分布解读】
    - min/max: 预测概率的极值范围
      - 如果 min 接近 0 且 max 接近 1，说明模型有区分度
      - 如果范围很窄（如 0.4-0.6），说明模型不确定性强
    
    - mean/median: 概率的集中趋势
      - mean 接近正样本比例，说明模型校准良好
      - 如果 mean 偏离很大，可能需要重新校准
    
    - 分位数: 概率分布的离散程度
      - 如果各分位数分布均匀，说明预测质量好
      - 如果集中在某个值附近，说明模型欠拟合
    
    【常见问题诊断】
    1. 所有预测概率都在 0.4-0.6 之间：
       - 模型欠拟合，特征区分度不足
       - 解决：增加特征、调整模型参数
    
    2. 预测概率极端（接近 0 或 1）：
       - 模型过拟合，可能在噪声上学习
       - 解决：增加正则化、减少特征、增加数据
    
    3. mean 远高于/低于正样本比例：
       - 模型偏乐观/悲观
       - 解决：调整分类阈值或重新训练
    
    Args:
        model: 训练好的 EnsembleModel 对象
        df: 用于验证的数据框
        feature_cols: 特征列名列表
    
    Returns:
        Dict: 包含 min, max, mean, median 的概率统计字典
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

    # 检查概率分布是否合理
    if probabilities.max() < 0.2:
        print("  ⚠️ 警告: 最大预测概率 < 20%，模型可能欠拟合")
    if probabilities.min() > 0.8:
        print("  ⚠️ 警告: 最小预测概率 > 80%，模型可能过拟合")
    if probabilities.mean() < 0.1 or probabilities.mean() > 0.9:
        print(f"  ⚠️ 警告: 平均概率 {probabilities.mean():.4f} 偏离 0.5 太远")
    if probabilities.max() - probabilities.min() < 0.1:
        print("  ⚠️ 警告: 概率分布范围太小 (< 0.1)，模型区分度不足")
    
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
    """
    主训练流程入口
    
    【命令行参数】
    --data: 训练数据文件名（默认 training_data.csv）
    --forward-days: 预测周期，单位天（默认 5）
    --threshold: 上涨阈值，小数形式（默认 0.02，即 2%）
    --balance: 样本平衡方法（默认 undersample）
    
    【使用示例】
    # 使用默认参数训练
    python retrain_model.py
    
    # 指定数据文件和预测周期
    python retrain_model.py --data my_data.csv --forward-days 10
    
    # 自定义阈值和平衡方法
    python retrain_model.py --threshold 0.03 --balance oversample
    
    【输出文件说明】
    1. ensemble_model.pkl
       - 位置: 脚本所在目录
       - 内容: 训练好的集成模型对象
       - 格式: Python pickle 序列化
       - 大小: 通常 50-200 MB
       - 用途: 由选股策略加载使用
    
    2. training_info.json
       - 位置: 脚本所在目录
       - 内容: 训练配置和验证结果
       - 格式: JSON 文本
       - 包含字段:
         * trained_at: 训练完成时间
         * forward_days: 预测周期
         * return_threshold: 上涨阈值
         * balance_method: 平衡方法
         * feature_count: 特征数量
         * validation: 验证统计结果
    
    【文件命名规则】
    - 模型文件固定命名为 ensemble_model.pkl
    - 训练信息固定命名为 training_info.json
    - 如需保留多个版本，请在训练后手动重命名文件
    - 建议命名格式: ensemble_model_YYYYMMDD.pkl
    
    【训练建议】
    1. 训练前检查数据质量，确保无缺失关键字段
    2. 首次训练建议使用小数据集测试流程
    3. 定期（每月/每季度）重新训练保持模型时效性
    4. 保存训练日志以便追溯问题
    """
    import argparse
    parser = argparse.ArgumentParser(description='Topaz 模型重新训练')
    parser.add_argument('--data', type=str, default='csi300_full_history.csv', help='训练数据文件（fetch_full_history.py生成）')
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