#!/usr/bin/env python3
"""
美股模型训练脚本
训练 QuantPilot 美股预测模型
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 检查依赖
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# 导入特征工程模块
from feature_engineer import FeatureEngineer


# 美股行业分类
US_INDUSTRY_MAP = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'META': 'Technology',
    'NVDA': 'Technology', 'AMD': 'Technology', 'INTC': 'Technology', 'AVGO': 'Technology',
    'CRM': 'Technology', 'ORCL': 'Technology',
    
    # Financial
    'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'GS': 'Financial',
    'V': 'Financial', 'MA': 'Financial',
    
    # Consumer
    'AMZN': 'Consumer', 'WMT': 'Consumer', 'KO': 'Consumer', 'MCD': 'Consumer',
    'NKE': 'Consumer', 'COST': 'Consumer',
    
    # Healthcare
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'ABT': 'Healthcare',
    
    # Industrial
    'CAT': 'Industrial', 'BA': 'Industrial', 'GE': 'Industrial', 'HON': 'Industrial',
    
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
    
    # Communication
    'T': 'Communication', 'VZ': 'Communication', 'TMUS': 'Communication',
    
    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities',
    
    # Real Estate
    'AMT': 'RealEstate', 'PLD': 'RealEstate',
    
    # Materials
    'LIN': 'Materials', 'APD': 'Materials',
}


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """加载并清洗数据"""
    print(f"加载数据: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # 数据格式：每个股票有标题行和日期行
    # Ticker, Price(Ticker), Close(AAPL), High(AAPL), Low(AAPL), Open(AAPL), Volume(AAPL)
    # AAPL, Date, NaN, NaN, NaN, NaN, NaN
    # AAPL, 2024-03-19, close, high, low, open, volume
    
    # 跳过 Price 列中包含 "Ticker" 或 "Date" 的行
    df = df[~df['Price'].isin(['Ticker', 'Date'])].copy()
    
    # 重命名列
    df.columns = ['code', 'date', 'close', 'high', 'low', 'open', 'volume']
    
    # 转换数据类型
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['close', 'high', 'low', 'open', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 删除无效行
    df = df.dropna(subset=['date', 'close']).copy()
    
    # 排序
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    
    # 添加行业
    df['industry'] = df['code'].map(US_INDUSTRY_MAP)
    
    print(f"数据形状: {df.shape}")
    print(f"股票数量: {df['code'].nunique()}")
    print(f"时间范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"行业分布:\n{df['industry'].value_counts()}")
    
    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """生成特征"""
    print("生成特征...")
    
    fe = FeatureEngineer()
    df = fe.generate_all_features(df)
    
    # 删除NaN过多的行
    nan_ratio = df.isnull().sum(axis=1) / len(df.columns)
    df = df[nan_ratio < 0.3].copy()
    
    print(f"特征数量: {len(df.columns)}")
    print(f"有效样本: {len(df)}")
    
    return df


def create_labels(df: pd.DataFrame, forward_days: int = 5, threshold: float = 0.02) -> pd.DataFrame:
    """创建预测标签"""
    print(f"创建标签: 未来{forward_days}天涨幅 > {threshold*100}%")
    
    # 计算未来收益
    df = df.sort_values(['code', 'date'])
    df['future_return'] = df.groupby('code')['close'].transform(
        lambda x: x.shift(-forward_days) / x - 1
    )
    
    # 创建三分类标签
    # 1: 上涨超过 threshold
    # 0: 横盘 (-threshold, threshold)
    # -1: 下跌超过 threshold
    df['label'] = 0
    df.loc[df['future_return'] > threshold, 'label'] = 1
    df.loc[df['future_return'] < -threshold, 'label'] = -1
    
    # 转换为分类标签 (0, 1, 2)
    df['label'] = df['label'] + 1  # -1->0, 0->1, 1->2
    
    print(f"标签分布:\n{df['label'].value_counts()}")
    
    return df


def prepare_training_data(df: pd.DataFrame) -> tuple:
    """准备训练数据"""
    print("准备训练数据...")
    
    # 选择特征列
    feature_cols = [c for c in df.columns if c not in [
        'code', 'date', 'industry', 'future_return', 'label'
    ]]
    
    # 删除含NaN的行
    df_clean = df.dropna(subset=feature_cols + ['label']).copy()
    
    X = df_clean[feature_cols].values
    y = df_clean['label'].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"训练样本: {len(X)}")
    print(f"特征数量: {len(feature_cols)}")
    
    return X_scaled, y, scaler, feature_cols


def train_ensemble_model(X, y, feature_cols, model_dir: str):
    """训练集成模型"""
    print("\n训练集成模型...")
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {}
    
    # 1. XGBoost
    if XGBOOST_AVAILABLE:
        print("训练 XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train, y_train)
        models['xgboost'] = xgb_model
        print(f"  XGBoost 训练完成")
    
    # 2. LightGBM
    if LIGHTGBM_AVAILABLE:
        print("训练 LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        models['lightgbm'] = lgb_model
        print(f"  LightGBM 训练完成")
    
    # 3. Random Forest
    print("训练 Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['random_forest'] = rf_model
    print(f"  Random Forest 训练完成")
    
    # 评估模型
    print("\n模型评估:")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"  {name}: Accuracy={acc:.4f}, F1={f1:.4f}")
    
    return models


def save_models(models, scaler, feature_cols, model_dir: str):
    """保存模型"""
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存主模型
    model_path = os.path.join(model_dir, 'us_ensemble_model.pkl')
    joblib.dump({
        'models': models,
        'feature_cols': feature_cols
    }, model_path)
    print(f"模型保存到: {model_path}")
    
    # 保存 scaler
    scaler_path = os.path.join(model_dir, 'us_ensemble_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler 保存到: {scaler_path}")
    
    # 保存状态
    status = {
        'trained': True,
        'train_time': datetime.now().isoformat(),
        'model_type': 'ensemble',
        'models': list(models.keys()),
        'feature_count': len(feature_cols),
        'market': 'US',
        'stocks': 30,
        'prediction_horizon': '5d'
    }
    status_path = os.path.join(model_dir, 'us_model_status.json')
    with open(status_path, 'w') as f:
        json.dump(status, f, indent=2)
    print(f"状态保存到: {status_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("QuantPilot 美股模型训练")
    print("=" * 60)
    
    model_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(model_dir, 'all_stocks_merged.csv')
    
    # 1. 加载数据
    df = load_and_clean_data(data_path)
    
    # 2. 生成特征
    df = generate_features(df)
    
    # 3. 创建标签
    df = create_labels(df, forward_days=5, threshold=0.02)
    
    # 4. 准备训练数据
    X, y, scaler, feature_cols = prepare_training_data(df)
    
    # 5. 训练模型
    models = train_ensemble_model(X, y, feature_cols, model_dir)
    
    # 6. 保存模型
    save_models(models, scaler, feature_cols, model_dir)
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()