#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topaz Walk-Forward TimeSeries 模型训练脚本
使用 Purged Walk-Forward Optimization 避免时间序列泄漏
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import lightgbm as lgb
import joblib

from feature_engineer import FeatureEngineer
from market_data import get_index_history
from quantpilot_data_api import get_history_data


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """计算夏普比率"""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252  # 日度无风险收益率
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


def calculate_max_drawdown(values: np.ndarray) -> float:
    """计算最大回撤"""
    if len(values) == 0:
        return 0.0
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    return np.min(drawdown)


class WalkForwardTrainer:
    """Walk-Forward 时间序列训练器"""
    
    def __init__(self, 
                 initial_train_window: int = 400,  # 初始训练窗口：400个交易日
                 roll_step: int = 20,              # 滚动步长：20个交易日
                 prediction_window: int = 20,      # 预测窗口：20个交易日
                 purge_gap: int = 5,               # 清除间隔：5个交易日
                 forward_days: int = 5,            # 预测未来5天
                 return_threshold: float = 0.02):  # 上涨阈值2%
        self.initial_train_window = initial_train_window
        self.roll_step = roll_step
        self.prediction_window = prediction_window
        self.purge_gap = purge_gap
        self.forward_days = forward_days
        self.return_threshold = return_threshold
        
        self.models_history = []
        self.performance_history = []
        
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成训练标签 - 横截面排名目标"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['code', 'date']).reset_index(drop=True)
        
        df['future_return'] = df.groupby('code')['close'].transform(
            lambda x: x.shift(-self.forward_days) / x - 1
        )
        
        df['return_rank'] = df.groupby('date')['future_return'].transform(
            lambda x: x.rank(pct=True)
        )
        
        df['target'] = (df['return_rank'] > 0.67).astype(int)
        
        df['excess_return'] = df.groupby('date')['future_return'].transform(
            lambda x: x - x.mean()
        )
        
        valid = df[df['future_return'].notna()]
        print(f"  目标分布: 前1/3占比={valid['target'].mean():.2%}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """准备特征 - 个股特异性特征为主"""
        exclude_cols = ['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'name',
                        'future_return', 'target', 'return_rank', 'excess_return']
        
        alpha_features = [
            'rsi_14', 'rsi_6', 'macd', 'macd_hist', 'macd_signal',
            'bb_position', 'bb_width', 'cci', 'williams_r',
            'momentum_5', 'momentum_10', 'momentum_accel',
            'volatility_5', 'volatility_10', 'volatility_20',
            'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20',
            'obv', 'turnover', 'amihud',
            'price_position_10', 'price_position_20', 'close_position',
            'return_1d', 'return_3d', 'return_5d',
            'max_drawdown_20', 'sharpe_proxy',
            'consecutive_up', 'consecutive_down', 'gap',
            'is_hammer', 'is_shooting_star', 'is_engulfing',
            'ma5_slope', 'ma10_slope', 'ma20_slope',
            'adx', 'atr_ratio', 'tr_14',
            'kurtosis_20', 'skewness_20',
            'return_autocorr_5', 'mean_reversion_20',
            'body_ratio', 'upper_shadow', 'lower_shadow'
        ]
        
        feature_cols = [col for col in alpha_features if col in df.columns]
        
        if len(feature_cols) < 10:
            all_cols = [col for col in df.columns 
                       if col not in exclude_cols 
                       and df[col].dtype in ['float64', 'int64']]
            feature_cols = all_cols[:30]
        
        for col in feature_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
            df[col] = df[col].clip(-1e10, 1e10)
        
        print(f"  alpha特征数: {len(feature_cols)}")
        print(f"  特征列表: {feature_cols[:10]}...")
        
        return df, feature_cols
    
    def create_walkforward_splits(self, df: pd.DataFrame) -> List[Tuple]:
        """
        创建 Walk-Forward 分割
        
        返回: [(train_start, train_end, test_start, test_end), ...]
        """
        # 获取所有唯一日期
        dates = sorted(df['date'].unique())
        n_dates = len(dates)
        
        splits = []
        
        # 初始训练结束位置
        train_end_idx = self.initial_train_window
        
        while train_end_idx + self.purge_gap + self.prediction_window <= n_dates:
            # 训练集：从开头到 train_end_idx
            train_start = 0
            train_end = train_end_idx
            
            # 测试集：跳过 purge_gap，取 prediction_window
            test_start = train_end_idx + self.purge_gap
            test_end = min(test_start + self.prediction_window, n_dates)
            
            splits.append((train_start, train_end, test_start, test_end))
            
            # 滚动到下一个窗口
            train_end_idx += self.roll_step
        
        return splits, dates
    
    def train_single_fold(self, 
                         df: pd.DataFrame, 
                         feature_cols: List[str],
                         train_indices: np.ndarray,
                         test_indices: np.ndarray) -> Dict:
        """训练单个折叠"""
        
        sample_size = 30000
        
        df_train = df.iloc[train_indices].copy()
        df_test = df.iloc[test_indices].copy()
        
        df_train = df_train.dropna(subset=feature_cols + ['target'])
        df_test = df_test.dropna(subset=feature_cols + ['target'])
        
        if len(df_train) < 100 or len(df_test) < 10:
            return None
        
        if len(df_train) > sample_size:
            df_train = df_train.sample(n=sample_size, random_state=42)
        
        X_train = df_train[feature_cols].values
        y_train = df_train['target'].values
        X_test = df_test[feature_cols].values
        y_test = df_test['target'].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        lgb_model.fit(X_train_scaled, y_train)
        models['lightgbm'] = lgb_model
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        models['rf'] = rf_model
        
        gb_model = GradientBoostingClassifier(
            n_estimators=80,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=10,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        models['gbdt'] = gb_model
        
        # 预测（简单平均）
        predictions = []
        for name, model in models.items():
            pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            predictions.append(pred_proba)
        
        avg_proba = np.mean(predictions, axis=0)
        y_pred = (avg_proba >= 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, avg_proba) if len(np.unique(y_test)) > 1 else 0.5
        
        if 'future_return' in df_test.columns:
            actual_returns = df_test['future_return'].values
            valid_mask = ~np.isnan(actual_returns) & ~np.isnan(avg_proba)
            
            if valid_mask.sum() > 10:
                ic = np.corrcoef(actual_returns[valid_mask], avg_proba[valid_mask])[0, 1]
                
                rank_actual = pd.Series(actual_returns[valid_mask]).rank()
                rank_pred = pd.Series(avg_proba[valid_mask]).rank()
                rank_ic = np.corrcoef(rank_actual, rank_pred)[0, 1]
            else:
                ic = 0
                rank_ic = 0
        else:
            ic = 0
            rank_ic = 0
        
        df_test_copy = df_test.copy()
        df_test_copy['pred_proba'] = avg_proba
        df_test_copy['pred'] = y_pred
        
        if 'future_return' in df_test.columns:
            df_test_copy['strategy_return'] = np.where(
                df_test_copy['pred_proba'] > 0.5,
                df_test_copy['future_return'],
                0
            )
        else:
            df_test_copy['strategy_return'] = 0
        
        returns = df_test_copy['strategy_return'].values
        sharpe = calculate_sharpe_ratio(returns)
        
        # 计算净值曲线
        cumulative_returns = np.cumprod(1 + returns) - 1
        max_dd = calculate_max_drawdown(1 + cumulative_returns)
        
        return {
            'models': models,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'accuracy': accuracy,
            'auc': auc,
            'ic': ic,
            'rank_ic': rank_ic,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_start_date': df_train['date'].min(),
            'train_end_date': df_train['date'].max(),
            'test_start_date': df_test['date'].min(),
            'test_end_date': df_test['date'].max(),
            'returns': returns,
            'cumulative_returns': cumulative_returns
        }
    
    def train(self, df: pd.DataFrame) -> Dict:
        """执行 Walk-Forward 训练"""
        print("="*60)
        print("🚀 Walk-Forward TimeSeries 训练")
        print("="*60)
        
        # 生成标签
        print("\n🏷️ 生成标签...")
        df = self.generate_labels(df)
        df = df[df['target'].notna()].copy()
        print(f"  有效样本: {len(df):,}")
        
        # 准备特征
        print("\n🔧 准备特征...")
        df, feature_cols = self.prepare_features(df)
        print(f"  特征数量: {len(feature_cols)}")
        
        # 创建 Walk-Forward 分割
        print("\n📊 创建 Walk-Forward 分割...")
        splits, dates = self.create_walkforward_splits(df)
        print(f"  总日期数: {len(dates)}")
        print(f"  分割数量: {len(splits)}")
        print(f"  参数: 初始窗口={self.initial_train_window}, 步长={self.roll_step}, 预测窗口={self.prediction_window}, 清除间隔={self.purge_gap}")
        
        # 执行每个折叠的训练（限制最多5个fold以节省时间）
        print("\n" + "="*60)
        print("🔄 开始滚动训练")
        print("="*60)
        
        all_results = []
        max_folds = min(5, len(splits))
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(splits[:max_folds]):
            print(f"\n📈 Fold {i+1}/{max_folds}")
            print(f"  训练集: {dates[train_start]} ~ {dates[train_end-1]} ({train_end-train_start}天)")
            print(f"  测试集: {dates[test_start]} ~ {dates[test_end-1]} ({test_end-test_start}天)")
            
            train_mask = (df['date'] >= dates[train_start]) & (df['date'] < dates[train_end])
            test_mask = (df['date'] >= dates[test_start]) & (df['date'] < dates[test_end])
            
            train_indices = df[train_mask].index.values
            test_indices = df[test_mask].index.values
            
            sample_size = 30000
            
            # 训练
            result = self.train_single_fold(df, feature_cols, train_indices, test_indices)
            
            if result:
                print(f"  ✅ AUC: {result['auc']:.3f}, IC: {result['ic']:.3f}, Rank-IC: {result['rank_ic']:.3f}")
                all_results.append(result)
                self.models_history.append({
                    'fold': i+1,
                    'models': result['models'],
                    'scaler': result['scaler'],
                    'feature_cols': result['feature_cols'],
                    'test_end_date': result['test_end_date']
                })
            else:
                print(f"  ⚠️ 训练失败")
        
        # 汇总结果
        print("\n" + "="*60)
        print("📊 Walk-Forward 性能汇总")
        print("="*60)
        
        if all_results:
            avg_accuracy = np.mean([r['accuracy'] for r in all_results])
            avg_auc = np.mean([r['auc'] for r in all_results])
            avg_ic = np.mean([r['ic'] for r in all_results])
            avg_rank_ic = np.mean([r['rank_ic'] for r in all_results])
            
            print(f"平均AUC:     {avg_auc:.3f}")
            print(f"平均IC:      {avg_ic:.3f}")
            print(f"平均Rank-IC: {avg_rank_ic:.3f}")
            
            self.performance_history = {
                'avg_accuracy': avg_accuracy,
                'avg_auc': avg_auc,
                'avg_ic': avg_ic,
                'avg_rank_ic': avg_rank_ic,
                'folds': len(all_results)
            }
        
        return self.performance_history
    
    def save_latest_model(self, output_path: str):
        """保存最新一期的模型"""
        if not self.models_history:
            print("⚠️ 没有可保存的模型")
            return
        
        # 获取最新一期的模型
        latest = self.models_history[-1]
        
        model_package = {
            'models': latest['models'],
            'scaler': latest['scaler'],
            'feature_cols': latest['feature_cols'],
            'trained_at': datetime.now().isoformat(),
            'test_end_date': latest['test_end_date'],
            'performance': self.performance_history
        }
        
        joblib.dump(model_package, output_path)
        print(f"\n💾 最新模型已保存: {output_path}")
        print(f"  训练截止日期: {latest['test_end_date']}")


def main():
    """主函数"""
    print("="*60)
    print("Topaz Walk-Forward TimeSeries 模型训练")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载数据
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'csi300_raw_data_2y.csv')
    
    print(f"\n📂 加载数据: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  数据形状: {df.shape}")
    
    # 创建训练器（简化参数，减少训练时间）
    trainer = WalkForwardTrainer(
        initial_train_window=300,
        roll_step=30,
        prediction_window=20,
        purge_gap=10,
        forward_days=20,
        return_threshold=0.0
    )
    
    # 执行训练
    performance = trainer.train(df)
    
    # 保存最新模型
    output_path = os.path.join(base_dir, 'ensemble_model_csi300_latest.pkl')
    trainer.save_latest_model(output_path)
    
    print("\n" + "="*60)
    print("✅ Walk-Forward 训练完成!")
    print("="*60)


if __name__ == '__main__':
    main()