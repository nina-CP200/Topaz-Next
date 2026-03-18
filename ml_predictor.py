#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topaz V3 - 真正的机器学习预测模块
支持过拟合检测和自动fallback到多因子评分
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

from ml_stock_analysis import MLStockAnalyzer
from topaz_data_api import get_cn_realtime_data, get_cn_history_data


class MLPredictor:
    """机器学习预测器 - 支持过拟合检测和Fallback"""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.dirname(os.path.abspath(__file__))
        
        # 模型
        self.model = None
        self.scaler = None
        self.feature_cols = []
        
        # 过拟合阈值
        self.overfit_threshold = 0.15  # 训练R2和验证R2差距超过15%视为过拟合
        self.min_r2 = 0.05  # R2低于这个值认为模型无效
        
        # 模型状态
        self.model_status = {
            'trained': False,
            'overfitted': False,
            'r2_train': 0,
            'r2_val': 0,
            'mse_val': 0,
            'use_fallback': True,  # 默认使用fallback
            'train_time': None,
            'samples': 0
        }
        
        # 多因子分析器（fallback用）
        self.factor_analyzer = MLStockAnalyzer()
        
        # 加载模型（如果存在）
        self._load_model()
    
    def _load_model(self):
        """加载已训练的模型"""
        model_path = os.path.join(self.data_dir, 'ml_model.pkl')
        scaler_path = os.path.join(self.data_dir, 'ml_scaler.pkl')
        status_path = os.path.join(self.data_dir, 'ml_model_status.json')
        
        if os.path.exists(model_path) and os.path.exists(status_path):
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                with open(status_path, 'r') as f:
                    self.model_status = json.load(f)
                print(f"✓ 加载已训练模型 (R2={self.model_status['r2_val']:.4f})")
            except Exception as e:
                print(f"加载模型失败: {e}")
    
    def _save_model(self):
        """保存模型"""
        model_path = os.path.join(self.data_dir, 'ml_model.pkl')
        scaler_path = os.path.join(self.data_dir, 'ml_scaler.pkl')
        status_path = os.path.join(self.data_dir, 'ml_model_status.json')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        with open(status_path, 'w') as f:
            json.dump(self.model_status, f, indent=2)
        
        print(f"✓ 模型已保存")
    
    def prepare_features(self, df: pd.DataFrame, target_days: int = 5, use_classification: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备训练特征
        
        Args:
            df: 原始数据
            target_days: 预测未来几天的收益
            use_classification: 使用分类（涨/跌）而非回归
        
        Returns:
            X: 特征矩阵
            y: 目标变量（0=跌，1=涨 或 收益率）
        """
        df = df.copy()
        
        # 计算未来收益（目标变量）
        df['future_return'] = df.groupby('code')['close'].shift(-target_days) / df['close'] - 1
        
        # 分类目标：涨=1，跌=0
        if use_classification:
            df['target'] = (df['future_return'] > 0).astype(int)
            target_col = 'target'
        else:
            target_col = 'future_return'
        
        # 特征列
        feature_cols = [
            # 价格位置
            'price_to_ma5', 'price_to_ma10', 'price_to_ma20',
            # 趋势
            'ma5_slope', 'ma10_slope', 'ma20_slope',
            # 波动率
            'volatility_5', 'volatility_10', 'volatility_20',
            # 动量
            'return_1d', 'return_5d', 'return_10d', 'return_20d',
            # 成交量
            'volume_ratio',
            # 技术指标
            'rsi', 'macd', 'macd_hist', 'bb_position', 'kdj_k', 'kdj_d'
        ]
        
        # 只保留存在的列
        self.feature_cols = [c for c in feature_cols if c in df.columns]
        
        # 添加行业特征（one-hot编码）
        if 'industry' in df.columns:
            industry_dummies = pd.get_dummies(df['industry'], prefix='ind')
            df = pd.concat([df, industry_dummies], axis=1)
            self.feature_cols.extend(industry_dummies.columns.tolist())
        
        # 移除NaN
        df_clean = df.dropna(subset=self.feature_cols + [target_col])
        
        X = df_clean[self.feature_cols]
        y = df_clean[target_col]
        
        return X, y
    
    def train(self, data_file: str = 'training_data.csv', 
              target_days: int = 5,
              test_size: float = 0.2,
              cv_folds: int = 5,
              use_classification: bool = True):
        """
        训练机器学习模型
        
        Args:
            data_file: 训练数据文件
            target_days: 预测未来几天的收益
            test_size: 验证集比例
            cv_folds: 交叉验证折数
            use_classification: 使用分类模型（涨/跌）而非回归
        """
        print("\n" + "="*60)
        print("机器学习模型训练")
        print("="*60)
        
        # 加载数据
        data_path = os.path.join(self.data_dir, data_file)
        if not os.path.exists(data_path):
            print(f"❌ 训练数据不存在: {data_path}")
            return False
        
        print(f"加载训练数据...")
        df = pd.read_csv(data_path)
        print(f"原始数据: {len(df):,} 行")
        
        # 准备特征
        print(f"\n准备特征...")
        X, y = self.prepare_features(df, target_days, use_classification)
        print(f"有效样本: {len(X):,}")
        print(f"特征数量: {len(self.feature_cols)}")
        
        if len(X) < 1000:
            print("❌ 样本数量不足，无法训练")
            return False
        
        # 标准化
        print(f"\n标准化特征...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 分割训练/验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        print(f"训练集: {len(X_train):,} 样本")
        print(f"验证集: {len(X_val):,} 样本")
        
        # 选择模型类型
        if use_classification:
            print(f"\n训练分类模型（预测涨跌）...")
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                min_samples_split=100,
                min_samples_leaf=50,
                learning_rate=0.1,
                random_state=42
            )
            self.model_type = 'classifier'
        else:
            print(f"\n训练回归模型...")
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1
            )
            self.model_type = 'regressor'
        
        self.model.fit(X_train, y_train)
        
        # 评估
        print(f"\n评估模型...")
        
        if self.model_type == 'classifier':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            # 训练集表现
            y_train_pred = self.model.predict(X_train)
            y_train_proba = self.model.predict_proba(X_train)[:, 1]
            
            # 验证集表现
            y_val_pred = self.model.predict(X_val)
            y_val_proba = self.model.predict_proba(X_val)[:, 1]
            
            # 指标
            train_acc = accuracy_score(y_train, y_train_pred)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_precision = precision_score(y_val, y_val_pred)
            val_recall = recall_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred)
            val_auc = roc_auc_score(y_val, y_val_proba)
            
            # 过拟合检测
            overfit_gap = train_acc - val_acc
            is_overfitted = overfit_gap > self.overfit_threshold
            is_valid = val_acc > 0.52  # 至少比随机好一点
            
            print(f"\n{'='*60}")
            print("分类模型评估结果")
            print("="*60)
            print(f"训练集准确率: {train_acc:.4f}")
            print(f"验证集准确率: {val_acc:.4f}")
            print(f"验证集精确率: {val_precision:.4f}")
            print(f"验证集召回率: {val_recall:.4f}")
            print(f"验证集 F1: {val_f1:.4f}")
            print(f"验证集 AUC: {val_auc:.4f}")
            print(f"\n过拟合差距: {overfit_gap:.4f} (阈值: {self.overfit_threshold})")
            
            # 更新状态
            self.model_status = {
                'trained': True,
                'model_type': 'classifier',
                'overfitted': is_overfitted,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'val_auc': val_auc,
                'overfit_gap': overfit_gap,
                'use_fallback': is_overfitted or not is_valid,
                'train_time': datetime.now().isoformat(),
                'samples': len(X),
                'features': len(self.feature_cols),
                'target_days': target_days
            }
            
        else:
            # 回归模型评估
            y_train_pred = self.model.predict(X_train)
            r2_train = r2_score(y_train, y_train_pred)
            mse_train = mean_squared_error(y_train, y_train_pred)
            
            y_val_pred = self.model.predict(X_val)
            r2_val = r2_score(y_val, y_val_pred)
            mse_val = mean_squared_error(y_val, y_val_pred)
            mae_val = mean_absolute_error(y_val, y_val_pred)
            
            # 交叉验证
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv_folds, scoring='r2')
            
            # 过拟合检测
            overfit_gap = r2_train - r2_val
            is_overfitted = overfit_gap > self.overfit_threshold
            is_valid = r2_val > self.min_r2
            
            print(f"\n{'='*60}")
            print("回归模型评估结果")
            print("="*60)
            print(f"训练集 R²: {r2_train:.4f}")
            print(f"验证集 R²: {r2_val:.4f}")
            print(f"验证集 MSE: {mse_val:.6f}")
            print(f"验证集 MAE: {mae_val:.4f}")
            print(f"交叉验证 R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"\n过拟合差距: {overfit_gap:.4f} (阈值: {self.overfit_threshold})")
            
            if is_overfitted:
                print("⚠️  检测到过拟合！将使用多因子Fallback")
            elif not is_valid:
                print("⚠️  模型效果不佳！将使用多因子Fallback")
            else:
                print("✓ 模型训练成功")
            
            # 更新状态
            self.model_status = {
                'trained': True,
                'model_type': 'regressor',
                'overfitted': is_overfitted,
                'r2_train': r2_train,
                'r2_val': r2_val,
                'mse_val': mse_val,
                'mae_val': mae_val,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'overfit_gap': overfit_gap,
                'use_fallback': is_overfitted or not is_valid,
                'train_time': datetime.now().isoformat(),
                'samples': len(X),
                'features': len(self.feature_cols),
                'target_days': target_days
            }
        
        # 保存模型
        self._save_model()
        
        # 特征重要性
        print(f"\n特征重要性 (Top 10):")
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return not self.model_status['use_fallback']
    
    def predict(self, code: str) -> Dict:
        """
        预测股票收益
        
        自动选择ML预测或多因子评分
        """
        result = {
            'code': code,
            'method': 'unknown',
            'predicted_return': 0,
            'confidence': 'low',
            'model_status': self.model_status
        }
        
        # 获取股票数据
        try:
            realtime = get_cn_realtime_data(code)
            history = get_cn_history_data(code, days=60)
            
            if not realtime or not history:
                result['error'] = '无法获取股票数据'
                return result
            
            result['name'] = realtime.get('name', '')
            result['current_price'] = realtime.get('price', 0)
            result['change_pct'] = realtime.get('change_pct', 0)
            
        except Exception as e:
            result['error'] = f'数据获取失败: {e}'
            return result
        
        # 检查是否使用Fallback
        if self.model_status.get('use_fallback', True):
            return self._predict_fallback(code, realtime, history, result)
        
        # ML预测
        return self._predict_ml(code, history, result)
    
    def _predict_ml(self, code: str, history: list, result: Dict) -> Dict:
        """使用ML模型预测"""
        if not self.model or not self.scaler:
            return self._predict_fallback(code, None, history, result)
        
        try:
            # 准备特征
            df = pd.DataFrame(history)
            df = self._calculate_features(df)
            
            # 取最新一行
            latest = df.iloc[-1:][self.feature_cols[:22]]  # 只取基础特征，不含行业
            
            if latest.isna().any().any():
                # 特征缺失，fallback
                return self._predict_fallback(code, None, history, result)
            
            # 标准化
            X = self.scaler.transform(latest)
            
            # 预测
            if self.model_type == 'classifier':
                pred_class = self.model.predict(X)[0]
                pred_proba = self.model.predict_proba(X)[0]
                
                # 分类结果转收益预期
                if pred_class == 1:  # 预测涨
                    predicted_return = (pred_proba[1] - 0.5) * 0.05  # 最高预期+2.5%
                else:  # 预测跌
                    predicted_return = -(0.5 - pred_proba[0]) * 0.05  # 最低预期-2.5%
                
                result['ml_confidence'] = max(pred_proba)
            else:
                predicted_return = self.model.predict(X)[0]
            
            result['method'] = 'ml'
            result['predicted_return'] = predicted_return
            result['confidence'] = 'high' if self.model_status.get('val_acc', 0) > 0.55 else 'medium'
            
            return result
            
        except Exception as e:
            print(f"ML预测失败: {e}")
            return self._predict_fallback(code, None, history, result)
    
    def _predict_fallback(self, code: str, realtime: Dict, history: list, result: Dict) -> Dict:
        """使用多因子评分作为Fallback"""
        result['method'] = 'multi_factor'
        
        # 使用多因子分析
        try:
            factor_result = self.factor_analyzer.analyze_stock(code)
            
            if factor_result:
                # 从多因子评分推断预期收益
                score = factor_result.get('total_score', 50)
                
                # 简单映射：分数高预期正收益，分数低预期负收益
                # 分数范围 0-100，映射到 -5% ~ +5%
                predicted_return = (score - 50) / 10 * 0.01
                
                result['predicted_return'] = predicted_return
                result['factor_score'] = score
                result['factor_rating'] = factor_result.get('rating', 'N/A')
                result['confidence'] = 'medium'
                
        except Exception as e:
            result['error'] = f'多因子分析失败: {e}'
            result['predicted_return'] = 0
            result['confidence'] = 'low'
        
        return result
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标特征"""
        df = df.copy()
        
        # 移动平均线
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        
        # 价格位置
        df['price_to_ma5'] = df['close'] / df['ma5'] - 1
        df['price_to_ma10'] = df['close'] / df['ma10'] - 1
        df['price_to_ma20'] = df['close'] / df['ma20'] - 1
        
        # 趋势
        df['ma5_slope'] = df['ma5'].diff(5) / df['ma5'].shift(5)
        df['ma10_slope'] = df['ma10'].diff(5) / df['ma10'].shift(5)
        df['ma20_slope'] = df['ma20'].diff(5) / df['ma20'].shift(5)
        
        # 波动率
        df['volatility_5'] = df['close'].pct_change().rolling(window=5).std()
        df['volatility_10'] = df['close'].pct_change().rolling(window=10).std()
        df['volatility_20'] = df['close'].pct_change().rolling(window=20).std()
        
        # 动量
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_20d'] = df['close'].pct_change(20)
        
        # 成交量
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma5']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_hist'] = df['macd'] - df['macd'].ewm(span=9, adjust=False).mean()
        
        # 布林带位置
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # KDJ
        low_min = df['low'].rolling(window=9).min()
        high_max = df['high'].rolling(window=9).max()
        df['kdj_k'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
        df['kdj_d'] = df['kdj_k'].rolling(window=3).mean()
        
        return df
    
    def get_status(self) -> Dict:
        """获取模型状态"""
        return {
            'model_loaded': self.model is not None,
            'use_fallback': self.model_status.get('use_fallback', True),
            **self.model_status
        }


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='ML预测器')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--predict', type=str, help='预测指定股票')
    parser.add_argument('--status', action='store_true', help='查看模型状态')
    parser.add_argument('--target-days', type=int, default=5, help='预测未来几天')
    parser.add_argument('--regression', action='store_true', help='使用回归模型（默认分类）')
    args = parser.parse_args()
    
    predictor = MLPredictor()
    
    if args.train:
        predictor.train(target_days=args.target_days, use_classification=not args.regression)
    elif args.predict:
        result = predictor.predict(args.predict)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.status:
        status = predictor.get_status()
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        # 默认显示状态
        status = predictor.get_status()
        print("\n模型状态:")
        print(f"  已训练: {status.get('trained', False)}")
        print(f"  模型类型: {status.get('model_type', 'unknown')}")
        print(f"  使用Fallback: {status.get('use_fallback', True)}")
        if status.get('model_type') == 'classifier':
            print(f"  验证准确率: {status.get('val_acc', 0):.4f}")
            print(f"  验证AUC: {status.get('val_auc', 0):.4f}")
        elif status.get('trained'):
            print(f"  验证R²: {status.get('r2_val', 0):.4f}")
            print(f"  过拟合差距: {status.get('overfit_gap', 0):.4f}")


if __name__ == '__main__':
    main()