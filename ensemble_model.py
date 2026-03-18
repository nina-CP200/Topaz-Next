#!/usr/bin/env python3
"""
模型集成 - Stacking + Meta-Learner
多模型融合提升准确率
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
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

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib


class EnsembleModel:
    """集成模型 - XGBoost + LightGBM + CatBoost Stacking"""
    
    def __init__(self, model_dir: str = '.'):
        self.model_dir = model_dir
        self.models = {}
        self.scaler = None
        self.meta_learner = None
        self.feature_cols = []
        self.model_status = {'trained': False}
        
        # 可用的基学习器
        self.available_models = []
        if XGBOOST_AVAILABLE:
            self.available_models.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            self.available_models.append('lightgbm')
        if CATBOOST_AVAILABLE:
            self.available_models.append('catboost')
        self.available_models.extend(['rf', 'gbdt'])
        
        # 加载已有模型
        self._load_models()
    
    def _load_models(self):
        """加载已保存的模型"""
        model_path = os.path.join(self.model_dir, 'ensemble_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'ensemble_scaler.pkl')
        status_path = os.path.join(self.model_dir, 'ensemble_status.json')
        
        if os.path.exists(model_path):
            try:
                saved = joblib.load(model_path)
                self.models = saved['models']
                self.meta_learner = saved['meta_learner']
                self.feature_cols = saved['feature_cols']
                
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                
                if os.path.exists(status_path):
                    with open(status_path, 'r') as f:
                        self.model_status = json.load(f)
                
                print(f"已加载集成模型: {list(self.models.keys())}")
            except Exception as e:
                print(f"加载模型失败: {e}")
    
    def _get_base_model(self, name: str, params: Dict = None):
        """获取基学习器"""
        if params is None:
            params = {}
        
        default_params = {
            'random_state': 42,
            'n_jobs': -1
        }
        
        if name == 'xgboost' and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                **default_params,
                **params
            )
        elif name == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                verbose=-1,
                **default_params,
                **params
            )
        elif name == 'catboost' and CATBOOST_AVAILABLE:
            return cb.CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                verbose=0,
                random_seed=42,
                **params
            )
        elif name == 'rf':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=50,
                min_samples_leaf=25,
                **default_params,
                **params
            )
        elif name == 'gbdt':
            return GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                min_samples_split=50,
                min_samples_leaf=25,
                **params
            )
        else:
            return None
    
    def train(self, df: pd.DataFrame, feature_cols: List[str],
              target_col: str = 'target',
              test_size: float = 0.2,
              n_folds: int = 5,
              use_optimization: bool = False):
        """
        训练集成模型
        
        Args:
            df: 训练数据
            feature_cols: 特征列
            target_col: 目标列
            test_size: 测试集比例
            n_folds: 交叉验证折数
            use_optimization: 是否使用Optuna调参
        """
        print("\n" + "="*60)
        print("训练集成模型 (Stacking)")
        print("="*60)
        
        self.feature_cols = feature_cols
        
        # 准备数据
        df_clean = df.dropna(subset=feature_cols + [target_col])
        X = df_clean[feature_cols].values
        y = df_clean[target_col].values
        
        print(f"训练样本: {len(X):,}")
        print(f"特征数量: {len(feature_cols)}")
        print(f"可用模型: {self.available_models}")
        
        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 分割
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"训练集: {len(X_train):,}, 验证集: {len(X_val):,}")
        
        # 训练基学习器
        print("\n训练基学习器...")
        base_predictions_train = np.zeros((len(X_train), len(self.available_models)))
        base_predictions_val = np.zeros((len(X_val), len(self.available_models)))
        
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for i, model_name in enumerate(self.available_models):
            print(f"\n  训练 {model_name}...")
            
            model = self._get_base_model(model_name)
            if model is None:
                print(f"    跳过 {model_name} (不可用)")
                continue
            
            # 使用交叉验证生成元特征
            cv_preds = cross_val_predict(model, X_train, y_train, cv=kfold, method='predict_proba')[:, 1]
            base_predictions_train[:, i] = cv_preds
            
            # 在全量训练集上训练
            model.fit(X_train, y_train)
            self.models[model_name] = model
            
            # 验证集预测
            val_pred = model.predict_proba(X_val)[:, 1]
            base_predictions_val[:, i] = val_pred
            
            # 单模型评估
            val_pred_label = (val_pred > 0.5).astype(int)
            acc = accuracy_score(y_val, val_pred_label)
            print(f"    验证准确率: {acc:.4f}")
        
        # 训练元学习器
        print("\n训练元学习器 (LogisticRegression)...")
        self.meta_learner = LogisticRegression(C=1.0, max_iter=1000)
        self.meta_learner.fit(base_predictions_train, y_train)
        
        # 最终预测
        final_pred_proba = self.meta_learner.predict_proba(base_predictions_val)[:, 1]
        final_pred = (final_pred_proba > 0.5).astype(int)
        
        # 评估
        accuracy = accuracy_score(y_val, final_pred)
        precision = precision_score(y_val, final_pred)
        recall = recall_score(y_val, final_pred)
        f1 = f1_score(y_val, final_pred)
        auc = roc_auc_score(y_val, final_pred_proba)
        
        print(f"\n{'='*60}")
        print("集成模型评估结果")
        print("="*60)
        print(f"验证准确率: {accuracy:.4f}")
        print(f"验证精确率: {precision:.4f}")
        print(f"验证召回率: {recall:.4f}")
        print(f"验证F1: {f1:.4f}")
        print(f"验证AUC: {auc:.4f}")
        
        # 各模型权重
        print(f"\n元学习器权重:")
        for i, model_name in enumerate(self.available_models):
            if model_name in self.models:
                print(f"  {model_name}: {self.meta_learner.coef_[0][i]:.4f}")
        
        # 保存模型
        self._save_models(accuracy, auc)
        
        return True
    
    def _save_models(self, accuracy: float, auc: float):
        """保存模型"""
        model_path = os.path.join(self.model_dir, 'ensemble_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'ensemble_scaler.pkl')
        
        joblib.dump({
            'models': self.models,
            'meta_learner': self.meta_learner,
            'feature_cols': self.feature_cols
        }, model_path)
        
        joblib.dump(self.scaler, scaler_path)
        
        # 保存状态
        self.model_status = {
            'trained': True,
            'n_models': len(self.models),
            'models': list(self.models.keys()),
            'val_acc': accuracy,
            'val_auc': auc,
            'train_time': datetime.now().isoformat()
        }
        
        status_path = os.path.join(self.model_dir, 'ensemble_status.json')
        with open(status_path, 'w') as f:
            json.dump(self.model_status, f, indent=2)
        
        print(f"\n模型已保存: {model_path}")
    
    def predict(self, X: np.ndarray) -> Dict:
        """
        预测
        
        Args:
            X: 特征矩阵 (已标准化)
        
        Returns:
            预测结果
        """
        if not self.models or self.meta_learner is None:
            return {'error': '模型未训练'}
        
        # 标准化
        X_scaled = self.scaler.transform(X)
        
        # 获取各模型预测
        base_preds = np.zeros((len(X), len(self.available_models)))
        
        for i, model_name in enumerate(self.available_models):
            if model_name in self.models:
                base_preds[:, i] = self.models[model_name].predict_proba(X_scaled)[:, 1]
        
        # 元学习器预测
        final_proba = self.meta_learner.predict_proba(base_preds)[:, 1]
        final_pred = (final_proba > 0.5).astype(int)
        
        return {
            'prediction': final_pred,
            'probability': final_proba,
            'method': 'ensemble'
        }
    
    def predict_single(self, features: Dict) -> Dict:
        """
        预测单只股票
        
        Args:
            features: 特征字典
        
        Returns:
            预测结果
        """
        X = np.array([[features.get(col, 0) for col in self.feature_cols]])
        result = self.predict(X)
        
        return {
            'method': 'ensemble',
            'prediction': '涨' if result['prediction'][0] == 1 else '跌',
            'probability': result['probability'][0],
            'confidence': max(result['probability'][0], 1 - result['probability'][0])
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性（平均）"""
        if not self.models:
            return pd.DataFrame()
        
        importance_dict = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[model_name] = model.feature_importances_
        
        if not importance_dict:
            return pd.DataFrame()
        
        df = pd.DataFrame(importance_dict, index=self.feature_cols)
        df['mean'] = df.mean(axis=1)
        df = df.sort_values('mean', ascending=False)
        
        return df
    
    def get_status(self) -> Dict:
        """获取模型状态"""
        return self.model_status


def optimize_hyperparameters(X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> Dict:
    """
    使用Optuna优化超参数
    
    Args:
        X: 特征
        y: 目标
        n_trials: 试验次数
    
    Returns:
        最优参数
    """
    if not OPTUNA_AVAILABLE:
        print("Optuna未安装，跳过优化")
        return {}
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
        
        if LIGHTGBM_AVAILABLE:
            model = lgb.LGBMClassifier(**params, verbose=-1, random_state=42)
        else:
            return 0.5
        
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params


def main():
    """测试集成模型"""
    print("可用模型:")
    print(f"  XGBoost: {'✓' if XGBOOST_AVAILABLE else '✗'}")
    print(f"  LightGBM: {'✓' if LIGHTGBM_AVAILABLE else '✗'}")
    print(f"  CatBoost: {'✓' if CATBOOST_AVAILABLE else '✗'}")
    print(f"  Optuna: {'✓' if OPTUNA_AVAILABLE else '✗'}")


if __name__ == '__main__':
    main()