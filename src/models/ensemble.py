#!/usr/bin/env python3
"""
================================================================================
集成模型模块 - Stacking + Meta-Learner
================================================================================

【模块功能】
本模块实现了基于 Stacking（堆叠）策略的集成学习模型，通过融合多个基学习器的预测
结果，显著提升股票涨跌预测的准确率和稳定性。

【模型架构说明】
采用两层架构设计：

┌─────────────────────────────────────────────────────────────────────┐
│                         Stacking 架构图                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│    输入特征 ──► ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐  │
│                │ XGBoost  │ │ LightGBM │ │ CatBoost │ │   RF    │  │
│                └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬────┘  │
│                     │            │            │            │        │
│                     ▼            ▼            ▼            ▼        │
│               预测概率P1     预测概率P2     预测概率P3     预测概率P4   │
│                     │            │            │            │        │
│                     └────────────┴────────────┴────────────┘        │
│                                        │                            │
│                                        ▼                            │
│                              ┌─────────────────┐                    │
│                              │  Meta-Learner   │                    │
│                              │ (LogisticReg.)  │                    │
│                              └────────┬────────┘                    │
│                                        │                            │
│                                        ▼                            │
│                                   最终预测结果                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

【基学习器说明】

1. XGBoost（梯度提升树）
   - 特点：高效、支持并行计算、内置正则化防止过拟合
   - 适用场景：中大规模数据集，特征交互复杂
   - 关键参数：n_estimators, max_depth, learning_rate, subsample
   
2. LightGBM（轻量级梯度提升机）
   - 特点：训练速度快、内存占用低、支持类别特征
   - 适用场景：大规模数据，需要快速训练
   - 关键参数：n_estimators, max_depth, num_leaves, learning_rate
   
3. CatBoost（类别型特征增强梯度提升）
   - 特点：自动处理类别特征、减少调参需求、鲁棒性强
   - 适用场景：包含大量类别特征的数据
   - 关键参数：iterations, depth, learning_rate, l2_leaf_reg
   
4. RandomForest（随机森林）
   - 特点：抗过拟合、可并行训练、稳定性好
   - 适用场景：特征维度高、样本量中等
   - 关键参数：n_estimators, max_depth, min_samples_split
   
5. GBDT（梯度提升决策树）
   - 特点：sklearn原生实现、兼容性好、可解释性强
   - 适用场景：需要sklearn生态兼容时
   - 关键参数：n_estimators, max_depth, learning_rate

【元学习器说明】
- 当前使用：LogisticRegression（逻辑回归）
- 作用：学习各基学习器的最优权重组合
- 优势：简单、快速、不易过拟合、可解释性强

【工作流程】
1. 训练阶段：
   - 使用K折交叉验证生成基学习器的元特征（out-of-fold预测）
   - 在完整训练集上训练每个基学习器
   - 用元特征训练元学习器，学习最优权重组合
   
2. 预测阶段：
   - 各基学习器独立预测，输出概率值
   - 元学习器根据学习到的权重进行集成预测
   - 输出最终预测结果和置信度

【性能优化建议】
- 数据量大时：优先使用LightGBM，减少训练时间
- 追求高精度：增加基学习器数量，使用Optuna调参
- 内存受限：减少n_estimators或使用更少基学习器
- 防止过拟合：降低max_depth、增加min_samples_split

================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 依赖导入
# ─────────────────────────────────────────────────────────────────────────────

# 标准库
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 梯度提升框架（可选依赖，按需安装）
# 安装命令：
#   pip install xgboost      # XGBoost
#   pip install lightgbm     # LightGBM
#   pip install catboost     # CatBoost
#   pip install optuna       # 超参数优化
# ─────────────────────────────────────────────────────────────────────────────

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

# sklearn 相关模块
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib


class EnsembleModel:
    """
    集成模型类 - XGBoost + LightGBM + CatBoost Stacking
    
    ════════════════════════════════════════════════════════════════════════════
    【类概述】
    实现基于 Stacking 策略的多模型集成，融合多个梯度提升模型的预测结果，
    通过元学习器自动学习最优权重组合，提升预测准确率和稳定性。
    
    【初始化参数】
    ────────────────────────────────────────────────────────────────────────────
    参数名        类型      默认值      说明
    ────────────────────────────────────────────────────────────────────────────
    model_dir     str       '.'        模型文件保存目录，用于持久化存储训练好的模型
                                      - 训练后会自动保存以下文件：
                                        • ensemble_model.pkl   (模型权重)
                                        • ensemble_scaler.pkl (标准化器)
                                        • ensemble_status.json(模型状态)
    
    【核心属性】
    ────────────────────────────────────────────────────────────────────────────
    属性名              类型                  说明
    ────────────────────────────────────────────────────────────────────────────
    models              Dict[str, Model]      训练后的基学习器字典，键为模型名称
    scaler              StandardScaler        特征标准化器，预测时需使用相同标准化
    meta_learner        LogisticRegression    元学习器，学习基学习器的最优权重
    feature_cols        List[str]             特征列名列表，预测时特征顺序必须一致
    model_status        Dict                  模型状态信息（训练时间、准确率等）
    available_models    List[str]             当前环境可用的基学习器列表
    
    【模型保存/加载机制】
    ────────────────────────────────────────────────────────────────────────────
    保存机制：
    1. 训练完成后自动调用 _save_models() 保存
    2. 使用 joblib 序列化模型，保存为 .pkl 文件
    3. 元学习器和基学习器打包保存在 ensemble_model.pkl
    4. 标准化器单独保存在 ensemble_scaler.pkl
    5. 训练状态保存为 ensemble_status.json（JSON格式，便于查看）
    
    加载机制：
    1. 初始化时自动调用 _load_models() 尝试加载已有模型
    2. 优先从 ensemble_model.pkl 加载完整模型包
    3. 兼容旧版本：如果模型包不含scaler，则单独加载 scaler 文件
    4. 加载成功后自动恢复 models, meta_learner, feature_cols 等属性
    
    【使用示例】
    ────────────────────────────────────────────────────────────────────────────
    >>> # 初始化并训练
    >>> ensemble = EnsembleModel(model_dir='./models')
    >>> ensemble.train(df, feature_cols=['ma5', 'ma10', 'rsi', ...])
    
    >>> # 预测（加载已有模型后直接预测）
    >>> ensemble = EnsembleModel(model_dir='./models')
    >>> result = ensemble.predict_single({'ma5': 10.5, 'ma10': 10.2, ...})
    
    ════════════════════════════════════════════════════════════════════════════
    """
    
    def __init__(self, model_dir: str = 'data/models'):
        """
        初始化集成模型
        
        Args:
            model_dir: 模型保存目录，默认为当前目录
                       指定目录后，训练好的模型会自动保存至此
                       下次初始化时会自动加载该目录下的已有模型
        """
        self.model_dir = model_dir
        self.models = {}
        self.scaler = None
        self.meta_learner = None
        self.feature_cols = []
        self.model_status = {'trained': False}
        
        # ─────────────────────────────────────────────────────────────────────
        # 检测当前环境可用的基学习器
        # 说明：不同环境可能安装了不同的梯度提升框架
        # - 如需使用特定模型，请先安装对应库
        # - 即使部分模型不可用，仍可使用其他可用模型进行集成
        # ─────────────────────────────────────────────────────────────────────
        self.available_models = []
        if XGBOOST_AVAILABLE:
            self.available_models.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            self.available_models.append('lightgbm')
        if CATBOOST_AVAILABLE:
            self.available_models.append('catboost')
        # RF和GBDT是sklearn内置，始终可用
        self.available_models.extend(['rf', 'gbdt'])
        
        # 尝试加载已保存的模型
        self._load_models()
    
    def _load_models(self):
        """
        加载已保存的模型
        
        【加载流程】
        1. 检查 ensemble_model.pkl 是否存在
        2. 存在则加载模型包（包含 models, meta_learner, feature_cols）
        3. 加载标准化器（支持新旧两种格式）
        4. 加载模型状态文件 ensemble_status.json
        
        【文件格式】
        - ensemble_model.pkl: joblib序列化的字典
          {
              'models': Dict[str, Model],      # 基学习器字典
              'meta_learner': LogisticRegression,  # 元学习器
              'feature_cols': List[str],       # 特征列名
              'scaler': StandardScaler         # 标准化器（新版格式）
          }
        - ensemble_scaler.pkl: 标准化器（旧版格式，兼容）
        - ensemble_status.json: 训练状态（JSON格式）
        """
        model_path = os.path.join(self.model_dir, 'ensemble_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'ensemble_scaler.pkl')
        status_path = os.path.join(self.model_dir, 'ensemble_status.json')
        
        if os.path.exists(model_path):
            try:
                saved = joblib.load(model_path)
                self.models = saved['models']
                self.meta_learner = saved.get('meta_learner', None)
                self.feature_cols = saved['feature_cols']
                
                # 检查是否包含标准化器（新格式）
                if 'scaler' in saved:
                    self.scaler = saved['scaler']
                elif os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                
                if os.path.exists(status_path):
                    with open(status_path, 'r') as f:
                        self.model_status = json.load(f)
                
                print(f"已加载集成模型: {list(self.models.keys())}")
            except Exception as e:
                print(f"加载模型失败: {e}")
    
    def _get_base_model(self, name: str, params: Dict = None):
        """
        获取基学习器实例
        
        【参数说明】
        ────────────────────────────────────────────────────────────────────────────
        参数名      类型        说明
        ────────────────────────────────────────────────────────────────────────────
        name        str        模型名称，可选值：
                              - 'xgboost'   : XGBoost分类器
                              - 'lightgbm'  : LightGBM分类器
                              - 'catboost'  : CatBoost分类器
                              - 'rf'        : 随机森林分类器
                              - 'gbdt'      : sklearn梯度提升分类器
        params      Dict       自定义参数，会覆盖默认参数
        
        【默认参数配置】
        ────────────────────────────────────────────────────────────────────────────
        模型          n_estimators  max_depth  learning_rate  其他参数
        ────────────────────────────────────────────────────────────────────────────
        xgboost       200           6          0.05           subsample=0.8, colsample_bytree=0.8
        lightgbm      200           6          0.05           subsample=0.8, colsample_bytree=0.8
        catboost      200(iters)    6          0.05           verbose=0
        rf            200           10         -              min_samples_split=50, min_samples_leaf=25
        gbdt          150           5          0.05           min_samples_split=50, min_samples_leaf=25
        
        【调优建议】
        ────────────────────────────────────────────────────────────────────────────
        过拟合时：
          - 降低 max_depth（如 4-5）
          - 降低 learning_rate，增加 n_estimators
          - 增加正则化参数（如 subsample=0.6-0.8）
          
        欠拟合时：
          - 增加 max_depth（如 8-10）
          - 增加 n_estimators（如 300-500）
          - 增加 learning_rate（如 0.1）
          
        训练速度慢：
          - 减少 n_estimators
          - 使用 LightGBM 替代 XGBoost
          - 增加最小样本数限制
        
        Returns:
            模型实例，如果模型不可用则返回 None
        """
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
        
        【参数说明】
        ────────────────────────────────────────────────────────────────────────────
        参数名            类型            默认值      说明
        ────────────────────────────────────────────────────────────────────────────
        df                pd.DataFrame    必填        训练数据集，需包含特征列和目标列
        feature_cols      List[str]       必填        特征列名列表，顺序需与预测时一致
                                                    建议特征：
                                                    - 技术指标：ma5, ma10, ma20, rsi, macd, kdj
                                                    - 价格特征：close, high, low, volume
                                                    - 波动特征：atr, bollinger_band_width
        target_col        str             'target'   目标列名，值为0/1（跌/涨）
        test_size         float           0.2        验证集比例，范围(0, 1)
                                                    - 数据量大时可适当增大（如0.3）
                                                    - 数据量小时建议0.15-0.2
        n_folds           int             5          交叉验证折数，用于生成元特征
                                                    - 数据量大时可用3-5折
                                                    - 数据量小时可用5-10折
        use_optimization  bool            False      是否使用Optuna进行超参数优化
                                                    注意：开启会显著增加训练时间
        
        【训练流程】
        ────────────────────────────────────────────────────────────────────────────
        1. 数据预处理
           - 移除含缺失值的样本
           - 特征标准化（StandardScaler）
           - 划分训练集和验证集（分层抽样）
        
        2. 基学习器训练（Stacking第一阶段）
           - 使用K折交叉验证生成元特征（out-of-fold预测）
           - 每个基学习器独立训练，输出预测概率
           - 保存所有基学习器模型
        
        3. 元学习器训练（Stacking第二阶段）
           - 输入：各基学习器的预测概率
           - 输出：加权后的最终预测
           - 使用逻辑回归学习最优权重组合
        
        4. 模型评估与保存
           - 计算验证集指标：Accuracy, Precision, Recall, F1, AUC
           - 保存模型文件和状态信息
        
        【调优建议】
        ────────────────────────────────────────────────────────────────────────────
        1. 提升准确率：
           - 增加高质量特征（技术指标、因子等）
           - 使用Optuna自动调参（设置use_optimization=True）
           - 增加基学习器多样性
        
        2. 加速训练：
           - 减少n_estimators或n_folds
           - 仅使用部分基学习器（在available_models中筛选）
           - 关闭超参数优化
        
        3. 防止过拟合：
           - 增加训练数据量
           - 降低max_depth
           - 增加正则化（subsample, colsample_bytree）
        
        4. 处理不平衡数据：
           - 在_get_base_model中设置class_weight='balanced'
           - 或在XGBoost/LightGBM中设置scale_pos_weight
        
        Returns:
            bool: 训练是否成功
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
        
        # 【重要修正】时序数据必须按时间划分，不能随机划分
        # 随机划分会导致同一股票不同日期同时出现在训练/验证集，造成数据泄露
        # 正确做法：按时间顺序，前80%训练，后20%验证
        
        split_idx = int(len(X_scaled) * (1 - test_size))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"训练集: {len(X_train):,}, 验证集: {len(X_val):,}")
        print(f"⚠️ 使用时间序列划分，避免数据泄露")
        
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
        """
        保存训练好的模型
        
        【保存机制】
        使用 joblib 进行模型序列化，保存以下文件：
        - ensemble_model.pkl: 包含所有模型的字典
        - ensemble_scaler.pkl: 特征标准化器
        - ensemble_status.json: 训练状态（JSON格式，便于查看）
        
        【文件内容】
        ensemble_status.json 结构：
        {
            "trained": true,
            "n_models": 5,
            "models": ["xgboost", "lightgbm", "catboost", "rf", "gbdt"],
            "val_acc": 0.85,
            "val_auc": 0.92,
            "train_time": "2024-01-15T10:30:00"
        }
        
        Args:
            accuracy: 验证集准确率，用于状态记录
            auc: 验证集AUC，用于状态记录
        """
        model_path = os.path.join(self.model_dir, 'ensemble_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'ensemble_scaler.pkl')
        
        # 保存模型（新版格式：包含 scaler）
        joblib.dump({
            'models': self.models,
            'meta_learner': self.meta_learner,
            'feature_cols': self.feature_cols,
            'scaler': self.scaler  # 新版：scaler 直接包含在模型文件中
        }, model_path)

        # 单独保存 scaler（兼容旧版本）
        joblib.dump(self.scaler, scaler_path)
        
        # 保存状态
        self.model_status = {
            'trained': True,
            'n_models': len(self.models),
            'models': list(self.models.keys()),
            'val_acc': accuracy,
            'val_auc': auc,
            'train_time': datetime.now().isoformat(),
            'scaler_included': True  # 标记新版格式
        }

        status_path = os.path.join(self.model_dir, 'ensemble_status.json')
        with open(status_path, 'w') as f:
            json.dump(self.model_status, f, indent=2)

        print(f"\n模型已保存: {model_path}")
        print(f"标准化器已保存: {scaler_path}")
    
    def predict(self, X: np.ndarray) -> Dict:
        """
        批量预测
        
        【输入格式】
        ────────────────────────────────────────────────────────────────────────────
        参数    类型              形状              说明
        ────────────────────────────────────────────────────────────────────────────
        X       np.ndarray       (n_samples, n_features)
                                              特征矩阵，列顺序必须与训练时的
                                              feature_cols 一致
                                              注意：输入原始特征值，函数内部会
                                              自动进行标准化
        
        【输出格式】
        ────────────────────────────────────────────────────────────────────────────
        返回值：Dict，包含以下字段：
        {
            'prediction': np.ndarray,    # 预测标签，形状(n_samples,)
                                         # 0=跌, 1=涨
            'probability': np.ndarray,   # 预测概率，形状(n_samples,)
                                         # 表示上涨概率，范围[0,1]
            'method': str                 # 预测方法标识，固定为'ensemble'
        }
        
        【使用示例】
        ────────────────────────────────────────────────────────────────────────────
        >>> # 假设训练时使用了5个特征
        >>> X = np.array([[10.5, 10.2, 0.8, 0.6, 1000],
        ...               [10.8, 10.5, 0.7, 0.5, 1200]])
        >>> result = ensemble.predict(X)
        >>> print(result['prediction'])  # [1, 0]
        >>> print(result['probability']) # [0.75, 0.32]
        
        Args:
            X: 特征矩阵，形状(n_samples, n_features)
        
        Returns:
            Dict: 包含prediction, probability, method的字典
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
        
        【输入格式】
        ────────────────────────────────────────────────────────────────────────────
        参数       类型       说明
        ────────────────────────────────────────────────────────────────────────────
        features   Dict      特征字典，键为特征名，值为特征值
                            键名必须与训练时的 feature_cols 一致
                            缺失的特征会自动填充为0
        
        【输出格式】
        ────────────────────────────────────────────────────────────────────────────
        返回值：Dict，包含以下字段：
        {
            'method': str,           # 预测方法，固定为'ensemble'
            'prediction': str,       # 预测结果，'涨' 或 '跌'
            'probability': float,    # 上涨概率，范围[0,1]
            'confidence': float      # 预测置信度，范围[0.5,1]
                                     # = max(probability, 1-probability)
        }
        
        【使用示例】
        ────────────────────────────────────────────────────────────────────────────
        >>> # 根据训练时的特征名提供特征值
        >>> features = {
        ...     'ma5': 10.5,
        ...     'ma10': 10.2,
        ...     'rsi': 65.0,
        ...     'macd': 0.15,
        ...     'volume': 1000000
        ... }
        >>> result = ensemble.predict_single(features)
        >>> print(result['prediction'])   # '涨'
        >>> print(result['probability'])  # 0.78
        >>> print(result['confidence'])   # 0.78
        
        Args:
            features: 特征字典
        
        Returns:
            Dict: 包含method, prediction, probability, confidence的字典
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
        """
        获取特征重要性（各模型平均）
        
        【返回值】
        DataFrame，索引为特征名，列为各模型的重要性分数：
        - 各列：各基学习器的特征重要性
        - 'mean'列：各特征的平均重要性（按此降序排列）
        
        【使用示例】
        >>> importance_df = ensemble.get_feature_importance()
        >>> print(importance_df.head(10))  # 查看最重要的10个特征
        
        【应用场景】
        - 特征选择：移除重要性低的特征，减少噪声
        - 模型解释：理解哪些特征对预测影响最大
        - 特征工程：针对重要特征进行优化
        
        Returns:
            pd.DataFrame: 特征重要性表，按平均重要性降序排列
        """
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
        """
        获取模型状态
        
        【返回值】
        Dict，包含以下字段：
        {
            'trained': bool,          # 是否已训练
            'n_models': int,          # 基学习器数量
            'models': List[str],      # 基学习器名称列表
            'val_acc': float,         # 验证集准确率
            'val_auc': float,         # 验证集AUC
            'train_time': str         # 训练时间（ISO格式）
        }
        
        Returns:
            Dict: 模型状态信息
        """
        return self.model_status


def optimize_hyperparameters(X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> Dict:
    """
    使用Optuna优化超参数
    
    【功能说明】
    使用贝叶斯优化搜索最优超参数组合，相比网格搜索更高效。
    需要 pip install optuna
    
    【参数说明】
    ────────────────────────────────────────────────────────────────────────────
    参数        类型          默认值      说明
    ────────────────────────────────────────────────────────────────────────────
    X           np.ndarray    必填       特征矩阵，形状(n_samples, n_features)
    y           np.ndarray    必填       目标标签，形状(n_samples,)
    n_trials    int           50         优化试验次数
                                      - 次数越多，可能找到更优参数
                                      - 但训练时间也会增加
                                      - 建议范围：30-100
    
    【搜索空间】
    ────────────────────────────────────────────────────────────────────────────
    参数                搜索范围              说明
    ────────────────────────────────────────────────────────────────────────────
    n_estimators        [100, 300]           树的数量，越多越精确但越慢
    max_depth           [3, 10]              树的最大深度，防止过拟合
    learning_rate       [0.01, 0.1]          学习率，越小越稳定但需要更多树
    subsample           [0.6, 1.0]           样本采样比例，防止过拟合
    colsample_bytree    [0.6, 1.0]           特征采样比例，防止过拟合
    
    【返回值】
    Dict: 最优参数组合，可直接传给模型
    
    【使用示例】
    ────────────────────────────────────────────────────────────────────────────
    >>> # 优化超参数
    >>> best_params = optimize_hyperparameters(X_train, y_train, n_trials=50)
    >>> print(best_params)
    {'n_estimators': 250, 'max_depth': 7, 'learning_rate': 0.03, ...}
    
    >>> # 使用最优参数训练模型
    >>> model = lgb.LGBMClassifier(**best_params)
    >>> model.fit(X_train, y_train)
    
    【注意事项】
    ────────────────────────────────────────────────────────────────────────────
    1. 优化过程可能较慢，建议在GPU或高性能CPU上运行
    2. 优化结果可能因随机种子不同而略有差异
    3. 优化后仍需验证在测试集上的表现，防止过拟合验证集
    
    Args:
        X: 特征矩阵
        y: 目标标签
        n_trials: 优化试验次数
    
    Returns:
        Dict: 最优超参数
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
    """
    模块测试入口
    
    运行 python ensemble_model.py 会输出当前环境可用的模型库：
    - XGBoost: 需要安装 xgboost
    - LightGBM: 需要安装 lightgbm
    - CatBoost: 需要安装 catboost
    - Optuna: 需要安装 optuna（用于超参数优化）
    """
    print("可用模型:")
    print(f"  XGBoost: {'✓' if XGBOOST_AVAILABLE else '✗'}")
    print(f"  LightGBM: {'✓' if LIGHTGBM_AVAILABLE else '✗'}")
    print(f"  CatBoost: {'✓' if CATBOOST_AVAILABLE else '✗'}")
    print(f"  Optuna: {'✓' if OPTUNA_AVAILABLE else '✗'}")


if __name__ == '__main__':
    main()