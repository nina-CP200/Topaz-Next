#!/usr/bin/env python3
"""
专业量化预测系统 v4.0
整合：特征工程 + 深度学习 + 模型集成 + 风控优化
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from feature_engineer import FeatureEngineer
from ensemble_model import EnsembleModel, XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE, CATBOOST_AVAILABLE
from risk_management import RiskManager

# 检查深度学习
try:
    from deep_learning import DeepLearningPredictor, TORCH_AVAILABLE
except ImportError:
    TORCH_AVAILABLE = False


class TopazPredictor:
    """Topaz预测系统 - 专业量化架构"""
    
    def __init__(self, data_dir: str = '.', model_dir: str = '.'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # 初始化各模块
        self.feature_engineer = FeatureEngineer()
        self.ensemble = EnsembleModel(model_dir)
        self.risk_manager = RiskManager()
        
        if TORCH_AVAILABLE:
            self.dl_predictor = DeepLearningPredictor(model_dir)
        else:
            self.dl_predictor = None
        
        self.feature_cols = []
        self.status = {
            'version': '4.0',
            'stage': 'initialized',
            'modules': {
                'feature_engineering': True,
                'deep_learning': TORCH_AVAILABLE,
                'ensemble': len(self.ensemble.available_models) > 0,
                'risk_management': True
            }
        }
    
    def train(self, 
              data_file: str = 'training_data.csv',
              target_days: int = 5,
              use_dl: bool = True,
              use_ensemble: bool = True):
        """
        训练完整预测系统
        
        Args:
            data_file: 训练数据文件
            target_days: 预测窗口
            use_dl: 是否使用深度学习
            use_ensemble: 是否使用集成模型
        """
        print("\n" + "="*70)
        print("Topaz 专业量化预测系统 v4.0 - 训练")
        print("="*70)
        
        # 加载数据
        data_path = os.path.join(self.data_dir, data_file)
        if not os.path.exists(data_path):
            print(f"❌ 数据文件不存在: {data_path}")
            return False
        
        print(f"\n1. 加载数据...")
        df = pd.read_csv(data_path)
        print(f"   原始数据: {len(df):,} 行")
        
        # 特征工程
        print(f"\n2. 特征工程...")
        df = self.feature_engineer.generate_all_features(df)
        self.feature_cols = self.feature_engineer.select_features(df)
        print(f"   生成特征: {len(self.feature_cols)} 个")
        
        # 准备目标变量
        print(f"\n3. 准备目标变量...")
        df['future_return'] = df.groupby('code')['close'].shift(-target_days) / df['close'] - 1
        
        # 三分类：涨(2) / 平(1) / 跌(0)
        df['target'] = pd.cut(
            df['future_return'], 
            bins=[-np.inf, -0.02, 0.02, np.inf],
            labels=[0, 1, 2]
        ).astype(float)
        
        # 或者二分类
        df['target_binary'] = (df['future_return'] > 0).astype(float)
        
        # 移除NaN
        df_clean = df.dropna(subset=self.feature_cols + ['target_binary'])
        print(f"   有效样本: {len(df_clean):,}")
        
        # 训练集成模型
        ensemble_acc = 0
        if use_ensemble and len(self.ensemble.available_models) > 0:
            print(f"\n4. 训练集成模型...")
            print(f"   可用模型: {self.ensemble.available_models}")
            self.ensemble.train(df_clean, self.feature_cols, target_col='target_binary')
            ensemble_acc = self.ensemble.model_status.get('val_acc', 0)
        
        # 训练深度学习模型
        dl_acc = 0
        if use_dl and TORCH_AVAILABLE and self.dl_predictor:
            print(f"\n5. 训练深度学习模型...")
            self.dl_predictor.train(
                df_clean, self.feature_cols,
                target_col='target_binary',
                model_type='lstm',
                epochs=30  # 快速训练
            )
            dl_acc = self.dl_predictor.model_status.get('val_acc', 0)
        
        # 选择最佳模型
        print(f"\n6. 选择最佳模型...")
        print(f"   集成模型准确率: {ensemble_acc:.4f}")
        print(f"   深度学习准确率: {dl_acc:.4f}")
        
        best_model = 'ensemble' if ensemble_acc >= dl_acc else 'dl'
        best_acc = max(ensemble_acc, dl_acc)
        
        print(f"   最佳模型: {best_model} ({best_acc:.4f})")
        
        # 保存状态
        self.status.update({
            'trained': True,
            'best_model': best_model,
            'best_acc': best_acc,
            'ensemble_acc': ensemble_acc,
            'dl_acc': dl_acc,
            'n_features': len(self.feature_cols),
            'train_time': datetime.now().isoformat()
        })
        
        self._save_status()
        
        return True
    
    def predict(self, code: str, history: pd.DataFrame = None) -> Dict:
        """
        预测单只股票
        
        Args:
            code: 股票代码
            history: 历史数据（可选）
        
        Returns:
            预测结果
        """
        result = {
            'code': code,
            'timestamp': datetime.now().isoformat()
        }
        
        # 获取历史数据
        if history is None:
            # 从数据目录加载
            data_path = os.path.join(self.data_dir, 'training_data.csv')
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                history = df[df['code'] == code].tail(100)
        
        if history is None or len(history) < 20:
            result['error'] = '历史数据不足'
            return result
        
        # 特征工程
        df_features = self.feature_engineer.generate_all_features(history)
        
        # 获取最新特征
        latest = df_features.iloc[-1:][self.feature_cols]
        if latest.isna().any().any():
            latest = latest.fillna(0)
        
        # 集成模型预测
        if self.status.get('best_model') == 'ensemble':
            try:
                features_dict = latest.iloc[0].to_dict()
                pred = self.ensemble.predict_single(features_dict)
                result.update(pred)
            except Exception as e:
                result['error'] = str(e)
        
        # 深度学习预测
        elif self.status.get('best_model') == 'dl' and self.dl_predictor:
            try:
                pred = self.dl_predictor.predict(history, code)
                result.update(pred)
            except Exception as e:
                result['error'] = str(e)
        
        # 风险评估
        if 'probability' in result:
            # 简化风险提示
            if result['confidence'] < 0.55:
                result['risk_warning'] = '预测置信度较低，建议谨慎'
        
        return result
    
    def batch_predict(self, codes: List[str], df: pd.DataFrame = None) -> pd.DataFrame:
        """
        批量预测
        
        Args:
            codes: 股票代码列表
            df: 全量数据
        
        Returns:
            预测结果DataFrame
        """
        results = []
        
        if df is None:
            data_path = os.path.join(self.data_dir, 'training_data.csv')
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
        
        for code in codes:
            try:
                result = self.predict(code, df[df['code'] == code] if df is not None else None)
                results.append(result)
            except Exception as e:
                results.append({'code': code, 'error': str(e)})
        
        return pd.DataFrame(results)
    
    def get_status(self) -> Dict:
        """获取系统状态"""
        return self.status
    
    def _save_status(self):
        """保存状态"""
        status_path = os.path.join(self.model_dir, 'topaz_status.json')
        with open(status_path, 'w') as f:
            json.dump(self.status, f, indent=2, ensure_ascii=False)
        print(f"\n状态已保存: {status_path}")
    
    def print_report(self):
        """打印系统报告"""
        print("\n" + "="*70)
        print("Topaz 预测系统报告")
        print("="*70)
        print(f"版本: {self.status.get('version', 'unknown')}")
        print(f"训练状态: {'已训练' if self.status.get('trained') else '未训练'}")
        
        if self.status.get('trained'):
            print(f"\n最佳模型: {self.status.get('best_model', 'unknown')}")
            print(f"最佳准确率: {self.status.get('best_acc', 0):.4f}")
            print(f"集成准确率: {self.status.get('ensemble_acc', 0):.4f}")
            print(f"深度学习准确率: {self.status.get('dl_acc', 0):.4f}")
            print(f"特征数量: {self.status.get('n_features', 0)}")
            print(f"训练时间: {self.status.get('train_time', 'unknown')}")
        
        print(f"\n模块状态:")
        for module, available in self.status.get('modules', {}).items():
            status = '✓' if available else '✗'
            print(f"  {status} {module}")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='Topaz预测系统')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--predict', type=str, help='预测指定股票')
    parser.add_argument('--status', action='store_true', help='查看状态')
    parser.add_argument('--report', action='store_true', help='打印报告')
    parser.add_argument('--no-dl', action='store_true', help='不使用深度学习')
    args = parser.parse_args()
    
    predictor = TopazPredictor()
    
    if args.train:
        predictor.train(use_dl=not args.no_dl)
    elif args.predict:
        result = predictor.predict(args.predict)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.status:
        status = predictor.get_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))
    elif args.report:
        predictor.print_report()
    else:
        predictor.print_report()


if __name__ == '__main__':
    main()