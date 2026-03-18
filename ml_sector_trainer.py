#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topaz V3 - 板块机器学习训练模块
按板块分类训练模型，提升预测准确率
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 导入现有模块
from topaz_data_api import get_cn_history_data, get_cn_realtime_data
from ml_stock_analysis import MLStockAnalyzer


class SectorMLTrainer:
    """板块机器学习训练器"""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.dirname(os.path.abspath(__file__))
        self.stocks = []
        self.stock_industry = {}
        self.industry_stocks = defaultdict(list)
        self.sector_models = {}
        
    def load_data(self):
        """加载成分股和行业映射"""
        # 加载成分股
        stocks_file = os.path.join(self.data_dir, "csi300_stocks.json")
        if os.path.exists(stocks_file):
            with open(stocks_file, 'r', encoding='utf-8') as f:
                self.stocks = json.load(f)
            print(f"✓ 加载 {len(self.stocks)} 只成分股")
        
        # 加载行业映射
        mapping_file = os.path.join(self.data_dir, "csi300_industry_map.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            self.stock_industry = mapping.get('stock_industry', {})
            
            # 构建行业->股票的反向映射
            for code, industry in self.stock_industry.items():
                self.industry_stocks[industry].append(code)
            
            print(f"✓ 加载 {len(self.stock_industry)} 只股票的行业映射")
            print(f"✓ 覆盖 {len(self.industry_stocks)} 个行业")
    
    def get_sector_history(self, industry: str, days: int = 60) -> pd.DataFrame:
        """获取某个板块所有股票的历史数据"""
        codes = self.industry_stocks.get(industry, [])
        if not codes:
            return None
        
        all_data = []
        for code in codes[:10]:  # 限制数量避免请求过多
            try:
                data = get_cn_history_data(code, days=days)
                if data:
                    for d in data:
                        d['code'] = code
                    all_data.extend(data)
                time.sleep(0.1)
            except:
                continue
        
        if not all_data:
            return None
        
        df = pd.DataFrame(all_data)
        return df
    
    def calculate_sector_features(self, industry: str) -> Dict:
        """计算板块特征"""
        codes = self.industry_stocks.get(industry, [])
        if not codes:
            return {}
        
        features = {
            'industry': industry,
            'stock_count': len(codes),
            'avg_change': 0,
            'avg_volume': 0,
            'avg_ma5': 0,
            'avg_ma10': 0,
            'avg_ma20': 0,
            'momentum': 0,  # 板块动量
            'volatility': 0,  # 板块波动率
            'trend': 'neutral'  # 板块趋势
        }
        
        changes = []
        volumes = []
        ma5_list = []
        ma10_list = []
        ma20_list = []
        
        for code in codes[:20]:  # 取前20只计算
            try:
                data = get_cn_realtime_data(code)
                if data:
                    changes.append(data.get('change_pct', 0))
                    volumes.append(data.get('volume', 0))
                    if 'ma5' in data:
                        ma5_list.append(data['ma5'])
                    if 'ma10' in data:
                        ma10_list.append(data['ma10'])
                    if 'ma20' in data:
                        ma20_list.append(data['ma20'])
                time.sleep(0.1)
            except:
                continue
        
        if changes:
            features['avg_change'] = np.mean(changes)
            features['volatility'] = np.std(changes)
            
            if features['avg_change'] > 1:
                features['trend'] = 'bullish'
            elif features['avg_change'] < -1:
                features['trend'] = 'bearish'
        
        if volumes:
            features['avg_volume'] = np.mean(volumes)
        
        if ma5_list:
            features['avg_ma5'] = np.mean(ma5_list)
        if ma10_list:
            features['avg_ma10'] = np.mean(ma10_list)
        if ma20_list:
            features['avg_ma20'] = np.mean(ma20_list)
        
        # 板块动量 = 平均涨跌幅 * 股票数量
        features['momentum'] = features['avg_change'] * features['stock_count']
        
        return features
    
    def train_sector_model(self, industry: str):
        """训练单个板块的模型"""
        codes = self.industry_stocks.get(industry, [])
        if len(codes) < 3:
            return None
        
        print(f"\n训练 {industry} 板块模型 ({len(codes)} 只股票)...")
        
        # 收集训练数据
        training_data = []
        for code in codes:
            try:
                # 获取历史数据
                history = get_cn_history_data(code, days=60)
                if not history or len(history) < 20:
                    continue
                
                # 计算因子
                for i in range(20, len(history) - 5):
                    window = history[i-20:i]
                    future = history[i:i+5]
                    
                    # 过去20天的因子
                    past_close = [d['close'] for d in window]
                    past_volume = [d['volume'] for d in window]
                    
                    # 未来5天收益
                    future_close = [d['close'] for d in future]
                    future_return = (future_close[-1] - past_close[-1]) / past_close[-1]
                    
                    # 计算特征
                    features = {
                        'ma5': np.mean(past_close[-5:]),
                        'ma10': np.mean(past_close[-10:]),
                        'ma20': np.mean(past_close),
                        'momentum': (past_close[-1] - past_close[-5]) / past_close[-5],
                        'volatility': np.std(past_close[-10:]) / np.mean(past_close[-10:]),
                        'volume_ratio': past_volume[-1] / np.mean(past_volume) if np.mean(past_volume) > 0 else 1,
                        'industry': industry
                    }
                    
                    training_data.append((features, future_return))
                
                time.sleep(0.1)
                
            except Exception as e:
                continue
        
        if len(training_data) < 10:
            print(f"  数据不足，跳过")
            return None
        
        # 转换为DataFrame
        X = pd.DataFrame([t[0] for t in training_data])
        X = X.select_dtypes(include=[np.number])  # 只保留数值特征
        y = np.array([t[1] for t in training_data])
        
        # 简单统计模型（可替换为更复杂的ML模型）
        model = {
            'industry': industry,
            'samples': len(training_data),
            'avg_return': np.mean(y),
            'std_return': np.std(y),
            'positive_ratio': np.sum(y > 0) / len(y),
            'feature_means': X.mean().to_dict(),
            'feature_stds': X.std().to_dict(),
            'train_time': datetime.now().isoformat()
        }
        
        print(f"  样本数: {model['samples']}")
        print(f"  平均收益: {model['avg_return']*100:.2f}%")
        print(f"  正收益比例: {model['positive_ratio']*100:.1f}%")
        
        return model
    
    def train_all_sectors(self):
        """训练所有板块模型"""
        print("\n" + "="*60)
        print("板块机器学习训练")
        print("="*60)
        
        self.load_data()
        
        trained = 0
        for industry in list(self.industry_stocks.keys())[:30]:  # 限制30个板块
            if len(self.industry_stocks[industry]) >= 3:
                model = self.train_sector_model(industry)
                if model:
                    self.sector_models[industry] = model
                    trained += 1
        
        # 保存模型
        model_file = os.path.join(self.data_dir, "sector_models.json")
        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump(self.sector_models, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 训练完成，保存 {trained} 个板块模型到 sector_models.json")
        
        return self.sector_models
    
    def predict_with_sector(self, code: str) -> Dict:
        """结合板块特征进行预测"""
        # 获取股票的行业
        industry = self.stock_industry.get(code, 'unknown')
        
        # 获取板块模型
        sector_model = self.sector_models.get(industry, {})
        
        # 获取个股数据
        stock_data = get_cn_realtime_data(code)
        if not stock_data:
            return {'error': '无法获取股票数据'}
        
        # 基础预测（多因子）
        base_prediction = {
            'code': code,
            'name': stock_data.get('name', ''),
            'current_price': stock_data.get('price', 0),
            'change_pct': stock_data.get('change_pct', 0),
            'industry': industry
        }
        
        # 板块增强预测
        if sector_model:
            # 个股因子
            stock_momentum = stock_data.get('change_pct', 0) / 100
            
            # 板块因子
            sector_momentum = sector_model.get('avg_return', 0)
            sector_positive_ratio = sector_model.get('positive_ratio', 0.5)
            
            # 组合预测
            predicted_return = (
                stock_momentum * 0.4 +  # 个股动量
                sector_momentum * 0.3 +  # 板块动量
                sector_positive_ratio * 0.1 - 0.05  # 板块正收益概率
            )
            
            base_prediction['sector_momentum'] = sector_momentum
            base_prediction['sector_positive_ratio'] = sector_positive_ratio
            base_prediction['predicted_return'] = predicted_return
            base_prediction['confidence'] = 'high' if sector_model.get('samples', 0) > 100 else 'medium'
        else:
            base_prediction['predicted_return'] = stock_data.get('change_pct', 0) / 100
            base_prediction['confidence'] = 'low'
        
        return base_prediction


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='板块机器学习训练')
    parser.add_argument('--train', action='store_true', help='训练板块模型')
    parser.add_argument('--predict', type=str, help='预测指定股票')
    parser.add_argument('--sector', type=str, help='查看指定板块信息')
    args = parser.parse_args()
    
    trainer = SectorMLTrainer()
    
    if args.train:
        trainer.train_all_sectors()
    elif args.predict:
        result = trainer.predict_with_sector(args.predict)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.sector:
        features = trainer.calculate_sector_features(args.sector)
        print(json.dumps(features, ensure_ascii=False, indent=2))
    else:
        # 默认：显示状态
        trainer.load_data()
        print(f"\n成分股: {len(trainer.stocks)}")
        print(f"行业映射: {len(trainer.stock_industry)}")
        print(f"行业数量: {len(trainer.industry_stocks)}")
        
        # 显示行业分布
        from collections import Counter
        ind_count = Counter(trainer.stock_industry.values())
        print("\n主要行业:")
        for ind, cnt in ind_count.most_common(10):
            print(f"  {ind}: {cnt} 只")


if __name__ == '__main__':
    main()