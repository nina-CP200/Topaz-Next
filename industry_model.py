#!/usr/bin/env python3
"""
行业分组ML模型 - QuantPilot
按申万一级行业分组训练独立的预测模型
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 申万一级行业分类
SW_INDUSTRIES = {
    '银行': ['601398', '601288', '601939', '601328', '600036', '600016', '601166', '600000', '601818', '600015'],
    '非银金融': ['601318', '601601', '600030', '601688', '600837', '601211', '600999', '601901', '600958', '601788'],
    '食品饮料': ['600519', '000858', '000568', '600887', '000333', '002304', '600809', '000596', '000860', '002568'],
    '医药生物': ['300760', '300122', '000661', '002007', '300347', '603259', '000963', '002821', '300003', '688180'],
    '电子': ['000725', '002475', '300408', '002241', '603501', '300661', '002600', '603160', '300628', '688981'],
    '计算机': ['002230', '300033', '600588', '002410', '300454', '688111', '300212', '002405', '688588', '300377'],
    '电力设备': ['300750', '300014', '002594', '300124', '601012', '002129', '600089', '600438', '300274', '002074'],
    '汽车': ['002594', '601238', '000625', '600104', '601633', '000338', '600066', '002920', '601799', '603799'],
    '有色金属': ['601899', '002460', '600547', '601600', '000960', '600111', '002466', '603993', '600489', '000962'],
    '化工': ['600309', '002493', '600426', '000703', '002648', '600586', '002466', '000912', '600989', '002648'],
    # 更多行业...
}

# 行业映射（股票代码 -> 行业）
def build_industry_map():
    """构建股票代码到行业的映射"""
    industry_map = {}
    for industry, codes in SW_INDUSTRIES.items():
        for code in codes:
            industry_map[code] = industry
    return industry_map

INDUSTRY_MAP = build_industry_map()


class IndustryMLModel:
    """行业分组ML模型"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.models = {}  # industry -> (clf, scaler)
        
    def get_industry(self, code):
        """获取股票所属行业"""
        return INDUSTRY_MAP.get(code, '其他')
    
    def prepare_features(self, df):
        """准备特征"""
        # 技术指标特征
        features = [
            'ma_5', 'ma_10', 'ma_20', 'ma_60',
            'rsi_14', 'rsi_6',
            'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'volume_ratio', 'amount_ratio',
            'price_change_5d', 'price_change_10d', 'price_change_20d',
            'volume_change_5d', 'volume_change_10d',
            'volatility_10d', 'volatility_20d',
            'high_low_ratio', 'open_close_ratio'
        ]
        
        # 过滤存在的特征
        available = [f for f in features if f in df.columns]
        return df[available].fillna(0)
    
    def prepare_target(self, df, horizon=5):
        """准备目标变量"""
        # 未来N天涨跌
        df['future_return'] = df.groupby('code')['close'].shift(-horizon) / df['close'] - 1
        df['target'] = (df['future_return'] > 0).astype(int)
        return df
    
    def train_industry_model(self, df, industry):
        """训练单个行业的模型"""
        # 准备数据
        X = self.prepare_features(df)
        y = df['target']
        
        # 移除NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 100:
            print(f"  [{industry}] 样本不足: {len(X)}")
            return None, None
        
        # 分割数据
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 训练模型（快速配置）
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=50,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42
        )
        clf.fit(X_train_scaled, y_train)
        
        # 验证
        train_acc = clf.score(X_train_scaled, y_train)
        val_acc = clf.score(X_val_scaled, y_val)
        gap = train_acc - val_acc
        
        print(f"  [{industry}] 训练: {train_acc:.2%}, 验证: {val_acc:.2%}, 差距: {gap:.2%}")
        
        # 检查过拟合
        if gap > 0.15 or val_acc < 0.52:
            print(f"  [{industry}] ⚠️ 过拟合或准确率过低，标记为不可用")
            return None, None
        
        return clf, scaler
    
    def train_all(self, df):
        """训练所有行业模型"""
        print("=" * 60)
        print("开始训练行业分组模型")
        print("=" * 60)
        
        # 准备目标变量
        df = self.prepare_target(df)
        
        # 按行业分组训练
        industries = df['industry'].unique() if 'industry' in df.columns else ['全部']
        
        for industry in industries:
            if industry == '全部':
                industry_df = df
            else:
                industry_df = df[df['industry'] == industry]
            
            if len(industry_df) < 200:
                continue
            
            print(f"\n训练 [{industry}] ({len(industry_df)} 样本)...")
            clf, scaler = self.train_industry_model(industry_df, industry)
            
            if clf is not None:
                self.models[industry] = (clf, scaler)
                
                # 保存模型
                joblib.dump(clf, self.models_dir / f'model_{industry}.pkl')
                joblib.dump(scaler, self.models_dir / f'scaler_{industry}.pkl')
        
        # 保存模型状态
        status = {
            'train_time': datetime.now().isoformat(),
            'industries': list(self.models.keys()),
            'total_models': len(self.models)
        }
        with open(self.models_dir / 'industry_status.json', 'w') as f:
            json.dump(status, f, indent=2)
        
        print(f"\n✓ 训练完成，共 {len(self.models)} 个行业模型")
        return self.models
    
    def predict(self, code, features):
        """预测单只股票"""
        industry = self.get_industry(code)
        
        if industry not in self.models:
            industry = '其他'
        
        if industry not in self.models:
            return 0.5  # 无模型时返回中性概率
        
        clf, scaler = self.models[industry]
        
        # 标准化
        X = scaler.transform([features])
        
        # 预测概率
        prob = clf.predict_proba(X)[0][1]
        return prob
    
    def load_models(self):
        """加载已保存的模型"""
        if not self.models_dir.exists():
            return False
        
        # 加载状态
        status_file = self.models_dir / 'industry_status.json'
        if not status_file.exists():
            return False
        
        with open(status_file) as f:
            status = json.load(f)
        
        # 加载模型
        for industry in status['industries']:
            model_file = self.models_dir / f'model_{industry}.pkl'
            scaler_file = self.models_dir / f'scaler_{industry}.pkl'
            
            if model_file.exists() and scaler_file.exists():
                clf = joblib.load(model_file)
                scaler = joblib.load(scaler_file)
                self.models[industry] = (clf, scaler)
        
        print(f"加载了 {len(self.models)} 个行业模型")
        return len(self.models) > 0


def main():
    """训练行业分组模型"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--test', action='store_true', help='测试模型')
    args = parser.parse_args()
    
    model = IndustryMLModel()
    
    if args.train:
        # 加载数据
        df = pd.read_pickle('csi300_features.pkl')
        
        # 添加行业列
        df['industry'] = df['code'].apply(model.get_industry)
        
        # 训练
        model.train_all(df)
    
    if args.test:
        # 测试加载
        model.load_models()
        print(f"模型状态: {model.models.keys()}")


if __name__ == '__main__':
    main()