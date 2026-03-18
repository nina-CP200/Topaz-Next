#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qlib 模型预测模块
使用 LightGBM 进行股票收益预测
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'qlib'))

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple
from datetime import datetime


def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建技术特征
    
    Parameters
    ----------
    df : pd.DataFrame
        包含 OHLCV 的历史数据
    
    Returns
    -------
    pd.DataFrame
        添加技术特征后的数据
    """
    data = df.copy()
    
    # 移动平均线
    data['ma5'] = data['close'].rolling(5).mean()
    data['ma10'] = data['close'].rolling(10).mean()
    data['ma20'] = data['close'].rolling(20).mean()
    
    # MACD
    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = exp1 - exp2
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # 布林带
    data['bb_middle'] = data['close'].rolling(20).mean()
    data['bb_std'] = data['close'].rolling(20).std()
    data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
    data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']
    data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    
    # 成交量特征
    data['volume_ma5'] = data['volume'].rolling(5).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma5']
    
    # 价格动量
    data['momentum_5d'] = data['close'].pct_change(5)
    data['momentum_10d'] = data['close'].pct_change(10)
    
    # 波动率
    data['volatility_5d'] = data['close'].pct_change().rolling(5).std()
    data['volatility_20d'] = data['close'].pct_change().rolling(20).std()
    
    # 价格位置
    data['high_low_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 0.001)
    
    return data


class QlibPredictor:
    """基于 Qlib 风格的股票收益预测器"""
    
    def __init__(self, model_params: Dict = None):
        """
        初始化预测器
        
        Parameters
        ----------
        model_params : Dict
            LightGBM 模型参数
        """
        self.model_params = model_params or {
            'objective': 'mse',
            'learning_rate': 0.05,
            'max_depth': 6,
            'num_leaves': 31,
            'n_estimators': 100,
            'random_state': 42,
            'verbosity': -1
        }
        self.model = None
        self.feature_cols = None
        self.trained_symbols = set()
    
    def train_with_history(self, history_data: Dict[str, pd.DataFrame], 
                           predict_days: int = 5) -> Dict:
        """
        使用历史数据训练模型
        
        Parameters
        ----------
        history_data : Dict[str, pd.DataFrame]
            {symbol: history_df, ...} 历史数据字典
        predict_days : int
            预测未来 N 天的收益率
            
        Returns
        -------
        Dict
            训练结果
        """
        all_features = []
        all_labels = []
        
        for symbol, df in history_data.items():
            if df is None or len(df) < 30:
                print(f"  {symbol} 数据不足，跳过")
                continue
            
            # 创建技术特征
            df_features = create_technical_features(df)
            
            # 检查并删除重复列
            df_features = df_features.loc[:, ~df_features.columns.duplicated()]
            
            # 生成标签：未来 N 天收益率
            df_features['label'] = df_features['close'].shift(-predict_days) / df_features['close'] - 1
            
            # 添加基本面特征（如果有）
            if 'pe_ratio' in df.columns:
                df_features['pe_ratio'] = df['pe_ratio']
            if 'pb_ratio' in df.columns:
                df_features['pb_ratio'] = df['pb_ratio']
            if 'roe' in df.columns:
                df_features['roe'] = df['roe']
            
            df_features['symbol'] = symbol
            
            # 删除缺失值
            df_features = df_features.dropna()
            
            if len(df_features) > 0:
                all_features.append(df_features)
                self.trained_symbols.add(symbol)
        
        if not all_features:
            raise ValueError("没有有效的训练数据")
        
        # 合并所有数据
        combined_df = pd.concat(all_features, ignore_index=False)
        
        # 删除重复列
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

        # 获取特征列（排除非数值列）
        exclude_cols = ['label', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'date', 'datetime']
        self.feature_cols = [c for c in combined_df.columns if c not in exclude_cols 
                            and combined_df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

        if not self.feature_cols:
            # 如果没有技术特征，使用基础特征
            self.feature_cols = ['open', 'high', 'low', 'close', 'volume']

        X = combined_df[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = combined_df['label']

        # 删除全零或全 NaN 的特征
        valid_cols = X.columns[(X != 0).any() & (~X.isna()).all()]
        X = X[valid_cols]
        self.feature_cols = list(valid_cols)
        
        # 划分训练/验证集 (时间序列，不能用随机划分)
        split_idx = int(len(X) * 0.8)
        X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"  训练样本：{len(X_train)}, 验证样本：{len(X_valid)}")
        print(f"  特征数量：{len(self.feature_cols)}")
        
        # 训练模型
        self.model = lgb.LGBMRegressor(**self.model_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        # 计算评估指标
        train_pred = self.model.predict(X_train)
        valid_pred = self.model.predict(X_valid)
        
        from sklearn.metrics import mean_squared_error, r2_score
        
        result = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'valid_mse': mean_squared_error(y_valid, valid_pred),
            'train_r2': r2_score(y_train, train_pred),
            'valid_r2': r2_score(y_valid, valid_pred),
            'trained_symbols': list(self.trained_symbols),
            'feature_importance': self.get_feature_importance()
        }
        
        return result
    
    def prepare_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        准备特征数据
        
        Parameters
        ----------
        feature_df : pd.DataFrame
            原始特征 DataFrame
            
        Returns
        -------
        pd.DataFrame
            处理后的特征
        """
        df = feature_df.copy()
        
        # 处理缺失值
        df = df.fillna(0)
        
        # 替换无穷值
        df = df.replace([np.inf, -np.inf], 0)
        
        # 标准化处理（Z-Score）
        for col in df.columns:
            if col in self.feature_cols or col.startswith('$'):
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
        
        return df
    
    def fit(self, feature_df: pd.DataFrame, label: pd.Series, 
            valid_ratio: float = 0.2) -> Dict:
        """
        训练模型
        
        Parameters
        ----------
        feature_df : pd.DataFrame
            特征数据
        label : pd.Series
            标签（未来收益率）
        valid_ratio : float
            验证集比例
            
        Returns
        -------
        Dict
            训练结果
        """
        # 准备特征
        X = self.prepare_features(feature_df)
        y = label
        
        # 对齐索引
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # 划分训练/验证集
        split_idx = int(len(X) * (1 - valid_ratio))
        X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 训练模型
        self.model = lgb.LGBMRegressor(**self.model_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # 计算评估指标
        train_pred = self.model.predict(X_train)
        valid_pred = self.model.predict(X_valid)
        
        from sklearn.metrics import mean_squared_error, r2_score
        
        result = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'valid_mse': mean_squared_error(y_valid, valid_pred),
            'train_r2': r2_score(y_train, train_pred),
            'valid_r2': r2_score(y_valid, valid_pred),
            'feature_importance': dict(zip(
                self.feature_cols,
                self.model.feature_importances_
            ))
        }
        
        return result
    
    def predict(self, feature_df: pd.DataFrame) -> pd.Series:
        """
        预测股票收益
        
        Parameters
        ----------
        feature_df : pd.DataFrame
            特征数据
            
        Returns
        -------
        pd.Series
            预测收益率，index 为 instrument
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 fit()")
        
        X = self.prepare_features(feature_df)
        predictions = self.model.predict(X)
        
        return pd.Series(predictions, index=feature_df.index, name='predicted_return')
    
    def get_feature_importance(self) -> Dict:
        """获取特征重要性"""
        if self.model is None:
            return {}
        return dict(zip(
            self.feature_cols,
            self.model.feature_importances_
        ))
    
    def predict_single(self, history_df: pd.DataFrame, current_data: Dict = None) -> float:
        """
        对单只股票进行预测
        
        Parameters
        ----------
        history_df : pd.DataFrame
            该股票的历史数据
        current_data : Dict
            当前基本面数据（可选）
            
        Returns
        -------
        float
            预测收益率 (%)
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train_with_history()")
        
        # 创建技术特征
        df_features = create_technical_features(history_df)
        
        # 添加基本面特征
        if current_data:
            for key in ['pe_ratio', 'pb_ratio', 'roe']:
                if key in current_data and current_data[key]:
                    df_features[key] = current_data[key]
        
        # 取最后一行数据
        df_features = df_features.dropna()
        if len(df_features) == 0:
            return 0.0
        
        last_row = df_features.iloc[[-1]]
        
        # 确保特征列一致
        X = last_row[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
        # 预测
        prediction = self.model.predict(X)[0]
        
        return prediction * 100  # 转换为百分比


class SimpleReturnPredictor:
    """
    简化版收益预测器（不依赖历史数据）
    使用多因子评分进行预测
    """
    
    # 行业因子权重配置
    INDUSTRY_WEIGHTS = {
        # 科技/芯片/互联网 - 看重成长和动量
        '科技': {'value': 0.15, 'quality': 0.25, 'momentum': 0.30, 'volatility': 0.15, 'dividend': 0.15},
        '芯片': {'value': 0.15, 'quality': 0.25, 'momentum': 0.30, 'volatility': 0.15, 'dividend': 0.15},
        '互联网': {'value': 0.15, 'quality': 0.25, 'momentum': 0.30, 'volatility': 0.15, 'dividend': 0.15},
        '电商': {'value': 0.15, 'quality': 0.25, 'momentum': 0.30, 'volatility': 0.15, 'dividend': 0.15},
        '社交': {'value': 0.15, 'quality': 0.25, 'momentum': 0.30, 'volatility': 0.15, 'dividend': 0.15},
        '新能源车': {'value': 0.15, 'quality': 0.25, 'momentum': 0.30, 'volatility': 0.15, 'dividend': 0.15},
        
        # 金融 - 看重价值和分红
        '银行': {'value': 0.30, 'quality': 0.20, 'momentum': 0.15, 'volatility': 0.15, 'dividend': 0.20},
        '证券': {'value': 0.30, 'quality': 0.20, 'momentum': 0.15, 'volatility': 0.15, 'dividend': 0.20},
        '保险': {'value': 0.30, 'quality': 0.20, 'momentum': 0.15, 'volatility': 0.15, 'dividend': 0.20},
        '投行': {'value': 0.30, 'quality': 0.20, 'momentum': 0.15, 'volatility': 0.15, 'dividend': 0.20},
        '支付': {'value': 0.25, 'quality': 0.25, 'momentum': 0.20, 'volatility': 0.15, 'dividend': 0.15},
        
        # 消费 - 看重质量和分红
        '消费': {'value': 0.20, 'quality': 0.30, 'momentum': 0.15, 'volatility': 0.15, 'dividend': 0.20},
        '零售': {'value': 0.20, 'quality': 0.30, 'momentum': 0.15, 'volatility': 0.15, 'dividend': 0.20},
        '饮料': {'value': 0.20, 'quality': 0.30, 'momentum': 0.15, 'volatility': 0.15, 'dividend': 0.20},
        '餐饮': {'value': 0.20, 'quality': 0.30, 'momentum': 0.15, 'volatility': 0.15, 'dividend': 0.20},
        '日用品': {'value': 0.20, 'quality': 0.30, 'momentum': 0.15, 'volatility': 0.15, 'dividend': 0.20},
        '白酒': {'value': 0.20, 'quality': 0.30, 'momentum': 0.15, 'volatility': 0.15, 'dividend': 0.20},
        
        # 医药 - 看重质量
        '医药': {'value': 0.20, 'quality': 0.35, 'momentum': 0.15, 'volatility': 0.15, 'dividend': 0.15},
        '医疗': {'value': 0.20, 'quality': 0.35, 'momentum': 0.15, 'volatility': 0.15, 'dividend': 0.15},
        
        # 制造业 - 看重质量
        '家电': {'value': 0.20, 'quality': 0.30, 'momentum': 0.20, 'volatility': 0.15, 'dividend': 0.15},
        '电子': {'value': 0.20, 'quality': 0.30, 'momentum': 0.20, 'volatility': 0.15, 'dividend': 0.15},
        '通信': {'value': 0.20, 'quality': 0.30, 'momentum': 0.20, 'volatility': 0.15, 'dividend': 0.15},
        '通信设备': {'value': 0.20, 'quality': 0.30, 'momentum': 0.20, 'volatility': 0.15, 'dividend': 0.15},
        '电池': {'value': 0.15, 'quality': 0.30, 'momentum': 0.25, 'volatility': 0.15, 'dividend': 0.15},
        '光伏': {'value': 0.15, 'quality': 0.30, 'momentum': 0.25, 'volatility': 0.15, 'dividend': 0.15},
        
        # 能源 - 看重价值和分红
        '水电': {'value': 0.25, 'quality': 0.25, 'momentum': 0.10, 'volatility': 0.15, 'dividend': 0.25},
        '稀土': {'value': 0.25, 'quality': 0.20, 'momentum': 0.20, 'volatility': 0.20, 'dividend': 0.15},
        '航运': {'value': 0.25, 'quality': 0.20, 'momentum': 0.20, 'volatility': 0.20, 'dividend': 0.15},
        
        # 指数ETF - 看重稳定和分红
        '指数': {'value': 0.20, 'quality': 0.20, 'momentum': 0.15, 'volatility': 0.20, 'dividend': 0.25},
        'ETF': {'value': 0.20, 'quality': 0.20, 'momentum': 0.15, 'volatility': 0.20, 'dividend': 0.25},
        
        # 周期/其他 - 均衡
        '防务': {'value': 0.20, 'quality': 0.25, 'momentum': 0.20, 'volatility': 0.20, 'dividend': 0.15},
        '传媒': {'value': 0.20, 'quality': 0.25, 'momentum': 0.20, 'volatility': 0.20, 'dividend': 0.15},
        '互联网金融': {'value': 0.15, 'quality': 0.30, 'momentum': 0.25, 'volatility': 0.15, 'dividend': 0.15},
        '软件': {'value': 0.15, 'quality': 0.30, 'momentum': 0.25, 'volatility': 0.15, 'dividend': 0.15},
        '贸易': {'value': 0.25, 'quality': 0.20, 'momentum': 0.20, 'volatility': 0.20, 'dividend': 0.15},
        '渔业': {'value': 0.25, 'quality': 0.20, 'momentum': 0.20, 'volatility': 0.20, 'dividend': 0.15},
    }
    
    def __init__(self, industry: str = None):
        # 默认权重
        self.factor_weights = {
            'value': 0.25,      # 价值因子
            'quality': 0.30,    # 质量因子
            'momentum': 0.20,   # 动量因子
            'volatility': 0.15, # 波动因子
            'dividend': 0.10    # 红利因子
        }
        
        # 根据行业调整权重
        if industry:
            self.set_industry_weights(industry)
    
    def set_industry_weights(self, industry: str):
        """根据行业设置因子权重"""
        # 尝试精确匹配
        if industry in self.INDUSTRY_WEIGHTS:
            self.factor_weights = self.INDUSTRY_WEIGHTS[industry].copy()
            return
        
        # 尝试模糊匹配
        for key in self.INDUSTRY_WEIGHTS:
            if key in industry or industry in key:
                self.factor_weights = self.INDUSTRY_WEIGHTS[key].copy()
                return
        
        # 使用默认权重
    
    def calculate_factor_scores(self, stock_data: Dict) -> Dict:
        """计算多因子评分"""
        scores = {}
        
        pe = stock_data.get('pe_ratio', 0) or 0
        pb = stock_data.get('pb_ratio', 0) or 0
        roe = stock_data.get('roe', 0) or 0
        change = stock_data.get('change', 0) or 0
        div_yield = stock_data.get('dividend_yield', 0) or 0
        
        # 判断是否有基本面数据
        has_fundamentals = pe > 0 or pb > 0 or roe > 0
        
        if not has_fundamentals:
            # 美股没有基本面数据，使用纯技术因子
            return self._calculate_technical_scores(stock_data)
        
        # 价值因子 (低 PE/PB 好)
        if pe > 0:
            pe_score = max(0, 100 - (pe - 10) * 2)
        else:
            pe_score = 50
        if pb > 0:
            pb_score = max(0, 100 - (pb - 1) * 20)
        else:
            pb_score = 50
        scores['value'] = (pe_score + pb_score) / 2
        
        # 质量因子 (高 ROE 好)
        if roe > 0:
            scores['quality'] = min(100, roe * 5)
        else:
            scores['quality'] = 30
        
        # 动量因子
        scores['momentum'] = max(0, min(100, 50 + change * 5))
        
        # 波动因子 (低波动好)
        scores['volatility'] = max(0, 100 - abs(change) * 15)
        
        # 红利因子
        scores['dividend'] = min(100, div_yield * 20)
        
        return scores
    
    def _calculate_technical_scores(self, stock_data: Dict) -> Dict:
        """
        纯技术因子评分（用于没有基本面数据的股票）
        """
        scores = {}
        change = stock_data.get('change', 0) or 0
        
        # 动量因子 - 最重要 (权重高)
        # 涨多了可能回调，跌多了可能反弹
        if change > 0:
            # 上涨趋势中，回调风险
            momentum_score = max(0, min(100, 60 - change * 3))
        else:
            # 下跌趋势中，反弹机会
            momentum_score = max(0, min(100, 50 - change * 2))
        scores['momentum'] = momentum_score
        
        # 波动因子 - 低波动更稳定
        volatility_score = max(0, 100 - abs(change) * 10)
        scores['volatility'] = volatility_score
        
        # 价值因子 - 默认50分
        scores['value'] = 50
        
        # 质量因子 - 默认50分
        scores['quality'] = 50
        
        # 分红因子 - 默认30分
        scores['dividend'] = 30
        
        return scores
    
    def predict(self, stock_data: Dict) -> float:
        """
        预测未来收益
        
        Parameters
        ----------
        stock_data : Dict
            股票数据
            
        Returns
        -------
        float
            预测收益率 (%)
        """
        scores = self.calculate_factor_scores(stock_data)
        
        # 判断是否使用技术因子模式（没有基本面数据）
        has_fundamentals = (stock_data.get('pe_ratio', 0) or 0) > 0 or \
                         (stock_data.get('pb_ratio', 0) or 0) > 0 or \
                         (stock_data.get('roe', 0) or 0) > 0
        
        if not has_fundamentals:
            # 美股纯技术模式：动量40%, 波动30%, 其他30%
            weights = {
                'value': 0.0,
                'quality': 0.0,
                'momentum': 0.40,
                'volatility': 0.30,
                'dividend': 0.30
            }
        else:
            weights = self.factor_weights
        
        weighted_score = sum(
            scores[k] * weights[k]
            for k in weights
        )
        
        # 转换为预测收益率 (-30% 到 +50%)
        predicted_return = max(-30, min(50, weighted_score - 50))
        
        return predicted_return
    
    def get_investment_advice(self, predicted_return: float, 
                               stock_data: Dict) -> Tuple[str, str]:
        """
        生成投资建议
        
        Returns
        -------
        Tuple[str, str]
            (建议，风险等级)
        """
        pe = stock_data.get('pe_ratio', 0) or 0
        pb = stock_data.get('pb_ratio', 0) or 0
        roe = stock_data.get('roe', 0) or 0
        
        # 风险评估
        risk_score = 0
        if pe > 30:
            risk_score += 30
        elif pe > 20:
            risk_score += 15
        if pb > 5:
            risk_score += 25
        elif pb > 3:
            risk_score += 10
        if roe < 5:
            risk_score += 25
        elif roe < 10:
            risk_score += 10
        
        if risk_score < 20:
            risk_level = '低风险'
        elif risk_score < 40:
            risk_level = '中风险'
        elif risk_score < 60:
            risk_level = '高风险'
        else:
            risk_level = '极高风险'
        
        # 投资建议
        if risk_level in ['高风险', '极高风险']:
            advice = '建议回避'
        elif predicted_return > 25:
            advice = '建议买入'
        elif predicted_return > 15:
            advice = '可考虑买入'
        elif predicted_return > 5:
            advice = '建议持有'
        else:
            advice = '建议观望'
        
        return advice, risk_level
