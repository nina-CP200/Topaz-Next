#!/usr/bin/env python3
"""
特征工程模块 - 专业量化因子库
参考: WorldQuant Alpha101, Barra风险模型
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """特征工程 - 生成100+因子"""
    
    def __init__(self):
        self.feature_names = []
    
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成所有特征
        
        Args:
            df: 原始数据 (code, date, open, high, low, close, volume)
        
        Returns:
            添加特征后的DataFrame
        """
        df = df.copy()
        
        # 按股票分组计算
        df = df.groupby('code', group_keys=False).apply(self._generate_features_for_stock)
        
        # 添加时间特征
        df = self._add_time_features(df)
        
        # 添加市场特征
        df = self._add_market_features(df)
        
        return df
    
    def _generate_features_for_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        """为单只股票生成特征"""
        
        # ===== 1. 价格位置因子 =====
        df = self._price_position_factors(df)
        
        # ===== 2. 动量因子 =====
        df = self._momentum_factors(df)
        
        # ===== 3. 波动率因子 =====
        df = self._volatility_factors(df)
        
        # ===== 4. 成交量因子 =====
        df = self._volume_factors(df)
        
        # ===== 5. 技术指标因子 =====
        df = self._technical_indicators(df)
        
        # ===== 6. 价格形态因子 =====
        df = self._pattern_factors(df)
        
        # ===== 7. 统计因子 =====
        df = self._statistical_factors(df)
        
        return df
    
    def _price_position_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """价格位置因子"""
        
        # 相对于均线的位置
        for period in [5, 10, 20, 60]:
            df[f'ma{period}'] = df['close'].rolling(period).mean()
            df[f'price_to_ma{period}'] = df['close'] / df[f'ma{period}'] - 1
            df[f'ma{period}_slope'] = df[f'ma{period}'].pct_change(5)
        
        # 相对于高低点的位置
        for period in [10, 20]:
            df[f'high{period}'] = df['high'].rolling(period).max()
            df[f'low{period}'] = df['low'].rolling(period).min()
            df[f'price_position_{period}'] = (df['close'] - df[f'low{period}']) / (df[f'high{period}'] - df[f'low{period}'])
        
        # 收盘价在当日范围的位置
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        
        return df
    
    def _momentum_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """动量因子"""
        
        # 收益率
        for period in [1, 3, 5, 10, 20, 60]:
            df[f'return_{period}d'] = df['close'].pct_change(period)
        
        # 价格动量（价格变化率）
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # ROC (Rate of Change)
        for period in [10, 20]:
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
        
        # 动量加速度
        df['momentum_accel'] = df['return_5d'] - df['return_5d'].shift(5)
        
        return df
    
    def _volatility_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """波动率因子"""
        
        # 历史波动率
        for period in [5, 10, 20, 60]:
            df[f'volatility_{period}'] = df['return_1d'].rolling(period).std() * np.sqrt(252)
        
        # 波动率变化
        df['volatility_change'] = df['volatility_5'] / df['volatility_20'] - 1
        
        # ATR (Average True Range)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_14'] = df['tr'].rolling(14).mean()
        df['atr_ratio'] = df['atr_14'] / df['close']
        
        # 偏度（收益分布不对称性）
        df['skewness_20'] = df['return_1d'].rolling(20).skew()
        
        # 峰度（尾部风险）
        df['kurtosis_20'] = df['return_1d'].rolling(20).kurt()
        
        # 上行/下行波动率
        up_returns = df['return_1d'].where(df['return_1d'] > 0, 0)
        down_returns = df['return_1d'].where(df['return_1d'] < 0, 0)
        df['up_volatility_20'] = up_returns.rolling(20).std() * np.sqrt(252)
        df['down_volatility_20'] = down_returns.rolling(20).std() * np.sqrt(252)
        
        return df
    
    def _volume_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量因子"""
        
        # 成交量均线
        for period in [5, 10, 20]:
            df[f'volume_ma{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma{period}']
        
        # 量价相关性
        df['price_volume_corr_10'] = df['close'].pct_change().rolling(10).corr(df['volume'].pct_change())
        
        # OBV (On Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_ma10'] = df['obv'].rolling(10).mean()
        
        # 成交量变化率
        df['volume_change'] = df['volume'].pct_change()
        
        # Amihud非流动性指标
        df['amihud'] = abs(df['return_1d']) / (df['volume'] + 1e-8)
        df['amihud_ma20'] = df['amihud'].rolling(20).mean()
        
        # 成交金额
        df['turnover'] = df['close'] * df['volume']
        df['turnover_ma5'] = df['turnover'].rolling(5).mean()
        
        return df
    
    def _technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """技术指标因子"""
        
        # RSI
        for period in [6, 14, 24]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-8)
            df[f'rsi_{period}'] = 100 - 100 / (1 + rs)
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # KDJ
        low_9 = df['low'].rolling(9).min()
        high_9 = df['high'].rolling(9).max()
        df['kdj_k'] = (df['close'] - low_9) / (high_9 - low_9 + 1e-8) * 100
        df['kdj_d'] = df['kdj_k'].rolling(3).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        
        # 布林带
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        
        # CCI
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['cci'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-8)
        
        # Williams %R
        df['williams_r'] = (df['high'].rolling(14).max() - df['close']) / (df['high'].rolling(14).max() - df['low'].rolling(14).min() + 1e-8) * -100
        
        # ADX (趋势强度)
        df['plus_dm'] = df['high'].diff().where(lambda x: x > 0, 0)
        df['minus_dm'] = (-df['low'].diff()).where(lambda x: x > 0, 0)
        df['tr_14'] = df['tr'].rolling(14).sum()
        df['plus_di'] = 100 * df['plus_dm'].rolling(14).sum() / (df['tr_14'] + 1e-8)
        df['minus_di'] = 100 * df['minus_dm'].rolling(14).sum() / (df['tr_14'] + 1e-8)
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-8)
        df['adx'] = df['dx'].rolling(14).mean()
        
        return df
    
    def _pattern_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """价格形态因子"""
        
        # K线实体和影线
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # 实体占比
        df['body_ratio'] = df['body'] / (df['high'] - df['low'] + 1e-8)
        
        # 阳线/阴线
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        
        # 跳空
        df['gap'] = df['open'] / df['close'].shift(1) - 1
        
        # 连续涨跌
        df['consecutive_up'] = (df['return_1d'] > 0).rolling(5).sum()
        df['consecutive_down'] = (df['return_1d'] < 0).rolling(5).sum()
        
        # 形态识别（简化版）
        # 锤子线
        df['is_hammer'] = (
            (df['lower_shadow'] > 2 * df['body']) &
            (df['upper_shadow'] < df['body'] * 0.5)
        ).astype(int)
        
        # 射击之星
        df['is_shooting_star'] = (
            (df['upper_shadow'] > 2 * df['body']) &
            (df['lower_shadow'] < df['body'] * 0.5)
        ).astype(int)
        
        # 吞没形态
        df['is_engulfing'] = (
            (df['is_bullish'] != df['is_bullish'].shift(1)) &
            (df['body'] > df['body'].shift(1))
        ).astype(int)
        
        return df
    
    def _statistical_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """统计因子"""
        
        # 均值回归
        df['mean_reversion_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        
        # 价格分位数
        df['price_percentile_20'] = df['close'].rolling(20).rank(pct=True)
        
        # 收益自相关
        df['return_autocorr_5'] = df['return_1d'].rolling(20).apply(
            lambda x: x.autocorr(lag=5) if len(x) > 5 else 0, raw=False
        )
        
        # 最大回撤
        cummax = df['close'].cummax()
        drawdown = (cummax - df['close']) / cummax
        df['max_drawdown_20'] = drawdown.rolling(20).max()
        
        # 信息比率相关
        df['sharpe_proxy'] = df['return_1d'].rolling(20).mean() / (df['return_1d'].rolling(20).std() + 1e-8)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """时间特征"""
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            
            # 月初月末效应
            df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
            df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
            
            # 周几效应（one-hot）
            for day in range(5):
                df[f'is_day_{day}'] = (df['day_of_week'] == day).astype(int)
        
        return df
    
    def _add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """市场特征（需要全市场数据）"""
        
        # 按日期计算市场统计
        if 'date' in df.columns:
            # 涨跌家数比
            market_stats = df.groupby('date').agg({
                'return_1d': ['mean', 'std', lambda x: (x > 0).sum() / len(x)]
            }).reset_index()
            market_stats.columns = ['date', 'market_return', 'market_volatility', 'advance_ratio']
            
            df = df.merge(market_stats, on='date', how='left')
            
            # 相对市场表现
            df['relative_return'] = df['return_1d'] - df['market_return']
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """获取所有特征名称"""
        return self.feature_names
    
    def select_features(self, df: pd.DataFrame, method: str = 'importance') -> List[str]:
        """
        特征选择
        
        Args:
            df: 数据框
            method: 选择方法 ('importance', 'correlation', 'shap')
        
        Returns:
            选择的特征列表
        """
        # 排除非特征列
        exclude_cols = ['code', 'date', 'open', 'high', 'low', 'close', 'volume',
                        'future_return', 'target']
        
        features = [col for col in df.columns if col not in exclude_cols]
        
        # 移除NaN过多的特征
        valid_features = []
        for feat in features:
            if df[feat].notna().mean() > 0.7:  # 至少70%有效值
                valid_features.append(feat)
        
        return valid_features


def main():
    """测试特征工程"""
    import sys
    
    # 示例：生成测试数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    codes = ['600000', '600519', '000001']
    
    data = []
    for code in codes:
        for date in dates:
            data.append({
                'code': code,
                'date': date,
                'open': 10 + np.random.randn() * 0.1,
                'high': 10.2 + np.random.randn() * 0.1,
                'low': 9.8 + np.random.randn() * 0.1,
                'close': 10 + np.random.randn() * 0.1,
                'volume': 1000000 + np.random.randint(-100000, 100000)
            })
    
    df = pd.DataFrame(data)
    
    # 生成特征
    fe = FeatureEngineer()
    df_features = fe.generate_all_features(df)
    
    # 获取有效特征
    features = fe.select_features(df_features)
    
    print(f"生成特征数: {len(features)}")
    print(f"数据形状: {df_features.shape}")
    print(f"\n示例特征:")
    for feat in features[:20]:
        print(f"  - {feat}")


if __name__ == '__main__':
    main()