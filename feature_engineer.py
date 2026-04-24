#!/usr/bin/env python3
"""
================================================================================
特征工程模块 - 专业量化因子库
================================================================================

【模块概述】
本模块实现了100+个量化因子，涵盖价格位置、动量、波动率、成交量、技术指标、价格形态、
统计因子等多个维度。因子设计参考了 WorldQuant Alpha101、Barra风险模型等经典方法论。

【因子分类说明】
================================================================================
1. 价格位置因子 (_price_position_factors)
   - 均线偏离度：衡量当前价格相对于移动平均线的位置
   - 区间位置：衡量当前价格在近期高低点区间中的相对位置
   - 收盘位置：衡量收盘价在当日K线中的位置，反映买卖力量

2. 动量因子 (_momentum_factors)
   - 收益率动量：多周期的历史收益率，捕捉趋势延续
   - 时间序列动量：跟随过去一段时间收益方向，是CTA策略核心
   - 价格动量：价格变化率，反映趋势强度
   - 动量加速度：动量变化的速度，识别趋势加速/减速
   - 均线交叉：经典的技术分析信号，捕捉中期趋势转折

3. 波动率因子 (_volatility_factors)
   - 历史波动率：收益率标准差，衡量价格波动程度
   - EWMA波动率：指数加权波动率，对近期波动更敏感
   - ATR：平均真实波幅，考虑跳空的价格波动度量
   - 偏度/峰度：收益分布的形状特征，识别尾部风险
   - 上行/下行波动：分离上涨和下跌时的波动特征

4. 成交量因子 (_volume_factors)
   - 量比：成交量相对于均量的倍数，识别异常放量/缩量
   - 量价相关性：成交量和价格变化的关系
   - OBV：能量潮指标，跟踪资金流向
   - Amihud非流动性：衡量价格对成交量的敏感度

5. 技术指标因子 (_technical_indicators)
   - RSI：相对强弱指标，衡量超买超卖
   - MACD：异同移动平均线，捕捉趋势方向和强度
   - KDJ：随机指标，衡量价格位置
   - 布林带：波动率通道，识别价格突破
   - CCI：商品通道指标
   - ADX：平均趋向指标，衡量趋势强度

6. 价格形态因子 (_pattern_factors)
   - K线形态：实体大小、影线比例
   - 经典形态：锤子线、射击之星、吞没形态等
   - 连续涨跌：统计连续上涨/下跌天数

7. 统计因子 (_statistical_factors)
   - 均值回归：Z-Score标准化后的价格位置
   - 价格分位数：价格在历史区间的百分位
   - 最大回撤：风险控制重要指标
   - 收益自相关：动量/反转信号的统计基础

【因子调优建议】
================================================================================
1. 参数调优：
   - 短周期参数(5/10日)：适合高频交易，但对噪音敏感
   - 长周期参数(60/120日)：信号更稳定，但滞后性强
   - 建议使用交叉验证选择最优参数组合

2. 因子筛选：
   - 使用 feature_importance 评估因子重要性
   - 计算因子间相关性，移除高度相关因子(>0.7)
   - 考虑因子单调性和IC值

3. 因子中性化：
   - 行业中性化：减去行业平均值
   - 市值中性化：回归剔除市值影响
   - 风险因子正交化：降低因子间共线性

4. 数据质量：
   - 注意处理缺失值和异常值
   - 建议至少使用3年以上历史数据
   - 考虑上市时间差异对因子的影响

【参考文献】
- WorldQuant Alpha101 因子库
- Barra 风险模型手册
- 《量化投资以Python为工具》蔡立耑
================================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    特征工程核心类 - 生成100+因子

    使用示例：
        fe = FeatureEngineer()
        df_features = fe.generate_all_features(df)
        features = fe.select_features(df_features)

    重要更新 (2026-04-24):
    - 添加了特征值范围自动校验和修复机制
    - KDJ、RSI、Williams %R、CCI 等技术指标现在会自动限制在合理范围内
    - 修复了训练数据与实时数据分布不一致导致的模型失效问题
    
    注意事项：
        - 输入数据需包含：code, date, open, high, low, close, volume 列
        - 因子计算会产生NaN值，建议使用前向填充或删除
        - 部分因子需要较长历史数据（如120日动量），请确保数据充足
    """
    
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
        """
        价格位置因子 - 衡量当前价格在历史价格区间中的相对位置
        
        ================================================================================
        【因子列表】
        ------------------------------------------------------------------------------
        | 因子名称              | 含义                          | 应用场景              |
        ------------------------------------------------------------------------------
        | ma{N}                 | N日移动平均线                 | 趋势判断              |
        | price_to_ma{N}        | 价格相对均线偏离度            | 均值回归策略          |
        | ma{N}_slope           | 均线斜率（5日变化率）          | 趋势强度              |
        | price_position_{N}    | 价格在N日高低点区间的位置      | 超买超卖判断          |
        | close_position        | 收盘价在当日K线中的位置        | 日内买卖力量          |
        ================================================================================
        
        【计算公式】
        - price_to_ma{N} = close / ma{N} - 1
        - ma{N}_slope = ma{N}的变化率 = (ma{N}_t - ma{N}_{t-5}) / ma{N}_{t-5}
        - price_position_{N} = (close - low_N) / (high_N - low_N)
          其中 high_N = N日最高价, low_N = N日最低价
        - close_position = (close - low) / (high - low)
        
        【参数说明】
        - period: [5, 10, 20, 60] 分别对应周线、双周、月线、季线
        - 可根据交易频率调整：短线用5/10，中线用20，长线用60
        
        【调优建议】
        - 均线周期可尝试：[3, 7, 14, 28, 56] 等斐波那契数列
        - price_to_ma 超过 ±0.1 通常视为偏离均值较远
        - price_position 接近0表示超卖，接近1表示超买
        """
        
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
        """
        动量因子 - 捕捉价格趋势延续性，是量化策略的核心因子类别
        
        ================================================================================
        【因子列表】
        ------------------------------------------------------------------------------
        | 因子名称              | 含义                          | 应用场景              |
        ------------------------------------------------------------------------------
        | return_{N}d           | N日收益率                     | 基础收益度量          |
        | tsmom_lb{N}           | 时间序列动量                  | CTA趋势跟踪策略       |
        | momentum_{N}          | 价格动量                      | 动量选股              |
        | roc_{N}               | 变化率指标                    | 超买超卖判断          |
        | momentum_accel_{N}    | 动量加速度                    | 趋势转折预警           |
        | ma_cross_{S}_{L}      | 均线交叉信号                  | 金叉死叉交易           |
        | trend_strength        | 趋势强度                      | 趋势策略过滤           |
        ================================================================================
        
        【计算公式】
        - return_{N}d = close_t / close_{t-N} - 1
        - tsmom_lb{N} = close_t / close_{t-N} - 1 （与return相同，但用于CTA语境）
        - momentum_{N} = close_t / close_{t-N} - 1
        - roc_{N} = (close_t - close_{t-N}) / close_{t-N} * 100
        - momentum_accel_{N} = return_{N}d_t - return_{N}d_{t-N}
        - ma_cross_{S}_{L} = ma{S} / ma{L} - 1
        - trend_strength = |ma5 - ma20| / ma20 * 100
        
        【参数说明】
        - 基础收益率周期: [1, 3, 5, 10, 20, 60] 覆盖短中长期
        - 时间序列动量回看期: [25, 60, 120]
          * 25天：约1个月，适合捕捉中期趋势
          * 60天：约1个季度，经典CTA参数
          * 120天：约半年，适合长期趋势跟踪
        - 均线交叉参数: [(5,20), (10,50), (20,60)]
          * 短期交叉(5/20)：信号频繁，适合短线
          * 中期交叉(10/50)：平衡信号频率和稳定性
          * 长期交叉(20/60)：信号少但可靠
        
        【调优建议】
        - 时间序列动量最优周期因市场而异，建议用滚动窗口优化
        - 动量加速度可用于识别趋势衰竭（负加速度）或加速（正加速度）
        - 可添加动量分位数因子：momentum_quantile = rank(momentum)
        - 考虑残差动量：剔除行业/市值影响后的动量
        """
        
        # 基础收益率
        for period in [1, 3, 5, 10, 20, 60]:
            df[f'return_{period}d'] = df['close'].pct_change(period)
        
        # 时间序列动量 - 多回看期（笔记实证最佳：25/60/120天）
        for lb in [25, 60, 120]:
            df[f'tsmom_lb{lb}'] = df['close'].pct_change(lb)
        
        # 价格动量（价格变化率）
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # ROC (Rate of Change)
        for period in [10, 20]:
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
        
        # 动量加速度（趋势加速/减速）
        for period in [5, 10, 20]:
            mom_col = f'return_{period}d'
            if mom_col in df.columns:
                df[f'momentum_accel_{period}'] = df[mom_col] - df[mom_col].shift(period)
        
        # 移动平均交叉信号（金叉/死叉）
        ma_pairs = [(5, 20), (10, 50), (20, 60)]
        for short, long in ma_pairs:
            short_ma = f'ma{short}'
            long_ma = f'ma{long}'
            if short_ma in df.columns and long_ma in df.columns:
                df[f'ma_cross_{short}_{long}'] = df[short_ma] / df[long_ma] - 1
        
        # 趋势强度
        if 'ma5' in df.columns and 'ma20' in df.columns:
            df['trend_strength'] = abs(df['ma5'] - df['ma20']) / df['ma20'] * 100
        
        return df
    
    def _volatility_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        波动率因子 - 衡量价格波动程度，是风险管理和仓位控制的核心
        
        ================================================================================
        【因子列表】
        ------------------------------------------------------------------------------
        | 因子名称              | 含义                          | 应用场景              |
        ------------------------------------------------------------------------------
        | volatility_{N}        | N日历史波动率（年化）          | 风险度量、止损设置     |
        | vol_ewma              | EWMA波动率（AQR方法）          | 动态风险监控          |
        | position_size         | 波动率调整后的头寸规模        | 仓位管理              |
        | vol_regime            | 波动率状态（高/低波动）        | 市场环境识别          |
        | atr_14                | 14日平均真实波幅              | 止损位设置            |
        | atr_ratio             | ATR占价格比例                 | 波动率标准化          |
        | skewness_{N}          | 收益分布偏度                  | 尾部风险识别          |
        | kurtosis_{N}          | 收益分布峰度                  | 极端事件风险评估       |
        | up/down_volatility    | 上行/下行波动率               | 非对称风险度量        |
        | vol_spike             | 波动率爆发信号                | 危机Alpha捕捉         |
        ================================================================================
        
        【计算公式】
        - volatility_{N} = std(return_1d, N) * sqrt(252)
          * 252为年化交易日数，年化便于不同周期比较
        - vol_ewma: 使用指数加权移动平均计算方差
          * EWMA方差 = sum(w_i * r_i^2) / sum(w_i)，w_i = (1-alpha)^i
          * alpha = 1 - 60/61 ≈ 0.0164，相当于约60日半衰期
        - position_size = target_vol / (vol_ewma + 0.01)
          * target_vol默认15%，高波动时减仓，低波动时加仓
        - ATR = mean(True Range, 14)
          * True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        - skewness = E[(r - mean)^3] / std^3
          * 负偏度表示下跌风险更大
        - kurtosis = E[(r - mean)^4] / std^4 - 3
          * 正峰度表示尾部更厚，极端事件更多
        
        【参数说明】
        - 波动率计算周期: [5, 10, 20, 60]
        - EWMA衰减因子: delta = 60/61 ≈ 0.9836
          * 半衰期约60天，对近期波动更敏感
          * 可调整为30/31(半衰期30天)或90/91(半衰期90天)
        - ATR周期: 14（标准参数）
        - 目标波动率: target_vol = 0.15 (15%年化)
          * 可根据策略风险偏好调整
        - position_size范围: [0.5, 2.0]
          * 限制极端仓位
        
        【调优建议】
        - 波动率计算可用Parkinson、Garman-Klass等估计器替代
        - 可添加波动率锥：比较当前波动率与历史分位数
        - VIX类指标：使用期权数据计算隐含波动率
        - 波动率状态切换：可用HMM模型识别高/低波动状态
        """
        
        # 历史波动率
        for period in [5, 10, 20, 60]:
            df[f'volatility_{period}'] = df['return_1d'].rolling(period).std() * np.sqrt(252)
        
        # EWMA 波动率（AQR 方法，权重中心60天）
        delta = 60 / 61  # ≈ 0.9836
        returns = df['return_1d'].fillna(0)
        var_ewma = returns.ewm(alpha=1-delta).var()
        df['vol_ewma'] = np.sqrt(var_ewma * 252)  # 年化
        
        # 波动率调整头寸规模因子
        target_vol = 0.15  # 目标波动率 15%
        df['position_size'] = target_vol / (df['vol_ewma'] + 0.01)
        df['position_size'] = df['position_size'].clip(0.5, 2.0)  # 限制范围
        
        # 波动率 regime
        vol_20 = df.get('volatility_20', df['return_1d'].rolling(20).std() * np.sqrt(252))
        vol_mean = vol_20.rolling(120).mean()
        df['vol_regime'] = (vol_20 - vol_mean) / (vol_20.rolling(120).std() + 0.01)
        
        # 波动率变化
        if 'volatility_5' in df.columns and 'volatility_20' in df.columns:
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
        
        # 偏度（收益分布不对称性）- 尾部风险指标
        df['skewness_20'] = df['return_1d'].rolling(20).skew()
        df['skewness_60'] = df['return_1d'].rolling(60).skew()
        
        # 峰度（尾部风险）
        df['kurtosis_20'] = df['return_1d'].rolling(20).kurt()
        
        # 上行/下行波动率
        up_returns = df['return_1d'].where(df['return_1d'] > 0, 0)
        down_returns = df['return_1d'].where(df['return_1d'] < 0, 0)
        df['up_volatility_20'] = up_returns.rolling(20).std() * np.sqrt(252)
        df['down_volatility_20'] = down_returns.rolling(20).std() * np.sqrt(252)
        
        # 波动率爆发信号（危机 Alpha）
        if 'volatility_20' in df.columns and 'volatility_60' in df.columns:
            vol_spike = df['volatility_20'] / df['volatility_60']
            df['vol_spike'] = (vol_spike > 1.5).astype(int)
        
        return df
    
    def _volume_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        成交量因子 - 分析成交量变化，识别资金流向和市场情绪
        
        ================================================================================
        【因子列表】
        ------------------------------------------------------------------------------
        | 因子名称              | 含义                          | 应用场景              |
        ------------------------------------------------------------------------------
        | volume_ma{N}          | N日成交量均线                 | 成交量基准            |
        | volume_ratio_{N}       | 量比（当日量/N日均量）        | 异常放量/缩量识别      |
        | price_volume_corr     | 量价相关性                    | 趋势确认              |
        | obv                   | 能量潮指标                    | 资金流向跟踪           |
        | obv_ma10              | OBV均线                       | OBV趋势判断           |
        | volume_change         | 成交量变化率                  | 量能变化              |
        | amihud                | Amihud非流动性指标            | 流动性风险度量        |
        | amihud_ma20           | 非流动性20日均值              | 流动性状态            |
        | turnover              | 成交金额                      | 市场活跃度            |
        ================================================================================
        
        【计算公式】
        - volume_ratio_{N} = volume_t / mean(volume, N)
          * >1.5 表示放量，<0.5 表示缩量
        - price_volume_corr = corr(close_pct_change, volume_pct_change, N)
          * 正相关：量价齐升，趋势健康
          * 负相关：量价背离，可能反转
        - OBV = sum(sign(close_t - close_{t-1}) * volume_t)
          * 累积资金流向，上涨日加成交量，下跌日减成交量
        - amihud = |return| / volume
          * 衡量单位成交量引起的价格变化
          * 值越大表示流动性越差
        
        【参数说明】
        - 成交量均线周期: [5, 10, 20]
        - 量价相关性窗口: 10日
        - OBV均线: 10日
        - Amihud均值: 20日
        
        【调优建议】
        - 可添加成交量加权平均价(VWAP)因子
        - 可计算成交量Z-Score：volume_zscore = (volume - mean) / std
        - 可添加资金流向指标(MFI)：考虑成交金额而非成交量
        - 对于A股，可结合换手率(换手率=成交量/流通股本)
        """
        
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
        
        # 添加模型期望的 volume_ratio（使用5日量比）
        df['volume_ratio'] = df['volume_ratio_5']

        # 对成交量特征进行对数变换，减少极端值影响
        volume_cols = ['volume_ma5', 'volume_ma10', 'volume_ma20', 'turnover', 'turnover_ma5']
        for col in volume_cols:
            if col in df.columns:
                # 对数变换: log(1 + x) 避免log(0)
                df[col] = np.log1p(df[col].clip(lower=0))

        return df
    
    def _technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        技术指标因子 - 经典技术分析指标的量化实现
        
        ================================================================================
        【因子列表】
        ------------------------------------------------------------------------------
        | 因子名称              | 含义                          | 应用场景              |
        ------------------------------------------------------------------------------
        | rsi_{N}               | 相对强弱指标                  | 超买超卖判断          |
        | macd                  | MACD快线                      | 趋势方向              |
        | macd_signal           | MACD慢线（信号线）            | 交易信号              |
        | macd_hist             | MACD柱状图                    | 趋势强度              |
        | kdj_k/d/j             | KDJ随机指标                   | 短线超买超卖          |
        | bb_mid/upper/lower    | 布林带中轨/上轨/下轨          | 波动通道              |
        | bb_position           | 价格在布林带中位置            | 突破/回归信号         |
        | bb_width              | 布林带宽度                    | 波动收缩/扩张         |
        | cci                   | 商品通道指标                  | 周期性高低点          |
        | williams_r            | 威廉指标                      | 超买超卖              |
        | adx                   | 平均趋向指标                  | 趋势强度              |
        ================================================================================
        
        【RSI - 相对强弱指标】
        ------------------------------------------------------------------------------
        【参数】N = [6, 14, 24]
        【公式】
          RSI = 100 - 100 / (1 + RS)
          RS = 平均上涨幅度 / 平均下跌幅度
          平均涨跌幅使用简单移动平均(SMA)计算
        【应用】
          - RSI > 70: 超买区，可能回调
          - RSI < 30: 超卖区，可能反弹
          - RSI = 50: 多空平衡
        【调优】
          - 短线用RSI_6，中线用RSI_14，长线用RSI_24
          - 可改用EMA替代SMA，对近期更敏感
        
        【MACD - 异同移动平均线】
        ------------------------------------------------------------------------------
        【参数】快线周期=12, 慢线周期=26, 信号线周期=9
        【公式】
          MACD = EMA(12) - EMA(26)
          Signal = EMA(MACD, 9)
          Histogram = MACD - Signal
        【应用】
          - MACD上穿Signal: 金叉，买入信号
          - MACD下穿Signal: 死叉，卖出信号
          - Histogram > 0且增大: 多头趋势加强
        【调优】
          - 短线可改为(5, 13, 4)
          - 长线可改为(19, 39, 9)
        
        【KDJ - 随机指标】
        ------------------------------------------------------------------------------
        【参数】N = 9（计算周期）, K平滑=3, D平滑=3
        【公式】
          RSV = (close - low_9) / (high_9 - low_9) * 100
          K = SMA(RSV, 3)
          D = SMA(K, 3)
          J = 3*K - 2*D
        【应用】
          - K > 80, D > 80: 超买
          - K < 20, D < 20: 超卖
          - K上穿D: 金叉
          - J > 100 或 J < 0: 极端超买/超卖
        
        【布林带 - Bollinger Bands】
        ------------------------------------------------------------------------------
        【参数】周期=20, 标准差倍数=2
        【公式】
          中轨 = MA(20)
          上轨 = 中轨 + 2 * std(20)
          下轨 = 中轨 - 2 * std(20)
          宽度 = (上轨 - 下轨) / 中轨
        【应用】
          - 价格触及上轨: 可能回调
          - 价格触及下轨: 可能反弹
          - 宽度收窄: 可能突破在即
        
        【其他指标】
        ------------------------------------------------------------------------------
        【CCI】参数: N=20, 常数=0.015
          - CCI > 100: 超买; CCI < -100: 超卖
        【Williams %R】参数: N=14
          - %R > -20: 超买; %R < -80: 超卖
        【ADX】参数: N=14
          - ADX > 25: 趋势明显; ADX < 20: 无明显趋势
        """
        
        # RSI
        for period in [6, 14, 24]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-8)
            df[f'rsi_{period}'] = (100 - 100 / (1 + rs)).clip(0, 100)

        # 添加模型期望的 rsi（使用14日RSI）
        df['rsi'] = df['rsi_14']
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # KDJ
        low_9 = df['low'].rolling(9).min()
        high_9 = df['high'].rolling(9).max()
        # 添加保护：当 high_9 == low_9 时避免除零，并将结果限制在 0-100 范围
        kdj_range = (high_9 - low_9 + 1e-8)
        df['kdj_k'] = ((df['close'] - low_9) / kdj_range * 100).clip(0, 100)
        df['kdj_d'] = df['kdj_k'].rolling(3).mean().clip(0, 100)
        df['kdj_j'] = (3 * df['kdj_k'] - 2 * df['kdj_d']).clip(-100, 200)
        
        # 布林带
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        
        # CCI (典型范围: -300 到 300，超出此范围视为异常)
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['cci'] = ((tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-8)).clip(-300, 300)
        
        # Williams %R (范围: -100 到 0)
        williams_r = (df['high'].rolling(14).max() - df['close']) / (df['high'].rolling(14).max() - df['low'].rolling(14).min() + 1e-8) * -100
        df['williams_r'] = williams_r.clip(-100, 0)
        
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
        """
        价格形态因子 - 基于K线形态特征识别市场信号
        
        ================================================================================
        【因子列表】
        ------------------------------------------------------------------------------
        | 因子名称              | 含义                          | 应用场景              |
        ------------------------------------------------------------------------------
        | body                  | K线实体大小                    | 趋势强度              |
        | upper_shadow          | 上影线长度                     | 卖压强度              |
        | lower_shadow          | 下影线长度                     | 买盘支撑              |
        | body_ratio            | 实体占K线比例                  | K线类型判断           |
        | is_bullish            | 是否阳线                       | 多空方向              |
        | gap                   | 跳空幅度                       | 缺口信号              |
        | consecutive_up/down   | 连续上涨/下跌天数              | 趋势延续/反转         |
        | is_hammer             | 锤子线形态                     | 底部反转信号          |
        | is_shooting_star      | 射击之星形态                   | 顶部反转信号          |
        | is_engulfing          | 吞没形态                       | 反转信号              |
        ================================================================================
        
        【计算公式】
        - body = |close - open|
        - upper_shadow = high - max(open, close)
        - lower_shadow = min(open, close) - low
        - body_ratio = body / (high - low)
        - is_bullish = 1 if close > open else 0
        - gap = open_t / close_{t-1} - 1
        - consecutive_up = sum(return_1d > 0, 5)
        
        【K线形态识别规则】
        ------------------------------------------------------------------------------
        【锤子线 (Hammer)】
          - 下影线 > 2倍实体
          - 上影线 < 0.5倍实体
          - 出现在下跌趋势末端，看涨反转信号
        
        【射击之星 (Shooting Star)】
          - 上影线 > 2倍实体
          - 下影线 < 0.5倍实体
          - 出现在上涨趋势末端，看跌反转信号
        
        【吞没形态 (Engulfing)】
          - 今日K线实体完全包含昨日实体
          - 阴阳属性相反
          - 强烈反转信号
        
        【调优建议】
        - 可添加更多经典形态：十字星、早晨之星、黄昏之星等
        - 可结合趋势背景判断形态有效性
        - 可添加形态确认：连续2-3根K线确认
        - 实体比例阈值可根据市场调整
        """
        
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
        """
        统计因子 - 基于统计学的价格分析和风险度量
        
        ================================================================================
        【因子列表】
        ------------------------------------------------------------------------------
        | 因子名称              | 含义                          | 应用场景              |
        ------------------------------------------------------------------------------
        | mean_reversion_{N}    | N日均值回归Z-Score            | 均值回归策略          |
        | price_percentile_{N}  | 价格在N日内的分位数           | 相对位置判断          |
        | return_autocorr_5     | 收益率自相关系数              | 动量/反转识别         |
        | max_drawdown_{N}      | N日最大回撤                   | 风险控制              |
        | dd_recovery           | 回撤恢复信号                  | 回撤结束判断          |
        | sharpe_proxy          | 20日夏普比率代理              | 风险调整收益          |
        | sharpe_60             | 60日夏普比率                  | 中期风险调整收益       |
        | tail_risk             | 尾部风险综合指标              | 极端风险度量          |
        | market_stress         | 市场压力信号                  | 危机识别              |
        | crisis_momentum       | 危机中的动量表现              | 危机Alpha策略         |
        ================================================================================
        
        【计算公式】
        - mean_reversion_{N} = (close - ma{N}) / std{N}
          * Z-Score标准化，>2表示显著高于均值，<-2表示显著低于均值
        - price_percentile_{N} = rank(close, N) / N
          * 当前价格在过去N天的百分位排名
        - return_autocorr = autocorr(return, lag=5)
          * 正自相关：动量效应；负自相关：反转效应
        - max_drawdown_{N} = max(cummax - close) / cummax over N days
        - sharpe_proxy = mean(return) / std(return) * sqrt(252)
          * 简化夏普比率，假设无风险利率为0
        
        【参数说明】
        - 均值回归窗口: [20, 60]
          * 20日：短期均值回归
          * 60日：中期均值回归
        - 自相关滞后期: 5天
        - 最大回撤窗口: [20, 60]
        
        【调优建议】
        - 可添加Hurst指数判断价格是否具有长期记忆性
        - 可添加分形维数衡量价格复杂度
        - 可添加协整关系因子进行配对交易
        - 可使用GARCH模型预测波动率
        """
        
        # 均值回归 - 多周期布林带 Z-Score
        for window in [20, 60]:
            ma = df['close'].rolling(window).mean()
            std = df['close'].rolling(window).std()
            df[f'mean_reversion_{window}'] = (df['close'] - ma) / (std + 0.01)
        
        # 价格分位数（回归信号）
        for window in [20, 60]:
            df[f'price_percentile_{window}'] = df['close'].rolling(window).rank(pct=True)
        
        # 收益自相关（动量/反转信号）
        df['return_autocorr_5'] = df['return_1d'].rolling(20).apply(
            lambda x: x.autocorr(lag=5) if len(x) > 5 else 0, raw=False
        )
        
        # 最大回撤
        cummax = df['close'].cummax()
        drawdown = (cummax - df['close']) / cummax
        df['max_drawdown_20'] = drawdown.rolling(20).max()
        df['max_drawdown_60'] = drawdown.rolling(60).max()
        
        # 回撤恢复信号
        if 'max_drawdown_20' in df.columns:
            dd = df['max_drawdown_20']
            df['dd_recovery'] = (dd < dd.shift(5)).astype(int)
        
        # 信息比率相关
        df['sharpe_proxy'] = df['return_1d'].rolling(20).mean() / (df['return_1d'].rolling(20).std() + 0.01)
        df['sharpe_60'] = df['return_1d'].rolling(60).mean() / (df['return_1d'].rolling(60).std() + 0.01)
        
        # 尾部风险综合指标
        if 'skewness_20' in df.columns and 'kurtosis_20' in df.columns:
            df['tail_risk'] = abs(df['skewness_20']) + abs(df['kurtosis_20']) / 10
        
        # 危机 Alpha 相关因子
        # 市场压力检测（如果市场数据存在）
        if 'market_return' in df.columns:
            market_vol = df['market_return'].rolling(60).std()
            df['market_stress'] = (df['market_return'] < -2 * market_vol).astype(int)
            
            # 长周期动量在危机中的表现代理
            if 'tsmom_lb120' in df.columns:
                df['crisis_momentum'] = df['tsmom_lb120'] * df['market_stress']
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        时间特征 - 捕捉日历效应和季节性规律
        
        ================================================================================
        【因子列表】
        ------------------------------------------------------------------------------
        | 因子名称              | 含义                          | 应用场景              |
        ------------------------------------------------------------------------------
        | day_of_week           | 星期几 (0=周一, 4=周五)       | 周内效应              |
        | month                 | 月份 (1-12)                   | 季节性效应            |
        | quarter               | 季度 (1-4)                    | 季度效应              |
        | is_month_start        | 是否月初                      | 月初效应              |
        | is_month_end          | 是否月末                      | 月末效应              |
        | is_day_{N}            | 星期N的哑变量                 | 周几效应分离          |
        ================================================================================
        
        【常见日历效应】
        - 周一效应：周一往往表现较差
        - 月末效应：月末流动性收紧，波动增加
        - 季度末效应：机构调仓，风格切换
        - 1月效应：1月往往表现较好
        
        【调优建议】
        - 可添加节假日相关特征
        - 可添加财报发布周期特征
        - 可添加期权到期日特征
        """
        
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
        """
        市场特征 - 计算市场整体指标和个股相对强度
        
        ================================================================================
        【因子列表】
        ------------------------------------------------------------------------------
        | 因子名称              | 含义                          | 应用场景              |
        ------------------------------------------------------------------------------
        | market_return         | 市场平均收益率                | 市场基准              |
        | market_volatility     | 市场波动率                    | 市场情绪              |
        | advance_ratio         | 上涨股票比例                  | 市场宽度              |
        | relative_return       | 相对市场超额收益              | 选股Alpha             |
        | market_strength       | 市场强度指标                  | 市场情绪判断          |
        ================================================================================
        
        【计算公式】
        - market_return = mean(个股收益率)
        - market_volatility = std(个股收益率)
        - advance_ratio = 上涨股票数 / 总股票数
        - relative_return = 个股收益 - 市场收益
        - market_strength = advance_ratio * 2 - 1
          * 范围[-1, 1]，>0表示多头市场，<0表示空头市场
        
        【注意事项】
        - 需要有多只股票数据才能计算市场特征
        - 股票数 < 5 时，市场特征设为0
        - 建议使用沪深300或全市场数据作为基准
        """
        
        if 'date' in df.columns and 'code' in df.columns:
            n_stocks = df['code'].nunique()
            if n_stocks >= 5:
                market_stats = df.groupby('date').agg({
                    'return_1d': ['mean', 'std', lambda x: (x > 0).sum() / len(x)]
                }).reset_index()
                market_stats.columns = ['date', 'market_return', 'market_volatility', 'advance_ratio']
                
                df = df.merge(market_stats, on='date', how='left')
                
                df['relative_return'] = df['return_1d'] - df['market_return']
                df['market_strength'] = df['advance_ratio'] * 2 - 1
        
        for col in ['market_return', 'market_volatility', 'advance_ratio', 'relative_return', 'market_strength']:
            if col not in df.columns:
                df[col] = 0.0
        
        return df
    
    def add_index_factors(self, df: pd.DataFrame, index_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        添加大盘指数因子 - 引入市场基准信息增强因子有效性
        
        ================================================================================
        【因子列表】
        ------------------------------------------------------------------------------
        | 因子名称              | 含义                          | 应用场景              |
        ------------------------------------------------------------------------------
        | index_close           | 指数收盘价                    | 市场基准              |
        | index_return_{N}d     | 指数N日收益率                | 市场趋势              |
        | index_ma_position     | 指数相对均线位置             | 市场趋势判断          |
        | index_volatility      | 指数波动率                    | 市场风险              |
        | relative_strength_{N}d| 相对指数的超额收益           | 选股Alpha             |
        | beta                  | 个股Beta系数                 | 系统性风险度量        |
        | combo_vol_rs          | 低波动+高相对强度组合        | 选股增强              |
        | combo_index_dd        | 指数低位+个股强势组合        | 抄底策略              |
        | market_regime         | 市场状态分类                 | 状态依赖策略          |
        ================================================================================
        
        【计算公式】
        - relative_strength_{N}d = 股票收益率 - 指数收益率
        - beta = cov(stock_return, index_return) / var(index_return)
          * beta > 1: 股票波动大于市场
          * beta < 1: 股票波动小于市场
        - combo_vol_rs = 1 when (index_vol < 30分位) & (relative_strength > 70分位)
        - combo_index_dd = 1 when (index_close < 30分位) & (drawdown > 70分位)
        - market_regime = 分位数分段(index_volatility, 5)
        
        【参数说明】
        - index_df: 指数数据，需包含 date, close 列
          * 推荐使用沪深300、中证500或上证指数
        - 相对强度计算周期: [1d, 5d, 20d]
        - beta计算窗口: 20日
        
        【调优建议】
        - 可添加行业指数因子进行行业中性化
        - 可添加风格指数（如成长/价值）因子
        - 可添加VIX类恐慌指标
        - beta可使用不同时间窗口计算
        """
        
        if index_df is None or len(index_df) == 0:
            # 没有指数数据，使用已有市场特征
            return df
        
        # 确保日期格式一致
        df['date'] = pd.to_datetime(df['date'])
        index_df = index_df.copy()
        index_df['date'] = pd.to_datetime(index_df['date'])
        
        # 计算指数因子
        index_df['index_close'] = index_df['close']
        index_df['index_return_1d'] = index_df['close'].pct_change(1)
        index_df['index_return_5d'] = index_df['close'].pct_change(5)
        index_df['index_return_20d'] = index_df['close'].pct_change(20)
        
        index_df['index_ma5'] = index_df['close'].rolling(5).mean()
        index_df['index_ma10'] = index_df['close'].rolling(10).mean()
        index_df['index_ma20'] = index_df['close'].rolling(20).mean()
        
        index_df['index_ma_position'] = index_df['close'] / index_df['index_ma20'] - 1
        index_df['index_volatility'] = index_df['close'].pct_change().rolling(20).std()
        
        # 选择需要合并的列（添加 index_close）
        index_cols = ['date', 'index_close', 'index_return_1d', 'index_return_5d', 'index_return_20d',
                      'index_ma_position', 'index_volatility']
        
        index_features = index_df[index_cols].copy()
        
        # 合并到个股数据
        df = df.merge(index_features, on='date', how='left')
        
        # 计算相对强度因子（添加 20d）
        if 'return_1d' in df.columns and 'index_return_1d' in df.columns:
            df['relative_strength_1d'] = df['return_1d'] - df['index_return_1d']
        if 'return_5d' in df.columns and 'index_return_5d' in df.columns:
            df['relative_strength_5d'] = df['return_5d'] - df['index_return_5d']
        if 'return_20d' in df.columns and 'index_return_20d' in df.columns:
            df['relative_strength_20d'] = df['return_20d'] - df['index_return_20d']
        
        # 计算 beta（个股相对于指数的波动率比率）
        if 'volatility_20' in df.columns and 'index_volatility' in df.columns:
            df['beta'] = df['volatility_20'] / (df['index_volatility'] * np.sqrt(252) + 1e-8)
        elif 'return_1d' in df.columns and 'index_return_1d' in df.columns:
            window = 20
            stock_var = df['return_1d'].rolling(window).var()
            index_var = df['index_return_1d'].rolling(window).var()
            cov = df['return_1d'].rolling(window).cov(df['index_return_1d'])
            df['beta'] = cov / (index_var + 1e-8)
        
        df['combo_vol_rs'] = 0
        if 'index_volatility' in df.columns and 'relative_strength_20d' in df.columns:
            vol_low = df['index_volatility'] < df['index_volatility'].rolling(60).quantile(0.3)
            rs_high = df['relative_strength_20d'] > df['relative_strength_20d'].rolling(60).quantile(0.7)
            df['combo_vol_rs'] = (vol_low & rs_high).astype(int)
        
        df['combo_index_dd'] = 0
        if 'index_close' in df.columns and 'max_drawdown_20' in df.columns:
            index_low = df['index_close'] < df['index_close'].rolling(60).quantile(0.3)
            dd_small = df['max_drawdown_20'] > df['max_drawdown_20'].rolling(60).quantile(0.7)
            df['combo_index_dd'] = (index_low & dd_small).astype(int)
        
        if 'index_volatility' in df.columns:
            df['market_regime'] = pd.cut(df['index_volatility'], bins=5, labels=[0,1,2,3,4]).astype(float)
        
        return df
        
        # 确保日期格式一致
        df['date'] = pd.to_datetime(df['date'])
        index_df = index_df.copy()
        index_df['date'] = pd.to_datetime(index_df['date'])
        
        # 计算指数因子
        index_df['index_close'] = index_df['close']
        index_df['index_return_1d'] = index_df['close'].pct_change(1)
        index_df['index_return_5d'] = index_df['close'].pct_change(5)
        index_df['index_return_20d'] = index_df['close'].pct_change(20)
        
        index_df['index_ma5'] = index_df['close'].rolling(5).mean()
        index_df['index_ma10'] = index_df['close'].rolling(10).mean()
        index_df['index_ma20'] = index_df['close'].rolling(20).mean()
        
        index_df['index_ma_position'] = index_df['close'] / index_df['index_ma20'] - 1
        index_df['index_volatility'] = index_df['close'].pct_change().rolling(20).std()
        
        # 选择需要合并的列（添加 index_close）
        index_cols = ['date', 'index_close', 'index_return_1d', 'index_return_5d', 'index_return_20d',
                      'index_ma_position', 'index_volatility']
        
        index_features = index_df[index_cols].copy()
        
        # 合并到个股数据
        df = df.merge(index_features, on='date', how='left')
        
        # 计算相对强度因子（添加 20d）
        if 'return_1d' in df.columns and 'index_return_1d' in df.columns:
            df['relative_strength_1d'] = df['return_1d'] - df['index_return_1d']
        if 'return_5d' in df.columns and 'index_return_5d' in df.columns:
            df['relative_strength_5d'] = df['return_5d'] - df['index_return_5d']
        if 'return_20d' in df.columns and 'index_return_20d' in df.columns:
            df['relative_strength_20d'] = df['return_20d'] - df['index_return_20d']
        
        # 计算 beta（个股相对于指数的波动率比率）
        if 'volatility_20' in df.columns and 'index_volatility' in df.columns:
            df['beta'] = df['volatility_20'] / (df['index_volatility'] * np.sqrt(252) + 1e-8)
        elif 'return_1d' in df.columns and 'index_return_1d' in df.columns:
            window = 20
            stock_var = df['return_1d'].rolling(window).var()
            index_var = df['index_return_1d'].rolling(window).var()
            cov = df['return_1d'].rolling(window).cov(df['index_return_1d'])
            df['beta'] = cov / (index_var + 1e-8)
        
        df['combo_vol_rs'] = 0
        if 'index_volatility' in df.columns and 'relative_strength_20d' in df.columns:
            vol_low = df['index_volatility'] < df['index_volatility'].rolling(60).quantile(0.3)
            rs_high = df['relative_strength_20d'] > df['relative_strength_20d'].rolling(60).quantile(0.7)
            df['combo_vol_rs'] = (vol_low & rs_high).astype(int)
        
        df['combo_index_dd'] = 0
        if 'index_close' in df.columns and 'max_drawdown_20' in df.columns:
            index_low = df['index_close'] < df['index_close'].rolling(60).quantile(0.3)
            dd_small = df['max_drawdown_20'] > df['max_drawdown_20'].rolling(60).quantile(0.7)
            df['combo_index_dd'] = (index_low & dd_small).astype(int)
        
        if 'index_volatility' in df.columns:
            df['market_regime'] = pd.cut(df['index_volatility'], bins=5, labels=[0,1,2,3,4]).astype(float)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """获取所有特征名称"""
        return self.feature_names
    
    def select_features(self, df: pd.DataFrame, method: str = 'importance') -> List[str]:
        """
        特征选择 - 筛选有效因子，提高模型性能
        
        ================================================================================
        【参数说明】
        - df: 包含所有因子的DataFrame
        - method: 特征选择方法
          * 'importance': 基于模型重要性（需配合训练使用）
          * 'correlation': 基于相关性筛选
          * 'shap': 基于SHAP值解释性
        
        【筛选规则】
        1. 排除非特征列：code, date, open, high, low, close, volume, future_return, target
        2. 移除NaN过多的特征（有效值比例 < 70%）
        3. 根据method进行进一步筛选
        
        【调优建议】
        - 可添加因子IC值筛选（IC > 0.02 或 IC < -0.02）
        - 可添加因子单调性检验
        - 可移除高相关性因子（相关系数 > 0.7）
        - 可使用PCA/LDA进行降维
        ================================================================================
        
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
    """
    特征工程测试函数 - 演示因子生成流程
    
    ================================================================================
    【使用示例】
    
    # 1. 基本使用
    fe = FeatureEngineer()
    df_features = fe.generate_all_features(df)
    
    # 2. 添加指数因子
    df_with_index = fe.add_index_factors(df_features, index_df)
    
    # 3. 特征选择
    features = fe.select_features(df_with_index)
    
    # 4. 获取有效特征列表
    print(f"生成特征数: {len(features)}")
    
    【注意事项】
    - 输入数据需包含：code, date, open, high, low, close, volume
    - 因子计算会产生NaN，需在模型训练前处理
    - 建议使用至少3年历史数据以确保因子稳定性
    ================================================================================
    """
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