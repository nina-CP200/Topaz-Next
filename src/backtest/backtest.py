#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Topaz 回测系统 (Backtest System)
================================================================================

模块说明:
---------
本模块实现了完整的股票量化策略回测框架，用于评估机器学习选股策略的历史表现。
通过模拟历史交易过程，计算策略的风险收益指标，帮助投资者了解策略的实际效果。

回测框架说明:
-----------
1. 数据准备:
   - 加载历史股票数据（包含价格、成交量等信息）
   - 加载预训练的机器学习模型（随机森林、XGBoost、LightGBM等集成模型）

2. 特征工程:
   - 使用 FeatureEngineer 类生成技术指标特征
   - 包括动量指标、RSI、移动平均线等技术分析指标

3. 交易模拟:
   - 按时间顺序遍历历史数据
   - 在每个调仓日生成买卖信号并执行交易
   - 记录每日持仓市值和现金余额

4. 绩效评估:
   - 计算年化收益率、Sharpe比率、最大回撤等关键指标
   - 输出交易记录供后续分析

回测注意事项:
-----------
1. 数据质量:
   - 确保数据来源可靠，避免未来函数（使用未来数据）
   - 注意处理停牌、涨跌停、分红除权等特殊情况
   - 数据应包含足够长的历史周期（建议至少2年）

2. 过拟合风险:
   - 策略在历史数据上表现好不代表未来表现好
   - 建议使用样本外数据进行验证
   - 参数不宜过多，避免过度优化

3. 交易成本:
   - 实际交易中需考虑手续费、滑点、冲击成本
   - 本回测框架简化了交易成本的计算

4. 市场环境:
   - 回测结果受市场环境影响较大
   - 不同市场环境下策略表现可能差异显著
   - 建议在多种市场环境下测试策略稳健性

输出指标:
--------
- Sharpe比率: 风险调整后收益
- 最大回撤: 从峰值到谷底的最大跌幅
- 年化收益: 按年计算的收益率
- 总交易次数: 策略执行期间的交易总数
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

from src.features.engineer import FeatureEngineer


class BacktestEngine:
    """
    回测引擎类
    
    该类负责执行完整的策略回测流程，包括：
    - 加载预训练的机器学习模型
    - 加载和处理历史股票数据
    - 按时间顺序模拟交易决策
    - 计算和输出策略绩效指标
    
    使用示例:
    --------
    >>> engine = BacktestEngine(initial_capital=1000000)
    >>> engine.load_model('model.pkl')
    >>> engine.prepare_data('data.csv')
    >>> engine.run()
    
    注意事项:
    --------
    - 确保数据文件包含完整的日期序列
    - 模型文件应包含 models, scaler, feature_cols 三个组件
    - 回测前请检查数据是否存在缺失值或异常值
    """
    
    def __init__(self,
                 initial_capital: float = 1000000,
                 lookback_days: int = 60,
                 rebalance_days: int = 20,
                 buy_threshold: float = 0.52,
                 sell_threshold: float = 0.35,
                 max_positions: int = 12):
        """
        初始化回测引擎
        
        参数说明:
        --------
        initial_capital : float
            初始资金（默认100万）
            建议根据实际投资规模设置，影响仓位管理
            
        lookback_days : int
            特征计算回溯天数（默认300天）
            用于计算技术指标所需的历史数据窗口
            较长的回溯期可计算更稳定的技术指标，但会减少有效回测天数
            
        rebalance_days : int
            调仓间隔天数（默认20天，约1个月）
            决定策略多久重新评估持仓并调整
            较短的间隔可更及时捕捉市场变化，但增加交易成本
            
        buy_threshold : float
            买入概率阈值（默认0.52）
            当股票上涨概率超过此阈值时考虑买入
            较高的阈值会减少交易频率，提高信号质量
            
        sell_threshold : float
            卖出概率阈值（默认0.35）
            当持仓股票上涨概率低于此阈值时考虑卖出
            较低的阈值会减少卖出频率，给予股票更多恢复时间
            
        max_positions : int
            最大持仓数量（默认12只）
            限制同时持有的股票数量，分散风险
            过多持仓可能分散收益，过少则风险集中
        """
        self.initial_capital = initial_capital
        self.lookback_days = lookback_days
        self.rebalance_days = rebalance_days
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.max_positions = max_positions
        
        self.cash = initial_capital
        self.holdings = {}
        self.daily_values = []
        self.trades = []
        
        self.models = None
        self.scaler = None
        self.feature_cols = None
        self.df = None
        self.all_dates = None
    
    def load_model(self, model_path: str):
        """
        加载预训练的机器学习模型
        
        参数说明:
        --------
        model_path : str
            模型文件路径（.pkl格式）
            
        模型文件结构:
        ------------
        模型文件应包含以下组件：
        - models: dict, 集成学习模型字典（如随机森林、XGBoost、LightGBM）
        - scaler: StandardScaler, 特征标准化器
        - feature_cols: list, 特征列名列表
        
        注意事项:
        --------
        - 模型应使用与回测数据相同的特征工程方法训练
        - 加载模型前请确保模型文件存在
        """
        model_data = joblib.load(model_path)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_cols = model_data['feature_cols']
        print(f"✅ 模型已加载: {model_path}")
    
    def prepare_data(self, data_path: str):
        """
        准备回测数据
        
        参数说明:
        --------
        data_path : str
            股票数据文件路径（CSV格式）
            
        数据格式要求:
        ------------
        CSV文件应包含以下列：
        - date: 交易日期
        - code: 股票代码
        - open: 开盘价
        - high: 最高价
        - low: 最低价
        - close: 收盘价
        - volume: 成交量
        
        数据质量检查:
        ------------
        - 确保数据按日期和股票代码排序
        - 检查是否存在缺失值和异常值
        - 验证日期连续性（处理停牌日）
        """
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(['code', 'date']).reset_index(drop=True)
        
        self.all_dates = sorted(self.df['date'].unique())
        print(f"✅ 数据已加载: {len(self.df):,} 条记录, {len(self.all_dates)} 个交易日")
    
    def run(self):
        """
        执行回测主流程
        
        回测逻辑说明:
        ------------
        1. 时间遍历:
           - 从第 lookback_days 天开始（确保有足够历史数据计算特征）
           - 到倒数第5天结束（预留数据避免边界问题）
           
        2. 调仓逻辑:
           - 每 rebalance_days 天执行一次调仓
           - 非调仓日仅更新持仓市值
           
        3. 股票筛选:
           - 对当日所有股票生成技术特征
           - 使用机器学习模型预测上涨概率
           - 结合动量指标计算综合评分
           
        4. 交易执行:
           - 卖出: 持仓股票评分低于卖出阈值
           - 买入: 候选股票评分高于买入阈值
           - 仓位管理: 根据评分高低分配不同仓位
           
        特征生成:
        --------
        使用 FeatureEngineer 类生成以下类型特征：
        - 价格动量: 5日、20日收益率
        - 技术指标: RSI、MACD、布林带等
        - 成交量指标: 成交量变化率等
        
        综合评分计算:
        ------------
        combined_score = ml_proba * 0.6 + (0.5 + momentum_score) * 0.4
        
        其中:
        - ml_proba: 机器学习模型预测上涨概率
        - momentum_score: 动量指标调整分数（基于5日/20日动量和RSI）
        
        动量评分规则:
        - 5日动量: >3% +0.15, >0 +0.05, <-3% -0.10
        - 20日动量: >10% +0.20, >5% +0.10, <-10% -0.15
        - RSI: <30 +0.10（超卖）, >70 -0.10（超买）
        """
        if self.models is None or self.df is None:
            print("❌ 请先加载模型和数据")
            return
        
        print("\n" + "=" * 60)
        print("🚀 开始回测")
        print("=" * 60)
        
        start_idx = self.lookback_days
        fe = FeatureEngineer()
        
        for i in range(start_idx, len(self.all_dates) - 5):
            current_date = self.all_dates[i]
            
            if (i - start_idx) % self.rebalance_days != 0:
                self._update_holdings_value(current_date)
                self._record_daily_value(current_date)
                continue
            
            if (i - start_idx) % 100 == 0:
                print(f"  处理: {current_date.strftime('%Y-%m-%d')} ({i}/{len(self.all_dates)})")
            
            history_start = self.all_dates[i - self.lookback_days]
            
            stocks_on_date = self.df[self.df['date'] == current_date]['code'].unique()
            predictions = []
            
            for code in stocks_on_date[:80]:
                stock_history = self.df[(self.df['code'] == code) & 
                                       (self.df['date'] >= history_start) & 
                                       (self.df['date'] <= current_date)].copy()
                
                if len(stock_history) < 30:
                    continue
                
                stock_history['code'] = code
                features = fe.generate_all_features(stock_history)
                features = features.fillna(0)
                
                latest = features.iloc[-1:]
                try:
                    X = latest[self.feature_cols].values
                    X_scaled = self.scaler.transform(X)
                    
                    probas = []
                    for name, model in self.models.items():
                        proba = model.predict_proba(X_scaled)[:, 1][0]
                        probas.append(proba)
                    
                    ml_proba = np.mean(probas)
                    
                    momentum_5d = latest['return_5d'].values[0] if 'return_5d' in latest.columns else 0
                    momentum_20d = latest['return_20d'].values[0] if 'return_20d' in latest.columns else 0
                    rsi = latest['rsi_14'].values[0] if 'rsi_14' in latest.columns else 50
                    
                    momentum_score = 0
                    if momentum_5d > 0.03:
                        momentum_score += 0.15
                    elif momentum_5d > 0:
                        momentum_score += 0.05
                    elif momentum_5d < -0.03:
                        momentum_score -= 0.10
                    
                    if momentum_20d > 0.10:
                        momentum_score += 0.20
                    elif momentum_20d > 0.05:
                        momentum_score += 0.10
                    elif momentum_20d < -0.10:
                        momentum_score -= 0.15
                    
                    if rsi < 30:
                        momentum_score += 0.10
                    elif rsi > 70:
                        momentum_score -= 0.10
                    
                    combined_score = ml_proba * 0.6 + (0.5 + momentum_score) * 0.4
                    
                    current_price = stock_history.iloc[-1]['close']
                    
                    predictions.append({
                        'code': code,
                        'probability': combined_score,
                        'ml_proba': ml_proba,
                        'momentum_score': momentum_score,
                        'price': current_price,
                        'momentum_5d': momentum_5d,
                        'momentum_20d': momentum_20d
                    })
                except Exception:
                    continue
            
            self._execute_trades(predictions, current_date)
            self._record_daily_value(current_date)
        
        self._calculate_metrics()
    
    def _execute_trades(self, predictions, date):
        """
        执行交易决策
        
        交易逻辑说明:
        ------------
        1. 卖出逻辑:
           - 遍历当前持仓
           - 若持仓股票的综合评分低于卖出阈值(sell_threshold)，则卖出
           - 卖出时按当前价格计算收益，增加现金余额
        
        2. 买入逻辑:
           - 筛选评分高于买入阈值(buy_threshold)的非持仓股票
           - 按评分从高到低排序
           - 最多买入前8只候选股票
           - 受最大持仓数(max_positions)和现金余额限制
        
        仓位管理规则:
        ------------
        根据综合评分动态调整单只股票的仓位比例：
        - 评分 > 0.65: 高信心，仓位20%
        - 评分 > 0.58: 中高信心，仓位15%
        - 评分 > 0.55: 中等信心，仓位10%
        - 评分 > 0.52 且动量为正: 低信心，仓位8%
        - 其他情况: 试探仓位5%
        
        买入数量计算:
        ------------
        - 仓位金额 = 现金余额 × 仓位比例
        - 买入股数 = int(仓位金额 / 股价 / 100) × 100
        - 股数取整到百股（符合A股交易规则）
        
        参数:
        ----
        predictions : list
            当日所有股票的预测结果列表
        date : datetime
            当前交易日期
        """
        for code in list(self.holdings.keys()):
            pred = next((p for p in predictions if p['code'] == code), None)
            if pred and pred['probability'] < self.sell_threshold:
                shares = self.holdings[code]['shares']
                price = pred['price']
                self.cash += shares * price
                self.trades.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'type': 'sell',
                    'code': code,
                    'shares': shares,
                    'price': price
                })
                del self.holdings[code]
        
        buy_candidates = [p for p in predictions 
                         if p['probability'] > self.buy_threshold 
                         and p['code'] not in self.holdings]
        buy_candidates.sort(key=lambda x: x['probability'], reverse=True)
        
        for pred in buy_candidates[:8]:
            if len(self.holdings) >= self.max_positions:
                break
            if self.cash < 20000:
                break
            
            score = pred['probability']
            mom_5d = pred.get('momentum_5d', 0)
            mom_20d = pred.get('momentum_20d', 0)
            
            if score > 0.65:
                position_pct = 0.20
            elif score > 0.58:
                position_pct = 0.15
            elif score > 0.55:
                position_pct = 0.10
            elif score > 0.52 and mom_5d > 0:
                position_pct = 0.08
            else:
                position_pct = 0.05
            
            amount = self.cash * position_pct
            shares = int(amount / pred['price'] / 100) * 100
            
            if shares > 0:
                self.holdings[pred['code']] = {
                    'shares': shares,
                    'cost': pred['price'],
                    'current_price': pred['price']
                }
                self.cash -= shares * pred['price']
                self.trades.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'type': 'buy',
                    'code': pred['code'],
                    'shares': shares,
                    'price': pred['price']
                })
    
    def _update_holdings_value(self, date):
        """
        更新持仓市值
        
        在非调仓日更新各持仓股票的当前价格，用于计算组合净值。
        
        参数:
        ----
        date : datetime
            当前交易日期
        """
        for code in self.holdings:
            stock_data = self.df[(self.df['code'] == code) & (self.df['date'] == date)]
            if len(stock_data) > 0:
                self.holdings[code]['current_price'] = stock_data.iloc[0]['close']
    
    def _record_daily_value(self, date):
        """
        记录每日净值
        
        计算并记录当日的组合总市值、现金余额、持仓市值。
        这些数据用于后续计算绩效指标。
        
        参数:
        ----
        date : datetime
            当前交易日期
        """
        holdings_value = sum(
            h['shares'] * h.get('current_price', h['cost'])
            for h in self.holdings.values()
        )
        total_value = self.cash + holdings_value
        
        self.daily_values.append({
            'date': date,
            'total_value': total_value,
            'cash': self.cash,
            'holdings_value': holdings_value
        })
    
    def _calculate_metrics(self):
        """
        计算并输出回测绩效指标
        
        性能指标说明:
        ------------
        1. Sharpe比率 (夏普比率):
           - 定义: 单位风险下的超额收益
           - 公式: Sharpe = (年化收益率 - 无风险利率) / 年化波动率
           - 本实现简化为: Sharpe = 年化收益率 / 年化波动率
           - 解读: 
             * Sharpe > 1: 优秀策略
             * Sharpe 0.5-1: 良好策略
             * Sharpe < 0.5: 策略效果有限
           - 注意: 高Sharpe不一定代表高收益，可能只是波动率低
        
        2. 最大回撤 (Maximum Drawdown):
           - 定义: 从历史最高点到后续最低点的最大跌幅
           - 公式: MaxDD = min((净值 - 历史最高净值) / 历史最高净值)
           - 解读:
             * MaxDD < 10%: 风险控制优秀
             * MaxDD 10-20%: 风险可控
             * MaxDD 20-30%: 中等风险
             * MaxDD > 30%: 高风险策略
           - 注意: 最大回撤反映了策略在最坏情况下的损失程度
        
        3. 年化收益率:
           - 定义: 将回测期间收益率折算为年化收益率
           - 公式: Annual_Return = 总收益率 × (252 / 回测天数)
           - 其中252为A股年均交易日数
           - 解读:
             * 年化收益 > 15%: 优秀
             * 年化收益 8-15%: 良好
             * 年化收益 < 8%: 一般
           - 注意: 需结合风险指标综合评估
        
        4. 总交易次数:
           - 反映策略的交易频率
           - 过高可能意味着过度交易和高交易成本
           - 过低可能意味着策略不够灵活
        
        计算方法:
        --------
        1. 从 daily_values 提取每日净值序列
        2. 计算日收益率序列
        3. 基于日收益率计算Sharpe比率
        4. 计算历史峰值序列，进而计算回撤序列
        5. 计算总收益率并年化
        """
        if len(self.daily_values) < 10:
            print("⚠️ 回测数据不足")
            return
        
        values = np.array([d['total_value'] for d in self.daily_values])
        returns = np.diff(values) / values[:-1]
        
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_dd = np.min(drawdown)
        
        total_return = (values[-1] - values[0]) / values[0]
        n_days = len(values)
        annual_return = total_return * 252 / n_days
        
        print("\n" + "=" * 60)
        print("📊 回测结果")
        print("=" * 60)
        print(f"  Sharpe比率:  {sharpe:.3f}")
        print(f"  最大回撤:    {max_dd:.2%}")
        print(f"  年化收益:    {annual_return:.2%}")
        print(f"  总交易次数:  {len(self.trades)}")
        print(f"  回测天数:    {n_days}")
        print("=" * 60)


def main():
    """
    主函数 - 执行回测流程
    
    执行步骤:
    ---------
    1. 定位模型文件（优先使用最新模型）
    2. 加载沪深300成分股历史数据
    3. 初始化回测引擎（配置参数）
    4. 加载模型和数据
    5. 运行回测并输出结果
    
    默认配置说明:
    ------------
    - initial_capital: 100万初始资金
    - lookback_days: 300天特征计算窗口
    - rebalance_days: 每20个交易日调仓一次
    - buy_threshold: 0.52 买入阈值
    - sell_threshold: 0.35 卖出阈值
    - max_positions: 最多持有12只股票
    
    注意事项:
    --------
    - 确保模型文件和数据文件存在于同一目录
    - 回测结果仅供参考，不构成投资建议
    - 实盘交易前请进行充分的样本外测试
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    model_path = os.path.join(base_dir, 'data/models/ensemble_model.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, 'data/models/ensemble_model_csi300_2y.pkl')
    
    data_path = os.path.join(base_dir, 'data/raw/csi300_full_history.csv')
    
    if not os.path.exists(model_path):
        print(f"❌ 未找到模型文件")
        return
    
    if not os.path.exists(data_path):
        print(f"❌ 未找到数据文件")
        return
    
    engine = BacktestEngine(
        initial_capital=1000000,
        lookback_days=300,
        rebalance_days=20,
        buy_threshold=0.52,
        sell_threshold=0.35,
        max_positions=12
    )
    
    engine.load_model(model_path)
    engine.prepare_data(data_path)
    engine.run()


if __name__ == '__main__':
    main()