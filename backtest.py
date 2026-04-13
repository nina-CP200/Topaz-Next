#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topaz 回测系统
使用过去2年数据模拟每日交易决策
输出：Sharpe比率、最大回撤、年化收益（简洁版）
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engineer import FeatureEngineer


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self,
                 initial_capital: float = 1000000,
                 lookback_days: int = 60,
                 rebalance_days: int = 20,
                 buy_threshold: float = 0.52,
                 sell_threshold: float = 0.35,
                 max_positions: int = 12):
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
        """加载模型"""
        model_data = joblib.load(model_path)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_cols = model_data['feature_cols']
        print(f"✅ 模型已加载: {model_path}")
    
    def prepare_data(self, data_path: str):
        """准备数据"""
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(['code', 'date']).reset_index(drop=True)
        
        self.all_dates = sorted(self.df['date'].unique())
        print(f"✅ 数据已加载: {len(self.df):,} 条记录, {len(self.all_dates)} 个交易日")
    
    def run(self):
        """执行回测"""
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
        """执行交易"""
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
        """更新持仓市值"""
        for code in self.holdings:
            stock_data = self.df[(self.df['code'] == code) & (self.df['date'] == date)]
            if len(stock_data) > 0:
                self.holdings[code]['current_price'] = stock_data.iloc[0]['close']
    
    def _record_daily_value(self, date):
        """记录每日净值"""
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
        """计算回测指标"""
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
    """主函数"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(base_dir, 'ensemble_model_csi300_latest.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, 'ensemble_model_csi300_2y.pkl')
    
    data_path = os.path.join(base_dir, 'csi300_raw_data_2y.csv')
    
    if not os.path.exists(model_path):
        print(f"❌ 未找到模型文件")
        return
    
    if not os.path.exists(data_path):
        print(f"❌ 未找到数据文件")
        return
    
    engine = BacktestEngine(
        initial_capital=1000000,
        lookback_days=60,
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