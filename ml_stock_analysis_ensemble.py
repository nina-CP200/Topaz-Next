#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantPilot 机器学习股票分析系统
使用预训练集成模型进行股票收益预测

模型：Ensemble (LightGBM + RF + GBDT)
验证准确率：60.19%, AUC: 0.647
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

# 启动时检查虚拟环境
def check_and_activate_venv():
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return True
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    home_dir = os.path.expanduser('~')
    
    venv_candidates = [
        os.path.join(home_dir, 'myenv'),
        os.path.join(home_dir, 'venv'),
    ]
    
    for venv_path in venv_candidates:
        python_path = os.path.join(venv_path, 'bin', 'python')
        if os.path.exists(python_path):
            print(f"🔄 检测到虚拟环境：{venv_path}")
            os.execv(python_path, [python_path] + sys.argv)
    
    return False

check_and_activate_venv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantpilot_data_api import get_stock_data, get_history_data, get_macro_indicators
from utils import parse_stock_list
from ensemble_model import EnsembleModel
from feature_engineer import FeatureEngineer


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """创建技术特征"""
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
    data['macd_hist'] = data['macd'] - data['macd_signal']
    
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
    data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'] + 0.001)
    
    # 成交量特征
    data['volume_ma5'] = data['volume'].rolling(5).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma5']
    
    # 价格动量
    data['momentum_5d'] = data['close'].pct_change(5)
    data['momentum_10d'] = data['close'].pct_change(10)
    data['momentum_20d'] = data['close'].pct_change(20)
    
    # 波动率
    data['volatility_5d'] = data['close'].pct_change().rolling(5).std()
    data['volatility_10d'] = data['close'].pct_change().rolling(10).std()
    data['volatility_20d'] = data['close'].pct_change().rolling(20).std()
    
    # 价格位置
    data['high_low_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 0.001)
    
    # 收益率
    data['returns'] = data['close'].pct_change()
    
    return data


class MLStockAnalyzer:
    """基于预训练集成模型的股票分析器"""
    
    def __init__(self, history_days: int = 60, batch: int = 0, limit: int = 0):
        self.history_days = history_days
        self.batch = batch
        self.limit = limit
        
        # 加载预训练模型
        print("🤖 加载预训练集成模型...")
        self.predictor = EnsembleModel(model_dir=os.path.dirname(os.path.abspath(__file__)))
        self.feature_engineer = FeatureEngineer()
        
        self.history_data = {}
        self.current_data = {}
        self.results = []
    
    def fetch_history_data(self, stocks: List) -> bool:
        """获取股票历史数据"""
        print(f"📈 获取历史数据（过去 {self.history_days} 天）...")
        
        success_count = 0
        for stock in stocks:
            symbol = stock[0]
            name = stock[1]
            market = stock[2] if len(stock) > 2 else 'A股'
            
            print(f"  获取 {symbol} ({name}) 历史数据...", end=' ')
            data = get_history_data(symbol, market, self.history_days)
            if data is not None and len(data) > 0:
                self.history_data[symbol] = data
                print(f"✓ {len(data)} 条")
                success_count += 1
            else:
                print("✗")
        
        print(f"\n  汇总：成功 {success_count}/{len(stocks)} 只")
        return success_count > 0
    
    def fetch_current_data(self, stocks: List) -> bool:
        """获取股票实时数据"""
        print("\n📊 获取股票实时数据...")
        
        success_count = 0
        for stock in stocks:
            symbol = stock[0]
            name = stock[1]
            market = stock[2] if len(stock) > 2 else 'A股'
            
            print(f"  获取 {symbol} ({name}) 实时数据...", end=' ')
            data = get_stock_data(symbol, market, name)
            if data:
                self.current_data[symbol] = data
                print(f"✓")
                success_count += 1
            else:
                print("✗")
        
        print(f"\n  成功获取 {success_count}/{len(stocks)} 只股票的实时数据")
        return success_count > 0
    
    def analyze_stock(self, symbol: str, current: Dict, history: pd.DataFrame) -> Dict:
        """分析单只股票"""
        # 使用 FeatureEngineer 生成特征
        if 'code' not in history.columns:
            history = history.copy()
            history['code'] = symbol
        
        df_features = self.feature_engineer.generate_all_features(history)
        
        # 获取最新特征
        df_features = df_features.fillna(0)
        if len(df_features) < 2:
            return None
        
        # 使用模型的特征列
        feature_cols = self.predictor.feature_cols
        latest = df_features.iloc[-1:][feature_cols]
        
        # 填充 NaN
        if latest.isna().any().any():
            latest = latest.fillna(0)
        
        # 构建特征向量
        X = latest.values
        
        # 模型预测
        pred = self.predictor.predict(X)
        
        if 'error' in pred:
            return None
        
        proba = pred['probability'][0]
        prediction = pred['prediction'][0]
        
        # 计算预期收益
        expected_return = (proba - 0.5) * 10  # 转换为百分比收益
        
        # 风险等级
        if proba >= 0.6:
            risk_level = '低风险'
        elif proba >= 0.5:
            risk_level = '中风险'
        elif proba >= 0.4:
            risk_level = '高风险'
        else:
            risk_level = '极高风险'
        
        # 投资建议
        if proba >= 0.6:
            advice = '建议买入'
        elif proba >= 0.5:
            advice = '建议持有'
        elif proba >= 0.4:
            advice = '建议观望'
        else:
            advice = '建议回避'
        
        return {
            'symbol': symbol,
            'name': current.get('name', symbol),
            'market': 'A股',
            'current_price': current.get('current_price', 0),
            'change_pct': current.get('change', 0),
            'pe_ratio': current.get('pe_ratio', 0),
            'pb_ratio': current.get('pb_ratio', 0),
            'roe': current.get('roe', 0),
            'predicted_return': expected_return,
            'probability': proba,
            'risk_level': risk_level,
            'advice': advice
        }
    
    def run(self, us_stocks_file: str = None, a_stocks_file: str = None):
        """运行分析"""
        # 读取股票列表
        stocks = []
        
        if a_stocks_file and os.path.exists(a_stocks_file):
            print(f"\n📂 读取 A股列表...")
            a_stocks = parse_stock_list(a_stocks_file)
            stocks.extend([(s[0], s[1], 'A股') for s in a_stocks])
            print(f"  读取 {len(a_stocks)} 只 A股")
        
        if us_stocks_file and os.path.exists(us_stocks_file):
            print(f"\n📂 读取美股列表...")
            us_stocks = parse_stock_list(us_stocks_file)
            stocks.extend([(s[0], s[1], '美股') for s in us_stocks])
            print(f"  读取 {len(us_stocks)} 只美股")
        
        if not stocks:
            print("❌ 未找到股票列表")
            return
        
        # 限制数量
        if self.limit > 0:
            stocks = stocks[:self.limit]
        
        # 获取历史数据
        if not self.fetch_history_data(stocks):
            print("❌ 获取历史数据失败")
            return
        
        # 获取实时数据
        if not self.fetch_current_data(stocks):
            print("❌ 获取实时数据失败")
            return
        
        # 分析每只股票
        print("\n📈 进行预测分析...")
        for symbol, name, market in stocks:
            if symbol in self.current_data and symbol in self.history_data:
                result = self.analyze_stock(
                    symbol,
                    self.current_data[symbol],
                    self.history_data[symbol]
                )
                if result:
                    self.results.append(result)
                    print(f"  完成 {symbol} 分析")
        
        print(f"\n  完成 {len(self.results)} 只股票分析")
    
    def print_results(self):
        """打印结果"""
        print("\n" + "=" * 80)
        print("分析结果")
        print("=" * 80)
        
        for r in self.results:
            market_label = f"({r['market']})"
            print(f"\n[ML] 【{r['symbol']}】{r['name']} {market_label}")
            print(f"  当前价格：¥{r['current_price']:.2f}  ({r['change_pct']:+.2f}%)")
            if r.get('pe_ratio', 0) > 0:
                print(f"  市盈率 (PE): {r['pe_ratio']:.2f}")
            if r.get('pb_ratio', 0) > 0:
                print(f"  市净率 (PB): {r['pb_ratio']:.2f}")
            if r.get('roe', 0) > 0:
                print(f"  净资产收益率 (ROE): {r['roe']:.2f}%")
            print(f"  预测收益：{r['predicted_return']:.1f}%")
            print(f"  概率：{r['probability']:.2%}")
            print(f"  风险等级：{r['risk_level']}")
            print(f"  投资建议：{r['advice']}")
        
        # 汇总统计
        self._print_summary()
    
    def _print_summary(self):
        """打印汇总统计"""
        print("\n" + "=" * 80)
        print("汇总统计")
        print("=" * 80)
        
        for market in ['美股', 'A股']:
            market_results = [r for r in self.results if r['market'] == market]
            if not market_results:
                continue
            
            print(f"\n【{market}】共 {len(market_results)} 只")
            
            buy = sum(1 for r in market_results if '买入' in r['advice'])
            hold = sum(1 for r in market_results if '持有' in r['advice'] or '观望' in r['advice'])
            avoid = sum(1 for r in market_results if '回避' in r['advice'])
            
            print(f"  建议买入：{buy} 只")
            print(f"  建议持有/观望：{hold} 只")
            print(f"  建议回避：{avoid} 只")
            
            low_risk = sum(1 for r in market_results if r['risk_level'] == '低风险')
            mid_risk = sum(1 for r in market_results if r['risk_level'] == '中风险')
            high_risk = sum(1 for r in market_results if r['risk_level'] in ['高风险', '极高风险'])
            
            print(f"  风险分布：低{low_risk}/中{mid_risk}/高{high_risk}")
        
        print()
        print("=" * 80)
        print("风险提示：本分析仅供参考，不构成投资建议。市场有风险，投资需谨慎。")
        print("=" * 80)


def main(batch=0, limit=0, market='all'):
    """主函数"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 股票列表在当前目录 - 使用 os.listdir 查找
    files = os.listdir(base_dir)
    us_stocks_file = None
    a_stocks_file = None
    
    for f in files:
        if '美股关注' in f and f.endswith('.md'):
            us_stocks_file = os.path.join(base_dir, f)
        if 'A股' in f and f.endswith('.md'):
            a_stocks_file = os.path.join(base_dir, f)
    
    if market == 'cn':
        us_stocks_file = None
        print("📊 市场模式：仅 A股")
    elif market == 'us':
        a_stocks_file = None
        print("📊 市场模式：仅美股")
    else:
        print("📊 市场模式：全部市场")
    
    # 创建分析器（使用预训练模型）
    analyzer = MLStockAnalyzer(history_days=60, batch=batch, limit=limit)
    
    # 运行分析
    analyzer.run(us_stocks_file, a_stocks_file)
    
    # 打印结果
    analyzer.print_results()


def parse_args():
    """解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(description='QuantPilot 股票分析系统 (预训练集成模型)')
    parser.add_argument('--batch', type=int, default=0, help='批量编号 (1,2,3,4)')
    parser.add_argument('--limit', type=int, default=0, help='每批数量')
    parser.add_argument('--cn', action='store_true', help='只分析 A股市场')
    parser.add_argument('--us', action='store_true', help='只分析美股市场')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if args.cn and args.us:
        print("⚠️ 不能同时指定 --cn 和 --us，将分析全部市场")
        market = 'all'
    elif args.cn:
        market = 'cn'
    elif args.us:
        market = 'us'
    else:
        market = 'all'
    
    main(batch=args.batch, limit=args.limit, market=market)
