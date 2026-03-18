#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topaz-Qlib 机器学习训练系统
使用真实历史数据训练 LightGBM 模型进行股票收益预测

功能:
1. 获取多只股票的历史数据
2. 自动构建技术特征（MACD, RSI, 布林带等）
3. 使用 LightGBM 训练回归模型
4. 预测未来收益并生成投资建议
"""

import os
import sys
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple


def check_and_activate_venv():
    """
    检查并自动激活虚拟环境
    """
    # 检查是否已在虚拟环境中
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # 已在虚拟环境中
        return True
    
    # 查找虚拟环境
    script_dir = os.path.dirname(os.path.abspath(__file__))
    home_dir = os.path.expanduser('~')
    
    # 可能的虚拟环境名称和位置
    venv_candidates = [
        os.path.join(home_dir, 'myenv'),
        os.path.join(home_dir, 'venv'),
        os.path.join(home_dir, 'env'),
        os.path.join(home_dir, '.venv'),
        os.path.join(script_dir, 'myenv'),
        os.path.join(script_dir, 'venv'),
    ]
    
    for venv_path in venv_candidates:
        python_path = os.path.join(venv_path, 'bin', 'python')
        if os.path.exists(python_path):
            print(f"🔄 检测到虚拟环境: {venv_path}")
            print("🔄 正在重启脚本...")
            
            # 用虚拟环境的 Python 重新运行
            os.execv(python_path, [python_path] + sys.argv)
    
    return False


# 启动时检查虚拟环境
check_and_activate_venv()

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topaz_data_api import get_stock_data, get_history_data, get_macro_indicators
from utils import parse_stock_list
from qlib_predictor import QlibPredictor, SimpleReturnPredictor


class MLStockAnalyzer:
    """基于机器学习的股票分析器"""
    
    def __init__(self, use_ml: bool = False, history_days: int = 60, batch: int = 0, limit: int = 0):
        """
        Parameters
        ----------
        use_ml : bool
            是否使用 ML 模型（True 使用 LightGBM，False 使用多因子评分）
        history_days : int
            获取历史数据的天数
        batch : int
            批量编号 (1,2,3,4)，0表示全部
        limit : int
            每批数量，0表示不限制
        """
        self.use_ml = use_ml
        self.history_days = history_days
        self.batch = batch
        self.limit = limit
        
        if use_ml:
            self.predictor = QlibPredictor(model_params={
                'objective': 'mse',
                'learning_rate': 0.05,
                'max_depth': 6,
                'num_leaves': 31,
                'n_estimators': 200,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbosity': -1
            })
        else:
            self.predictor = SimpleReturnPredictor()
        
        self.history_data = {}
        self.current_data = {}
        self.stock_industries = {}  # 跟踪每只股票的行业
        self.results = []
    
    def fetch_history_data(self, stocks: List) -> bool:
        """
        批量获取股票历史数据
        
        Parameters
        ----------
        stocks : List
            [(symbol, name, market), ...] 或 [(symbol, name, market, industry), ...]
            
        Returns
        -------
        bool
            是否成功获取至少一只股票的数据
        """
        print(f"📈 获取历史数据（过去 {self.history_days} 天）...")
        print(f"  数据源：A 股 - 新浪财经 | 美股 - FMP (已配置) / Finnhub (免费不支持)")

        success_count = 0
        a_success = 0
        us_success = 0

        for stock in stocks:
            symbol = stock[0]
            name = stock[1]
            market = stock[2] if len(stock) > 2 else 'A 股'
            
            print(f"  获取 {symbol} ({name}) 历史数据...", end=' ')

            data = get_history_data(symbol, market, self.history_days)
            if data is not None and len(data) > 0:
                self.history_data[symbol] = data
                print(f"✓ {len(data)} 条")
                success_count += 1
                if market == 'A 股':
                    a_success += 1
                else:
                    us_success += 1
            else:
                print("✗")

        print(f"\n  汇总：成功 {success_count}/{len(stocks)} 只")
        print(f"    A 股：{a_success} 只 (可用于 ML 训练)")
        if us_success > 0:
            print(f"    美股：{us_success} 只 (FMP API - 可用于 ML 训练)")
        else:
            print(f"    美股：0 只 (未配置 FMP API Key 或获取失败)")

        if a_success > 0 or us_success > 0:
            return True
        return success_count > 0
    
    def fetch_current_data(self, stocks: List) -> bool:
        """
        批量获取股票当前数据
        
        Parameters
        ----------
        stocks : List
            [(symbol, name, market), ...] 或 [(symbol, name, market, industry), ...]
        """
        print("📊 获取股票实时数据...")
        
        success_count = 0
        for stock in stocks:
            symbol = stock[0]
            name = stock[1]
            market = stock[2] if len(stock) > 2 else 'A 股'
            industry = stock[3] if len(stock) > 3 else None
            
            data = get_stock_data(symbol, market, name)
            if data:
                self.current_data[symbol] = data
                # 记录行业
                if industry:
                    self.stock_industries[symbol] = industry
                success_count += 1
        
        print(f"  成功获取 {success_count}/{len(stocks)} 只股票的实时数据")
        return success_count > 0
    
    def train_model(self) -> Dict:
        """
        训练 ML 模型
        
        Returns
        -------
        Dict
            训练结果
        """
        if not self.use_ml:
            return {}
        
        print("\n🤖 训练 LightGBM 模型...")
        
        try:
            result = self.predictor.train_with_history(
                self.history_data,
                predict_days=5  # 预测未来 5 天收益率
            )
            
            print(f"\n  训练完成!")
            print(f"    训练集 R²: {result['train_r2']:.4f}")
            print(f"    验证集 R²: {result['valid_r2']:.4f}")
            print(f"    训练股票数：{len(result['trained_symbols'])}")
            
            return result
            
        except Exception as e:
            print(f"  训练失败：{e}")
            print("  切换到简化版多因子模型...")
            self.use_ml = False
            self.predictor = SimpleReturnPredictor()
            return {}
    
    def analyze_stocks(self) -> List[Dict]:
        """
        分析股票并生成预测

        Returns
        -------
        List[Dict]
            分析结果
        """
        results = []
        
        # 为每只股票使用行业特定的多因子预测器
        for symbol, current in self.current_data.items():
            market = '美股' if current.get('currency') == 'USD' else 'A 股'
            
            # 获取该股票的行业
            industry = self.stock_industries.get(symbol)
            
            # 创建行业特定的预测器
            factor_predictor = SimpleReturnPredictor(industry=industry)
            
            if self.use_ml and symbol in self.history_data:
                # 使用 ML 模型预测
                try:
                    predicted_return = self.predictor.predict_single(
                        self.history_data[symbol],
                        current
                    )
                except Exception as e:
                    # ML 预测失败，回退到多因子模型
                    predicted_return = factor_predictor.predict(current)
            else:
                # 使用多因子模型
                predicted_return = factor_predictor.predict(current)

            # 使用多因子预测器生成投资建议（基于基本面数据）
            advice, risk_level = factor_predictor.get_investment_advice(
                predicted_return, current
            )

            result = {
                'symbol': symbol,
                'name': current.get('name', ''),
                'market': market,
                'industry': industry,
                'current_price': current.get('current_price', 0),
                'currency': current.get('currency', 'CNY'),
                'change': current.get('change', 0),
                'pe_ratio': current.get('pe_ratio', 0),
                'pb_ratio': current.get('pb_ratio', 0),
                'roe': current.get('roe', 0),
                'dividend_yield': current.get('dividend_yield', 0),
                'predicted_return': predicted_return,
                'risk_level': risk_level,
                'advice': advice,
                'use_ml': self.use_ml
            }
            results.append(result)
        
        self.results = results
        return results
    
    def print_feature_importance(self, top_n: int = 10):
        """打印特征重要性"""
        if not self.use_ml:
            return
        
        importance = self.predictor.get_feature_importance()
        if not importance:
            return
        
        print("\n📊 特征重要性 Top {}".format(top_n))
        print("-" * 50)
        
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, imp) in enumerate(sorted_items[:top_n]):
            print(f"  {i+1}. {feature}: {imp:.2f}")
    
    def run(self, us_stocks_file: str = None, a_stocks_file: str = None) -> List[Dict]:
        """
        运行完整分析流程
        
        Parameters
        ----------
        us_stocks_file : str
            美股列表文件路径
        a_stocks_file : str
            A 股列表文件路径
        """
        print("=" * 80)
        print("Topaz-Qlib 机器学习股票分析系统")
        print("数据源：Topaz API (Finnhub + 腾讯财经)")
        if self.use_ml:
            print("模型：LightGBM + 技术特征")
        else:
            print("模型：多因子评分")
        print("=" * 80)
        print(f"运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 读取股票列表
        all_stocks = []
        if us_stocks_file and os.path.exists(us_stocks_file):
            us_stocks = parse_stock_list(us_stocks_file)
            print(f"📂 读取美股列表：{len(us_stocks)} 只")
            for s in us_stocks:
                all_stocks.append((s[0], s[1], '美股', s[2]))  # (symbol, name, market, industry)
        
        if a_stocks_file and os.path.exists(a_stocks_file):
            a_stocks = parse_stock_list(a_stocks_file)
            print(f"📂 读取 A 股列表：{len(a_stocks)} 只")
            for s in a_stocks:
                all_stocks.append((s[0], s[1], 'A 股', s[2]))  # (symbol, name, market, industry)
        
        if not all_stocks:
            print("未找到股票列表，使用示例股票")
        
        # 如果没有股票，使用 A 股示例
        if not all_stocks:
            all_stocks = [
                ('600519', '贵州茅台', 'A 股', '白酒'),
                ('000858', '五粮液', 'A 股', '白酒'),
                ('000333', '美的集团', 'A 股', '家电'),
                ('601318', '中国平安', 'A 股', '保险'),
                ('600036', '招商银行', 'A 股', '银行'),
            ]
        
        # 跟踪每只股票的行业
        for stock in all_stocks:
            if len(stock) >= 4:
                self.stock_industries[stock[0]] = stock[3]  # symbol -> industry
        
        print()
        
        # 批量模式：只处理指定范围
        if self.batch > 0 and self.limit > 0:
            start_idx = (self.batch - 1) * self.limit
            end_idx = start_idx + self.limit
            all_stocks = all_stocks[start_idx:end_idx]
            print(f"📦 批量模式: 第{self.batch}批，每批{self.limit}只")
            print(f"   本批股票: {len(all_stocks)} 只")
            print()
        
        # 获取宏观数据
        print("📊 获取宏观经济数据...")
        macro = get_macro_indicators()
        for key, data in macro.items():
            print(f"  {data['name']}: {data['current_price']:.2f} ({data['change']:+.2f}%)")
        print()
        
        # 获取历史数据
        if self.use_ml:
            if not self.fetch_history_data(all_stocks):
                print("\n  无法获取历史数据，切换到简化版模型")
                self.use_ml = False
                self.predictor = SimpleReturnPredictor()
            print()
        
        # 获取当前数据
        if not self.fetch_current_data(all_stocks):
            print("\n  无法获取实时数据，请检查网络或 API Key")
            return []
        print()
        
        # 训练模型
        if self.use_ml:
            self.train_model()
            print()
        
        # 分析股票
        print("📈 进行预测分析...")
        self.results = self.analyze_stocks()
        print(f"  完成 {len(self.results)} 只股票分析")
        
        # 打印特征重要性
        if self.use_ml:
            self.print_feature_importance()
        
        return self.results
    
    def print_results(self):
        """打印分析结果"""
        if not self.results:
            print("没有分析结果")
            return
        
        print("\n" + "=" * 80)
        print("分析结果")
        print("=" * 80)
        print()
        
        for r in self.results:
            currency = '$' if r['currency'] == 'USD' else '¥'
            price = r['current_price']
            
            model_tag = "[ML]" if r.get('use_ml') else "[因子]"
            print(f"{model_tag} 【{r['symbol']}】{r['name']} ({r['market']})")
            print(f"  当前价格：{currency}{price:.2f}  ({r['change']:+.2f}%)")
            
            if r['pe_ratio'] and r['pe_ratio'] > 0:
                print(f"  市盈率 (PE): {r['pe_ratio']:.2f}")
            if r['pb_ratio'] and r['pb_ratio'] > 0:
                print(f"  市净率 (PB): {r['pb_ratio']:.2f}")
            if r['roe'] and r['roe'] > 0:
                print(f"  净资产收益率 (ROE): {r['roe']:.2f}%")
            if r['dividend_yield'] and r['dividend_yield'] > 0:
                print(f"  股息率：{r['dividend_yield']:.2f}%")
            
            print(f"  预测收益：{r['predicted_return']:.1f}%")
            print(f"  风险等级：{r['risk_level']}")
            print(f"  投资建议：{r['advice']}")
            print()
        
        # 汇总统计
        self._print_summary()
    
    def _print_summary(self):
        """打印汇总统计"""
        print("=" * 80)
        print("汇总统计")
        print("=" * 80)
        
        for market in ['美股', 'A 股']:
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


def main(batch=0, limit=0):
    """
    主函数
    
    Parameters
    ----------
    batch : int
        批量编号 (1,2,3,4)，0表示全部
    limit : int
        每批数量，0表示不限制
    """
    # 确定股票列表文件路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    topaz_dir = os.path.expanduser("~/.openclaw/workspace-topaz/topaz-v3")
    
    us_stocks_file = os.path.join(topaz_dir, "美股关注股票列表.md")
    a_stocks_file = os.path.join(topaz_dir, "A股关注股票列表.md")
    
    # 创建分析器（默认使用多因子模型）
    analyzer = MLStockAnalyzer(use_ml=False, history_days=60, batch=batch, limit=limit)
    
    # 运行分析
    analyzer.run(us_stocks_file, a_stocks_file)
    
    # 打印结果
    analyzer.print_results()


def parse_args():
    """解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(description='Topaz 股票分析系统')
    parser.add_argument('--batch', type=int, default=0, help='批量编号 (1,2,3,4)')
    parser.add_argument('--limit', type=int, default=0, help='每批数量')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(batch=args.batch, limit=args.limit)
