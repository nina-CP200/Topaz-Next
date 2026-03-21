#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantPilot - 训练数据收集器
收集沪深300成分股历史数据，用于机器学习训练
"""

import os
import json
import time
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from quantpilot_data_api import get_cn_history_data, get_cn_realtime_data


class TrainingDataCollector:
    """训练数据收集器"""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.dirname(os.path.abspath(__file__))
        self.stocks = []
        self.stock_industry = {}
        self.collected = 0
        self.failed = []
        self.rate_limited = False
        self.session_start = None
        
        # 加载股票列表
        self._load_stock_list()
    
    def _load_stock_list(self):
        """加载成分股列表"""
        stocks_file = os.path.join(self.data_dir, "csi300_stocks.json")
        if os.path.exists(stocks_file):
            with open(stocks_file, 'r', encoding='utf-8') as f:
                self.stocks = json.load(f)
        
        mapping_file = os.path.join(self.data_dir, "csi300_industry_map.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            self.stock_industry = mapping.get('stock_industry', {})
        
        print(f"加载 {len(self.stocks)} 只成分股，{len(self.stock_industry)} 只有行业映射")
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 确保按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        
        # 移动平均线
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma60'] = df['close'].rolling(window=60).mean()
        
        # 移动平均线斜率（趋势）
        df['ma5_slope'] = df['ma5'].diff(5) / df['ma5'].shift(5)
        df['ma10_slope'] = df['ma10'].diff(5) / df['ma10'].shift(5)
        df['ma20_slope'] = df['ma20'].diff(5) / df['ma20'].shift(5)
        
        # 相对位置
        df['price_to_ma5'] = df['close'] / df['ma5'] - 1
        df['price_to_ma10'] = df['close'] / df['ma10'] - 1
        df['price_to_ma20'] = df['close'] / df['ma20'] - 1
        
        # 波动率
        df['volatility_5'] = df['close'].pct_change().rolling(window=5).std()
        df['volatility_10'] = df['close'].pct_change().rolling(window=10).std()
        df['volatility_20'] = df['close'].pct_change().rolling(window=20).std()
        
        # 成交量变化
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma10'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma5']
        
        # 价格动量
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_20d'] = df['close'].pct_change(20)
        
        # RSI (14日)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 布林带
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # KDJ
        low_min = df['low'].rolling(window=9).min()
        high_max = df['high'].rolling(window=9).max()
        df['kdj_k'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
        df['kdj_d'] = df['kdj_k'].rolling(window=3).mean()
        
        return df
    
    def collect_stock_data(self, code: str, days: int = 500) -> Optional[pd.DataFrame]:
        """收集单只股票的历史数据"""
        try:
            data = get_cn_history_data(code, days=days)
            if not data or len(data) < 60:
                print(f"  {code}: 数据不足 ({len(data) if data else 0} 天)")
                return None
            
            df = pd.DataFrame(data)
            df['code'] = code
            df['industry'] = self.stock_industry.get(code, 'unknown')
            
            # 计算技术指标
            df = self.calculate_technical_indicators(df)
            
            return df
            
        except Exception as e:
            error_msg = str(e).lower()
            if 'rate' in error_msg or 'limit' in error_msg or '429' in error_msg:
                print(f"  {code}: API 限流，需要休息")
                self.rate_limited = True
            else:
                print(f"  {code}: 获取失败 - {e}")
            return None
    
    def collect_all(self, 
                    max_stocks: int = 300,
                    days: int = 500,
                    output_file: str = "training_data.csv",
                    progress_file: str = "collection_progress.json",
                    resume: bool = True):
        """收集所有股票数据"""
        
        print("\n" + "="*60)
        print("训练数据收集器")
        print("="*60)
        
        # 检查进度
        start_idx = 0
        if resume and os.path.exists(os.path.join(self.data_dir, progress_file)):
            with open(os.path.join(self.data_dir, progress_file), 'r') as f:
                progress = json.load(f)
            start_idx = progress.get('last_idx', 0)
            self.collected = progress.get('collected', 0)
            self.failed = progress.get('failed', [])
            print(f"从断点恢复: 已收集 {self.collected} 只，从第 {start_idx+1} 只继续")
        
        self.session_start = datetime.now()
        
        # 准备股票列表
        stocks_to_collect = self.stocks[start_idx:max_stocks]
        total = len(stocks_to_collect)
        
        print(f"准备收集 {total} 只股票，每只 {days} 天数据\n")
        
        all_data = []
        
        # 如果已有数据文件，先加载
        output_path = os.path.join(self.data_dir, output_file)
        if resume and os.path.exists(output_path):
            try:
                existing = pd.read_csv(output_path)
                all_data = [existing]
                print(f"加载已有数据: {len(existing)} 条记录")
            except:
                pass
        
        for i, stock in enumerate(stocks_to_collect):
            code = stock['code']
            name = stock['name']
            
            print(f"[{start_idx + i + 1}/{len(self.stocks)}] {code} {name}...", end='')
            
            # 检查是否需要休息
            if self.rate_limited:
                print("\n\n⚠️ API 限流，休息 1 小时后继续...")
                # 保存进度
                self._save_progress(progress_file, start_idx + i)
                # 等待 1 小时
                time.sleep(3600)
                self.rate_limited = False
                print(f"\n继续收集...")
            
            # 获取数据
            df = self.collect_stock_data(code, days)
            
            if df is not None:
                all_data.append(df)
                self.collected += 1
                print(f" ✓ ({len(df)} 条)")
            else:
                self.failed.append(code)
                print(f" ✗")
            
            # 每收集 10 只保存一次
            if (i + 1) % 10 == 0:
                self._save_data(all_data, output_path)
                self._save_progress(progress_file, start_idx + i + 1)
                print(f"  [检查点: {self.collected} 只成功, {len(self.failed)} 只失败]")
            
            # 请求间隔
            time.sleep(0.3)
        
        # 最终保存
        if all_data:
            self._save_data(all_data, output_path)
        
        # 保存最终进度
        self._save_progress(progress_file, len(self.stocks), done=True)
        
        # 统计
        print("\n" + "="*60)
        print("收集完成!")
        print("="*60)
        print(f"成功: {self.collected} 只")
        print(f"失败: {len(self.failed)} 只")
        print(f"数据文件: {output_path}")
        
        return all_data
    
    def _save_data(self, all_data: list, output_path: str):
        """保存数据到CSV"""
        if not all_data:
            return
        
        combined = pd.concat(all_data, ignore_index=True)
        
        # 选择有用的列
        columns = [
            'date', 'code', 'industry',
            'open', 'high', 'low', 'close', 'volume',
            'ma5', 'ma10', 'ma20', 'ma60',
            'ma5_slope', 'ma10_slope', 'ma20_slope',
            'price_to_ma5', 'price_to_ma10', 'price_to_ma20',
            'volatility_5', 'volatility_10', 'volatility_20',
            'volume_ma5', 'volume_ratio',
            'return_1d', 'return_5d', 'return_10d', 'return_20d',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_position', 'kdj_k', 'kdj_d'
        ]
        
        # 只保存存在的列
        save_cols = [c for c in columns if c in combined.columns]
        combined[save_cols].to_csv(output_path, index=False)
    
    def _save_progress(self, progress_file: str, last_idx: int, done: bool = False):
        """保存收集进度"""
        progress = {
            'last_idx': last_idx,
            'collected': self.collected,
            'failed': self.failed,
            'last_update': datetime.now().isoformat(),
            'done': done
        }
        with open(os.path.join(self.data_dir, progress_file), 'w') as f:
            json.dump(progress, f, indent=2)


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='训练数据收集')
    parser.add_argument('--max', type=int, default=300, help='最大收集股票数量')
    parser.add_argument('--days', type=int, default=500, help='历史数据天数')
    parser.add_argument('--output', type=str, default='training_data.csv', help='输出文件')
    parser.add_argument('--no-resume', action='store_true', help='不从断点恢复')
    args = parser.parse_args()
    
    collector = TrainingDataCollector()
    collector.collect_all(
        max_stocks=args.max,
        days=args.days,
        output_file=args.output,
        resume=not args.no_resume
    )


if __name__ == '__main__':
    main()