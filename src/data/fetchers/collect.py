#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
QuantPilot - 训练数据收集器
================================================================================

模块说明：
    本模块用于收集沪深300成分股的历史行情数据，并计算技术指标和大盘因子，
    生成用于机器学习模型训练的特征数据集。

数据收集流程：
    1. 加载沪深300成分股列表和行业映射
    2. 收集沪深300指数历史数据（用于计算大盘因子）
    3. 遍历每只成分股：
       a. 获取历史K线数据（开高低收量）
       b. 计算技术指标（MA、RSI、MACD、KDJ、布林带等）
       c. 合并大盘因子（指数收益、相对强度、Beta等）
    4. 保存合并后的数据到CSV文件
    5. 支持断点续传，每10只股票保存一次进度

输出文件说明：
    - training_data.csv: 训练数据集，包含技术指标和大盘因子
    - collection_progress.json: 收集进度文件，用于断点恢复

注意事项：
    1. API限流处理：遇到限流时自动休眠1小时后继续
    2. 数据完整性：每只股票至少需要60天数据才会被纳入数据集
    3. 内存管理：数据量大时定期保存，避免内存溢出
    4. 网络超时：请求超时设置为15秒，失败股票会记录到failed列表
    5. 断点续传：可通过 --no-resume 参数禁用，默认启用

作者：QuantPilot Team
版本：1.0.0
"""

import os
import json
import time
import csv
import requests  # 添加 requests 导入
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from src.data.api import get_history_data, get_stock_data


class TrainingDataCollector:
    """
    训练数据收集器
    
    功能：
        - 收集沪深300成分股历史数据
        - 计算技术指标和大盘因子
        - 支持断点续传
    """
    
    def __init__(self, data_dir: str = None):
        """
        初始化数据收集器
        
        参数：
            data_dir: 数据目录路径，默认为当前脚本所在目录
                     该目录需包含以下文件：
                     - csi300_stocks.json: 成分股列表
                     - csi300_industry_map.json: 行业映射表
        
        属性说明：
            stocks: 成分股列表，格式为 [{'code': '000001', 'name': '平安银行'}, ...]
            stock_industry: 股票-行业映射，格式为 {'000001': '银行', ...}
            collected: 已成功收集的股票数量
            failed: 收集失败的股票代码列表
            rate_limited: 是否触发了API限流
            index_data: 沪深300指数历史数据DataFrame
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'config')
        self.stocks = []
        self.stock_industry = {}
        self.collected = 0
        self.failed = []
        self.rate_limited = False
        self.session_start = None
        self.index_data = None
        
        self._load_stock_list()
    
    def _load_stock_list(self):
        """
        加载成分股列表和行业映射
        
        文件格式：
            csi300_stocks.json:
                [
                    {"code": "000001", "name": "平安银行"},
                    {"code": "000002", "name": "万科A"},
                    ...
                ]
            
            csi300_industry_map.json:
                {
                    "stock_industry": {
                        "000001": "银行",
                        "000002": "房地产",
                        ...
                    }
                }
        
        异常处理：
            文件不存在时不会报错，stocks和stock_industry将保持为空
        """
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
    
    def collect_index_data(self, days: int = 500) -> Optional[pd.DataFrame]:
        """
        收集沪深300指数历史数据
        
        参数：
            days: 获取最近N个交易日的数据，默认500天
        
        返回：
            DataFrame或None，包含以下列：
                - date: 交易日期
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - volume: 成交量
        
        数据源：
            新浪财经API: https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData
        
        注意：
            该数据用于计算大盘因子，如指数收益率、相对强度、Beta等
        """
        print("\n收集沪深300指数数据...")
        try:
            index_code = '000300'  # 沪深300
            url = f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
            params = {
                "symbol": f"sh{index_code}",
                "scale": 240,  # 日线
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            data = response.json()
            
            if data and isinstance(data, list):
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['day'].str.split(' ').str[0])
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                
                df = df.sort_values('date').reset_index(drop=True)
                df = df.tail(days)  # 取最近 days 天
                
                self.index_data = df
                print(f"  成功获取 {len(df)} 天指数数据")
                return df
            else:
                print("  获取指数数据失败")
                return None
        except Exception as e:
            print(f"  获取指数数据失败: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        参数：
            df: 股票历史数据DataFrame，必须包含以下列：
                - date: 交易日期
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - volume: 成交量
        
        返回：
            添加技术指标后的DataFrame，新增以下指标：
            
            移动平均线类：
                - ma5, ma10, ma20, ma60: 5/10/20/60日均线
                - ma5_slope, ma10_slope, ma20_slope: 均线斜率（5日变化率）
                - price_to_ma5/10/20: 价格相对均线的位置（偏离度）
            
            波动率类：
                - volatility_5/10/20: 5/10/20日滚动波动率
            
            成交量类：
                - volume_ma5, volume_ma10: 成交量均线
                - volume_ratio: 量比（当日成交量/5日均量）
            
            动量类：
                - return_1d/5d/10d/20d: 1/5/10/20日收益率
            
            技术指标：
                - rsi: 14日RSI指标
                - macd, macd_signal, macd_hist: MACD指标及信号线
                - bb_upper, bb_lower, bb_middle, bb_position: 布林带及位置
                - kdj_k, kdj_d: KDJ指标
            
            大盘因子（需先调用collect_index_data）：
                - index_close, index_return_1d/5d/20d: 指数收盘价及收益率
                - index_ma_position: 指数相对20日均线位置
                - index_volatility: 指数波动率
                - relative_strength_1d/5d/20d: 个股相对指数的超额收益
                - beta: 个股Beta系数
        
        注意：
            前60行数据因窗口期不足，部分指标为NaN，训练时需处理
        """
        if 'date' not in df.columns:
            if df.index.name == 'datetime' or hasattr(df.index, 'name'):
                df['date'] = df.index
            elif 'day' in df.columns:
                df['date'] = pd.to_datetime(df['day'])
        
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
        
        # ========== 新增：大盘因子 ==========
        if self.index_data is not None:
            df = self._add_index_factors(df)
        
        return df
    
    def _add_index_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加大盘因子（私有方法）
        
        参数：
            df: 股票数据DataFrame，需包含date列和return列
        
        返回：
            合并大盘因子后的DataFrame，新增列：
                - index_close: 沪深300收盘价
                - index_return_1d/5d/20d: 指数收益率
                - index_ma_position: 指数相对20日均线位置
                - index_volatility: 指数20日波动率
                - relative_strength_1d/5d/20d: 个股相对指数的超额收益
                - beta: 个股Beta系数（20日滚动协方差计算）
        
        计算方法：
            - Beta = Cov(个股收益, 指数收益) / Var(指数收益)
            - 相对强度 = 个股收益 - 指数收益
        
        前置条件：
            需要先调用collect_index_data()获取指数数据
        """
        try:
            # 确保日期格式一致
            df['date'] = pd.to_datetime(df['date'])
            index_df = self.index_data.copy()
            
            # 计算指数因子
            index_df['index_return_1d'] = index_df['close'].pct_change(1)
            index_df['index_return_5d'] = index_df['close'].pct_change(5)
            index_df['index_return_20d'] = index_df['close'].pct_change(20)
            index_df['index_ma20'] = index_df['close'].rolling(20).mean()
            index_df['index_ma_position'] = index_df['close'] / index_df['index_ma20'] - 1
            index_df['index_volatility'] = index_df['close'].pct_change().rolling(20).std()
            
            # 选择需要合并的列
            index_cols = ['date', 'close', 'index_return_1d', 'index_return_5d', 
                          'index_return_20d', 'index_ma_position', 'index_volatility']
            index_features = index_df[index_cols].copy()
            index_features = index_features.rename(columns={'close': 'index_close'})
            
            # 合并
            df = df.merge(index_features, on='date', how='left')
            
            # 计算相对强度
            df['relative_strength_1d'] = df['return_1d'] - df['index_return_1d']
            df['relative_strength_5d'] = df['return_5d'] - df['index_return_5d']
            df['relative_strength_20d'] = df['return_20d'] - df['index_return_20d']
            
            # Beta（个股相对于大盘的弹性）
            if len(df) >= 20:
                covariance = df['return_1d'].rolling(20).cov(df['index_return_1d'])
                variance = df['index_return_1d'].rolling(20).var()
                df['beta'] = covariance / (variance + 1e-10)
            
        except Exception as e:
            print(f"  添加大盘因子失败: {e}")
        
        return df
    
    def collect_stock_data(self, code: str, days: int = 500) -> Optional[pd.DataFrame]:
        """
        收集单只股票的历史数据
        
        参数：
            code: 股票代码，格式如 '000001'、'600000'
            days: 获取最近N个交易日的数据，默认500天
        
        返回：
            DataFrame或None（获取失败时返回None）
            DataFrame包含原始K线数据和技术指标，详见calculate_technical_indicators()
        
        数据验证：
            - 数据少于60天的股票将被过滤（窗口期不足）
            - API限流时会设置self.rate_limited=True
        
        异常处理：
            - 遇到rate/limit/429关键词时识别为限流
            - 其他异常记录失败但不中断程序
        """
        try:
            df = get_history_data(code, days=days)
            if df is None or len(df) < 60:
                print(f"  {code}: 数据不足 ({len(df) if df is not None else 0} 天)")
                return None
            
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
        """
        收集所有股票数据（主入口函数）
        
        参数：
            max_stocks: 最大收集股票数量，默认300（沪深300全部）
            days: 每只股票获取的历史数据天数，默认500天
            output_file: 输出CSV文件名，默认'training_data.csv'
            progress_file: 进度文件名，默认'collection_progress.json'
            resume: 是否启用断点续传，默认True
        
        断点续传机制：
            - 读取progress_file获取上次中断位置
            - 从上次成功的索引继续收集
            - 加载已存在的output_file数据
        
        限流处理：
            - 检测到API限流时休眠1小时
            - 休眠前保存当前进度
        
        检查点机制：
            - 每收集10只股票保存一次数据和进度
            - 防止程序崩溃导致数据丢失
        
        返回：
            所有股票数据的DataFrame列表
        
        注意：
            - 调用前确保已准备好csi300_stocks.json和csi300_industry_map.json
            - 数据量大时建议使用resume=True
        """
        print("\n" + "="*60)
        print("训练数据收集器")
        print("="*60)
        
        # 先收集指数数据
        self.collect_index_data(days)
        
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
        """
        保存数据到CSV文件（私有方法）
        
        参数：
            all_data: DataFrame列表，每个DataFrame为一只股票的数据
            output_path: 输出文件完整路径
        
        输出格式（CSV列说明）：
            基础信息：
                - date: 交易日期
                - code: 股票代码
                - industry: 所属行业
            
            价格数据：
                - open, high, low, close: 开高低收
                - volume: 成交量
            
            移动平均线：
                - ma5, ma10, ma20, ma60: 均线值
                - ma5_slope, ma10_slope, ma20_slope: 均线斜率
                - price_to_ma5, price_to_ma10, price_to_ma20: 价格偏离度
            
            波动率：
                - volatility_5, volatility_10, volatility_20: 滚动波动率
            
            成交量：
                - volume_ma5: 5日均量
                - volume_ratio: 量比
            
            收益率：
                - return_1d, return_5d, return_10d, return_20d: 滚动收益率
            
            技术指标：
                - rsi: RSI指标
                - macd, macd_signal, macd_hist: MACD指标
                - bb_position: 布林带位置
                - kdj_k, kdj_d: KDJ指标
            
            大盘因子：
                - index_close: 指数收盘价
                - index_return_1d/5d/20d: 指数收益率
                - index_ma_position: 指数均线位置
                - index_volatility: 指数波动率
                - relative_strength_1d/5d/20d: 相对强度
                - beta: Beta系数
        
        注意：
            - 只保存存在的列，避免因缺失列导致错误
            - 不保存索引列
        """
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
            'bb_position', 'kdj_k', 'kdj_d',
            # 新增大盘因子
            'index_close', 'index_return_1d', 'index_return_5d', 'index_return_20d',
            'index_ma_position', 'index_volatility',
            'relative_strength_1d', 'relative_strength_5d', 'relative_strength_20d',
            'beta'
        ]
        
        # 只保存存在的列
        save_cols = [c for c in columns if c in combined.columns]
        combined[save_cols].to_csv(output_path, index=False)
    
    def _save_progress(self, progress_file: str, last_idx: int, done: bool = False):
        """
        保存收集进度（私有方法）
        
        参数：
            progress_file: 进度文件名
            last_idx: 上次处理到的股票索引
            done: 是否已完成全部收集
        
        进度文件格式（JSON）：
            {
                "last_idx": 150,          // 上次处理到的索引
                "collected": 148,         // 已成功收集数量
                "failed": ["000001"],     // 失败的股票代码列表
                "last_update": "2024-01-15T10:30:00",  // 最后更新时间
                "done": false             // 是否全部完成
            }
        
        用途：
            - 支持断点续传，程序崩溃后可从上次位置继续
            - 记录失败的股票代码，便于后续重试
        """
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
    """
    主函数 - 命令行入口
    
    命令行参数：
        --max: 最大收集股票数量，默认300
        --days: 历史数据天数，默认500
        --output: 输出文件名，默认training_data.csv
        --no-resume: 禁用断点续传，从头开始收集
    
    使用示例：
        # 收集全部300只股票，500天数据
        python collect_training_data.py
        
        # 收集前50只股票
        python collect_training_data.py --max 50
        
        # 收集100天数据，输出到指定文件
        python collect_training_data.py --days 100 --output my_data.csv
        
        # 从头开始收集（忽略进度文件）
        python collect_training_data.py --no-resume
    """
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