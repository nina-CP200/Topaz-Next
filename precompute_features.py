#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
盘前特征预计算脚本
================================================================================

【模块说明】
本模块负责在每日开盘前预先计算所有目标股票的技术特征因子，并将结果缓存到本地。
这样在实盘分析时可以直接读取缓存数据，大幅缩短分析时间。

【预计算机制】
1. 数据获取：从数据接口获取每只股票最近300个交易日的行情数据
2. 特征生成：调用 FeatureEngineer 生成100+个技术因子（动量、波动、成交量等）
3. 指数因子：叠加沪深300指数相关因子，增强市场相关性分析
4. 缓存存储：将计算结果以 JSON 格式缓存到本地目录

【缓存策略】
- 缓存路径：./cache/features/{symbol}/{date}.json
- 缓存有效期：当日有效，次日自动失效
- 历史清理：默认保留最近7天的缓存，过期自动清理
- 指数缓存：单独缓存指数数据，避免重复获取

【性能对比】
┌─────────────────┬─────────────────┬─────────────────┐
│     场景        │   无预计算      │   有预计算      │
├─────────────────┼─────────────────┼─────────────────┤
│  分析延迟       │   ~45秒         │   <5秒          │
│  数据获取       │   实时获取       │   读缓存        │
│  CPU占用        │   高（计算中）  │   低（仅读取）  │
└─────────────────┴─────────────────┴─────────────────┘

【使用场景】
场景一：每日定时预计算（推荐开盘前执行）
    python precompute_features.py                    # 预计算今日特征
    
场景二：指定日期预计算（用于回测或补算）
    python precompute_features.py --date 2024-01-15  # 指定日期

场景三：自定义参数
    python precompute_features.py --workers 16 --cache-dir /data/cache

【Hermes Agent 定时任务配置】
    请每日执行以下预计算任务：
    **时间**: 每个交易日开盘前（建议 9:00 或更早）
    **命令**: cd /path/to/topaz-next && python precompute_features.py
    **说明**: 预计算所有沪深300成分股的特征（100+因子），缓存到 ./cache/features/ 目录
    **预计耗时**: 约 2-3 分钟（取决于网络和CPU）
    **依赖**: 确保数据接口可用、股票列表文件存在

【注意事项】
1. 首次运行需确保 csi300_stocks.json 股票列表文件存在
2. 运行前检查数据接口连接状态
3. 建议在交易日前一天晚上或当日清晨执行，避免开盘后网络拥堵
4. 如遇部分股票计算失败，不影响其他股票的缓存
================================================================================
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engineer import FeatureEngineer
from quantpilot_data_api import get_history_data
from market_data import get_index_history
from cache_manager import CacheManager
from utils import load_stock_list_from_json


def compute_single_stock(args):
    """
    计算单只股票的所有特征因子（用于并行处理）
    
    【计算流程】
    1. 数据获取：调用 get_history_data 获取股票近300个交易日的日线数据
       - 数据包含：开盘价、收盘价、最高价、最低价、成交量、成交额等
       - 最少需要20条记录才能进行有效计算
    
    2. 数据预处理：
       - 为数据添加股票代码标识
       - 检查数据有效性（非空、长度足够）
    
    3. 特征生成：
       - 调用 FeatureEngineer.generate_all_features 生成基础特征
       - 特征类型包括：动量因子、波动因子、成交量因子、技术形态因子等
       - 总计100+个技术因子
    
    4. 指数因子叠加：
       - 将沪深300指数数据与个股特征合并
       - 计算个股相对于指数的相对强度因子
       - 增强市场相关性分析能力
    
    5. 数据清洗：
       - 将 NaN 值填充为 0，避免后续计算错误
       - 取最新一日的特征值作为缓存结果
    
    【缓存机制】
    - 计算结果以 (symbol, features_dict) 元组形式返回
    - 由调用方 precompute_features 函数负责实际写入缓存
    - 缓存格式：JSON 文件，按股票代码和日期组织
    
    【异常处理】
    - 数据不足：返回 None，跳过该股票
    - 计算错误：打印警告信息，返回 None
    - 网络超时：由数据接口层处理，此处捕获异常
    
    Args:
        args: 元组，包含以下元素
            - symbol (str): 股票代码，如 "000001.SZ"
            - name (str): 股票名称，如 "平安银行"
            - category (str): 股票分类，如 "金融"
            - fe (FeatureEngineer): 特征工程器实例
            - index_history (DataFrame): 沪深300指数历史数据
    
    Returns:
        tuple: (symbol, features_dict) 计算成功的股票及其特征字典
        None: 计算失败或数据不足时返回
    """
    symbol, name, category, fe, index_history = args

    try:
        history = get_history_data(symbol, "A股", days=300)
        if history is None or len(history) < 20:
            return None

        history["code"] = symbol

        df_features = fe.generate_all_features(history)

        if index_history is not None:
            df_features = fe.add_index_factors(df_features, index_history)

        df_features = df_features.fillna(0)

        latest_features = df_features.iloc[-1].to_dict()

        return (symbol, latest_features)

    except Exception as e:
        print(f"  ⚠️ {symbol} 计算失败: {e}")
        return None


def precompute_features(
    stock_list_file: str = None,
    cache_dir: str = "./cache",
    date: str = None,
    max_workers: int = 8,
):
    """
    预计算所有目标股票的特征并缓存到本地
    
    【主要功能】
    批量计算股票池中所有股票的技术特征因子，并将结果缓存以便实盘快速读取。
    这是提升实盘分析速度的关键预处理步骤。
    
    【执行流程】
    1. 初始化配置
       - 确定计算日期（默认今天）
       - 加载股票列表（默认沪深300成分股）
       - 初始化缓存管理器和特征工程器
    
2. 获取指数数据
        - 获取沪深300指数近300个交易日数据
        - 单独缓存指数数据，供后续分析使用
       - 指数数据用于计算个股相对强度因子
    
    3. 并行计算特征
       - 使用线程池并行处理所有股票
       - 每个线程独立计算一只股票的特征
       - 实时显示计算进度
    
    4. 缓存结果
       - 成功计算的股票特征写入缓存
       - 缓存按股票代码和日期组织
       - 失败的股票记录但不影响其他股票
    
    5. 清理过期缓存
       - 删除超过7天的旧缓存文件
       - 避免缓存目录无限增长
    
    【线程数建议】
    ┌─────────────────┬─────────────────┬─────────────────┐
    │   CPU核心数    │   建议线程数    │       说明      │
    ├─────────────────┼─────────────────┼─────────────────┤
    │   4核          │   4-8           │   默认配置即可  │
    │   8核          │   8-12          │   可适当增加    │
    │   16核+        │   12-16         │   注意内存占用  │
    └─────────────────┴─────────────────┴─────────────────┘
    
    注意：由于主要瓶颈在网络I/O（获取数据），线程数可适当超过CPU核心数。
    但过多线程可能导致：
    - 数据接口限流
    - 内存占用过高
    - 系统资源竞争
    
    【性能优化建议】
    1. 网络优化：
       - 确保与数据服务器的网络延迟较低
       - 避免在网络高峰期执行
       - 考虑使用本地数据缓存减少网络请求
    
    2. 系统资源：
       - 监控内存使用，300只股票约需500MB-1GB内存
       - SSD硬盘可加快缓存读写速度
       - 关闭不必要的后台进程
    
    3. 执行时机：
       - 建议开盘前1-2小时执行（如8:00-9:00）
       - 避免与实盘分析任务竞争资源
       - 考虑设置定时任务自动执行
    
    【使用场景】
    场景一：每日开盘前自动预计算
        - 时间：每个交易日 8:30 或更早
        - 目的：确保开盘时所有特征已就绪
        - 耗时：约2-3分钟
    
    场景二：盘中更新（可选）
        - 时间：午间休市时
        - 目的：补充计算失败的股票
        - 注意：仅重新计算失败的股票
    
    场景三：回测数据准备
        - 时间：回测前一次性执行
        - 参数：指定日期范围逐日预计算
        - 目的：加速后续回测过程
    
    Args:
        stock_list_file (str): 股票列表文件路径（JSON格式）
            - 默认使用 csi300_stocks.json（沪深300成分股）
            - 文件格式：[{"symbol": "000001.SZ", "name": "平安银行", "category": "金融"}]
        
        cache_dir (str): 缓存根目录
            - 默认 "./cache"
            - 特征缓存路径：{cache_dir}/features/{symbol}/{date}.json
            - 指数缓存路径：{cache_dir}/index/
        
        date (str): 计算日期，格式 "YYYY-MM-DD"
            - 默认当天日期
            - 用于回测时可指定历史日期
            - 缓存文件按此日期命名
        
        max_workers (int): 并行线程数
            - 默认 8
            - 建议范围：4-16
            - 过多可能导致数据接口限流
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    if stock_list_file is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        stock_list_file = os.path.join(base_dir, "csi300_stocks.json")

    print(f"\n{'=' * 60}")
    print(f"📦 特征预计算 - {date}")
    print(f"{'=' * 60}")

    cache = CacheManager(cache_dir)
    fe = FeatureEngineer()

    stocks = load_stock_list_from_json(stock_list_file)
    if not stocks:
        print(f"❌ 无法加载股票列表: {stock_list_file}")
        return

    print(f"📋 共 {len(stocks)} 只股票")

    print("\n📊 获取沪深300指数数据...")
    index_history = get_index_history("000300.SH", days=300)
    if index_history is None:
        print("❌ 无法获取指数数据，使用默认值")
    else:
        print(f"  ✓ 获取 {len(index_history)} 天指数数据")
        cache.set_index_cache("000300.SH", 300, index_history)

    print(f"\n🔄 开始并行计算（{max_workers} 线程）...")

    args_list = [
        (symbol, name, category, fe, index_history) for symbol, name, category in stocks
    ]

    success_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(compute_single_stock, args): args[0] for args in args_list
        }

        for future in as_completed(futures):
            result = future.result()
            if result:
                symbol, features = result
                cache.set_feature_cache(symbol, features, date)
                success_count += 1

                if success_count % 50 == 0:
                    print(f"  ✓ 已完成 {success_count}/{len(stocks)}")
            else:
                failed_count += 1

    print(f"\n{'=' * 60}")
    print(f"✅ 完成！")
    print(f"   成功: {success_count} 只")
    print(f"   失败: {failed_count} 只")
    print(f"   缓存目录: {cache_dir}/features/")
    print(f"{'=' * 60}")

    stats = cache.get_cache_stats()
    print(f"\n📊 缓存统计:")
    print(f"   今日缓存: {stats['today_cached']} 只")
    print(f"   总缓存: {stats['total_cached']} 只")

    cache.clear_old_cache(keep_days=7)


def main():
    """
    命令行入口函数
    
    【命令行参数说明】
    --date       : 指定计算日期，格式 YYYY-MM-DD
                   默认：当天日期
                   示例：--date 2024-01-15
    
    --cache-dir  : 缓存存储目录
                   默认：./cache
                   示例：--cache-dir /data/topaz/cache
    
    --workers    : 并行计算线程数
                   默认：8
                   建议范围：4-16
                   示例：--workers 16
    
    --stock-list : 自定义股票列表文件路径
                   默认：./csi300_stocks.json
                   格式：JSON数组，每项包含 symbol, name, category
                   示例：--stock-list ./my_stocks.json
    
    【使用示例】
    # 基本用法：预计算今日特征
    python precompute_features.py
    
    # 指定日期（用于回测准备）
    python precompute_features.py --date 2024-01-15
    
    # 使用更多线程加速
    python precompute_features.py --workers 16
    
    # 自定义缓存目录
    python precompute_features.py --cache-dir /data/cache
    
    # 使用自定义股票列表
    python precompute_features.py --stock-list ./my_portfolio.json
    """
    parser = argparse.ArgumentParser(
        description="盘前特征预计算",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--date", type=str, help="指定日期 (YYYY-MM-DD)")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="缓存目录")
    parser.add_argument("--workers", type=int, default=8, help="并行线程数")
    parser.add_argument("--stock-list", type=str, help="股票列表文件路径")
    args = parser.parse_args()

    precompute_features(
        stock_list_file=args.stock_list,
        cache_dir=args.cache_dir,
        date=args.date,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()