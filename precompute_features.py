#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盘前特征预计算脚本

每日定时运行（如开盘前 9:00），预计算所有沪深300成分股的特征并缓存。
实盘分析时直接读取缓存，无需实时计算，可将延迟从 ~45秒降至 <5秒。

使用方法:
    python precompute_features.py                    # 预计算今日特征
    python precompute_features.py --date 2024-01-15  # 指定日期

Hermes Agent 定时任务 Prompt:
    请每日执行以下预计算任务：
    **时间**: 每个交易日开盘前（建议 9:00 或更早）
    **命令**: cd /path/to/topaz-next && python precompute_features.py
    **说明**: 预计算所有沪深300成分股的特征（100+因子），缓存到 ./cache/features/ 目录
    **预计耗时**: 约 2-3 分钟
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
from utils import parse_stock_list


def compute_single_stock(args):
    """
    计算单只股票特征（用于并行）

    Args:
        args: (symbol, name, category, fe, index_history)

    Returns:
        (symbol, features_dict) 或 None
    """
    symbol, name, category, fe, index_history = args

    try:
        history = get_history_data(symbol, "A股", days=60)
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
    预计算所有股票的特征并缓存

    Args:
        stock_list_file: 股票列表文件路径
        cache_dir: 缓存目录
        date: 指定日期（默认今天）
        max_workers: 并行线程数
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    if stock_list_file is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        stock_list_file = os.path.join(base_dir, "csi300_stock_list.md")

    print(f"\n{'=' * 60}")
    print(f"📦 特征预计算 - {date}")
    print(f"{'=' * 60}")

    cache = CacheManager(cache_dir)
    fe = FeatureEngineer()

    stocks = parse_stock_list(stock_list_file)
    if not stocks:
        print(f"❌ 无法加载股票列表: {stock_list_file}")
        return

    print(f"📋 共 {len(stocks)} 只股票")

    print("\n📊 获取沪深300指数数据...")
    index_history = get_index_history("000300.SH", days=60)
    if index_history is None:
        print("❌ 无法获取指数数据，使用默认值")
    else:
        print(f"  ✓ 获取 {len(index_history)} 天指数数据")
        cache.set_index_cache("000300.SH", 60, index_history)

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
