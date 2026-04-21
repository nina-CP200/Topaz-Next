#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存管理器 - 指数数据、特征数据缓存
用于降低实时分析延迟，预计算数据可直接读取
"""

import os
import joblib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd


class CacheManager:
    """统一缓存管理"""

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self._memory_cache: Dict[str, Any] = {}
        self._cache_date: str = datetime.now().strftime("%Y-%m-%d")

    def _check_date(self):
        """检查是否跨天，跨天清空内存缓存"""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._cache_date:
            self._memory_cache = {}
            self._cache_date = today

    def get_feature_cache(self, symbol: str, date: str = None) -> Optional[Dict]:
        """
        获取预计算的特征缓存

        Args:
            symbol: 股票代码
            date: 日期（默认今天）

        Returns:
            特征字典或None
        """
        self._check_date()
        if date is None:
            date = self._cache_date

        key = f"{symbol}_{date}"

        if key in self._memory_cache:
            return self._memory_cache[key]

        cache_file = self.cache_dir / "features" / f"{key}.pkl"
        if cache_file.exists():
            try:
                data = joblib.load(cache_file)
                self._memory_cache[key] = data
                return data
            except Exception as e:
                print(f"加载缓存失败: {e}")
                return None

        return None

    def set_feature_cache(self, symbol: str, features: Dict, date: str = None):
        """
        保存特征缓存

        Args:
            symbol: 股票代码
            features: 特征字典
            date: 日期（默认今天）
        """
        if date is None:
            date = self._cache_date

        key = f"{symbol}_{date}"
        self._memory_cache[key] = features

        feature_dir = self.cache_dir / "features"
        feature_dir.mkdir(exist_ok=True)
        cache_file = feature_dir / f"{key}.pkl"

        try:
            joblib.dump(features, cache_file)
        except Exception as e:
            print(f"保存缓存失败: {e}")

    def get_index_cache(self, symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
        """
        获取指数数据缓存

        Args:
            symbol: 指数代码
            days: 天数

        Returns:
            DataFrame或None
        """
        self._check_date()
        key = f"index_{symbol}_{days}"

        if key in self._memory_cache:
            return self._memory_cache[key]

        return None

    def set_index_cache(self, symbol: str, days: int, data: pd.DataFrame):
        """
        保存指数数据缓存

        Args:
            symbol: 指数代码
            days: 天数
            data: DataFrame
        """
        self._check_date()
        key = f"index_{symbol}_{days}"
        self._memory_cache[key] = data

    def clear_old_cache(self, keep_days: int = 7):
        """
        清理过期缓存文件

        Args:
            keep_days: 保留天数
        """
        feature_dir = self.cache_dir / "features"
        if not feature_dir.exists():
            return

        cutoff_date = datetime.now() - pd.Timedelta(days=keep_days)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")

        removed = 0
        for cache_file in feature_dir.glob("*.pkl"):
            filename = cache_file.stem
            if "_" in filename:
                file_date = filename.split("_")[-1]
                if file_date < cutoff_str:
                    cache_file.unlink()
                    removed += 1

        if removed > 0:
            print(f"清理了 {removed} 个过期缓存文件")

    def get_cache_stats(self) -> Dict:
        """
        获取缓存统计信息

        Returns:
            统计字典
        """
        feature_dir = self.cache_dir / "features"

        total_files = 0
        today_files = 0
        if feature_dir.exists():
            files = list(feature_dir.glob("*.pkl"))
            total_files = len(files)
            today_str = self._cache_date
            today_files = len([f for f in files if today_str in f.stem])

        return {
            "total_cached": total_files,
            "today_cached": today_files,
            "memory_cached": len(self._memory_cache),
            "cache_date": self._cache_date,
        }


if __name__ == "__main__":
    cache = CacheManager()
    stats = cache.get_cache_stats()
    print("缓存统计:")
    print(f"  文件缓存总数: {stats['total_cached']}")
    print(f"  今日缓存: {stats['today_cached']}")
    print(f"  内存缓存: {stats['memory_cached']}")
    print(f"  缓存日期: {stats['cache_date']}")
