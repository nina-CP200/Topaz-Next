#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
板块分析模块 - 板块涨幅排名（仅供参考，不做买卖推荐）

注意：板块分析存在幸存者偏差，回测结果不可信。
仅作为参考信号，不构成投资建议。
"""

from typing import Dict, List
from collections import defaultdict
import numpy as np


def analyze_sectors(results: List[Dict], market_regime: str = "sideways") -> Dict:
    """
    按板块聚合，计算涨幅排名

    注意：仅展示数据，不做推荐
    """
    sectors = defaultdict(lambda: {
        "stocks": [], "returns_5d": [], "returns_20d": [],
    })

    for r in results:
        industry = r.get("industry", "")
        if not industry:
            continue
        d = sectors[industry]
        d["stocks"].append(r)
        d["returns_5d"].append(r.get("return_5d", 0))
        d["returns_20d"].append(r.get("return_20d", 0))

    sector_results = []
    for name, data in sectors.items():
        n = len(data["stocks"])
        if n == 0:
            continue

        avg_ret_20d = np.mean(data["returns_20d"]) if data["returns_20d"] else 0
        avg_ret_5d = np.mean(data["returns_5d"]) if data["returns_5d"] else 0

        sector_results.append({
            "name": name,
            "momentum_20d": round(avg_ret_20d * 100, 2),
            "momentum_5d": round(avg_ret_5d * 100, 2),
            "stock_count": n,
            "top_stocks": sorted(data["stocks"], key=lambda x: x.get("composite_score", 0), reverse=True)[:3],
        })

    sector_results.sort(key=lambda x: x["momentum_20d"], reverse=True)

    return {
        "sectors": sector_results,
        "market_regime": market_regime,
    }


def print_sector_report(sector_data: Dict):
    """打印板块排名（仅供参考）"""
    sectors = sector_data.get("sectors", [])

    if not sectors:
        print("\n⚠️ 无板块数据")
        return

    print("\n" + "=" * 80)
    print("📊 板块涨幅排名（仅供参考，不构成投资建议）")
    print("=" * 80)

    # 全部板块排名
    print(f"\n📋 板块排名（共{len(sectors)}个）")
    print("-" * 80)
    for i, s in enumerate(sectors, 1):
        top = ", ".join([t["name"] for t in s["top_stocks"][:2]])
        print(f"  #{i:2d} {s['name']:6s} 20日:{s['momentum_20d']:+.1f}% 5日:{s['momentum_5d']:+.1f}% {s['stock_count']}只 关注: {top}")
