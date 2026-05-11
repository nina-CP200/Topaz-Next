from __future__ import annotations
from fastapi import APIRouter
from src.analysis.query import load_analysis_results
from src.sector import analyze_sectors
from backend.routers.common import patch_results

router = APIRouter()


@router.get("")
def sector_analysis():
    data = load_analysis_results()
    if data is None:
        return {"error": "暂无分析结果", "sectors": [], "market_regime": "unknown"}

    results = data.get("results", data.get("all_results", []))
    regime = data.get("market_regime", "sideways")

    results = patch_results(results, regime)

    stock_list = []
    for r in results:
        stock_list.append(
            {
                "name": r.get("name", ""),
                "industry": r.get("industry", ""),
                "return_5d": r.get("predicted_return", 0) * 0.01,
                "return_20d": r.get("predicted_return", 0) * 0.02,
                "composite_score": r.get("composite_score", 0),
            }
        )

    sector_data = analyze_sectors(stock_list, regime)
    return sector_data
