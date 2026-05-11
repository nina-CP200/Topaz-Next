from __future__ import annotations
import os

from src.utils.utils import load_stock_list_from_json
from src.strategy import suggest_position

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_industry_map():
    stock_list = load_stock_list_from_json(os.path.join(PROJECT_ROOT, "config/csi300_stocks.json"))
    m = {}
    for symbol, name, industry in stock_list:
        code = symbol.replace(".SH", "").replace(".SZ", "")
        m[code] = industry
        m[symbol] = industry
    return m


_industry_map = None


def _get_industry(symbol: str) -> str:
    global _industry_map
    if _industry_map is None:
        _industry_map = _load_industry_map()
    return _industry_map.get(symbol, _industry_map.get(symbol.replace(".SH", "").replace(".SZ", ""), "其他"))


def patch_results(results: list, regime: str = "sideways") -> list:
    patched = []
    for r in results:
        if "industry" not in r or not r.get("industry"):
            r["industry"] = _get_industry(r.get("symbol", ""))
        if "composite_score" not in r or r.get("composite_score") is None:
            r["composite_score"] = r.get("probability", 0.5)
        if "reasons" not in r:
            r["reasons"] = []
        if "position_pct" not in r:
            r["position_pct"] = suggest_position(r.get("composite_score", 0.5), regime)
        if "predicted_return" not in r:
            r["predicted_return"] = (r.get("composite_score", 0.5) - 0.5) * 20
        if "risk_level" not in r:
            p = r.get("composite_score", 0.5)
            r["risk_level"] = "低风险" if p >= 0.65 else "中风险" if p >= 0.5 else "高风险"
        patched.append(r)
    return patched
