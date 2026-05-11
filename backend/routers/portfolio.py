from __future__ import annotations
import json
import os
from fastapi import APIRouter, HTTPException
from backend.schemas import PortfolioSave, PortfolioAnalyzeRequest
from src.portfolio import analyze_portfolio as _analyze_portfolio
from src.utils.utils import load_stock_list_from_json
from src.data.api import get_stock_data
from backend.routers.common import PROJECT_ROOT

router = APIRouter()

PORTFOLIO_FILE = os.path.join(PROJECT_ROOT, "data/raw/portfolio.json")
CSI300_STOCK_LIST = os.path.join(PROJECT_ROOT, "config/csi300_stocks.json")


def _load_holdings() -> list[dict]:
    if not os.path.exists(PORTFOLIO_FILE):
        return []
    with open(PORTFOLIO_FILE, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "holdings" in data:
        return data["holdings"]
    return []


def _save_holdings(holdings: list[dict]):
    os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(holdings, f, ensure_ascii=False, indent=2)


def _lookup_stock_name(symbol: str) -> str:
    stock_list = load_stock_list_from_json(CSI300_STOCK_LIST)
    for s in stock_list:
        if s[0] == symbol or s[0].replace(".SH", "").replace(".SZ", "") == symbol.replace(".SH", "").replace(".SZ", ""):
            return s[1]
    try:
        data = get_stock_data(symbol, "A股")
        if data and data.get("name"):
            return data["name"]
    except Exception:
        pass
    return symbol


@router.get("")
def get_portfolio():
    holdings = _load_holdings()
    stock_list = load_stock_list_from_json(CSI300_STOCK_LIST)
    name_map = {}
    for s in stock_list:
        key = s[0].replace(".SH", "").replace(".SZ", "")
        name_map[key] = s[1]

    for h in holdings:
        if not h.get("name"):
            code = h.get("symbol", "").replace(".SH", "").replace(".SZ", "")
            h["name"] = name_map.get(code, "") or _lookup_stock_name(code)
    return {"holdings": holdings}


@router.put("")
def save_portfolio(data: PortfolioSave):
    holdings = []
    for h in data.holdings:
        item = h.model_dump()
        if not item.get("name"):
            item["name"] = _lookup_stock_name(item["symbol"])
        holdings.append(item)
    _save_holdings(holdings)
    return {"message": "持仓已保存", "count": len(holdings)}


@router.post("/analyze")
def analyze(req: PortfolioAnalyzeRequest):
    if not req.holdings:
        raise HTTPException(status_code=400, detail="持仓为空")

    csi300 = set()
    for s in load_stock_list_from_json(CSI300_STOCK_LIST):
        csi300.add(s[0].replace(".SH", "").replace(".SZ", ""))

    normalized = []
    failed = []
    for h in req.holdings:
        item = {"shares": h.shares, "cost": h.cost, "name": h.name}
        code = h.symbol.strip()
        base = code.replace(".SH", "").replace(".SZ", "")
        if base not in csi300:
            failed.append({"code": code, "reason": f"{h.name or base} 不是沪深300成分股，无法分析"})
            continue
        if code.startswith("6") and not code.endswith(".SH"):
            code = f"{code}.SH"
        elif (code.startswith("0") or code.startswith("3")) and not code.endswith(".SZ"):
            code = f"{code}.SZ"
        item["code"] = code
        normalized.append(item)

    if not normalized:
        return {
            "holdings": [], "failed": failed,
            "total_value": 0, "total_cost": 0, "total_pnl": 0, "total_pnl_pct": 0,
            "industry_allocation": {}, "regime": "sideways",
            "buy_count": 0, "sell_count": 0, "loss_count": 0,
        }

    result = _analyze_portfolio(normalized)
    result.setdefault("failed", []).extend(failed)
    return result
