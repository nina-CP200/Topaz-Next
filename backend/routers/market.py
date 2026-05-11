from __future__ import annotations
from fastapi import APIRouter, Query
from src.data.market import (
    get_index_data,
    get_index_history,
    get_market_sentiment,
    get_market_adjusted_thresholds,
)
from src.config import get_market_state_manager

router = APIRouter()


@router.get("/overview")
def market_overview():
    index_data = get_index_data("000300.SH")
    sentiment = get_market_sentiment()
    manager = get_market_state_manager()

    if index_data:
        regime_description = manager.get_status_summary()
        current_regime = manager.state["current_regime"]
        thresholds = get_market_adjusted_thresholds(current_regime)
        return {
            "regime": current_regime,
            "regime_description": regime_description
            or "N/A",
            "index_price": index_data.get("price", 0),
            "index_change_pct": index_data.get("change_pct", 0),
            "index_high": index_data.get("high", 0),
            "index_low": index_data.get("low", 0),
            "index_open": index_data.get("open", 0),
            "index_prev_close": index_data.get("prev_close", 0),
            "index_name": index_data.get("name", "沪深300"),
            "sentiment_advance_ratio": sentiment.get("advance_ratio", 0)
            if sentiment
            else 0,
            "sentiment_up_count": sentiment.get("up_count", 0) if sentiment else 0,
            "sentiment_down_count": sentiment.get("down_count", 0)
            if sentiment
            else 0,
            "sentiment_limit_up": sentiment.get("limit_up", 0) if sentiment else 0,
            "sentiment_limit_down": sentiment.get("limit_down", 0)
            if sentiment
            else 0,
            "buy_threshold": thresholds.get("buy_threshold", 0.55),
            "sell_threshold": thresholds.get("sell_threshold", 0.35),
            "position_max": thresholds.get("position_max", 0.6),
        }
    return {"error": "无法获取市场数据"}


@router.get("/history")
def market_history(days: int = Query(60, ge=10, le=500)):
    df = get_index_history("000300.SH", days=days)
    if df is None or df.empty:
        return {"error": "无法获取历史数据", "data": []}
    df = df.tail(days)
    data = df[["date", "open", "high", "low", "close", "volume"]].copy()
    return {
        "data": data.to_dict(orient="records"),
        "index_name": "沪深300",
        "code": "000300.SH",
    }
