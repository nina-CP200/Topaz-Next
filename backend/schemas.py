from __future__ import annotations
from pydantic import BaseModel
from typing import Optional


class HoldingItem(BaseModel):
    symbol: str
    name: str = ""
    shares: int
    cost: float


class PortfolioSave(BaseModel):
    holdings: list[HoldingItem]


class PortfolioAnalyzeRequest(BaseModel):
    holdings: list[HoldingItem]


class StockResultItem(BaseModel):
    symbol: str
    name: str
    industry: str
    current_price: float
    change_pct: float
    probability: float
    composite_score: float
    reasons: list[str]
    advice: str
    position_pct: float
    model_confidence: float
    rank: int
    predicted_return: float
    risk_level: str


class DailyAnalysisResult(BaseModel):
    date: str
    market_regime: str
    model_confidence: float
    advance_ratio: float
    recommended_position: float
    total_stocks: int
    results: list[StockResultItem]


class MarketOverview(BaseModel):
    regime: str
    regime_description: str
    index_price: float
    index_change_pct: float
    index_high: float
    index_low: float
    index_open: float
    index_prev_close: float
    index_name: str
    sentiment_advance_ratio: float
    sentiment_up_count: int
    sentiment_down_count: int
    sentiment_limit_up: int
    sentiment_limit_down: int
    buy_threshold: float
    sell_threshold: float
    position_max: float


class HoldingAnalysis(BaseModel):
    symbol: str
    name: str
    industry: str
    shares: int
    cost: float
    current_price: float
    market_value: float
    pnl: float
    pnl_pct: float
    change_today: float
    composite_score: float
    advice: str
    reasons: list[str]
    rsi: float
    ret_5d: float
    ret_20d: float


class PortfolioAnalysisResult(BaseModel):
    holdings: list[HoldingAnalysis]
    total_value: float
    total_cost: float
    total_pnl: float
    total_pnl_pct: float
    industry_allocation: dict
    regime: str
    buy_count: int
    sell_count: int
    loss_count: int


class SectorItem(BaseModel):
    name: str
    momentum_20d: float
    momentum_5d: float
    stock_count: int
    top_stocks: list[dict]


class SectorAnalysis(BaseModel):
    sectors: list[SectorItem]
    market_regime: str
