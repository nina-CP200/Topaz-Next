#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略层 - 综合评分、理由生成、仓位建议

在 ML 概率基础上，结合技术指标给出复合评分，
输出明确的买入/卖出列表 + 理由 + 仓位建议。
"""


def compute_composite_score(ml_proba: float, features: dict) -> tuple:
    """
    综合评分 = ML概率 × 50% + 技术面 × 30% + 动量 × 20%

    Returns:
        (composite_score, reasons)
    """
    # ========== 1. 技术面评分 (30%) ==========
    # RSI
    rsi = features.get('rsi', 50)
    if rsi < 30:
        rsi_score = 0.8  # 超卖，看涨
    elif rsi < 45:
        rsi_score = 0.6
    elif rsi < 55:
        rsi_score = 0.5
    elif rsi < 70:
        rsi_score = 0.4
    else:
        rsi_score = 0.2  # 超买，看跌

    # MACD
    macd_hist = features.get('macd_hist', 0)
    macd = features.get('macd', 0)
    macd_signal = features.get('macd_signal', 0)
    if macd > macd_signal and macd_hist > 0:
        macd_score = 0.8  # 金叉+红柱
    elif macd > macd_signal:
        macd_score = 0.6  # 金叉
    elif macd_hist < 0:
        macd_score = 0.3  # 绿柱
    else:
        macd_score = 0.5

    # 均线位置
    price_to_ma20 = features.get('price_to_ma20', 1.0)
    if price_to_ma20 > 1.03:
        ma_score = 0.8  # 远高于20日线
    elif price_to_ma20 > 1.0:
        ma_score = 0.6  # 在20日线上方
    elif price_to_ma20 > 0.97:
        ma_score = 0.4  # 在20日线下方
    else:
        ma_score = 0.2  # 远低于20日线

    technical_score = rsi_score * 0.35 + macd_score * 0.35 + ma_score * 0.30

    # ========== 2. 动量评分 (20%) ==========
    ret_1d = features.get('return_1d', 0)
    ret_5d = features.get('return_5d', 0)
    ret_20d = features.get('return_20d', 0)

    # 短期动量
    if ret_5d > 0.05:
        short_mom = 0.8
    elif ret_5d > 0.02:
        short_mom = 0.65
    elif ret_5d > 0:
        short_mom = 0.55
    elif ret_5d > -0.02:
        short_mom = 0.45
    elif ret_5d > -0.05:
        short_mom = 0.35
    else:
        short_mom = 0.2

    # 中期趋势
    if ret_20d > 0.08:
        mid_mom = 0.8
    elif ret_20d > 0.03:
        mid_mom = 0.65
    elif ret_20d > 0:
        mid_mom = 0.55
    elif ret_20d > -0.03:
        mid_mom = 0.45
    elif ret_20d > -0.08:
        mid_mom = 0.35
    else:
        mid_mom = 0.2

    momentum_score = short_mom * 0.5 + mid_mom * 0.5

    # ========== 3. 综合评分 ==========
    composite = ml_proba * 0.50 + technical_score * 0.30 + momentum_score * 0.20

    # ========== 4. 生成理由 ==========
    reasons = []

    # RSI 理由
    if rsi < 30:
        reasons.append("RSI超卖")
    elif rsi > 70:
        reasons.append("RSI超买")

    # MACD 理由
    if macd > macd_signal and macd_hist > 0:
        reasons.append("MACD金叉")
    elif macd < macd_signal and macd_hist < 0:
        reasons.append("MACD死叉")

    # 均线理由
    if price_to_ma20 > 1.03:
        reasons.append("站稳20日线上方")
    elif price_to_ma20 < 0.97:
        reasons.append("跌破20日线")

    # 动量理由
    if ret_5d > 0.03:
        reasons.append("短期动量强")
    elif ret_5d < -0.03:
        reasons.append("短期动量弱")

    if ret_20d > 0.05:
        reasons.append("中期趋势向上")
    elif ret_20d < -0.05:
        reasons.append("中期趋势向下")

    # 成交量理由
    volume_ratio = features.get('volume_ratio', 1.0)
    if volume_ratio > 1.5:
        reasons.append("放量")
    elif volume_ratio < 0.5:
        reasons.append("缩量")

    if not reasons:
        reasons.append("指标中性")

    return composite, reasons


def get_advice_from_score(composite: float, model_confidence: float = 0.5) -> str:
    """根据综合评分返回投资建议"""
    if composite >= 0.65:
        advice = "建议买入"
    elif composite >= 0.55:
        advice = "可以关注"
    elif composite >= 0.45:
        advice = "建议观望"
    elif composite >= 0.35:
        advice = "建议回避"
    else:
        advice = "建议卖出"

    if model_confidence >= 0.8 and composite >= 0.55:
        advice = "强烈" + advice.replace("建议", "").replace("可以", "")
        if not advice.startswith("强烈"):
            advice = "强烈建议" + advice

    return advice


def suggest_position(composite: float, market_regime: str) -> float:
    """根据综合评分和市场环境建议仓位比例"""
    base = {
        "bull": 0.25, "recovery": 0.20, "pullback": 0.18,
        "sideways": 0.15, "bear": 0.10,
    }
    max_per_stock = base.get(market_regime, 0.15)

    # 评分越高，仓位越接近上限
    ratio = max(0.3, min(1.0, (composite - 0.3) / 0.5))
    return round(max_per_stock * ratio, 2)
