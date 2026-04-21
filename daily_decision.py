#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topaz 每日投资决策系统
根据 ML 分析结果生成投资建议并更新虚拟投资组合
支持大盘环境判断和条件策略

运行模式：
  --execute  : 执行交易（默认）
  --preview  : 预告模式，只生成决策建议，不执行交易
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ensemble_model import EnsembleModel
from feature_engineer import FeatureEngineer
from quantpilot_data_api import get_history_data, get_stock_data
from utils import parse_stock_list
from cache_manager import CacheManager
from market_data import (
    get_index_data,
    get_index_history,
    get_market_sentiment,
    judge_market_environment,
    get_market_adjusted_thresholds,
)


def load_portfolio(portfolio_file: str) -> Dict:
    """加载投资组合"""
    if os.path.exists(portfolio_file):
        with open(portfolio_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "initial_capital": 1000000,
        "cash": 1000000,
        "holdings": {},
        "trades": [],
        "daily_values": [],
    }


def save_portfolio(portfolio: Dict, portfolio_file: str):
    """保存投资组合"""
    with open(portfolio_file, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=2, ensure_ascii=False)


def analyze_stocks(
    stock_list_file: str, use_csi300_model: bool = False, portfolio: Dict = None
) -> Dict:
    """分析股票列表（优化版：缓存+并行+批量预测）

    改进:
    1. 优先读取预计算特征缓存（预计算完成后延迟 <3s）
    2. 无缓存时使用线程池并行计算（延迟降至 ~8s）
    3. 指数数据缓存，避免重复请求
    """

    import joblib

    cache = CacheManager()
    today = datetime.now().strftime("%Y-%m-%d")

    # ========== 市场环境判断（放宽定义）==========
    from market_data import get_index_history, get_market_sentiment

    index_history_30 = get_index_history("000300.SH", days=30)
    sentiment = get_market_sentiment()

    adv_ratio = 0.5
    if sentiment:
        adv_ratio = sentiment.get("advance_ratio", 0.5)

    # 计算20日收益
    ret_20d = 0
    if index_history_30 is not None and len(index_history_30) >= 20:
        ret_20d = (
            index_history_30["close"].iloc[-1] / index_history_30["close"].iloc[-20] - 1
        )

    # 环境分类（放宽）
    if adv_ratio > 0.55 and ret_20d > 0:
        detailed_regime = "bull"
        model_confidence = 0.7  # IC=0.10
    elif adv_ratio > 0.55 and ret_20d < -0.02:
        detailed_regime = "recovery"
        model_confidence = 0.9  # IC=0.15（最佳）
    elif adv_ratio < 0.45 and ret_20d < -0.02:
        detailed_regime = "bear"
        model_confidence = 0.6  # IC=0.06（改善）
    elif adv_ratio < 0.45 and ret_20d > 0.02:
        detailed_regime = "pullback"
        model_confidence = 0.8  # IC=0.11
    else:
        detailed_regime = "sideways"
        model_confidence = 0.5  # IC=0.03

    print(f"\n📊 市场环境: {detailed_regime}")
    print(f"   上涨比例: {adv_ratio:.1%}, 20日收益: {ret_20d * 100:.1f}%")
    print(f"   模型置信度: {model_confidence:.0%}")

    # 环境效果说明 + 动态仓位建议（震荡分散、熊市保守策略）
    regime_effect = {
        "recovery": "✅ 最佳环境（IC=0.15）- 建议95%仓位",
        "pullback": "✅ 有效环境（IC=0.11）- 建议80%仓位",
        "bull": "✅ 有效环境（IC=0.10）- 建议70%仓位",
        "bear": "⚠️ 熊市保守（IC=0.06）- 建议20%仓位，严格止损",
        "sideways": "📊 震荡分散（IC=0.03）- 建议50%仓位，diversify多股",
    }

    position_advice = {
        "recovery": 0.95,
        "pullback": 0.80,
        "bull": 0.70,
        "bear": 0.20,  # 熊市降低到20%
        "sideways": 0.50,  # 震荡市提高到50%，但单股上限降低
    }

    recommended_position = position_advice.get(detailed_regime, 0.50)

    print(f"   {regime_effect.get(detailed_regime, '未知')}")
    print(f"   📌 建议最大仓位: {recommended_position:.0%}")

    # ========== 加载分组模型 ==========
    ensemble = None

    if use_csi300_model:
        if os.path.exists("ensemble_model_regime_based.pkl"):
            model_data = joblib.load("ensemble_model_regime_based.pkl")
            models_by_regime = model_data.get("models_by_regime", {})

            if detailed_regime in models_by_regime:
                selected = models_by_regime[detailed_regime]
                print(f"📦 已加载 {detailed_regime} 环境模型")
            else:
                selected = models_by_regime.get(
                    "sideways", models_by_regime.get("bull", {})
                )
                print(f"📦 已加载兜底模型")

            if selected:
                ensemble = {
                    "model": selected["model"],
                    "scaler": selected["scaler"],
                    "feature_cols": selected["features"],
                }
        else:
            print("⚠️ 未找到分组模型")
            return {"all_results": [], "watchlist_results": []}
    else:
        # 非CSI300模式：加载默认 ensemble_model_csi300_latest.pkl
        if os.path.exists("ensemble_model_csi300_latest.pkl"):
            model_data = joblib.load("ensemble_model_csi300_latest.pkl")
            # 使用 lightgbm 子模型
            ensemble = {
                "model": model_data["models"]["lightgbm"],
                "scaler": model_data["scaler"],
                "feature_cols": model_data["feature_cols"],
            }
            print(f"📦 已加载默认 ensemble_model_csi300_latest.pkl")
        else:
            print("⚠️ 未找到默认模型 ensemble_model_csi300_latest.pkl")
            return {"all_results": [], "watchlist_results": []}

    fe = FeatureEngineer()

    stocks = parse_stock_list(stock_list_file)

    # 加载关注列表
    base_dir = os.path.dirname(os.path.abspath(__file__))
    watchlist_file = os.path.join(base_dir, "A股关注股票列表.md")
    watchlist_symbols = set()
    if os.path.exists(watchlist_file):
        try:
            watchlist_stocks = parse_stock_list(watchlist_file)
            watchlist_symbols = set([s for s, n, c in watchlist_stocks])
            print(f"📋 加载关注列表: {len(watchlist_symbols)} 只股票")
        except Exception as e:
            print(f"⚠️ 加载关注列表失败: {e}")

    # 选择要分析的股票：持仓 + 关注 + 全量沪深300
    holdings = portfolio.get("holdings", {}) if portfolio else {}
    holding_symbols = set(holdings.keys())

    # 持仓股票
    holding_stocks = [(s, n, c) for s, n, c in stocks if s in holding_symbols]

    # 关注列表股票（不包括持仓）
    watch_stocks = [
        (s, n, c)
        for s, n, c in stocks
        if s in watchlist_symbols and s not in holding_symbols
    ]

    # 全量分析：使用所有沪深300成分股（不包括持仓和关注）
    all_csi300_stocks = [
        (s, n, c)
        for s, n, c in stocks
        if s not in holding_symbols and s not in watchlist_symbols
    ]

    # 合并股票列表（持仓优先，然后关注，然后全量）
    selected_stocks = holding_stocks + watch_stocks + all_csi300_stocks

    print(
        f"📋 分析股票: {len(holding_stocks)} 持仓 + {len(watch_stocks)} 关注 + {len(all_csi300_stocks)} 全量 = {len(selected_stocks)} 只"
    )

    results = []

    # 获取沪深300指数历史数据（使用缓存）
    print("📊 获取沪深300指数历史数据...")
    index_history = cache.get_index_cache("000300.SH", 60)
    if index_history is None:
        index_history = get_index_history("000300.SH", days=60)
        if index_history is not None:
            cache.set_index_cache("000300.SH", 60, index_history)
    if index_history is None:
        print("  ⚠️ 无法获取指数数据，将缺少指数因子")
    else:
        print(f"  ✓ 获取 {len(index_history)} 天指数数据")

    # ========== 第一步：尝试从缓存读取特征 ==========
    print("\n📦 尝试读取预计算特征缓存...")
    cached_results = []
    need_compute = []

    for symbol, name, category in selected_stocks:
        cached_features = cache.get_feature_cache(symbol, today)
        if cached_features:
            # 只对持仓和关注列表取实时价，其他用缓存close初筛
            need_realtime = symbol in holding_symbols or symbol in watchlist_symbols
            current = get_stock_data(symbol, "A股", name) if need_realtime else None
            current_price = (
                current.get("current_price", cached_features.get("close", 0))
                if current
                else cached_features.get("close", 0)
            )
            feature_cols = ensemble["feature_cols"]
            X = np.array([[cached_features.get(f, 0) for f in feature_cols]])
            X = np.clip(X, -1e10, 1e10)
            X_scaled = ensemble["scaler"].transform(X)
            proba = ensemble["model"].predict_proba(X_scaled)[:, 1][0]

            expected_return = (proba - 0.5) * 20

            if proba >= 0.65:
                risk_level = "低风险"
            elif proba >= 0.50:
                risk_level = "中风险"
            elif proba >= 0.40:
                risk_level = "高风险"
            else:
                risk_level = "极高风险"

            if proba >= 0.60:
                advice = "建议买入"
            elif proba >= 0.50:
                advice = "建议持有"
            elif proba >= 0.40:
                advice = "建议观望"
            else:
                advice = "建议回避"

            if model_confidence < 0.3:
                if advice == "建议买入":
                    advice = "谨慎买入"
                elif advice == "建议持有":
                    advice = "建议观望"
            elif model_confidence >= 0.8:
                advice = advice.replace("建议", "强烈")

            cached_results.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "current_price": current_price,
                    "change_pct": current.get(
                        "change", cached_features.get("return_1d", 0) * 100
                    )
                    if current
                    else (
                        cached_features.get("return_1d", 0) * 100
                        if cached_features.get("return_1d")
                        else 0
                    ),
                    "pe_ratio": 0,
                    "pb_ratio": 0,
                    "roe": 0,
                    "probability": proba,
                    "predicted_return": expected_return,
                    "risk_level": risk_level,
                    "advice": advice,
                    "model_confidence": model_confidence,
                }
            )
        else:
            need_compute.append((symbol, name, category))

    results.extend(cached_results)
    print(f"  ✓ 缓存命中: {len(cached_results)} 只，需计算: {len(need_compute)} 只")

    # ========== 第二步：并行计算未缓存的股票 ==========
    if need_compute:
        print(f"\n🔄 并行计算 {len(need_compute)} 只股票...")

        def analyze_one(args):
            symbol, name, category = args
            try:
                history = get_history_data(symbol, "A股", days=60)
                current = get_stock_data(symbol, "A股", name)
                if history is None or not current:
                    return None

                history["code"] = symbol
                df_features = fe.generate_all_features(history)
                if index_history is not None:
                    df_features = fe.add_index_factors(df_features, index_history)
                df_features = df_features.fillna(0)

                feature_cols = ensemble["feature_cols"]
                missing = [f for f in feature_cols if f not in df_features.columns]
                for f in missing:
                    df_features[f] = 0

                latest = df_features.iloc[-1:][feature_cols]
                X = latest.values
                X = np.clip(X, -1e10, 1e10)
                X_scaled = ensemble["scaler"].transform(X)
                proba = ensemble["model"].predict_proba(X_scaled)[:, 1][0]

                # 缓存计算结果供下次使用
                cache.set_feature_cache(symbol, df_features.iloc[-1].to_dict(), today)

                expected_return = (proba - 0.5) * 20

                if proba >= 0.65:
                    risk_level = "低风险"
                elif proba >= 0.50:
                    risk_level = "中风险"
                elif proba >= 0.40:
                    risk_level = "高风险"
                else:
                    risk_level = "极高风险"

                if proba >= 0.60:
                    advice = "建议买入"
                elif proba >= 0.50:
                    advice = "建议持有"
                elif proba >= 0.40:
                    advice = "建议观望"
                else:
                    advice = "建议回避"

                if model_confidence < 0.3:
                    if advice == "建议买入":
                        advice = "谨慎买入"
                    elif advice == "建议持有":
                        advice = "建议观望"
                elif model_confidence >= 0.8:
                    advice = advice.replace("建议", "强烈")

                return {
                    "symbol": symbol,
                    "name": name,
                    "current_price": current.get("current_price", 0),
                    "change_pct": current.get("change", 0),
                    "pe_ratio": current.get("pe_ratio", 0),
                    "pb_ratio": current.get("pb_ratio", 0),
                    "roe": current.get("roe", 0),
                    "probability": proba,
                    "predicted_return": expected_return,
                    "risk_level": risk_level,
                    "advice": advice,
                    "model_confidence": model_confidence,
                }
            except Exception as e:
                return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(analyze_one, args): args[0] for args in need_compute
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        print(f"  ✓ 并行计算完成")

    # 提取关注列表的分析结果
    watchlist_analysis = [r for r in results if r["symbol"] in watchlist_symbols]

    return {
        "all_results": results,
        "watchlist_results": watchlist_analysis,
        "market_regime": detailed_regime,
        "model_confidence": model_confidence,
        "advance_ratio": adv_ratio,
        "recommended_position": recommended_position,
    }


def generate_decision(
    results: List[Dict], portfolio: Dict, watchlist_results: List[Dict] = None
) -> Dict:
    """生成投资决策（结合大盘环境）"""

    # ========== 大盘环境判断 ==========
    print("\n📊 判断大盘环境...")
    index_data = None
    sentiment = None
    try:
        index_data = get_index_data()
        sentiment = get_market_sentiment()
        market_env = judge_market_environment(index_data, sentiment)
        market_thresholds = get_market_adjusted_thresholds(market_env)

        print(
            f"  沪深300: {index_data['price']:.2f} ({index_data['change_pct']:+.2f}%)"
        )
        if sentiment:
            print(
                f"  市场情绪: 上涨{sentiment['up_count']}家, 下跌{sentiment['down_count']}家, 上涨比例{sentiment['advance_ratio']:.1%}"
            )
        print(f"  环境判断: {market_env} - {market_thresholds['description']}")
    except Exception as e:
        print(f"  大盘判断失败: {e}，使用默认参数")
        market_env = "sideways"
        market_thresholds = get_market_adjusted_thresholds("sideways")

    # 从大盘阈值中获取参数
    BUY_THRESHOLD = market_thresholds["buy_threshold"]
    SELL_THRESHOLD = market_thresholds["sell_threshold"]
    POSITION_MAX = market_thresholds["position_max"]
    SINGLE_MAX = market_thresholds["single_max"]

    # ========== 综合评分（建议1：概率 × 预测收益）==========
    def calc_score(stock):
        """综合评分 = 概率 × (1 + 预测收益/10)"""
        prob = stock["probability"]
        ret = stock.get("predicted_return", 0)
        return prob * (1 + ret / 10)

    for r in results:
        r["score"] = calc_score(r)

    # 筛选建议买入的股票（根据大盘环境调整 - 震荡分散、熊市保守）
    if market_env == "bull":
        # 牛市：买入建议买入和持有的股票
        buy_candidates = [r for r in results if r["advice"] in ["建议买入", "建议持有"]]
    elif market_env == "bear":
        # 熊市保守策略：只买入最强的股票（强烈建议买入 + 概率>0.85）
        buy_candidates = [
            r
            for r in results
            if r["advice"] in ["强烈建议买入", "建议买入"] and r["probability"] > 0.85
        ]
    elif market_env == "sideways":
        # 震荡分散策略：放宽门槛，捕捉更多机会（建议买入 + 建议/谨慎持有，概率>55%）
        buy_candidates = [
            r
            for r in results
            if r["probability"] > 0.55
            and r["advice"] in ["建议买入", "强烈建议买入", "建议持有", "谨慎买入"]
        ]
    else:
        # 反弹/复苏：买入建议买入的股票
        buy_candidates = [r for r in results if r["advice"] == "建议买入"]

    buy_candidates.sort(key=lambda x: x["score"], reverse=True)

    # 筛选现有持仓
    holdings = portfolio.get("holdings", {})
    hold_candidates = [r for r in results if r["symbol"] in holdings]

    decisions = {
        "buy": [],
        "sell": [],
        "hold": [],
        "watchlist": watchlist_results if watchlist_results else [],
        "all_results": results,
        "market_info": {
            "environment": market_env,
            "index_price": index_data.get("price", 0) if index_data else 0,
            "index_change": index_data.get("change_pct", 0) if index_data else 0,
            "advance_ratio": sentiment.get("advance_ratio", 0.5) if sentiment else 0.5,
            "description": market_thresholds["description"],
        },
    }

    # ========== 卖出决策 ==========

    # 参数（根据大盘环境调整 - 震荡分散、熊市保守）
    if market_env == "bear":
        # 熊市保守：严格止损，不贪婪止盈
        STOP_LOSS_PCT = -0.04  # 熊市止损更严格 4%
        TAKE_PROFIT_PCT = 0.08  # 熊市止盈更保守 8%
    elif market_env == "sideways":
        # 震荡分散：适度止损，快速止盈（震荡市机会多，有利润就走）
        STOP_LOSS_PCT = -0.06  # 震荡市止损 6%
        TAKE_PROFIT_PCT = 0.08  # 震荡市止盈 8%（快速兑现）
    else:
        STOP_LOSS_PCT = -0.08  # 其他环境止损 8%
        TAKE_PROFIT_PCT = 0.15 if market_env == "bull" else 0.12  # 牛市止盈更高

    AVOID_THRESHOLD = SELL_THRESHOLD
    SWAP_PROFIT_THRESHOLD = 0.05
    SWAP_SCORE_THRESHOLD = 0.65
    SWAP_SCORE_GAP = 0.15 if market_env != "bull" else 0.20  # 牛市换仓门槛更高

    # 计算总资产（用于持仓集中度检查）
    cash = portfolio.get("cash", 0)
    holdings_value = sum(
        h["shares"] * h.get("current_price", h["cost"]) for h in holdings.values()
    )
    total_value = cash + holdings_value

    for stock in hold_candidates:
        symbol = stock["symbol"]
        if symbol not in holdings:
            continue

        holding = holdings[symbol]
        cost_price = holding["cost"]
        current_price = stock["current_price"]
        pnl_pct = (current_price - cost_price) / cost_price
        prob = stock["probability"]
        score = stock["score"]
        position_value = holding["shares"] * current_price
        position_pct = position_value / total_value if total_value > 0 else 0

        sell_reason = None

        # 条件1: 止损
        if pnl_pct < STOP_LOSS_PCT:
            sell_reason = f"止损: 亏损{pnl_pct:.1%} (阈值{STOP_LOSS_PCT:.0%})"

        # 条件2: 止盈 + 评分下降
        elif pnl_pct > TAKE_PROFIT_PCT and prob < 0.7:
            sell_reason = f"止盈: 盈利{pnl_pct:.1%}, 评分{prob:.1%}"

        # 条件3: ML 建议"回避"
        elif stock["advice"] == "建议回避" or prob < AVOID_THRESHOLD:
            sell_reason = f"ML建议回避(概率{prob:.1%})"

        # 条件4: 换仓（盈利 + 评分下降 + 有更好机会）
        elif pnl_pct > SWAP_PROFIT_THRESHOLD and prob < SWAP_SCORE_THRESHOLD:
            for candidate in buy_candidates[:5]:
                if candidate["score"] - score > SWAP_SCORE_GAP:
                    sell_reason = f"换仓: 盈利{pnl_pct:.1%}, 评分{prob:.1%} → {candidate['name']}评分{candidate['probability']:.1%}"
                    break

        # 条件5: 持仓集中度处理（综合判断）
        if position_pct > 0.25 and not sell_reason:
            # 集中度高，但需要综合看
            if prob < 0.5 and pnl_pct < 0:
                # 低评分 + 亏损 → 减仓
                sell_reason = f"风控减仓: 持仓{position_pct:.1%}, 评分{prob:.1%}, 亏损{pnl_pct:.1%}"
            # 其他情况：持有观察，不盲目减仓

        if sell_reason:
            decisions["sell"].append(
                {
                    "symbol": symbol,
                    "name": stock["name"],
                    "shares": holding["shares"],
                    "price": current_price,
                    "amount": holding["shares"] * current_price,
                    "probability": prob,
                    "score": score,
                    "reason": sell_reason,
                }
            )

    # ========== 买入决策 ==========

    # 计算可用现金（当前现金 + 预计卖出金额）
    cash = portfolio.get("cash", 0)
    sell_amount = sum(s["amount"] for s in decisions["sell"])
    available_cash = cash + sell_amount

    # 建议2：动态仓位分配（震荡分散、熊市保守策略）
    def get_position_pct(stock, is_adding_position=False, pnl_pct=0):
        """根据评分和大盘环境动态分配仓位

        Args:
            stock: 股票信息
            is_adding_position: 是否加仓
            pnl_pct: 当前盈亏比例
        """
        score = stock["score"]

        if is_adding_position:
            # 加仓逻辑：抄底（亏损时加仓）
            if market_env == "bear":
                # 熊市禁止加仓
                return 0
            elif market_env == "sideways":
                # 震荡市谨慎加仓（只在深度亏损+高评分时）
                if pnl_pct < -0.08 and score > 0.75:
                    return min(0.06, SINGLE_MAX)  # 震荡市加仓保守 6%
                else:
                    return 0
            elif pnl_pct < -0.05 and score > 0.7:
                return min(0.15, SINGLE_MAX)
            elif pnl_pct < 0 and score > 0.6:
                return min(0.08, SINGLE_MAX * 0.5)
            else:
                return 0
        else:
            # 新建仓逻辑（根据大盘环境调整）
            if market_env == "bull":
                # 牛市：可以更激进
                if score > 0.85:
                    return min(0.25, SINGLE_MAX)
                elif score > 0.75:
                    return min(0.20, SINGLE_MAX)
                elif score > 0.65:
                    return min(0.15, SINGLE_MAX)
                else:
                    return min(0.10, SINGLE_MAX)
            elif market_env == "bear":
                # 熊市保守：只买最强的，仓位很小
                if score > 0.90:
                    return min(0.10, SINGLE_MAX)  # 熊市最强股票 10%
                elif score > 0.85:
                    return min(0.06, SINGLE_MAX)  # 熊市次强 6%
                else:
                    return 0  # 熊市不买其他股票
            elif market_env == "sideways":
                # 震荡分散策略：单股仓位低，买更多股票
                if score > 0.80:
                    return min(0.10, SINGLE_MAX)  # 震荡市高分 10%
                elif score > 0.70:
                    return min(0.08, SINGLE_MAX)  # 震荡市中分 8%
                elif score > 0.60:
                    return min(0.06, SINGLE_MAX)  # 震荡市低分 6%
                else:
                    return min(0.04, SINGLE_MAX)  # 震荡市最低分 4%
            else:
                # 反弹/复苏：中性
                if score > 0.85:
                    return min(0.20, SINGLE_MAX)
                elif score > 0.75:
                    return min(0.15, SINGLE_MAX)
                elif score > 0.65:
                    return min(0.10, SINGLE_MAX)
                else:
                    return min(0.06, SINGLE_MAX)

    # 优先处理：抄底加仓（已有持仓 + 亏损 + 高评分）
    for stock in hold_candidates:
        symbol = stock["symbol"]
        if symbol in [s["symbol"] for s in decisions["sell"]]:
            continue  # 已计划卖出
        if symbol not in holdings:
            continue

        holding = holdings[symbol]
        cost_price = holding["cost"]
        current_price = stock["current_price"]
        pnl_pct = (current_price - cost_price) / cost_price

        # 只在亏损时考虑加仓（抄底）
        if pnl_pct >= 0:
            continue  # 盈利不加仓

        score = stock["score"]
        if score < 0.6:
            continue  # 评分太低不加仓

        if available_cash < 30000:
            break

        position_pct = get_position_pct(stock, is_adding_position=True, pnl_pct=pnl_pct)
        if position_pct <= 0:
            continue

        amount = min(available_cash * position_pct, 150000)  # 加仓上限 15 万
        shares = int(amount / current_price / 100) * 100

        if shares > 0:
            decisions["buy"].append(
                {
                    "symbol": symbol,
                    "name": stock["name"],
                    "shares": shares,
                    "price": current_price,
                    "amount": shares * current_price,
                    "probability": stock["probability"],
                    "score": score,
                    "position_pct": position_pct,
                    "reason": f"抄底加仓: 亏损{pnl_pct:.1%}, 评分{stock['probability']:.1%}, 加仓{position_pct:.0%}",
                }
            )
            available_cash -= shares * current_price

    # 新建仓（非持仓股票）
    # 震荡市买更多股票（最多10个），其他环境保持5个
    max_new_positions = 10 if market_env == "sideways" else 5

    for stock in buy_candidates[:max_new_positions]:
        # 跳过已在持仓中的股票（已在上面的抄底加仓处理）
        if stock["symbol"] in holdings:
            continue

        if available_cash < 30000:
            break

        position_pct = get_position_pct(stock)
        # 根据大盘环境调整最大金额
        if market_env == "bull":
            max_amount = 250000
        elif market_env == "bear":
            max_amount = 80000  # 熊市单股上限更低
        elif market_env == "sideways":
            max_amount = 100000  # 震荡市单股上限低，但可以买更多
        else:
            max_amount = 200000

        amount = min(available_cash * position_pct, max_amount)
        shares = int(amount / stock["current_price"] / 100) * 100

        if shares > 0:
            decisions["buy"].append(
                {
                    "symbol": stock["symbol"],
                    "name": stock["name"],
                    "shares": shares,
                    "price": stock["current_price"],
                    "amount": shares * stock["current_price"],
                    "probability": stock["probability"],
                    "score": stock["score"],
                    "position_pct": position_pct,
                    "reason": f"评分{stock['score']:.2f}(概率{stock['probability']:.1%}, 预测收益{stock['predicted_return']:.1f}%), 仓位{position_pct:.0%}",
                }
            )
            available_cash -= shares * stock["current_price"]

    # ========== 持有决策 ==========

    for stock in hold_candidates:
        if stock["symbol"] not in [s["symbol"] for s in decisions["sell"]]:
            decisions["hold"].append(
                {
                    "symbol": stock["symbol"],
                    "name": stock["name"],
                    "shares": holdings[stock["symbol"]]["shares"],
                    "price": stock["current_price"],
                    "probability": stock["probability"],
                    "score": stock["score"],
                    "reason": f"评分{stock['score']:.2f}, {stock['advice']}",
                }
            )

    return decisions


def update_portfolio(portfolio: Dict, decisions: Dict) -> Dict:
    """更新投资组合"""
    holdings = portfolio.get("holdings", {})
    trades = portfolio.get("trades", [])
    cash = portfolio.get("cash", 0)

    today = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M")

    # 先执行卖出，获得现金
    for sell in decisions["sell"]:
        symbol = sell["symbol"]
        if symbol in holdings:
            holding = holdings[symbol]
            cash += sell["amount"]
            trades.append(
                {
                    "date": today,
                    "time": time_str,
                    "type": "sell",
                    "code": symbol,
                    "name": sell["name"],
                    "shares": sell["shares"],
                    "price": sell["price"],
                    "amount": sell["amount"],
                    "reason": sell["reason"],
                }
            )
            del holdings[symbol]

    # 执行买入（严格校验现金）
    for buy in decisions["buy"]:
        # 【关键】现金校验：不允许孖展交易
        if cash < buy["amount"]:
            print(
                f"  ⚠️ 现金不足，跳过买入 {buy['symbol']} {buy['name']} (需要 ¥{buy['amount']:,.0f}，可用 ¥{cash:,.0f})"
            )
            continue

        symbol = buy["symbol"]
        if symbol in holdings:
            # 加仓
            old_shares = holdings[symbol]["shares"]
            old_cost = holdings[symbol]["cost"]
            new_shares = old_shares + buy["shares"]
            new_cost = (
                old_cost * old_shares + buy["price"] * buy["shares"]
            ) / new_shares

            holdings[symbol]["shares"] = new_shares
            holdings[symbol]["cost"] = new_cost
        else:
            # 新建仓
            holdings[symbol] = {
                "name": buy["name"],
                "shares": buy["shares"],
                "cost": buy["price"],
                "current_price": buy["price"],
            }

        cash -= buy["amount"]
        trades.append(
            {
                "date": today,
                "time": time_str,
                "type": "buy",
                "code": symbol,
                "name": buy["name"],
                "shares": buy["shares"],
                "price": buy["price"],
                "amount": buy["amount"],
                "reason": buy["reason"],
            }
        )

    # 更新持仓现价
    for symbol in holdings:
        try:
            current = get_stock_data(symbol, "A股", holdings[symbol]["name"])
            if current:
                holdings[symbol]["current_price"] = current.get(
                    "current_price", holdings[symbol]["cost"]
                )
        except Exception as e:
            print(f"  ⚠️ 更新 {symbol} 价格失败: {e}")

    # 计算总资产
    holdings_value = sum(
        h["shares"] * h.get("current_price", h["cost"]) for h in holdings.values()
    )
    total_value = cash + holdings_value
    pnl = total_value - portfolio["initial_capital"]
    pnl_pct = pnl / portfolio["initial_capital"] * 100

    # 记录每日净值
    portfolio["daily_values"].append(
        {
            "date": today,
            "time": time_str,
            "cash": round(cash, 2),
            "holdings_value": round(holdings_value, 2),
            "total_value": round(total_value, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
        }
    )

    portfolio["cash"] = round(cash, 2)
    portfolio["holdings"] = holdings
    portfolio["trades"] = trades
    portfolio["holdings_value"] = round(holdings_value, 2)
    portfolio["total_value"] = round(total_value, 2)
    portfolio["pnl"] = round(pnl, 2)
    portfolio["pnl_pct"] = round(pnl_pct, 2)

    return portfolio


def print_report(decisions: Dict, portfolio: Dict):
    """打印报告"""
    print("\n" + "=" * 80)
    print("📊 Topaz 每日投资决策报告")
    print("=" * 80)
    print(f"报告时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # 市场环境信息
    regime = decisions.get("market_regime", "sideways")
    confidence = decisions.get("model_confidence", 0.5)
    adv_ratio = decisions.get("advance_ratio", 0.5)

    print(f"\n📈 市场环境")
    print(
        f"  环境: {regime} | 模型置信度: {confidence:.0%} | 上涨比例: {adv_ratio:.1%}"
    )

    # 环境效果说明
    regime_effect = {
        "recovery": "✅ 最佳环境（IC=0.15, Spread=4.85%）",
        "pullback": "✅ 有效环境（IC=0.11, Spread=3.69%）",
        "bull": "✅ 有效环境（IC=0.10, Spread=4.61%）",
        "bear": "⚠️ 熊市保守（IC=0.06）- 20%仓位，严格止损4%，只买最强",
        "sideways": "📊 震荡分散（IC=0.03）- 50%仓位，diversify多股，快速止盈",
    }
    effect_note = regime_effect.get(regime, "未知")
    print(f"  {effect_note}")

    # 动态仓位建议（风控改进）
    recommended_position = decisions.get("recommended_position", 0.50)
    print(f"  📌 建议最大仓位: {recommended_position:.0%}")

    all_results = decisions.get("all_results", [])
    if all_results:
        sorted_by_prob = sorted(
            all_results, key=lambda x: x["probability"], reverse=True
        )

        print("\n🟢 Top 5 建议买入（最高概率）")
        print("-" * 80)
        for i, stock in enumerate(sorted_by_prob[:5], 1):
            print(
                f"  #{i} {stock['symbol']} {stock['name']}: 概率 {stock['probability']:.1%} | 预期收益 {stock['predicted_return']:+.1f}%"
            )

        print("\n🔴 Bottom 5 建议回避（最低概率）")
        print("-" * 80)
        for i, stock in enumerate(sorted_by_prob[-5:][::-1], 1):
            print(
                f"  #{i} {stock['symbol']} {stock['name']}: 概率 {stock['probability']:.1%} | 预期收益 {stock['predicted_return']:+.1f}%"
            )

        holdings = portfolio.get("holdings", {})
        if holdings:
            low_prob_holdings = [
                r
                for r in all_results
                if r["symbol"] in holdings and r["probability"] < 0.40
            ]
            if low_prob_holdings:
                print("\n⚠️ 持仓低概率警告（<40%）")
                print("-" * 80)
                for stock in low_prob_holdings:
                    print(
                        f"  {stock['symbol']} {stock['name']}: 概率 {stock['probability']:.1%} | 建议 {stock['advice']}"
                    )

    if "market_info" in decisions:
        mi = decisions["market_info"]
        print(f"\n📈 大盘环境")
        print(f"  沪深300: {mi['index_price']:.2f} ({mi['index_change']:+.2f}%)")
        print(f"  上涨比例: {mi['advance_ratio']:.1%}")
        print(f"  环境判断: {mi['environment']} - {mi['description']}")

    # 关注列表分析
    if decisions.get("watchlist"):
        print("\n📋 关注股票分析")
        print("-" * 80)
        # 按概率排序
        watchlist_sorted = sorted(
            decisions["watchlist"], key=lambda x: x["probability"], reverse=True
        )
        for stock in watchlist_sorted[:10]:  # 只显示前10只
            prob = stock["probability"]
            prob_str = f"{prob:.1%}"
            if prob >= 0.60:
                emoji = "🟢"  # 建议买入
            elif prob >= 0.50:
                emoji = "🟡"  # 建议持有
            elif prob >= 0.40:
                emoji = "🟠"  # 建议观望
            else:
                emoji = "🔴"  # 建议回避
            print(
                f"  {emoji} {stock['symbol']} {stock['name']}: {prob_str} - {stock['advice']}"
            )
        if len(watchlist_sorted) > 10:
            print(f"  ... 还有 {len(watchlist_sorted) - 10} 只关注股票")

    # 买入决策
    if decisions["buy"]:
        print("\n✅ 建议买入")
        for buy in decisions["buy"]:
            print(
                f"  {buy['symbol']} {buy['name']}: {buy['shares']}股 @ ¥{buy['price']:.2f} = ¥{buy['amount']:,.0f}"
            )
            print(f"    理由：{buy['reason']}")

    # 卖出决策
    if decisions["sell"]:
        print("\n❌ 建议卖出")
        for sell in decisions["sell"]:
            print(
                f"  {sell['symbol']} {sell['name']}: {sell['shares']}股 @ ¥{sell['price']:.2f} = ¥{sell['amount']:,.0f}"
            )
            print(f"    理由：{sell['reason']}")

    # 持有决策
    if decisions["hold"]:
        print("\n📌 继续持有")
        for hold in decisions["hold"]:
            print(
                f"  {hold['symbol']} {hold['name']}: {hold['shares']}股 @ ¥{hold['price']:.2f}"
            )
            print(f"    理由：{hold['reason']}")

    # 持仓汇总
    print("\n" + "=" * 80)
    print("💼 持仓汇总")
    print("=" * 80)
    print(f"现金：¥{portfolio['cash']:,.2f}")
    print(f"持仓市值：¥{portfolio['holdings_value']:,.2f}")
    print(f"总资产：¥{portfolio['total_value']:,.2f}")
    print(f"累计盈亏：¥{portfolio['pnl']:,.2f} ({portfolio['pnl_pct']:+.2f}%)")

    print("\n" + "=" * 80)
    print("风险提示：本分析仅供参考，不构成投资建议。市场有风险，投资需谨慎。")
    print("=" * 80)


def find_stock_list_file(base_dir: str, prefix: str) -> str:
    """查找股票列表文件"""
    for f in os.listdir(base_dir):
        if f.startswith(prefix) and f.endswith(".md"):
            return os.path.join(base_dir, f)
    return None


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Topaz 每日投资决策系统")
    parser.add_argument(
        "--preview", action="store_true", help="预告模式：只生成决策建议，不执行交易"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=True,
        help="执行模式：分析并执行交易（默认）",
    )
    parser.add_argument("--csi300", action="store_true", help="使用沪深300专用模型")
    args = parser.parse_args()

    # 判断运行模式
    is_preview_mode = args.preview
    use_csi300 = args.csi300

    base_dir = os.path.dirname(os.path.abspath(__file__))
    portfolio_file = os.path.join(base_dir, "virtual_portfolio.json")

    # 如果使用沪深300模型，使用对应的股票列表
    # 默认都使用沪深300全量列表进行分析
    stock_list_file = os.path.join(base_dir, "csi300_stock_list.md")
    if not os.path.exists(stock_list_file):
        print(f"❌ 未找到沪深300股票列表文件")
        return

    # 加载投资组合
    print("📂 加载投资组合...")
    portfolio = load_portfolio(portfolio_file)

    # 分析股票
    if use_csi300:
        print("📈 分析沪深300成分股（使用分组模型）...")
        analysis_data = analyze_stocks(
            stock_list_file, use_csi300_model=True, portfolio=portfolio
        )
    else:
        print("📈 分析沪深300成分股（使用默认模型）...")
        analysis_data = analyze_stocks(stock_list_file, portfolio=portfolio)

    results = analysis_data["all_results"]
    watchlist_results = analysis_data["watchlist_results"]
    print(f"  完成 {len(results)} 只股票分析")

    # 生成决策
    print("🤖 生成投资决策...")
    decisions = generate_decision(results, portfolio, watchlist_results)

    # 根据模式处理
    if is_preview_mode:
        print("\n⚠️ [预告模式] 以下决策仅供参考，不执行实际交易")
        print_report(decisions, portfolio)
        print("\n📌 预告完成，投资组合未更新")
    else:
        # 执行模式：更新投资组合
        print("💼 更新投资组合...")
        portfolio = update_portfolio(portfolio, decisions)

        # 保存
        save_portfolio(portfolio, portfolio_file)
        print(f"✓ 投资组合已保存：{portfolio_file}")

        # 打印报告
        print_report(decisions, portfolio)


if __name__ == "__main__":
    main()
