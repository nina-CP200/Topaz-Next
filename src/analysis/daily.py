#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Topaz 每日股票分析系统
================================================================================

【模块说明】
本模块是 Topaz 量化交易系统的核心决策引擎，负责：
1. 判断当前市场环境（牛市/熊市/震荡/反弹/回调）
2. 根据市场环境动态调整建议仓位
3. 加载预训练的机器学习模型（支持按市场环境分组加载）
4. 对股票列表进行批量分析，计算上涨概率
5. 生成投资建议（买入/持有/观望/回避）

【使用方法】
命令行执行：
    python daily_decision.py              # 使用默认模型分析
    python daily_decision.py --csi300     # 使用沪深300专用分组模型

【依赖文件】
- csi300_stocks.json          : 沪深300成分股列表（股票代码、名称、行业分类）
- ensemble_model.pkl          : 默认集成模型文件
- ensemble_model_regime_based.pkl  : 按市场环境分组的模型文件（需使用 --csi300 参数）

【输出说明】
控制台输出包含：
- 市场环境判断（bull/bear/sideways/recovery/pullback）
- 模型置信度（0.5-0.9）
- 建议最大仓位（20%-95%）
- Top 5 建议买入股票（按上涨概率排序）
- Bottom 5 建议回避股票

【配置参数说明】
--------------------------------------------------------------------------------
市场环境判断阈值（第51-65行）：
- adv_ratio > 0.55 且 ret_20d > 0      → bull（牛市），置信度 0.7
- adv_ratio > 0.55 且 ret_20d < -0.02   → recovery（反弹），置信度 0.9
- adv_ratio < 0.45 且 ret_20d < -0.02   → bear（熊市），置信度 0.6
- adv_ratio < 0.45 且 ret_20d > 0.02    → pullback（回调），置信度 0.8
- 其他情况                                → sideways（震荡），置信度 0.5

仓位建议映射（第71-78行）：
- recovery  → 95%  （反弹期，高仓位）
- pullback  → 80%  （回调期，较高仓位）
- bull      → 70%  （牛市期，中高仓位）
- sideways  → 50%  （震荡期，中性仓位）
- bear      → 20%  （熊市期，低仓位）

风险等级划分阈值（第144-151行、第214-221行）：
- 概率 >= 0.65 → 低风险
- 概率 >= 0.50 → 中风险
- 概率 >= 0.40 → 高风险
- 概率 <  0.40 → 极高风险

投资建议划分阈值（第153-160行、第223-230行）：
- 概率 >= 0.60 → 建议买入
- 概率 >= 0.50 → 建议持有
- 概率 >= 0.40 → 建议观望
- 概率 <  0.40 → 建议回避

【预期收益计算公式】
expected_return = (probability - 0.5) * 20
即：概率 0.6 对应预期收益 2%，概率 0.7 对应 4%，以此类推

【注意事项】
1. 模型文件需预先训练生成（使用 train_ensemble_model.py）
2. 本分析不构成投资建议，仅供参考
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.models.ensemble import EnsembleModel
from src.features.engineer import FeatureEngineer
from src.data.api import get_history_data, get_stock_data
from src.utils.utils import load_stock_list_from_json
from src.data.cache import CacheManager
from src.data.market import (
    get_index_data,
    get_index_history,
    get_market_sentiment,
)


def analyze_stocks(stock_list_file: str, use_csi300_model: bool = False) -> Dict:
    """
    分析股票列表，生成投资建议
    
    【功能说明】
    这是本模块的核心函数，完成以下任务：
    1. 判断市场环境并计算模型置信度
    2. 加载对应的机器学习模型
    3. 批量计算股票特征并预测上涨概率
    4. 生成风险等级和投资建议
    
    【参数说明】
    Args:
        stock_list_file: 股票列表 JSON 文件路径
            - 文件格式：[["000001.SZ", "平安银行", "金融"], ...]
            - 每个元素包含：股票代码、股票名称、行业分类
        
        use_csi300_model: 是否使用沪深300分组模型
            - False（默认）：使用 ensemble_model.pkl 单一模型
            - True：使用 ensemble_model_regime_based.pkl，根据市场环境选择对应模型
    
    【返回值说明】
    Returns:
        Dict: 分析结果字典，包含以下字段：
            - all_results: List[Dict] - 所有股票的分析结果
                每个元素包含：
                * symbol: 股票代码
                * name: 股票名称
                * current_price: 当前价格
                * change_pct: 当日涨跌幅(%)
                * probability: 模型预测的上涨概率(0-1)
                * predicted_return: 预期收益率(%)
                * risk_level: 风险等级（低风险/中风险/高风险/极高风险）
                * advice: 投资建议（建议买入/建议持有/建议观望/建议回避）
                * model_confidence: 模型置信度
            - market_regime: str - 市场环境（bull/bear/sideways/recovery/pullback）
            - model_confidence: float - 模型置信度(0.5-0.9)
            - advance_ratio: float - 市场上涨比例
            - recommended_position: float - 建议最大仓位(0-1)
    
    【调用示例】
    >>> # 使用默认模型分析
    >>> result = analyze_stocks("csi300_stocks.json")
    >>> print(f"市场环境: {result['market_regime']}")
    >>> for stock in result['all_results'][:5]:
    ...     print(f"{stock['symbol']}: {stock['advice']}")
    
    >>> # 使用分组模型分析
    >>> result = analyze_stocks("csi300_stocks.json", use_csi300_model=True)
    
    【注意事项】
    - 首次运行较慢（需计算特征），后续会使用缓存加速
    - 并行处理使用 8 个线程，可根据 CPU 核心数调整
    """
    import joblib

    # 初始化缓存管理器，用于存储特征计算结果，避免重复计算
    cache = CacheManager()
    
    # 获取当前日期，用于缓存键值
    today = datetime.now().strftime("%Y-%m-%d")

    # -------------------------------------------------------------------------
    # 第一步：获取市场数据，用于判断市场环境
    # -------------------------------------------------------------------------
    # 获取沪深300指数最近30天的历史数据
    index_history_30 = get_index_history("000300.SH", days=30)
    
    # 获取市场情绪数据（包含上涨股票占比等信息）
    sentiment = get_market_sentiment()

    # -------------------------------------------------------------------------
    # 第二步：计算市场环境指标
    # -------------------------------------------------------------------------
    # adv_ratio: 市场上涨股票占比，反映市场整体热度
    # 默认值 0.5 表示中性市场
    adv_ratio = 0.5
    if sentiment:
        adv_ratio = sentiment.get("advance_ratio", 0.5)

    # ret_20d: 沪深300指数最近20天的收益率
    # 用于判断市场短期趋势方向
    ret_20d = 0
    if index_history_30 is not None and len(index_history_30) >= 20:
        ret_20d = (
            index_history_30["close"].iloc[-1] / index_history_30["close"].iloc[-20] - 1
        )

    # -------------------------------------------------------------------------
    # 第三步：判断市场环境（核心逻辑）
    # -------------------------------------------------------------------------
    # 根据上涨比例和20日收益率两个维度判断市场环境
    # 这决定了后续使用哪个模型以及建议仓位
    if adv_ratio > 0.55 and ret_20d > 0:
        # 上涨比例高 + 正收益 = 牛市
        # 牛市中追高需谨慎，置信度设为 0.7
        detailed_regime = "bull"
        model_confidence = 0.7
    elif adv_ratio > 0.55 and ret_20d < -0.02:
        # 上涨比例高 + 负收益 = 反弹初期
        # 反弹信号较强，置信度设为 0.9（最高）
        detailed_regime = "recovery"
        model_confidence = 0.9
    elif adv_ratio < 0.45 and ret_20d < -0.02:
        # 上涨比例低 + 负收益 = 熊市
        # 熊市中模型预测可能失准，置信度设为 0.6
        detailed_regime = "bear"
        model_confidence = 0.6
    elif adv_ratio < 0.45 and ret_20d > 0.02:
        # 上涨比例低 + 正收益 = 回调整理
        # 回调可能是买入机会，置信度设为 0.8
        detailed_regime = "pullback"
        model_confidence = 0.8
    else:
        # 其他情况 = 震荡市
        # 震荡市方向不明，置信度设为 0.5（最低）
        detailed_regime = "sideways"
        model_confidence = 0.5

    # 输出市场环境判断结果
    print(f"\n📊 市场环境: {detailed_regime}")
    print(f"   上涨比例: {adv_ratio:.1%}, 20日收益: {ret_20d * 100:.1f}%")
    print(f"   模型置信度: {model_confidence:.0%}")

    # -------------------------------------------------------------------------
    # 第四步：根据市场环境确定建议仓位
    # -------------------------------------------------------------------------
    # 不同市场环境下的最大持仓建议
    # 用户可根据自身风险偏好调整这些阈值
    position_advice = {
        "recovery": 0.95,   # 反弹期：激进建仓
        "pullback": 0.80,   # 回调期：积极建仓
        "bull": 0.70,       # 牛市期：适度持仓
        "bear": 0.20,       # 熊市期：轻仓观望
        "sideways": 0.50,   # 震荡期：半仓操作
    }
    recommended_position = position_advice.get(detailed_regime, 0.50)
    print(f"   📌 建议最大仓位: {recommended_position:.0%}")

    # -------------------------------------------------------------------------
    # 第五步：加载机器学习模型
    # -------------------------------------------------------------------------
    ensemble = None
    
    if use_csi300_model:
        # 使用分组模型：根据当前市场环境选择对应训练好的模型
        # 这种方式可以提高预测精度，因为不同市场环境下股票表现规律不同
        if os.path.exists("data/models/ensemble_model_regime_based.pkl"):
            model_data = joblib.load("data/models/ensemble_model_regime_based.pkl")
            models_by_regime = model_data.get("models_by_regime", {})
            
            # 尝试加载当前市场环境对应的模型
            if detailed_regime in models_by_regime:
                selected = models_by_regime[detailed_regime]
                print(f"📦 已加载 {detailed_regime} 环境模型")
            else:
                # 如果没有对应环境的模型，使用兜底模型
                selected = models_by_regime.get("sideways", models_by_regime.get("bull", {}))
                print(f"📦 已加载兜底模型")
            
            if selected:
                ensemble = {
                    "model": selected["model"],           # 训练好的模型对象
                    "scaler": selected["scaler"],          # 特征标准化器
                    "feature_cols": selected["features"], # 特征列名列表
                }
        else:
            print("⚠️ 未找到分组模型")
            return {"all_results": [], "market_regime": detailed_regime}
    else:
        # 使用默认单一模型：适用于通用场景
        if os.path.exists("data/models/ensemble_model.pkl"):
            model_data = joblib.load("data/models/ensemble_model.pkl")
            
            # 处理两种模型格式
            if "scaler" in model_data:
                # 新格式：scaler 在模型文件中
                ensemble = {
                    "model": model_data.get("models", {}).get("lightgbm", model_data.get("model")),
                    "scaler": model_data["scaler"],
                    "feature_cols": model_data["feature_cols"],
                }
            else:
                # 兼容旧格式：scaler 在单独文件
                ensemble = {
                    "model": model_data.get("models", {}).get("lightgbm", model_data.get("model")),
                    "scaler": joblib.load("data/models/ensemble_scaler.pkl") if os.path.exists("data/models/ensemble_scaler.pkl") else None,
                    "feature_cols": model_data["feature_cols"],
                }
                if ensemble["scaler"] is None:
                    from sklearn.preprocessing import StandardScaler
                    ensemble["scaler"] = StandardScaler()
            
            print(f"📦 已加载默认 data/models/ensemble_model.pkl")
        else:
            print("⚠️ 未找到默认模型")
            return {"all_results": [], "market_regime": detailed_regime}

    # -------------------------------------------------------------------------
    # 第六步：加载股票列表并准备特征工程
    # -------------------------------------------------------------------------
    fe = FeatureEngineer()
    stocks = load_stock_list_from_json(stock_list_file)
    print(f"📋 分析股票: {len(stocks)} 只")

    results = []

    # 获取沪深300指数历史数据，用于计算指数相关因子
    # 这些因子可以帮助模型理解个股相对于大盘的表现
    index_history = cache.get_index_cache("000300.SH", 60)
    if index_history is None:
        index_history = get_index_history("000300.SH", days=300)
        if index_history is not None:
            cache.set_index_cache("000300.SH", 60, index_history)
    if index_history is None:
        print("  ⚠️ 无法获取指数数据")
    else:
        print(f"  ✓ 获取 {len(index_history)} 天指数数据")

    # -------------------------------------------------------------------------
    # 第七步：尝试使用缓存的特征数据（加速处理）
    # -------------------------------------------------------------------------
    # 如果当天已经计算过特征，直接使用缓存结果
    # 这可以大幅减少计算时间，特别是对于大批量股票
    print("\n📦 尝试读取预计算特征缓存...")
    cached_results = []
    need_compute = []

    for symbol, name, category in stocks:
        # 尝试获取该股票的特征缓存
        cached_features = cache.get_feature_cache(symbol, today)
        
        if cached_features:
            # 缓存命中：直接使用特征进行预测
            feature_cols = ensemble["feature_cols"]
            
            # 构建特征向量，缺失特征用0填充
            X = np.array([[cached_features.get(f, 0) for f in feature_cols]])
            
            # 限制特征值范围，防止极端值影响预测
            X = np.clip(X, -1e10, 1e10)
            
            # 特征标准化（必须与训练时使用相同的 scaler）
            X_scaled = ensemble["scaler"].transform(X)
            
            # 预测上涨概率（predict_proba 返回 [下跌概率, 上涨概率]）
            proba = ensemble["model"].predict_proba(X_scaled)[:, 1][0]
            
            # 计算预期收益：概率偏离 0.5 越多，预期收益越大
            # 公式：(概率 - 0.5) * 20 表示概率每增加 0.1，预期收益增加 2%
            expected_return = (proba - 0.5) * 20

            # 判断风险等级（基于上涨概率）
            if proba >= 0.65:
                risk_level = "低风险"
            elif proba >= 0.50:
                risk_level = "中风险"
            elif proba >= 0.40:
                risk_level = "高风险"
            else:
                risk_level = "极高风险"

            # 判断投资建议（基于上涨概率）
            if proba >= 0.60:
                advice = "建议买入"
            elif proba >= 0.50:
                advice = "建议持有"
            elif proba >= 0.40:
                advice = "建议观望"
            else:
                advice = "建议回避"

            # 如果模型置信度高（>= 0.8），加强建议语气
            if model_confidence >= 0.8:
                advice = advice.replace("建议", "强烈")

            cached_results.append({
                "symbol": symbol,
                "name": name,
                "current_price": cached_features.get("close", 0),
                "change_pct": cached_features.get("return_1d", 0) * 100,
                "probability": proba,
                "predicted_return": expected_return,
                "risk_level": risk_level,
                "advice": advice,
                "model_confidence": model_confidence,
            })
        else:
            # 缓存未命中：加入待计算列表
            need_compute.append((symbol, name, category))

    results.extend(cached_results)
    print(f"  ✓ 缓存命中: {len(cached_results)} 只，需计算: {len(need_compute)} 只")

    # -------------------------------------------------------------------------
    # 第八步：并行计算缺失的股票特征
    # -------------------------------------------------------------------------
    # 对于缓存未命中的股票，需要实时计算特征
    # 使用多线程并行处理以提高效率
    if need_compute:
        print(f"\n🔄 并行计算 {len(need_compute)} 只股票...")

        def analyze_one(args):
            """
            分析单只股票的内部函数
            
            Args:
                args: 元组 (symbol, name, category)
            
            Returns:
                Dict: 股票分析结果，失败返回 None
            """
            symbol, name, category = args
            try:
                # 获取股票历史数据（300天），用于计算技术指标
                history = get_history_data(symbol, "A股", days=300)
                
                # 获取股票当前实时数据（价格、涨跌幅等）
                current = get_stock_data(symbol, "A股", name)
                
                if history is None or not current:
                    return None

                # 为历史数据添加股票代码标识
                history["code"] = symbol
                
                # 生成技术因子特征（动量、波动率、成交量等）
                df_features = fe.generate_all_features(history)
                
                # 添加指数相关因子（相对强弱、行业表现等）
                if index_history is not None:
                    df_features = fe.add_index_factors(df_features, index_history)
                
                # 填充缺失值为0，确保特征向量完整
                df_features = df_features.fillna(0)

                # 获取模型所需的特征列
                feature_cols = ensemble["feature_cols"]
                
                # 检查并填充缺失的特征列
                missing = [f for f in feature_cols if f not in df_features.columns]
                for f in missing:
                    df_features[f] = 0

                # 提取最新一行的特征值
                latest = df_features.iloc[-1:][feature_cols]
                X = latest.values
                
                # 限制特征值范围，防止极端值
                X = np.clip(X, -1e10, 1e10)
                
                # 特征标准化
                X_scaled = ensemble["scaler"].transform(X)
                
                # 预测上涨概率
                proba = ensemble["model"].predict_proba(X_scaled)[:, 1][0]

                # 缓存计算结果，避免重复计算（移除代码中的 .SH/.SZ 后缀以统一缓存键）
                clean_symbol = symbol.replace('.SH', '').replace('.SZ', '')
                cache.set_feature_cache(clean_symbol, df_features.iloc[-1].to_dict(), today)

                # 计算预期收益
                expected_return = (proba - 0.5) * 20

                # 判断风险等级
                if proba >= 0.65:
                    risk_level = "低风险"
                elif proba >= 0.50:
                    risk_level = "中风险"
                elif proba >= 0.40:
                    risk_level = "高风险"
                else:
                    risk_level = "极高风险"

                # 判断投资建议
                if proba >= 0.60:
                    advice = "建议买入"
                elif proba >= 0.50:
                    advice = "建议持有"
                elif proba >= 0.40:
                    advice = "建议观望"
                else:
                    advice = "建议回避"

                # 高置信度时加强建议
                if model_confidence >= 0.8:
                    advice = advice.replace("建议", "强烈")

                return {
                    "symbol": symbol,
                    "name": name,
                    "current_price": current.get("current_price", 0),
                    "change_pct": current.get("change", 0),
                    "probability": proba,
                    "predicted_return": expected_return,
                    "risk_level": risk_level,
                    "advice": advice,
                    "model_confidence": model_confidence,
                }
            except Exception:
                return None

        # 使用线程池并行处理（8个工作线程）
        # max_workers 可根据 CPU 核心数调整
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(analyze_one, args): args[0] for args in need_compute}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        print(f"  ✓ 并行计算完成")

    # -------------------------------------------------------------------------
    # 返回完整的分析结果
    # -------------------------------------------------------------------------
    return {
        "all_results": results,           # 所有股票分析结果
        "market_regime": detailed_regime,  # 市场环境
        "model_confidence": model_confidence,  # 模型置信度
        "advance_ratio": adv_ratio,        # 市场上涨比例
        "recommended_position": recommended_position,  # 建议仓位
    }


def print_report(analysis_data: Dict):
    """
    打印分析报告到控制台
    
    【功能说明】
    将分析结果格式化输出为可读的报告，包括：
    - 市场环境概述
    - Top 5 推荐买入股票
    - Bottom 5 建议回避股票
    - 风险分布统计
    
    【参数说明】
    Args:
        analysis_data: 分析结果字典，由 analyze_stocks() 返回
            - all_results: List[Dict] - 所有股票分析结果
            - market_regime: str - 市场环境
            - model_confidence: float - 模型置信度
            - advance_ratio: float - 上涨比例
            - recommended_position: float - 建议仓位
    
    【返回值】
    Returns:
        None（直接打印到控制台）
    
    【调用示例】
    >>> result = analyze_stocks("csi300_stocks.json")
    >>> print_report(result)
    """
    # 打印报告头部
    print("\n" + "=" * 80)
    print("📊 Topaz 每日股票分析报告")
    print("=" * 80)
    print(f"报告时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # 提取市场环境信息
    regime = analysis_data.get("market_regime", "sideways")
    confidence = analysis_data.get("model_confidence", 0.5)
    adv_ratio = analysis_data.get("advance_ratio", 0.5)
    recommended_position = analysis_data.get("recommended_position", 0.50)

    # 打印市场环境摘要
    print(f"\n📈 市场环境")
    print(f"  环境: {regime} | 模型置信度: {confidence:.0%} | 上涨比例: {adv_ratio:.1%}")
    print(f"  📌 建议最大仓位: {recommended_position:.0%}")

    # 打印股票分析结果
    results = analysis_data.get("all_results", [])
    if results:
        # 按上涨概率降序排序
        sorted_by_prob = sorted(results, key=lambda x: x["probability"], reverse=True)

        # 打印 Top 5 推荐买入股票（概率最高）
        print("\n🟢 Top 5 建议买入（最高概率）")
        print("-" * 80)
        for i, stock in enumerate(sorted_by_prob[:5], 1):
            print(f"  #{i} {stock['symbol']} {stock['name']}: 概率 {stock['probability']:.1%} | 预期收益 {stock['predicted_return']:+.1f}%")

        # 打印 Bottom 5 建议回避股票（概率最低）
        print("\n🔴 Bottom 5 建议回避（最低概率）")
        print("-" * 80)
        for i, stock in enumerate(sorted_by_prob[-5:][::-1], 1):
            print(f"  #{i} {stock['symbol']} {stock['name']}: 概率 {stock['probability']:.1%} | 预期收益 {stock['predicted_return']:+.1f}%")

        # 统计投资建议分布
        buy_count = len([r for r in results if r["advice"] in ["建议买入", "强烈建议买入"]])
        hold_count = len([r for r in results if r["advice"] in ["建议持有", "强烈建议持有"]])
        avoid_count = len([r for r in results if r["advice"] == "建议回避"])

        # 打印风险分布统计
        print(f"\n📊 风险分布: 低风险{len([r for r in results if r['risk_level']=='低风险'])} / 中风险{len([r for r in results if r['risk_level']=='中风险'])} / 高风险{len([r for r in results if r['risk_level']=='高风险'])}")

    # 打印报告尾部和风险提示
    print("\n" + "=" * 80)
    print("风险提示：本分析仅供参考，不构成投资建议。市场有风险，投资需谨慎。")
    print("=" * 80)


def main():
    """
    主函数：程序入口
    
    【功能说明】
    1. 解析命令行参数
    2. 加载股票列表
    3. 执行股票分析
    4. 发送报告到 Slack（可选）
    5. 打印分析报告
    
    【命令行参数】
    --csi300: 使用沪深300专用分组模型（按市场环境加载不同模型）
    
    【调用示例】
    命令行执行：
        python daily_decision.py              # 默认模型
        python daily_decision.py --csi300     # 分组模型
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Topaz 每日股票分析系统")
    parser.add_argument("--csi300", action="store_true", help="使用沪深300专用模型")
    args = parser.parse_args()

    # 定位股票列表文件路径
    stock_list_file = "config/csi300_stocks.json"
    
    # 检查股票列表文件是否存在
    if not os.path.exists(stock_list_file):
        print(f"❌ 未找到沪深300股票列表文件")
        return

    # 执行股票分析
    print("📈 分析沪深300成分股...")
    analysis_data = analyze_stocks(stock_list_file, use_csi300_model=args.csi300)

    results = analysis_data["all_results"]
    print(f"  完成 {len(results)} 只股票分析")

    # 尝试发送报告到 Slack（可选功能）
    try:
        from src.reports.sender import send_score_ranking
        slack_ok = send_score_ranking(
            results=results,
            market_regime=analysis_data.get("market_regime", "sideways"),
            model_confidence=analysis_data.get("model_confidence", 0.5),
            advance_ratio=analysis_data.get("advance_ratio", 0.5),
        )
        if slack_ok:
            print("  ✓ 评分排名已发送至 Slack")
    except Exception as e:
        print(f"  ⚠️ 发送报告失败: {e}")

    # 保存分析结果到 JSON 文件（供查询脚本使用）
    if results:
        import json
        from datetime import datetime
        
        # 添加排名信息
        sorted_results = sorted(results, key=lambda x: x.get("probability", 0), reverse=True)
        for i, r in enumerate(sorted_results):
            r["rank"] = i + 1
        
        output_file = "data/raw/latest_analysis_results.json"
        output_data = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "market_regime": analysis_data.get("market_regime", "sideways"),
            "model_confidence": analysis_data.get("model_confidence", 0.5),
            "advance_ratio": analysis_data.get("advance_ratio", 0.5),
            "recommended_position": analysis_data.get("recommended_position", 0.5),
            "total_stocks": len(results),
            "results": sorted_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ 分析结果已保存: {output_file}")

    # 打印分析报告
    print_report(analysis_data)


if __name__ == "__main__":
    main()