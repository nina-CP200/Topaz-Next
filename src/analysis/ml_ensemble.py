#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
QuantPilot 机器学习股票分析系统 - 集成模型版
================================================================================

【模块说明】
本模块是 QuantPilot 股票分析系统的核心预测引擎，使用预训练的集成机器学习模型
对股票未来收益进行预测。通过整合多种技术指标和基本面数据，为投资者提供量化
投资参考。

【使用的模型】
- Ensemble 集成模型 (LightGBM + Random Forest + GBDT)
- 验证准确率：60.19%
- AUC 指标：0.647

【分析流程】
1. 初始化阶段
   └── 加载预训练的集成模型 (EnsembleModel)
   └── 初始化特征工程器 (FeatureEngineer)

2. 数据获取阶段
   └── 从数据源获取股票历史数据 (默认60天)
   └── 获取股票实时行情数据

3. 特征工程阶段
   └── 使用 FeatureEngineer 生成技术指标特征
   └── 补充模型所需但特征工程未生成的特征（指数相关特征等）

4. 模型预测阶段
   └── 将特征向量输入集成模型
   └── 获取预测概率和分类结果
   └── 计算预期收益和风险等级

5. 结果输出阶段
   └── 打印每只股票的详细分析结果
   └── 输出市场整体汇总统计

【核心类说明】
- MLStockAnalyzer: 主分析器类，协调数据获取、特征生成和模型预测
- FeatureEngineer: 特征工程类，生成技术指标特征
- EnsembleModel: 集成模型类，封装多个基模型的预测逻辑

【依赖模块】
- quantpilot_data_api: 数据获取接口
- ensemble_model: 集成模型定义
- feature_engineer: 特征工程模块
- utils: 工具函数（股票列表加载等）

【使用场景】
1. 日常选股：对沪深300成分股进行批量分析，筛选潜在投资标的
2. 风险评估：评估持仓股票的风险等级，辅助仓位管理
3. 定期复盘：定期运行分析，跟踪市场热点和风险变化
4. 策略验证：验证技术指标组合对股票收益的预测能力

【注意事项】
- 本模块依赖于预训练模型文件，需确保模型文件存在
- 分析结果仅供参考，不构成投资建议
- 市场有风险，投资需谨慎

作者：QuantPilot Team
版本：v1.0
最后更新：2024年
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple


# 启动时检查虚拟环境
# ============================================================================
# 虚拟环境检查与激活
# ============================================================================
def check_and_activate_venv():
    """
    检查并激活 Python 虚拟环境
    
    功能说明：
    - 检查当前是否已在虚拟环境中运行
    - 如果不在虚拟环境中，尝试查找并激活 ~/myenv 或 ~/venv
    - 通过 os.execv 重新启动当前脚本，使用虚拟环境的 Python 解释器
    
    返回值：
    - True: 已在虚拟环境中或成功激活
    - False: 未找到可用的虚拟环境
    
    使用场景：
    - 确保脚本在正确的 Python 环境中运行
    - 避免因依赖缺失导致的运行错误
    """
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        return True

    script_dir = os.path.dirname(os.path.abspath(__file__))
    home_dir = os.path.expanduser("~")

    venv_candidates = [
        os.path.join(home_dir, "myenv"),
        os.path.join(home_dir, "venv"),
    ]

    for venv_path in venv_candidates:
        python_path = os.path.join(venv_path, "bin", "python")
        if os.path.exists(python_path):
            print(f"🔄 检测到虚拟环境：{venv_path}")
            os.execv(python_path, [python_path] + sys.argv)

    return False


check_and_activate_venv()
from src.data.api import get_stock_data, get_history_data
from src.utils.utils import load_stock_list_from_json
from src.models.ensemble import EnsembleModel
from src.features.engineer import FeatureEngineer


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建技术分析特征
    
    功能说明：
    基于股票历史数据生成一系列技术指标特征，用于机器学习模型输入。
    注意：此函数为辅助函数，主要特征由 FeatureEngineer 类生成。
    
    参数：
    - df: pd.DataFrame, 包含股票历史数据，需包含以下列：
        - close: 收盘价
        - high: 最高价
        - low: 最低价
        - volume: 成交量
    
    返回值：
    - pd.DataFrame: 添加了技术指标特征的数据框
    
    生成的特征类别：
    1. 移动平均线：ma5, ma10, ma20
    2. MACD 指标：macd, macd_signal, macd_hist
    3. RSI 指标：rsi (14日相对强弱指数)
    4. 布林带：bb_middle, bb_std, bb_upper, bb_lower, bb_position
    5. 成交量特征：volume_ma5, volume_ratio
    6. 价格动量：momentum_5d, momentum_10d, momentum_20d
    7. 波动率：volatility_5d, volatility_10d, volatility_20d
    8. 价格位置：high_low_position
    9. 收益率：returns
    """
    data = df.copy()

    # 移动平均线
    data["ma5"] = data["close"].rolling(5).mean()
    data["ma10"] = data["close"].rolling(10).mean()
    data["ma20"] = data["close"].rolling(20).mean()

    # MACD
    exp1 = data["close"].ewm(span=12, adjust=False).mean()
    exp2 = data["close"].ewm(span=26, adjust=False).mean()
    data["macd"] = exp1 - exp2
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    # RSI
    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["rsi"] = 100 - (100 / (1 + rs))

    # 布林带
    data["bb_middle"] = data["close"].rolling(20).mean()
    data["bb_std"] = data["close"].rolling(20).std()
    data["bb_upper"] = data["bb_middle"] + 2 * data["bb_std"]
    data["bb_lower"] = data["bb_middle"] - 2 * data["bb_std"]
    data["bb_position"] = (data["close"] - data["bb_lower"]) / (
        data["bb_upper"] - data["bb_lower"] + 0.001
    )

    # 成交量特征
    data["volume_ma5"] = data["volume"].rolling(5).mean()
    data["volume_ratio"] = data["volume"] / data["volume_ma5"]

    # 价格动量
    data["momentum_5d"] = data["close"].pct_change(5)
    data["momentum_10d"] = data["close"].pct_change(10)
    data["momentum_20d"] = data["close"].pct_change(20)

    # 波动率
    data["volatility_5d"] = data["close"].pct_change().rolling(5).std()
    data["volatility_10d"] = data["close"].pct_change().rolling(10).std()
    data["volatility_20d"] = data["close"].pct_change().rolling(20).std()

    # 价格位置
    data["high_low_position"] = (data["close"] - data["low"]) / (
        data["high"] - data["low"] + 0.001
    )

    # 收益率
    data["returns"] = data["close"].pct_change()

    return data


class MLStockAnalyzer:
    """
    ============================================================================
    基于预训练集成模型的股票分析器
    ============================================================================
    
    【类功能说明】
    本类是股票分析系统的核心组件，负责协调数据获取、特征工程和模型预测
    的完整流程，为股票投资提供量化分析结果。
    
    【初始化参数说明】
    - history_days: int, 默认60
        获取历史数据的天数，用于计算技术指标和生成特征
        建议值：30-120天，太短可能导致特征不稳定，太长增加计算开销
    
    - batch: int, 默认0
        批量处理编号，用于大规模股票列表的分批处理
        当设置为 1,2,3,4 时，会将股票列表分成4批，只处理指定批次
    
    - limit: int, 默认0
        每批处理的股票数量限制，0表示不限制
        用于测试或小规模分析场景
    
    【核心属性说明】
    - predictor: EnsembleModel
        预训练的集成模型实例，用于执行预测
    
    - feature_engineer: FeatureEngineer
        特征工程器实例，用于生成技术指标特征
    
    - history_data: Dict[str, pd.DataFrame]
        存储各股票的历史数据，key为股票代码
    
    - current_data: Dict[str, Dict]
        存储各股票的实时行情数据，key为股票代码
    
    - results: List[Dict]
        存储分析结果列表，每只股票一个结果字典
    
    【分析流程说明】
    1. 初始化 → 加载预训练模型和特征工程器
    2. run() → 执行完整分析流程
       ├── 读取股票列表 (从 JSON 文件)
       ├── 获取历史数据 (fetch_history_data)
       ├── 获取实时数据 (fetch_current_data)
       └── 执行预测分析 (analyze_stock)
    3. print_results() → 输出分析结果
    
    【使用示例】
    >>> analyzer = MLStockAnalyzer(history_days=300)
    >>> analyzer.run("csi300_stocks.json")
    >>> analyzer.print_results()
    
    【注意事项】
    - 需要确保预训练模型文件存在
    - 数据获取依赖 quantpilot_data_api 模块
    - 分析结果仅供参考，不构成投资建议
    ============================================================================
    """

    def __init__(self, history_days: int = 300, batch: int = 0, limit: int = 0):
        """
        初始化股票分析器
        
        参数说明：
        - history_days: 历史数据天数，默认300天
        - batch: 批量编号，用于分批处理，默认0（不分批）
        - limit: 每批数量限制，默认0（不限制）
        """
        self.history_days = history_days
        self.batch = batch
        self.limit = limit

        # 加载预训练模型
        print("🤖 加载预训练集成模型...")
        self.predictor = EnsembleModel(
            model_dir="data/models"
        )
        self.feature_engineer = FeatureEngineer()

        self.history_data = {}
        self.current_data = {}
        self.results = []

    def fetch_history_data(self, stocks: List) -> bool:
        """
        获取股票历史数据
        
        功能说明：
        从数据源获取指定股票列表的历史行情数据，用于后续特征计算和模型预测。
        
        参数：
        - stocks: List[Tuple], 股票列表
            每个元素为元组格式：(股票代码, 股票名称, 市场)
            示例：[("000001", "平安银行", "A股"), ("600000", "浦发银行", "A股")]
        
        返回值：
        - bool: 是否成功获取至少一只股票的数据
        
        数据存储：
        - 成功获取的数据存储在 self.history_data 字典中
        - key: 股票代码, value: DataFrame 格式的历史数据
        """
        print(f"📈 获取历史数据（过去 {self.history_days} 天）...")

        success_count = 0
        for stock in stocks:
            symbol = stock[0]
            name = stock[1]
            market = stock[2] if len(stock) > 2 else "A股"

            print(f"  获取 {symbol} ({name}) 历史数据...", end=" ")
            data = get_history_data(symbol, market, self.history_days)
            if data is not None and len(data) > 0:
                self.history_data[symbol] = data
                print(f"✓ {len(data)} 条")
                success_count += 1
            else:
                print("✗")

        print(f"\n  汇总：成功 {success_count}/{len(stocks)} 只")
        return success_count > 0

    def fetch_current_data(self, stocks: List) -> bool:
        """
        获取股票实时数据
        
        功能说明：
        从数据源获取指定股票列表的实时行情数据，包括当前价格、涨跌幅、
        市盈率、市净率、ROE等基本面指标。
        
        参数：
        - stocks: List[Tuple], 股票列表
            每个元素为元组格式：(股票代码, 股票名称, 市场)
        
        返回值：
        - bool: 是否成功获取至少一只股票的数据
        
        数据存储：
        - 成功获取的数据存储在 self.current_data 字典中
        - key: 股票代码, value: Dict 格式的实时数据
        
        数据字段说明：
        - current_price: 当前价格
        - change: 涨跌幅（百分比）
        - pe_ratio: 市盈率
        - pb_ratio: 市净率
        - roe: 净资产收益率
        """
        print("\n📊 获取股票实时数据...")

        success_count = 0
        for stock in stocks:
            symbol = stock[0]
            name = stock[1]
            market = stock[2] if len(stock) > 2 else "A股"

            print(f"  获取 {symbol} ({name}) 实时数据...", end=" ")
            data = get_stock_data(symbol, market, name)
            if data:
                self.current_data[symbol] = data
                print(f"✓")
                success_count += 1
            else:
                print("✗")

        print(f"\n  成功获取 {success_count}/{len(stocks)} 只股票的实时数据")
        return success_count > 0

    def analyze_stock(self, symbol: str, current: Dict, history: pd.DataFrame) -> Dict:
        """
        分析单只股票
        
        功能说明：
        对单只股票执行完整的分析流程，包括特征生成、模型预测、风险评估
        和投资建议生成。
        
        参数：
        - symbol: str, 股票代码（如 "000001"）
        - current: Dict, 实时行情数据，包含以下字段：
            - name: 股票名称
            - current_price: 当前价格
            - change: 涨跌幅
            - pe_ratio: 市盈率
            - pb_ratio: 市净率
            - roe: 净资产收益率
        - history: pd.DataFrame, 历史行情数据，需包含：
            - close, high, low, volume 等列
        
        返回值：
        - Dict: 分析结果字典，包含以下字段：
            - symbol: 股票代码
            - name: 股票名称
            - market: 市场（如 "A股"）
            - current_price: 当前价格
            - change_pct: 涨跌幅
            - pe_ratio: 市盈率
            - pb_ratio: 市净率
            - roe: 净资产收益率
            - predicted_return: 预测收益率（百分比）
            - probability: 上涨概率（0-1之间）
            - risk_level: 风险等级（低风险/中风险/高风险/极高风险）
            - advice: 投资建议（建议买入/建议持有/建议观望/建议回避）
        
        分析流程：
        1. 使用 FeatureEngineer 生成技术指标特征
        2. 补充模型所需但特征工程未生成的特征（指数相关）
        3. 调用集成模型进行预测
        4. 根据预测概率计算预期收益和风险等级
        5. 生成投资建议
        
        风险等级划分标准：
        - 低风险: probability >= 0.6
        - 中风险: 0.5 <= probability < 0.6
        - 高风险: 0.4 <= probability < 0.5
        - 极高风险: probability < 0.4
        
        投资建议标准：
        - 建议买入: probability >= 0.6
        - 建议持有: 0.5 <= probability < 0.6
        - 建议观望: 0.4 <= probability < 0.5
        - 建议回避: probability < 0.4
        """
        # 使用 FeatureEngineer 生成特征
        if "code" not in history.columns:
            history = history.copy()
            history["code"] = symbol

        df_features = self.feature_engineer.generate_all_features(history)

        # 获取最新特征
        df_features = df_features.fillna(0)
        if len(df_features) < 2:
            return None

        # 添加模型期望但特征工程未生成的特征（使用默认值0）
        required_features = [
            "index_close",
            "index_return_1d",
            "index_return_5d",
            "index_return_20d",
            "index_ma_position",
            "index_volatility",
            "relative_strength_1d",
            "relative_strength_5d",
            "relative_strength_20d",
            "beta",
        ]
        for feat in required_features:
            if feat not in df_features.columns:
                df_features[feat] = 0.0

        # 使用模型的特征列
        feature_cols = self.predictor.feature_cols
        latest = df_features.iloc[-1:][feature_cols]

        # 填充 NaN
        if latest.isna().any().any():
            latest = latest.fillna(0)

        # 构建特征向量
        X = latest.values

        # 模型预测
        pred = self.predictor.predict(X)

        if "error" in pred:
            return None

        proba = pred["probability"][0]
        prediction = pred["prediction"][0]

        # 计算预期收益
        expected_return = (proba - 0.5) * 10  # 转换为百分比收益

        # 风险等级
        if proba >= 0.6:
            risk_level = "低风险"
        elif proba >= 0.5:
            risk_level = "中风险"
        elif proba >= 0.4:
            risk_level = "高风险"
        else:
            risk_level = "极高风险"

        # 投资建议
        if proba >= 0.6:
            advice = "建议买入"
        elif proba >= 0.5:
            advice = "建议持有"
        elif proba >= 0.4:
            advice = "建议观望"
        else:
            advice = "建议回避"

        return {
            "symbol": symbol,
            "name": current.get("name", symbol),
            "market": "A股",
            "current_price": current.get("current_price", 0),
            "change_pct": current.get("change", 0),
            "pe_ratio": current.get("pe_ratio", 0),
            "pb_ratio": current.get("pb_ratio", 0),
            "roe": current.get("roe", 0),
            "predicted_return": expected_return,
            "probability": proba,
            "risk_level": risk_level,
            "advice": advice,
        }

    def run(self, a_stocks_file: str = None):
        """
        运行完整分析流程
        
        功能说明：
        执行股票分析的完整流程，从读取股票列表到生成预测结果。
        
        参数：
        - a_stocks_file: str, A股股票列表文件路径（JSON格式）
            文件格式：[["股票代码", "股票名称"], ...]
            示例：[["000001", "平安银行"], ["600000", "浦发银行"]]
        
        返回值：
        - None（结果存储在 self.results 中）
        
        执行流程：
        1. 读取股票列表文件
        2. 根据 limit 参数限制分析数量
        3. 获取历史数据（默认60天）
        4. 获取实时行情数据
        5. 对每只股票执行预测分析
        6. 将结果存储到 self.results
        
        使用场景：
        - 日常选股分析
        - 定期市场扫描
        - 批量风险评估
        """
        # 读取股票列表
        stocks = []

        if a_stocks_file and os.path.exists(a_stocks_file):
            print(f"\n📂 读取 A股列表...")
            a_stocks = load_stock_list_from_json(a_stocks_file)
            stocks.extend([(s[0], s[1], "A股") for s in a_stocks])
            print(f"  读取 {len(a_stocks)} 只 A股")

        if not stocks:
            print("❌ 未找到股票列表")
            return

        # 限制数量
        if self.limit > 0:
            stocks = stocks[: self.limit]

        # 获取历史数据
        if not self.fetch_history_data(stocks):
            print("❌ 获取历史数据失败")
            return

        # 获取实时数据
        if not self.fetch_current_data(stocks):
            print("❌ 获取实时数据失败")
            return

        # 分析每只股票
        print("\n📈 进行预测分析...")
        for symbol, name, market in stocks:
            if symbol in self.current_data and symbol in self.history_data:
                result = self.analyze_stock(
                    symbol, self.current_data[symbol], self.history_data[symbol]
                )
                if result:
                    self.results.append(result)
                    print(f"  完成 {symbol} 分析")

        print(f"\n  完成 {len(self.results)} 只股票分析")

    def print_results(self):
        """
        打印分析结果
        
        功能说明：
        格式化输出每只股票的详细分析结果和市场汇总统计。
        
        输出格式说明：
        ────────────────────────────────────────────────────────────────────────
        【单只股票输出格式】
        
        [ML] 【股票代码】股票名称 (市场)
          当前价格：¥XX.XX  (+/-X.XX%)
          市盈率 (PE): XX.XX
          市净率 (PB): XX.XX
          净资产收益率 (ROE): XX.XX%
          预测收益：+/-X.X%
          概率：XX.XX%
          风险等级：低风险/中风险/高风险/极高风险
          投资建议：建议买入/建议持有/建议观望/建议回避
        
        【汇总统计输出格式】
        
        【A 股】共 XX 只
          建议买入：XX 只
          建议持有/观望：XX 只
          建议回避：XX 只
          风险分布：低X/中X/高X
        ────────────────────────────────────────────────────────────────────────
        
        输出字段解释：
        - 当前价格：股票最新成交价
        - 涨跌幅：当日涨跌百分比
        - 市盈率 (PE)：股价/每股收益，反映估值水平
        - 市净率 (PB)：股价/每股净资产，反映估值水平
        - 净资产收益率 (ROE)：净利润/净资产，反映盈利能力
        - 预测收益：模型预测的未来收益（百分比）
        - 概率：模型预测上涨的概率（>50%看涨，<50%看跌）
        - 风险等级：基于概率的风险评估
        - 投资建议：基于概率的操作建议
        
        返回值：
        - None（直接打印到控制台）
        """
        print("\n" + "=" * 80)
        print("分析结果")
        print("=" * 80)

        for r in self.results:
            market_label = f"({r['market']})"
            print(f"\n[ML] 【{r['symbol']}】{r['name']} {market_label}")
            print(f"  当前价格：¥{r['current_price']:.2f}  ({r['change_pct']:+.2f}%)")
            if r.get("pe_ratio", 0) > 0:
                print(f"  市盈率 (PE): {r['pe_ratio']:.2f}")
            if r.get("pb_ratio", 0) > 0:
                print(f"  市净率 (PB): {r['pb_ratio']:.2f}")
            if r.get("roe", 0) > 0:
                print(f"  净资产收益率 (ROE): {r['roe']:.2f}%")
            print(f"  预测收益：{r['predicted_return']:.1f}%")
            print(f"  概率：{r['probability']:.2%}")
            print(f"  风险等级：{r['risk_level']}")
            print(f"  投资建议：{r['advice']}")

        # 汇总统计
        self._print_summary()

    def _print_summary(self):
        """
        打印汇总统计信息
        
        功能说明：
        对所有分析结果进行统计汇总，输出市场整体情况概览。
        
        输出内容说明：
        1. 按投资建议统计：
           - 建议买入：上涨概率 >= 60% 的股票数量
           - 建议持有/观望：上涨概率在 40%-60% 之间的股票数量
           - 建议回避：上涨概率 < 40% 的股票数量
        
        2. 按风险等级统计：
           - 低风险：上涨概率 >= 60%
           - 中风险：上涨概率在 50%-60%
           - 高风险：上涨概率 < 50%
        
        3. 风险提示：
           - 本分析仅供参考，不构成投资建议
           - 市场有风险，投资需谨慎
        
        返回值：
        - None（直接打印到控制台）
        
        使用场景：
        - 快速了解市场整体风险偏好
        - 评估投资组合的风险分布
        - 辅助仓位管理决策
        """
        print("\n" + "=" * 80)
        print("汇总统计")
        print("=" * 80)

        market_results = [r for r in self.results if r["market"] == "A股"]
        if market_results:
            print(f"\n【A 股】共 {len(market_results)} 只")

            buy = sum(1 for r in market_results if "买入" in r["advice"])
            hold = sum(
                1
                for r in market_results
                if "持有" in r["advice"] or "观望" in r["advice"]
            )
            avoid = sum(1 for r in market_results if "回避" in r["advice"])

            print(f"  建议买入：{buy} 只")
            print(f"  建议持有/观望：{hold} 只")
            print(f"  建议回避：{avoid} 只")

            low_risk = sum(1 for r in market_results if r["risk_level"] == "低风险")
            mid_risk = sum(1 for r in market_results if r["risk_level"] == "中风险")
            high_risk = sum(
                1 for r in market_results if r["risk_level"] in ["高风险", "极高风险"]
            )

            print(f"  风险分布：低{low_risk}/中{mid_risk}/高{high_risk}")

        print()
        print("=" * 80)
        print("风险提示：本分析仅供参考，不构成投资建议。市场有风险，投资需谨慎。")
        print("=" * 80)


def main(batch=0, limit=0):
    """
    主函数入口
    
    功能说明：
    创建分析器实例并执行完整的股票分析流程，使用沪深300成分股作为分析对象。
    
    参数：
    - batch: int, 批量编号（1,2,3,4），用于分批处理大规模股票列表
        - 0: 不分批，处理全部股票
        - 1-4: 处理对应批次的股票
    - limit: int, 每批处理的股票数量限制
        - 0: 不限制，处理批次内全部股票
        - >0: 只处理指定数量的股票（用于测试）
    
    执行流程：
    1. 确定股票列表文件路径（默认使用沪深300成分股）
    2. 创建 MLStockAnalyzer 实例
    3. 调用 run() 执行分析
    4. 调用 print_results() 输出结果
    
    使用示例：
    # 分析全部沪深300成分股
    >>> main()
    
    # 分批处理，每批处理50只
    >>> main(batch=1, limit=50)
    
    返回值：
    - None
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 使用沪深300股票列表
    a_stocks_file = "config/csi300_stocks.json"

    print("📊 分析 A 股市场")

    # 创建分析器（使用预训练模型）
    analyzer = MLStockAnalyzer(history_days=300, batch=batch, limit=limit)

    # 运行分析
    analyzer.run(a_stocks_file)

    # 打印结果
    analyzer.print_results()


def parse_args():
    """
    解析命令行参数
    
    功能说明：
    解析脚本运行时的命令行参数，支持分批处理和数量限制。
    
    命令行参数：
    --batch: int, 批量编号
        - 用途：将大规模股票列表分成多批处理，避免单次运行时间过长
        - 取值：1, 2, 3, 4
        - 默认：0（不分批）
        - 示例：python ml_stock_analysis_ensemble.py --batch 1
    
    --limit: int, 每批数量限制
        - 用途：限制每批处理的股票数量，用于测试或快速验证
        - 默认：0（不限制）
        - 示例：python ml_stock_analysis_ensemble.py --batch 1 --limit 10
    
    返回值：
    - argparse.Namespace: 解析后的参数对象
        - batch: 批量编号
        - limit: 数量限制
    
    使用示例：
    # 分析全部股票
    >>> python ml_stock_analysis_ensemble.py
    
    # 分批处理第1批，每批50只
    >>> python ml_stock_analysis_ensemble.py --batch 1 --limit 50
    
    # 测试模式，只分析10只股票
    >>> python ml_stock_analysis_ensemble.py --limit 10
    """
    import argparse

    parser = argparse.ArgumentParser(description="Topaz-Next A 股分析系统")
    parser.add_argument("--batch", type=int, default=0, help="批量编号 (1,2,3,4)")
    parser.add_argument("--limit", type=int, default=0, help="每批数量")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(batch=args.batch, limit=args.limit)
