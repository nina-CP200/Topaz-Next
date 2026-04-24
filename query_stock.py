#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票查询脚本 - 查询特定股票的评分和排名

使用方法：
  python query_stock.py 600519      # 纯数字代码
  python query_stock.py 600519.SH   # 带.SH后缀
  python query_stock.py 600519.sh   # 小写后缀也支持
  python query_stock.py 000001.SZ   # 深圳股票

功能：
  - 查询上一次分析结果
  - 显示评分概率和排名
  - 显示风险等级和投资建议
  - 支持多种代码格式输入
"""

import json
import os
import sys
import argparse
from datetime import datetime


def normalize_stock_code(code: str) -> str:
    """
    标准化股票代码格式
    
    支持输入格式：
      - 600519 (纯数字)
      - 600519.SH (大写后缀)
      - 600519.sh (小写后缀)
      - 600519.SZ (深圳)
    
    返回标准格式：
      - 600519.SH (上海)
      - 000001.SZ (深圳)
    """
    # 去除空格和特殊字符
    code = code.strip().upper()
    
    # 如果已有后缀，直接返回
    if '.SH' in code or '.SZ' in code:
        return code
    
    # 纯数字代码，需要判断交易所
    # 6开头 = 上海，0/3开头 = 深圳，68开头 = 上海科创板
    if len(code) == 6 and code.isdigit():
        if code.startswith('6'):
            return f"{code}.SH"
        elif code.startswith('0') or code.startswith('3'):
            return f"{code}.SZ"
        else:
            # 其他情况默认深圳
            return f"{code}.SZ"
    
    # 其他格式，尝试直接返回
    return code


def load_analysis_results() -> dict:
    """
    加载最近的分析结果
    
    Returns:
        dict: 分析结果数据，如果文件不存在返回 None
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_file = os.path.join(base_dir, "latest_analysis_results.json")
    
    if not os.path.exists(result_file):
        return None
    
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"✗ 读取结果文件失败: {e}")
        return None


def query_stock(stock_code: str) -> None:
    """
    查询股票评分和排名
    
    Args:
        stock_code: 股票代码（支持多种格式）
    """
    # 标准化代码
    normalized_code = normalize_stock_code(stock_code)
    
    print("=" * 60)
    print("股票查询")
    print("=" * 60)
    print(f"查询代码: {stock_code}")
    print(f"标准代码: {normalized_code}")
    print()
    
    # 加载分析结果
    data = load_analysis_results()
    
    if data is None:
        print("✗ 未找到分析结果文件")
        print("  请先运行分析: python daily_decision.py")
        return
    
    # 显示分析日期
    print(f"分析日期: {data.get('date', '未知')}")
    print(f"市场环境: {data.get('market_regime', '未知')}")
    print(f"分析股票数: {data.get('total_stocks', 0)}")
    print()
    
    # 查找股票
    results = data.get('results', [])
    stock_info = None
    
    for r in results:
        if r.get('symbol') == normalized_code:
            stock_info = r
            break
    
    if stock_info is None:
        print(f"✗ 未找到股票: {normalized_code}")
        print("  可能原因:")
        print("    1. 股票代码错误")
        print("    2. 该股票不在沪深300成分股中")
        print("    3. 分析结果文件需要更新")
        print()
        print("  提示: 请检查 csi300_stocks.json 是否包含该股票")
        return
    
    # 显示查询结果
    print("-" * 60)
    print("查询结果")
    print("-" * 60)
    
    name = stock_info.get('name', '未知')
    rank = stock_info.get('rank', 0)
    total = data.get('total_stocks', 300)
    probability = stock_info.get('probability', 0)
    predicted_return = stock_info.get('predicted_return', 0)
    current_price = stock_info.get('current_price', 0)
    change_pct = stock_info.get('change_pct', 0)
    risk_level = stock_info.get('risk_level', '未知')
    advice = stock_info.get('advice', '未知')
    model_confidence = data.get('model_confidence', 0.5)
    
    print(f"股票名称: {name}")
    print(f"股票代码: {normalized_code}")
    print()
    
    print(f"【排名信息】")
    print(f"  总排名: 第 {rank} 名 / 共 {total} 只")
    print(f"  排名百分位: {rank/total*100:.1f}%")
    
    # 排名等级
    if rank <= 30:
        rank_grade = "Top 10% ⭐⭐⭐"
    elif rank <= 100:
        rank_grade = "Top 33% ⭐⭐"
    elif rank <= 200:
        rank_grade = "中等 ⭐"
    else:
        rank_grade = "Bottom 33%"
    print(f"  排名等级: {rank_grade}")
    print()
    
    print(f"【评分信息】")
    print(f"  上涨概率: {probability:.1%}")
    print(f"  预期收益: {predicted_return:+.2f}%")
    print(f"  模型置信度: {model_confidence:.0%}")
    print()
    
    print(f"【价格信息】")
    print(f"  当前价格: ¥{current_price:.2f}")
    print(f"  今日涨跌: {change_pct:+.2f}%")
    print()
    
    print(f"【风险评估】")
    print(f"  风险等级: {risk_level}")
    print(f"  投资建议: {advice}")
    
    print()
    print("=" * 60)
    print("风险提示：本分析仅供参考，不构成投资建议。")
    print("=" * 60)


def list_top_stocks(n: int = 10) -> None:
    """
    显示排名前N的股票
    
    Args:
        n: 显示数量，默认10
    """
    data = load_analysis_results()
    
    if data is None:
        print("✗ 未找到分析结果文件")
        print("  请先运行分析: python daily_decision.py")
        return
    
    print("=" * 60)
    print(f"Top {n} 股票排名")
    print("=" * 60)
    print(f"分析日期: {data.get('date', '未知')}")
    print()
    
    results = data.get('results', [])
    
    for i, r in enumerate(results[:n]):
        symbol = r.get('symbol', '')
        name = r.get('name', '')
        prob = r.get('probability', 0)
        ret = r.get('predicted_return', 0)
        advice = r.get('advice', '')
        
        print(f"{i+1:3d}. {symbol} {name}")
        print(f"     概率: {prob:.1%} | 预期收益: {ret:+.2f}% | 建议: {advice}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="股票查询脚本 - 查询特定股票的评分和排名",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python query_stock.py 600519        # 查询贵州茅台
  python query_stock.py 000001.SZ     # 查询平安银行
  python query_stock.py --top 20      # 显示前20名股票
        """
    )
    
    parser.add_argument("code", nargs="?", help="股票代码（如 600519 或 600519.SH）")
    parser.add_argument("--top", type=int, default=0, help="显示排名前N的股票")
    
    args = parser.parse_args()
    
    if args.top > 0:
        list_top_stocks(args.top)
    elif args.code:
        query_stock(args.code)
    else:
        parser.print_help()
        print()
        print("提示: 请提供股票代码或使用 --top 参数")


if __name__ == "__main__":
    main()