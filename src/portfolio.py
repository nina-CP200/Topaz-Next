#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
持仓分析模块

功能：
1. 读取用户持仓（股票代码+数量+成本价）
2. 自动抓取实时数据
3. 匹配行业分类
4. 计算持仓盈亏和风险
5. 给出调仓建议
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

from src.data.api import get_stock_data, get_history_data
from src.features.engineer import FeatureEngineer
from src.strategy import compute_composite_score, get_advice_from_score, suggest_position
from src.utils.utils import load_stock_list_from_json
from src.config import get_market_state_manager
from src.data.market import get_market_adjusted_thresholds, get_index_history


def load_portfolio(file_path: str) -> List[Dict]:
    """
    加载持仓文件

    文件格式（JSON）：
    [
        {"code": "600519", "name": "贵州茅台", "shares": 100, "cost": 1680.00},
        {"code": "000858", "name": "五粮液", "shares": 200, "cost": 150.00}
    ]

    或者简化格式（每行一只股票）：
    600519 100 1680
    000858 200 150
    """
    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # 尝试 JSON 格式
    if content.startswith('['):
        try:
            data = json.loads(content)
            return data
        except json.JSONDecodeError:
            pass

    # 尝试简化格式：code shares cost
    portfolio = []
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 2:
            code = parts[0]
            shares = int(parts[1]) if len(parts) > 1 else 0
            cost = float(parts[2]) if len(parts) > 2 else 0
            portfolio.append({
                "code": code,
                "name": "",
                "shares": shares,
                "cost": cost,
            })

    return portfolio


def analyze_portfolio(portfolio: List[Dict]) -> Dict:
    """
    分析持仓

    Returns:
        {
            "holdings": [...],  # 每只股票的分析结果
            "failed": [...],    # 分析失败的股票
            "total_value": float,
            "total_cost": float,
            "total_pnl": float,
            "total_pnl_pct": float,
            "industry_allocation": {...},
            "risk_summary": {...},
        }
    """
    # 加载行业映射
    industry_map = {}
    industry_file = "config/csi300_industry_map.json"
    if os.path.exists(industry_file):
        with open(industry_file, 'r', encoding='utf-8') as f:
            industry_map = json.load(f)

    # 获取市场环境
    state_manager = get_market_state_manager()
    regime = state_manager.state.get("current_regime", "sideways")
    thresholds = get_market_adjusted_thresholds(regime)

    # 加载指数数据（用于相对强弱计算）
    index_history = get_index_history("000300.SH", days=60)

    fe = FeatureEngineer()
    holdings = []
    failed = []  # 记录失败的股票
    industry_data = {}

    for item in portfolio:
        code = item.get("code", "")
        shares = item.get("shares", 0)
        cost = item.get("cost", 0)

        # 验证股票代码格式
        if not code:
            failed.append({"code": code, "reason": "代码为空"})
            continue

        # 补全代码后缀
        if code.startswith("6") and not code.endswith(".SH"):
            symbol = f"{code}.SH"
        elif (code.startswith("0") or code.startswith("3")) and not code.endswith(".SZ"):
            symbol = f"{code}.SZ"
        else:
            symbol = code

        # 获取实时数据
        current_data = get_stock_data(symbol, "A股")
        if not current_data:
            failed.append({"code": symbol, "reason": "无法获取实时数据（可能代码不存在或API异常）"})
            continue

        current_price = current_data.get("current_price", 0)
        if current_price == 0:
            failed.append({"code": symbol, "reason": "当前价格为0（可能已停牌或退市）"})
            continue

        change_pct = current_data.get("change", 0)

        # 获取历史数据并计算特征
        history = get_history_data(symbol, "A股", days=300)
        if history is None or len(history) < 30:
            failed.append({"code": symbol, "reason": f"历史数据不足（仅{len(history) if history is not None else 0}天，需30天）"})
            continue

        history["code"] = symbol
        df_features = fe.generate_all_features(history)
        if index_history is not None:
            df_features = fe.add_index_factors(df_features, index_history)
        df_features = df_features.fillna(0)

        # 计算综合评分
        latest = df_features.iloc[-1]
        rsi = latest.get('rsi', 50)
        macd_hist = latest.get('macd_hist', 0)
        price_to_ma20 = latest.get('price_to_ma20', 1.0)
        ret_5d = latest.get('return_5d', 0)
        ret_20d = latest.get('return_20d', 0)

        # 简化评分（不依赖 ML 模型）
        composite, reasons = compute_composite_score(0.5, latest.to_dict())
        advice = get_advice_from_score(composite, 0.7)

        # 计算盈亏
        market_value = current_price * shares
        cost_value = cost * shares
        pnl = market_value - cost_value
        pnl_pct = (current_price / cost - 1) * 100 if cost > 0 else 0

        # 行业
        industry = industry_map.get(symbol, "未知")

        holding = {
            "symbol": symbol,
            "name": item.get("name", ""),
            "industry": industry,
            "shares": shares,
            "cost": cost,
            "current_price": current_price,
            "market_value": round(market_value, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "change_today": change_pct,
            "composite_score": round(composite, 3),
            "advice": advice,
            "reasons": reasons,
            "rsi": round(rsi, 1),
            "ret_5d": round(ret_5d * 100, 2),
            "ret_20d": round(ret_20d * 100, 2),
        }
        holdings.append(holding)

        # 行业聚合
        if industry not in industry_data:
            industry_data[industry] = {"value": 0, "pnl": 0, "stocks": []}
        industry_data[industry]["value"] += market_value
        industry_data[industry]["pnl"] += pnl
        industry_data[industry]["stocks"].append(holding["name"] or symbol)

    # 汇总
    total_value = sum(h["market_value"] for h in holdings)
    total_cost = sum(h["cost"] * h["shares"] for h in holdings)
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

    # 行业占比
    industry_allocation = {}
    for ind, data in industry_data.items():
        industry_allocation[ind] = {
            "value": round(data["value"], 2),
            "pct": round(data["value"] / total_value * 100, 2) if total_value > 0 else 0,
            "pnl": round(data["pnl"], 2),
            "stocks": data["stocks"],
        }

    # 风险统计
    buy_count = sum(1 for h in holdings if "买入" in h["advice"])
    sell_count = sum(1 for h in holdings if "卖出" in h["advice"] or "回避" in h["advice"])
    loss_count = sum(1 for h in holdings if h["pnl_pct"] < 0)

    return {
        "holdings": holdings,
        "failed": failed,
        "total_value": round(total_value, 2),
        "total_cost": round(total_cost, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "industry_allocation": industry_allocation,
        "regime": regime,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "loss_count": loss_count,
    }


def print_portfolio_report(result: Dict):
    """打印持仓分析报告"""
    holdings = result.get("holdings", [])
    failed = result.get("failed", [])

    if not holdings and not failed:
        print("\n⚠️ 无持仓数据")
        return

    regime = result.get("regime", "sideways")

    print("\n" + "=" * 80)
    print("💼 持仓分析报告")
    print("=" * 80)

    # 汇总
    print(f"\n📊 持仓汇总")
    print(f"  总市值: ¥{result['total_value']:,.2f}")
    print(f"  总成本: ¥{result['total_cost']:,.2f}")
    pnl_emoji = "📈" if result['total_pnl'] >= 0 else "📉"
    print(f"  总盈亏: {pnl_emoji} ¥{result['total_pnl']:+,.2f} ({result['total_pnl_pct']:+.2f}%)")
    print(f"  市场环境: {regime}")
    print(f"  买入信号: {result['buy_count']}只 | 卖出信号: {result['sell_count']}只 | 亏损: {result['loss_count']}只")

    # 失败的股票
    if failed:
        print(f"\n⚠️ 无法分析的股票（{len(failed)}只）")
        print("-" * 80)
        for f in failed:
            print(f"  {f['code']}: {f['reason']}")

    # 个股详情
    print(f"\n📋 个股详情（共{len(holdings)}只）")
    print("-" * 80)
    for i, h in enumerate(sorted(holdings, key=lambda x: x['pnl_pct'], reverse=True), 1):
        pnl_sign = "+" if h['pnl_pct'] >= 0 else ""
        print(f"  #{i} {h['symbol']} {h['name']} [{h['industry']}]")
        print(f"     市值:¥{h['market_value']:,.0f} 盈亏:{pnl_sign}¥{h['pnl']:,.0f} ({pnl_sign}{h['pnl_pct']:.1f}%)")
        print(f"     评分:{h['composite_score']:.2f} {h['advice']} | RSI:{h['rsi']:.0f} 5日:{h['ret_5d']:+.1f}%")
        print(f"     理由: {' + '.join(h['reasons'])}")

    # 行业分布
    print(f"\n🏭 行业分布")
    print("-" * 80)
    for ind, data in sorted(result['industry_allocation'].items(), key=lambda x: x[1]['pct'], reverse=True):
        stocks = ", ".join(data['stocks'][:3])
        print(f"  {ind}: {data['pct']:.1f}% (¥{data['value']:,.0f}) | {stocks}")

    # ========== 操作建议（核心部分） ==========
    print(f"\n{'=' * 80}")
    print("🎯 操作建议")
    print(f"{'=' * 80}")

    # 分类持仓
    sell_list = [h for h in holdings if "卖出" in h["advice"] or "回避" in h["advice"]]
    hold_list = [h for h in holdings if "观望" in h["advice"] or "持有" in h["advice"]]
    buy_list = [h for h in holdings if "买入" in h["advice"]]

    # 建议卖出
    if sell_list:
        print(f"\n🔴 建议卖出（{len(sell_list)}只）")
        print("-" * 80)
        for h in sell_list:
            print(f"  {h['symbol']} {h['name']}")
            print(f"    盈亏: {h['pnl_pct']:+.1f}% | 评分: {h['composite_score']:.2f}")
            print(f"    原因: {', '.join(h['reasons'])}")
            print(f"    建议: 清仓，止损离场")

    # 建议减仓
    reduce_list = [h for h in holdings if h['pnl_pct'] < -10 and h['advice'] not in ["建议回避", "建议卖出"]]
    if reduce_list:
        print(f"\n🟡 建议减仓（{len(reduce_list)}只）")
        print("-" * 80)
        for h in reduce_list:
            print(f"  {h['symbol']} {h['name']}")
            print(f"    盈亏: {h['pnl_pct']:+.1f}% | 评分: {h['composite_score']:.2f}")
            print(f"    原因: {', '.join(h['reasons'])}")
            print(f"    建议: 减仓至半仓，降低风险敞口")

    # 建议持有
    if hold_list:
        print(f"\n🟢 建议持有（{len(hold_list)}只）")
        print("-" * 80)
        for h in hold_list:
            print(f"  {h['symbol']} {h['name']}")
            print(f"    盈亏: {h['pnl_pct']:+.1f}% | 评分: {h['composite_score']:.2f}")
            print(f"    原因: {', '.join(h['reasons'])}")
            print(f"    建议: 继续持有，暂不操作")

    # 建议加仓
    if buy_list:
        print(f"\n🔵 建议加仓（{len(buy_list)}只）")
        print("-" * 80)
        for h in buy_list:
            print(f"  {h['symbol']} {h['name']}")
            print(f"    盈亏: {h['pnl_pct']:+.1f}% | 评分: {h['composite_score']:.2f}")
            print(f"    原因: {', '.join(h['reasons'])}")
            print(f"    建议: 逢低加仓，提升仓位")

    # 行业集中度提醒
    high_concentration = {ind: data for ind, data in result['industry_allocation'].items() if data['pct'] > 40}
    if high_concentration:
        print(f"\n⚠️ 行业集中度提醒")
        print("-" * 80)
        for ind, data in high_concentration.items():
            print(f"  {ind} 占比 {data['pct']:.1f}%，建议分散配置")

    # 整体建议
    print(f"\n📝 整体建议")
    print("-" * 80)
    if sell_list:
        print(f"  优先处理 {len(sell_list)} 只建议卖出的股票")
    if reduce_list:
        print(f"  降低 {len(reduce_list)} 只亏损较大股票的仓位")
    if high_concentration:
        print(f"  注意行业集中度风险，适当分散")
    if not sell_list and not reduce_list and not high_concentration:
        print(f"  当前持仓状况良好，暂无需调整")


def interactive_menu():
    """交互式持仓管理菜单"""
    portfolio = []

    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        print("=" * 60)
        print("  Topaz 持仓分析")
        print("=" * 60)

        # 显示当前持仓
        if portfolio:
            print(f"\n📋 当前持仓（{len(portfolio)}只）")
            print("-" * 60)
            for i, p in enumerate(portfolio, 1):
                print(f"  {i}. {p['code']} {p['name']} | {p['shares']}股 | 成本¥{p['cost']:.2f}")
        else:
            print("\n  （暂无持仓）")

        # 菜单
        print(f"\n{'=' * 60}")
        print("  1. 添加股票")
        print("  2. 修改股票")
        print("  3. 删除股票")
        print("  4. 分析持仓")
        print("  5. 退出")
        print(f"{'=' * 60}")

        choice = input("\n请选择操作 (1-5): ").strip()

        if choice == "1":
            # 添加股票
            print("\n--- 添加股票 ---")
            code = input("股票代码（如 600519）: ").strip()
            if not code:
                continue

            # 验证代码格式
            clean_code = code.replace(".SH", "").replace(".SZ", "")
            if not clean_code.isdigit() or len(clean_code) != 6:
                print("⚠️ 股票代码必须是6位数字（如 600519）")
                input("按回车继续...")
                continue

            name = input("股票名称（可选，回车跳过）: ").strip()
            try:
                shares = int(input("持有数量（股）: ").strip() or "0")
                if shares <= 0:
                    print("⚠️ 数量必须大于0")
                    input("按回车继续...")
                    continue
                cost = float(input("成本价（元）: ").strip() or "0")
                if cost <= 0:
                    print("⚠️ 成本价必须大于0")
                    input("按回车继续...")
                    continue
            except ValueError:
                print("⚠️ 输入格式错误")
                input("按回车继续...")
                continue

            portfolio.append({
                "code": clean_code,
                "name": name,
                "shares": shares,
                "cost": cost,
            })
            print(f"✓ 已添加 {clean_code}")

        elif choice == "2":
            # 修改股票
            if not portfolio:
                print("⚠️ 暂无持仓可修改")
                input("按回车继续...")
                continue

            try:
                idx = int(input(f"修改第几只 (1-{len(portfolio)}): ").strip()) - 1
                if idx < 0 or idx >= len(portfolio):
                    print("⚠️ 序号无效")
                    input("按回车继续...")
                    continue
            except ValueError:
                print("⚠️ 输入格式错误")
                input("按回车继续...")
                continue

            p = portfolio[idx]
            print(f"\n当前: {p['code']} {p['name']} | {p['shares']}股 | 成本¥{p['cost']:.2f}")
            print("（直接回车保持原值）")

            new_code = input(f"股票代码 [{p['code']}]: ").strip() or p['code']
            # 验证代码格式
            clean_code = new_code.replace(".SH", "").replace(".SZ", "")
            if not clean_code.isdigit() or len(clean_code) != 6:
                print("⚠️ 股票代码必须是6位数字")
                input("按回车继续...")
                continue
            new_code = clean_code

            new_name = input(f"股票名称 [{p['name']}]: ").strip() or p['name']
            new_shares = input(f"持有数量 [{p['shares']}]: ").strip()
            new_cost = input(f"成本价 [{p['cost']}]: ").strip()

            p['code'] = new_code
            p['name'] = new_name
            if new_shares:
                try:
                    val = int(new_shares)
                    if val <= 0:
                        print("⚠️ 数量必须大于0，未修改")
                    else:
                        p['shares'] = val
                except ValueError:
                    print("⚠️ 数量格式错误，未修改")
            if new_cost:
                try:
                    val = float(new_cost)
                    if val <= 0:
                        print("⚠️ 成本价必须大于0，未修改")
                    else:
                        p['cost'] = val
                except ValueError:
                    print("⚠️ 成本价格式错误，未修改")

            print(f"✓ 已修改 {p['code']}")

        elif choice == "3":
            # 删除股票
            if not portfolio:
                print("⚠️ 暂无持仓可删除")
                input("按回车继续...")
                continue

            try:
                idx = int(input(f"删除第几只 (1-{len(portfolio)}): ").strip()) - 1
                if idx < 0 or idx >= len(portfolio):
                    print("⚠️ 序号无效")
                    input("按回车继续...")
                    continue
            except ValueError:
                print("⚠️ 输入格式错误")
                input("按回车继续...")
                continue

            removed = portfolio.pop(idx)
            print(f"✓ 已删除 {removed['code']} {removed['name']}")

        elif choice == "4":
            # 分析持仓
            if not portfolio:
                print("⚠️ 请先添加持仓")
                input("按回车继续...")
                continue

            print(f"\n🔄 正在分析 {len(portfolio)} 只股票...")
            result = analyze_portfolio(portfolio)

            # 保存结果
            output_file = "data/raw/portfolio_analysis.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print_portfolio_report(result)
            input("\n按回车返回菜单...")

        elif choice == "5":
            # 退出
            if portfolio:
                save = input("\n是否保存持仓到文件？(y/n): ").strip().lower()
                if save == 'y':
                    filename = input("文件名 (如 portfolio.json): ").strip() or "portfolio.json"
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(portfolio, f, ensure_ascii=False, indent=2)
                    print(f"✓ 已保存到 {filename}")
            print("\n👋 再见！")
            break

        else:
            print("⚠️ 无效选择")


def main():
    """命令行入口"""
    # 检查是否有文件参数
    if len(sys.argv) >= 2:
        # 从文件加载
        portfolio_file = sys.argv[1]
        print("📈 加载持仓...")
        portfolio = load_portfolio(portfolio_file)
        if not portfolio:
            print("  ❌ 无法加载持仓文件")
            sys.exit(1)

        print(f"  ✓ 加载 {len(portfolio)} 只股票")
        print("\n🔄 分析持仓...")
        result = analyze_portfolio(portfolio)

        output_file = "data/raw/portfolio_analysis.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 结果已保存: {output_file}")

        print_portfolio_report(result)
    else:
        # 交互式模式
        interactive_menu()


if __name__ == "__main__":
    main()
