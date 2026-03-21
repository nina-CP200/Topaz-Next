#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A 股交易日估值分析脚本
- 检测是否为交易日
- 运行估值模型
- 输出报告
"""

import subprocess
import sys
from datetime import datetime, date
from pathlib import Path

# 中国 A股 2026 年休市日（需定期更新）
HOLIDAYS_2026 = [
    # 元旦
    date(2026, 1, 1),
    date(2026, 1, 2),
    date(2026, 1, 3),
    # 春节
    date(2026, 2, 16),
    date(2026, 2, 17),
    date(2026, 2, 18),
    date(2026, 2, 19),
    date(2026, 2, 20),
    date(2026, 2, 21),
    date(2026, 2, 22),
    # 清明
    date(2026, 4, 4),
    date(2026, 4, 5),
    date(2026, 4, 6),
    # 五一
    date(2026, 5, 1),
    date(2026, 5, 2),
    date(2026, 5, 3),
    date(2026, 5, 4),
    date(2026, 5, 5),
    # 端午
    date(2026, 5, 31),
    date(2026, 6, 1),
    date(2026, 6, 2),
    # 中秋
    date(2026, 10, 3),
    date(2026, 10, 4),
    date(2026, 10, 5),
    # 国庆
    date(2026, 10, 1),
    date(2026, 10, 2),
    date(2026, 10, 6),
    date(2026, 10, 7),
    date(2026, 10, 8),
]

def is_trading_day(check_date: date = None) -> tuple[bool, str]:
    """
    检查是否为 A股交易日
    
    Returns:
        (is_trading, reason)
    """
    if check_date is None:
        check_date = date.today()
    
    # 周末不交易
    if check_date.weekday() >= 5:
        return False, f"周末休市 ({check_date.strftime('%Y-%m-%d')} 是 {'周六' if check_date.weekday() == 5 else '周日'})"
    
    # 节假日不交易
    if check_date in HOLIDAYS_2026:
        return False, f"节假日休市 ({check_date.strftime('%Y-%m-%d')})"
    
    return True, f"交易日 ({check_date.strftime('%Y-%m-%d')})"


def run_analysis():
    """
    运行估值分析
    """
    script_dir = Path(__file__).parent
    analysis_script = script_dir / "ml_stock_analysis.py"
    
    print(f"=== A股估值分析 ===")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查交易日
    is_trading, reason = is_trading_day()
    print(f"交易日检测: {reason}")
    
    if not is_trading:
        print("非交易日，不运行分析")
        return False
    
    # 运行分析
    print("\n开始运行估值模型...")
    try:
        result = subprocess.run(
            ["python3", str(analysis_script), "--cn"],
            cwd=str(script_dir),
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        if result.returncode == 0:
            print("分析完成")
            print(result.stdout)
            return True
        else:
            print(f"分析失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("分析超时")
        return False
    except Exception as e:
        print(f"分析出错: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="A 股交易日估值分析")
    parser.add_argument("--check", action="store_true", help="只检查是否为交易日")
    parser.add_argument("--date", type=str, help="指定日期 (YYYY-MM-DD)")
    args = parser.parse_args()
    
    if args.date:
        y, m, d = map(int, args.date.split("-"))
        check_date = date(y, m, d)
    else:
        check_date = None
    
    if args.check:
        is_trading, reason = is_trading_day(check_date)
        print(f"交易日检测: {reason}")
        print(f"结果: {'是交易日' if is_trading else '非交易日'}")
        sys.exit(0 if is_trading else 1)
    
    # 运行分析
    success = run_analysis()
    sys.exit(0 if success else 1)