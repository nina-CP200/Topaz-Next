#!/usr/bin/env python3
"""
Update portfolio with latest prices and send report
Runs every hour during A-share trading hours
"""

import json
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# Portfolio file
PORTFOLIO_FILE = Path(__file__).parent / "virtual_portfolio.json"

# Stock codes to fetch
STOCK_CODES = ["600036", "600900", "000300"]  # 招商银行, 长江电力, 沪深300


def get_realtime_quote(code: str) -> dict:
    """Fetch realtime quote from Eastmoney API"""
    # Determine market
    if code.startswith("6"):
        secid = f"1.{code}"
    else:
        secid = f"0.{code}"
    
    url = f"https://push2.eastmoney.com/api/qt/stock/get?secid={secid}&fields=f43,f44,f45,f46,f47,f48,f50,f51,f52,f58,f60,f169,f170"
    
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            if data.get("data"):
                d = data["data"]
                return {
                    "code": code,
                    "price": d.get("f43", 0) / 100 if d.get("f43") else 0,
                    "change_pct": d.get("f170", 0) / 100 if d.get("f170") else 0,
                    "volume": d.get("f47", 0),
                    "amount": d.get("f48", 0),
                }
    except Exception as e:
        print(f"Error fetching {code}: {e}")
    
    return {"code": code, "price": 0, "change_pct": 0}


def update_portfolio():
    """Update portfolio with latest prices"""
    if not PORTFOLIO_FILE.exists():
        print(f"Portfolio file not found: {PORTFOLIO_FILE}")
        return None
    
    with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
        portfolio = json.load(f)
    
    # Fetch latest prices
    prices = {}
    for code in STOCK_CODES:
        quote = get_realtime_quote(code)
        prices[code] = quote["price"]
        print(f"{code}: ¥{quote['price']:.2f} ({quote['change_pct']:+.2f}%)")
    
    # Update holdings
    holdings_value = 0
    for code, holding in portfolio.get("holdings", {}).items():
        if code in prices and prices[code] > 0:
            holding["current_price"] = prices[code]
            holdings_value += holding["shares"] * prices[code]
        else:
            # Keep previous price if fetch failed
            holdings_value += holding["shares"] * holding.get("current_price", holding.get("cost_price", 0))
    
    # Update benchmark
    benchmark_value = prices.get("000300", portfolio.get("daily_values", [{}])[-1].get("benchmark_value", 4593.11))
    
    # Calculate total and PnL
    total_value = portfolio["cash"] + holdings_value
    pnl = total_value - portfolio["initial_capital"]
    pnl_pct = (pnl / portfolio["initial_capital"]) * 100
    
    # Record daily value
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M")
    
    daily_record = {
        "date": today,
        "time": time_now,
        "cash": portfolio["cash"],
        "holdings_value": holdings_value,
        "total_value": total_value,
        "pnl": pnl,
        "pnl_pct": round(pnl_pct, 2),
        "benchmark_value": benchmark_value,
        "benchmark_pct": 0  # Calculate if we have benchmark start value
    }
    
    # Update or append daily record
    if portfolio.get("daily_values") and portfolio["daily_values"][-1].get("date") == today:
        portfolio["daily_values"][-1] = daily_record
    else:
        portfolio["daily_values"].append(daily_record)
    
    # Save
    with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
        json.dump(portfolio, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Portfolio Updated ===")
    print(f"Total: ¥{total_value:,.0f}")
    print(f"PnL: ¥{pnl:,.0f} ({pnl_pct:+.2f}%)")
    print(f"Holdings: ¥{holdings_value:,.0f}")
    print(f"Cash: ¥{portfolio['cash']:,.0f}")
    
    return {
        "total_value": total_value,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "holdings_value": holdings_value,
        "holdings": portfolio.get("holdings", {}),
        "cash": portfolio["cash"],
        "time": time_now
    }


if __name__ == "__main__":
    result = update_portfolio()
    if result:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Portfolio update completed")