#!/usr/bin/env python3
"""获取实时行情数据"""
import json
import urllib.request

def get_quote(code):
    secid = f"1.{code}" if code.startswith("6") else f"0.{code}"
    url = f"https://push2.eastmoney.com/api/qt/stock/get?secid={secid}&fields=f43,f44,f45,f46,f47,f48,f50,f51,f52,f58,f60,f169,f170"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            if data.get("data"):
                d = data["data"]
                return {
                    "price": d.get("f43", 0) / 100,
                    "open": d.get("f46", 0) / 100,
                    "high": d.get("f44", 0) / 100,
                    "low": d.get("f45", 0) / 100,
                    "volume": d.get("f47", 0),
                    "amount": d.get("f48", 0),
                    "change_pct": d.get("f170", 0) / 100,
                }
    except Exception as e:
        print(f"Error: {e}")
    return None

stocks = [
    ("600036", "招商银行"),
    ("600900", "长江电力"),
    ("000300", "沪深300"),
]

output = []
for code, name in stocks:
    q = get_quote(code)
    if q:
        output.append(f"{name} ({code}):")
        output.append(f"  价格: {q['price']:.2f}")
        output.append(f"  涨跌: {q['change_pct']:+.2f}%")
        output.append(f"  开高低: {q['open']:.2f} / {q['high']:.2f} / {q['low']:.2f}")
        output.append("")

print("\n".join(output))

# 写入文件
with open("/tmp/quotes.json", "w") as f:
    json.dump({code: get_quote(code) for code, _ in stocks}, f, indent=2)
print("数据已保存到 /tmp/quotes.json")