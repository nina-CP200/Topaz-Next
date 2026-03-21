#!/usr/bin/env python3
"""A股实时行情获取脚本 - 使用东方财富公开API"""

import urllib.request
import json
import ssl
from datetime import datetime

# A股关注股票列表
A_SHARE_STOCKS = [
    ("1.600519", "贵州茅台"),
    ("1.000858", "五粮液"),
    ("1.600036", "招商银行"),
    ("1.600111", "北方稀土"),
    ("0.002465", "海格通信"),
    ("1.601318", "中国平安"),
    ("1.000001", "平安银行"),
    ("0.002415", "海康威视"),
    ("1.600900", "长江电力"),
    ("1.601166", "兴业银行"),
    ("1.000300", "沪深300"),  # 基准指数
]

def fetch_quotes():
    """获取实时行情"""
    # 构建股票代码列表
    secids = ",".join([code for code, _ in A_SHARE_STOCKS])
    
    # 东方财富实时行情API
    url = f"https://push2.eastmoney.com/api/qt/ulist.np/get?fltt=2&secids={secids}&fields=f2,f3,f12,f14,f15,f16,f17,f18,f5,f6"
    
    # 忽略SSL验证（仅用于测试）
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://quote.eastmoney.com/'
        })
        response = urllib.request.urlopen(req, timeout=15, context=ctx)
        data = json.loads(response.read().decode())
        return data
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None

def main():
    print("=" * 80)
    print(f"A股关注股票实时行情 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("=" * 80)
    
    data = fetch_quotes()
    if not data or not data.get('data'):
        print("无法获取数据")
        return
    
    quotes = []
    for item in data['data'].get('diff', []):
        code = item.get('f12', 'N/A')
        name = item.get('f14', 'N/A')
        price = item.get('f2', 0) or 0
        change_pct = item.get('f3', 0) or 0
        high = item.get('f15', 0) or 0
        low = item.get('f16', 0) or 0
        open_ = item.get('f17', 0) or 0
        prev_close = item.get('f18', 0) or 0
        volume = item.get('f5', 0) or 0  # 成交量（手）
        amount = item.get('f6', 0) or 0  # 成交额（万）
        
        quotes.append({
            'code': code,
            'name': name,
            'price': price,
            'change_pct': change_pct,
            'high': high,
            'low': low,
            'open': open_,
            'prev_close': prev_close,
            'volume': volume,
            'amount': amount
        })
    
    # 打印行情表
    print(f"{'代码':<10} {'名称':<10} {'现价':>10} {'涨跌%':>8} {'最高':>10} {'最低':>10} {'成交额(万)':>12}")
    print("-" * 80)
    
    for q in quotes:
        print(f"{q['code']:<10} {q['name']:<10} {q['price']:>10.2f} {q['change_pct']:>7.2f}% {q['high']:>10.2f} {q['low']:>10.2f} {q['amount']:>12.0f}")
    
    print("=" * 80)
    
    # 保存到文件
    output_file = "/home/emmmoji/.openclaw/workspace-topaz/topaz-v3/data/a_share_quotes_today.json"
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'quotes': quotes
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据已保存到: {output_file}")
    
    return quotes

if __name__ == "__main__":
    main()