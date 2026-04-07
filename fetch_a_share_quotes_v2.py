#!/usr/bin/env python3
"""A股实时行情获取脚本 - 多数据源备份版

数据源优先级：
1. 腾讯财经 (首选)
2. 新浪财经 (备用1)
3. 东方财富 (备用2)
"""

import urllib.request
import json
import ssl
import time
from datetime import datetime

# 忽略SSL验证
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# A股关注股票列表 - 从 A股关注股票列表.md 读取
import json
import os
from utils import parse_stock_list

def load_stock_list():
    """从 A股关注股票列表.md 加载股票列表（统一数据源）"""
    # 优先使用 A股关注股票列表.md
    md_path = os.path.join(os.path.dirname(__file__), 'A股关注股票列表.md')
    try:
        stocks = parse_stock_list(md_path)
        if stocks:
            # 转换为腾讯/新浪格式
            result = []
            for symbol, name, category in stocks:
                # 去掉后缀 .SH/.SZ
                code = symbol.replace('.SH', '').replace('.SZ', '')
                if code.startswith('6'):
                    tencent_code = f'sh{code}'
                else:
                    tencent_code = f'sz{code}'
                result.append((tencent_code, name))
            print(f"  从 A股关注股票列表.md 加载 {len(result)} 只股票")
            return result
    except Exception as e:
        print(f"警告: 无法加载 A股关注股票列表.md: {e}")
    
    # 备用：从 csi300_stocks.json 加载（全部，不只是前10只）
    config_path = os.path.join(os.path.dirname(__file__), 'csi300_stocks.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            stocks = json.load(f)
        result = []
        for s in stocks:
            code = s['code']
            name = s['name']
            if code.startswith('6'):
                tencent_code = f'sh{code}'
            else:
                tencent_code = f'sz{code}'
            result.append((tencent_code, name))
        print(f"  从 csi300_stocks.json 加载 {len(result)} 只股票")
        return result
    except Exception as e:
        print(f"警告: 无法加载股票列表，使用默认列表: {e}")
        # 默认列表
        return [
            ('sh600519', '贵州茅台'),
            ('sz000858', '五粮液'),
            ('sh600036', '招商银行'),
            ('sh600111', '北方稀土'),
            ('sz002465', '海格通信'),
            ('sh601318', '中国平安'),
            ('sh600900', '长江电力'),
            ('sh601166', '兴业银行'),
            ('sh601888', '中国中免'),
            ('sz000333', '美的集团'),
        ]

# 加载股票列表
STOCKS_TENCENT = load_stock_list()
STOCKS_SINA = STOCKS_TENCENT

def fetch_from_tencent():
    """从腾讯财经获取实时行情"""
    print("[数据源1] 尝试腾讯财经...")
    
    codes = [code for code, _ in STOCKS_TENCENT]
    url = f"https://qt.gtimg.cn/q={','.join(codes)}"
    
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://stock.qq.com/'
        })
        response = urllib.request.urlopen(req, timeout=10, context=ssl_context)
        data = response.read().decode('gbk')
        
        # 解析腾讯数据格式
        quotes = []
        for line in data.strip().split(';'):
            if not line or 'v_' not in line:
                continue
            
            parts = line.split('~')
            if len(parts) < 45:
                continue
            
            code = parts[2]
            name = parts[1]
            price = float(parts[3]) if parts[3] else 0
            prev_close = float(parts[4]) if parts[4] else 0
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
            high = float(parts[33]) if parts[33] else 0
            low = float(parts[34]) if parts[34] else 0
            volume = float(parts[36]) if parts[36] else 0
            
            quotes.append({
                'code': code,
                'name': name,
                'price': price,
                'change_pct': change_pct,
                'high': high,
                'low': low,
                'volume': volume
            })
        
        if quotes:
            print(f"  ✓ 成功获取 {len(quotes)} 只股票数据")
            return quotes
        else:
            print("  ✗ 未获取到数据")
            return None
            
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return None

def fetch_from_sina():
    """从新浪财经获取实时行情"""
    print("[数据源2] 尝试新浪财经...")
    
    codes = [code for code, _ in STOCKS_SINA]
    url = f"https://hq.sinajs.cn/list={','.join(codes)}"
    
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://finance.sina.com.cn/'
        })
        response = urllib.request.urlopen(req, timeout=10, context=ssl_context)
        data = response.read().decode('gbk')
        
        # 解析新浪数据格式
        quotes = []
        for line in data.strip().split('\n'):
            if not line or '=' not in line:
                continue
            
            code_part, data_part = line.split('=')
            code = code_part.split('_')[-1]
            
            fields = data_part.strip('"').split(',')
            if len(fields) < 10:
                continue
            
            name = fields[0]
            price = float(fields[3]) if fields[3] else 0
            prev_close = float(fields[2]) if fields[2] else 0
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
            high = float(fields[4]) if fields[4] else 0
            low = float(fields[5]) if fields[5] else 0
            volume = float(fields[8]) if fields[8] else 0
            
            quotes.append({
                'code': code,
                'name': name,
                'price': price,
                'change_pct': change_pct,
                'high': high,
                'low': low,
                'volume': volume
            })
        
        if quotes:
            print(f"  ✓ 成功获取 {len(quotes)} 只股票数据")
            return quotes
        else:
            print("  ✗ 未获取到数据")
            return None
            
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return None

def fetch_from_eastmoney():
    """从东方财富获取实时行情 (原脚本)"""
    print("[数据源3] 尝试东方财富...")
    
    # 转换代码格式
    A_SHARE_STOCKS = [
        ("1.600519", "贵州茅台"),
        ("1.000858", "五粮液"),
        ("1.600036", "招商银行"),
        ("1.600111", "北方稀土"),
        ("0.002465", "海格通信"),
        ("1.601318", "中国平安"),
        ("1.600900", "长江电力"),
        ("1.601166", "兴业银行"),
        ("1.601888", "中国中免"),
        ("0.000333", "美的集团"),
    ]
    
    secids = ",".join([code for code, _ in A_SHARE_STOCKS])
    url = f"https://push2.eastmoney.com/api/qt/ulist.np/get?fltt=2&secids={secids}&fields=f2,f3,f12,f14,f15,f16,f17,f18,f5,f6"
    
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://quote.eastmoney.com/'
        })
        response = urllib.request.urlopen(req, timeout=15, context=ssl_context)
        data = json.loads(response.read().decode())
        
        if not data or not data.get('data'):
            print("  ✗ 未获取到数据")
            return None
        
        quotes = []
        for item in data['data'].get('diff', []):
            code = item.get('f12', 'N/A')
            name = item.get('f14', 'N/A')
            price = item.get('f2', 0) or 0
            change_pct = item.get('f3', 0) or 0
            high = item.get('f15', 0) or 0
            low = item.get('f16', 0) or 0
            volume = item.get('f5', 0) or 0
            
            quotes.append({
                'code': code,
                'name': name,
                'price': price,
                'change_pct': change_pct,
                'high': high,
                'low': low,
                'volume': volume
            })
        
        if quotes:
            print(f"  ✓ 成功获取 {len(quotes)} 只股票数据")
            return quotes
        else:
            print("  ✗ 未获取到数据")
            return None
            
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return None

def fetch_quotes_with_retry():
    """带重试机制的数据获取"""
    max_retries = 3
    
    for attempt in range(max_retries):
        print(f"\n尝试 {attempt + 1}/{max_retries}:")
        
        # 尝试腾讯
        quotes = fetch_from_tencent()
        if quotes:
            return quotes, 'tencent'
        
        time.sleep(1)
        
        # 尝试新浪
        quotes = fetch_from_sina()
        if quotes:
            return quotes, 'sina'
        
        time.sleep(1)
        
        # 尝试东方财富
        quotes = fetch_from_eastmoney()
        if quotes:
            return quotes, 'eastmoney'
        
        if attempt < max_retries - 1:
            print(f"等待 3 秒后重试...")
            time.sleep(3)
    
    return None, None

def main():
    print("=" * 80)
    print(f"A股实时行情获取 - 多数据源备份版 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("=" * 80)
    
    quotes, source = fetch_quotes_with_retry()
    
    if not quotes:
        print("\n✗ 所有数据源均失败")
        return None
    
    print(f"\n✓ 使用数据源: {source}")
    print(f"✓ 成功获取 {len(quotes)} 只股票数据\n")
    
    # 打印行情表
    print(f"\n{'代码':<12} {'名称':<10} {'现价':>10} {'涨跌%':>8} {'最高':>10} {'最低':>10}")
    print("-" * 80)
    
    for q in quotes:
        print(f"{q['code']:<12} {q['name']:<10} {q['price']:>10.2f} {q['change_pct']:>7.2f}% {q['high']:>10.2f} {q['low']:>10.2f}")
    
    print("=" * 80)
    
    # 保存到文件
    output_file = "/home/emmmoji/.openclaw/workspace-topaz/topaz-v3/data/a_share_quotes_today.json"
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'source': source,
            'quotes': quotes
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据已保存到: {output_file}")
    
    return quotes

if __name__ == "__main__":
    main()
