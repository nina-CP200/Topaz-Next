#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topaz - 智能股票估值分析系统（V3版）- 完整版
支持：A股（腾讯财经）+ 美股（Finnhub API）
"""

import requests
import re
from datetime import datetime
from typing import Dict, List, Tuple

# ==================== 配置 ====================
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOCK_LIST_FILE = f"{BASE_DIR}/美股关注股票列表.md"
A股_LIST_FILE = f"{BASE_DIR}/A股关注股票列表.md"

# Finnhub API Key - 从环境变量读取
from dotenv import load_dotenv
load_dotenv()

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    raise ValueError("❌ 未找到 FINNHUB_API_KEY，请复制 .env.example 为 .env 并配置 API Key")

# ==================== 数据获取 ====================

def get_finnhub_us_stock(symbol: str, symbol_name: str = None) -> Dict:
    """
    从 Finnhub 获取美股数据
    """
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
        response = requests.get(url, timeout=10)
        
        data = response.json()
        
        # Finnhub 返回格式: {"c": 当前价, "d": 涨跌, "dp": 涨跌幅, "h": 最高, "l": 最低, "o": 开盘, "pc": 昨收, "t": 时间}
        if 'c' in data and data['c'] > 0:
            return {
                'symbol': symbol,
                'name': symbol_name or symbol,
                'current_price': data.get('c', 0),
                'prev_close': data.get('pc', 0),
                'open': data.get('o', 0),
                'change': data.get('dp', 0),  # 涨跌幅%
                'change_amount': data.get('d', 0),  # 涨跌额
                'high': data.get('h', 0),
                'low': data.get('l', 0),
                'currency': 'USD'
            }
        
        return {}

    except Exception as e:
        print(f"获取美股 {symbol} 失败: {e}")
        return {}

def get_finnhub_company_info(symbol: str) -> Dict:
    """
    从 Finnhub 获取公司基本信息（PE、股息等）
    """
    try:
        url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={FINNHUB_API_KEY}"
        response = requests.get(url, timeout=10)
        
        data = response.json()
        
        if data:
            return {
                'market_cap': data.get('marketCapitalization', 0) * 1e6,  # 转换为美元
                'pe_ratio': data.get('peForward', 0) or data.get('peTrailing', 0),
                'dividend_yield': data.get('lastDividend', 0),
                'exchange': data.get('exchange', ''),
                'industry': data.get('finnhubIndustry', ''),
                'weburl': data.get('weburl', '')
            }
        
        return {}

    except Exception as e:
        return {}

def get_tencent_china_stock(symbol: str, symbol_name: str = None) -> Dict:
    """
    从腾讯财经获取A股数据
    """
    try:
        symbol = symbol.upper()
        if symbol.endswith('.SH'):
            api_symbol = 'sh' + symbol.replace('.SH', '')
        elif symbol.endswith('.SZ'):
            api_symbol = 'sz' + symbol.replace('.SZ', '')
        else:
            api_symbol = symbol
            
        url = f"http://qt.gtimg.cn/q={api_symbol}"
        response = requests.get(url, timeout=10)
        
        data = response.text
        if not data.startswith('v_'):
            return {}

        parts = data.split('~')

        if len(parts) >= 55:
            return {
                'symbol': symbol,
                'name': symbol_name or parts[1],
                'current_price': float(parts[3]),
                'prev_close': float(parts[4]),
                'change': float(parts[32]) if parts[32] else 0,  # 涨跌幅%
                'change_amount': float(parts[31]) if parts[31] else 0,
                'volume': float(parts[36]),
                'turnover': float(parts[37]),
                'date': parts[30],
                'pe_ratio': float(parts[43]) * 10 if parts[43] else 0,  # PE需要*10
                'pb_ratio': float(parts[46]) if parts[46] else 0,
                'roe': float(parts[52]) if parts[52] else 0,  # ROE
                'dividend_yield': 0
            }

        return {}

    except Exception as e:
        return {}

# ==================== 多因子分析 ====================

def calculate_factor_scores(data: Dict, is_us_stock: bool = False) -> Dict:
    """
    计算多因子评分（0-100分）
    """
    scores = {'value': 0, 'quality': 0, 'momentum': 0, 'volatility': 0, 'dividend': 0}

    price = data.get('current_price', 0)
    if price == 0:
        return scores

    pe_ratio = data.get('pe_ratio', 0)
    pb_ratio = data.get('pb_ratio', 0)
    roe = data.get('roe', 0)
    change = data.get('change', 0)
    dividend_yield = data.get('dividend_yield', 0)

    # 1. 价值因子
    if pe_ratio > 0:
        if pe_ratio < 15:
            value_score = 100
        elif pe_ratio > 40:
            value_score = 0
        else:
            value_score = 100 - (pe_ratio - 15) * 3
    else:
        value_score = 50  # 无PE数据时给中等分

    if pb_ratio > 0:
        if pb_ratio < 1:
            pb_score = 100
        elif pb_ratio > 5:
            pb_score = 0
        else:
            pb_score = 100 - (pb_ratio - 1) * 25
    else:
        pb_score = 50

    scores['value'] = (value_score + pb_score) / 2

    # 2. 质量因子
    if roe > 20:
        roe_score = 100
    elif roe < 5:
        roe_score = 0
    else:
        roe_score = (roe - 5) * 5

    margin_score = max(0, 100 - pe_ratio * 2.5) if pe_ratio > 0 else 50
    
    if change > 20:
        growth_score = 100
    elif change < -10:
        growth_score = 0
    else:
        growth_score = max(0, (change + 10) * 5)

    scores['quality'] = (roe_score + margin_score + growth_score) / 3

    # 3. 动量因子
    if change > 20:
        momentum_score = 100
    elif change < -20:
        momentum_score = 0
    else:
        momentum_score = (change + 20) * 2.5
    scores['momentum'] = momentum_score

    # 4. 波动因子
    if abs(change) < 2:
        volatility_score = 100
    elif abs(change) > 5:
        volatility_score = 0
    else:
        volatility_score = 100 - abs(change) * 20
    scores['volatility'] = volatility_score

    # 5. 红利因子
    if dividend_yield > 4:
        dividend_score = 100
    elif dividend_yield < 0.5:
        dividend_score = 0
    else:
        dividend_score = dividend_yield * 25
    scores['dividend'] = dividend_score

    return scores

def predict_future_returns(factor_scores: Dict, market_change: float, market: str = '美股') -> Dict:
    """
    基于多因子评分预测未来3个月收益
    保守型参数：
    - 质量因子权重提高（基本面更重要）
    - 波动因子权重提高（低波动优先）
    - 动量因子权重降低（不追涨）
    - A股额外保守
    """
    # 保守型因子权重
    if market == 'A股':
        # A股：格外保守 - 质量和波动权重更高
        weights = {'value': 0.15, 'quality': 0.30, 'momentum': 0.15, 'volatility': 0.25, 'dividend': 0.15}
    else:
        # 美股：保守但适中
        weights = {'value': 0.20, 'quality': 0.28, 'momentum': 0.15, 'volatility': 0.22, 'dividend': 0.15}

    weighted_score = (
        factor_scores['value'] * weights['value'] +
        factor_scores['quality'] * weights['quality'] +
        factor_scores['momentum'] * weights['momentum'] +
        factor_scores['volatility'] * weights['volatility'] +
        factor_scores['dividend'] * weights['dividend']
    )

    base_return = weighted_score
    
    # 保守调整：市场调整更保守
    if market == 'A股':
        # A股预测收益额外保守
        market_factor = 0.7 if market_change > 0 else 0.5
    else:
        market_factor = 0.8 if market_change > 0 else 0.6
    
    market_adjusted_return = base_return * market_factor
    
    # 保守范围：-30% 到 +40%
    predicted_return = max(-30, min(40, market_adjusted_return))

    return {'predicted_return': predicted_return, 'weighted_score': weighted_score, 'base_return': base_return}

def calculate_risk_level(factor_scores: Dict, data: Dict, market_change: float, market: str = '美股') -> tuple:
    """
    计算风险等级（保守型）
    - A股：风险阈值更低，更容易判定为高风险
    """
    risk_level = '低风险'
    risk_factors = []
    base_risk_score = 0

    pe_ratio = data.get('pe_ratio', 0)
    pb_ratio = data.get('pb_ratio', 0)
    roe = data.get('roe', 0)
    change = data.get('change', 0)

    # 保守型风险阈值
    # A股更严格：PE>20就算高，PB>3就算高
    if market == 'A股':
        # A股保守阈值
        if pe_ratio > 20:
            base_risk_score += 35
            risk_factors.append(f"高PE({pe_ratio:.1f})")
        if pe_ratio > 30:
            base_risk_score += 25

        if pb_ratio > 3:
            base_risk_score += 35
            risk_factors.append(f"高PB({pb_ratio:.1f})")
        if pb_ratio > 5:
            base_risk_score += 25

        # 质量风险更严格
        if roe < 10:
            base_risk_score += 30
            risk_factors.append(f"低ROE({roe:.1f}%)")
        if roe < 5:
            base_risk_score += 30

        # 波动风险更敏感
        if abs(change) > 5:
            base_risk_score += 25
            risk_factors.append(f"大波动({change:.1f}%)")
        if abs(change) > 8:
            base_risk_score += 25
    else:
        # 美股保守阈值（比之前严格）
        if pe_ratio > 25:
            base_risk_score += 30
            risk_factors.append(f"高PE({pe_ratio:.1f})")
        if pe_ratio > 40:
            base_risk_score += 20

        if pb_ratio > 4:
            base_risk_score += 30
            risk_factors.append(f"高PB({pb_ratio:.1f})")
        if pb_ratio > 6:
            base_risk_score += 20

        if roe < 8:
            base_risk_score += 25
            risk_factors.append(f"低ROE({roe:.1f}%)")
        if roe < 0:
            base_risk_score += 30

        if abs(change) > 8:
            base_risk_score += 20
            risk_factors.append(f"大波动({change:.1f}%)")
        if abs(change) > 12:
            base_risk_score += 20

    # 市场风险
    if market_change < -1:
        base_risk_score += 15
        risk_factors.append(f"市场下跌({market_change:.1f}%)")

    # A股额外风险加成
    if market == 'A股':
        base_risk_score += 15  # A股市场固有风险加成

    # 保守型风险等级判定（阈值更低）
    if base_risk_score < 20:
        risk_level = '低风险'
    elif base_risk_score < 45:
        risk_level = '中风险'
    elif base_risk_score < 70:
        risk_level = '高风险'
    else:
        risk_level = '极高风险'

    return risk_level, {'risk_score': base_risk_score, 'factors': risk_factors}

def generate_investment_advice(factor_scores: Dict, predicted_return: float, risk_level: str, market: str = '美股') -> tuple:
    """
    生成投资建议（保守型）
    核心原则：低风险 + 高预测收益 才推荐买入
    """
    # 保守判断
    if risk_level in ['极高风险', '高风险']:
        return '建议回避', '高'
    
    if risk_level == '中风险':
        # 中风险：需要预测收益>20%才考虑
        if predicted_return > 20:
            return '建议观望', '中'
        else:
            return '建议回避', '高'
    
    # 低风险
    if predicted_return > 25:
        return '建议买入', '中'
    elif predicted_return > 15:
        return '可考虑买入', '中'
    elif predicted_return > 5:
        return '建议持有', '中'
    else:
        return '建议观望', '中'

# ==================== 主函数 ====================

def parse_stock_list(file_path: str) -> List[Tuple[str, str, str]]:
    """解析股票列表文件"""
    stocks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    sections = content.split('## ')
    for section in sections:
        if not section.strip():
            continue
        lines = section.split('\n')
        category = lines[0].strip() if lines else 'Unknown'

        for line in lines[1:]:
            line = line.strip()
            # 跳过空行、分隔线、标题行
            if not line or line.startswith('|') and '---' in line:
                continue
            # 跳过标题行（包含"代码"或"名称"）
            if '代码' in line or '名称' in line:
                continue
            if line.startswith('|') and '|' in line[1:]:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3 and parts[1] and (parts[1][0].isdigit() or parts[1].isalpha()):
                    stocks.append((parts[1], parts[2], category))

    return stocks

def main():
    print("=" * 80)
    print("Topaz - 智能股票估值分析系统（V3版）")
    print("多因子量化分析 + 未来3个月收益预测")
    print("数据源: Finnhub (美股) + 腾讯财经 (A股)")
    print("=" * 80)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("📂 读取关注股票列表...")
    us_stocks = parse_stock_list(STOCK_LIST_FILE)
    a_stocks = parse_stock_list(A股_LIST_FILE)
    print(f"  美股: {len(us_stocks)} 只")
    print(f"  A股: {len(a_stocks)} 只")
    print()

    all_results = []
    
    # 美股数据获取
    print("📊 获取美股数据 (Finnhub)...")
    for symbol, name, category in us_stocks:
        print(f"  获取 {symbol} ({name})...", end=' ')
        
        data = get_finnhub_us_stock(symbol, name)
        
        if data:
            # 获取公司详细信息（PE等）
            info = get_finnhub_company_info(symbol)
            data.update(info)
            
            print("✓")
            
            factor_scores = calculate_factor_scores(data, True)
            predicted_return = predict_future_returns(factor_scores, data.get('change', 0), '美股')
            risk_level, risk_info = calculate_risk_level(factor_scores, data, data.get('change', 0), '美股')
            advice, confidence = generate_investment_advice(factor_scores, predicted_return['predicted_return'], risk_level, '美股')
            
            all_results.append({
                'market': '美股', 'symbol': symbol, 'name': name,
                'category': category, 'data': data, 'factor_scores': factor_scores,
                'predicted_return': predicted_return, 'risk_level': risk_level,
                'risk_info': risk_info, 'advice': advice, 'confidence': confidence
            })
        else:
            print("✗")

    # A股数据获取
    print("📊 获取A股数据 (腾讯财经)...")
    for symbol, name, category in a_stocks:
        print(f"  获取 {symbol} ({name})...", end=' ')
        
        data = get_tencent_china_stock(symbol, name)
        
        if data:
            print("✓")
            
            factor_scores = calculate_factor_scores(data, False)
            predicted_return = predict_future_returns(factor_scores, data.get('change', 0), 'A股')
            risk_level, risk_info = calculate_risk_level(factor_scores, data, data.get('change', 0), 'A股')
            advice, confidence = generate_investment_advice(factor_scores, predicted_return['predicted_return'], risk_level, 'A股')
            
            all_results.append({
                'market': 'A股', 'symbol': symbol, 'name': name,
                'category': category, 'data': data, 'factor_scores': factor_scores,
                'predicted_return': predicted_return, 'risk_level': risk_level,
                'risk_info': risk_info, 'advice': advice, 'confidence': confidence
            })
        else:
            print("✗")

    print(f"\n  完成！共获取 {len(all_results)} 只股票数据")
    print()

    if not all_results:
        print("未获取到数据，请检查网络连接")
        return

    # 输出结果
    print("=" * 80)
    print("分析结果")
    print("=" * 80)
    print()

    for result in all_results:
        market = result['market']
        symbol = result['symbol']
        name = result['name']
        category = result['category']
        data = result['data']
        
        currency = data.get('currency', '¥')
        price_str = f"${data.get('current_price', 0):.2f}" if currency == 'USD' else f"¥{data.get('current_price', 0):.2f}"
        
        print(f"【{symbol}】{name} ({market})")
        print(f"  行业: {category}")
        print(f"  当前价格: {price_str}")
        print(f"  涨跌幅: {data.get('change', 0):.2f}%")
        
        pe = data.get('pe_ratio', 0)
        if pe > 0:
            print(f"  市盈率: {pe:.2f}")
        
        pb = data.get('pb_ratio', 0)
        if pb > 0:
            print(f"  市净率: {pb:.2f}")
        
        div = data.get('dividend_yield', 0)
        if div > 0:
            print(f"  股息率: {div:.2f}%")
        
        scores = result['factor_scores']
        print(f"  价值因子: {scores['value']:.1f}/100")
        print(f"  质量因子: {scores['quality']:.1f}/100")
        print(f"  动量因子: {scores['momentum']:.1f}/100")
        print(f"  波动因子: {scores['volatility']:.1f}/100")
        print(f"  红利因子: {scores['dividend']:.1f}/100")
        print(f"  未来3个月收益预测: {result['predicted_return']['predicted_return']:.1f}%")
        print(f"  风险等级: {result['risk_level']}")
        
        if result['risk_info']['factors']:
            print(f"  风险因素: {', '.join(result['risk_info']['factors'])}")
        
        print(f"  投资建议: {result['advice']} (置信度: {result['confidence']})")
        print()

    # 总结
    print("=" * 80)
    print("总结")
    print("=" * 80)

    for market in ['美股', 'A股']:
        market_results = [r for r in all_results if r['market'] == market]
        if not market_results:
            continue
            
        print(f"\n【{market}】")
        
        strong_buy = sum(1 for r in market_results if r['advice'].startswith('强烈推荐'))
        buy = sum(1 for r in market_results if r['advice'] == '建议买入')
        hold = sum(1 for r in market_results if r['advice'] in ['建议持有', '可考虑买入'])
        avoid = sum(1 for r in market_results if r['advice'] == '建议回避')

        print(f"  强烈推荐买入: {strong_buy} 只")
        print(f"  建议买入: {buy} 只")
        print(f"  建议持有/观望: {hold} 只")
        print(f"  建议回避: {avoid} 只")

        low = sum(1 for r in market_results if r['risk_level'] == '低风险')
        medium = sum(1 for r in market_results if r['risk_level'] == '中风险')
        high = sum(1 for r in market_results if r['risk_level'] == '高风险')
        very_high = sum(1 for r in market_results if r['risk_level'] == '极高风险')

        print(f"  风险等级: 低{low}/中{medium}/高{high}/极高{very_high}")

    print()
    print("=" * 80)
    print("风险提示")
    print("=" * 80)
    print("1. 本分析仅供参考，不构成投资建议")
    print("2. 未来收益预测基于多因子模型，存在不确定性")
    print("3. 市场有风险，投资需谨慎")
    print("4. 建议结合基本面、行业趋势等多维度分析")
    print("5. 实际投资决策请自行评估风险承受能力")
    print()

if __name__ == '__main__':
    main()
