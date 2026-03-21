#!/usr/bin/env python3
"""
美股预测脚本
使用训练好的美股模型进行预测
"""

import numpy as np
import pandas as pd
import os
import json
import joblib
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from feature_engineer import FeatureEngineer
from topaz_data_api import get_history_data


def parse_us_stock_list(md_file: str) -> dict:
    """
    从美股关注股票列表.md解析股票代码和行业
    
    Args:
        md_file: md文件路径
        
    Returns:
        {股票代码: 行业} 字典
    """
    industry_map = {}
    
    if not os.path.exists(md_file):
        print(f"⚠️ 美股列表文件不存在: {md_file}")
        return industry_map
    
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 当前板块名（作为行业）
    current_section = 'Other'
    
    for line in content.split('\n'):
        # 检测标题行 ## 板块名
        if line.startswith('## '):
            section_name = line[3:].strip()
            # 映射板块名到行业
            section_map = {
                '科技': 'Technology',
                '金融': 'Financial',
                '消费': 'Consumer',
                '医疗': 'Healthcare',
                'ETF指数': 'ETF',
                '其他': 'Other',
            }
            current_section = section_map.get(section_name, section_name)
            continue
        
        # 跳过表头和分隔符
        if line.startswith('| 股票代码') or line.startswith('|---') or line.startswith('| ---'):
            continue
        
        # 解析表格行 | AAPL | Apple Inc | 科技 |
        if line.startswith('|') and '|' in line[1:]:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 4:
                symbol = parts[1].strip()
                # 验证是有效股票代码（大写字母）
                if symbol and re.match(r'^[A-Z]+$', symbol):
                    industry_map[symbol] = current_section
    
    print(f"📋 从 {md_file} 加载 {len(industry_map)} 只美股")
    return industry_map


# 美股行业分类 - 从 md 文件加载
US_STOCK_LIST_FILE = os.path.join(os.path.dirname(__file__), '美股关注股票列表.md')
US_INDUSTRY_MAP = parse_us_stock_list(US_STOCK_LIST_FILE)

# 如果加载失败，使用默认列表
if not US_INDUSTRY_MAP:
    print("⚠️ 使用默认美股列表")
    US_INDUSTRY_MAP = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'META': 'Technology',
        'NVDA': 'Technology', 'AMD': 'Technology', 'INTC': 'Technology', 'AVGO': 'Technology',
        'CRM': 'Technology', 'ORCL': 'Technology',
        'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'GS': 'Financial',
        'V': 'Financial', 'MA': 'Financial',
        'AMZN': 'Consumer', 'WMT': 'Consumer', 'KO': 'Consumer', 'MCD': 'Consumer',
        'NKE': 'Consumer', 'COST': 'Consumer',
        'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'ABT': 'Healthcare',
        'CAT': 'Industrial', 'BA': 'Industrial', 'GE': 'Industrial', 'HON': 'Industrial',
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
        'T': 'Communication', 'VZ': 'Communication', 'TMUS': 'Communication',
        'NEE': 'Utilities', 'DUK': 'Utilities',
        'AMT': 'RealEstate', 'PLD': 'RealEstate',
        'LIN': 'Materials', 'APD': 'Materials',
    }

LABEL_MEANING = {0: '下跌', 1: '横盘', 2: '上涨'}


def load_model(model_dir: str):
    """加载模型"""
    model_path = os.path.join(model_dir, 'us_ensemble_model.pkl')
    scaler_path = os.path.join(model_dir, 'us_ensemble_scaler.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model_data = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model_data['models'], scaler, model_data['feature_cols']


def fetch_stock_data(symbol: str, days: int = 120) -> pd.DataFrame:
    """
    获取美股历史数据
    
    Args:
        symbol: 股票代码
        days: 获取天数
    
    Returns:
        DataFrame with columns: ticker, date, open, high, low, close, volume
    """
    print(f"  获取 {symbol} 历史数据...")
    
    try:
        # 使用 topaz_data_api 获取数据
        df = get_history_data(symbol, market='美股', days=days)
        
        if df is None or len(df) < 60:
            print(f"  ⚠️ {symbol} 数据不足 (需要至少60天), 获取到 {len(df) if df is not None else 0} 条")
            return None
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        # 处理日期列
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        else:
            print(f"  ⚠️ {symbol} 缺少 date 列")
            return None
        
        # 添加 ticker
        df['ticker'] = symbol
        
        # 确保必要列存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"  ⚠️ {symbol} 缺少列: {missing}")
            return None
        
        # 按日期升序排列
        df = df.sort_values('date').reset_index(drop=True)
        
        return df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        import traceback
        print(f"  ✗ 获取失败: {e}")
        traceback.print_exc()
        return None


def prepare_features(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """准备特征"""
    # 重命名列为特征工程期望的格式 (code -> ticker)
    df = df.rename(columns={'ticker': 'code'})
    
    # 添加行业信息
    df['industry'] = df['code'].map(US_INDUSTRY_MAP).fillna('Technology')
    
    # 生成特征
    fe = FeatureEngineer()
    df = fe.generate_all_features(df)
    
    # 确保所有特征列都存在
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    X = df[feature_cols].values
    return X, df


def predict_stock(symbol: str, models: dict, scaler, feature_cols: list) -> dict:
    """
    预测单只股票
    
    Returns:
        预测结果字典或 None
    """
    # 获取数据
    df = fetch_stock_data(symbol, days=120)
    
    if df is None:
        return None
    
    # 准备特征
    X, df_features = prepare_features(df, feature_cols)
    
    # 使用最新一行
    X_latest = X[-1:].reshape(1, -1)
    
    # 处理 NaN
    X_latest = np.nan_to_num(X_latest, nan=0.0)
    
    # 标准化
    X_scaled = scaler.transform(X_latest)
    
    # 模型预测
    all_probs = []
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_scaled)
            all_probs.append(probs[0])
    
    if not all_probs:
        return None
    
    # 平均概率
    avg_probs = np.mean(all_probs, axis=0)
    prediction = np.argmax(avg_probs)
    confidence = avg_probs[prediction]
    
    # 当前价格信息
    current_price = df.iloc[-1]['close']
    prev_close = df.iloc[-2]['close'] if len(df) > 1 else current_price
    change_pct = (current_price - prev_close) / prev_close * 100
    
    return {
        'symbol': symbol,
        'prediction': LABEL_MEANING[prediction],
        'confidence': float(confidence),
        'current_price': float(current_price),
        'change_pct': float(change_pct),
        'probs': {LABEL_MEANING[i]: float(p) for i, p in enumerate(avg_probs)}
    }


def main(symbols: list = None):
    """
    主预测函数
    
    Args:
        symbols: 要预测的股票列表，None 则使用全部
    """
    print("=" * 60)
    print("Topaz V3 美股预测")
    print("=" * 60)
    print(f"预测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    model_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 加载模型
    print("\n加载模型...")
    models, scaler, feature_cols = load_model(model_dir)
    print(f"模型数量: {len(models)}")
    print(f"特征数量: {len(feature_cols)}")
    
    # 要预测的股票
    if symbols is None:
        symbols = list(US_INDUSTRY_MAP.keys())
    
    print(f"\n预测 {len(symbols)} 只股票...")
    print("-" * 60)
    
    # 预测每只股票
    results = []
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] {symbol}")
        try:
            result = predict_stock(symbol, models, scaler, feature_cols)
            if result:
                results.append(result)
                print(f"  ✓ 预测: {result['prediction']} (置信度: {result['confidence']:.1%})")
                print(f"  当前价格: ${result['current_price']:.2f} ({result['change_pct']:+.2f}%)")
        except Exception as e:
            print(f"  ✗ 预测失败: {e}")
    
    # 汇总
    print("\n" + "=" * 60)
    print("预测汇总")
    print("=" * 60)
    
    up = [r for r in results if r['prediction'] == '上涨']
    down = [r for r in results if r['prediction'] == '下跌']
    neutral = [r for r in results if r['prediction'] == '横盘']
    
    print(f"\n📈 看涨 ({len(up)} 只):")
    for r in sorted(up, key=lambda x: -x['confidence'])[:5]:
        print(f"  {r['symbol']}: ${r['current_price']:.2f} | {r['confidence']:.1%}")
    
    print(f"\n📉 看跌 ({len(down)} 只):")
    for r in sorted(down, key=lambda x: -x['confidence'])[:5]:
        print(f"  {r['symbol']}: ${r['current_price']:.2f} | {r['confidence']:.1%}")
    
    print(f"\n➡️ 横盘 ({len(neutral)} 只)")
    
    # 保存结果
    output_path = os.path.join(model_dir, 'us_predictions_today.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total': len(results),
            'up': len(up),
            'down': len(down),
            'neutral': len(neutral),
            'predictions': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: {output_path}")
    
    return results


if __name__ == '__main__':
    main()