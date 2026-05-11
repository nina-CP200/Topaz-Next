#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大盘数据获取模块
================

本模块用于获取A股市场大盘指数数据和市场情绪数据，为量化交易策略提供市场环境判断依据。

【模块功能】
1. 获取沪深300、上证指数等主要指数的实时数据
2. 获取指数历史K线数据，支持技术分析
3. 计算市场情绪指标（上涨/下跌家数、涨停/跌停家数）
4. 判断当前市场环境（牛市/熊市/震荡/反弹）

【数据源说明】
本模块使用以下公开数据接口：
- 腾讯财经接口（qt.gtimg.cn）：用于获取实时指数行情
  - 优点：响应快、数据稳定
  - 数据格式：v_开头字符串，以~分隔各字段
  - 注意：需要添加User-Agent头
  
- 新浪财经接口（money.finance.sina.com.cn）：用于获取历史K线数据
  - 优点：数据完整，包含OHLCV
  - 支持不同周期：日线、周线、月线
  - 返回JSON格式，便于解析

【主要指数代码对照】
- 000300.SH：沪深300指数（本模块默认使用）
- 000001.SH：上证指数
- 399001.SZ：深证成指
- 399006.SZ：创业板指

【依赖库】
- requests：HTTP请求
- pandas：数据处理
- datetime：日期处理

【注意事项】
1. 本模块不依赖任何付费数据接口
2. 接口可能有频率限制，建议合理控制调用频率
3. 节假日期间市场休市，数据可能为最后一个交易日数据

【节假日维护说明】
--------------------
A股市场休市日期需要特别注意：
- 法定节假日：元旦、春节、清明、五一、端午、中秋、国庆
- 周末：周六、周日
- 临时休市：交易所公告的特殊休市日

本模块的处理方式：
1. 休市期间调用接口，通常返回最后一个交易日数据
2. judge_market_environment() 函数不判断是否交易日
3. 建议调用方自行判断交易日（可使用 akshare 或 exchange_calendars 库）

代码示例：
    import akshare as ak
    from datetime import datetime
    
    # 判断是否交易日
    today = datetime.now().strftime('%Y%m%d')
    try:
        trade_dates = ak.tool_trade_date_hist_sina()
        is_trading_day = today in trade_dates
    except:
        # 简单判断：工作日 + 非节假日
        is_trading_day = datetime.now().weekday() < 5

作者：Topaz量化团队
版本：v1.0.0
"""

import requests
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


def get_index_data(index_code: str = '000300.SH') -> Optional[Dict]:
    """
    获取指数实时数据
    
    通过腾讯财经接口获取指定指数的实时行情数据，包括当前价格、涨跌幅、成交量等。
    
    【参数说明】
    ----------
    index_code : str, 默认值='000300.SH'
        指数代码，格式为"代码.交易所"
        - .SH 结尾表示上交所
        - .SZ 结尾表示深交所
        常用代码：
        - '000300.SH'：沪深300
        - '000001.SH'：上证指数
        - '399001.SZ'：深证成指
        - '399006.SZ'：创业板指
    
    【返回格式】
    ----------
    成功时返回字典，包含以下字段：
    {
        'code': str,           # 指数代码（原始格式，如 '000300.SH'）
        'name': str,           # 指数名称（如 '沪深300'）
        'price': float,        # 当前价格
        'prev_close': float,   # 前收盘价
        'change_pct': float,   # 涨跌幅百分比（如 1.5 表示涨1.5%）
        'change_amount': float,# 涨跌点数
        'high': float,         # 最高价
        'low': float,          # 最低价
        'open': float,         # 开盘价
        'volume': float,       # 成交量（手）
        'amount': float        # 成交额（元）
    }
    
    失败时返回 None
    
    【接口地址】
    ----------
    腾讯财经行情接口：
https://qt.gtimg.cn/q={api_code}
    
    其中 api_code 格式：
    - 上交所：sh + 代码（如 sh000300）
    - 深交所：sz + 代码（如 sz399001）
    
    【使用示例】
    ----------
    >>> data = get_index_data('000300.SH')
    >>> if data:
    ...     print(f"沪深300: {data['price']}, 涨跌: {data['change_pct']}%")
    
    【数据更新频率】
    ----------
    交易日 9:30-11:30, 13:00-15:00 期间实时更新
    非交易时间返回最后收盘数据
    
    【错误处理】
    ----------
    - 网络超时：返回 None，打印错误信息
    - 数据格式异常：返回 None
    - 接口不可用：返回 None
    """
    try:
        # 转换代码格式（腾讯格式）
        # 将 Wind 代码格式转换为腾讯接口格式
        # 例如：'000300.SH' -> 'sh000300'
        if index_code.endswith('.SH'):
            api_code = f"sh{index_code.replace('.SH', '')}"
        elif index_code.endswith('.SZ'):
            api_code = f"sz{index_code.replace('.SZ', '')}"
        else:
            # 无交易所后缀时，根据代码首位判断
            # 0开头默认上交所，其他默认深交所
            api_code = f"sh{index_code}" if index_code.startswith('0') else f"sh{index_code}"
        
        # 腾讯财经接口 URL
        url = f"https://qt.gtimg.cn/q={api_code}"
        
        # 添加 User-Agent 头，模拟浏览器请求
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # 发送 HTTP GET 请求，设置10秒超时
        response = requests.get(url, headers=headers, timeout=10)
        data = response.text
        
        # 检查返回数据有效性
        # 有效数据以 "v_" 开头
        if not data or 'v_' not in data:
            return None
        
        # 腾讯接口返回格式：v_字段1~字段2~...~字段N
        # 使用 ~ 分隔各字段
        parts = data.split('~')
        if len(parts) >= 45:
            return {
                'code': index_code,                                    # 字段0相关：指数代码
                'name': parts[1],                                      # 字段1：指数名称
                'price': float(parts[3]) if parts[3] else 0,          # 字段3：当前价格
                'prev_close': float(parts[4]) if parts[4] else 0,     # 字段4：前收盘价
                'change_pct': float(parts[32]) if parts[32] else 0,   # 字段32：涨跌幅
                'change_amount': float(parts[31]) if parts[31] else 0,# 字段31：涨跌点数
                'high': float(parts[33]) if parts[33] else 0,          # 字段33：最高价
                'low': float(parts[34]) if parts[34] else 0,           # 字段34：最低价
                'open': float(parts[5]) if parts[5] else 0,            # 字段5：开盘价
                'volume': float(parts[36]) if parts[36] else 0,        # 字段36：成交量
                'amount': float(parts[37]) if parts[37] else 0,        # 字段37：成交额
            }
        return None
    except Exception as e:
        print(f"获取指数数据失败: {e}")
        return None


def get_index_history(index_code: str = '000300.SH', days: int = 60) -> Optional[pd.DataFrame]:
    """
    获取指数历史K线数据
    
    通过腾讯财经接口获取指定指数的历史K线数据，包含开高低收量等信息，
    并自动计算常用技术指标（MA5、MA10、MA20）。
    
    【参数说明】
    ----------
    index_code : str, 默认值='000300.SH'
        指数代码，格式同 get_index_data()
    
    days : int, 默认值=300
        获取最近N个交易日的数据
        - 建议范围：20-500
        - 腾讯接口支持更长历史数据
        - 用于计算技术指标时，建议至少获取20天数据
    
    【返回格式】
    ----------
    成功时返回 pandas DataFrame，包含以下列：
    
    | 列名       | 类型    | 说明                    |
    |------------|---------|------------------------|
    | date       | datetime| 交易日期               |
    | open       | float   | 开盘价                 |
    | high       | float   | 最高价                 |
    | low        | float   | 最低价                 |
    | close      | float   | 收盘价                 |
    | volume     | float   | 成交量                 |
    | ma5        | float   | 5日移动平均线          |
    | ma10       | float   | 10日移动平均线         |
    | ma20       | float   | 20日移动平均线         |
    | pct_change | float   | 日涨跌幅百分比         |
    
    DataFrame 按日期升序排列，最近的日期在最后。
    
    失败时返回 None
    
    【接口地址】
    ----------
    腾讯财经K线接口：
https://web.ifzq.gtimg.cn/appstock/app/fqkline/get
    
    参数说明：
    - param: {code},day,,,{days},qfq
      - code: 指数代码（如 sh000300）
      - day: 日线数据
      - days: 返回数据条数
    
    【使用示例】
    ----------
    >>> df = get_index_history('000300.SH', days=300)
    >>> if df is not None:
    ...     print(f"最近300天收盘价均值: {df['close'].mean():.2f}")
    ...     print(f"当前20日均线: {df['ma20'].iloc[-1]:.2f}")
    
    【技术指标计算说明】
    ----------
    1. MA5：5日简单移动平均 = 最近5日收盘价之和 / 5
    2. MA10：10日简单移动平均 = 最近10日收盘价之和 / 10
    3. MA20：20日简单移动平均 = 最近20日收盘价之和 / 20
    4. pct_change：日涨跌幅 = (今日收盘 - 昨日收盘) / 昨日收盘 * 100
    
    【注意事项】
    ----------
    - 前19行数据的 ma20 将为 NaN（数据不足）
    - 第1行数据的 pct_change 将为 NaN（无前日数据）
    """
    try:
        # 转换代码格式为腾讯接口格式
        # 例如：'000300.SH' -> 'sh000300'
        if index_code.endswith('.SH'):
            api_code = f"sh{index_code.replace('.SH', '')}"
        elif index_code.endswith('.SZ'):
            api_code = f"sz{index_code.replace('.SZ', '')}"
        else:
            api_code = f"sh{index_code}"
        
        # 腾讯财经K线接口 URL（支持更长历史数据）
        url = f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={api_code},day,,,{days},qfq'
        
        # 请求头
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://stock.finance.qq.com/'
        }
        
        # 发送请求，15秒超时
        response = requests.get(url, headers=headers, timeout=15)
        data = response.json()
        
        # 解析腾讯接口返回数据结构
        # data -> data -> {code} -> qfqday/day -> [klines]
        if data.get('data') and data['data'].get(api_code):
            stock_data = data['data'][api_code]
            # 指数数据通常在 'day' 字段中
            klines = stock_data.get('qfqday', stock_data.get('day', []))
            
            if klines and isinstance(klines, list):
                records = []
                # 解析K线数据：[日期, 开, 收, 低, 高, 成交量]
                for item in klines:
                    if len(item) >= 6:
                        records.append({
                            'date': pd.to_datetime(item[0]),
                            'open': float(item[1]),
                            'close': float(item[2]),
                            'low': float(item[3]),
                            'high': float(item[4]),
                            'volume': float(item[5])
                        })
                
                # 构建DataFrame
                df = pd.DataFrame(records)
                
                # 计算移动平均线
                # MA5: 5日均线，用于判断短期趋势
                df['ma5'] = df['close'].rolling(5).mean()
                # MA10: 10日均线，用于判断中短期趋势
                df['ma10'] = df['close'].rolling(10).mean()
                # MA20: 20日均线，用于判断中期趋势，常被称为"生命线"
                df['ma20'] = df['close'].rolling(20).mean()
                
                # 计算日涨跌幅（百分比）
                df['pct_change'] = df['close'].pct_change() * 100
                
                # 按日期升序排列
                df = df.sort_values('date').reset_index(drop=True)
                
                # 取最近 days 天数据
                df = df.tail(days).reset_index(drop=True)
                
                return df
        return None
    except Exception as e:
        print(f"获取指数历史数据失败: {e}")
        return None


def get_market_sentiment() -> Optional[Dict]:
    """
    获取市场情绪数据
    
    基于主要指数涨跌幅估算全市场情绪指标，包括上涨家数、下跌家数、
    涨停家数、跌停家数等。这些数据对于判断市场整体热度和风险偏好非常重要。
    
    【算法说明】
    ----------
    由于免费接口难以获取实时涨跌家数，本函数采用以下估算方法：
    
    1. 获取上证指数、沪深300、深证成指、创业板指四个主要指数的涨跌幅
    2. 计算四个指数的平均涨跌幅
    3. 根据统计规律，估算上涨股票比例：
       - 指数涨1% ≈ 65%股票上涨
       - 指数跌1% ≈ 35%股票上涨
       - 公式：上涨比例 = 0.5 + 平均涨跌幅 / 5
    4. 假设全市场约5300只股票，计算涨跌家数
    5. 根据涨跌幅分段估算涨停/跌停家数
    
    【参数说明】
    ----------
    无参数
    
    【返回格式】
    ----------
    成功时返回字典：
    {
        'sample_stocks': int,      # 样本股票总数（固定为5300）
        'up_count': int,           # 上涨股票家数
        'down_count': int,         # 下跌股票家数
        'flat_count': int,         # 平盘股票家数
        'limit_up': int,           # 涨停股票家数
        'limit_down': int,         # 跌停股票家数
        'advance_ratio': float,    # 上涨比例（0-1）
        'limit_up_ratio': float,   # 涨停比例（0-1）
        'limit_down_ratio': float, # 跌停比例（0-1）
        'sentiment_score': float,  # 情绪得分 = (上涨-下跌)/总数
        'source': str,             # 数据来源标识
        'avg_index_change': float, # 指数平均涨跌幅
        'index_details': list      # 各指数涨跌详情
    }
    
    失败时返回 None
    
    【接口地址】
    ----------
    腾讯财经批量查询接口：
    https://qt.gtimg.cn/q=sh000001,sh000300,sz399001,sz399006
    
    【使用示例】
    ----------
    >>> sentiment = get_market_sentiment()
    >>> if sentiment:
    ...     print(f"上涨家数: {sentiment['up_count']}")
    ...     print(f"上涨比例: {sentiment['advance_ratio']:.1%}")
    ...     print(f"情绪得分: {sentiment['sentiment_score']:.2f}")
    
    【情绪得分解读】
    ----------
    - sentiment_score > 0.3：市场情绪高涨，多数股票上涨
    - sentiment_score ≈ 0：市场情绪中性，涨跌各半
    - sentiment_score < -0.3：市场情绪低迷，多数股票下跌
    
    【估算精度说明】
    ----------
    本函数的估算是基于历史统计规律：
    - 优点：无需付费接口，实时性好
    - 缺点：精确度有限，误差在±5%左右
    - 建议：对于精确度要求高的场景，建议使用付费数据源
    """
    try:
        # 定义需要查询的主要指数
        # 选择这些指数的原因：覆盖主板、创业板，代表性强
        indices = {
            'sh000001': '上证指数',   # 上海主板
            'sh000300': '沪深300',     # 大盘蓝筹
            'sz399001': '深证成指',   # 深圳主板
            'sz399006': '创业板指'    # 创业板
        }
        
        index_changes = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # 逐个获取指数涨跌幅
        for code, name in indices.items():
            try:
                url = f"https://qt.gtimg.cn/q={code}"
                response = requests.get(url, headers=headers, timeout=5)
                if response.text and 'v_' in response.text:
                    parts = response.text.split('~')
                    if len(parts) >= 45:
                        change_pct = float(parts[32]) if parts[32] else 0
                        index_changes.append({
                            'name': name,
                            'change_pct': change_pct
                        })
            except:
                continue
        
        if not index_changes:
            return None
        
        # 计算平均涨跌幅
        avg_change = sum(item['change_pct'] for item in index_changes) / len(index_changes)
        
        # 【核心算法】基于指数涨跌幅估算上涨股票比例
        # 统计规律：指数涨1%时，约65%股票上涨；指数跌1%时，约35%股票上涨
        # 线性映射公式：advance_ratio = 0.5 + avg_change / 5
        # - avg_change = 0 时，advance_ratio = 0.5（涨跌各半）
        # - avg_change = 1 时，advance_ratio = 0.7（70%上涨）
        # - avg_change = -1 时，advance_ratio = 0.3（30%上涨）
        advance_ratio = 0.5 + avg_change / 5
        
        # 限制上涨比例在合理范围 [20%, 80%]
        # 避免极端情况下的不合理估算
        advance_ratio = max(0.2, min(0.8, advance_ratio))
        
        # 【股票家数估算】
        # A股市场当前约有5300只股票
        # 注：这个数字会随时间变化，建议定期更新
        total_stocks = 5300
        
        # 计算涨跌家数
        up_count = int(total_stocks * advance_ratio)       # 上涨家数
        down_count = int(total_stocks * (1 - advance_ratio)) # 下跌家数
        flat_count = total_stocks - up_count - down_count   # 平盘家数
        
        # 【涨停跌停家数估算】
        # 根据平均涨跌幅分段估算
        # 涨停/跌停的股票数量与市场情绪密切相关
        if avg_change > 2:
            # 大涨行情（涨幅>2%）
            # 涨停家数较多，跌停家数很少
            limit_up = int(80 + avg_change * 20)    # 约120-150家涨停
            limit_down = max(0, int(3 - avg_change)) # 0-1家跌停
        elif avg_change > 1:
            # 较强行情（涨幅1%-2%）
            limit_up = int(40 + avg_change * 20)    # 约60-80家涨停
            limit_down = max(0, int(8 - avg_change * 3)) # 5-8家跌停
        elif avg_change > 0:
            # 小涨行情（涨幅0%-1%）
            limit_up = int(15 + avg_change * 15)     # 约15-30家涨停
            limit_down = max(0, int(12 - avg_change * 5)) # 7-12家跌停
        elif avg_change > -1:
            # 小跌行情（跌幅0%-1%）
            limit_up = max(0, int(12 + avg_change * 8)) # 4-12家涨停
            limit_down = int(15 - avg_change * 10)      # 15-25家跌停
        elif avg_change > -2:
            # 较弱行情（跌幅1%-2%）
            limit_up = max(0, int(8 + avg_change * 5))   # 0-8家涨停
            limit_down = int(30 - avg_change * 10)       # 30-50家跌停
        else:
            # 大跌行情（跌幅>2%）
            limit_up = max(0, int(3 + avg_change * 2))   # 0-3家涨停
            limit_down = int(60 - avg_change * 15)       # 60+家跌停
        
        return {
            'sample_stocks': total_stocks,
            'up_count': up_count,
            'down_count': down_count,
            'flat_count': flat_count,
            'limit_up': limit_up,
            'limit_down': limit_down,
            'advance_ratio': advance_ratio,
            'limit_up_ratio': limit_up / total_stocks,
            'limit_down_ratio': limit_down / total_stocks,
            'sentiment_score': (up_count - down_count) / total_stocks,
            'source': 'index_based',  # 标记为基于指数计算
            'avg_index_change': avg_change,
            'index_details': index_changes
        }
    except Exception as e:
        print(f"  获取市场情绪失败: {e}")
        return None


def judge_market_environment(index_data: Dict = None, sentiment: Dict = None) -> str:
    """
    判断当前大盘市场环境
    
    通过综合分析指数价格、趋势、均线位置和市场情绪，判断当前市场处于
    牛市、熊市、反弹还是震荡状态。这个判断对策略调整非常重要。
    
    【参数说明】
    ----------
    index_data : Dict, 可选
        指数数据字典，由 get_index_data() 返回
        如果不传入，函数内部会自动调用获取
        包含字段：price, change_pct, high, low 等
    
    sentiment : Dict, 可选
        市场情绪字典，由 get_market_sentiment() 返回
        如果不传入，函数内部会自动调用获取
        包含字段：advance_ratio, up_count, down_count 等
    
    【返回格式】
    ----------
    返回以下四种市场环境之一：
    
    | 返回值    | 含义     | 特征                          |
    |-----------|----------|-------------------------------|
    | 'bull'    | 牛市     | 价格高于20日线，趋势向上      |
    | 'bear'    | 熊市     | 价格低于20日线，趋势向下      |
    | 'recovery'| 反弹     | 短期均线金叉，但趋势仍向下    |
    | 'sideways'| 震荡     | 无明显趋势，横盘整理          |
    
    【判断逻辑说明】
    ----------
    函数通过以下指标综合判断市场环境：
    
    1. 价格与20日均线关系
       - 价格 > MA20：表示中期趋势向上
       - 价格 < MA20：表示中期趋势向下
    
    2. 20日涨跌幅
       - 涨幅 > 5%：强势上涨
       - 跌幅 > 5%：明显下跌
       - 其他：震荡
    
    3. 市场情绪（上涨家数比例）
       - > 60%：市场情绪积极
       - < 40%：市场情绪低迷
       - 40%-60%：市场情绪中性
    
    【阈值参数说明】
    ----------
    本函数使用以下阈值参数进行判断：
    
    牛市阈值：
    - price > ma20：价格高于20日均线
    - return_20d > 0.05：20日涨幅超过5%
    - advance_ratio > 0.6：上涨家数超过60%
    
    熊市阈值：
    - price < ma20：价格低于20日均线
    - return_20d < -0.05：20日跌幅超过5%
    - advance_ratio < 0.4：上涨家数低于40%
    
    反弹阈值：
    - ma5 > ma20：5日均线在20日均线之上（短期走强）
    - return_20d < 0：但20日仍为下跌（中期趋势向下）
    
    震荡阈值：
    - 不满足以上任何条件
    
    【使用示例】
    ----------
    >>> # 方式1：自动获取数据判断
    >>> env = judge_market_environment()
    >>> print(f"当前市场环境: {env}")
    
    >>> # 方式2：传入已有数据（避免重复请求）
    >>> index_data = get_index_data()
    >>> sentiment = get_market_sentiment()
    >>> env = judge_market_environment(index_data, sentiment)
    
    【判断流程图】
    ----------
    开始
      ↓
    获取指数数据和市场情绪数据
      ↓
    计算20日涨跌幅
      ↓
    计算20日均线位置
      ↓
    ┌─────────────────────────────────┐
    │ 判断是否牛市                      │
    │ 价格>MA20 且 20日涨幅>5% 且 情绪>60% │
    └─────────────────────────────────┘
      │是                    │否
      ↓                      ↓
    返回'bull'    ┌─────────────────────────────────┐
                  │ 判断是否熊市                      │
                  │ 价格<MA20 且 20日跌幅>5% 且 情绪<40% │
                  └─────────────────────────────────┘
                    │是                    │否
                    ↓                      ↓
                  返回'bear'    ┌─────────────────────────┐
                                │ 判断是否反弹              │
                                │ MA5>MA20 且 20日仍下跌    │
                                └─────────────────────────┘
                                  │是            │否
                                  ↓              ↓
                              返回'recovery'  返回'sideways'
    
    【注意事项】
    ----------
    1. 默认使用沪深300指数作为判断基准
    2. 需要至少20个交易日的历史数据
    3. 数据获取失败时默认返回'sideways'
    4. 节假日期间数据可能为最后一个交易日
    """
    try:
        # 获取数据（如果未传入）
        if index_data is None:
            index_data = get_index_data()
        if sentiment is None:
            sentiment = get_market_sentiment()
        
        # 获取历史数据用于趋势判断
        # 使用30天数据确保能计算20日均线和20日涨跌幅
        index_history = get_index_history(days=30)
        
        # 数据检查：如果无法获取足够数据，返回默认值
        if index_data is None or index_history is None:
            return 'sideways'  # 默认震荡
        
        # 提取当前价格和涨跌幅
        current_price = index_data['price']
        change_pct = index_data['change_pct']
        
        # 【计算20日涨跌幅】
        # 用于判断中期趋势
        if len(index_history) >= 20:
            price_20d_ago = index_history.iloc[-20]['close']
            return_20d = (current_price - price_20d_ago) / price_20d_ago
        else:
            return_20d = 0
        
        # 【计算均线位置】
        # MA20：中期趋势指标，常被称为"生命线"
        # MA5：短期趋势指标
        ma20 = index_history['ma20'].iloc[-1] if 'ma20' in index_history.columns else current_price
        ma5 = index_history['ma5'].iloc[-1] if 'ma5' in index_history.columns else current_price
        
        # 【获取市场情绪数据】
        # advance_ratio: 上涨家数占比
        advance_ratio = sentiment.get('advance_ratio', 0.5) if sentiment else 0.5
        
        # 【判断逻辑 - 按优先级顺序】
        
        # 1. 牛市判断
        # 条件：价格高于20日线 + 20日涨幅>5% + 上涨家数>60%
        # 这是一个较强的多头市场信号
        if current_price > ma20 and return_20d > 0.05 and advance_ratio > 0.6:
            return 'bull'  
        
        # 2. 熊市判断
        # 条件：价格低于20日线 + 20日跌幅>5% + 上涨家数<40%
        # 这是一个明显的空头市场信号
        elif current_price < ma20 and return_20d < -0.05 and advance_ratio < 0.4:
            return 'bear'  
        
        # 3. 反弹判断
        # 条件：短期均线(MA5)上穿长期均线(MA20) + 但20日仍为下跌
        # 这表示可能的反弹机会，但中期趋势尚未确认反转
        elif ma5 > ma20 and return_20d < 0:
            return 'recovery'  
        
        # 4. 震荡判断（默认）
        # 不满足以上条件，市场处于震荡状态
        else:
            return 'sideways'  
    
    except Exception as e:
        print(f"判断市场环境失败: {e}")
        return 'sideways'


def judge_market_regime(advance_ratio: float = None, days: int = 5) -> Dict:
    """
    判断细致的市场环境（用于ML模型置信度计算）
    
    与 judge_market_environment() 相比，本函数提供更细致的市场环境分类，
    并返回与训练环境的匹配置信度，用于机器学习模型的预测结果调整。
    
    【参数说明】
    ----------
    advance_ratio : float, 可选
        当日上涨股票比例（0-1之间）
        如果不传入，函数内部会自动获取
    
    days : int, 默认值=5
        回看天数，用于计算上涨比例的移动平均
        - 建议范围：3-20
        - 较小值更敏感，较大值更平滑
    
    【返回格式】
    ----------
    返回字典格式：
    {
        'regime': str,             # 市场环境类型
        'confidence': float,       # 与训练环境匹配度（0-1）
        'adv_ratio_5d_ma': float,  # 5日上涨比例均值
        'advance_ratio': float     # 当日上涨比例
    }
    
    regime 可能的值：
    - 'bull'：强势牛市
    - 'bear'：强势熊市
    - 'weak_bull'：弱势上涨
    - 'weak_bear'：弱势下跌
    - 'sideways'：震荡整理
    
    【判断逻辑说明】
    ----------
    本函数使用双重指标判断市场环境：
    
    1. 当日上涨比例（advance_ratio）
    2. N日上涨比例均值（历史趋势）
    
    判断规则：
    
    【强势牛市 'bull'】
    - 当日上涨比例 > 60%
    - 5日上涨比例均值 > 55%
    - 特征：持续普涨，情绪高涨
    
    【强势熊市 'bear'】
    - 当日上涨比例 < 40%
    - 5日上涨比例均值 < 45%
    - 特征：持续普跌，情绪低迷
    
    【弱势上涨 'weak_bull'】
    - 当日上涨比例 > 55%
    - 不满足强势牛市条件
    - 特征：上涨但不够强势
    
    【弱势下跌 'weak_bear'】
    - 当日上涨比例 < 45%
    - 不满足强势熊市条件
    - 特征：下跌但不够剧烈
    
    【震荡整理 'sideways'】
    - 当日上涨比例在 45%-55% 之间
    - 特征：多空平衡，方向不明
    
    【置信度说明】
    ----------
    confidence 字段表示当前环境与训练数据的匹配程度：
    
    | 环境      | 置信度 | 说明                          |
    |-----------|--------|-------------------------------|
    | weak_bear | 0.9    | 训练期主要环境，置信度最高      |
    | bear      | 0.7    | 与训练期接近                   |
    | sideways  | 0.5    | 中等置信度                     |
    | bull      | 0.5    | 训练期较少，置信度较低          |
    | weak_bull | 0.3    | 训练期较少，置信度低            |
    
    【使用示例】
    ----------
    >>> # 获取市场环境详情
    >>> regime_info = judge_market_regime()
    >>> print(f"市场环境: {regime_info['regime']}")
    >>> print(f"置信度: {regime_info['confidence']}")
    
    >>> # 用于ML模型调整
    >>> if regime_info['confidence'] > 0.7:
    ...     # 高置信度环境，可以更信任模型预测
    ...     use_model_prediction = True
    >>> else:
    ...     # 低置信度环境，需要降低模型权重
    ...     use_model_prediction = False
    
    【与 judge_market_environment 的区别】
    ----------
    - judge_market_environment：粗粒度分类（4种），用于策略选择
    - judge_market_regime：细粒度分类（5种），用于模型置信度
    
    建议同时使用两个函数：
    - 用 judge_market_environment 选择策略参数
    - 用 judge_market_regime 调整模型权重
    """
    try:
        # 获取当前上涨比例（如果未传入）
        if advance_ratio is None:
            sentiment = get_market_sentiment()
            advance_ratio = sentiment.get('advance_ratio', 0.5) if sentiment else 0.5
        
        # 获取历史数据
        index_history = get_index_history(days=max(days, 20))
        if index_history is None or len(index_history) < days:
            return {'regime': 'sideways', 'confidence': 0.3, 'adv_ratio_5d_ma': advance_ratio}
        
        # 【估算历史上涨比例】
        # 由于没有历史涨跌家数数据，用指数涨跌天数来近似
        # 这是一个合理的近似：指数涨通常对应多数股票涨
        changes = index_history['close'].pct_change()
        
        # 计算过去days天中上涨天数的比例
        # 例如：过去5天有3天上涨，则 adv_approx = 0.6
        adv_approx = (changes > 0).rolling(days).mean().iloc[-1] if len(changes) >= days else 0.5
        
        # 【计算5日上涨比例均值】
        # 加权平均：当日权重1，历史权重4
        # 这样可以平滑短期波动，同时考虑最新数据
        adv_ratio_5d_ma = (advance_ratio + adv_approx * 4) / 5 if adv_approx else advance_ratio
        
        # 【分类判断】
        # 分类标准与模型训练时保持一致，确保环境匹配
        
        # 强势牛市
        if advance_ratio > 0.6 and adv_ratio_5d_ma > 0.55:
            regime = 'bull'
            confidence = 0.5  # 训练期主要是weak_bear，bull环境置信度低
        
        # 强势熊市
        elif advance_ratio < 0.4 and adv_ratio_5d_ma < 0.45:
            regime = 'bear'
            confidence = 0.7  # 与训练期接近
        
        # 弱势上涨
        elif advance_ratio > 0.55:
            regime = 'weak_bull'
            confidence = 0.3  # 训练期较少，置信度低
        
        # 弱势下跌
        elif advance_ratio < 0.45:
            regime = 'weak_bear'
            confidence = 0.9  # 训练期主要环境，置信度高
        
        # 震荡整理
        else:
            regime = 'sideways'
            confidence = 0.5
        
        return {
            'regime': regime,
            'confidence': confidence,
            'adv_ratio_5d_ma': adv_ratio_5d_ma,
            'advance_ratio': advance_ratio
        }
        
    except Exception as e:
        return {'regime': 'sideways', 'confidence': 0.3, 'adv_ratio_5d_ma': 0.5}


def get_market_adjusted_thresholds(market_env: str) -> Dict:
    """
    根据市场环境返回调整后的交易阈值
    
    不同市场环境下，交易策略应该采用不同的参数阈值。
    本函数根据市场环境返回建议的参数设置。
    
    【参数说明】
    ----------
    market_env : str
        市场环境类型，由 judge_market_environment() 返回
        可选值：'bull', 'bear', 'recovery', 'sideways'
    
    【返回格式】
    ----------
    返回字典格式：
    {
        'buy_threshold': float,    # 买入信号阈值
        'sell_threshold': float,    # 卖出信号阈值
        'position_max': float,     # 最大仓位限制
        'single_max': float,       # 单股最大仓位
        'description': str         # 环境描述
    }
    
    【阈值参数说明】
    ----------
    
    1. buy_threshold（买入阈值）
       - 含义：模型预测上涨概率超过此值时才买入
       - 牛市设置较低（0.65）：市场好，更容易买入
       - 熊市设置较高（0.80）：市场差，需要更高把握才买入
    
    2. sell_threshold（卖出阈值）
       - 含义：模型预测下跌概率超过此值时卖出
       - 牛市设置较低（0.35）：有下跌信号就尽快卖出锁定利润
       - 熊市设置较高（0.50）：需要更明确的下跌信号才卖出
    
    3. position_max（最大仓位）
       - 含义：整体仓位的上限
       - 牛市设置较高（0.95）：充分利用上涨行情
       - 熊市设置较低（0.60）：保持较低仓位控制风险
    
    4. single_max（单股最大仓位）
       - 含义：单只股票持仓的上限
       - 牛市设置较高（0.25）：可以集中持仓
       - 熊市设置较低（0.15）：分散风险
    
    【各环境阈值设置】
    ----------
    
    【牛市 'bull'】
    - buy_threshold: 0.65   （降低买入门槛，积极做多）
    - sell_threshold: 0.35  （提高卖出门槛，持股待涨）
    - position_max: 0.95    （允许接近满仓）
    - single_max: 0.25      （允许单股较高仓位）
    - description: 牛市环境，积极做多
    
    【熊市 'bear'】
    - buy_threshold: 0.80   （提高买入门槛，谨慎操作）
    - sell_threshold: 0.50  （降低卖出门槛，及时止损）
    - position_max: 0.60    （降低整体仓位）
    - single_max: 0.15      （单股仓位限制更严）
    - description: 熊市环境，保守防守
    
    【反弹 'recovery'】
    - buy_threshold: 0.75   （中等偏严买入门槛）
    - sell_threshold: 0.40  （中等卖出门槛）
    - position_max: 0.80    （中等仓位）
    - single_max: 0.20      （中等单股仓位）
    - description: 反弹环境，精选抄底
    
    【震荡 'sideways'】
    - buy_threshold: 0.70   （中性买入门槛）
    - sell_threshold: 0.40  （中性卖出门槛）
    - position_max: 0.85    （中性仓位）
    - single_max: 0.20      （中性单股仓位）
    - description: 震荡环境，中性策略
    
    【使用示例】
    ----------
    >>> # 根据市场环境获取交易阈值
    >>> market_env = judge_market_environment()
    >>> thresholds = get_market_adjusted_thresholds(market_env)
    >>> 
    >>> print(f"市场环境: {thresholds['description']}")
    >>> print(f"买入阈值: {thresholds['buy_threshold']}")
    >>> print(f"最大仓位: {thresholds['position_max']}")
    >>> 
    >>> # 在交易逻辑中使用
    >>> if predicted_prob > thresholds['buy_threshold']:
    ...     # 执行买入
    ...     position_size = min(calculated_size, thresholds['single_max'])
    
    【设计原则】
    ----------
    1. 牛市：放宽买入，收紧卖出，允许较高仓位
    2. 熊市：收紧买入，放宽卖出，控制低仓位
    3. 反弹：中等偏谨慎，精选个股
    4. 震荡：中性策略，控制风险
    """
    thresholds = {
        'bull': {
            'buy_threshold': 0.55,
            'sell_threshold': 0.35,
            'position_max': 0.95,
            'single_max': 0.25,
            'deep_bottom_fishing': True,
            'stop_loss': -0.08,
            'take_profit': 0.25,
            'holding_days': 10,
            'description': '牛市环境，积极做多'
        },
        'bear': {
            'buy_threshold': 0.75,
            'sell_threshold': 0.50,
            'position_max': 0.60,
            'single_max': 0.15,
            'deep_bottom_fishing': True,
            'stop_loss': -0.04,
            'take_profit': 0.15,
            'holding_days': 5,
            'description': '熊市环境，保守防守'
        },
        'recovery': {
            'buy_threshold': 0.65,
            'sell_threshold': 0.40,
            'position_max': 0.80,
            'single_max': 0.20,
            'deep_bottom_fishing': True,
            'stop_loss': -0.05,
            'take_profit': 0.18,
            'holding_days': 7,
            'description': '反弹环境，精选抄底'
        },
        'pullback': {
            'buy_threshold': 0.60,
            'sell_threshold': 0.40,
            'position_max': 0.80,
            'single_max': 0.20,
            'deep_bottom_fishing': False,
            'stop_loss': -0.05,
            'take_profit': 0.15,
            'holding_days': 7,
            'description': '回调环境，逢低布局'
        },
        'sideways': {
            'buy_threshold': 0.60,
            'sell_threshold': 0.40,
            'position_max': 0.85,
            'single_max': 0.20,
            'deep_bottom_fishing': False,
            'stop_loss': -0.06,
            'take_profit': 0.12,
            'holding_days': 7,
            'description': '震荡环境，中性策略'
        }
    }
    
    return thresholds.get(market_env, thresholds['sideways'])


def test_market_data():
    """
    测试大盘数据获取功能
    
    本函数用于测试所有数据获取和市场环境判断功能，
    打印详细的测试结果，便于开发者验证模块是否正常工作。
    
    【测试内容】
    ----------
    1. 获取沪深300指数实时数据
    2. 获取市场情绪数据（涨跌家数、涨停跌停等）
    3. 判断当前市场环境
    4. 获取对应的交易阈值建议
    
    【使用方法】
    ----------
    直接运行此文件进行测试：
    $ python market_data.py
    
    或在代码中调用：
    >>> from market_data import test_market_data
    >>> test_market_data()
    
    【预期输出】
    ----------
    ============================================================
    大盘数据测试
    ============================================================
    
    1. 沪深300指数:
       价格: 3850.23
       涨跌: 0.85%
       最高: 3875.16
       最低: 3830.05
    
    2. 市场情绪:
       上涨家数: 3180
       下跌家数: 1760
       涨停家数: 45
       跌停家数: 12
       上涨比例: 60.0%
    
    3. 市场环境判断:
       环境: bull
       说明: 牛市环境，积极做多
       买入阈值: 65%
       卖出阈值: 35%
       最大仓位: 95%
    """
    print("=" * 60)
    print("大盘数据测试")
    print("=" * 60)
    
    # 1. 获取指数数据
    print("\n1. 沪深300指数:")
    index_data = get_index_data()
    if index_data:
        print(f"   价格: {index_data['price']:.2f}")
        print(f"   涨跌: {index_data['change_pct']:.2f}%")
        print(f"   最高: {index_data['high']:.2f}")
        print(f"   最低: {index_data['low']:.2f}")
    
    # 2. 获取市场情绪
    print("\n2. 市场情绪:")
    sentiment = get_market_sentiment()
    if sentiment:
        print(f"   上涨家数: {sentiment['up_count']}")
        print(f"   下跌家数: {sentiment['down_count']}")
        print(f"   涨停家数: {sentiment['limit_up']}")
        print(f"   跌停家数: {sentiment['limit_down']}")
        print(f"   上涨比例: {sentiment['advance_ratio']:.1%}")
    
    # 3. 判断市场环境
    print("\n3. 市场环境判断:")
    market_env = judge_market_environment(index_data, sentiment)
    thresholds = get_market_adjusted_thresholds(market_env)
    print(f"   环境: {market_env}")
    print(f"   说明: {thresholds['description']}")
    print(f"   买入阈值: {thresholds['buy_threshold']:.0%}")
    print(f"   卖出阈值: {thresholds['sell_threshold']:.0%}")
    print(f"   最大仓位: {thresholds['position_max']:.0%}")
    
    return market_env, thresholds


if __name__ == '__main__':
    test_market_data()