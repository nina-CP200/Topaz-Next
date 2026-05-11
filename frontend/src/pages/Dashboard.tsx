import { useEffect, useState } from 'react'
import { Card, Row, Col, Statistic, Spin, Tag, Typography } from 'antd'
import { ArrowUpOutlined, ArrowDownOutlined, RiseOutlined, FallOutlined } from '@ant-design/icons'
import ReactECharts from 'echarts-for-react'
import { getMarketOverview, getMarketHistory } from '../api'
import type { MarketOverview, KLineItem } from '../api'

const { Title } = Typography

const regimeColors: Record<string, string> = {
  bull: '#52c41a',
  recovery: '#1677ff',
  pullback: '#faad14',
  sideways: '#d9d9d9',
  bear: '#ff4d4f',
}

const regimeLabels: Record<string, string> = {
  bull: '牛市',
  recovery: '复苏',
  pullback: '回调',
  sideways: '震荡',
  bear: '熊市',
}

export default function Dashboard() {
  const [overview, setOverview] = useState<MarketOverview | null>(null)
  const [klineData, setKlineData] = useState<KLineItem[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getMarketOverview(), getMarketHistory(60)])
      .then(([ov, hist]) => {
        setOverview(ov)
        setKlineData(hist.data || [])
      })
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <Spin size="large" style={{ display: 'block', marginTop: 120 }} />

  return (
    <div>
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <Title level={4} style={{ margin: 0 }}>{overview?.index_name}</Title>
                <div style={{ fontSize: 36, fontWeight: 600, marginTop: 8 }}>
                  {overview?.index_price?.toFixed(2)}
                </div>
                <Tag
                  color={overview && overview.index_change_pct >= 0 ? '#f5222d' : '#52c41a'}
                  style={{ fontSize: 14, padding: '2px 8px', marginTop: 4 }}
                >
                  {overview && overview.index_change_pct >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                  {overview ? Math.abs(overview.index_change_pct).toFixed(2) : '-'}%
                </Tag>
              </div>
              <div>
                <Statistic
                  title="市场状态"
                  value={regimeLabels[overview?.regime || ''] || overview?.regime}
                  valueStyle={{ color: regimeColors[overview?.regime || ''] || '#fff' }}
                />
              </div>
              <div>
                <Statistic title="开" value={overview?.index_open?.toFixed(2)} />
                <Statistic title="昨收" value={overview?.index_prev_close?.toFixed(2)} />
              </div>
              <div>
                <Statistic title="高" value={overview?.index_high?.toFixed(2)} />
                <Statistic title="低" value={overview?.index_low?.toFixed(2)} />
              </div>
            </div>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col span={16}>
          <Card title="沪深300 日K线">
            <ReactECharts
              style={{ height: 400 }}
              option={{
                tooltip: { trigger: 'axis' },
                xAxis: { type: 'category', data: klineData.map(d => d.date), axisLabel: { rotate: 45, fontSize: 10 } },
                yAxis: { type: 'value', scale: true },
                grid: { left: 60, right: 20, bottom: 60 },
                dataZoom: [{ type: 'inside' }],
                series: [{
                  type: 'candlestick',
                  data: klineData.map(d => [d.open, d.close, d.low, d.high]),
                  itemStyle: {
                    color: '#f5222d',
                    color0: '#52c41a',
                    borderColor: '#f5222d',
                    borderColor0: '#52c41a',
                  },
                }],
              }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Row gutter={[0, 16]}>
            <Col span={24}>
              <Card>
                <Statistic
                  title="买入阈值"
                  value={overview?.buy_threshold}
                  prefix={<RiseOutlined />}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Card>
            </Col>
            <Col span={24}>
              <Card>
                <Statistic
                  title="卖出阈值"
                  value={overview?.sell_threshold}
                  prefix={<FallOutlined />}
                  valueStyle={{ color: '#ff4d4f' }}
                />
              </Card>
            </Col>
            <Col span={24}>
              <Card>
                <Statistic
                  title="建议仓位上限"
                  value={overview?.position_max ? `${(overview.position_max * 100).toFixed(0)}%` : '-'}
                />
              </Card>
            </Col>
          </Row>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col span={24}>
          <Card title="市场情绪">
            <Row gutter={24}>
              <Col span={4}><Statistic title="涨跌比" value={overview?.sentiment_advance_ratio ? `${(overview.sentiment_advance_ratio * 100).toFixed(1)}%` : '-'} /></Col>
              <Col span={4}><Statistic title="上涨" value={overview?.sentiment_up_count || 0} valueStyle={{ color: '#f5222d' }} /></Col>
              <Col span={4}><Statistic title="下跌" value={overview?.sentiment_down_count || 0} valueStyle={{ color: '#52c41a' }} /></Col>
              <Col span={4}><Statistic title="涨停" value={overview?.sentiment_limit_up || 0} valueStyle={{ color: '#f5222d' }} /></Col>
              <Col span={4}><Statistic title="跌停" value={overview?.sentiment_limit_down || 0} valueStyle={{ color: '#52c41a' }} /></Col>
            </Row>
          </Card>
        </Col>
      </Row>
    </div>
  )
}
