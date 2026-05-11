import { useEffect, useState } from 'react'
import { Table, Spin, Tag, Card, Row, Col, Statistic } from 'antd'
import ReactECharts from 'echarts-for-react'
import { getSectors } from '../api'
import type { SectorAnalysis } from '../api'

export default function Sectors() {
  const [data, setData] = useState<SectorAnalysis | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getSectors()
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <Spin size="large" style={{ display: 'block', marginTop: 80 }} />
  if (!data) return <div>暂无行业数据</div>

  const sectors = data.sectors || []
  const sorted20d = [...sectors].sort((a, b) => b.momentum_20d - a.momentum_20d)

  const barOption = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 120, right: 40, top: 10, bottom: 30 },
    xAxis: { type: 'value' },
    yAxis: {
      type: 'category',
      data: sorted20d.map(s => s.name).reverse(),
      axisLabel: { fontSize: 11 },
    },
    series: [{
      type: 'bar',
      data: sorted20d.map(s => ({
        value: +(s.momentum_20d * 100).toFixed(2),
        itemStyle: { color: s.momentum_20d >= 0 ? '#f5222d' : '#52c41a' },
      })),
      barWidth: 12,
    }],
  }

  const columns = [
    { title: '行业', dataIndex: 'name', key: 'name', width: 120 },
    {
      title: '20日动量', dataIndex: 'momentum_20d', key: 'momentum_20d', width: 100,
      sorter: (a: any, b: any) => a.momentum_20d - b.momentum_20d,
      render: (v: number) => <Tag color={v >= 0 ? '#f5222d' : '#52c41a'}>{(v * 100).toFixed(2)}%</Tag>,
    },
    {
      title: '5日动量', dataIndex: 'momentum_5d', key: 'momentum_5d', width: 100,
      sorter: (a: any, b: any) => a.momentum_5d - b.momentum_5d,
      render: (v: number) => <Tag color={v >= 0 ? '#f5222d' : '#52c41a'}>{(v * 100).toFixed(2)}%</Tag>,
    },
    { title: '成分股数', dataIndex: 'stock_count', key: 'stock_count', width: 80 },
    {
      title: 'Top 股票', dataIndex: 'top_stocks', key: 'top_stocks',
      render: (stocks: { name: string; composite_score: number }[]) =>
        stocks.slice(0, 3).map(s => `${s.name}(${s.composite_score?.toFixed(3)})`).join(' / '),
    },
  ]

  return (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col span={4}>
          <Card size="small">
            <Statistic title="市场状态" value={data.market_regime} />
          </Card>
        </Col>
        <Col span={4}>
          <Card size="small">
            <Statistic title="行业数量" value={sectors.length} />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col span={10}>
          <Card title="20日动量排行" size="small">
            <ReactECharts style={{ height: 500 }} option={barOption} />
          </Card>
        </Col>
        <Col span={14}>
          <Card title="行业详情" size="small">
            <Table
              dataSource={sectors}
              columns={columns}
              rowKey="name"
              size="small"
              pagination={false}
              scroll={{ y: 460 }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  )
}
