import { useEffect, useState, useCallback } from 'react'
import {
  Table, Button, InputNumber, Input, Space, message, Spin,
  Card, Row, Col, Statistic, Tag, Popconfirm, Typography,
} from 'antd'
import { PlusOutlined, DeleteOutlined, ThunderboltOutlined } from '@ant-design/icons'
import ReactECharts from 'echarts-for-react'
import { getPortfolio, savePortfolio, analyzePortfolio } from '../api'
import type { HoldingItem, PortfolioAnalysis } from '../api'

const { Text } = Typography

const adviceColors: Record<string, string> = {
  '强烈建议买入': '#f5222d',
  '建议买入': '#ff7a45',
  '可以关注': '#faad14',
  '建议观望': '#d9d9d9',
  '建议回避': '#52c41a',
  '建议卖出': '#237804',
  '强烈建议卖出': '#135200',
}

const LOCAL_KEY = 'topaz_portfolio'

export default function Portfolio() {
  const [holdings, setHoldings] = useState<HoldingItem[]>([])
  const [loading, setLoading] = useState(true)
  const [analysis, setAnalysis] = useState<PortfolioAnalysis | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [saving, setSaving] = useState(false)

  const loadFromLocal = useCallback(() => {
    const saved = localStorage.getItem(LOCAL_KEY)
    if (saved) {
      try { return JSON.parse(saved) as HoldingItem[] } catch { return null }
    }
    return null
  }, [])

  useEffect(() => {
    const local = loadFromLocal()
    if (local) {
      setHoldings(local)
      setLoading(false)
    }
    getPortfolio()
      .then(r => {
        if (r.holdings?.length) {
          setHoldings(r.holdings)
          localStorage.setItem(LOCAL_KEY, JSON.stringify(r.holdings))
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [loadFromLocal])

  const persistToBackend = useCallback((list: HoldingItem[]) => {
    setSaving(true)
    localStorage.setItem(LOCAL_KEY, JSON.stringify(list))
    savePortfolio(list)
      .catch(() => message.warning('后端保存失败，已保留本地数据'))
      .finally(() => setSaving(false))
  }, [])

  const addRow = () => {
    const newHolding: HoldingItem = { symbol: '', name: '', shares: 100, cost: 10 }
    const list = [...holdings, newHolding]
    setHoldings(list)
  }

  const deleteRow = (idx: number) => {
    const list = holdings.filter((_, i) => i !== idx)
    setHoldings(list)
    persistToBackend(list)
  }

  const updateField = (idx: number, field: keyof HoldingItem, value: string | number) => {
    const list = holdings.map((h, i) => (i === idx ? { ...h, [field]: value } : h))
    setHoldings(list)
  }

  const handleAnalyze = () => {
    if (!holdings.length) { message.warning('请先添加持仓'); return }
    setAnalyzing(true)
    persistToBackend(holdings)
    analyzePortfolio(holdings)
      .then(setAnalysis)
      .catch((e: any) => message.error(e?.response?.data?.detail || '分析失败'))
      .finally(() => setAnalyzing(false))
  }

  const columns = [
    {
      title: '代码', dataIndex: 'symbol', key: 'symbol', width: 120,
      render: (_: any, __: any, idx: number) => (
        <Input
          size="small"
          placeholder="如 600519"
          value={holdings[idx].symbol}
          onChange={e => updateField(idx, 'symbol', e.target.value)}
          onBlur={() => persistToBackend(holdings)}
        />
      ),
    },
    {
      title: '名称', dataIndex: 'name', key: 'name', width: 120,
      render: (_: any, __: any, idx: number) => (
        <Input
          size="small"
          placeholder="自动填充"
          value={holdings[idx].name}
          onChange={e => updateField(idx, 'name', e.target.value)}
          onBlur={() => persistToBackend(holdings)}
        />
      ),
    },
    {
      title: '股数', dataIndex: 'shares', key: 'shares', width: 100,
      render: (_: any, __: any, idx: number) => (
        <InputNumber
          size="small"
          min={1}
          value={holdings[idx].shares}
          onChange={v => updateField(idx, 'shares', v || 0)}
          onBlur={() => persistToBackend(holdings)}
          style={{ width: '100%' }}
        />
      ),
    },
    {
      title: '成本价', dataIndex: 'cost', key: 'cost', width: 100,
      render: (_: any, __: any, idx: number) => (
        <InputNumber
          size="small"
          min={0.01}
          step={0.01}
          value={holdings[idx].cost}
          onChange={v => updateField(idx, 'cost', v || 0)}
          onBlur={() => persistToBackend(holdings)}
          style={{ width: '100%' }}
          prefix="¥"
        />
      ),
    },
    {
      title: '操作', key: 'action', width: 60,
      render: (_: any, __: any, idx: number) => (
        <Popconfirm title="删除这支持仓？" onConfirm={() => deleteRow(idx)}>
          <Button size="small" danger icon={<DeleteOutlined />} />
        </Popconfirm>
      ),
    },
  ]

  if (loading) return <Spin size="large" style={{ display: 'block', marginTop: 80 }} />

  return (
    <div>
      <Space style={{ marginBottom: 16 }}>
        <Button icon={<PlusOutlined />} onClick={addRow}>添加持仓</Button>
        <Button
          type="primary"
          icon={<ThunderboltOutlined />}
          loading={analyzing}
          onClick={handleAnalyze}
        >
          分析当前组合
        </Button>
        {saving && <Spin size="small" />}
      </Space>

      <Table
        dataSource={holdings}
        columns={columns}
        rowKey={(_, idx) => String(idx)}
        size="small"
        pagination={false}
        style={{ marginBottom: 24 }}
      />

      {analysis && (
        <div>
          {(analysis as any).failed?.length > 0 && (
            <Card size="small" style={{ marginBottom: 16, borderColor: '#ff4d4f' }}>
              <Text type="danger" strong>以下股票分析失败：</Text>
              {(analysis as any).failed.map((f: any, i: number) => (
                <div key={i} style={{ marginTop: 4 }}>代码 {f.code} — {f.reason}</div>
              ))}
            </Card>
          )}
          <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
            <Col span={4}>
              <Card size="small">
                <Statistic title="总市值" value={analysis.total_value?.toFixed(2)} prefix="¥" />
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small">
                <Statistic title="总成本" value={analysis.total_cost?.toFixed(2)} prefix="¥" />
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small">
                <Statistic
                  title="总盈亏"
                  value={analysis.total_pnl?.toFixed(2)}
                  prefix="¥"
                  valueStyle={{ color: analysis.total_pnl >= 0 ? '#f5222d' : '#52c41a' }}
                />
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small">
                <Statistic
                  title="盈亏%"
                  value={analysis.total_pnl_pct?.toFixed(2)}
                  suffix="%"
                  valueStyle={{ color: analysis.total_pnl_pct >= 0 ? '#f5222d' : '#52c41a' }}
                />
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small">
                <Statistic title="买入建议" value={analysis.buy_count} valueStyle={{ color: '#f5222d' }} />
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small">
                <Statistic title="亏损数量" value={analysis.loss_count} valueStyle={{ color: '#52c41a' }} />
              </Card>
            </Col>
          </Row>

          <Row gutter={[16, 16]}>
            <Col span={8}>
              <Card title="行业分布" size="small">
                <ReactECharts
                  style={{ height: 300 }}
                  option={{
                    tooltip: { trigger: 'item', formatter: '{b}: {c}%' },
                    series: [{
                      type: 'pie',
                      radius: ['40%', '70%'],
                      data: Object.entries(analysis.industry_allocation || {}).map(([k, v]) => ({
                        name: k,
                        value: +(v as any).pct?.toFixed(1) || 0,
                      })),
                      label: { formatter: '{b}\n{d}%' },
                    }],
                  }}
                />
              </Card>
            </Col>
            <Col span={16}>
                  <Card title="持仓建议" size="small">
                {analysis.holdings.map(h => (
                  <Card
                    key={h.symbol}
                    size="small"
                    style={{ marginBottom: 8 }}
                    extra={<Tag color={adviceColors[h.advice] || '#d9d9d9'}>{h.advice}</Tag>}
                  >
                    <Row gutter={16}>
                      <Col span={4}><Text strong>{h.name}</Text><br /><Text type="secondary">{h.symbol}</Text></Col>
                      <Col span={3}><Text>盈亏</Text><br /><Text style={{ color: h.pnl >= 0 ? '#f5222d' : '#52c41a' }}>{h.pnl_pct?.toFixed(2)}%</Text></Col>
                      <Col span={2}><Text>评分</Text><br /><Text>{h.composite_score?.toFixed(3)}</Text></Col>
                      <Col span={2}><Text>现价</Text><br /><Text>¥{h.current_price?.toFixed(2)}</Text></Col>
                      <Col span={2}><Text>持股</Text><br /><Text>{h.shares}股</Text></Col>
                      <Col span={2}><Text>RSI</Text><br /><Text>{h.rsi?.toFixed(1)}</Text></Col>
                      <Col span={9}><Text>理由</Text><br /><Text type="secondary">{h.reasons?.join('; ')}</Text></Col>
                    </Row>
                  </Card>
                ))}
              </Card>
            </Col>
          </Row>
        </div>
      )}
    </div>
  )
}
