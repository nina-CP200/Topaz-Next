import { useEffect, useState, useRef } from 'react'
import { Table, Tag, Spin, Input, Select, Space, Button, message, Row, Col, Statistic, Card, Progress, Modal } from 'antd'
import { ReloadOutlined, ArrowUpOutlined, ArrowDownOutlined, SendOutlined, FileTextOutlined } from '@ant-design/icons'
import { getDailyAnalysis, refreshAnalysis, getAnalysisReport, sendSlackReport } from '../api'
import type { DailyAnalysis } from '../api'

const adviceColors: Record<string, string> = {
  '强烈买入': '#f5222d',
  '建议买入': '#ff7a45',
  '可以关注': '#faad14',
  '建议观望': '#d9d9d9',
  '建议回避': '#52c41a',
  '建议卖出': '#237804',
  '强烈建议卖出': '#135200',
}

export default function Stocks() {
  const [data, setData] = useState<DailyAnalysis | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [search, setSearch] = useState('')
  const [adviceFilter, setAdviceFilter] = useState<string | undefined>(undefined)
  const [reportText, setReportText] = useState('')
  const [reportModal, setReportModal] = useState(false)
  const [sending, setSending] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const load = () => {
    setLoading(true)
    getDailyAnalysis()
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    load()
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [])

  const pollStatus = () => {
    if (pollRef.current) clearInterval(pollRef.current)
    setProgress(3)
    let tick = 0
    pollRef.current = setInterval(async () => {
      tick++
      try {
        const r = await fetch('/api/analysis/status')
        const s = await r.json()
        if (s.status === 'done' || s.status === 'error') {
          if (pollRef.current) clearInterval(pollRef.current)
          pollRef.current = null
          setProgress(100)
          setRefreshing(false)
          setTimeout(load, 500)
          if (s.status === 'error') message.error(`分析失败: ${s.message}`)
          return
        }
      } catch {}
      setProgress(p => {
        const step = 100 - p
        const increment = Math.max(0.5, step * 0.08 + Math.random() * 1.5)
        return Math.min(p + increment, 99)
      })
    }, 2500)
  }

  const handleRefresh = () => {
    setRefreshing(true)
    setProgress(2)
    refreshAnalysis()
      .then(pollStatus)
      .catch(() => {
        setRefreshing(false)
        message.error('触发分析失败')
      })
  }

  const results = data?.results || []
  const filtered = results.filter(r => {
    if (search && !r.name.includes(search) && !r.symbol.includes(search)) return false
    if (adviceFilter && r.advice !== adviceFilter) return false
    return true
  })

  const columns = [
    { title: '排名', dataIndex: 'rank', key: 'rank', width: 60 },
    { title: '代码', dataIndex: 'symbol', key: 'symbol', width: 110 },
    { title: '名称', dataIndex: 'name', key: 'name', width: 120 },
    { title: '行业', dataIndex: 'industry', key: 'industry', width: 100 },
    {
      title: '现价', dataIndex: 'current_price', key: 'current_price', width: 80,
      render: (v: number) => v?.toFixed(2),
    },
    {
      title: '涨跌幅', dataIndex: 'change_pct', key: 'change_pct', width: 80,
      render: (v: number) => (
        <span style={{ color: v >= 0 ? '#f5222d' : '#52c41a' }}>
          {v >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />} {Math.abs(v).toFixed(2)}%
        </span>
      ),
    },
    {
      title: 'ML概率', dataIndex: 'probability', key: 'probability', width: 80,
      render: (v: number) => v ? `${(v * 100).toFixed(1)}%` : '-',
    },
    {
      title: '综合评分', dataIndex: 'composite_score', key: 'composite_score', width: 90,
      sorter: (a: any, b: any) => a.composite_score - b.composite_score,
      render: (v: number) => <span style={{ fontWeight: 600, color: v >= 0.65 ? '#f5222d' : v >= 0.45 ? '#faad14' : '#52c41a' }}>{v?.toFixed(3)}</span>,
    },
    {
      title: '建议', dataIndex: 'advice', key: 'advice', width: 100,
      render: (v: string) => <Tag color={adviceColors[v] || '#d9d9d9'}>{v}</Tag>,
    },
    {
      title: '建议仓位', dataIndex: 'position_pct', key: 'position_pct', width: 90,
      render: (v: number) => v ? `${(v * 100).toFixed(0)}%` : '-',
    },
  ]

  return (
    <div>
      {refreshing && (
        <Card size="small" style={{ marginBottom: 16 }}>
          <Progress percent={progress} status="active" strokeColor="#1677ff" />
          <div style={{ textAlign: 'center', color: '#888', marginTop: 4 }}>正在分析沪深300成分股，请稍候...</div>
        </Card>
      )}

      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        {data && (
          <>
            <Col span={4}><Card size="small"><Statistic title="市场状态" value={data.market_regime} /></Card></Col>
            <Col span={4}><Card size="small"><Statistic title="模型置信度" value={`${(data.model_confidence * 100).toFixed(0)}%`} /></Card></Col>
            <Col span={4}><Card size="small"><Statistic title="涨跌比" value={`${(data.advance_ratio * 100).toFixed(1)}%`} /></Card></Col>
            <Col span={4}><Card size="small"><Statistic title="建议仓位" value={`${(data.recommended_position * 100).toFixed(0)}%`} /></Card></Col>
            <Col span={4}><Card size="small"><Statistic title="分析日期" value={data.date} /></Card></Col>
          </>
        )}
      </Row>

      <Space style={{ marginBottom: 16 }}>
        <Input.Search
          placeholder="搜索代码/名称"
          value={search}
          onChange={e => setSearch(e.target.value)}
          style={{ width: 200 }}
          allowClear
        />
        <Select
          placeholder="按建议筛选"
          value={adviceFilter}
          onChange={setAdviceFilter}
          allowClear
          style={{ width: 140 }}
          options={[
            { label: '强烈买入', value: '强烈买入' },
            { label: '建议买入', value: '建议买入' },
            { label: '可以关注', value: '可以关注' },
            { label: '建议观望', value: '建议观望' },
            { label: '建议回避', value: '建议回避' },
            { label: '建议卖出', value: '建议卖出' },
            { label: '强烈建议卖出', value: '强烈建议卖出' },
          ]}
        />
        <Button icon={<ReloadOutlined />} loading={refreshing} onClick={handleRefresh}>
          重新分析
        </Button>
        {data && (
          <>
            <Button icon={<FileTextOutlined />} onClick={async () => {
              const r = await getAnalysisReport()
              if (r.text) { setReportText(r.text); setReportModal(true) }
              else message.error(r.error || '暂无报告')
            }}>生成报告</Button>
            <Button icon={<SendOutlined />} loading={sending} onClick={async () => {
              setSending(true)
              try {
                const r = await sendSlackReport()
                message.success(r.message)
              } catch (e: any) {
                message.error(e?.response?.data?.detail || '发送失败')
              }
              setSending(false)
            }}>发送到Slack</Button>
          </>
        )}
      </Space>

      <Modal
        title="策略报告"
        open={reportModal}
        onCancel={() => setReportModal(false)}
        footer={null}
        width={700}
      >
        <pre style={{
          background: '#1f1f1f', color: '#d9d9d9', padding: 16,
          borderRadius: 8, fontSize: 13, lineHeight: 1.7, whiteSpace: 'pre-wrap',
          maxHeight: 500, overflow: 'auto',
        }}>{reportText}</pre>
      </Modal>

      {loading ? <Spin size="large" style={{ display: 'block', marginTop: 80 }} /> : (
        <Table
          dataSource={filtered}
          columns={columns}
          rowKey="symbol"
          size="small"
          scroll={{ y: 600 }}
          pagination={{ pageSize: 50, showSizeChanger: false }}
        />
      )}
    </div>
  )
}
