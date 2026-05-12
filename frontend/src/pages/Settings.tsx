import { useEffect, useState } from 'react'
import { Card, Input, Button, message, Spin, Typography, Alert } from 'antd'
import { getSlackConfig, saveSlackConfig } from '../api'

const { Title, Text, Paragraph } = Typography

export default function Settings() {
  const [token, setToken] = useState('')
  const [channel, setChannel] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    getSlackConfig()
      .then(cfg => {
        setToken(cfg.token || '')
        setChannel(cfg.channel || '')
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const handleSave = () => {
    setSaving(true)
    saveSlackConfig({ token, channel })
      .then(() => message.success('Slack 配置已保存'))
      .catch(() => message.error('保存失败'))
      .finally(() => setSaving(false))
  }

  if (loading) return <Spin size="large" style={{ display: 'block', marginTop: 80 }} />

  return (
    <div style={{ maxWidth: 600 }}>
      <Title level={4}>系统设置</Title>

      <Card title="Slack 推送" size="small" style={{ marginBottom: 16 }}>
        <Paragraph type="secondary">
          配置后将可在沪深300页面一键发送分析报告到 Slack 频道。
        </Paragraph>

        <div style={{ marginBottom: 12 }}>
          <Text>Bot Token</Text>
          <Input.Password
            placeholder="xoxb-xxxxxxxxxxxx-xxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx"
            value={token}
            onChange={e => setToken(e.target.value)}
            style={{ marginTop: 4 }}
          />
        </div>

        <div style={{ marginBottom: 12 }}>
          <Text>频道 ID</Text>
          <Input
            placeholder="C0XXXXXXXX"
            value={channel}
            onChange={e => setChannel(e.target.value)}
            style={{ marginTop: 4 }}
          />
        </div>

        <Button type="primary" loading={saving} onClick={handleSave}>保存</Button>
      </Card>

      <Alert
        type="info"
        showIcon
        message="如何获取 Token？"
        description={
          <ol style={{ margin: 0, paddingLeft: 20 }}>
            <li>访问 <a href="https://api.slack.com/apps" target="_blank">api.slack.com/apps</a> 创建 App</li>
            <li>在 OAuth & Permissions 中添加 Bot Token 权限：<Text code>chat:write</Text>、<Text code>chat:write.public</Text></li>
            <li>Install to Workspace 后复制 Bot User OAuth Token</li>
            <li>在 Slack 中右键频道 → 复制链接 → 从 URL 中提取频道 ID</li>
            <li>邀请 Bot 到频道：<Text code>/invite @Bot名称</Text></li>
          </ol>
        }
      />
    </div>
  )
}
