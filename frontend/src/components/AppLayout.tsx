import { Layout, Menu } from 'antd'
import {
  DashboardOutlined,
  StockOutlined,
  PieChartOutlined,
  WalletOutlined,
  QuestionCircleOutlined,
  SettingOutlined,
} from '@ant-design/icons'
import { Outlet, useNavigate, useLocation } from 'react-router-dom'

const { Sider, Content } = Layout

const menuItems = [
  { key: '/dashboard', icon: <DashboardOutlined />, label: '市场总览' },
  { key: '/stocks', icon: <StockOutlined />, label: '沪深300' },
  { key: '/sectors', icon: <PieChartOutlined />, label: '行业板块' },
  { key: '/portfolio', icon: <WalletOutlined />, label: '我的组合' },
  { key: '/about', icon: <QuestionCircleOutlined />, label: '说明' },
  { key: '/settings', icon: <SettingOutlined />, label: '设置' },
]

export default function AppLayout() {
  const navigate = useNavigate()
  const location = useLocation()

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider width={180} theme="dark">
        <div style={{ height: 48, margin: 16, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontWeight: 600, fontSize: 16 }}>
          Topaz-Next
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
        />
      </Sider>
      <Layout>
        <Content style={{ margin: 16, padding: 24, background: '#141414', borderRadius: 8, overflow: 'auto' }}>
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  )
}
