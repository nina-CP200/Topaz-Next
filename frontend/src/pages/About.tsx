import { Card, Typography, Tag } from 'antd'

const { Title, Text, Paragraph } = Typography

export default function About() {
  return (
    <div style={{ maxWidth: 800 }}>
      <Title level={3}>Topaz-Next 使用说明</Title>

      <Card size="small" style={{ marginBottom: 16 }}>
        <Title level={5}>系统概述</Title>
        <Paragraph>
          Topaz-Next 是一个本地运行的量化分析工具，基于沪深300成分股的市场数据和机器学习模型，
          提供市场环境判断、个股评分排名、行业板块分析和组合持仓诊断。
        </Paragraph>
      </Card>

      <Card size="small" style={{ marginBottom: 16 }}>
        <Title level={5}>页面说明</Title>

        <Text strong>市场总览</Text>
        <Paragraph style={{ marginBottom: 8 }}>
          展示沪深300指数实时行情、日K线图、市场情绪（涨跌比/涨停跌停）和当前市场状态下的仓位建议阈值。
        </Paragraph>

        <Text strong>沪深300</Text>
        <Paragraph style={{ marginBottom: 8 }}>
          对300只成分股进行综合评分排序。支持按代码/名称搜索、按建议类型筛选。点击"重新分析"触发完整的ML模型预测+技术因子计算，
          整个过程约1-2分钟，进度条会指示当前状态。
        </Paragraph>

        <Text strong>行业板块</Text>
        <Paragraph style={{ marginBottom: 8 }}>
          按行业聚合成分股评分，展示各行业20日/5日动量排名（柱状图+表格），以及每个行业的前3名股票。
        </Paragraph>

        <Text strong>我的组合</Text>
        <Paragraph style={{ marginBottom: 8 }}>
          管理自选持仓（增删改），数据自动保存在浏览器和服务器。点击"分析当前组合"对每只持仓进行盈亏计算、
          行业分布分析和买卖建议。仅支持沪深300成分股。
        </Paragraph>
      </Card>

      <Card size="small" style={{ marginBottom: 16 }}>
        <Title level={5}>评分体系</Title>
        <Paragraph>
          综合评分 = <Tag color="#f5222d">ML模型概率 × 50%</Tag> + <Tag color="#1677ff">技术面评分 × 30%</Tag> + <Tag color="#faad14">动量评分 × 20%</Tag>
        </Paragraph>
        <Text strong>ML模型</Text>
        <Paragraph style={{ marginBottom: 4 }}>
          基于 LightGBM + 随机森林 + GBDT 的集成模型，输入65+个技术因子，预测5日上涨概率。
        </Paragraph>
        <Text strong>技术面评分</Text>
        <Paragraph style={{ marginBottom: 4 }}>
          RSI（35%）+ MACD（35%）+ 均线位置（30%）。
        </Paragraph>
        <Text strong>动量评分</Text>
        <Paragraph style={{ marginBottom: 4 }}>
          5日动量（50%）+ 20日动量（50%）。
        </Paragraph>
      </Card>

      <Card size="small" style={{ marginBottom: 16 }}>
        <Title level={5}>市场状态说明</Title>
        <div><Tag color="#52c41a">牛市</Tag> 价格在MA20和MA60上方，均线向上</div>
        <div><Tag color="#1677ff">复苏</Tag> 价格回到MA20上方但MA60仍在下方</div>
        <div><Tag color="#faad14">回调</Tag> 价格跌破MA20但MA60仍在支撑</div>
        <div><Tag color="#d9d9d9" style={{ color: '#333' }}>震荡</Tag> 无明显趋势方向</div>
        <div style={{ marginBottom: 0 }}><Tag color="#ff4d4f">熊市</Tag> 价格在MA20和MA60下方，均线向下</div>
      </Card>

      <Card size="small">
        <Text type="secondary">
          免责声明：本工具仅供学习研究参考，不构成任何投资建议。
          所有数据来源于腾讯/新浪财经公开API，模型预测结果仅供参考。
          市场有风险，投资需谨慎。
        </Text>
      </Card>
    </div>
  )
}
