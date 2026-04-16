# Topaz-V3 + Qlib 量化模型整合

## 修改内容

### 1. 新增 `qlib_model.py`
Qlib 风格的量化预测模型：
- `QlibStylePredictor`: 多因子评分预测器
- `LightGBMPredictor`: LightGBM 机器学习预测器（可选）

### 2. 修改 `topaz_analysis_v3_full.py`
- 导入 Qlib 模型
- 使用 Qlib 模型进行收益预测和风险评估
- 保持原有内置模型作为备选

## 运行方式

```bash
cd $(dirname "$0")  # 使用相对路径
python3 topaz_analysis_v3_full.py
```

## 依赖安装

```bash
pip install -r requirements.txt
```

## 输出示例

```
✅ Qlib 量化模型已加载
【600519.SH】贵州茅台 (A 股)
  当前价格：¥1455.02
  市盈率：14.50
  价值因子：45.5/100
  质量因子：100.0/100
  未来 3 个月收益预测：13.9%
  风险等级：中风险
  投资建议：建议持有
```

## 模型说明

### Qlib 多因子模型
| 因子 | 权重 | 说明 |
|------|------|------|
| 价值因子 | 25% | 低 PE/PB 优先 |
| 质量因子 | 30% | 高 ROE 优先 |
| 动量因子 | 20% | 近期涨幅 |
| 波动因子 | 15% | 低波动优先 |
| 红利因子 | 10% | 高股息优先 |

### 风险评估
- 低风险：PE<20, PB<3, ROE>10%
- 中风险：PE<30, PB<5, ROE>5%
- 高风险：PE>30 或 PB>5 或 ROE<5%

## 文件结构

```
topaz-v3/
├── qlib_model.py              # Qlib 量化模型（新增）
├── topaz_analysis_v3_full.py  # 主程序（已修改）
├── topaz_analysis_v3_full.py.bak  # 原备份
├── requirements.txt           # 依赖（已更新）
├── README_QLIB.md            # 本文档
└── ...
```
