# QuantSelect 量化选股系统

两套独立选股引擎，覆盖**中线月度再平衡**和**短线 T+1~3 择时**两种策略风格。

## 项目结构

```
quant_select/
├── core/                       # 中线月度再平衡系统
│   ├── rebalance.py            #   月度再平衡编排（5阶段流程）
│   ├── macro_timing.py         #   宏观择时：21项指标打分 + 仓位推导 + 风格估值配置
│   ├── micro_factors.py        #   微观因子：12因子两轮计算
│   ├── selector.py             #   选股：行业Z-Score + 风格分层
│   ├── stock_pool.py           #   股票池：A股全市场 + 港股通Top200
│   ├── style_rotation.py       #   风格轮动：大小盘/价值成长分类
│   ├── industry_zscore.py      #   行业Z-Score标准化
│   ├── risk_control.py         #   前置+后置风控（6+5项规则）
│   ├── trader.py               #   持仓模拟回测
│   ├── data_cache.py           #   Pickle缓存层（MD5键控 + TTL）
│   └── factor_calc.py          #   旧版4因子（兼容保留）
│
├── short_term/                 # 短线T+1~3选股系统（新增）
│   ├── config.py               #   硬约束、因子权重、模型参数
│   ├── data_fetcher.py         #   日线OHLCV + 资金流向 + 指数数据
│   ├── alpha_factors.py        #   47个Alpha158风格因子
│   ├── custom_factors.py       #   量价结构 + 资金流向 + RPS因子
│   ├── factor_engine.py        #   预处理：填充→缩尾→标准化→IC筛选
│   ├── filters.py              #   四级硬约束过滤器
│   ├── model_trainer.py        #   XGBoost训练/预测/持久化
│   ├── daily_pipeline.py       #   每日选股管线编排
│   ├── backtest_engine.py      #   短线回测引擎
│   ├── main_short.py           #   CLI入口
│   ├── model_cache/            #   训练好的模型文件
│   └── output/                 #   选股结果JSON快照
│
├── config/                     # 共享配置
│   ├── settings.py             #   权重、风控参数、缓存TTL
│   ├── macro_config.py         #   21项宏观指标定义（新增3项热度+3项估值）
│   ├── manual_macro.py         #   手动宏观覆盖
│   └── ggt_codes.json          #   港股通成分股列表
│
├── utils/common.py             # 工具函数
├── risk/filter_rule.py         # 旧版风控（兼容）
├── output/                     # 中线系统月报输出
├── cache/                      # API调用缓存
├── log/                        # 运行日志
├── main.py                     # 中线系统CLI入口
└── requirements.txt            # Python依赖
```

## 中线月度再平衡系统

适用场景：T+30 月度调仓，稳健风格，宏观驱动仓位 + 微观驱动选股。

### 策略逻辑

1. **宏观择时**：21项宏观指标（PMI/CPI/M2/信用利差/北向资金/量比极值/涨跌比/ERP/PE分位/PE价差等）→ 加权打分 → 推导股票仓位（30%~100%）和风格配置（大小盘/价值成长）
2. **微观选股**：12因子两轮计算（批量财务 + 逐只量价）→ 行业Z-Score标准化 → 风格分层选股
3. **风控**：6项前置 + 5项后置校验

**宏观指标体系（8大维度 / 21项）：**

| 维度 | 指标 | 权重合计 |
|------|------|----------|
| 经济增长 | PMI、工业增加值 | 8% |
| 通胀物价 | CPI、PPI、大宗商品 | 4.5% |
| 流动性信用 | 10Y国债、M2、社融、信用利差 | 10% |
| 外部跨境 | 中美利差、汇率、北向资金 | 4.5% |
| 市场情绪 | 全市场成交额、两融余额 | 3% |
| 市场热度 | 量比极值、涨跌比、权益风险溢价(ERP) | 8% |
| 风格估值 | 上证PE分位、创业板PE分位、PE比值 | 5% |
| 政策信心 | 货币政策、财政政策、会议定调、消费者信心(代理) | 6% |

> **热度/冷度因子**（v2.0 新增）：量比极值用 center 型打分（0.7~1.3x 为正常，>2.5x 过热/ <0.3x 冰点），涨跌比监测 20 日单边极端程度，ERP = 盈利收益率 − 10Y 国债利率衡量股债性价比。
>
> **风格估值因子**（v2.0 新增）：上证 PE 历史分位指导大盘配置，创业板 PE 分位指导成长配置，PE 比值监测风格泡沫。风格配置 = PE 估值信号(50%) + 60 日动量(50%)。

### 使用

```bash
# 日常使用：编辑 config/manual_macro.py 持久化配置，运行即可
python main.py --month 2026-05

# 临时覆盖（不改配置文件，一次生效）
python main.py --month 2026-05 --manual-macro monetary=8,fiscal=7
# 优先级：CLI --manual-macro > config/manual_macro.py > API 自动获取

# 结果输出至 output/portfolio_2026-05.json 和 output/report_2026-05.md
```

---

## 短线 T+1~3 选股系统

适用场景：每日选股，T日收盘后出信号 → T+1开盘买入 → T+3收盘卖出，追求高胜率高频交易。

### 策略逻辑

**六步每日管线：**

```
Step 1 数据获取   → 全A股日线（tqdm进度条，腾讯数据源） + 资金流向 + 指数数据
Step 2 因子计算   → 47个Alpha因子 + 量价结构 + 资金流向 + RPS
Step 3 因子预处理 → 缺失值填充 → 极值缩尾 → Z-Score标准化
Step 4 硬约束过滤 → 剔除ST → 基础 → 趋势 → 量价 → 资金（四级递进）
Step 5 模型预测   → XGBoost输出3日上涨概率
Step 6 综合排序   → 概率≥0.45 + 因子得分排序 → Top 3
```

**四级硬约束：**

| 级别 | 规则 |
|------|------|
| 基础 | 剔除ST/退市；流通市值20-500亿；日均成交额≥3亿；换手率≥1%；上市≥30天 |
| 趋势 | MA5>MA10>MA20；股价站上MA60；RPS(20)≥70 |
| 量价 | 当日涨幅1%-9%；放量≥1.1倍5日均量；不放天量(<3倍) |
| 资金 | 主力净流入>0；大单净流入>0；3日累计主力净流入为正 |

### 使用

```bash
# 训练/更新模型（建议每月1次）
python -m short_term.main_short --mode train

# 每日选股（默认今天，缓存过期自动刷新）
python -m short_term.main_short --mode select

# 指定日期选股
python -m short_term.main_short --mode select --date 20260527

# 清理过期缓存后选股（确保用最新收盘价）
rm cache/*.pkl && python -m short_term.main_short --mode select

# 回测
python -m short_term.main_short --mode backtest \
    --start 2025-01-01 --end 2026-05-01

# 自定义初始资金
python -m short_term.main_short --mode backtest \
    --start 2025-01-01 --end 2026-05-01 --capital 500000
```

> **自动化**：已配置工作日 17:00 自动执行选股（`.claude/scheduled_tasks.json`），先清过期缓存再跑管线。

**选股输出示例：**

```
==============================================================
  选股结果 (Top 3)
==============================================================
  1. 600xxx 某某股份
     模型概率: 82.3%  因子得分: 0.852
     综合得分: 0.835  → 买入 20% 仓位
  2. 000xxx 某某科技
     模型概率: 76.1%  因子得分: 0.721
     综合得分: 0.745  → 买入 20% 仓位
  3. 300xxx 某某医疗
     模型概率: 72.5%  因子得分: 0.693
     综合得分: 0.712  → 买入 20% 仓位
==============================================================
```

结果同时保存至 `short_term/output/selection_YYYYMMDD.json`。

### 回测基准

| 指标 | 合格线 | 说明 |
|------|--------|------|
| 胜率 | ≥58% | 盈利交易占比 |
| 盈亏比 | ≥2.0 | 平均盈利/平均亏损 |
| 年化收益 | ≥30% | 年化复合收益率 |
| 最大回撤 | ≤18% | 净值最大回撤幅度 |
| 夏普比率 | ≥1.5 | 风险调整后收益 |

回测参数：手续费万2.5 + 滑点0.1%、单票仓位20%、最多持仓5票、T+3卖出。

---

## 环境要求

```bash
pip install -r requirements.txt
```

依赖：`akshare`（数据源）、`pandas`、`numpy`、`scipy`、`matplotlib`、`xgboost`、`scikit-learn`

数据源：通过 akshare 获取 A 股实时行情、财务数据、资金流向，所有 API 调用自动缓存至 `cache/` 目录。日线优先走东方财富，失败回退腾讯数据源；全量拉取使用 tqdm 进度条。

## 注意事项

- 两套系统**独立运行**，分别有各自的 CLI 入口和输出目录
- 短线系统的 XGBoost 模型建议**每月重新训练**以保持因子有效性
- 实盘操作前务必在回测模式下验证策略在当前市场环境的表现
- 短线资金流向数据依赖 akshare 的 `stock_individual_fund_flow` 接口，可能受数据源延迟影响
