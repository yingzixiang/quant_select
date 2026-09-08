# quant_select 工程架构文档

> 本文件是「量化工程智能体」上岗第一任务沉淀的共享知识基座，供后续所有量化工程任务复用。
> 生成时间：2026-09-08　｜　覆盖代码库：`/Users/zaleying/Project/quant_select`

## 一、工程定位

quant_select 是一个**三引擎并存的 A 股 / 港股量化选股与回测工程**，按持仓周期分为：

| 引擎 | 目录 | 策略周期 | 决策核心 | 入口 |
|------|------|----------|----------|------|
| 中线月度再平衡 | `core/` + `main.py` | T+30 月度调仓 | 宏观择时 + 微观 12 因子 | `python main.py` |
| 短线选股 | `short_term/` | T+1 买入 → T+3 卖出 | 47 因子 + XGBoost + 五级硬约束 | `python -m short_term.main_short` |
| ~~日内 T+0~~ | ~~`intraday/`~~ → `archive/` | 分钟级底仓回转 | （已归档，见 `CONVERGENCE_DESIGN.md`） | — |

> **收敛状态**：三引擎收敛方案 B 已采纳（2026-09-08）。日内引擎已归档到 `archive/intraday/`，当前活跃引擎为**中线月度 + 短线**两套，抽共享底座的重构（P1~P4）按 `CONVERGENCE_DESIGN.md` 分阶段推进。

两套系统**相互独立运行**，各有 CLI 入口、配置与输出目录，仅共享 `core/data_cache.py`（缓存层）与 `utils/common.py`（工具函数）等少量基座。

## 二、技术栈与运行环境

- **语言**：Python 3.13（`.venv` 已就绪，`requirements.txt`：akshare/pandas/numpy/scipy/matplotlib/xgboost/scikit-learn）
- **数据源**：AKShare（免费，无需账号）。日线优先东方财富（`stock_zh_a_hist`），失败回退腾讯（`stock_zh_a_hist_tx`），再回退新浪（`stock_zh_a_daily`）
- **缓存**：`core/data_cache.py` 提供 `cached_api_call(func, *args, max_age_seconds=…)`，MD5 键控 + pickle 落盘 + TTL 过期，统一写到 `cache/*.pkl`
- **模型**：XGBoost 二分类（预测「未来 3 日上涨概率」），模型 pickle 持久化于 `short_term/model_cache/`

## 三、目录结构

```
quant_select/
├── main.py                  # 中线月度再平衡 CLI 入口
├── cli.py                   # 统一策略 CLI 入口（收敛重构 P3 新增）
├── base/                    # 共享底座（收敛重构新增）
│   ├── strategy.py          #   Strategy 接口 + Signal/Portfolio 数据模型
│   ├── adapters.py          #   两套引擎 → Strategy 的薄适配器
│   ├── data_fetch.py        #   统一数据获取（日线多源回退/资金流/指数）
│   └── backtest.py          #   统一回测绩效指标层
├── core/                    # 中线月度再平衡系统
│   ├── rebalance.py         #   月度再平衡编排（5 阶段流程）
│   ├── macro_timing.py      #   21 项宏观指标打分 → 股债配比 + 风格判定
│   ├── micro_factors.py     #   12 因子两轮计算（批量财务 → Top200 逐只量价）
│   ├── selector.py          #   行业 Z-Score 打分 + 风格分层选股
│   ├── stock_pool.py        #   A股全市场 + 港股通 Top200
│   ├── style_rotation.py    #   大小盘/价值成长分池 + 席位分配
│   ├── industry_zscore.py   #   行业分组 Z-Score（小行业回退板块）
│   ├── risk_control.py      #   6 项前置 + 5 项后置风控
│   ├── contrarian_state.py  #   逆向仓位修正状态机（持久化到 output/）
│   ├── trader.py            #   A股持仓模拟回测 + 债券收益
│   ├── data_cache.py        #   Pickle 缓存层（MD5 键控 + TTL）
│   └── factor_calc.py       #   旧版 4 因子（兼容保留）
├── short_term/              # 短线 T+1~3 选股系统
│   ├── main_short.py        #   CLI 入口（train/select/backtest）
│   ├── daily_pipeline.py    #   每日选股六步管线编排
│   ├── data_fetcher.py      #   日线/资金流/指数（东财→腾讯→新浪回退）
│   ├── alpha_factors.py     #   47 个 Alpha158/191 风格因子
│   ├── custom_factors.py    #   量价结构 + 资金流向 + RPS + 趋势强度
│   ├── factor_engine.py     #   预处理：填充→缩尾→标准化→IC 筛选
│   ├── filters.py           #   五级硬约束过滤器
│   ├── model_trainer.py     #   XGBoost 训练/预测/持久化
│   ├── backtest_engine.py   #   回测引擎（Position/Trade/指标）
│   ├── backtest_runner.py   #   完整管线回测运行器
│   └── model_cache/ output/ #   模型文件 / 选股结果 JSON
├── archive/                 # 已归档代码（不再维护，git 保留历史）
│   ├── intraday/            #   日内 T+0 底仓回转系统（已冻结）
│   └── parameter_search.py  #   日内参数搜索（依赖 intraday，一并归档）
├── broker/                  # 下单对接层（信号 → 委托最后一公里）
│   ├── base.py              #   Broker 抽象契约 + Order/Position 数据模型
│   ├── paper.py             #   PaperBroker 模拟盘（本地撮合 + JSON 落盘）
│   ├── bridge.py            #   信号桥：选股结果 → 委托 + 出场规则
│   ├── live.py              #   真实券商适配器注册点（渠道待选型）
│   ├── config.py            #   交易成本/整手/仓位约束
│   └── __main__.py          #   CLI（--query / --file / --dry-run）
├── config/                  # 共享配置
│   ├── settings.py          #   中线：日期/权重/风控参数/缓存 TTL
│   ├── macro_config.py      #   21 项宏观指标定义（API + 打分阈值表）
│   ├── manual_macro.py      #   手动宏观覆盖（持久化配置）
│   └── ggt_codes.json       #   港股通成分股列表
├── utils/common.py          # 工具函数（安全转换/分位/年化波动）
├── risk/filter_rule.py      # 旧版风控（兼容）
├── output/                  # 中线月报 JSON + 报告 MD
├── cache/                   # API 调用 pickle 缓存
└── *.md（README / PROJECT_OVERVIEW / query / 短线量化选股 / 日内T交易）
```

## 四、中线月度再平衡核心链路（`core/`）

`main.py` → `core/rebalance.py::run_monthly_rebalance()`，五阶段：

```
[Phase 1] 宏观择时   fetch_all_macro_indicators → compute_macro_score
                     → derive_allocation（股债配比，含逆向修正）
                     → determine_style_allocation（大小盘/价值成长）
[Phase 2] 股票池     get_total_stocks（A股全市场）→ filter_risk_stocks（剔ST）
                     get_hk_stocks（港股通 Top200 按成交额）
[Phase 3] A股因子    compute_all_micro_factors（12 因子两轮）→ compute_micro_scores
[Phase 4] 港股因子   get_hk_stock_factors（4 因子）→ _score_hk_stocks
[Phase 5] 选股+风控  select_target_stocks_v2（风格分层 + 行业上限）
                     → post_check_portfolio（后置校验）
```

关键机制：

- **宏观择时**：21 项指标（8 维度）加权打分 → `股票仓位 = FLOOR(15%) + 85% × 宏观总分`，剩余为债券。打分类型有 `threshold`（分段）、`center`（区间最优）、`percentile`（历史分位）、`manual`（手动）、`proxy`（代理）。
- **逆向修正**：`contrarian_state.py` 状态机持久化 `output/contrarian_state.json`，连续 ≥2 个月悲观/过热才分阶段调仓（每月 ±2%），单月极端只预警。
- **微观 12 因子**：估值（PB/PE/股息）+ 质量（ROE/负债率/毛利率）+ 成长（利润/营收/研发增速）+ 量价（60 日动量/30 日波动）+ 流动性（20 日均成交额）。第一轮 `stock_yjbb_em` 批量财务筛 Top200，第二轮逐只补 PB/量价/股息。
- **打分**：`industry_zscore.py` 按申万一级行业分组 Z-Score（小行业回退七大板块），统一方向后加权求和；港股 4 因子池内等权。
- **风控**：6 项前置（部分因无批量 API 而跳过/委托）+ 5 项后置（单票 ≤5%、行业 ≤20%、组合 PE/PB vs 沪深 300 等）。

## 五、短线选股核心链路（`short_term/`）

`short_term/daily_pipeline.py::run_daily_selection()`，六步：

```
Step 1 数据获取   全A股列表（东财 spot，含市值/成交额）→ 活跃度 Top1000 预筛
                  → 批量日线（并发）→ 资金流向 → 沪深300 指数
Step 2 因子计算   47 Alpha 因子 + 量价结构/资金流向/RPS/趋势强度自定义因子
Step 3 因子预处理 缺失填充 → 极值缩尾(1%~99%) → Z-Score（可选 IC 筛选）
Step 4 硬约束过滤 五级递进：基础→趋势→量价→资金→强势确认
Step 5 模型预测   XGBoost 输出 3 日上涨概率
Step 6 综合排序   final_score = 概率×0.55 + 因子×0.30 + 趋势强度×0.15 → Top3
```

五级硬约束（`short_term/filters.py`）：
1. **基础**：剔 ST/退市；流通市值 20–500 亿；日均成交额 ≥3 亿；换手率 ≥1%；上市 ≥30 天
2. **趋势**：MA5>MA10>MA20；股价站上 MA20/MA60；RPS(20)≥70
3. **量价**：当日涨幅 1%–9%；量比 1.1–3 倍 5 日均量
4. **资金**：主力净流入>0；大单净流入>0；3 日资金趋势为正（无数据则软放行）
5. **强势确认**：收阳 + 近 3 日 ≥2 日收阳 + 量价共振 + 超额收益（满足 3/4）

训练：`--mode train` 多日期采样（2 年窗口、每 3 天采样），标签为「3 日收益 >5% 为正 / <0 为负」。建议每月重训。

## 六、日内 T+0 核心链路（`archive/intraday/`，已归档）

> ⚠️ 本节仅作历史参考。日内引擎已于 2026-09-08 归档到 `archive/intraday/`，不再维护（详见 `CONVERGENCE_DESIGN.md` 方案 B）。

依据 `日内T交易.md` 的十一层架构实现（纯方案落代码），基于底仓日内回转（T+1 规则约束）：

```
[Layer 0] 标的池筛选（流动性 ≥6 亿、振幅 ≥3.2%、风险排除，4–6 只）
[Layer 1] 全域环境过滤（指数连跌/涨跌比/振幅异常 → 空仓）
[Layer 2] 行情分型（震荡/多头趋势/空头趋势/反转/跳空，含滞回确认）
[Layer 3] 自适应指标（ATR/MA5·10·20/量能/K 线形态）
[Layer 4] 分场景交易规则（range/bull/bear/reversal/gap）
[Layer 5] 动态仓位管理（底仓 15%–45%，单票/全账户上限）
[Layer 6] 全域硬风控（单笔 -0.7%、单票日 -1.4%、全账户分级止损）
[Layer 7] 分行情参数矩阵（narrow_range/normal/high_active 三套）
[Layer 8] 信号校验防骗线 + [Layer 9] 回测验收 + [Layer 10] 参数敏感性搜索
```

交易窗口 09:35–14:40，1min/5min K 线；回测验收标准（胜率 ≥57%、盈亏比 ≥1.3、单日回撤 <7%、假信号率 <18%）。

## 六.b、下单对接（`broker/`，v1 模拟盘）

新增下单层，补齐「信号 → 委托」最后一公里。当前只实现了**券商无关抽象 + 本地模拟盘**，未绑定任何真实券商 SDK。

- **`Broker` 契约**（`base.py`）：`connect` / `get_cash` / `get_positions` / `place_order` / `cancel_order` / `get_orders`，含 `Order`/`Position` 数据模型。
- **`PaperBroker`**（`paper.py`）：本地即时撮合，手续费万2.5 + 印花税0.05%(卖)、整手约束、资金/持仓校验、JSON 落盘 `output/paper_account.json`（跨运行续接）。
- **信号桥**（`bridge.py`）：读 `short_term/output/selection_*.json` → 按 position 百分比 + 最新价算整手 → 生成买单 + 止损/止盈/卖出日规则。
- **适配器注册点**（`live.py`）：`BrokerFactory.create("paper", ...)`，真实渠道在选定后继承 `Broker` 注册进 `BROKER_BACKENDS`。

```bash
python -m broker --query                                        # 查模拟盘账户
python -m broker --file short_term/output/selection_YYYYMMDD.json --capital 1000000 --dry-run  # 干跑
python -m broker --file short_term/output/selection_YYYYMMDD.json --capital 1000000            # 模拟盘下单
```

## 六.c、共享底座与统一策略接口（`base/`，收敛重构 P1~P3）

三引擎收敛重构按 `CONVERGENCE_DESIGN.md` 推进，已落地：

- **`base/data_fetch.py`**（P1）：统一数据获取——日线多源回退（东财→腾讯→新浪）、资金流向、指数、股票列表、流通市值。`short_term/data_fetcher.py` 已退化为兼容 shim。
- **`base/backtest.py`**（P2）：统一回测绩效指标层——胜率/盈亏比/年化/最大回撤/夏普，消除两套回测的年化口径漂移。
- **`base/strategy.py`**（P3）：`Strategy` 抽象接口 + `Signal`/`Portfolio` 数据模型，回测/下单/盯盘未来只依赖该接口。
- **`base/adapters.py`**（P3）：`MonthlyStrategy`（包装 `core.rebalance`）与 `SwingStrategy`（包装 `short_term.daily_pipeline`）薄适配器，把两套输出归一化到 `Portfolio`。
- **`cli.py`**（P3）：统一策略 CLI。老入口 `main.py` / `short_term.main_short` 保留。

```bash
python cli.py --strategy monthly --month 2026-05
python cli.py --strategy monthly --backtest 2025-06 2026-05
python cli.py --strategy swing --date 20260903
```

## 七、配置与运行方式

### 中线月度
```bash
python main.py --month 2026-05                                  # 指定月再平衡
python main.py --month 2026-05 --manual-macro monetary=8,fiscal=7  # 临时覆盖
python main.py --backtest 2025-06 2026-05                       # 多期月度回测（逐月选股+持仓收益汇总）
# 持久化覆盖：编辑 config/manual_macro.py（优先级：CLI > manual_macro.py > API）
# 输出：output/portfolio_2026-05.json + output/report_2026-05.md
```

### 短线选股
```bash
python -m short_term.main_short --mode train                 # 训练/更新模型
python -m short_term.main_short --mode select                # 今日选股
python -m short_term.main_short --mode select --date 20260527
rm cache/*.pkl && python -m short_term.main_short --mode select  # 清缓存选股
python -m short_term.main_short --mode backtest --start 2025-01-01 --end 2026-05-01
python -m short_term.backtest_runner --start 2026-01-01 --end 2026-05-01 --prob 0.5
```

### 日内 T+0（已归档，不再使用）
```bash
# 已归档到 archive/intraday/，如需复活：git mv archive/intraday intraday 后修复依赖
python -m archive.intraday.main_intraday --mode backtest --start 2026-01-01 --end 2026-06-01
```

### 自动化运行（multica autopilot）

平台定时自动化，跑完把报告以 issue 评论 @zale 送达（`create_issue` 模式，绑定「量化工程智能体」agent）：

| 任务 | autopilot id | cron（Asia/Shanghai） | 内容 |
|------|--------------|----------------------|------|
| 每日短线选股 | `4e81f939-705d-460f-a809-c75d05603f5e` | `10 18 * * 1-5`（工作日 18:10） | 清缓存 → `main_short --mode select` → Top3 日报 |
| 每月中线再平衡 | `cde123ae-3736-40b0-a601-a79843578013` | `0 8 15 * *`（每月 15 号 08:00） | `main.py --month YYYY-MM` → 宏观+组合月报 |

管理命令：`multica autopilot list` / `trigger-update` / `delete` / `trigger`（手动跑一次）。

## 八、关键注意事项（上手必读）

1. **手动宏观指标必须配置**：`config/manual_macro.py` 中 `social_financing`（API 滞后约 6 个月）、`credit_spread`（无 API）、`monetary_policy`/`fiscal_policy`/`meeting_tone`（定性）必须手动给分，否则走中性分或代理。
2. **数据源回退链**：短线日线 东财→腾讯→新浪；`stock_zh_a_hist_tx` 的 `amount` 单位是万元（需 ×10000），`stock_zh_a_hist` 的 `amount` 已是成交额元。
3. **模型需定期重训**：XGBoost 每月一次，否则因子漂移导致概率失真（`daily_pipeline` 有概率全低于阈值时的因子回退逻辑）。
4. **缓存清理**：`cache/*.pkl` TTL 过期自动失效；实盘前 `rm cache/*.pkl` 强制刷新收盘数据。
5. **已知未完成**：部分前置风控（上市天数/审计意见）因无批量 API 跳过；多期回测与港股回测为 v1 近似实现（口径见下）。
6. **状态文件**：`output/contrarian_state.json` 是逆向修正的跨月状态，删除会重置连续月计数。
7. **两套系统独立**：中线/短线各自输出目录互不干扰；`cache/` 共享但键由函数名+参数 MD5 区分。日内引擎已归档（`archive/`），不参与维护。

## 九、后续可推进方向与疑问

- **多期回测**（已实现 v1）：`main.py --backtest START END` 逐月选股 + 持仓收益汇总。口径：月首决策、量价窗口前推 3 个月、财报取最近季末、持仓区间 = 月首 → 下月首；合并净值 = A 股（含债）×A股数 + 港股（纯股）×港股数 按票数加权。属近似实现，未做停牌/涨跌停/交易成本处理。
- **港股回测**（已实现 v1）：`core/trader.py::simulate_hk_holding_return`，`stock_hk_daily(前复权)` 等权，单月与多期均已接入。
- **实盘/下单对接**（已实现 v1 模拟盘）：`broker/` 抽象层 + PaperBroker + 信号桥已跑通；真实券商 SDK 待 zale 确认渠道后接入（候选：QMT/miniQMT、PTrade、easytrader 等）。
- **数据质量治理**：社融等宏观指标 API 滞后、资金流向覆盖率低，需明确数据可靠性分级。
- **待 zale 决策**：① ~~三引擎收敛~~ → **已采纳方案 B**（归档日内、保留中线+短线，重构方案见 `CONVERGENCE_DESIGN.md`，按 P1~P4 推进）；② 实盘接入的券商/渠道（含是否需要自动盯盘 + 止损止盈触发）；③ 短线模型重训的自动化节奏；④ 多期回测是否需要更严格口径（复现单月流程 + 更真实的成本模型）。
