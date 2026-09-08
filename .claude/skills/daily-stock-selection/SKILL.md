---
name: daily-stock-selection
description: 每日收盘后自动执行短线选股管线（17:00 触发）
---

# 每日短线选股

## 触发条件
每个交易日 17:00 自动执行（A股收盘后约2小时，确保数据已更新）

## 执行流程

1. **清理过期缓存** — 删除 `cache/*.pkl` 中超过6小时的缓存，确保拉取最新收盘数据
2. **运行选股管线** — 执行 `python -m short_term.main_short --mode select`
3. **检查结果** — 如果选股成功，输出结果摘要；如果失败（无股票通过过滤），记录原因
4. **保存日志** — 结果保存在 `short_term/output/selection_YYYYMMDD.json`

## 注意事项
- 如果今天不是交易日（周末/节假日），日线数据不会更新，跳过执行
- 如果 OOM（exit code 137），降低并发重试：`python -m short_term.main_short --mode select` 前设置并发参数
- 选股成功后，告知用户 Top 3 推荐标的
