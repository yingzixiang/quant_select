"""
intraday/backtest_engine.py
日内 T+0 回测引擎 —— 核心模拟循环

逐日、逐 K 线模拟完整交易流程：
数据预加载 → 交易日分组 → 逐日循环 → 逐 bar 循环 → 指标计算 → 分型 →
环境过滤 → 信号生成 → 信号校验 → 执行开仓 → 检查持仓 → 收盘清仓 → 指标计算

无未来函数：bar[t] 的信号在 bar[t+1] 入场
"""
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from intraday.config import (
    BACKTEST_ACCEPTANCE, TRANSACTION_COST, BACKTEST_DEFAULTS,
    PARAMETER_MATRICES, POSITION, SCENARIO_RULES,
)
from intraday.market_classifier import (
    MarketType, classify_market_with_hysteresis, initial_classify,
    get_allowed_directions,
)
from intraday.indicators import (
    IndicatorState, compute_indicator_state, get_center_window_end_index,
    get_time_phase,
)
from intraday.env_filter import (
    check_global_environment, batch_check_sectors, EnvStatus,
)
from intraday.signal_engine import (
    Signal, SignalDirection, generate_signals,
)
from intraday.signal_validator import validate_signals_batch
from intraday.position_manager import (
    PositionManager, AccountState, get_position_ratio,
    check_account_stop, check_account_position_limits,
)
from intraday.risk_manager import (
    RiskManager, TradeRecord, RiskAction, is_close_only_time,
    should_cancel_all,
)
from intraday.stock_pool import (
    StockPoolItem, detect_opening_gap, get_active_codes, get_pool_item,
)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class CompletedTrade:
    """已完成交易记录"""
    stock_code: str
    direction: str
    scenario: str
    open_time: str
    close_time: str
    open_price: float
    close_price: float
    shares: int
    profit_pct: float
    profit_amount: float
    is_win: bool
    stop_reason: str
    max_adverse_pct: float = 0.0
    max_favorable_pct: float = 0.0


@dataclass
class BacktestResult:
    """回测结果"""
    metrics: dict = field(default_factory=dict)
    trades: List[CompletedTrade] = field(default_factory=list)
    daily_values: List[Tuple[str, float]] = field(default_factory=list)
    daily_stats: List[dict] = field(default_factory=list)
    acceptance: dict = field(default_factory=dict)
    signal_log: List[dict] = field(default_factory=list)


# ============================================================
# 回测引擎
# ============================================================

class IntradayBacktestEngine:
    """日内 T+0 回测引擎"""

    def __init__(self,
                 initial_capital: float = 1_000_000,
                 param_set_name: str = "normal",
                 period: str = "5"):
        """
        Args:
            initial_capital: 初始资金
            param_set_name: 参数矩阵名称（"narrow_range"/"normal"/"high_active"）
            period: K 线周期（"5" 或 "1"）
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.param_set = PARAMETER_MATRICES[param_set_name]
        self.period = period
        self.period_minutes = int(period)

        # 交易成本
        self.commission = TRANSACTION_COST["commission_rate"]
        self.slippage = TRANSACTION_COST["slippage_rate"]

        # 结果收集
        self.all_trades: List[CompletedTrade] = []
        self.daily_values: List[Tuple[str, float]] = []
        self.daily_stats: List[dict] = []
        self.signal_log: List[dict] = []
        self.false_signals: int = 0
        self.total_signals: int = 0

    # ================================================================
    # 主回测循环
    # ================================================================

    def run(self,
            stock_minute_dict: Dict[str, pd.DataFrame],
            index_minute_df: pd.DataFrame,
            sector_minute_dict: Dict[str, pd.DataFrame],
            sector_map: Dict[str, str],
            volume_profile: Dict[Tuple[str, str], float],
            pool_items: List[StockPoolItem],
            base_positions: Dict[str, int]) -> BacktestResult:
        """
        执行完整回测

        Args:
            stock_minute_dict: {code: 分钟 K 线 DataFrame}
            index_minute_df: 指数分钟数据
            sector_minute_dict: {sector_name: 分钟数据}
            sector_map: {code: sector_name}
            volume_profile: {(code, time_str): avg_volume}
            pool_items: 标的池
            base_positions: {code: 底仓股数}

        Returns:
            BacktestResult
        """
        print(f"\n{'='*60}")
        print(f"  日内 T+0 回测引擎")
        print(f"  参数矩阵: {self.param_set['name']}")
        print(f"  初始资金: {self.initial_capital:,.0f}  |  K线周期: {self.period}min")
        print(f"  手续费: {self.commission:.4%}  |  滑点: {self.slippage:.2%}")
        print(f"{'='*60}\n")
        sys.stdout.flush()

        # ---- Step 0：分组交易日 ----
        trading_days = self._group_trading_days(index_minute_df)
        if not trading_days:
            print("[回测] 无有效交易日！")
            return BacktestResult()

        # 过滤：只保留标的池中股票也有数据的交易日
        valid_codes = [item.code for item in pool_items if item.code in stock_minute_dict]
        print(f"[回测] 标的池: {len(valid_codes)} 只 | 交易日: {len(trading_days)} 天")
        print(f"[回测] 日期范围: {list(trading_days.keys())[0]} ~ {list(trading_days.keys())[-1]}")
        sys.stdout.flush()

        # ---- Step 1：逐日循环 ----
        account = AccountState(initial_capital=self.initial_capital)
        day_count = 0

        for day_date, day_indices in trading_days.items():
            day_count += 1
            daily_result = self._simulate_day(
                day_date, day_indices,
                stock_minute_dict, index_minute_df,
                sector_minute_dict, sector_map,
                volume_profile, pool_items,
                base_positions, valid_codes,
                account,
            )

            if daily_result:
                self.daily_values.append((day_date, daily_result["end_value"]))
                self.daily_stats.append(daily_result)

            if day_count % 10 == 0:
                net_value = self.daily_values[-1][1] if self.daily_values else self.initial_capital
                print(f"[回测] 进度: {day_count}/{len(trading_days)} 天, "
                      f"净值: {net_value:,.0f}, "
                      f"交易: {len(self.all_trades)} 笔")
                sys.stdout.flush()

        # ---- Step 2：计算指标 ----
        result = BacktestResult()
        result.trades = self.all_trades
        result.daily_values = self.daily_values
        result.daily_stats = self.daily_stats
        result.signal_log = self.signal_log

        metrics = self._compute_metrics()
        result.metrics = metrics

        # 验收检查
        result.acceptance = self._check_acceptance(metrics)

        self._print_report(metrics, result.acceptance)

        return result

    # ================================================================
    # 单日模拟
    # ================================================================

    def _simulate_day(self,
                      day_date: str,
                      day_indices: Tuple[int, int],
                      stock_minute_dict: dict,
                      index_minute_df: pd.DataFrame,
                      sector_minute_dict: dict,
                      sector_map: dict,
                      volume_profile: dict,
                      pool_items: List[StockPoolItem],
                      base_positions: dict,
                      valid_codes: List[str],
                      account: AccountState) -> Optional[dict]:
        """
        模拟单个交易日

        Args:
            day_date: 日期字符串 "YYYY-MM-DD"
            day_indices: (start_idx, end_idx) 在 index_minute_df 中的索引范围
        """
        idx_start, idx_end = day_indices
        n_bars = idx_end - idx_start
        if n_bars < 10:
            return None

        # ---- 初始化当日状态 ----
        # 重置所有 pool items
        for item in pool_items:
            item.blocked = False
            item.block_reason = ""
            item.gap_level = 0.0
            item.gap_scenario = 0

        # 初始化每个标的的管理器
        pos_managers: Dict[str, PositionManager] = {}
        risk_managers: Dict[str, RiskManager] = {}
        indicator_states: Dict[str, IndicatorState] = {}
        market_types: Dict[str, MarketType] = {}
        confirm_counts: Dict[str, int] = {}
        open_trades: Dict[str, List[TradeRecord]] = defaultdict(list)

        for code in valid_codes:
            base_shares = base_positions.get(code, BACKTEST_DEFAULTS["base_position_shares"])
            pos_managers[code] = PositionManager(code, base_shares)
            risk_managers[code] = RiskManager(code)
            market_types[code] = MarketType.RANGE
            confirm_counts[code] = 0

        # ---- 中枢窗口索引 ----
        code0 = valid_codes[0]
        center_end_idx = get_center_window_end_index(
            stock_minute_dict[code0].iloc[idx_start:idx_end],
            self.period_minutes
        )

        # ---- 检测开盘跳空 ----
        for code in valid_codes:
            stock_df = stock_minute_dict[code]
            day_bars = stock_df.iloc[idx_start:idx_end]
            if len(day_bars) < 2:
                continue

            # 前日收盘价（取上一交易日最后一根 bar 的 close）
            prev_close = float(day_bars.iloc[0]["close"])
            for item in pool_items:
                if item.code == code:
                    gap_pct, gap_scenario = detect_opening_gap(day_bars, prev_close)
                    item.gap_level = gap_pct
                    item.gap_scenario = gap_scenario
                    break

        # 当日开盘价（用于缺口场景）
        day_opens = {}
        for code in valid_codes:
            bar0 = stock_minute_dict[code].iloc[idx_start]
            day_opens[code] = float(bar0["open"])

        # ---- 逐 bar 循环 ----
        day_pnl = 0.0
        day_signals_generated = 0
        day_false_signals = 0
        initial_capital = self.initial_capital + account.daily_pnl_amount

        for bar_offset in range(n_bars):
            bar_idx = idx_start + bar_offset

            # 当前 bar 时间
            idx_bar = index_minute_df.iloc[bar_idx]
            bar_time = str(idx_bar.get("timestamp", ""))
            time_str = bar_time[-8:-3] if len(bar_time) >= 8 else bar_time[:5]  # HH:MM

            # ---- 时段过滤 ----
            phase = get_time_phase(bar_time)
            if phase in ("pre_trading", "closing_auction"):
                continue  # 禁止交易时段

            # ---- 14:40 强制平仓 ----
            if phase == "close_only" or is_close_only_time(bar_time):
                for code in valid_codes:
                    self._close_all_open_trades(
                        code, open_trades, stock_minute_dict,
                        bar_idx, bar_time, "TIME_CLOSE",
                        pos_managers, account, idx_start,
                    )
                break  # 日循环结束

            # ---- A. 更新指标 ----
            for code in valid_codes:
                stock_df = stock_minute_dict[code]
                day_bars = stock_df.iloc[idx_start:bar_idx + 1]
                if len(day_bars) < 3:
                    continue

                bk = bar_time[:5]
                vol_ref = volume_profile.get((code, bk))
                indicator_states[code] = compute_indicator_state(
                    day_bars, len(day_bars) - 1,
                    vol_ref=vol_ref,
                    atr_multiplier=self.param_set["atr_multiplier"],
                    atr_window=20,
                )

            # ---- B. 更新行情分型（每 10 分钟） ----
            minute_of_day = self._time_to_minutes(bar_time)
            if minute_of_day >= 605:  # 10:05 后
                bars_after_center = (minute_of_day - 605) // self.period_minutes
                if bars_after_center >= 0 and bars_after_center % max(1, 10 // self.period_minutes) == 0:
                    for code in valid_codes:
                        stock_df = stock_minute_dict[code]
                        day_bars = stock_df.iloc[idx_start:bar_idx + 1]
                        if len(day_bars) < 10:
                            continue
                        new_type, new_confirm, _ = classify_market_with_hysteresis(
                            day_bars, len(day_bars) - 1,
                            market_types[code], confirm_counts[code],
                        )
                        if new_type != market_types[code]:
                            market_types[code] = new_type
                            confirm_counts[code] = new_confirm

            # ---- C. 全局环境检查 ----
            env_result = check_global_environment(index_minute_df, bar_idx)
            env_blocked = env_result["block_open"]

            # ---- D. 板块过滤 ----
            sector_blocks = batch_check_sectors(
                valid_codes, sector_map, sector_minute_dict,
                bar_idx - idx_start,  # 板块数据的相对索引
            )

            # ---- E. 更新动态淘汰 ----
            for code in valid_codes:
                stock_df = stock_minute_dict[code]
                if len(stock_df) > bar_idx:
                    cur_bar = stock_df.iloc[bar_idx]
                    cur_price = float(cur_bar["close"])
                    prev_close = float(stock_df.iloc[idx_start - 1]["close"]) if idx_start > 0 else cur_price

                    from intraday.stock_pool import check_dynamic_elimination
                    check_dynamic_elimination(cur_price, prev_close, None, pool_items)

            # ---- F. 检查已有持仓（止损/止盈） ----
            for code in valid_codes:
                stock_df = stock_minute_dict[code]
                if bar_idx >= len(stock_df):
                    continue

                cur_bar = stock_df.iloc[bar_idx]
                ind = indicator_states.get(code)

                for trade in open_trades[code][:]:
                    if ind is None:
                        continue

                    rm = risk_managers[code]
                    action, reason = rm.check_stop_loss(trade, cur_bar.to_dict(), ind)

                    if action != RiskAction.HOLD:
                        close_price = self._get_execution_price(
                            float(cur_bar["close"]), trade.direction == "short"
                        )
                        self._close_trade(
                            code, trade, close_price, bar_time, action.name,
                            pos_managers, account, open_trades,
                        )

            # ---- G. 生成信号 ----
            if not env_blocked:
                for code in valid_codes:
                    item = get_pool_item(pool_items, code)
                    if item is None or item.blocked:
                        continue
                    if code in sector_blocks:
                        continue

                    stock_df = stock_minute_dict[code]
                    if bar_idx >= len(stock_df):
                        continue

                    cur_bar = stock_df.iloc[bar_idx]
                    ind = indicator_states.get(code)
                    if ind is None:
                        continue

                    day_bars = stock_df.iloc[idx_start:bar_idx + 1]

                    signals = generate_signals(
                        stock_code=code,
                        timestamp=bar_time,
                        ind=ind,
                        cur_bar=cur_bar.to_dict(),
                        market_type=market_types[code],
                        gap_scenario=item.gap_scenario,
                        gap_pct=item.gap_level,
                        day_open=day_opens.get(code, float(cur_bar["close"])),
                        bar_history=day_bars,
                        env_blocked=env_blocked,
                        stock_blocked=code in sector_blocks,
                    )

                    # ---- H. 信号校验 ----
                    valid_signals = validate_signals_batch(
                        signals, stock_minute_dict, bar_idx,
                        indicator_states, index_minute_df, bar_idx,
                    )

                    day_signals_generated += len(signals)
                    day_false_signals += len(signals) - len(valid_signals)

                    # ---- I. 执行信号 ----
                    for sig in valid_signals:
                        pm = pos_managers[code]
                        rm = risk_managers[code]

                        # 仓位限制
                        direction_str = sig.direction.value
                        can_open, reason = pm.check_position_limits(direction_str)
                        if not can_open:
                            rm.record_signal_failure(direction_str, bar_time[:5])
                            continue

                        # 频次限制
                        blocked, freq_reason = rm.check_frequency_limit(direction_str, bar_time[:5])
                        if blocked:
                            continue

                        # 计算仓位
                        is_weak = (market_types[code] == MarketType.REVERSAL)
                        sig.size_shares = pm.allocate_trade_size(
                            market_types[code], item.gap_scenario, is_weak, account
                        )

                        if sig.size_shares <= 0:
                            continue

                        # 全账户红线
                        trade_value = sig.size_shares * sig.entry_price
                        within, limit_reason = check_account_position_limits(
                            account, self._current_capital(account), trade_value
                        )
                        if not within:
                            continue

                        # 执行入场（下一 bar 价格，模拟延迟）
                        is_buy = (sig.direction == SignalDirection.LONG)
                        # 用下一根 bar 的开盘价（无未来函数）
                        if bar_offset + 1 < n_bars:
                            next_bar = stock_minute_dict[code].iloc[idx_start + bar_offset + 1]
                            exec_price = float(next_bar["open"])
                        else:
                            exec_price = sig.entry_price

                        exec_price = rm.get_execution_price(exec_price, is_buy)

                        # 创建交易记录
                        trade = TradeRecord(
                            stock_code=code,
                            direction=direction_str,
                            open_time=bar_time[:5],
                            open_price=exec_price,
                            shares=sig.size_shares,
                            stop_loss_price=sig.stop_loss_price,
                            take_profit_tiers=sig.take_profit_tiers,
                            reverse_retreat_atr=sig.reverse_retreat_atr,
                            scenario=sig.scenario.value,
                            high_price=exec_price,
                            low_price=exec_price,
                        )

                        # 记录
                        pm.register_open(direction_str, sig.size_shares, exec_price)
                        account.total_trade_value += exec_price * sig.size_shares
                        rm.record_signal_executed(direction_str)
                        open_trades[code].append(trade)

                        # 更新当日成本
                        cost = exec_price * sig.size_shares * (self.commission + self.slippage)
                        account.daily_pnl_amount -= cost

                        # 信号日志
                        self.signal_log.append({
                            "date": day_date,
                            "time": bar_time[:5],
                            "code": code,
                            "direction": direction_str,
                            "scenario": sig.scenario.value,
                            "price": exec_price,
                            "shares": sig.size_shares,
                            "conditions_met": sig.conditions_met,
                        })

            # 更新当日净值
            # （每日结束时统一计算）

        # ---- 收盘清仓 ----
        for code in valid_codes:
            self._close_all_open_trades(
                code, open_trades, stock_minute_dict,
                idx_end - 1, "15:00", "END_OF_DAY",
                pos_managers, account, idx_start,
            )

        # 计算当日净值
        end_value = self._current_capital(account)

        return {
            "date": day_date,
            "start_value": initial_capital,
            "end_value": end_value,
            "day_return": (end_value - initial_capital) / initial_capital if initial_capital > 0 else 0,
            "signals_generated": day_signals_generated,
            "false_signals": day_false_signals,
            "trades_closed": len([t for t in self.all_trades
                                  if t.open_time.startswith(day_date[:10])]),
        }

    # ================================================================
    # 辅助方法
    # ================================================================

    def _close_trade(self,
                     code: str,
                     trade: TradeRecord,
                     close_price: float,
                     close_time: str,
                     stop_reason: str,
                     pm: PositionManager,
                     account: AccountState,
                     open_trades: Dict[str, List]):
        """平仓一笔交易"""
        direction = trade.direction
        shares = trade.shares

        # 如果是阶梯止盈①（部分卖出），剩余仓位继续持有
        partial_close = (stop_reason == "TAKE_PROFIT_TIER1")

        if partial_close:
            close_ratio = SCENARIO_RULES["range_trading"]["staggered_tp_1_ratio"]
            close_shares = int(shares * close_ratio / 100) * 100
            trade.shares -= close_shares  # 剩余仓位
            trade.partial_closed = True
        else:
            close_shares = shares

        # 计算盈亏
        if direction == "long":
            profit_pct = (close_price - trade.open_price) / trade.open_price * 100
        else:
            profit_pct = (trade.open_price - close_price) / trade.open_price * 100

        # 扣除交易成本
        cost_pct = (self.commission + self.slippage) * 2  # 双边
        profit_pct -= cost_pct * 100

        profit_amount = close_shares * trade.open_price * profit_pct / 100

        # 记录
        pm.register_close(direction, profit_pct, profit_amount)
        account.daily_pnl_amount += profit_amount
        account.daily_pnl_pct = account.daily_pnl_amount / self.initial_capital * 100

        if not partial_close:
            open_trades[code].remove(trade)

        # 添加到全局交易记录
        self.all_trades.append(CompletedTrade(
            stock_code=code,
            direction=direction,
            scenario=trade.scenario,
            open_time=trade.open_time,
            close_time=close_time,
            open_price=trade.open_price,
            close_price=close_price,
            shares=close_shares,
            profit_pct=profit_pct,
            profit_amount=profit_amount,
            is_win=profit_pct > 0,
            stop_reason=stop_reason,
        ))

        # 假信号统计
        self.total_signals += 1
        if profit_pct <= 0 and not partial_close:
            self.false_signals += 1

    def _close_all_open_trades(self,
                                code: str,
                                open_trades: Dict[str, List],
                                stock_minute_dict: dict,
                                bar_idx: int,
                                close_time: str,
                                reason: str,
                                pos_managers: dict,
                                account: AccountState,
                                day_start_idx: int):
        """强制平仓某标的的全部持仓"""
        for trade in list(open_trades.get(code, [])):
            stock_df = stock_minute_dict[code]
            if bar_idx < len(stock_df):
                close_price = float(stock_df.iloc[bar_idx]["close"])
            else:
                close_price = trade.open_price

            pm = pos_managers[code]
            self._close_trade(code, trade, close_price, close_time,
                              reason, pm, account, open_trades)

    def _current_capital(self, account: AccountState) -> float:
        """当前总资产（忽略未平仓浮盈）"""
        return self.initial_capital + account.daily_pnl_amount

    def _get_execution_price(self, price: float, is_sell: bool) -> float:
        """计算含滑点的执行价格"""
        if is_sell:
            return price * (1 - self.slippage)
        else:
            return price * (1 + self.slippage)

    def _time_to_minutes(self, time_str: str) -> int:
        """HH:MM:SS → 分钟数"""
        try:
            parts = time_str[-8:-3].split(":") if len(time_str) >= 8 else time_str.split(":")
            return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            return 0

    def _group_trading_days(self, index_df: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
        """
        将指数分钟数据按交易日分组

        Returns:
            {"YYYY-MM-DD": (start_idx, end_idx)}
        """
        if "timestamp" not in index_df.columns:
            return {}

        days = {}
        dates = index_df["timestamp"].dt.date.unique()

        for d in dates:
            mask = index_df["timestamp"].dt.date == d
            indices = index_df[mask].index
            if len(indices) >= 20:  # 至少 20 根 bar（100 分钟）
                days[str(d)] = (int(indices[0]), int(indices[-1]))

        return days

    # ================================================================
    # 指标计算
    # ================================================================

    def _compute_metrics(self) -> dict:
        """计算回测核心指标"""
        trades = self.all_trades
        values = self.daily_values

        if not trades:
            return {"error": "无交易记录", "total_trades": 0}

        # 胜率
        wins = [t for t in trades if t.is_win]
        win_rate = len(wins) / len(trades)

        # 盈亏比
        win_profits = [t.profit_pct for t in wins]
        loss_trades = [t for t in trades if not t.is_win]
        avg_win = np.mean(win_profits) if win_profits else 0
        avg_loss = abs(np.mean([t.profit_pct for t in loss_trades])) if loss_trades else 1
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # 总收益
        total_profit = sum(t.profit_amount for t in trades)
        total_return = total_profit / self.initial_capital

        # 年化收益
        if len(values) >= 2:
            dates = [datetime.strptime(v[0], "%Y-%m-%d") for v in values]
            days = (dates[-1] - dates[0]).days
            years = max(days / 252, 0.01)
            annual_return = (1 + total_return) ** (1 / years) - 1
        else:
            annual_return = total_return

        # 最大回撤（日频）
        if values:
            vals = pd.Series({v[0]: v[1] for v in values}).sort_index()
            peak = vals.cummax()
            drawdown = (vals - peak) / peak
            max_drawdown = abs(float(drawdown.min()))
        else:
            max_drawdown = 0

        # 夏普比率
        if len(values) >= 5:
            vals_series = pd.Series({v[0]: v[1] for v in values}).sort_index()
            daily_ret = vals_series.pct_change().dropna()
            sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
        else:
            sharpe = 0

        # 假信号率
        false_rate = self.false_signals / max(self.total_signals, 1)

        return {
            "initial_capital": self.initial_capital,
            "final_value": self.initial_capital + total_profit,
            "total_return": total_return,
            "annual_return": annual_return,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "total_trades": len(trades),
            "win_trades": len(wins),
            "loss_trades": len(loss_trades),
            "avg_win_pct": avg_win,
            "avg_loss_pct": -avg_loss,
            "false_signal_rate": false_rate,
            "total_signals": self.total_signals,
            "false_signals": self.false_signals,
        }

    def _check_acceptance(self, metrics: dict) -> dict:
        """验收检查"""
        bm = BACKTEST_ACCEPTANCE
        return {
            "win_rate": (metrics.get("win_rate", 0) >= bm["min_win_rate"],
                         f"{metrics.get('win_rate', 0):.1%} >= {bm['min_win_rate']:.0%}"),
            "profit_loss_ratio": (metrics.get("profit_loss_ratio", 0) >= bm["min_profit_loss_ratio"],
                                   f"{metrics.get('profit_loss_ratio', 0):.2f} >= {bm['min_profit_loss_ratio']}"),
            "max_drawdown": (metrics.get("max_drawdown", 1) < bm["max_daily_drawdown"],
                             f"{metrics.get('max_drawdown', 0):.2%} < {bm['max_daily_drawdown']:.0%}"),
            "false_signal_rate": (metrics.get("false_signal_rate", 1) < bm["max_false_signal_rate"],
                                   f"{metrics.get('false_signal_rate', 0):.1%} < {bm['max_false_signal_rate']:.0%}"),
        }

    def _print_report(self, metrics: dict, acceptance: dict):
        """打印回测报告"""
        bm = BACKTEST_ACCEPTANCE

        print(f"\n{'='*60}")
        print(f"  日内 T+0 回测报告")
        print(f"{'='*60}")
        print(f"  初始资金:     {metrics.get('initial_capital', 0):,.0f}")
        print(f"  最终净值:     {metrics.get('final_value', 0):,.0f}")
        print(f"  总收益率:     {metrics.get('total_return', 0):+.2%}")
        print(f"  年化收益率:   {metrics.get('annual_return', 0):+.2%}")
        print(f"  {'─'*56}")
        print(f"  胜率:         {metrics.get('win_rate', 0):.1%}  "
              f"[合格 >= {bm['min_win_rate']:.0%}]  "
              f"{'✓' if acceptance.get('win_rate', (False,))[0] else '✗'}")
        print(f"  盈亏比:       {metrics.get('profit_loss_ratio', 0):.2f}  "
              f"[合格 >= {bm['min_profit_loss_ratio']}]  "
              f"{'✓' if acceptance.get('profit_loss_ratio', (False,))[0] else '✗'}")
        print(f"  最大回撤:     {metrics.get('max_drawdown', 0):.2%}  "
              f"[合格 < {bm['max_daily_drawdown']:.0%}]  "
              f"{'✓' if acceptance.get('max_drawdown', (False,))[0] else '✗'}")
        print(f"  夏普比率:     {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  假信号率:     {metrics.get('false_signal_rate', 0):.1%}  "
              f"[合格 < {bm['max_false_signal_rate']:.0%}]  "
              f"{'✓' if acceptance.get('false_signal_rate', (False,))[0] else '✗'}")
        print(f"  {'─'*56}")
        print(f"  总交易次数:   {metrics.get('total_trades', 0)}")
        print(f"  盈利/亏损:    {metrics.get('win_trades', 0)}/{metrics.get('loss_trades', 0)}")
        print(f"  平均盈利:     {metrics.get('avg_win_pct', 0):+.2%}")
        print(f"  平均亏损:     {metrics.get('avg_loss_pct', 0):+.2%}")
        print(f"  总信号数:     {metrics.get('total_signals', 0)}")
        pass_count = sum(1 for v in acceptance.values() if v[0])
        print(f"  验收通过:     {pass_count}/{len(acceptance)}")
        print(f"{'='*60}\n")
        sys.stdout.flush()
