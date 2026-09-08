"""
短线回测引擎：逐日模拟 T日选股 → T+1买入 → T+N卖出
计算胜率、盈亏比、年化收益、最大回撤、夏普比率
"""
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

from short_term.config import (
    COMMISSION_RATE, SLIPPAGE, SINGLE_STOCK_WEIGHT,
    MAX_POSITION_COUNT, HOLD_DAYS, BACKTEST_BENCHMARKS,
)


class Position:
    """持仓记录"""
    def __init__(self, code: str, buy_date: str, buy_price: float,
                 shares: int, cost: float, weight: float):
        self.code = code
        self.buy_date = buy_date
        self.buy_price = buy_price
        self.shares = shares
        self.cost = cost        # 买入总成本（含手续费）
        self.weight = weight    # 仓位权重


class BacktestEngine:
    """短线回测引擎"""

    def __init__(self, initial_capital: float = 100_0000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.available_cash = initial_capital

        self.positions: List[Position] = []
        self.closed_trades: List[Dict] = []
        self.daily_values: Dict[str, float] = {}  # date → total_value

    def _buy(self, code: str, date: str, price: float, weight: float = SINGLE_STOCK_WEIGHT):
        """模拟买入"""
        # 计算买入金额（含滑点）
        buy_price = price * (1 + SLIPPAGE)
        target_amount = self.capital * weight

        # 不能超过可用现金
        actual_amount = min(target_amount, self.available_cash)
        if actual_amount < buy_price * 100:  # 至少买1手
            return None

        shares = int(actual_amount / buy_price / 100) * 100  # 整手
        if shares == 0:
            return None

        cost = shares * buy_price * (1 + COMMISSION_RATE)
        if cost > self.available_cash:
            shares = int(self.available_cash / (buy_price * (1 + COMMISSION_RATE)) / 100) * 100
            if shares == 0:
                return None
            cost = shares * buy_price * (1 + COMMISSION_RATE)

        pos = Position(
            code=code, buy_date=date, buy_price=buy_price,
            shares=shares, cost=cost, weight=weight,
        )
        self.positions.append(pos)
        self.available_cash -= cost

        return pos

    def _sell(self, position: Position, date: str, price: float) -> float:
        """模拟卖出，返回收益"""
        sell_price = price * (1 - SLIPPAGE)
        revenue = position.shares * sell_price * (1 - COMMISSION_RATE)
        profit = revenue - position.cost
        profit_pct = profit / position.cost

        self.closed_trades.append({
            "code": position.code,
            "buy_date": position.buy_date,
            "sell_date": date,
            "buy_price": position.buy_price,
            "sell_price": sell_price,
            "shares": position.shares,
            "profit": profit,
            "profit_pct": profit_pct,
            "win": profit > 0,
        })

        self.available_cash += revenue
        return profit

    def _total_value(self) -> float:
        """当前总资产（现金 + 持仓市值）"""
        # 简化：未卖出持仓按买入成本计算
        holding_value = sum(p.cost for p in self.positions)
        return self.available_cash + holding_value

    def run(self, kline_dict: dict, selection_dates: List[str],
            select_func) -> Dict:
        """
        执行回测

        Args:
            kline_dict: {code: kline_df}，每个DataFrame需有date列
            selection_dates: 选股日期列表（逐日）
            select_func: 选股函数 f(date, kline_dict) → [(code, score)]

        Returns:
            dict: 回测指标
        """
        print(f"\n[回测] ===== 短线回测开始 =====")
        print(f"[回测] 初始资金: {self.initial_capital:,.0f}")
        print(f"[回测] 回测区间: {selection_dates[0]} ~ {selection_dates[-1]}")
        print(f"[回测] 手续费: {COMMISSION_RATE:.4%}  滑点: {SLIPPAGE:.2%}")
        print(f"[回测] 单票仓位: {SINGLE_STOCK_WEIGHT:.0%}  持仓周期: T+{HOLD_DAYS}")

        for i, date in enumerate(selection_dates):
            # 1. 处理到期持仓（T+3卖出）
            sell_date = datetime.strptime(date, "%Y%m%d")

            for pos in self.positions[:]:
                buy_dt = datetime.strptime(pos.buy_date, "%Y%m%d")
                hold_days = (sell_date - buy_dt).days

                if hold_days >= HOLD_DAYS:
                    # 卖出
                    sell_price = self._get_close_price(kline_dict, pos.code, date)
                    if sell_price is None:
                        # 无价格数据，用买入价（保本卖出）
                        sell_price = pos.buy_price

                    profit = self._sell(pos, date, sell_price)
                    self.positions.remove(pos)

            # 2. 选股并买入
            max_positions = MAX_POSITION_COUNT
            current_count = len(self.positions)

            if current_count < max_positions:
                selections = select_func(date, kline_dict)
                buy_count = min(len(selections), max_positions - current_count)

                for j in range(buy_count):
                    code = selections[j][0]

                    # 避免重复持仓
                    if any(p.code == code for p in self.positions):
                        continue

                    buy_price = self._get_open_price(kline_dict, code, date)
                    if buy_price is None:
                        continue

                    pos = self._buy(code, date, buy_price)
                    if pos is None:
                        continue

            # 3. 记录每日净值
            self.daily_values[date] = self._total_value()

            if (i + 1) % 50 == 0:
                print(f"[回测] 进度: {i+1}/{len(selection_dates)}, 净值: {self._total_value():,.0f}")
                sys.stdout.flush()

        # 清仓剩余持仓
        last_date = selection_dates[-1]
        for pos in self.positions[:]:
            sell_price = self._get_close_price(kline_dict, pos.code, last_date)
            if sell_price is None:
                sell_price = pos.buy_price
            self._sell(pos, last_date, sell_price)
            self.positions.remove(pos)

        self.daily_values[last_date] = self._total_value()

        # 计算指标
        metrics = self._compute_metrics()
        self._print_report(metrics)

        return metrics

    def _get_close_price(self, kline_dict: dict, code: str, date: str) -> float:
        """获取某日收盘价"""
        df = kline_dict.get(code)
        if df is None:
            return None

        date_col = None
        for col_name in ["date", "日期"]:
            if col_name in df.columns:
                date_col = col_name
                break

        if date_col is None:
            # 假设按顺序排列
            return float(df["close"].iloc[-1])

        match = df[df[date_col].astype(str).str[:10] == date]
        if match.empty:
            return None

        return float(match["close"].iloc[-1])

    def _get_open_price(self, kline_dict: dict, code: str, date: str) -> float:
        """获取某日开盘价"""
        df = kline_dict.get(code)
        if df is None:
            return None

        date_col = None
        for col_name in ["date", "日期"]:
            if col_name in df.columns:
                date_col = col_name
                break

        if date_col is None:
            return float(df["open"].iloc[-1])

        match = df[df[date_col].astype(str).str[:10] == date]
        if match.empty:
            return None

        return float(match["open"].iloc[-1])

    def _compute_metrics(self) -> Dict:
        """计算回测核心指标（复用统一指标层 base/backtest.py）"""
        from base.backtest import compute_backtest_metrics

        trades = self.closed_trades
        if not trades:
            return {"error": "无交易记录"}

        trades_pct = [t["profit_pct"] for t in trades]
        equity = pd.Series(self.daily_values).sort_index()

        metrics = compute_backtest_metrics(
            trades_pct=trades_pct,
            equity=equity,
            initial_capital=self.initial_capital,
            final_value=self._total_value(),
        )
        metrics["initial_capital"] = self.initial_capital
        metrics["final_value"] = self._total_value()
        return metrics

    def _print_report(self, metrics: Dict):
        """打印回测报告"""
        bm = BACKTEST_BENCHMARKS

        print(f"\n{'='*60}")
        print(f"  短线回测报告")
        print(f"{'='*60}")
        print(f"  初始资金:     {metrics.get('initial_capital', 0):,.0f}")
        print(f"  最终净值:     {metrics.get('final_value', 0):,.0f}")
        print(f"  总收益率:     {metrics.get('total_return', 0):.2%}")
        print(f"  年化收益率:   {metrics.get('annual_return', 0):.2%}  "
              f"[合格线: >= {bm['annual_return']:.0%}] {'✓' if metrics.get('annual_return', 0) >= bm['annual_return'] else '✗'}")
        print(f"  胜率:         {metrics.get('win_rate', 0):.2%}  "
              f"[合格线: >= {bm['win_rate']:.0%}] {'✓' if metrics.get('win_rate', 0) >= bm['win_rate'] else '✗'}")
        print(f"  盈亏比:       {metrics.get('profit_loss_ratio', 0):.2f}  "
              f"[合格线: >= {bm['profit_loss_ratio']:.1f}] {'✓' if metrics.get('profit_loss_ratio', 0) >= bm['profit_loss_ratio'] else '✗'}")
        print(f"  最大回撤:     {metrics.get('max_drawdown', 0):.2%}  "
              f"[合格线: <= {bm['max_drawdown']:.0%}] {'✓' if metrics.get('max_drawdown', 0) <= bm['max_drawdown'] else '✗'}")
        print(f"  夏普比率:     {metrics.get('sharpe_ratio', 0):.2f}  "
              f"[合格线: >= {bm['sharpe_ratio']:.1f}] {'✓' if metrics.get('sharpe_ratio', 0) >= bm['sharpe_ratio'] else '✗'}")
        print(f"  总交易次数:   {metrics.get('total_trades', 0)}")
        print(f"  平均盈利:     {metrics.get('avg_win_pct', 0):.2%}")
        print(f"  平均亏损:     {metrics.get('avg_loss_pct', 0):.2%}")
        print(f"{'='*60}\n")


def simple_select_func(date: str, kline_dict: dict, top_n: int = 3) -> List[Tuple[str, float]]:
    """
    简化选股函数（用于回测中无模型时的fallback）
    基于当日涨跌幅和量比排序

    Returns:
        [(code, score), ...]
    """
    candidates = []
    for code, df in kline_dict.items():
        if df is None or len(df) < 6:
            continue

        close = df["close"].astype(float)
        vol = df["volume"].astype(float)

        if len(close) < 2:
            continue

        day_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
        vol_ratio = vol.iloc[-1] / vol.tail(6).head(5).mean() if vol.tail(6).head(5).mean() > 0 else 0

        # 选涨幅3-5%且放量的
        if 0.03 <= day_change <= 0.05 and vol_ratio >= 1.3:
            score = day_change * 0.5 + min(vol_ratio / 3, 1.0) * 0.5
            candidates.append((code, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_n]
