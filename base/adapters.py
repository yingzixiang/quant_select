"""
薄适配器：把现有两套引擎包装成统一 Strategy 接口（收敛重构 P3）

不重写引擎内部，仅在边缘把两套输出（中线 snapshot dict / 短线 selection DataFrame）
归一化到 base.strategy.Portfolio，让统一 CLI 与未来下单/盯盘能按同一接口消费。

老入口（main.py / short_term.main_short）保持不变，本模块为新增的统一门面。
"""
import os

import pandas as pd

from base.strategy import Strategy, Signal, Portfolio


class MonthlyStrategy(Strategy):
    """中线月度再平衡引擎适配器（包装 core.rebalance.run_monthly_rebalance）"""
    name = "monthly"

    def run(self, **kwargs) -> Portfolio:
        from core.rebalance import run_monthly_rebalance

        snapshot = run_monthly_rebalance(**kwargs)
        if not snapshot:
            return Portfolio(strategy=self.name)

        signals = []
        for d in snapshot.get("portfolio_details", []):
            weight = d.get("weight") if d.get("weight") else None
            signals.append(Signal(
                code=d["code"],
                name=d.get("name", ""),
                market="HK" if d.get("market") == "港" else "A",
                weight=weight if weight else 0.0,
                score=float(d.get("score") or 0.0),
                exit_rule={"industry": d.get("industry", ""),
                           "pb": d.get("pb"), "roe": d.get("roe")},
            ))

        style = snapshot.get("style_allocation", {})
        post = snapshot.get("post_check", {})
        return Portfolio(
            strategy=self.name,
            date=snapshot.get("month", ""),
            signals=signals,
            stock_weight=float(snapshot.get("stock_weight", 1.0)),
            bond_weight=float(snapshot.get("bond_weight", 0.0)),
            style=dict(style) if style else {},
            meta={"macro_score": snapshot.get("macro_score")},
            risk={
                "passed": post.get("passed", True),
                "violations": post.get("violations", []),
                "warnings": post.get("warnings", []),
            },
        )


class SwingStrategy(Strategy):
    """短线 T+1~3 选股引擎适配器（包装 short_term.daily_pipeline.run_daily_selection）"""
    name = "swing"

    def run(self, **kwargs) -> Portfolio:
        from short_term.daily_pipeline import run_daily_selection

        target_date = kwargs.get("target_date")
        df = run_daily_selection(target_date=target_date)
        if df is None or df.empty:
            return Portfolio(strategy=self.name, date=target_date or "")

        signals = []
        for _, row in df.iterrows():
            signals.append(Signal(
                code=str(row.get("code", "")),
                name=str(row.get("name", "")),
                market="A",
                weight=0.0,  # 短线单票仓位由 broker 层按 position 百分比计算
                score=float(row.get("final_score") or 0.0),
                exit_rule={
                    "probability": row.get("probability"),
                    "buy_date": row.get("buy_date"),
                    "sell_date": row.get("sell_date"),
                    "stop_loss": row.get("stop_loss"),
                    "stop_profit": row.get("stop_profit"),
                    "position": row.get("position"),
                },
            ))

        return Portfolio(
            strategy=self.name,
            date=target_date or "",
            signals=signals,
            stock_weight=1.0,
        )


STRATEGIES = {
    "monthly": MonthlyStrategy,
    "swing": SwingStrategy,
}


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"未知策略 {name!r}，可用: {list(STRATEGIES)}")
    return STRATEGIES[name]()
