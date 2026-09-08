"""
信号 → 下单桥：把短线选股结果（short_term/output/selection_*.json）转成下单委托

选股 JSON 每条记录含 code/name/position("20%")/buy_date/sell_date/stop_loss/stop_profit。
本桥负责：
  1. 读取选股结果
  2. 按 position 百分比与资金计算整手股数
  3. 生成买入委托（价格由 price_provider 提供）
  4. 提炼止损/止盈/卖出日规则（供后续盯盘程序消费）
"""
import json
import re
from datetime import datetime, timedelta

from broker.base import Order
from broker.config import MIN_LOT, MAX_POSITION_COUNT, SINGLE_STOCK_WEIGHT


def _pct(text: str):
    """从 '20%' / '-5% 严格止损' 中解析出小数比例"""
    m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", text or "")
    return float(m.group(1)) / 100.0 if m else None


def load_selection(path: str) -> dict:
    """读取选股结果 JSON"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def akshare_price_provider(code: str):
    """用 akshare 日线取最新收盘价作为委托参考价（真实接入用）"""
    import akshare as ak

    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=start, end_date=end, adjust="")
        if df is not None and len(df) > 0:
            return float(df["收盘"].iloc[-1])
    except Exception:
        pass
    return None


def build_orders(selection: dict, capital: float, price_provider) -> tuple:
    """
    将选股结果转为买入委托 + 出场规则

    Args:
        selection: load_selection 的返回值
        capital: 账户总资金（用于按仓位百分比计算金额）
        price_provider: callable(code) -> float 或 None，返回最新价

    Returns:
        (orders: list[Order], exit_rules: list[dict])
    """
    orders = []
    exit_rules = []
    skipped = []

    for s in selection.get("selections", [])[:MAX_POSITION_COUNT]:
        code = s.get("code")
        name = s.get("name", "")
        price = price_provider(code)
        if not price:
            skipped.append(f"{code} {name}（无价格，跳过）")
            continue

        pos_pct = _pct(s.get("position")) or SINGLE_STOCK_WEIGHT
        target_value = capital * pos_pct
        qty = int(target_value / price / MIN_LOT) * MIN_LOT
        if qty <= 0:
            skipped.append(f"{code} {name}（资金不足一手，跳过）")
            continue

        orders.append(Order(
            code=code,
            name=name,
            side="buy",
            quantity=qty,
            price=round(price, 3),
            reason="短线选股信号",
        ))

        exit_rules.append({
            "code": code,
            "name": name,
            "buy_date": s.get("buy_date", ""),
            "sell_date": s.get("sell_date", ""),
            "stop_loss_pct": _pct(s.get("stop_loss")),
            "stop_profit_pct": _pct(s.get("stop_profit")),
        })

    return orders, exit_rules, skipped
