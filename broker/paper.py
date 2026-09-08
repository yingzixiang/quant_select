"""
模拟盘券商：本地撮合 + JSON 落盘

用于在没有真实账户的情况下跑通「信号 → 下单 → 持仓」全链路。
成交按委托价立即撮合（不做对手盘/滑点），交易成本按 config 计。
状态持久化到 PAPER_STATE_FILE，重复运行会从上次状态续接。
"""
import json
import os
import uuid

from broker.base import Broker, Order, Position
from broker.config import (
    COMMISSION_RATE, STAMP_TAX_RATE, MIN_LOT, PAPER_STATE_FILE,
)


class PaperBroker(Broker):
    def __init__(self, initial_cash: float, state_file: str = PAPER_STATE_FILE):
        self.state_file = state_file
        self.cash = float(initial_cash)
        self.positions = {}   # code -> {"quantity": int, "avg_price": float, "name": str}
        self.orders = []      # list[dict]
        self._load()

    # ---------- 持久化 ----------
    def _load(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.cash = float(state.get("cash", self.cash))
            self.positions = state.get("positions", {})
            self.orders = state.get("orders", [])
        except Exception:
            # 状态文件损坏时从空账户开始，不阻塞交易链路
            pass

    def _save(self):
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        state = {
            "cash": self.cash,
            "positions": self.positions,
            "orders": self.orders,
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    # ---------- Broker 契约 ----------
    def connect(self) -> bool:
        return True

    def get_cash(self) -> float:
        return self.cash

    def get_positions(self) -> list:
        return [
            Position(code=code, quantity=p["quantity"], avg_price=p["avg_price"], name=p.get("name", ""))
            for code, p in self.positions.items()
            if p["quantity"] > 0
        ]

    def place_order(self, order: Order) -> Order:
        if order.quantity <= 0 or order.price <= 0:
            order.status = "rejected"
            order.order_id = f"R{uuid.uuid4().hex[:8]}"
            self.orders.append(order.to_dict())
            self._save()
            return order

        if order.side == "buy":
            if order.quantity % MIN_LOT != 0:
                order.status = "rejected"
                order.order_id = f"R{uuid.uuid4().hex[:8]}"
                self.orders.append(order.to_dict())
                self._save()
                return order
            cost = order.quantity * order.price * (1 + COMMISSION_RATE)
            if cost > self.cash:
                order.status = "rejected"
                order.order_id = f"R{uuid.uuid4().hex[:8]}"
                self.orders.append(order.to_dict())
                self._save()
                return order
            self.cash -= cost
            self._add_position(order.code, order.name, order.quantity, order.price)
        elif order.side == "sell":
            pos = self.positions.get(order.code)
            if not pos or pos["quantity"] < order.quantity:
                order.status = "rejected"
                order.order_id = f"R{uuid.uuid4().hex[:8]}"
                self.orders.append(order.to_dict())
                self._save()
                return order
            revenue = order.quantity * order.price * (1 - COMMISSION_RATE - STAMP_TAX_RATE)
            self.cash += revenue
            self._reduce_position(order.code, order.quantity)
        else:
            order.status = "rejected"
            order.order_id = f"R{uuid.uuid4().hex[:8]}"
            self.orders.append(order.to_dict())
            self._save()
            return order

        order.status = "filled"
        order.order_id = f"P{uuid.uuid4().hex[:8]}"
        order.filled_price = order.price
        order.filled_at = order.created_at
        self.orders.append(order.to_dict())
        self._save()
        return order

    def cancel_order(self, order_id: str) -> bool:
        for o in self.orders:
            if o.get("order_id") == order_id and o.get("status") == "pending":
                o["status"] = "cancelled"
                self._save()
                return True
        return False

    def get_orders(self) -> list:
        return [Order.from_dict(o) for o in self.orders]

    # ---------- 内部 ----------
    def _add_position(self, code: str, name: str, quantity: int, price: float):
        pos = self.positions.get(code)
        if not pos:
            self.positions[code] = {"quantity": quantity, "avg_price": price, "name": name}
            return
        total_qty = pos["quantity"] + quantity
        pos["avg_price"] = (pos["avg_price"] * pos["quantity"] + price * quantity) / total_qty
        pos["quantity"] = total_qty
        pos["name"] = name or pos.get("name", "")

    def _reduce_position(self, code: str, quantity: int):
        pos = self.positions.get(code)
        if not pos:
            return
        pos["quantity"] -= quantity
        if pos["quantity"] <= 0:
            self.positions.pop(code, None)
