"""
逆向仓位修正的状态机：跟踪极端区间持续时长，分阶段缓慢调整仓位。

原则：
  - 不因单月极端信号就大幅调仓（防抄底抄在半山腰）
  - 连续 2+ 个月的冰点 + 便宜信号，才逐步抬升仓位
  - 连续 2+ 个月的过热 + 泡沫信号，才逐步压低仓位
  - 环境回暖立即重置计数
"""
import json
import os
from datetime import datetime

STATE_FILE = "output/contrarian_state.json"


def _load_state() -> dict:
    """加载逆向状态"""
    if not os.path.exists(STATE_FILE):
        return {
            "pessimistic_months": 0,
            "optimistic_months": 0,
            "last_month": None,
            "updated_at": None,
        }
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {
            "pessimistic_months": 0,
            "optimistic_months": 0,
            "last_month": None,
            "updated_at": None,
        }


def _save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    state["updated_at"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def compute_contrarian_correction(
    macro_score: float,
    indicator_scores: dict,
    month_label: str = None,
) -> tuple:
    """
    计算逆向修正和原因列表。

    Args:
        macro_score: 宏观总分 (0~1)
        indicator_scores: 所有指标得分字典
        month_label: 当前月份标识（如 "2026-06"），用于状态持久化

    Returns:
        (correction_float, reasons_list)
    """
    indicator_scores = indicator_scores or {}

    # ── 1. 判定当前是否处于极端区间 ──
    is_pessimistic = macro_score < 0.35
    is_optimistic = macro_score > 0.65
    has_poor_signal = False
    has_hot_signal = False

    if is_pessimistic:
        erp = indicator_scores.get("erp", {})
        adr = indicator_scores.get("advance_decline_ratio", {})
        if erp.get("score", 0) >= 0.8 or adr.get("raw_value", 0.5) < 0.3:
            has_poor_signal = True

    if is_optimistic:
        to = indicator_scores.get("turnover_extreme", {})
        gpe = indicator_scores.get("growth_pe_percentile", {})
        if to.get("raw_value", 1.0) > 2.0 or gpe.get("raw_value", 0.5) > 0.8:
            has_hot_signal = True

    # ── 2. 加载历史状态 ──
    state = _load_state()
    last_month = state.get("last_month")

    # 如果跨月了（或首次运行），更新计数
    if month_label and month_label != last_month:
        # 悲观区
        if has_poor_signal:
            state["pessimistic_months"] = state.get("pessimistic_months", 0) + 1
        else:
            state["pessimistic_months"] = 0

        # 乐观区
        if has_hot_signal:
            state["optimistic_months"] = state.get("optimistic_months", 0) + 1
        else:
            state["optimistic_months"] = 0

        state["last_month"] = month_label
        _save_state(state)

    poor_count = state.get("pessimistic_months", 0)
    hot_count = state.get("optimistic_months", 0)

    # ── 3. 分阶段计算修正 ──
    correction = 0.0
    reasons = []

    if poor_count >= 2:
        # 悲观信号连续 ≥2 个月 → 缓慢加仓
        # 每月 +2%，第8个月到顶 +15%
        ramp = min((poor_count - 1) * 0.02, 0.15)
        correction += ramp

        erp = indicator_scores.get("erp", {})
        adr = indicator_scores.get("advance_decline_ratio", {})
        erp_v = erp.get("raw_value", None)
        adr_v = adr.get("raw_value", None)

        detail = []
        if erp_v is not None and erp.get("score", 0) >= 0.8:
            detail.append(f"ERP={erp_v:.1f}%（股极便宜）")
        if adr_v is not None and adr_v < 0.3:
            detail.append(f"涨跌比={adr_v:.0%}（恐慌冰点）")

        # 几个月后到顶
        months_to_max = max(1, int((0.15 - ramp) / 0.02) + poor_count)
        reasons.append(
            f"连续{poor_count}个月悲观（{' + '.join(detail)}）→ 逆向加仓 +{ramp:.0%}（预计第{months_to_max}个月到顶+15%）"
        )

    if hot_count >= 2:
        # 过热信号连续 ≥2 个月 → 缓慢减仓
        # 每月 -2%，第7个月到顶 -12%
        ramp = min((hot_count - 1) * 0.02, 0.12)
        correction -= ramp

        to = indicator_scores.get("turnover_extreme", {})
        gpe = indicator_scores.get("growth_pe_percentile", {})
        to_v = to.get("raw_value", None)
        gpe_v = gpe.get("raw_value", None)

        detail = []
        if to_v is not None and to_v > 2.0:
            detail.append(f"量比={to_v:.1f}x（全民炒股）")
        if gpe_v is not None and gpe_v > 0.8:
            detail.append(f"创业板PE分位={gpe_v:.0%}（泡沫）")

        months_to_max = max(1, int((0.12 - ramp) / 0.02) + hot_count)
        reasons.append(
            f"连续{hot_count}个月过热（{' + '.join(detail)}）→ 逆向减仓 -{ramp:.0%}（预计第{months_to_max}个月到顶-12%）"
        )

    # 刚检测到但不满2个月 → 只预警，不操作
    if poor_count == 1 and has_poor_signal:
        reasons.append("⚠️ 首次检测到悲观信号，进入观察期（需连续2月确认后才逆向加仓）")
    if hot_count == 1 and has_hot_signal:
        reasons.append("⚠️ 首次检测到过热信号，进入观察期（需连续2月确认后才逆向减仓）")

    return correction, reasons
