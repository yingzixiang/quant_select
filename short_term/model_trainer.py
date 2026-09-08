"""
XGBoost 二分类模型：训练、预测、评估、持久化
预测 "3日上涨概率"，用于短线选股排序
"""
import os
import pickle
import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta

from short_term.config import (
    XGB_PARAMS, TRAIN_WINDOW_DAYS, TEST_WINDOW_DAYS,
    LABEL_FORWARD_DAYS, LABEL_POSITIVE_THRESHOLD, LABEL_NEGATIVE_THRESHOLD,
    PREDICT_PROB_THRESHOLD, TOP_N_SELECT,
    MODEL_OUTPUT_DIR, MIN_FACTOR_IC, MAX_FACTOR_COUNT,
)


def prepare_labels(kline_dict: dict, base_date: str) -> pd.Series:
    """
    准备训练标签：计算未来3日收益率

    Args:
        kline_dict: {code: kline_df}，kline_df 需按日期升序排列
        base_date: 基准日期 YYYYMMDD（该日收盘后选股）

    Returns:
        Series: index=code, value=未来3日收益率
    """
    labels = {}
    target_date = datetime.strptime(base_date, "%Y%m%d")
    end_date = target_date + timedelta(days=LABEL_FORWARD_DAYS + 5)  # 多取几天容错

    for code, df in kline_dict.items():
        if df is None or len(df) < LABEL_FORWARD_DAYS + 1:
            continue

        close = df["close"].astype(float)

        # 找到 base_date 对应的行
        date_col = None
        for col_name in ["date", "日期"]:
            if col_name in df.columns:
                date_col = col_name
                break

        if date_col is None:
            base_idx = len(df) - 1
        else:
            dates = pd.to_datetime(df[date_col]).dt.strftime("%Y%m%d")
            match = df[dates == base_date]
            if match.empty:
                # 非交易日，取 base_date 之前最近的一个交易日
                earlier = df[dates <= base_date]
                if earlier.empty:
                    continue
                base_idx = earlier.index[-1]
            else:
                base_idx = match.index[-1]

        # 需要base_idx之后有 LABEL_FORWARD_DAYS 天的数据
        if base_idx + LABEL_FORWARD_DAYS >= len(close):
            continue

        base_price = close.iloc[base_idx]
        forward_price = close.iloc[base_idx + LABEL_FORWARD_DAYS]

        if base_price > 0:
            forward_return = (forward_price - base_price) / base_price
            labels[code] = forward_return

    return pd.Series(labels)


def labels_to_binary(labels: pd.Series) -> pd.Series:
    """
    将连续收益率转为二分类标签：
    3日收益 > 5%  → 1（正样本）
    3日收益 < 0   → 0（负样本）
    0~5%之间的样本丢弃（避免噪声标签）
    """
    binary = pd.Series(index=labels.index, dtype=int)
    for code in labels.index:
        ret = labels[code]
        if ret > LABEL_POSITIVE_THRESHOLD:
            binary[code] = 1
        elif ret < LABEL_NEGATIVE_THRESHOLD:
            binary[code] = 0
        else:
            binary[code] = -1  # 标记为丢弃

    # 只保留有效标签
    valid = binary[binary >= 0]
    print(f"[模型] 标签分布: 正样本 {(valid == 1).sum()}，负样本 {(valid == 0).sum()}")
    return valid


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  X_valid: np.ndarray = None, y_valid: np.ndarray = None,
                  scale_pos_weight: float = None):
    """
    训练XGBoost二分类模型（修复时序泄漏：无验证集时不再随机切分）

    Args:
        X_train: 训练特征矩阵
        y_train: 训练标签
        X_valid: 验证特征（用于early stopping）
        y_valid: 验证标签
        scale_pos_weight: 正样本权重（None 则自动按 负样本数/正样本数 平衡）

    Returns:
        trained XGBoost model
    """
    import xgboost as xgb

    params = XGB_PARAMS.copy()
    early_stopping = params.pop("early_stopping_rounds", 30)

    if scale_pos_weight is None:
        n_neg = int((np.asarray(y_train) == 0).sum())
        n_pos = int((np.asarray(y_train) == 1).sum())
        scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    model = xgb.XGBClassifier(**params)
    if X_valid is not None and y_valid is not None:
        model.fit(
            X_train, y_train,
            sample_weight=_sample_weight(y_train, scale_pos_weight),
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=False,
        )
    else:
        # 无验证集时不再用随机 train_test_split（会时序泄漏），
        # 直接全量训练；样本权重仅用于损失，不改变标签分布
        model.fit(
            X_train, y_train,
            sample_weight=_sample_weight(y_train, scale_pos_weight),
            verbose=False,
        )

    return model


def _sample_weight(y, scale_pos_weight: float):
    """正样本权重 = scale_pos_weight，负样本权重 = 1.0"""
    y = np.asarray(y)
    return np.where(y == 1, scale_pos_weight, 1.0)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    评估模型性能

    Returns:
        dict: {auc, accuracy, precision, recall, feature_importance}
    """
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        "auc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }

    # 特征重要性 Top20
    importance = model.feature_importances_
    feature_imp = pd.Series(importance).sort_values(ascending=False).head(20)
    metrics["feature_importance_top20"] = feature_imp.to_dict()

    return metrics


def predict_probability(model, X: np.ndarray, codes: list) -> pd.DataFrame:
    """
    预测上涨概率

    Args:
        model: trained XGBoost model
        X: 特征矩阵
        codes: 对应的股票代码列表

    Returns:
        DataFrame: code, probability, rank
    """
    if X.shape[0] == 0:
        return pd.DataFrame(columns=["code", "probability", "rank"])

    prob = model.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        "code": codes,
        "probability": prob,
    })
    result = result.sort_values("probability", ascending=False)
    result["rank"] = range(1, len(result) + 1)
    return result


def prepare_multi_date_dataset(kline_dict: dict, date_list: list,
                                alpha_func=None) -> tuple:
    """
    多日期采样训练集：对每个日期计算因子和标签，堆叠成大训练集

    Args:
        kline_dict: {code: kline_df}，kline_df 需覆盖全时段
        date_list: 训练日期列表 ["YYYYMMDD", ...]
        alpha_func: 因子计算函数，默认 compute_alpha_factors_single

    Returns:
        (X, y, codes_used, feature_names) or None
    """
    from short_term.alpha_factors import compute_alpha_factors_single

    if alpha_func is None:
        alpha_func = compute_alpha_factors_single

    all_X = []
    all_y = []
    all_y_cont = []
    all_date_codes = []
    feature_names = None

    for di, train_date in enumerate(date_list):
        # 计算标签
        labels = prepare_labels(kline_dict, train_date)
        binary_labels = labels_to_binary(labels)
        if len(binary_labels) < 20:
            continue

        # 对每个有标签的股票，计算该日期的因子
        date_str = f"{train_date[:4]}-{train_date[4:6]}-{train_date[6:8]}"
        rows = []
        valid_codes = []

        for code in binary_labels.index:
            df = kline_dict.get(code)
            if df is None or len(df) < 60:
                continue

            # 截取 train_date 之前的数据
            date_col = "date"
            df_before = df[df[date_col].astype(str).str[:10] <= date_str]
            if len(df_before) < 60:
                continue

            # Alpha因子
            factors = alpha_func(df_before)
            if factors is None or len(factors) < 10:
                continue

            # 补充自定义量价/K线因子（从kline直接计算，无需外部数据）
            close = df_before["close"].astype(float)
            vol = df_before["volume"].astype(float)
            open_ = df_before["open"].astype(float)
            high = df_before["high"].astype(float)
            low = df_before["low"].astype(float)

            hl_range = high.iloc[-1] - low.iloc[-1]
            factors["candle_body_ratio"] = (close.iloc[-1] - open_.iloc[-1]) / hl_range if hl_range != 0 else 0
            body_high = max(open_.iloc[-1], close.iloc[-1])
            factors["upper_shadow_ratio"] = (high.iloc[-1] - body_high) / hl_range if hl_range != 0 else 0
            body_low = min(open_.iloc[-1], close.iloc[-1])
            factors["lower_shadow_ratio"] = (body_low - low.iloc[-1]) / hl_range if hl_range != 0 else 0
            factors["open_gap"] = open_.iloc[-1] / close.iloc[-2] - 1 if len(close) >= 2 and close.iloc[-2] != 0 else 0
            factors["intraday_volatility"] = hl_range / close.iloc[-1] if close.iloc[-1] != 0 else 0

            # 量价结构因子
            if len(close) >= 20:
                c5 = open_.tail(5).corr(vol.tail(5))
                c20 = open_.tail(20).corr(vol.tail(20))
                factors["vol_price_corr_diff"] = c5 - c20 if not (np.isnan(c5) or np.isnan(c20)) else 0
            else:
                factors["vol_price_corr_diff"] = 0

            if len(close) >= 2:
                vc = vol.iloc[-1] - vol.iloc[-2]
                pc = close.iloc[-1] - close.iloc[-2]
                factors["vol_price_slope"] = vc / pc if abs(pc) >= 1e-6 else 0
            else:
                factors["vol_price_slope"] = 0

            avg_v5 = vol.tail(5).mean()
            vr = vol.iloc[-1] / avg_v5 if avg_v5 > 0 else 1.0
            is_pb = close.iloc[-1] < close.iloc[-2] if len(close) >= 2 else False
            factors["shrink_pullback"] = (0.7 - vr) / 0.7 if (is_pb and vr < 0.7) else 0

            m5v = vol.tail(6).head(5).mean()
            m20v = vol.tail(21).head(20).mean() if len(vol) >= 21 else m5v
            bl = max(m5v, m20v)
            factors["volume_breakout_strength"] = min(vol.iloc[-1] / bl / 3.0, 1.0) if bl > 0 else 0

            # 资金趋势代理
            ret3 = close.pct_change(3).iloc[-1] if len(close) >= 4 else 0
            vr3 = vol.tail(3).mean() / vol.tail(6).head(3).mean() if len(vol) >= 6 else 1
            factors["ret_3d"] = ret3
            factors["money_trend_3d"] = ret3 * vr3

            rows.append(factors)
            valid_codes.append(code)

        if len(rows) < 20:
            continue

        if feature_names is None:
            feature_names = sorted(rows[0].keys())

        # 对齐特征
        X_day = np.array([[r.get(f, 0.0) for f in feature_names] for r in rows])
        y_day = binary_labels.loc[valid_codes].values
        y_cont_day = labels.loc[valid_codes].values

        # 去除 NaN/Inf
        valid_rows = ~np.isnan(X_day).any(axis=1) & ~np.isinf(X_day).any(axis=1)
        X_day = X_day[valid_rows]
        y_day = y_day[valid_rows]
        y_cont_day = y_cont_day[valid_rows]

        if len(y_day) >= 20:
            all_X.append(X_day)
            all_y.append(y_day)
            all_y_cont.append(y_cont_day)
            all_date_codes.extend([(train_date, c) for c in np.array(valid_codes)[valid_rows]])

        if (di + 1) % 10 == 0:
            total_samples = sum(len(x) for x in all_X)
            print(f"[多日期采样] 进度: {di+1}/{len(date_list)}  累计样本: {total_samples}")
            sys.stdout.flush()

    if not all_X:
        return None

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    y_cont = np.concatenate(all_y_cont)
    print(f"[多日期采样] 完成: {len(date_list)} 个日期 → {X.shape[0]} 样本 × {X.shape[1]} 特征")
    print(f"[多日期采样] 正样本: {y.sum()} ({(y.sum()/len(y))*100:.1f}%)")

    # 因子 IC 筛选（训练时真正生效）
    from short_term.factor_engine import select_features_by_ic
    X, feature_names = select_features_by_ic(
        X, y_cont, feature_names, min_abs_ic=MIN_FACTOR_IC, max_factors=MAX_FACTOR_COUNT,
    )

    return X, y, all_date_codes, feature_names


def rolling_train(kline_dict: dict = None, factor_df: pd.DataFrame = None,
                  train_end_date: str = None, feature_cols: list = None,
                  X_multi: np.ndarray = None, y_multi: np.ndarray = None,
                  feature_names_multi: list = None) -> dict:
    """
    训练模型（支持单日和多日两种模式）

    单日模式: 传入 kline_dict + factor_df + train_end_date
    多日模式: 传入 X_multi + y_multi + feature_names_multi

    Returns:
        dict: {model, metrics, feature_cols, train_end_date}
    """
    import xgboost as xgb

    # 多日期模式
    if X_multi is not None and y_multi is not None:
        print(f"\n[模型] 多日期训练: {X_multi.shape[0]} 样本 × {X_multi.shape[1]} 特征")
        print(f"[模型] 正样本率: {y_multi.sum()/len(y_multi):.2%}")

        if y_multi.sum() < 10 or (len(y_multi) - y_multi.sum()) < 10:
            print("[模型] 样本不均衡，无法训练")
            return None

        model = train_xgboost(X_multi, y_multi)
        from sklearn.metrics import roc_auc_score
        y_prob = model.predict_proba(X_multi)[:, 1]
        auc = roc_auc_score(y_multi, y_prob)
        print(f"[模型] 训练AUC: {auc:.4f}")

        return {
            "model": model,
            "metrics": {"auc": auc, "n_samples": len(y_multi),
                        "positive_ratio": y_multi.sum() / len(y_multi)},
            "feature_cols": feature_names_multi,
            "train_end_date": "multi_date",
        }

    # 单日期模式（兼容旧接口）
    if kline_dict is None or factor_df is None or train_end_date is None:
        print("[模型] 参数不足")
        return None

    target_date = datetime.strptime(train_end_date, "%Y%m%d")
    train_start = target_date - timedelta(days=TRAIN_WINDOW_DAYS + 30)
    train_start_str = train_start.strftime("%Y%m%d")

    print(f"\n[模型] 滚动训练: {train_start_str} ~ {train_end_date}")

    # 准备标签
    labels = prepare_labels(kline_dict, train_end_date)
    binary_labels = labels_to_binary(labels)

    # 对齐因子和标签
    common_codes = factor_df.index.intersection(binary_labels.index)
    if len(common_codes) < 100:
        print(f"[模型] 训练样本不足: {len(common_codes)}")
        return None

    if feature_cols is None:
        feature_cols = [c for c in factor_df.columns if c in factor_df.columns]
    else:
        feature_cols = [c for c in feature_cols if c in factor_df.columns]

    X = factor_df.loc[common_codes, feature_cols].values
    y = binary_labels.loc[common_codes].values

    # 去除含有 NaN 的行
    valid_rows = ~np.isnan(X).any(axis=1)
    X = X[valid_rows]
    y = y[valid_rows]

    if len(y) < 100 or y.sum() < 10 or (len(y) - y.sum()) < 10:
        print(f"[模型] 有效样本不足: total={len(y)}, positive={y.sum()}")
        return None

    print(f"[模型] 训练样本: {len(y)}，正样本率: {y.sum()/len(y):.2%}")

    # 训练
    model = train_xgboost(X, y)

    # 评估（简单验证）
    from sklearn.metrics import roc_auc_score
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
    print(f"[模型] 训练AUC: {auc:.4f}")

    return {
        "model": model,
        "metrics": {"auc": auc, "n_samples": len(y), "positive_ratio": y.sum() / len(y)},
        "feature_cols": feature_cols,
        "train_end_date": train_end_date,
    }


def save_model(model_info: dict, filename: str = None):
    """保存模型到磁盘"""
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    if filename is None:
        date_str = model_info.get("train_end_date", datetime.now().strftime("%Y%m%d"))
        filename = f"xgboost_{date_str}.pkl"

    path = os.path.join(MODEL_OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model_info, f)
    print(f"[模型] 已保存至 {path}")


def load_model(filename: str = None) -> dict:
    """从磁盘加载模型"""
    if filename is None:
        # 加载最新的模型
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        files = sorted([f for f in os.listdir(MODEL_OUTPUT_DIR) if f.endswith(".pkl")])
        if not files:
            print("[模型] 未找到已保存的模型")
            return None
        filename = files[-1]

    path = os.path.join(MODEL_OUTPUT_DIR, filename)
    if not os.path.exists(path):
        print(f"[模型] 模型文件不存在: {path}")
        return None

    with open(path, "rb") as f:
        model_info = pickle.load(f)
    print(f"[模型] 已加载: {path}")
    return model_info
