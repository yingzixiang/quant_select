import json
import os
import time
import requests
import akshare as ak

GGT_CACHE = os.path.join(os.path.dirname(__file__), "..", "config", "ggt_codes.json")


def get_total_stocks():
    """获取A股全市场标的（code + name）"""
    df = ak.stock_info_a_code_name()
    return df


def _fetch_ggt_codes_fresh():
    """从东方财富拉取港股通成分股（akshare → 多子域名兜底），失败返回 None"""
    # 方案A: 优先用 akshare 内置函数（自带重试）
    try:
        df = ak.stock_hk_ggt_components_em()
        if df is not None and len(df) > 100:
            codes = [str(c) for c in df["代码"].tolist()]
            if len(codes) > 400:
                return sorted(codes)
    except Exception:
        pass

    # 方案B: 多子域名轮询
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "https://data.eastmoney.com/",
    })
    subdomains = [str(n) for n in [1, 5, 9, 15, 25, 35, 39, 46, 54, 57, 63, 72, 76, 80, 88, 90, 95, 99]]

    for sd in subdomains:
        url = f"https://{sd}.push2.eastmoney.com/api/qt/clist/get"
        ggt_codes = set()
        for page in range(1, 8):
            params = {
                "pn": str(page), "pz": "100", "po": "1", "np": "1",
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                "fltt": "2", "fid": "f12",
                "fs": "b:DLMK0146,b:DLMK0144",
                "fields": "f12",
            }
            try:
                r = session.get(url, params=params, timeout=10)
                items = r.json().get("data", {}).get("diff", [])
                if not items:
                    break
                for item in items:
                    ggt_codes.add(str(item["f12"]))
                time.sleep(0.8)
            except Exception:
                break
        if len(ggt_codes) > 100:
            return sorted(ggt_codes)
    return None


def _load_ggt_codes():
    """加载港股通列表：config/ggt_codes.json 为主要数据源，无过期限制"""
    if os.path.exists(GGT_CACHE):
        with open(GGT_CACHE) as f:
            return set(json.load(f))

    # 兜底：尝试 API 拉取并保存到 config/
    codes = _fetch_ggt_codes_fresh()
    if codes:
        os.makedirs(os.path.dirname(GGT_CACHE), exist_ok=True)
        with open(GGT_CACHE, "w") as f:
            json.dump(codes, f)
        return set(codes)
    return None


def get_hk_stocks(top_n=200, ggt_only=True):
    """获取港股标的，按成交额排序取Top N，可选仅港股通"""
    df = ak.stock_hk_spot()
    df = df[df["成交额"] > 0]

    if ggt_only:
        ggt_codes = _load_ggt_codes()
        if ggt_codes is not None:
            df = df[df["代码"].isin(ggt_codes)]
            print(f"  港股通过滤: 成分股 {len(ggt_codes)} 只 → 池内 {len(df)} 只")
        else:
            print(f"  港股通列表获取失败，使用全港股（{len(df)} 只）")

    df = df.sort_values("成交额", ascending=False)
    df = df.head(top_n)
    return df[["代码", "中文名称"]].rename(columns={"中文名称": "name", "代码": "code"})


def filter_risk_stocks(stock_df):
    """风控过滤：ST、上市不足1年"""
    mask = ~stock_df["name"].str.contains("ST", na=False)
    stock_df = stock_df[mask]
    return stock_df["code"].tolist()
