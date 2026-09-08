import pandas as pd
from config.settings import MIN_LIST_DAYS

def base_stock_filter(stock_df):
    """剔除ST、次新股"""
    stock_df["list_date"] = pd.to_datetime(stock_df["list_date"])
    end_dt = pd.to_datetime(stock_df["list_date"].max())
    # 上市天数过滤
    stock_df = stock_df[(end_dt - stock_df["list_date"]).dt.days > MIN_LIST_DAYS]
    # 剔除ST
    stock_df = stock_df[~stock_df["name"].str.contains("ST")]
    return stock_df