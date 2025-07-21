# data.py 

import pandas as pd


def load_market_data(path: str = "data/rb_20250606_primary.csv", n: int = 5):
    """
    读取并预处理原始 A 股行情数据。

    参数
    ----------
    path : str, 可选
        CSV 文件的本地路径，默认为 ``"data/rb_20250606_primary.csv"``。
    n: int 可选
        预测 n 秒之后的收益率

    返回
    ----------
    pandas.DataFrame
        处理后的行情特征表，包含基础字段以及 ``target``（20 日远期收益）列。
    """
    # TODO: 这里暂时读取本地文件

    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df = df["close"].resample("1s").ohlc()
    df["target"] = df["close"].pct_change(periods=-n)
    return df
