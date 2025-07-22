# data.py 

import pandas as pd
import numpy as np


def load_market_data(path: str = "data/rb_20250606_primary.csv", multiplier: int = 10, n: int = 5):
    """
    读取并预处理原始 A 股行情数据。

    参数
    ----------
    path : str, 可选
        CSV 文件的本地路径，默认为 ``"data/rb_20250606_primary.csv"``。
    multiplier: int
    n: int 可选
        预测 n 秒之后的收益率

    返回
    ----------
    pandas.DataFrame
        处理后的行情特征表，包含基础字段以及 ``target``（20 日远期收益）列。
    """
    # TODO: 这里暂时读取本地文件

    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")

    df = df.sort_index()

    df['d_vol'] = df['volume'].diff()
    df['d_amt'] = df['amount'].diff()
    df['d_oi']  = df['openInterest'].diff()

    df.loc[df['d_vol'] <= 0, ['d_vol', 'd_amt']] = np.nan

    df['trade_price'] = df['d_amt'] / df['d_vol'] / multiplier

    one_sec = df.resample('1s')

    ohlc = pd.DataFrame({
        'open' : one_sec['last'].first(),
        'high' : one_sec['trade_price'].max(),
        'low'  : one_sec['trade_price'].min(),
        'close': one_sec['last'].last(),
        'volume': one_sec['d_vol'].sum(),        # 可选：这一秒真正的成交量
        'amount': one_sec['d_amt'].sum(),
        'openInterest': one_sec['d_oi'].sum(),
    })

    mask = ohlc['high'].isna()
    ohlc.loc[mask, 'high'] = one_sec['last'].max()[mask]
    ohlc.loc[mask, 'low']  = one_sec['last'].min()[mask]

    ohlc['target'] = ohlc['close'].pct_change(periods=-n, fill_method=None)
    return ohlc
