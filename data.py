# data.py
# TODO: 逻辑需要更加细化，
# 1. 过滤不合格的交易时间--针对不同的交易品种 
# 2. 怎么样弹性地满足不同跨度因子的需求，整理成 numpy.array 高效地计算 ic，现在处理方式为 直接concat，很不合理
#   a. 不同的时间窗口; 
#   b. 不同的 symbol;


from typing import List, Dict
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

def load_and_process(file: Path, multiplier: int, n: int) -> pd.DataFrame:
    df = pd.read_parquet(file)
    return process_tick_data(df, multiplier=multiplier, n=n)

def load_symbol_dfs(
    directory: str,
    symbols: Dict[str, int],
    start_date: str,
    end_date: str,
    n_jobs: int = 4,
    n: int = 10
) -> Dict[str, List[pd.DataFrame]]:
    dir_path = Path(directory)
    dt_start = datetime.strptime(start_date, "%Y%m%d").date()
    dt_end   = datetime.strptime(end_date,   "%Y%m%d").date()

    symbol_dfs: Dict[str, List[pd.DataFrame]] = {}

    for sym, mul in symbols.items():
        files = []
        for file in dir_path.glob(f"{sym}_*.parquet"):
            date_str = file.stem.split("_", 1)[1]
            try:
                file_date = datetime.strptime(date_str, "%Y%m%d").date()
            except ValueError:
                continue
            if dt_start <= file_date <= dt_end:
                files.append(file)
        files = sorted(files)

        processed_list = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(load_and_process)(file, mul, n)
            for file in files
        )
        symbol_dfs[sym] = processed_list

    return symbol_dfs

def process_tick_data(df: pd.DataFrame, multiplier: int = 10, n: int = 5):

    df = df.copy()
    df = df.set_index("timestamp").sort_index()

    df["d_vol"] = df["volume"].diff()
    df["d_amt"] = df["amount"].diff()
    df["d_oi"] = df["openInterest"].diff()

    df.loc[df["d_vol"] <= 0, ["d_vol", "d_amt"]] = np.nan

    df["trade_price"] = df["d_amt"] / df["d_vol"] / multiplier

    df['ts_ceil'] = df.index.to_series().dt.ceil('s')

    group = df.groupby('ts_ceil', sort=True)
    ohlc = pd.DataFrame({
        'open':         group['last'].first(),
        'high':         group['trade_price'].max(),
        'low':          group['trade_price'].min(),
        'close':        group['last'].last(),
        'volume':       group['d_vol'].sum(),
        'amount':       group['d_amt'].sum(),
        'openInterest': group['d_oi'].sum(),
    })

    ohlc.index.name = 'timestamp'

    mask = ohlc['high'].isna()
    if mask.any():
        last_max = group['last'].max()
        last_min = group['last'].min()
        ohlc.loc[mask, 'high'] = last_max[mask]
        ohlc.loc[mask, 'low']  = last_min[mask]

    ohlc["target"] = ohlc["close"].pct_change(periods=-n, fill_method=None)
    return ohlc


