# backtest_for_factor_0715_sz.py
import pandas as pd
import numpy as np
import pymysql
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import logging
from tqdm import tqdm
from typing import List
import pathlib
import os
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # Windows系统

# 配置
symbol = "ma"  # 品种代码
start_date = "2025-04-01"  # 开始日期
# end_date = "2025-07-10"  # 结束日期
end_date = "2025-07-24"  # 结束日期

window = 40


def get_symbol_info(symbol: str, field: str):
    """
    从数据库的 SymbolInfo 表中获取指定合约的某个字段值。

    Parameters
    ----------
    symbol : str
        品种代码，例如 'jm'。
    field : str
        要查询的字段名，如 'ExchangeCode', 'TickSize', 'Multiplier'。

    Returns
    -------
    Any
        指定字段对应的值。

    Raises
    ------
    ValueError
        如果查询不到对应的记录。
    pymysql.MySQLError
        数据库操作出错时抛出。
    """
    try:
        conn = pymysql.connect(
            host="192.168.1.118",
            user="replay1",
            password="replayMvt*",
            db="mvtdb",
            connect_timeout=5,
        )

        with conn.cursor() as cursor:
            sql = f"SELECT {field} FROM SymbolInfo WHERE Symbol = %s"
            cursor.execute(sql, (symbol,))
            result = cursor.fetchone()

            if result is None:
                raise ValueError(f"No result found for symbol: {symbol}")

            return result[0]

    except pymysql.MySQLError as e:
        print("数据库错误：", e)
    finally:
        try:
            conn.close()
        except:
            pass


def get_tick_files(
    symbol: str, start_date: str, end_date: str, tradetype: str = "Primary"
):
    """
    根据指定品种和日期范围，从本地文件系统和数据库查询匹配的 tick 数据文件路径。

    Parameters
    ----------
    symbol : str
        品种代码，如 'jm'。
    start_date : str
        开始日期，格式 'YYYY-MM-DD'.
    end_date : str
        结束日期，格式 'YYYY-MM-DD'.
    tradetype : str, optional
        交易类型，默认为 'Primary'.

    Returns
    -------
    tuple[list[pathlib.Path], list, list[str]]
        target_files : 匹配到的 CSV 文件路径列表。
        period_data : 查询到的交易时段记录。
        trading_days : 按升序排序的交易日列表（YYYYMMDD 格式）。
    """
    db_config = {
        "user": "replay1",
        "password": "replayMvt*",
        "host": "192.168.1.118",
        "port": 3306,
        "database": "mvtdb",
        "charset": "utf8mb4",
    }

    # SQL 查询
    sql_instruments = f"""
        SELECT TradingDay, InstrumentID 
        FROM TradeableInstrument 
        WHERE Symbol = '{symbol}' 
          AND TradingDay BETWEEN '{start_date}' AND '{end_date}' 
          AND TradeableType = '{tradetype}'
    """

    sql_trading_period = f"""
        SELECT * FROM SymbolTradingPeriod WHERE symbol = '{symbol}'
    """

    target_files = []

    foldername_dict = {
        "SHFE": "microvast-zx_920",
        "DCE": "microvast-gtja_950",
        "CZCE": "microvast-zx_960",
        "INE": "microvast-zx_920",
        "GFEX": "microvast-zx_901",
    }

    ExchangeCode = get_symbol_info(symbol, field="ExchangeCode")

    folder_name = foldername_dict[ExchangeCode]

    trading_days = []

    try:
        connection = pymysql.connect(**db_config)
        with connection.cursor() as cursor:
            # 查询主力合约
            cursor.execute(sql_instruments)
            results = cursor.fetchall()

            # 查询交易时段
            cursor.execute(sql_trading_period)
            period_data = cursor.fetchall()

            # 遍历查询结果，查找 tick 文件
            for row in results:
                trading_day = row[0].strftime("%Y%m%d")  # 格式化为 YYYYMMDD
                trading_days.append(trading_day)
                instrument_id = row[1]  # 合约代码（如 '2509'）

                PATH = pathlib.Path(
                    *[
                        "T:",
                        folder_name,
                        "Future",
                        ExchangeCode,
                        symbol,
                        trading_day,
                        f"{instrument_id}_{trading_day}.csv",
                    ]
                )

                if os.path.exists(PATH):
                    target_files.append(PATH)

    except pymysql.MySQLError as e:
        print("数据库查询错误:", e)
    finally:
        if connection:
            connection.close()

    return target_files, period_data, sorted(trading_days)


def load_and_preprocess_parallel(
    tick_files: List[str], n_jobs: int = -1
) -> pd.DataFrame:
    """
    并行加载并合并多日 tick CSV 文件。

    Parameters
    ----------
    tick_files : list of str
        需要加载的文件路径列表。
    n_jobs : int, optional
        并行作业数，-1 表示使用所有 CPU 核心。

    Returns
    -------
    pd.DataFrame
        将所有文件内容合并后的 DataFrame，包含原始列和解析后的 timestamp。
    """

    def _load_single_tick_file(file: str) -> pd.DataFrame:
        df = pd.read_csv(file)
        if "timestamp" not in df.columns:
            columns = [
                "timestamp",
                "instrumentID",
                "exchangeID",
                "last",
                "iopv",
                "bid1",
                "bid2",
                "bid3",
                "bid4",
                "bid5",
                "ask1",
                "ask2",
                "ask3",
                "ask4",
                "ask5",
                "bidSize1",
                "bidSize2",
                "bidSize3",
                "bidSize4",
                "bidSize5",
                "askSize1",
                "askSize2",
                "askSize3",
                "askSize4",
                "askSize5",
                "volume",
                "amount",
                "openInterest",
                "updateTime",
                "tradingPhaseCode",
                "indicativeAuctionPrice",
                "indexPrice",
                "epochTime",
            ]
            df.columns = columns
        df["timestamp"] = pd.to_datetime(df.timestamp)
        return df

    dfs = Parallel(n_jobs=n_jobs)(
        delayed(_load_single_tick_file)(file)
        for file in tqdm(tick_files, desc="并行加载")
    )

    df = pd.concat(dfs, ignore_index=True)
    return df


def assign_segment(
    df: pd.DataFrame, time_period: pd.DataFrame, xcol: str = "timestamp"
) -> pd.Series:
    """
    根据时间戳映射到上午盘、下午盘或夜盘。

    Parameters
    ----------
    df : pd.DataFrame
        包含时间戳列的 DataFrame。
    time_period : pd.DataFrame
        来自 SymbolTradingPeriod 表的交易时段信息。
    xcol : str, optional
        时间戳列名称，默认 'timestamp'.

    Returns
    -------
    pd.Series
        与 df 对齐的 segment 标签序列，可为 'morning', 'afternoon', 'night' 或 None。
    """
    # 1. 把 timestamp 转成 hhmmss 整数
    t = df[xcol].dt.hour * 10000 + df[xcol].dt.minute * 100 + df[xcol].dt.second

    # 2️. 上午盘 & 下午盘 的掩码
    mask_am = ((t > 85900) & (t < 101500)) | ((t > 102900) & (t < 113000))
    mask_pm = (t > 132900) & (t < 150000)

    # 3. 根据 time_period["end_time"] 决定夜盘逻辑
    end_times = set(time_period["end_time"].tolist())
    if 23000 in end_times:
        mask_night = (t > 205900) | (t < 23000)
    elif 10000 in end_times:
        mask_night = (t > 205900) | (t < 10000)
    elif 240000 in end_times:
        mask_night = t > 205900
    elif 233000 in end_times:
        mask_night = (t > 205900) & (t < 233000)
    elif 230000 in end_times:
        mask_night = (t > 205900) & (t < 230000)
    else:
        mask_night = pd.Series(False, index=df.index)

    # 4. 用 np.select 一次性贴标签
    df["segment"] = np.select(
        [mask_am, mask_pm, mask_night], ["morning", "afternoon", "night"], default=None
    )
    return df["segment"]


def align_segment(df: pd.DataFrame, trading_days: List[str]) -> pd.Series:
    """
    将夜盘交易的跨日数据映射到对应的交易日。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含 'timestamp' 和 'segment' 列的 DataFrame。
    trading_days : list[str]
        按日期升序排列的交易日列表（YYYYMMDD）。

    Returns
    -------
    pd.Series
        与 df 行数一致的交易日映射（YYYYMMDD 字符串）。
    """

    trading_days_arr = (
        pd.to_datetime(trading_days).normalize().values.astype("datetime64[D]")
    )

    ts = df["timestamp"]
    dates = ts.dt.normalize().values.astype("datetime64[D]")
    minutes = ts.dt.hour.to_numpy(dtype="int32") * 60 + ts.dt.minute.to_numpy(
        dtype="int32"
    )
    is_night = df["segment"].eq("night").to_numpy()

    idx_right = np.searchsorted(trading_days_arr, dates, side="right")
    next_td = trading_days_arr[np.minimum(idx_right, len(trading_days_arr) - 1)]

    # ---- 规则 1：夜盘 & >= 20:30 —— 总是映射到 next_td ----
    mask_late = is_night & (minutes >= 20 * 60 + 30)

    # ---- 规则 2：夜盘 & <= 02:30 ----
    mask_early = is_night & (minutes <= 2 * 60 + 30)
    idx_left = np.searchsorted(trading_days_arr, dates, side="left")
    match_curr = (idx_left < len(trading_days_arr)) & (
        trading_days_arr[idx_left] == dates
    )  # 日期存在于 trading_days
    mapped_early = np.where(
        match_curr, trading_days_arr[idx_left], next_td
    )  # 存在→当天，否则→next_td

    # ---- 结果合并 ----
    mapped = dates.copy()  # 默认就是当天（含日盘、非夜盘行）
    mapped[mask_late] = next_td[mask_late]  # 规则 1
    mapped[mask_early] = mapped_early[mask_early]  # 规则 2

    return pd.to_datetime(mapped).strftime("%Y%m%d")


def calculate_future_returns(df: pd.DataFrame, M: int) -> pd.DataFrame:
    """
    计算未来 M 跳的收益率。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含 'trading_day_mapped', 'segment', 'timestamp', 'mid_price' 列。
    M : int
        lookahead 步数。

    Returns
    -------
    pd.DataFrame
        在原 DataFrame 基础上添加 'future_price' 和 'future_return'，并剔除无法计算的行。
    """
    df1 = df.copy()
    df1 = df1.sort_values(["trading_day_mapped", "segment", "timestamp"])
    df1["future_price"] = df1.groupby(["segment", "trading_day_mapped"])[
        "mid_price"
    ].shift(-M)
    df1["future_return"] = ((df1["future_price"] / df1["mid_price"]) - 1.0) * 10000
    df1 = df1.dropna(subset=["future_price"])
    return df1


def get_R_score(factor: pd.Series, label: pd.Series) -> float:
    if len(factor) == 0:
        return np.nan

    X = factor.values.reshape(-1, 1)
    y = label

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    return model.score(X, y)


def evaluate_factor_by_segment(factor: pd.Series, df: pd.DataFrame) -> dict:
    """
    按不同交易时段(segment)评估因子对未来收益的解释能力。

    Parameters
    ----------
    factor : pd.Series
        根据原始 df 生成的因子序列，索引与 df 对齐。
    df : pd.DataFrame
        包含 'future_return', 'segment', 'is_limit' 列的 DataFrame。

    Returns
    -------
    dict
        键为 'all' 或具体时段标签，值为对应的 R²。
    """
    # 1) 合并
    data = pd.concat(
        [factor.rename("factor"), df[["future_return", "segment", "is_limit"]]], axis=1
    )
    # 2) 丢掉 NaN 并排除涨跌停
    data = data.dropna().loc[~data["is_limit"]]

    results = {}
    # —— 全时段 R² ——
    if len(data) >= 2:
        X_all = data["factor"].values.reshape(-1, 1)
        y_all = data["future_return"].values
        results["all"] = LinearRegression().fit(X_all, y_all).score(X_all, y_all)
    else:
        # 样本太少，直接全返回空
        return {}

    # —— 分段 R² ——
    for seg, grp in data.groupby("segment"):
        if len(grp) < 2:
            results[seg] = np.nan
            continue

        X = grp["factor"].values.reshape(-1, 1)
        y = grp["future_return"].values
        results[seg] = LinearRegression().fit(X, y).score(X, y)

    return results


def example_price_oi_divergence(df: pd.DataFrame, periods: int) -> pd.Series:
    """
    示例因子：价格与持仓量背离信号。

    Parameters
    ----------
    df : pd.DataFrame
        包含 'mid_price' 和 'openInterest' 的原始市场数据。
    periods : int
        计算差分的步长。

    Returns
    -------
    pd.Series
        取值 -1, 0, 1，代表不同的背离信号。
    """
    g = df.groupby(["trading_day_mapped", "segment"])
    price_diff = g["mid_price"].diff(periods)
    price_denominator = g["mid_price"].shift(periods)  # 修复：分母也分组计算
    price_change = price_diff * 10000 / price_denominator

    oi_change = g["openInterest"].diff(periods)

    signal = np.zeros(len(df))
    signal[(price_change > 0) & (oi_change < 0)] = -1
    signal[(price_change < 0) & (oi_change < 0)] = 1

    return pd.Series(signal, index=df.index, name="signal")


def example_calculate_momentum(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    示例因子：基于分组的动量信号。

    Parameters
    ----------
    df : pd.DataFrame
        包含 'mid_price' 列和分组键的原始数据。
    window : int, optional
        计算动量的时间差窗口，默认为 50。

    Returns
    -------
    pd.Series
        重命名为 'momentum' 的动量值序列。
    """
    g = df.groupby(["trading_day_mapped", "segment"])
    return g["mid_price"].diff(window).div(df["mid_price"]).rename("momentum")


def evaluate_factor(
    df: pd.DataFrame, factor_func: callable, verbose: bool = True, *args, **kwargs
) -> dict:
    """
    计算因子并打印各交易时段的 R²（决定系数）。

    Parameters
    ----------
    df : pd.DataFrame
        包含市场行情数据和标签列的数据框，需包含用于计算因子的所有原始字段。

    factor_func : callable
        用于计算因子的函数，应接受 df 和可选参数，返回一个 pd.Series。
        其签名应为：factor_func(df, *args, **kwargs) -> pd.Series

    verbose : bool, optional
        是否打印每个交易时段对应的 R²，默认为 True。

    *args, **kwargs :
        传递给 factor_func 的附加参数。

    Returns
    -------
    dict
        各交易时间段（如 '夜盘'、'上午盘' 等）对应的 R² 值。
        键为时段名，值为 float（或 np.nan）。

    Notes
    -----
    - 该函数首先调用 factor_func 生成因子 Series；
    - 然后使用 `.align(df)` 将因子与原始数据对齐（按索引交集）；
    - 接着对每个交易时段计算该因子与标签的回归拟合优度 R²；
    - 最终结果以 logging 输出（若 verbose=True），并返回字典形式结果。
    """
    # 计算因子值（Series），并与 df 对齐
    factor = factor_func(df, *args, **kwargs)
    aligned_factor, aligned_df = factor.align(df, join="inner")

    # 分时段计算 R²
    r_squared = evaluate_factor_by_segment(aligned_factor, aligned_df)

    # 打印结果（可选）
    if verbose:
        formatted = {seg: f"{val:.6f}" for seg, val in r_squared.items()}
        logging.info(f"{factor_func.__name__} 的 R²: {formatted}")

    return r_squared


def factor_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
    factor_func,
    tradetype: str = "Primary",
    verbose: bool = False,
    *args,
    **kwargs,
) -> dict:
    """
    一站式因子分析接口：
    - 拉取 tick 文件、加载预处理、贴标签、计算收益
    - 生成因子、对齐数据、分段回归评估 R²

    Parameters
    ----------
    symbol : str
        品种代码
    start_date : str
        开始日期，'YYYY-MM-DD'
    end_date : str
        结束日期，'YYYY-MM-DD'
    factor_func : callable
        因子函数，签名为 func(df, *args, **kwargs) -> pd.Series
    tradetype : str
        交易类型，默认 'Primary'
    *args, **kwargs :
        传给因子函数的参数

    Returns
    -------
    dict
        evaluate_factor_by_segment 的输出，各时段 R²
    """
    files, periods, days = get_tick_files(symbol, start_date, end_date, tradetype)
    df = load_and_preprocess_parallel(files)
    period_df = pd.DataFrame(
        periods, columns=["id", "symbol", "segment", "start_time", "end_time"]
    )
    df["segment"] = assign_segment(df, period_df)
    df["trading_day_mapped"] = align_segment(df, days)
    df = df.dropna(subset=["segment"])

    df["mid_price"] = (df["bid1"] + df["ask1"]) / 2.0
    df["is_limit"] = (
        df["bid1"].isna() | df["ask1"].isna() | (df["bid1"] == 0) | (df["ask1"] == 0)
    )
    df["mid_price"] = np.where(
        df["is_limit"],
        df["bid1"].fillna(0) + df["ask1"].fillna(0),
        (df["bid1"] + df["ask1"]) / 2.0,
    )

    df = calculate_future_returns(df, window)
    factor = factor_func(df, *args, **kwargs)
    aligned_factor, aligned_df = factor.align(df, join="inner")
    return evaluate_factor_by_segment(aligned_factor, aligned_df, verbose=verbose)

def secret_factor(df: pd.DataFrame):
    from operators import signed_log
    df["factor"] = signed_log(df.askSize1 /  (df.bidSize1 + 20))
    return df["factor"]



if __name__ == "__main__":

    tick_files, period_data, trading_days = get_tick_files(symbol, start_date, end_date)

    df = load_and_preprocess_parallel(tick_files)

    period_data = pd.DataFrame(
        period_data, columns=["id", "symbol", "segment", "start_time", "end_time"]
    )

    df["segment"] = assign_segment(df, period_data)

    df["trading_day_mapped"] = align_segment(df, trading_days)

    df = df.loc[~df.segment.isna()]

    logging.info(
        "================================数据准备完毕============================================"
    )

    df["mid_price"] = (df["bid1"] + df["ask1"]) / 2.0

    # 涨跌停判断（即 bid1 或 ask1 有缺失或为0）
    df["is_limit"] = (
        df["bid1"].isna() | df["ask1"].isna() | (df["bid1"] == 0) | (df["ask1"] == 0)
    )

    # 计算 mid_price：如果一边无效，则使用另一边；两边都无效则返回 NaN；否则取平均
    df["mid_price"] = np.where(
        df["is_limit"],
        df["bid1"].fillna(0) + df["ask1"].fillna(0),
        (df["bid1"] + df["ask1"]) / 2.0,
    )

    logging.info(
        "==============================以下是因子计算参数设置======================================="
    )
    df = calculate_future_returns(df, window)

    print(df.columns)
    r_squared1 = evaluate_factor(df, example_price_oi_divergence, periods=15)
    r_squared2 = evaluate_factor(df, example_calculate_momentum, window=50)
    r_squared2 = evaluate_factor(df, secret_factor)
