import pandas as pd
import numpy as np

# -----------------------------
# Time-Series Operators (TS)
# -----------------------------

def ref(series: pd.Series, period: int) -> pd.Series:
    """Ref(x, t)：滞后期 t 的值"""
    return series.shift(period)

def ts_mean(series: pd.Series, window: int) -> pd.Series:
    """Mean(x, t)：滚动均值"""
    return series.rolling(window).mean()

def ts_med(series: pd.Series, window: int) -> pd.Series:
    """Med(x, t)：滚动中位数"""
    return series.rolling(window).median()

def ts_sum(series: pd.Series, window: int) -> pd.Series:
    """Sum(x, t)：滚动求和"""
    return series.rolling(window).sum()

def ts_std(series: pd.Series, window: int) -> pd.Series:
    """Std(x, t)：滚动标准差"""
    return series.rolling(window).std()

def ts_var(series: pd.Series, window: int) -> pd.Series:
    """Var(x, t)：滚动方差"""
    return series.rolling(window).var()

def ts_max(series: pd.Series, window: int) -> pd.Series:
    """Max(x, t)：滚动最大值"""
    return series.rolling(window).max()

def ts_min(series: pd.Series, window: int) -> pd.Series:
    """Min(x, t)：滚动最小值"""
    return series.rolling(window).min()

def ts_mad(series: pd.Series, window: int) -> pd.Series:
    """Mad(x, t)：滚动平均绝对偏差"""
    return series.rolling(window).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )

def ts_delta(series: pd.Series, period: int = 1) -> pd.Series:
    """Delta(x, t)：与 t 期前的差值"""
    return series.diff(period)

def ts_rank(series: pd.Series, window: int) -> pd.Series:
    """滚动排序百分位"""
    def _rank(x):
        return pd.Series(x).rank(pct=True).iloc[-1]
    return series.rolling(window).apply(_rank, raw=True)

def ts_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Corr(x, y, t)：滚动皮尔森相关系数"""
    return x.rolling(window).corr(y)

def ts_cov(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Cov(x, y, t)：滚动协方差"""
    return x.rolling(window).cov(y)

def ts_wma(series: pd.Series, window: int) -> pd.Series:
    """WMA(x, t)：滚动线性加权平均"""
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )

def ts_ema(series: pd.Series, span: int) -> pd.Series:
    """EMA(x, t)：指数移动平均（span 可理解为 t）"""
    return series.ewm(span=span, adjust=False).mean()


# -----------------------------
# Utility Operators
# -----------------------------

def decay_linear(series: pd.Series, window: int) -> pd.Series:
    """线性衰减加权平均（同 WMA）"""
    weights = np.arange(1, window + 1)[::-1]
    return series.rolling(window).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )

def ts_zscore(series: pd.Series, window: int) -> pd.Series:
    """时序标准分"""
    return (series - ts_mean(series, window)) / ts_std(series, window)

def ts_return(series: pd.Series, period: int = 1) -> pd.Series:
    """周期收益率"""
    return series.pct_change(period)


