# utility.py

import numpy as np
import pandas as pd

def zscore_normalize(series: pd.Series) -> pd.Series:
    """
    Z-score 标准化处理：消除因子的尺度影响
    """
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros_like(series), index=series.index)
    return (series - mean) / std

def winsorize(series: pd.Series, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.Series:
    """
    异常值截断处理：防止 outlier 扰动因子稳定性
    """
    lower = series.quantile(lower_quantile)
    upper = series.quantile(upper_quantile)
    return series.clip(lower=lower, upper=upper)

def information_coefficient(factor: np.ndarray, target: np.ndarray) -> float:
    """
    信息系数（IC）：衡量因子预测未来收益的能力，Pearson 相关系数
    """
    if len(factor) != len(target):
        raise ValueError("Factor and target length mismatch")
    if np.std(factor) == 0 or np.std(target) == 0:
        return 0.0
    
    ic = np.corrcoef(factor, target)[0, 1]
    return float(np.nan_to_num(ic))