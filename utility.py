# utility.py
import numpy as np
import pandas as pd
import os
import random
import torch


def zscore_normalize(alpha: np.ndarray) -> pd.Series:
    """
    对因子序列进行Z-score标准化。

    该函数对输入数组进行去均值除以标准差处理，
    并将结果限制为有限值，所有NaN或无穷值替换为0。

    Args:
        alpha (np.ndarray): 原始因子值数组，dtype可包含NaN。

    Returns:
        pd.Series: 标准化后的序列，长度与输入相同，无NaN。
    """
    a = np.asarray(alpha, dtype=np.float64)
    mean = np.nanmean(a)
    std = np.nanstd(a)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros_like(a))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (a - mean) / std
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(z)


def winsorize(
    alpha: np.ndarray, lower_quantile: float = 0.01, upper_quantile: float = 0.99
) -> pd.Series:
    """
    对因子序列进行截尾处理，限制极端值。

    使用指定上下分位数计算阈值，并将超出范围的值裁剪到边界，
    所有NaN或无穷值替换为对应边界值。

    Args:
        alpha (np.ndarray): 原始因子值数组。
        lower_quantile (float): 下分位点，默认0.01。
        upper_quantile (float): 上分位点，默认0.99。

    Returns:
        pd.Series: 截尾后的序列，长度与输入相同，无NaN。
    """
    a = np.asarray(alpha, dtype=float)
    with np.errstate(invalid="ignore", over="ignore"):
        lower = np.nanpercentile(a, lower_quantile * 100)
        upper = np.nanpercentile(a, upper_quantile * 100)
        clipped = np.clip(a, lower, upper)
    clipped = np.nan_to_num(clipped, nan=lower, posinf=upper, neginf=lower)
    return pd.Series(clipped)


def information_coefficient(factor: np.ndarray, target: np.ndarray) -> float:
    """
    计算因子与目标的Pearson相关系数（信息系数）。

    若输入长度不匹配或有效样本少于2，返回0；
    若任意方差为0，返回0。

    Args:
        factor (np.ndarray): 因子值数组。
        target (np.ndarray): 目标值数组（未来收益）。

    Returns:
        float: Pearson相关系数，范围[-1,1]，或0表示无效。
    """
    if len(factor) != len(target):
        raise ValueError("Factor and target length mismatch")
    mask = np.isfinite(factor) & np.isfinite(target)
    if mask.sum() < 2:
        return 0.0
    f, t = factor[mask], target[mask]
    if np.std(f) == 0 or np.std(t) == 0:
        return 0.0
    return float(np.corrcoef(f, t)[0, 1])


def set_random_seed(seed: int = 42):
    """
    固定全局随机数种子，确保实验可复现。

    Args:
        seed (int): 随机数种子，默认42。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
