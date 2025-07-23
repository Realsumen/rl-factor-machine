# utility.py

import numpy as np
import pandas as pd
import os
import random
import torch


def zscore_normalize(alpha: np.ndarray) -> pd.Series:
    """
    Z-score 标准化处理：消除因子的尺度影响
    """
    a = np.asarray(alpha, dtype=np.float64)
    # 用 nanmean/nanstd 自动跳过 NaN
    mean = np.nanmean(a)
    std = np.nanstd(a)
    # 如果全是 NaN 或者方差为 0
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros_like(a))
    return pd.Series((a - mean) / std)

def winsorize(alpha: np.ndarray, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.Series:
    """
    异常值截断处理：防止 outlier 扰动因子稳定性
    """
    a = np.asarray(alpha, dtype=float)
    lower = np.nanpercentile(a, lower_quantile * 100)
    upper = np.nanpercentile(a, upper_quantile * 100)
    clipped = np.clip(a, lower, upper)
    return pd.Series(clipped)

def information_coefficient(factor: np.ndarray, target: np.ndarray) -> float:
    """
    信息系数（IC）：衡量因子预测未来收益的能力，Pearson 相关系数
    """
    if len(factor) != len(target):
        raise ValueError("Factor and target length mismatch")
    
    # 1. 同时有效（非 NaN/Inf）的掩码
    mask = np.isfinite(factor) & np.isfinite(target)
    if mask.sum() < 2:               # 样本太少无法算相关
        return 0.0
    
    f, t = factor[mask], target[mask]

    # 2. σ=0 说明缺乏波动，相关性定义无意义，直接返回 0
    if np.std(f) == 0 or np.std(t) == 0:
        return 0.0

    return float(np.corrcoef(f, t)[0, 1])

def set_random_seed(SEED: int = 42):
    """
    固定全局随机数种子，确保实验可复现。
    参数：
        seed (int): 随机数种子，默认 42。
    """
    # 1. 系统级别
    os.environ["PYTHONHASHSEED"] = str(SEED)

    # 2. Python 自带的 random
    random.seed(SEED)

    # 3. NumPy
    np.random.seed(SEED)

    # 4. PyTorch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # 以下两行可保证 cuDNN 的确定性（可能会略微牺牲一点速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False