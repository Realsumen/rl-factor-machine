# data.py 

import pandas as pd


def load_market_data(path: str = "rb_20250606_primary.csv"):
    """
    Load and= preprocess raw A-share market data.
    Returns: DataFrame with features and target 20-day returns.
    """
    # TODO: 这里暂时读取本地文件

    df = pd.read_csv(path)
    return df
