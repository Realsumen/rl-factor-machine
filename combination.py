# combination.py
import numpy as np
from utility import zscore_normalize, winsorize, information_coefficient
from sklearn.linear_model import ElasticNet
from scipy.optimize import minimize
from typing import List, Dict, Union, Tuple, Callable
import operators
import pandas as pd


class AlphaCombinationModel:
    """
    因子线性组合管理器，实现论文中算法1的核心思想。

    功能:
      - 维护一个最多包含 `max_pool_size` 条归一化因子序列的因子池。
      - 每当有新因子生成时，执行截尾（Winsorize）和 Z-score 标准化，并计算该因子的单因子 IC（Information Coefficient）。
      - 将新因子加入因子池后，通过凸优化（SLSQP）求解最优线性权重，以最大化组合 IC。
      - 若因子池超过上限，则剔除对组合贡献度最小的因子。
      - 在多次计算中对标准化序列、IC 值、最优权重等中间结果进行缓存，以减少重复开销。

    Attributes:
        max_pool_size (int): 因子池最大容量。超过后将删除贡献最小的因子。
        alphas (List[np.ndarray]): 原始因子序列列表。
        norm_alphas (List[np.ndarray]): 归一化后的因子序列列表。
        ic_list (List[float]): 对应每条因子的单因子 IC 列表。
        weights (List[float]): 当前组合中各因子的线性权重。
        expr_list (List[str]): 保存每条因子对应的 RPN 表达式。
        _cache (Dict): 缓存用于存储中间计算结果。
        data (pd.DataFrame): 注入的行情数据。
        _target (np.ndarray): 注入的目标列（未来收益）数组。
    """

    def __init__(
        self,
        max_pool_size: int = 50,
        combiner: Union[str, Callable] = "lsqp",
        combiner_args: Tuple = (),
        combiner_kwargs: Dict = None,
    ):
        """
        初始化 AlphaCombinationModel。

        Args:
            max_pool_size (int): 因子池的最大容量，上限内优先保留贡献度高的因子。
        """
        self.max_pool_size = max_pool_size
        self.combiner = combiner
        self.combiner_args = combiner_args
        self.combiner_kwargs = combiner_kwargs or {}

        self.alphas = []  # 原始因子序列列表
        self.norm_alphas = []  # 归一化后的因子序列列表
        self.ic_list = []  # 对应的单因子 IC 值列表
        self.weights = []  # 当前组合的线性权重列表
        self._cache = {}  # 缓存字典，用于存储中间结果
        self.expr_list = []  # 新增：对应每条因子的 RPN 表达式字符串

    def inject_data(self, df: pd.DataFrame, target_col: str) -> None:
        """
        注入市场行情数据和目标序列，用于 IC 计算与权重优化。

        Args:
            df (pd.DataFrame): 行情特征表，包含基础字段和目标列。
            target_col (str): DataFrame 中代表未来收益的列名，用于 IC 计算。

        Raises:
            ValueError: 当 target_col 不在 df 列时抛出。
        """
        # TODO: 需要更细致的训练集 / 验证集 分割逻辑，添加新的数据集作为验证集 etc
        self.data: pd.DataFrame = df
        self._target = df[target_col].values.astype(np.float64)

    def add_alpha_expr(self, expr: str) -> float:
        """
        根据 RPN 表达式计算新因子，并将其加入因子池。

        Args:
            expr (str): RPN 格式的表达式，例如 "close 5 ts_mean"。

        Returns:
            float: 该因子的单 IC 值，可作为强化学习的 reward。

        Raises:
            ValueError: 当表达式格式错误或运算失败时。
        """
        raw, norm, ic = self._compute_alpha_and_ic(expr)
        self._update_pool(raw, norm, ic, expr)
        return ic

    def _reoptimize_weights(self):
        A = np.vstack(self.norm_alphas).T
        y = self._load_validation_target()

        if isinstance(self.combiner, str):
            if self.combiner == "lsqp":
                w = self._reoptimize_weights_lsqp(
                    A, y, *self.combiner_args, **self.combiner_kwargs
                )
            elif self.combiner == "elastic_net":
                w = self._reoptimize_weights_enet(
                    A, y, *self.combiner_args, **self.combiner_kwargs
                )
            else:
                raise ValueError(f"Unknown combiner: {self.combiner}")
        elif callable(self.combiner):
            w = self.combiner(A, y, *self.combiner_args, **self.combiner_kwargs)
        else:
            raise TypeError("combiner 必须是 str 或者 可调用")

        self.weights = np.asarray(w, dtype=float).ravel().tolist()
        self._cache["weights"] = w.copy()

    def _reoptimize_weights_lsqp(self, A, y, *args, **kwargs):
        """原始 SLSQP + L1 约束"""

        def objective(w):
            combo = A.dot(w)
            ic = np.corrcoef(combo, y)[0, 1]
            return -np.nan_to_num(ic)

        cons = {"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1}
        x0 = np.ones(A.shape[1]) / A.shape[1]
        res = minimize(objective, x0, constraints=cons, method="SLSQP")
        return res.x

    def _reoptimize_weights_enet(self, A, y, *args, **kwargs):
        """ElasticNet 拟合后再归一化 ∑|w|=1"""
        import threading
        print(f"[enet] 开始 fit by {threading.current_thread().name}")
        model = ElasticNet(
            fit_intercept=False,
            random_state=42,
            **kwargs,
        )
        model.fit(A, y)
        print("  [enet] fit 完成", flush=True)
        raise RuntimeError("ddd")
        w = model.coef_
        s = np.sum(np.abs(w))
        print("  [enet] 返回 w=", w, flush=True)
        return (w / s) if s > 0 else w

    def evaluate_alpha(self, expr: str) -> float:
        """只评估 IC，不入池。"""
        _, _, ic = self._compute_alpha_and_ic(expr)
        return ic

    def _compute_alpha_and_ic(
        self, expr: str, in_pool: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        统一计算入口：expr -> raw alpha, norm alpha, ic（全程带缓存）
        """
        # 1) raw
        key_raw = ("raw_alpha", expr)
        if key_raw in self._cache:
            raw = self._cache[key_raw]
        else:
            raw = self._compute_alpha_from_expr(expr)
            raw = np.nan_to_num(raw, nan=0.0)
            if in_pool:
                self._cache[key_raw] = raw

        # 2) norm
        key_norm = ("norm_alpha", expr)
        if key_norm in self._cache:
            norm = self._cache[key_norm]
        else:
            norm = self._maybe_normalize(raw)
            if in_pool:
                self._cache[key_norm] = norm

        # 3) ic
        key_ic = ("expr_ic", expr)
        if key_ic in self._cache:
            ic = self._cache[key_ic]
        else:
            if np.all(np.isnan(raw)):
                ic = -1.0
            else:
                target = self._load_validation_target()
                ic = information_coefficient(norm, target)
            if in_pool:
                self._cache[key_ic] = ic

        return raw, norm, ic

    def _update_pool(
        self, raw: np.ndarray, norm: np.ndarray, ic: float, expr: str
    ) -> None:
        """
        给定已经算好的 raw/norm/ic，把它们塞进因子池并重优化权重。
        """
        # 1. 加入池
        self.alphas.append(raw)
        self.norm_alphas.append(norm)
        self.ic_list.append(ic)
        self.expr_list.append(expr)

        # 2. 超出容量时剔除贡献最小的因子
        if len(self.alphas) > self.max_pool_size:
            contrib = [abs(w * ic_) for w, ic_ in zip(self.weights, self.ic_list)]
            idx = int(np.argmin(contrib))
            for lst in (    
                self.alphas,
                self.norm_alphas,
                self.ic_list,
                self.weights,
                self.expr_list,
            ):
                lst.pop(idx)

        # 3. 最后才更新权重
        if len(self.norm_alphas) == 1:
            self.weights = [1.0]
        else:
            self._reoptimize_weights()

    def score(self) -> float:
        """
        计算当前因子组合在验证集上的加权 IC。

        Returns:
            float: 组合因子的 Pearson IC 值。
        """
        A = np.vstack(self.norm_alphas).T
        combo = A.dot(np.array(self.weights))
        target = self._load_validation_target()
        return information_coefficient(combo, target)

    def _load_validation_target(self) -> np.ndarray:
        """
        获取注入的目标序列数组（未来收益或方向）。

        Returns:
            np.ndarray: 目标序列数值数组。

        Raises:
            AttributeError: 若未调用 `inject_data` 注入数据时。
        """
        if not hasattr(self, "_target"):
            raise AttributeError("请先调用 inject_data() 注入行情和目标序列")
        return self._target

    def _compute_alpha_from_expr(self, expr: str) -> np.ndarray:
        """
        解析逆波兰表达式（RPN），执行算子运算，生成原始因子序列。

        Args:
            expr (str): 形如 "close 5 ts_mean" 的 RPN 表达式字符串。

        Returns:
            np.ndarray: 计算得到的因子值数组，dtype=float64。

        Raises:
            AttributeError: 若未注入 `data` 时调用。
            ValueError: 表达式格式错误（未知 token、参数不足或最终栈深 != 1）。
        """
        if not hasattr(self, "data"):
            raise AttributeError(
                "AlphaCombinationModel 需先注入行情 DataFrame 到 self.data"
            )

        tokens: List[str] = expr.strip().split()
        stack: List[Union[pd.Series, float]] = []

        func_map: Dict[str, Tuple[Callable, int]] = operators.FUNC_MAP

        for tk in tokens:
            if tk in self.data.columns:
                stack.append(self.data[tk])
            elif _is_float(tk):
                val = float(tk)
                if val.is_integer():
                    stack.append(int(val))
                else:
                    stack.append(float(tk))
            elif tk in func_map:
                fn, arity, _ = func_map[tk]
                if len(stack) < arity:
                    raise ValueError(f"RPN 表达式参数不足：{tk}")
                args = [stack.pop() for _ in range(arity)][
                    ::-1
                ]  # 注意：弹栈顺序需反转以保持原来顺序
                res = fn(*args)
                stack.append(res)
            else:
                raise ValueError(f"未知 token：{tk}")

        if len(stack) != 1:
            raise ValueError(f"RPN 表达式最终栈深度应为 1, 计算时为: {len(stack)}")
        output = stack[0]
        if np.isscalar(output):
            series = pd.Series(output, index=self.data.index)
        elif isinstance(output, pd.Series):
            series = output.reindex(self.data.index)
        else:
            series = pd.Series(output, index=self.data.index)

        return series.values.astype(np.float64)

    def _maybe_normalize(self, alpha: np.ndarray) -> np.ndarray:
        """
        对原始因子序列执行截尾（winsorize）和 Z-score 标准化。

        Args:
            alpha (np.ndarray): 原始因子值数组。

        Returns:
            np.ndarray: 归一化后的因子序列。
        """
        return winsorize(zscore_normalize(alpha))

        # === 1. RPN 解析与执行 ====================================================


def _is_float(str) -> bool:
    """
    判断字符串是否可转换为浮点数。

    Args:
        s (str): 待检测字符串。

    Returns:
        bool: 若能安全转换为 float，则返回 True，否则 False。
    """
    try:
        float(str)
        return True
    except ValueError:
        return False
