# combination.py
import numpy as np
from utility import zscore_normalize, winsorize, information_coefficient
from scipy.optimize import minimize
from typing import List, Dict, Union, Tuple, Callable
import operators
import pandas as pd
import inspect

class AlphaCombinationModel:
    """
    因子线性组合管理器，实现论文中算法1的核心思想。

    功能：
    1. 维护一个最多包含 max_pool_size 条归一化因子序列的因子池。
    2. 每当有新因子生成时，对其进行截尾与 Z-score 归一化，并计算信息系数（IC）。
    3. 将新因子加入因子池后，使用凸优化求解最优线性权重，以最大化组合 IC。
    4. 如果因子池大小超出上限，则剔除对组合贡献最小的因子。
    5. 对中间计算（归一化序列、IC 值、权重解等）进行缓存，避免重复开销。
    """
    def __init__(self, max_pool_size: int = 50):
        """
        初始化因子组合模型。

        参数：
        - max_pool_size：因子池最大容量，超过后会剔除贡献度最低的因子。
        """
        self.max_pool_size = max_pool_size
        self.alphas = []         # 原始因子序列列表
        self.norm_alphas = []    # 归一化后的因子序列列表
        self.ic_list = []        # 对应的单因子 IC 值列表
        self.weights = []        # 当前组合的线性权重列表
        self._cache = {}         # 缓存字典，用于存储中间结果
        self.expr_list = []      # 新增：对应每条因子的 RPN 表达式字符串

    def inject_data(self, df: pd.DataFrame, target_col) -> None:
        """
        在主程序中加载好行情后，把 df 注入模型，
        target_col 是未来收益列，用于计算 IC。
        """
        self.data: pd.DataFrame = df
        self._target = df[target_col].values.astype(np.float64)

    def update_with(self, new_alpha: np.ndarray, expr: str):
        """
        添加并更新新因子至因子池，同时保存其表达式。

        步骤：
        1. 对 new_alpha 做 winsorize 截尾与 z-score 归一化。
        2. 计算该因子的单因子 IC，并缓存结果。
        3. 将归一化因子与其 IC 加入池中，重新求解最优权重。
        4. 若因子池超限，剔除 |w·IC| 最小的那个因子。

        参数：
        - new_alpha：原始因子值数组（numpy）。
        - expr     ：该因子对应的 RPN 表达式字符串。
        """
        # 归一化
        norm = winsorize(zscore_normalize(new_alpha))
        key = ('ic', tuple(norm))
        if key in self._cache:
            ic = self._cache[key]
        else:
            target = self._load_validation_target()
            ic = information_coefficient(norm, target)
            self._cache[key] = ic

        # 更新因子池
        self.alphas.append(new_alpha)
        self.norm_alphas.append(norm)
        self.ic_list.append(ic)
        self.expr_list.append(expr)          # 记录表达式

        # 若是首因子直接赋权 1；否则重优化权重
        if len(self.norm_alphas) == 1:
            self.weights = [1.0]
        else:
            self._reoptimize_weights()

        # 超限时剔除贡献最小因子
        if len(self.alphas) > self.max_pool_size:
            contrib = [abs(w * ic) for w, ic in zip(self.weights, self.ic_list)]
            idx = contrib.index(min(contrib))
            for lst in (self.alphas, self.norm_alphas, self.ic_list, self.weights):
                lst.pop(idx)

    def add_alpha_expr(self, expr: str) -> float:
        """
        根据表达式计算因子、IC，并把它加入因子池。
        返回该因子的单因子 IC，方便上层当作 reward 使用。
        """
        new_alpha = self._compute_alpha_from_expr(expr)
        ic = self.evaluate_alpha(expr)            # 里面自带缓存
        self.update_with(new_alpha, expr)
        return ic

    def _reoptimize_weights(self):
        """
        求解最优线性权重，使组合 IC 最大。

        实现细节：
        - 目标：最小化负的组合 IC。
        - 约束：权重绝对值之和等于 1（L1 归一化）。
        - 求解器：使用 scipy.optimize.minimize，默认 SLSQP 方法。
        """
        A = np.vstack(self.norm_alphas).T
        target = self._load_validation_target()

        def objective(w):
            combo = A.dot(w)
            ic = np.corrcoef(combo, target)[0, 1]
            return -np.nan_to_num(ic)

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1})
        x0 = np.ones(A.shape[1]) / A.shape[1]
        res = minimize(objective, x0, constraints=cons, method='SLSQP')
        self.weights = res.x.tolist()
        self._cache['weights'] = res.x.copy()

    def evaluate_alpha(self, expr: str) -> float:
        """
        根据给定的表达式（RPN 格式）计算原始因子序列，
        然后做归一化并返回其单因子 IC（带缓存）。

        参数：
        - expr：因子表达式（逆波兰表示法字符串）。

        返回：
        - 该表达式对应因子的 IC 值。
        """
        new_alpha = self._compute_alpha_from_expr(expr)
        norm = self._maybe_normalize(new_alpha)
        key = ('expr_ic', expr)
        if key not in self._cache:
            target = self._load_validation_target()
            self._cache[key] = information_coefficient(norm, target)
        return self._cache[key]

    def score(self) -> float:
        """
        计算并返回当前加权组合在验证集上的信息系数（IC）。

        返回：
        - 组合因子的 IC。
        """
        A = np.vstack(self.norm_alphas).T
        combo = A.dot(np.array(self.weights))
        target = self._load_validation_target()
        return information_coefficient(combo, target)

    def _load_validation_target(self) -> np.ndarray:
        """
        加载验证集的目标序列（如未来收益或方向标签）。
        该方法需由用户实现以接入实际数据。
        """
        if not hasattr(self, "_target"):
            raise AttributeError("请先调用 inject_data() 注入行情和目标序列")
        return self._target

    def _compute_alpha_from_expr(self, expr: str) -> np.ndarray:
        """
        解析 RPN 表达式并返回对应的 numpy.ndarray 因子序列。

        设计约定
        ----------
        • 基础变量：直接写列名，例如 'close'、'volume'；必须存在于 self.data 中  
        • 常量      ：写成数字字符串，如 '5'、'0.3'，自动转 float  
        • 一元/二元/多元算子：对应 operators.py 中的函数名，例如
              "close 5 ts_mean"    # → ts_mean(close, 5)
              "high low - ts_max"  # → ts_max(high - low)
              "open close ts_corr 20" # → ts_corr(open, close, 20)
        • 每个函数的“参数个数” (= arity) 通过反射自动推断，兼容新算子零改动
        """
        if not hasattr(self, "data"):
            raise AttributeError(
                "AlphaCombinationModel 需先注入行情 DataFrame 到 self.data"
            )

        tokens: List[str] = expr.strip().split()
        stack: List[Union[pd.Series, float]] = []

        # 预构建 “函数名 → (Callable, arity)” 映射
        func_map: Dict[str, Tuple[Callable, int]] = {}
        for name, fn in inspect.getmembers(operators, inspect.isfunction):
            sig = inspect.signature(fn)
            func_map[name] = (fn, len(sig.parameters))

        for tk in tokens:
            # -- 1. 基础变量 ----------------------------------------------------
            if tk in self.data.columns:
                stack.append(self.data[tk])
            # -- 2. 数值常量 ----------------------------------------------------
            elif _is_float(tk):
                stack.append(float(tk))
            # -- 3. 函数 / 运算符 ----------------------------------------------
            elif tk in func_map:
                fn, arity = func_map[tk]
                if len(stack) < arity:
                    raise ValueError(f"RPN 表达式参数不足：{tk}")
                # 注意：弹栈顺序需反转以保持原来顺序
                args = [stack.pop() for _ in range(arity)][::-1]
                res = fn(*args)
                stack.append(res)
            # -- 4. 支持最常见的四则运算符 -------------------------------------
            elif tk in {"+", "-", "*", "/"}:
                if len(stack) < 2:
                    raise ValueError(f"RPN 表达式参数不足：{tk}")
                b, a = stack.pop(), stack.pop()
                if tk == "+": res = a + b
                elif tk == "-": res = a - b
                elif tk == "*": res = a * b
                elif tk == "/": res = a / (b.replace(0, np.nan) if isinstance(b, pd.Series) else (b if b != 0 else np.nan))
                stack.append(res)
            else:
                raise ValueError(f"未知 token：{tk}")

        if len(stack) != 1:
            raise ValueError("RPN 表达式最终栈深度应为 1")
        # 返回 numpy 数组，后续会做 winsorize + z-score
        series = stack[0]
        return np.asarray(series.values, dtype=np.float64)
    
    def _maybe_normalize(self, alpha: np.ndarray) -> np.ndarray:
        """
        对原始因子序列执行 winsorize 截尾和 z-score 归一化。
        """
        return winsorize(zscore_normalize(alpha))

        # === 1. RPN 解析与执行 ====================================================


def _is_float(str) -> bool:
    """
    判断字符串 s 是否能被转换成 float。
    返回 True 表示可以, False 表示不行。
    """
    try:
        float(str)
        return True
    except ValueError:
        return False