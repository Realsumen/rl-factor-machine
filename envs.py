# env.py
from combination import AlphaCombinationModel
from tokenizer import AlphaTokenizer
from typing import List


class AlphaGenerationEnv:
    """
    自定义强化学习环境：生成逆波兰表达式（RPN）的序列决策过程。

    - **状态**：当前已生成的 token ID 列表
    - **动作**：下一个 token 的 ID
    - **奖励**：组合模型评估的单因子 IC
    - **终止条件**：生成 `[SEP]` 或达到 `max_len`
    """

    def __init__(
        self, combo_model: AlphaCombinationModel, tokenizer: AlphaTokenizer, max_len=20
    ):
        """
        初始化环境。

        参数
        ----------
        combo_model : AlphaCombinationModel
            负责因子池管理与 IC 计算的组合模型。
        tokenizer : AlphaTokenizer
            RPN ↔ token 序列转换器。
        max_len : int, 可选
            生成序列的最⼤长度（含 `[BOS]` 与 `[SEP]`），默认 20。
        """
        self.combo_model = combo_model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.reset()

    def reset(self):
        """
        重新开始一条新序列。

        返回
        ----------
        List[int]
            仅包含 `[BOS]` 的初始序列。
        """
        self.sequence = [self.tokenizer.bos_token_id]
        self.done = False
        return self._get_obs()

    def step(self, action: int):
        """
        执行一步生成并返回环境转移结果。

        参数
        ----------
        action : int
            选定的 token ID。

        返回
        ----------
        obs : List[int]
            新的 token 序列。
        reward : float
            若已结束则为该表达式的单因子 IC，否则为 0。
        done : bool
            是否到达终止状态。
        info : dict
            预留调试信息，当前为空字典。
        """
        self.sequence.append(action)
        token = self.tokenizer.decode([action])
        reward = 0.0
        if token == self.tokenizer.sep_token or len(self.sequence) >= self.max_len:
            expr = self.tokenizer.decode(self.sequence[1:-1])
            reward = self.combo_model.evaluate_alpha(expr)
            self.done = True
        obs = self._get_obs()
        return obs, reward, self.done, {}

    def _get_obs(self):
        """
        获取当前观测值（token 序列）。

        返回
        ----------
        List[int]
            当前生成序列（含 `[BOS]`，可能含 `[SEP]`）。
        """
        return self.sequence

    def valid_actions(self) -> List[int]:
        """
        计算当前状态下的合法动作集合，用于 Invalid-Action-Mask。

        规则（基于逆波兰表达式 RPN）
        ------------------------------------------------------------
        1. 维护一个“栈深”计数：
           • 操作数（基础字段或常量）   → 栈深 +1
           • 算子（arity = n）         → 栈深 −(n-1)
        2. 任何前缀片段都必须满足栈深 ≥ 0（否则语法非法）。
        3. 仅当栈深 == 1 时才允许收尾 `[SEP]`。
        4. 生成过程中禁止再次选择 `[BOS]`、`[PAD]`。
        5. 若已接近最大长度，只允许 `[SEP]`。
        """
        # ---------------- 终止态快速返回 ----------------
        if self.done:
            return []  # Episode 已结束

        # ---------- 1. 当前栈深 -------------------------------------------------
        #    跳过首个 [BOS]；若最后已是 [SEP]，说明逻辑上应已 done，但稳妥起见再校验
        sep_str = self.tokenizer.id_to_token[self.tokenizer.sep_token_id]
        tokens = [self.tokenizer.id_to_token[t] for t in self.sequence[1:]]
        if tokens and tokens[-1] == sep_str:
            return []

        #   构建 「算子 → arity」 映射（含四则运算）
        import inspect, operators

        op_arity = {"+": 2, "-": 2, "*": 2, "/": 2}
        for name, fn in inspect.getmembers(operators, inspect.isfunction):
            op_arity[name] = len(inspect.signature(fn).parameters)

        def _delta(tok: str) -> int:
            """token 对栈深的增量"""
            if tok in op_arity:  # 算子
                return 1 - op_arity[tok]
            if tok in self.tokenizer.special_tokens:  # [BOS]/[SEP]/[PAD]
                return 0
            return 1  # 操作数

        for tk in tokens:
            stack_depth += _delta(tk)

         # ---------- 2. 若长度已至上限-1，则只能收尾 ------------------------------
        if len(self.sequence) >= self.max_len - 1:
            return [self.tokenizer.sep_token_id]

        # ---------- 3. 枚举所有 token，筛选合法动作 -----------------------------
        valid: List[int] = []
        remaining_steps = self.max_len - len(self.sequence) - 1  # 预留结尾 [SEP]

        for tid in range(self.tokenizer.vocab_size):
            tok = self.tokenizer.id_to_token[tid]

            # 跳过无意义或禁止的特殊标记
            if tok in ("[BOS]", "[PAD]"):
                continue

            # ---- A. 结束符 `[SEP]` ----
            if tok == "[SEP]":
                if stack_depth == 1:               # 仅当栈深恰好为 1 可收尾
                    valid.append(tid)
                continue

            # ---- B. 普通 token ----
            nd = stack_depth + _delta(tok)
            # 合法性 1：中途栈深不得为负
            if nd < 0:
                continue
            # 合法性 2：后续仍需有可能在剩余步数内归结到栈深 1
            #   最悲观情形：后面全用二元算子，每步栈深 -1，再加结尾 [SEP]
            #   所需最少步 = (nd - 1)   （降到 1） + 1（收尾）
            if nd - 1 + 1 > remaining_steps:
                continue

            valid.append(tid)

        return valid
