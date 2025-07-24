# alpha_generation_env.py
from combination import AlphaCombinationModel
from tokenizer import AlphaTokenizer
from typing import List
from operators import FUNC_MAP


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
        self.sequence: List[int] = [self.tokenizer.bos_token_id]
        self.done: bool = False
        self._stack_types: List[str] = []
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

        # —— 记录前一时刻的 info ——
        info = {
            "prev_stack_types": self._stack_types.copy(),
            "prev_depth": len(self._stack_types),
            "prev_valid_actions": self.valid_actions(),
            "action_id": action,
            "action_token": self.tokenizer.id_to_token[action],
        }

        # —— 执行动作：更新 sequence & stack_types ——
        self.sequence.append(action)
        tok = info["action_token"]
        if tok in self.tokenizer.operand_type_map:
            self._stack_types.append(self.tokenizer.operand_type_map[tok])
        elif tok in FUNC_MAP:
            _, arity, _ = FUNC_MAP[tok]
            for _ in range(arity):
                self._stack_types.pop()
            self._stack_types.append("Series")
        else:
            pass

        # —— 再补充依赖于“后”状态的 info ——
        info.update(
            {
                "new_stack_types": self._stack_types.copy(),
                "new_depth": len(self._stack_types),
                "remaining": self.max_len - len(self.sequence),
            }
        )

        # —— 计算 reward & done ——
        reward = 0.0
        if action == self.tokenizer.sep_token_id or len(self.sequence) >= self.max_len:
            expr = self.tokenizer.decode(self.sequence, remove_special_tokens=True)
            try:
                reward = self.combo_model.add_alpha_expr(expr)
            except ValueError as e:
                reward = -1.0
                info["error"] = str(e)
            self.done = True

        obs = self._get_obs()
        return obs, reward, self.done, info

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
        获取当前状态下所有合法动作的 token ID。

        根据当前表达式的栈状态与剩余步数，筛选出能够构成
        合法逆波兰表达式的下一个 token。

        包含如下合法性检查：
        - 栈深是否满足操作符的参数需求；
        - 栈顶参数类型是否与操作符定义匹配；
        - 剩余步数是否足够完成表达式闭合；
        - 对于操作数，保证不会超出最大长度限制。

        返回
        ----------
        List[int]
            所有合法 token 的 ID 列表。
        """
        depth = len(self._stack_types)
        remaining = self.max_len - len(self.sequence)

        if remaining == 1:
            return [self.tokenizer.sep_token_id] if depth == 1 else []

        valid = []
        
        if depth == 1:
            valid.append(self.tokenizer.sep_token_id)

        for tok_id, tok in self.tokenizer.id_to_token.items():
            if tok in ("[PAD]", "[BOS]", "[SEP]"):
                continue

            # 时间序列算子前，只有整型常量才合法
            if tok in FUNC_MAP and tok.startswith("ts_"):
                last_tok = self.tokenizer.id_to_token[self.sequence[-1]]
                last_type = self.tokenizer.operand_type_map.get(last_tok)
                if last_type != "Scalar_INT":
                    continue
                if last_tok == "CONST_1":
                    continue
                if tok == "ts_kurt" and last_tok == "CONST_3":
                    continue

            if tok in FUNC_MAP:
                fn, arity, param_types = FUNC_MAP[tok]
                if depth < arity:
                    continue
                if not all(
                    _type_compatible(g, r)
                    for g, r in zip(self._stack_types[-arity:], param_types)
                ):
                    continue
                if depth - arity + 1 >= remaining:
                    continue
                valid.append(tok_id)
            else:
                # 常量不能连着常量
                if depth > 0 \
                and self.tokenizer.operand_type_map[tok].startswith("Scalar") \
                and self._stack_types[-1].startswith("Scalar"):
                    continue
                if depth + 1 < remaining:
                    valid.append(tok_id)
        return valid


def _type_compatible(given: str, required: str) -> bool:
    if given == required:
        return True
    if required == "Any":
        return True
    if given == "Scalar_INT" and required == "Scalar_FLOAT":
        return True
    return False
