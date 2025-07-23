# env.py
from combination import AlphaCombinationModel
from tokenizer import AlphaTokenizer
from typing import List
from rpn_type_sim import RPNTypeSimulator



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
        self.type_sim = RPNTypeSimulator(tokenizer)
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
        reward = 0.0
        if action == self.tokenizer.sep_token_id or len(self.sequence) >= self.max_len:
            expr = self.tokenizer.decode(self.sequence, remove_special_tokens=True)
            try:
                reward = self.combo_model.evaluate_alpha(expr)
            except ValueError as e:
                reward = -1.0
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
        valid = []
        prefix = [self.tokenizer.id_to_token[t] for t in self.sequence[1:]]
        remaining = self.max_len - len(self.sequence) - 1

        for tid in range(self.tokenizer.vocab_size):
            tok = self.tokenizer.id_to_token[tid]
            if tok in ("[BOS]", "[PAD]"):
                continue
            if tok == "[SEP]":
                if self.type_sim.simulate(tuple(prefix)) == (self.type_sim.S,):
                    valid.append(tid)
                continue
            if self.type_sim.is_valid_append(prefix, tok, remaining):
                valid.append(tid)
        return valid

