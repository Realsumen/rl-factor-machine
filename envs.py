# env.py
from combination import AlphaCombinationModel
from typing import List

class AlphaGenerationEnv:
    """
    自定义强化学习环境，无需依赖 OpenAI Gym。
    State: token ID 序列
    Action: 下一个 token ID
    Reward: 表达式结束时组合模型返回的 IC
    """
    def __init__(self, combo_model, tokenizer, max_len=20):
        self.combo_model = combo_model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.reset()

    def reset(self):
        self.sequence = [self.tokenizer.bos_token_id]
        self.done = False
        return self._get_obs()

    def step(self, action: int):
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
        # 返回 list[int]；PPO 里再转 tensor
        return self.sequence
    
    def valid_actions(self) -> List[int]:
        """
        返回当前状态下合法 token_id 的列表，用于 Invalid-Action-Mask。
        简化实现：只防止过长 & 必须以 sep 终止；更细规则见 Appendix C。
        """
        if self.done:
            return []                      # episode 已结束
        if len(self.sequence) >= self.max_len - 1:
            return [self.tokenizer.sep_token_id]  # 只能收尾
        # 否则全部 token 都可选
        return list(range(self.tokenizer.vocab_size))
    

