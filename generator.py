from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from envs import AlphaGenerationEnv


class PolicyNetwork(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 1) -> None:
        """
        策略网络，用于生成下一个 token 的概率分布。

        Args:
            vocab_size: token vocabulary 大小，即动作空间大小。
            hidden_dim: LSTM 隐藏层维度。
            num_layers: LSTM 层数。
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_logits = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: 输入 token 序列，形状为 (batch_size, seq_len)。
            hidden: LSTM 的隐藏状态 (h, c)，形状为 (num_layers, batch_size, hidden_dim)。

        Returns:
            logits: 下一个 token 的 logits，形状为 (batch_size, vocab_size)。
            hidden: 更新后的隐藏状态。
        """
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc_logits(out[:, -1, :])
        return logits, hidden

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化隐藏状态为全零。

        Args:
            batch_size: 批量大小。
            device: 所在设备（cpu 或 cuda）。

        Returns:
            初始化的 (h, c) 状态。
        """
        num_layers = self.lstm.num_layers
        hidden_dim = self.lstm.hidden_size
        h0 = torch.zeros(num_layers, batch_size, hidden_dim, device=device)
        c0 = torch.zeros(num_layers, batch_size, hidden_dim, device=device)
        return (h0, c0)


class ValueNetwork(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 1) -> None:
        """
        价值网络，用于估计当前 token 序列的状态价值 V(s)

        Args:
            vocab_size: token vocabulary 大小。
            hidden_dim: LSTM 隐藏层维度。
            num_layers: LSTM 层数。
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_value = nn.Linear(hidden_dim, 1)

    def forward(
        self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: 输入 token 序列，形状为 (batch_size, seq_len)。
            hidden: LSTM 隐藏状态。

        Returns:
            value: 每个序列的状态价值，形状为 (batch_size,)。
            hidden: 更新后的隐藏状态。
        """
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        value = self.fc_value(out[:, -1, :]).squeeze(-1)
        return value, hidden

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        同 PolicyNetwork，初始化 LSTM 隐藏状态。
        """
        num_layers = self.lstm.num_layers
        hidden_dim = self.lstm.hidden_size
        h0 = torch.zeros(num_layers, batch_size, hidden_dim, device=device)
        c0 = torch.zeros(num_layers, batch_size, hidden_dim, device=device)
        return (h0, c0)


class RLAlphaGenerator:
    """
    使用 PPO 在 AlphaGenerationEnv 上训练生成 alpha 表达式的策略。
    """

    def __init__(self, env: AlphaGenerationEnv, config: Dict[str, Any]) -> None:
        """
        Args:
            env: 强化学习环境，需支持 reset(), step(action), valid_actions() 接口。
            config: 包含网络和 PPO 超参数的配置字典。
        """
        self.env = env
        self.vocab_size = config["vocab_size"]
        self.hidden_dim = config.get("hidden_dim", 128)
        self.device = config.get("device", "cpu")

        self.policy_net = PolicyNetwork(self.vocab_size, self.hidden_dim).to(
            self.device
        )
        self.value_net = ValueNetwork(self.vocab_size, self.hidden_dim).to(self.device)

        self.policy_optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=config.get("lr_policy", 3e-4)
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value_net.parameters(), lr=config.get("lr_value", 1e-3)
        )

        self.gamma = config.get("gamma", 1.0)
        self.clip_eps = config.get("clip_eps", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_coef = config.get("value_coef", 0.5)
        self.update_epochs = config.get("update_epochs", 4)
        self.batch_size = config.get("batch_size", 64)
        self.max_seq_len = config.get("max_seq_len", 20)

    def train(self, num_iterations: int) -> None:
        """
        运行 PPO 训练循环，更新策略和值函数。

        Args:
            num_iterations: 训练轮数，每轮会进行一次 trajectory 收集与 PPO 更新。
        """
        ...

    def _collect_trajectories(
        self,
    ) -> Tuple[
        List[torch.Tensor], List[int], List[torch.Tensor], List[float], List[float]
    ]:
        """
        从环境中采样若干条完整轨迹，用于 PPO 更新。

        Returns:
            states: token 序列 (List[Tensor])。
            actions: 选择的动作 token_id (List[int])。
            logps: 对应动作的 log 概率 (List[Tensor])。
            returns: 每个步骤对应的回报值 (List[float])。
            advantages: 每个步骤的 advantage（此处为简化直接用 return）。
        """
        states, actions, logps, rewards, dones, values = [], [], [], [], [], []

        obs = torch.tensor([self.env.reset()], device=self.device)
        h_p, h_v = self.policy_net.init_hidden(1, self.device), self.value_net.init_hidden(1, self.device)

        while len(states) < self.batch_size:
            logits, h_p = self.policy_net(obs, h_p)

            # -------- Invalid-action-mask --------
            mask = torch.tensor([self.env.action_mask()], device=self.device)
            logits[mask == 0] = -1e9
            dist = Categorical(logits=logits)
            
            action = dist.sample()
            logp = dist.log_prob(action)

            value, h_v = self.value_net(action)

            next_obs, reward, done, _ = self.env.step(action.item())
            states.append(obs.squeeze(0))
            actions.append(action)
            logps.append(logp)
            rewards.append(torch.tensor(reward, device=self.device, dtype=torch.float32))
            dones.append(done)
            values.append(value.squeeze(0))

            if done:
                obs = torch.tensor([self.env.reset()], device=self.device)
                h_p, h_v = self.policy_net.init_hidden(1, self.device), self.value_net.init_hidden(1, self.device)
            else:
                obs = torch.tensor([next_obs], device=self.device)
        
        # ===== 计算 GAE(λ=1) → advantage = return - value =====
        returns, advantages, R = [], [], torch.tensor(0.0, device=self.device)
        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            R = r + self.gamma * R * (1.0 - float(d))
            returns.insert(0, R)
            advantages.insert(0, R - v)
        
        return (
            torch.stack(states),
            torch.stack(actions).squeeze(-1),
            torch.stack(logps),
            torch.stack(returns),
            torch.stack(advantages),
        )
    
    