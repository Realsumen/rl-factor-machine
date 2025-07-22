from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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
        参数:
            x: 输入 token 序列，形状为 (batch_size, seq_len)。
            hidden: LSTM 隐藏状态。

        返回值:
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
    使用 PPO 在 `AlphaGenerationEnv` 中训练策略网络，自动生成高 IC 的 Alpha 表达式。
    """

    def __init__(self, env: AlphaGenerationEnv, config: Dict[str, Any]) -> None:
        """
        参数:
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
        用 PPO 训练策略 & 价值网络。
        每轮：
          ① 与环境交互，采样 batch_size 步（或更多）完整 episode
          ② 计算优势 (A = R - V) 并标准化
          ③ 对策略 / 价值网络做多次 epoch 更新
        """
        for it in range(1, num_iterations + 1):
            # ------ 采样轨迹 -------------------------------------------------
            states, actions, old_logps, returns, advantages = self._collect_trajectories()

            # Advantage 标准化以稳定训练
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 打包成 DataLoader，方便多轮 epoch shuffle
            ds = TensorDataset(states, actions, old_logps, returns, advantages)
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

            for _ in range(self.update_epochs):
                for s, a, logp_old, ret, adv in loader:
                    s = s.to(self.device)
                    a = a.to(self.device)
                    logp_old = logp_old.to(self.device).detach()
                    ret = ret.to(self.device)
                    adv = adv.to(self.device)

                    # ----- 1. 重新计算策略 & log π(a|s) ---------------------
                    h0_p = self.policy_net.init_hidden(s.size(0), self.device)
                    logits, _ = self.policy_net(s, h0_p)
                    dist = Categorical(logits=logits)
                    logp = dist.log_prob(a)
                    entropy = dist.entropy().mean()

                    # ----- 2. 计算 PPO clip 损失 ---------------------------
                    ratio = (logp - logp_old).exp()
                    pg_loss = -torch.min(
                        ratio * adv,
                        torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv,
                    ).mean()

                    # ----- 3. 计算价值函数损失 -----------------------------
                    h0_v = self.value_net.init_hidden(s.size(0), self.device)
                    value_pred, _ = self.value_net(s, h0_v)
                    value_loss = F.mse_loss(value_pred.squeeze(-1), ret)

                    # ----- 4. 总损失 & 反向传播 ----------------------------
                    loss = pg_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
                    self.policy_optimizer.step()
                    self.value_optimizer.step()

            # ------ 打印监控信息 --------------------------------------------
            if it % 10 == 0:
                with torch.no_grad():
                    avg_ret = returns.mean().item()
                    combo_ic = self.env.combo_model.score()
                print(f"[Iter {it:04d}]  AvgReturn={avg_ret:+.4f}   ComboIC={combo_ic:+.4f}")

    def _collect_trajectories(
        self,
    ) -> Tuple[
        List[torch.Tensor], List[int], List[torch.Tensor], List[float], List[float]
    ]:
        """
        从环境中采样一批完整轨迹，并计算回报与优势值（GAE λ=1）。

        返回
        ----------
        states : torch.Tensor
            形状 ``(T, seq_len)`` 的 token 序列张量。
        actions : torch.Tensor
            长度 ``T`` 的动作 ID。
        logps : torch.Tensor
            对应动作的对数概率。
        returns : torch.Tensor
            折现回报。
        advantages : torch.Tensor
            Advantage 值（此处为 `return - value`）。
        """
        states, actions, logps, rewards, dones, values = [], [], [], [], [], []

        obs = torch.tensor([self.env.reset()], device=self.device)
        h_p, h_v = self.policy_net.init_hidden(1, self.device), self.value_net.init_hidden(1, self.device)

        while len(states) < self.batch_size:
            logits, h_p = self.policy_net(obs, h_p)

            # -------- Invalid-action-mask --------
            valid = self.env.valid_actions()
            if len(valid) == 0:
                # 无合法动作 → 强制收尾并惩罚
                sep_id = self.env.tokenizer.sep_token_id
                obs, _, _, _ = self.env.step(sep_id)
                action = torch.tensor(sep_id, device=self.device)
                logp = torch.tensor(0.0, device=self.device)
                print(obs, h_v)
                value, h_v = self.value_net(obs, h_v)
                reward = -1.0
                done = True
            else:
                mask = torch.full((self.vocab_size,), float('-inf'), device=self.device)
                mask[valid] = 0.0                                   # 合法动作设置成 0，其它仍为 −inf
                logits = logits + mask                              # 非法动作 logits 变为 −inf
                dist = Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

                value, h_v = self.value_net(obs, h_v)

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
    
    