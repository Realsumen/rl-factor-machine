## 1. 数据侧：一次性准备，后续只做增量

| 步骤      | 关键点                                                                 | 代码接口                                                |
| ------- | ------------------------------------------------------------------- | --------------------------------------------------- |
| **加载**  | 读取日频或分钟级 CSV，形成 DataFrame；保留必要字段（open、high、low、close、volume、vwap…）。 | `data.load_market_data()`                           |
| **划分**  | 训练／验证／测试 = 8:1:1 或按年份切分，保证验证集不泄漏。                                   | —                                                   |
| **目标**  | 一般选取 *n-day forward return*；在注入前就写入 df。                             | `AlphaCombinationModel.inject_data(df, target_col)` |
| **预处理** | 对每只股票做前复权 → 缺口填补 → 对齐交易日；无需提前标准化，组合模型内部有 winsorize + z-score。       | —                                                   |

---

## 2. 组合层（Combination Model）：持续更新的因子池

1. **池管理**

   * 每来一条新因子：winsorize → z-score → 计算单因子 IC。
   * 把它插入 `self.norm_alphas`、`self.ic_list`，再用 **L1 约束** 的凸优化重新算权重。
   * 池子超限时，按照 |w·IC| 最小的删。

2. **评价缓存**

   * IC 计算、权重解都进 `self._cache`，表达式相同就不用再跑。
   * IC 的键：`('expr_ic', expr)`；方便在环境里直接拿 reward。

3. **开放接口**

   * `evaluate_alpha(expr)`：给 RL 环境实时打分。
   * `score()`：拿当前组合在验证集的 IC，当作柱状图监控。

> **落地要点**：
>
> * 先 warm-up 几十个 **人写** 或 **随机** 因子，让权重解有初始稳定性。
> * 组合优化可批量、异步；主线程生成表达式时，权重优化可放线程池。

---

## 3. 强化学习环境：把公式空间离散化

| 组件                      | 职责                                                                                                                            |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Tokenizer**           | 把 RPN 表达式 ↔ token id；保证常量映射到最近 bucket，减少词表尺寸。                                                                                 |
| **AlphaGenerationEnv**  | *State* = 当前 token 序列；*Action* = 下一个 token；*Reward* = `combo.evaluate_alpha(expr)`；合法动作用 `valid_actions()` 控制；`max_len` 确保不炸。 |
| **Invalid-Action Mask** | 生成 logits 后，对不合法 id 置 -1e9；否则策略网络会学到非法 token。                                                                                 |

> **技巧**
>
> * `max_len-1` 之前禁止输出 `[SEP]`，否则会狂刷无意义短式子。
> * 终止奖励用「**组合 IC 的绝对值增量**」而非单因子 IC，能让 agent 学会互补性。

---

## 4. PPO 代理：策略 & 价值网络

1. **网络**

   * `embedding → LSTM → FC` 输出 logits / value。
   * `hidden_dim` 128\~256 足够；要多 GPU 时只把 LSTM 拆 DataParallel，embedding 共享。

2. **采样循环** (`_collect_trajectories`)

   * 单环境或 vectorized\_env 都行；核心是 **batch 收齐再更新**。
   * 走完一条表达式即 `done=True`，重新 `reset`，权重池已含新因子。

3. **优势估计**

   * 论文用 GAE，你这里 λ=1、γ=1 简化为 return-value。
   * 如 reward = ΔIC，可加 √T 衰减，避免长式子 reward 偏高。

4. **更新**

   * 多个 `update_epochs`，shuffle batch。
   * `entropy_coef` 逐渐线性降温，让后期更 exploitation。

---

## 5. 训练节奏与工程化

1. **Curriculum**

   * 先限制算子集合、常量粒度，等 agent 有收益再逐步解锁更多算子。
2. **并行**

   * 「组合权重优化」和「PPO update」CPU 线程化； GPU 负责前向采样 + 反向。
3. **缓存落磁盘**

   * 把 `expr → ic`、`expr → alpha ndarray` 存 LMDB；重启继续训练。
4. **监控**

   * TensorBoard：单因子 IC、组合 IC、池子大小、策略熵。

---

## 6. 评估与回测

1. **离线指标**：

   * **IC / RankIC**：验证集 & 测试集。
   * 稳定性：滚动窗口 IC 的均值±方差。

2. **在线回测**：

   * 组合输出作为 *score*，日内 top-k drop-n 调仓；交易成本、滑点、持仓上限要写死。
   * 跟基准指数对比，画 equity curve、max DD、Sharpe、Turnover。

3. **稳健性实验**：

   * **冷启动**：全部删权重重算，看组合能否迅速恢复。
   * **信息泄漏**：不同截面（大盘/小盘、行业 neutral）做分组测试。

---

## 7. 可扩展方向

* **多目标奖励**：IC 提升 + 表达式长度惩罚，权重可调。
* **树-LSTM/Transformer**：用 AST 结构编码表达式，比纯序列更鲁棒。
* **非线性组合**：把组合模型换成 *Elastic Net*、*浅层 GBDT*，但仍保持可解释。
* **在线学习**：每天收盘后新生成少量因子 → 滚动更新池子 → 第二天盘中实时打分。

---

### 小结

> **核心逻辑**：
> **RL 负责探索表达式空间** → **Combination Model 负责“评审 + 投票”** → **IC 增量变成 PPO 的回报**。
> 这样就把「因子挖掘」和「协同组合」联成一个闭环，避免了“单因子好看、组合却拉胯”的老问题。
