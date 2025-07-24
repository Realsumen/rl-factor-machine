# %% [markdown]
# ### 这个 notebook 用来读取文件，并测试一些方法

# %%
import os
import numpy as np


def get_all_py_files(root_dir: str) -> list:
    """
    获取指定目录及其子目录下的所有 .py 文件路径。

    Args:
        root_dir (str): 起始目录路径。

    Returns:
        list: 包含所有 .py 文件完整路径的列表。
    """
    py_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".py"):
                py_files.append(os.path.join(dirpath, file))
    return py_files


def write_all_py_contents_to_output(
    py_files: list, output_path: str = "src.txt"
) -> None:
    """
    将所有 .py 文件的内容写入一个文本文件中，并打印文件名作为分隔。

    Args:
        py_files (list): .py 文件路径列表。
        output_path (str): 输出文件路径。
    """
    with open(output_path, "w", encoding="utf-8") as out_file:
        for path in py_files:
            out_file.write(f"{'=' * 80}\n")
            out_file.write(f"File: {path}\n")
            out_file.write(f"{'-' * 80}\n")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    out_file.write(f.read())
            except Exception as e:
                out_file.write(f"⚠️ Error reading {path}: {e}\n")
            out_file.write("\n\n")


if True:
    root_directory = "."  # 当前目录
    all_py_files = get_all_py_files(root_directory)
    write_all_py_contents_to_output(all_py_files)
    print(f"📄 所有 Python 文件内容已写入 src.txt（共 {len(all_py_files)} 个文件）")


# %%
import importlib
from utility import set_random_seed


def reload_components():
    """
    重新加载以下模块，以便在开发过程中即时生效：
      - data.load_market_data
      - tokenizer.AlphaTokenizer
      - combination.AlphaCombinationModel
      - envs.AlphaGenerationEnv
      - generator.RLAlphaGenerator
    """
    import data, tokenizer, combination, alpha_generation_env, generator

    importlib.reload(data)
    importlib.reload(tokenizer)
    importlib.reload(combination)
    importlib.reload(alpha_generation_env)
    importlib.reload(generator)

    # 重新绑定到本地名称（可选）
    from data import load_market_data
    from tokenizer import AlphaTokenizer
    from combination import AlphaCombinationModel
    from alpha_generation_env import AlphaGenerationEnv
    from generator import RLAlphaGenerator

    return {
        "load_market_data": load_market_data,
        "AlphaTokenizer": AlphaTokenizer,
        "AlphaCombinationModel": AlphaCombinationModel,
        "AlphaGenerationEnv": AlphaGenerationEnv,
        "RLAlphaGenerator": RLAlphaGenerator,
    }


components = reload_components()


# %% [markdown]
# ### 1.测试 AlphaCombinationModel._compute_alpha_from_expr

# %%
from combination import AlphaCombinationModel
from data import load_market_data

df = load_market_data()
model = AlphaCombinationModel()
model.inject_data(df, target_col="target")

# 表达式：close 的 100 秒均值
expr = "close 100 ts_mean"
alpha = model._compute_alpha_from_expr(expr)

print("alpha shape:", alpha.shape)
print("alpha sample:", alpha[~np.isnan(alpha)][:5])


# %% [markdown]
# ###  2. 测试 AlphaCombinationModel.add_alpha_expr

# %%
ic = model.add_alpha_expr("high low sub 100 ts_max")
print("该因子的 IC 为：", ic)
print("当前池中因子数：", len(model.alphas))


# %% [markdown]
# ### 3. 测试 AlphaTokenizer.encode / decode

# %%
from tokenizer import AlphaTokenizer

tokenizer = AlphaTokenizer()

expr = "close 5 ts_mean"
ids = tokenizer.encode(expr)
decoded = tokenizer.decode(ids)

print("Token IDs:", ids)
print("Decoded expr:", decoded)


# %% [markdown]
# ### 4. 测试 AlphaGenerationEnv.reset / step

# %%
from alpha_generation_env import AlphaGenerationEnv
from combination import AlphaCombinationModel
from tokenizer import AlphaTokenizer
from data import load_market_data

df = load_market_data()
combo = AlphaCombinationModel()
combo.inject_data(df, target_col="target")
tokenizer = AlphaTokenizer()
env = AlphaGenerationEnv(combo, tokenizer)

obs = env.reset()
print("初始状态 token IDs:", obs)

valid = env.valid_actions()
action = valid[1]
obs2, reward, done, info = env.step(action)
print("新状态:", obs2)
print("Reward:", reward, "Done:", done)


# %% [markdown]
# ### 5. 测试 PolicyNetwork / ValueNetwork 输出维度

# %%
import torch
from generator import PolicyNetwork, ValueNetwork

vocab_size = 50
seq_len = 6
hidden_dim = 64
device = "cpu"

x = torch.randint(0, vocab_size, (1, seq_len))  # batch_size=1
policy = PolicyNetwork(vocab_size, hidden_dim).to(device)
value = ValueNetwork(vocab_size, hidden_dim).to(device)

h0_p = policy.init_hidden(1, device)
logits, _ = policy(x, h0_p)
print("Policy logits shape:", logits.shape)  # 应为 (1, vocab_size)

h0_v = value.init_hidden(1, device)
v, _ = value(x, h0_v)
print("Value estimate shape:", v.shape)  # 应为 (1,)


# %% [markdown]
# ### 6. 测试 RLAlphaGenerator._collect_trajectories

# %%
from generator import RLAlphaGenerator
from alpha_generation_env import AlphaGenerationEnv
from combination import AlphaCombinationModel
from tokenizer import AlphaTokenizer
from data import load_market_data
from utility import set_random_seed

# reload_components()

set_random_seed(10)

df = load_market_data()
combo = AlphaCombinationModel()
combo.inject_data(df, "target")
tokenizer = AlphaTokenizer()
env = AlphaGenerationEnv(combo, tokenizer, max_len=20)

cfg = dict(
    vocab_size=tokenizer.vocab_size,
    hidden_dim=64,
    batch_size=1280,
    device="cpu",
)

agent = RLAlphaGenerator(env, cfg)

s, a, logp, ret, adv = agent._collect_trajectories()
print("Sample states shape:", s.shape)
print("Sample actions shape:", a.shape)
print("Sample rewards (returns):", ret.shape, ret)


# %%
