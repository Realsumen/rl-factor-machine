# train.py
import os
import yaml
import pandas as pd

from utility import set_random_seed
from data import load_market_data
from combination import AlphaCombinationModel
from tokenizer import AlphaTokenizer
from alpha_generation_env import AlphaGenerationEnv
from generator import RLAlphaGenerator


def main(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_random_seed(cfg.get("random_seed", 42))

    data_cfg = cfg["data"]
    df = load_market_data(
        path=data_cfg["path"], multiplier=data_cfg["multiplier"], n=data_cfg["n"]
    )

    combo = AlphaCombinationModel(max_pool_size=cfg["model"]["max_pool_size"])
    combo.inject_data(df, target_col=data_cfg["target_col"])

    tokenizer = AlphaTokenizer()
    env = AlphaGenerationEnv(
        combo_model=combo, tokenizer=tokenizer, max_len=cfg["env"]["max_len"]
    )

    gen_cfg = cfg["generator"]
    gen_cfg["vocab_size"] = tokenizer.vocab_size
    gen_cfg["max_seq_len"] = cfg["env"]["max_len"]
    agent = RLAlphaGenerator(env=env, config=gen_cfg)

    print(f"Starting training for {gen_cfg['num_iterations']} iterations...")
    agent.train(num_iterations=gen_cfg["num_iterations"])

    out_cfg = cfg["output"]
    os.makedirs(os.path.dirname(out_cfg["alphas_weights_path"]), exist_ok=True)
    results = pd.DataFrame(
        {"expr": combo.expr_list, "ic": combo.ic_list, "weight": combo.weights}
    )
    results.to_csv(out_cfg["alphas_weights_path"], index=False)
    print(f"Saved discovered alphas and weights to {out_cfg['alphas_weights_path']}")


if __name__ == "__main__":
    main()
