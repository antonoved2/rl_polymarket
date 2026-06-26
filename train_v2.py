"""
Multi-asset PPO training — одна модель на BTC+ETH+SOL.
"""

import sys, os, json, time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).parent))
from environment_v2 import PolymarketMultiEnv


def make_env(**kwargs):
    def _init():
        env = PolymarketMultiEnv(**kwargs)
        return Monitor(env)
    return _init


def train(
    total_steps: int = 200_000,
    seed: int = 42,
    save_dir: str = "models",
):
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("Multi-Asset PPO Training (BTC+ETH+SOL)")
    print(f"Total steps: {total_steps:,}")
    print("=" * 60)

    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([make_env(seed=seed)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="auto",
        seed=seed,
    )

    print("\n[Training] Starting...")
    start = time.time()
    model.learn(total_timesteps=total_steps, progress_bar=False)
    elapsed = time.time() - start

    model_path = os.path.join(save_dir, f"ppo_multi_steps{total_steps}")
    model.save(model_path)
    print(f"\n[Saved] {model_path}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    env.close()
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(total_steps=args.total_steps, seed=args.seed)
