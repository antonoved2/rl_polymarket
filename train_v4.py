"""
PPO Training v4 — финальная версия с TA индикаторами (45 фичей).

Улучшения:
- 300K шагов обучения
- Reward shaping: +1 за прибыль, -1 за убыток
- 45 фичей: 20 base + 25 TA (RSI, MACD, Bollinger, EMA, SMA, ATR, Stochastic)
- Увеличенный batch_size=256 для стабильности
- n_epochs=20 для лучшей оптимизации
- gamma=0.995 для долгосрочной перспективы
- Уменьшенный clip_range=0.1 для консервативного обучения
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).parent))
from environment_v3 import PolymarketEnvV3


class WinRateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_pnls = []
        self.episode_wins = []
        self.episode_trade_counts = []

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            latest = self.model.ep_info_buffer[-1]
            if 'episode' in latest:
                self.episode_rewards.append(latest['episode']['r'])
            if 'total_pnl' in latest:
                pnl = latest['total_pnl']
                self.episode_pnls.append(pnl)
                self.episode_wins.append(1 if pnl > 0 else 0)
            if 'trade_count' in latest:
                self.episode_trade_counts.append(latest['trade_count'])

        if self.n_calls % 5000 == 0 and len(self.episode_rewards) > 0:
            recent = min(200, len(self.episode_rewards))
            w = self.episode_wins[-recent:]
            p = self.episode_pnls[-recent:]
            wr = np.mean(w) * 100 if w else 0
            avg_pnl = np.mean(p) if p else 0
            print(f"\n[Step {self.n_calls:,}] WR: {wr:.1f}% | Avg P&L: ${avg_pnl:.2f} | Episodes: {len(self.episode_rewards)}")
        return True


def make_env(asset, data_path, seed=42):
    def _init():
        return PolymarketEnvV3(
            data_path=data_path,
            asset=asset,
            initial_capital=1000.0,
            position_size_pct=0.10,
            taker_fee=0.025,
            max_steps_per_episode=90,
            seed=seed,
        )
    return _init


def train(
    asset="btc",
    data_path="/home/antonov5/.openclaw/workspace/data_collector/data/expanded/expanded_snapshots.jsonl",
    total_steps=300_000,
    seed=42,
    save_dir="/home/antonov5/.openclaw/workspace/rl_polymarket/models",
):
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"PPO v4 Training — {asset.upper()} (45 features with TA)")
    print(f"Total steps: {total_steps:,}")
    print("=" * 60)

    env = DummyVecEnv([make_env(asset, data_path, seed=seed)])

    model = PPO(
        "MlpPolicy", env,
        learning_rate=5e-5,
        n_steps=1024,
        batch_size=256,
        n_epochs=20,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="auto",
        seed=seed,
    )

    callback = WinRateCallback()

    print("\n[Training] Starting...")
    start = time.time()
    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=False)
    elapsed = time.time() - start

    model_path = os.path.join(save_dir, f"ppo_v4_{asset}_steps{total_steps}")
    model.save(model_path)
    print(f"\n[Saved] {model_path}")

    metrics = {
        "total_steps": total_steps,
        "elapsed_sec": elapsed,
        "n_episodes": len(callback.episode_rewards),
        "avg_reward": float(np.mean(callback.episode_rewards)) if callback.episode_rewards else 0,
        "avg_pnl": float(np.mean(callback.episode_pnls)) if callback.episode_pnls else 0,
        "win_rate": float(np.mean(callback.episode_wins)) if callback.episode_wins else 0,
    }
    metrics_path = os.path.join(save_dir, f"metrics_v4_{asset}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Saved] {metrics_path}")

    env.close()
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="btc", choices=["btc", "eth", "sol"])
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(asset=args.asset, total_steps=args.total_steps, seed=args.seed)
