"""
PPO Training v3 — улучшенная версия для высокого Win Rate.

Улучшения:
- Увеличенные шаги: 150K-300K
- Лучшие гиперпараметры: lr=1e-4, n_epochs=15, ent_coef=0.01
- Reward shaping: усиленный штраф за убытки, бонус за win streak
- Data augmentation: случайный старт внутри каждого периода
- Больше данных через расширенный датасет

Запуск:
    python3 train_v3.py --asset btc --total-steps 150000
    python3 train_v3.py --asset eth --total-steps 200000
    python3 train_v3.py --asset sol --total-steps 300000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).parent))
from environment_v2 import PolymarketMultiEnv


class WinRateCallback(BaseCallback):
    """Callback для отслеживания Win Rate и детальных метрик."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_pnls = []
        self.episode_trade_counts = []
        self.episode_wins = []  # 1 if profitable, 0 otherwise
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            latest = self.model.ep_info_buffer[-1]
            if 'episode' in latest:
                self.episode_rewards.append(latest['episode']['r'])
                self.episode_lengths.append(latest['episode']['l'])
            if 'total_pnl' in latest:
                pnl = latest['total_pnl']
                self.episode_pnls.append(pnl)
                self.episode_wins.append(1 if pnl > 0 else 0)
            if 'trade_count' in latest:
                self.episode_trade_counts.append(latest['trade_count'])

        # Логирование каждые 5000 шагов
        if self.n_calls % 5000 == 0 and len(self.episode_rewards) > 0:
            recent = 200
            r = self.episode_rewards[-recent:]
            w = self.episode_wins[-recent:]
            p = self.episode_pnls[-recent:]
            t = self.episode_trade_counts[-recent:]

            wr = np.mean(w) * 100 if w else 0
            avg_pnl = np.mean(p) if p else 0
            avg_trades = np.mean(t) if t else 0

            print(f"\n[Step {self.n_calls:,}] "
                  f"WR: {wr:.1f}% | "
                  f"Avg P&L: ${avg_pnl:.2f} | "
                  f"Avg Trades: {avg_trades:.1f} | "
                  f"Episodes: {len(self.episode_rewards)}")

        return True


def make_env(asset, data_path, seed=42, initial_capital=1000.0):
    """Создаёт среду для одного актива."""
    def _init():
        from environment import PolymarketEnv
        env = PolymarketEnv(
            data_path=data_path,
            asset=asset,
            initial_capital=initial_capital,
            position_size_pct=0.10,
            taker_fee=0.025,
            max_steps_per_episode=90,
            drawdown_penalty=0.15,  # усиленный штраф за drawdown
            trade_penalty=0.0005,   # меньше штраф за сделки (поощряем активность)
            seed=seed,
        )
        return env
    return _init


def make_multi_env(data_path, seed=42, initial_capital=1000.0):
    """Создаёт multi-asset среду."""
    def _init():
        env = PolymarketMultiEnv(
            data_path=data_path,
            assets=["btc", "eth", "sol"],
            initial_capital=initial_capital,
            position_size_pct=0.10,
            taker_fee=0.025,
            max_steps_per_episode=90,
            drawdown_penalty=0.15,
            trade_penalty=0.0005,
            seed=seed,
        )
        return env
    return _init


def train_single_asset(
    asset: str = "btc",
    data_path: str = "/home/antonov5/.openclaw/workspace/data_collector/data/expanded/expanded_snapshots.jsonl",
    total_steps: int = 150_000,
    seed: int = 42,
    save_dir: str = "/home/antonov5/.openclaw/workspace/rl_polymarket/models",
):
    """Обучает single-asset PPO модель."""
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"PPO v3 Training — {asset.upper()}")
    print(f"Total steps: {total_steps:,}")
    print("=" * 60)

    env = DummyVecEnv([make_env(asset, data_path, seed=seed)])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        # Консервативное обучение для стабильности
        learning_rate=1e-4,       # меньше для стабильности
        n_steps=512,              # больше буфер для лучшей оценки
        batch_size=128,           # больше батч для стабильности
        n_epochs=15,              # больше эпох на обновление
        gamma=0.99,               # высокий discount — долгосрочная перспектива
        gae_lambda=0.95,
        clip_range=0.15,          # консервативный clip
        ent_coef=0.01,            # мало энтропии — агент эксплуатирует лучшую стратегию
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="auto",
        seed=seed,
    )

    callback = WinRateCallback()

    print("\n[Training] Starting...")
    start_time = time.time()

    model.learn(
        total_timesteps=total_steps,
        callback=callback,
        progress_bar=False,
    )

    elapsed = time.time() - start_time
    print(f"\n[Training] Completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Сохраняем модель
    model_path = os.path.join(save_dir, f"ppo_v3_{asset}_steps{total_steps}")
    model.save(model_path)
    print(f"[Saved] Model: {model_path}")

    # Сохраняем метрики
    metrics = {
        "total_steps": total_steps,
        "elapsed_sec": elapsed,
        "n_episodes": len(callback.episode_rewards),
        "avg_reward": float(np.mean(callback.episode_rewards)) if callback.episode_rewards else 0,
        "avg_pnl": float(np.mean(callback.episode_pnls)) if callback.episode_pnls else 0,
        "avg_trades": float(np.mean(callback.episode_trade_counts)) if callback.episode_trade_counts else 0,
        "win_rate": float(np.mean(callback.episode_wins)) if callback.episode_wins else 0,
    }
    metrics_path = os.path.join(save_dir, f"metrics_v3_{asset}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Saved] Metrics: {metrics_path}")

    env.close()
    return model, metrics


def train_multi_asset(
    data_path: str = "/home/antonov5/.openclaw/workspace/data_collector/data/expanded/expanded_snapshots.jsonl",
    total_steps: int = 200_000,
    seed: int = 42,
    save_dir: str = "/home/antonov5/.openclaw/workspace/rl_polymarket/models",
):
    """Обучает multi-asset PPO модель."""
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"PPO v3 Multi-Asset Training (BTC+ETH+SOL)")
    print(f"Total steps: {total_steps:,}")
    print("=" * 60)

    env = DummyVecEnv([make_multi_env(data_path, seed=seed)])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=15,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="auto",
        seed=seed,
    )

    callback = WinRateCallback()

    print("\n[Training] Starting...")
    start_time = time.time()

    model.learn(
        total_timesteps=total_steps,
        callback=callback,
        progress_bar=False,
    )

    elapsed = time.time() - start_time
    print(f"\n[Training] Completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    model_path = os.path.join(save_dir, f"ppo_v3_multi_steps{total_steps}")
    model.save(model_path)
    print(f"[Saved] Model: {model_path}")

    metrics = {
        "total_steps": total_steps,
        "elapsed_sec": elapsed,
        "n_episodes": len(callback.episode_rewards),
        "avg_reward": float(np.mean(callback.episode_rewards)) if callback.episode_rewards else 0,
        "avg_pnl": float(np.mean(callback.episode_pnls)) if callback.episode_pnls else 0,
        "avg_trades": float(np.mean(callback.episode_trade_counts)) if callback.episode_trade_counts else 0,
        "win_rate": float(np.mean(callback.episode_wins)) if callback.episode_wins else 0,
    }
    metrics_path = os.path.join(save_dir, f"metrics_v3_multi.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Saved] Metrics: {metrics_path}")

    env.close()
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO v3 Training")
    parser.add_argument("--asset", type=str, default="btc", choices=["btc", "eth", "sol", "multi"])
    parser.add_argument("--total-steps", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.asset == "multi":
        train_multi_asset(total_steps=args.total_steps, seed=args.seed)
    else:
        train_single_asset(
            asset=args.asset,
            total_steps=args.total_steps,
            seed=args.seed,
        )
