"""
Скрипт обучения PPO агента для Polymarket.

Использует Stable-Baselines3 с кастомной средой.
Обучение на исторических данных с walk-forward валидацией.

Запуск:
    python3 train.py --asset btc --total-steps 100000
    python3 train.py --asset btc --total-steps 100000 --eval  # с оценкой
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
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# Добавляем путь к среде
sys.path.insert(0, str(Path(__file__).parent))
from environment import PolymarketEnv


class TrainingMetricsCallback(BaseCallback):
    """Кастомный callback для логирования метрик."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_pnls = []
        self.episode_trade_counts = []

    def _on_step(self) -> bool:
        # Собираем информацию из info
        if len(self.model.ep_info_buffer) > 0:
            latest = self.model.ep_info_buffer[-1]
            if 'episode' in latest:
                self.episode_rewards.append(latest['episode']['r'])
                self.episode_lengths.append(latest['episode']['l'])
            if 'total_pnl' in latest:
                self.episode_pnls.append(latest['total_pnl'])
            if 'trade_count' in latest:
                self.episode_trade_counts.append(latest['trade_count'])

        # Логирование каждые 1000 шагов
        if self.n_calls % 1000 == 0 and len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-100:]
            recent_pnls = self.episode_pnls[-100:] if self.episode_pnls else [0]
            recent_trades = self.episode_trade_counts[-100:] if self.episode_trade_counts else [0]

            print(f"\n[Step {self.n_calls}] "
                  f"Avg Reward: {np.mean(recent_rewards):.4f} | "
                  f"Avg P&L: ${np.mean(recent_pnls):.2f} | "
                  f"Avg Trades: {np.mean(recent_trades):.1f} | "
                  f"Episodes: {len(self.episode_rewards)}")

        return True


def make_env(asset: str, data_path: str, initial_capital: float = 1000.0,
             position_size_pct: float = 0.10, taker_fee: float = 0.025,
             max_steps: int = 90, seed: Optional[int] = None) -> gym.Env:
    """Создаёт и оборачивает среду."""
    def _init():
        env = PolymarketEnv(
            data_path=data_path,
            asset=asset,
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            taker_fee=taker_fee,
            max_steps_per_episode=max_steps,
            seed=seed,
        )
        env = Monitor(env)
        return env
    return _init


def train(
    asset: str = "btc",
    data_path: str = "/home/antonov5/.openclaw/workspace/data_collector/data/expanded/expanded_snapshots.jsonl",
    total_steps: int = 100_000,
    n_envs: int = 1,
    eval_freq: int = 10_000,
    eval_episodes: int = 20,
    save_dir: str = "/home/antonov5/.openclaw/workspace/rl_polymarket/models",
    seed: int = 42,
):
    """Обучает PPO агента."""

    print("=" * 60)
    print(f"PPO Training — {asset.upper()}")
    print(f"Total steps: {total_steps:,}")
    print(f"Environments: {n_envs}")
    print("=" * 60)

    # Создаём директорию для моделей
    os.makedirs(save_dir, exist_ok=True)

    # Создаём среду
    if n_envs == 1:
        env = DummyVecEnv([make_env(asset, data_path, seed=seed)])
    else:
        env = SubprocVecEnv([
            make_env(asset, data_path, seed=seed + i) for i in range(n_envs)
        ])

    # Параметры PPO — оптимизированы для нашей задачи
    model = PPO(
        policy="MlpPolicy",
        env=env,
        # Скорость обучения
        learning_rate=3e-4,
        # Размер буфера (сколько шагов собираем перед обновлением)
        n_steps=256,
        # Размер мини-батча для обновления
        batch_size=64,
        # Количество эпох на каждый буфер
        n_epochs=10,
        # Коэффициент дисконтирования (короткий горизонт — 15 мин)
        gamma=0.95,
        # GAE lambda
        gae_lambda=0.95,
        # PPO clip
        clip_range=0.2,
        # Entropy coefficient (поощряем исследование, но не слишком)
        # Низкое значение → агент чаще выбирает HOLD
        ent_coef=0.03,
        # Value function coefficient
        vf_coef=0.5,
        # Максимный градиент
        max_grad_norm=0.5,
        # Логирование
        verbose=1,
        # Tensorboard
        tensorboard_log=None,
        # Сид
        seed=seed,
        # Устройство
        device="auto",
    )

    # Callbacks
    metrics_callback = TrainingMetricsCallback()

    # Обучение
    print("\n[Training] Starting...")
    start_time = time.time()

    model.learn(
        total_timesteps=total_steps,
        callback=metrics_callback,
        progress_bar=False,
    )

    elapsed = time.time() - start_time
    print(f"\n[Training] Completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Сохраняем модель
    model_path = os.path.join(save_dir, f"ppo_{asset}_steps{total_steps}")
    model.save(model_path)
    print(f"[Saved] Model: {model_path}")

    # Сохраняем метрики
    metrics = {
        "total_steps": total_steps,
        "elapsed_sec": elapsed,
        "n_episodes": len(metrics_callback.episode_rewards),
        "avg_reward": float(np.mean(metrics_callback.episode_rewards)) if metrics_callback.episode_rewards else 0,
        "avg_pnl": float(np.mean(metrics_callback.episode_pnls)) if metrics_callback.episode_pnls else 0,
        "avg_trades": float(np.mean(metrics_callback.episode_trade_counts)) if metrics_callback.episode_trade_counts else 0,
    }
    metrics_path = os.path.join(save_dir, f"metrics_{asset}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Saved] Metrics: {metrics_path}")

    env.close()
    return model, metrics


def evaluate(
    model_path: str,
    asset: str = "btc",
    data_path: str = "/home/antonov5/.openclaw/workspace/data_collector/data/expanded/expanded_snapshots.jsonl",
    n_episodes: int = 50,
    seed: int = 123,
):
    """Оценивает обученную модель на out-of-sample данных."""

    print("=" * 60)
    print(f"Evaluation — {asset.upper()}")
    print(f"Episodes: {n_episodes}")
    print("=" * 60)

    # Загружаем модель
    model = PPO.load(model_path)

    # Создаём среду
    env = DummyVecEnv([make_env(asset, data_path, seed=seed)])

    # Оценка
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_episodes,
        deterministic=True, return_episode_rewards=False,
    )

    print(f"\n[Results]")
    print(f"  Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}")

    # Детальная статистика
    episode_stats = []
    obs = env.reset()
    for _ in range(n_episodes):
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                episode_stats.append(info[0])

    if episode_stats:
        capitals = [s.get("final_capital", 0) for s in episode_stats if "final_capital" in s]
        pnls = [s.get("total_pnl", 0) for s in episode_stats if "total_pnl" in s]
        trades = [s.get("trade_count", 0) for s in episode_stats if "trade_count" in s]

        if capitals:
            print(f"  Avg Final Capital: ${np.mean(capitals):.2f}")
            print(f"  Avg P&L: ${np.mean(pnls):.2f}")
            print(f"  Avg Trades/Episode: {np.mean(trades):.1f}")
            print(f"  Profitable Episodes: {sum(1 for p in pnls if p > 0)}/{len(pnls)}")

    env.close()
    return mean_reward, std_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training for Polymarket")
    parser.add_argument("--asset", type=str, default="btc", choices=["btc", "eth", "sol"])
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    parser.add_argument("--eval-only", type=str, default=None, help="Evaluate existing model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.eval_only:
        evaluate(args.eval_only, asset=args.asset, seed=args.seed)
    else:
        model, metrics = train(
            asset=args.asset,
            total_steps=args.total_steps,
            n_envs=args.n_envs,
            seed=args.seed,
        )

        if args.eval:
            model_path = f"/home/antonov5/.openclaw/workspace/rl_polymarket/models/ppo_{args.asset}_steps{args.total_steps}"
            evaluate(model_path, asset=args.asset, seed=args.seed + 100)
