#!/usr/bin/env python3
"""
PPO Training v9 — Risk-aware training with drawdown penalties.

Same 95 features as v8, but environment teaches risk management:
  - Drawdown penalty in reward (starts at 5% DD)
  - Consecutive loss penalty (3+ losses → penalty)
  - Dynamic position sizing (halve size after 10% DD)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).parent))
from environment_v5 import PolymarketEnvV7 as PolymarketEnv, MIN_HOLD_STEPS


class ProfitCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_stats = []

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            latest = self.model.ep_info_buffer[-1]
            info = latest.get('episode', {})
            stats = {
                'reward': info.get('r', 0),
                'length': info.get('l', 0),
                'capital': latest.get('capital', 1000),
                'total_pnl': latest.get('total_pnl', 0),
                'trade_count': latest.get('trade_count', 0),
                'wins': latest.get('wins', 0),
                'losses': latest.get('losses', 0),
                'episode_trades': latest.get('episode_trades', 0),
            }
            self.episode_stats.append(stats)

        if self.n_calls % 10000 == 0 and len(self.episode_stats) > 0:
            recent = min(500, len(self.episode_stats))
            pnls = [s['total_pnl'] for s in self.episode_stats[-recent:]]
            caps = [s['capital'] for s in self.episode_stats[-recent:]]
            wins = sum(s['wins'] for s in self.episode_stats[-recent:])
            losses = sum(s['losses'] for s in self.episode_stats[-recent:])
            trades = sum(s['episode_trades'] for s in self.episode_stats[-recent:])
            total = wins + losses
            wr = wins / total * 100 if total > 0 else 0
            avg_pnl = np.mean(pnls)
            avg_cap = np.mean(caps)
            avg_trades = trades / recent
            print(f"\n[Step {self.n_calls:,}] WR: {wr:.1f}% | Avg P&L: ${avg_pnl:.2f} | "
                  f"Avg Cap: ${avg_cap:.2f} | Avg Trades/Ep: {avg_trades:.1f}")
        return True


def make_env(asset, data_path, seed=42):
    def _init():
        return PolymarketEnv(
            data_path=data_path,
            asset=asset,
            initial_capital=1000.0,
            position_size_pct=0.02,
            taker_fee=0.025,
            max_steps_per_episode=90,
            min_hold_steps=MIN_HOLD_STEPS,
            seed=seed,
        )
    return _init


def train(
    asset="btc",
    data_path="/home/antonov5/.openclaw/workspace/rl_polymarket/data/expanded_snapshots_v4.jsonl",
    total_steps=500_000,
    seed=42,
    save_dir="/home/antonov5/.openclaw/workspace/rl_polymarket/models",
):
    print("=" * 60)
    print(f"  PPO Training v9 — {asset.upper()} — Risk-Aware")
    print(f"  Steps: {total_steps:,}")
    print(f"  Data: {data_path}")
    print("=" * 60)

    if not os.path.exists(data_path):
        print(f"[ERROR] Data file not found: {data_path}")
        return

    n_envs = 4
    env = DummyVecEnv([make_env(asset, data_path, seed=seed + i) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(asset, data_path, seed=999)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
    )

    profit_callback = ProfitCallback()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=20000,
        deterministic=True,
        render=False,
        n_eval_episodes=20,
        verbose=1,
    )

    print("\n[Train] Starting training...")
    start_time = time.time()
    model.learn(total_timesteps=total_steps, callback=[profit_callback, eval_callback])
    elapsed = time.time() - start_time
    print(f"\n[Train] Completed in {elapsed/3600:.1f}h")

    final_path = os.path.join(save_dir, f"ppo_v9_{asset}_steps{total_steps}")
    model.save(final_path)
    print(f"[Train] Saved to {final_path}")

    # Evaluate
    print("\n[Eval] Evaluating model...")
    all_stats = []
    for ep in range(50):
        eval_e = make_env(asset, data_path, seed=9000 + ep)()
        obs, _ = eval_e.reset(seed=9000 + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_e.step(int(action))
            done = terminated or truncated
        all_stats.append(info)

    capitals = [s.get('capital', 1000) for s in all_stats]
    pnls = [s.get('total_pnl', 0) for s in all_stats]
    wins = sum(s.get('wins', 0) for s in all_stats)
    losses = sum(s.get('losses', 0) for s in all_stats)
    total = wins + losses

    stats = {
        'avg_capital': float(np.mean(capitals)),
        'avg_pnl': float(np.mean(pnls)),
        'win_rate': float(wins / total * 100) if total > 0 else 0,
        'total_trades': total,
        'std_capital': float(np.std(capitals)),
        'min_capital': float(min(capitals)),
        'max_capital': float(max(capitals)),
    }
    print(f"[Eval] Results:")
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    stats_path = os.path.join(save_dir, f"ppo_v9_{asset}_eval.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    return model, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="btc")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    data_path = args.data or "/home/antonov5/.openclaw/workspace/rl_polymarket/data/expanded_snapshots_v4.jsonl"
    train(asset=args.asset, data_path=data_path, total_steps=args.steps, seed=args.seed)
