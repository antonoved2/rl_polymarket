"""
PPO Training v9 — aggressive time penalty to force SELL usage.

Key changes from v8:
- TIME_PENALTY: 0.001 → 0.01 (10x increase)
- Model must learn to exit quickly via SELL=3
- Same 4 actions, no hard TP/SL
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).parent))
from environment_v3 import PolymarketEnvV3


class ProfitCallback(BaseCallback):
    """Track profit and win rate during training."""
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
        return PolymarketEnvV3(
            data_path=data_path,
            asset=asset,
            initial_capital=1000.0,
            position_size_pct=0.10,
            taker_fee=0.025,
            max_steps_per_episode=90,
            min_hold_steps=5,
            seed=seed,
        )
    return _init


def evaluate_model(model, asset, data_path, n_episodes=50):
    """Evaluate model on separate data."""
    env = DummyVecEnv([make_env(asset, data_path, seed=i) for i in range(4)])
    
    all_stats = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = [False]
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done[0]:
                all_stats.append(info[0])
    
    if not all_stats:
        return {}
    
    capitals = [s.get('capital', 1000) for s in all_stats]
    pnls = [s.get('total_pnl', 0) for s in all_stats]
    wins = sum(s.get('wins', 0) for s in all_stats)
    losses = sum(s.get('losses', 0) for s in all_stats)
    total = wins + losses
    
    return {
        'avg_capital': np.mean(capitals),
        'avg_pnl': np.mean(pnls),
        'win_rate': wins / total * 100 if total > 0 else 0,
        'total_trades': total,
        'std_capital': np.std(capitals),
        'min_capital': min(capitals),
        'max_capital': max(capitals),
    }


def train(
    asset="btc",
    data_path="/opt/rl_trader/data/expanded_snapshots.jsonl",
    total_steps=500_000,
    seed=42,
    save_dir="/home/antonov5/.openclaw/workspace/rl_polymarket/models",
):
    print("=" * 60)
    print(f"  PPO Training v5 — {asset.upper()}")
    print(f"  Steps: {total_steps:,}")
    print(f"  Data: {data_path}")
    print("=" * 60)

    # Create env
    n_envs = 4
    env = DummyVecEnv([make_env(asset, data_path, seed=seed + i) for i in range(n_envs)])
    
    # Eval env
    eval_env = DummyVecEnv([make_env(asset, data_path, seed=999)])

    # PPO model with tuned hyperparameters
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
        ent_coef=0.01,  # encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
    )

    # Callbacks
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

    # Train
    print("\n[Train] Starting training...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_steps,
        callback=[profit_callback, eval_callback],
    )

    elapsed = time.time() - start_time
    print(f"\n[Train] Completed in {elapsed/3600:.1f}h")

    # Save final model
    final_path = os.path.join(save_dir, f"ppo_v5_{asset}_steps{total_steps}")
    model.save(final_path)
    print(f"[Train] Saved to {final_path}")

    # Evaluate
    print("\n[Eval] Evaluating model...")
    stats = evaluate_model(model, asset, data_path, n_episodes=50)
    print(f"[Eval] Results:")
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save stats
    stats_path = os.path.join(save_dir, f"ppo_v5_{asset}_eval.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[Eval] Stats saved to {stats_path}")

    return model, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="btc")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(asset=args.asset, total_steps=args.steps, seed=args.seed)
