#!/usr/bin/env python3
"""
Analyze PPO v11 trades — pure model-driven, 4 actions.
Tracks: entries, exits, PnL distribution, action usage.
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from stable_baselines3 import PPO
from environment_v4 import PolymarketEnvV4

def analyze(model_path, asset="btc", n_episodes=200):
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    all_trades = []  # completed trades with PnL
    all_entries = []
    all_data = []

    for ep in range(n_episodes):
        env = PolymarketEnvV4(
            data_path="/opt/rl_trader/data/expanded_snapshots.jsonl",
            asset=asset,
            initial_capital=1000.0,
            position_size_pct=0.10,
            taker_fee=0.025,
            max_steps_per_episode=90,
            min_hold_steps=3,
            seed=ep * 7 + 13,
        )

        obs, _ = env.reset(seed=ep * 7 + 13)
        done = False
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            had_position = env.position is not None
            old_pnl = env.total_pnl
            old_capital = env.capital

            obs, reward, terminated, truncated, info = env.step(action)

            d = {
                "episode": ep,
                "step": step,
                "action": action,
                "had_position": had_position,
                "has_position": env.position is not None,
                "capital": env.capital,
                "up_price": env.raw_data[env.current_data_idx]["up_price"] if env.current_data_idx < len(env.raw_data) else 0.5,
                "pnl_delta": env.total_pnl - old_pnl,
            }
            all_data.append(d)

            # Entry detected
            if env.position is not None and not had_position:
                p = env.position
                all_entries.append({
                    "episode": ep,
                    "step": step,
                    "side": "UP" if p.side == 1 else "DOWN",
                    "entry_price": p.entry_price,
                    "capital_before": old_capital,
                })

            # Exit detected (position was open, now closed)
            if had_position and env.position is None:
                pnl = env.total_pnl - old_pnl
                all_trades.append({
                    "episode": ep,
                    "step": step,
                    "pnl": pnl,
                    "capital_after": env.capital,
                })

            done = terminated or truncated
            step += 1

    # ═══════════════════════════════════════════════════════════════════
    # Analysis
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"  V11 ANALYSIS ({n_episodes} episodes)")
    print(f"{'='*60}")

    # 1. Action distribution
    actions = [d["action"] for d in all_data]
    total = len(actions)
    print(f"\n📊 ACTION DISTRIBUTION:")
    print(f"  HOLD:     {actions.count(0):6d} ({actions.count(0)/total*100:.1f}%)")
    print(f"  BUY_UP:   {actions.count(1):6d} ({actions.count(1)/total*100:.1f}%)")
    print(f"  BUY_DOWN: {actions.count(2):6d} ({actions.count(2)/total*100:.1f}%)")
    print(f"  SELL:     {actions.count(3):6d} ({actions.count(3)/total*100:.1f}%)")

    # 2. Entries
    print(f"\n📈 ENTRIES: {len(all_entries)}")
    if all_entries:
        up_entries = [e for e in all_entries if e["side"] == "UP"]
        down_entries = [e for e in all_entries if e["side"] == "DOWN"]
        print(f"  BUY_UP:   {len(up_entries)}")
        print(f"  BUY_DOWN: {len(down_entries)}")

        # Entry price
        prices = [e["entry_price"] for e in all_entries]
        print(f"\n💰 ENTRY PRICE:")
        print(f"  Mean:   {np.mean(prices):.3f}")
        print(f"  Median: {np.median(prices):.3f}")
        print(f"  Min:    {np.min(prices):.3f}")
        print(f"  Max:    {np.max(prices):.3f}")

        # Price buckets
        buckets = [(0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, 0.85)]
        print(f"  Buckets:")
        for lo, hi in buckets:
            cnt = sum(1 for p in prices if lo <= p < hi)
            print(f"    [{lo:.2f}-{hi:.2f}): {cnt}")

        # Entry timing
        steps = [e["step"] for e in all_entries]
        print(f"\n⏱️  ENTRY STEP:")
        print(f"  Mean:   {np.mean(steps):.1f}")
        print(f"  Median: {np.median(steps):.1f}")

    # 3. Completed trades (with PnL)
    print(f"\n🔴 COMPLETED TRADES: {len(all_trades)}")
    if all_trades:
        pnls = [t["pnl"] for t in all_trades]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p <= 0)

        print(f"  Wins:   {wins} ({wins/len(pnls)*100:.1f}%)")
        print(f"  Losses: {losses} ({losses/len(pnls)*100:.1f}%)")
        print(f"\n  PnL Distribution:")
        print(f"    Mean:   ${np.mean(pnls):+.2f}")
        print(f"    Median: ${np.median(pnls):+.2f}")
        print(f"    Std:    ${np.std(pnls):.2f}")
        print(f"    Min:    ${np.min(pnls):+.2f}")
        print(f"    Max:    ${np.max(pnls):+.2f}")

        # PnL buckets
        buckets = [(-999, -100), (-100, -50), (-50, -10), (-10, 0), (0, 10), (10, 50), (50, 100), (100, 999)]
        print(f"    PnL buckets:")
        for lo, hi in buckets:
            cnt = sum(1 for p in pnls if lo <= p < hi)
            if cnt > 0:
                print(f"      [{lo:+d}, {hi:+d}): {cnt}")

    # 4. Episode PnL
    print(f"\n📈 EPISODE PnL:")
    ep_pnls = []
    for ep in range(n_episodes):
        ep_data = [d for d in all_data if d["episode"] == ep]
        if ep_data:
            ep_pnl = ep_data[-1]["capital"] - 1000.0
            ep_pnls.append(ep_pnl)

    if ep_pnls:
        wins_ep = sum(1 for p in ep_pnls if p > 0)
        losses_ep = sum(1 for p in ep_pnls if p <= 0)
        print(f"  Profitable episodes: {wins_ep}/{len(ep_pnls)} ({wins_ep/len(ep_pnls)*100:.1f}%)")
        print(f"  Mean PnL:   ${np.mean(ep_pnls):+.2f}")
        print(f"  Median PnL: ${np.median(ep_pnls):+.2f}")
        print(f"  Std PnL:    ${np.std(ep_pnls):.2f}")
        print(f"  Min PnL:    ${np.min(ep_pnls):+.2f}")
        print(f"  Max PnL:    ${np.max(ep_pnls):+.2f}")

    # 5. SELL analysis
    print(f"\n🔴 SELL ACTION ANALYSIS:")
    sell_data = [d for d in all_data if d["action"] == 3]
    print(f"  Total SELL: {len(sell_data)}")
    if sell_data:
        sell_with_pos = sum(1 for d in sell_data if d["had_position"])
        sell_without_pos = sum(1 for d in sell_data if not d["had_position"])
        print(f"  With position:    {sell_with_pos}")
        print(f"  Without position: {sell_without_pos}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/opt/rl_trader/models/ppo_v5_btc_steps500000"
    analyze(model_path, n_episodes=200)
