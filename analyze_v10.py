#!/usr/bin/env python3
"""
Analyze PPO v10 edge-based trades.
Tracks: entries by edge, time-to-exit, PnL distribution.
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from stable_baselines3 import PPO
from environment_v4 import PolymarketEnvV4, compute_fair_price, compute_edge

def analyze(model_path, asset="btc", n_episodes=200):
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    all_trades = []  # completed trades
    all_entries = []  # entry points
    all_sells = []  # sell points
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
        ep_obs_list = []

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
                "edge": info.get("edge", 0),
                "fair_price": info.get("fair_price", 0.5),
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
                    "edge": p.entry_edge,
                    "fair_price": p.entry_fair_price,
                    "capital_before": old_capital,
                })

            # Exit detected
            if had_position and env.position is None:
                pnl = env.total_pnl - old_pnl
                all_sells.append({
                    "episode": ep,
                    "step": step,
                    "pnl": pnl,
                    "steps_held": step - env.current_step + 1 if env.current_step > 0 else 0,
                })

            ep_obs_list.append(obs)
            done = terminated or truncated
            step += 1

    # ═══════════════════════════════════════════════════════════════════
    # Analysis
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"  V10 EDGE-BASED ANALYSIS ({n_episodes} episodes)")
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

        # Edge at entry
        edges = [e["edge"] for e in all_entries]
        print(f"\n💰 EDGE AT ENTRY:")
        print(f"  Mean:   {np.mean(edges):.4f} ({np.mean(edges)*100:.2f}%)")
        print(f"  Median: {np.median(edges):.4f}")
        print(f"  Min:    {np.min(edges):.4f}")
        print(f"  Max:    {np.max(edges):.4f}")

        # Edge buckets
        buckets = [(0, 0.03), (0.03, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.50)]
        print(f"  Edge distribution:")
        for lo, hi in buckets:
            cnt = sum(1 for e in edges if lo <= e < hi)
            print(f"    [{lo:.2f}-{hi:.2f}): {cnt}")

        # Entry price
        prices = [e["entry_price"] for e in all_entries]
        print(f"\n💵 ENTRY PRICE:")
        print(f"  Mean:   {np.mean(prices):.3f}")
        print(f"  Median: {np.median(prices):.3f}")
        print(f"  Min:    {np.min(prices):.3f}")
        print(f"  Max:    {np.max(prices):.3f}")

        # Fair price at entry
        fairs = [e["fair_price"] for e in all_entries]
        print(f"\n📐 FAIR PRICE AT ENTRY:")
        print(f"  Mean:   {np.mean(fairs):.3f}")
        print(f"  Median: {np.median(fairs):.3f}")

    # 3. Sells
    print(f"\n🔴 SELLS (exits): {len(all_sells)}")
    if all_sells:
        pnls = [s["pnl"] for s in all_sells]
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
        buckets = [(-999, -5), (-5, -2), (-2, 0), (0, 2), (2, 5), (5, 999)]
        print(f"    PnL buckets:")
        for lo, hi in buckets:
            cnt = sum(1 for p in pnls if lo <= p < hi)
            print(f"      [{lo:+.0f}, {hi:+.0f}): {cnt}")

    # 4. Edge analysis — when does model see edge?
    print(f"\n📊 EDGE DISTRIBUTION (all timesteps):")
    all_edges = [d["edge"] for d in all_data]
    print(f"  Mean:   {np.mean(all_edges):.4f}")
    print(f"  Median: {np.median(all_edges):.4f}")
    print(f"  Std:    {np.std(all_edges):.4f}")

    # How often is edge > 3%?
    big_edge = sum(1 for e in all_edges if abs(e) > 0.03)
    print(f"  |edge| > 3%: {big_edge} ({big_edge/len(all_edges)*100:.1f}%)")
    big_edge_up = sum(1 for e in all_edges if e > 0.03)
    big_edge_down = sum(1 for e in all_edges if e < -0.03)
    print(f"    edge > 3% (UP cheap):   {big_edge_up}")
    print(f"    edge < -3% (UP expensive): {big_edge_down}")

    # 5. Episode PnL
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

    print(f"\n{'='*60}")


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/opt/rl_trader/models/ppo_v5_btc_steps500000"
    analyze(model_path, n_episodes=200)
