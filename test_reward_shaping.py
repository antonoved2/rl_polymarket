"""
Test script to verify reward shaping improvements.
Runs a few episodes and prints rewards for different actions.
"""
import sys
sys.path.insert(0, '.')

import environment_v3
import numpy as np

def test_reward_shaping():
    """Test if SELL actions get proper rewards."""
    
    print("=" * 60)
    print("TESTING REWARD SHAPING")
    print("=" * 60)
    
    # Create environment
    env = environment_v3.PolymarketEnvV3(
        asset='btc',
        initial_capital=10000.0,
        data_path='/home/antonov5/.openclaw/workspace/data_collector/data/expanded/expanded_snapshots.jsonl'
    )
    
    print(f"\nN_FEATURES: {environment_v3.N_FEATURES}")
    print(f"Reward constants:")
    print(f"  TAKE_PROFIT_REWARD: {environment_v3.TAKE_PROFIT_REWARD}")
    print(f"  STOP_LOSS_PENALTY: {environment_v3.STOP_LOSS_PENALTY}")
    print(f"  EXIT_BONUS: {environment_v3.EXIT_BONUS}")
    print(f"  HOLD_PENALTY: {environment_v3.HOLD_PENALTY}")
    
    # Run a few episodes
    for episode in range(3):
        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1}")
        print(f"{'='*60}")
        
        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0
        
        while not done and step < 50:  # Limit to 50 steps
            # Alternate between actions for testing
            if step < 10:
                action = 0  # HOLD
            elif step < 20:
                action = 1  # BUY_UP
            elif step < 30:
                action = 2  # BUY_DOWN
            else:
                action = 3  # SELL
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            action_names = ["HOLD", "BUY_UP", "BUY_DOWN", "SELL"]
            print(f"Step {step:2d} | {action_names[action]:8s} | Reward: {reward:+.4f} | Capital: ${info['capital']:.2f}")
            
            step += 1
            done = terminated or truncated
        
        stats = env.get_episode_stats()
        print(f"\nEpisode {episode + 1} Stats:")
        print(f"  Total PnL: ${stats['total_pnl']:.2f}")
        print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"  Total Reward: {total_reward:.4f}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_reward_shaping()
