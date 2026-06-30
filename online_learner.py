"""
Online Learning Module for RL Live Trader v2
=============================================
Feedback loop: live trades → experience replay → periodic PPO fine-tuning → A/B validation → safe model swap

Architecture:
1. ExperienceReplayBuffer — stores (obs, action, reward, next_obs, done) from live trading
2. OnlinePPOTrainer — fine-tunes PPO from replay buffer every N hours
3. ABValidator — compares candidate model vs production on recent data
4. ModelSwapManager — atomic model swap with rollback

Flow:
- Every trade close → store transition in replay buffer
- Every N hours → fine-tune PPO on replay buffer → validate → swap if better
- If validation fails → keep old model, log warning
"""

import os
import json
import time
import shutil
import numpy as np
import gymnasium as gym
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
from typing import Optional, Dict, Any, List, Tuple

# ─── Config ────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Replay Buffer
    "replay_max_size": 5000,        # reduced from 10000
    "replay_min_for_train": 25,     # reduced from 200
    "replay_save_path": "/opt/rl_trader/replay_buffer",

    # Fine-tune Schedule
    "finetune_interval_hours": 0.25,  # 15 minutes instead of 4
    "finetune_steps": 3000,         # reduced from 5000
    "finetune_lr": 5e-4,
    "finetune_batch_size": 64,
    "finetune_n_epochs": 3,

    # A/B Validation
    "validation_window": 100,       # reduced from 200
    "validation_min_improvement": 0.02,
    "validation_metric": "avg_reward",

    # Model Management
    "models_dir": "/opt/rl_trader/models",
    "production_model": "ppo_v10_btc_steps1000000.zip",  # Will be overridden by bot's model
    "candidate_prefix": "ppo_v10_online_",       # Will be dynamically determined
    "max_candidates": 3,
    "rollback_on_failure": True,
}


# ─── Experience Replay Buffer ─────────────────────────────────────────────

class ExperienceReplayBuffer:
    """
    Stores live trading transitions: (obs, action, reward, next_obs, done)
    with trade metadata. Persisted to disk as numpy arrays + JSON metadata.
    """

    def __init__(self, max_size: int = 10000, save_path: str = None):
        self.max_size = max_size
        self.save_path = Path(save_path) if save_path else None

        # Ring buffer for transitions
        self.observations = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)
        self.next_observations = deque(maxlen=max_size)
        self.dones = deque(maxlen=max_size)

        # Metadata per transition
        self.metadata = deque(maxlen=max_size)  # dict per transition

        # Stats
        self.total_stored = 0
        self.last_save_time = 0

    def __len__(self):
        return len(self.observations)

    def add(self, obs: np.ndarray, action: int, reward: float,
            next_obs: np.ndarray, done: bool, metadata: Dict = None):
        """Store a transition from live trading."""
        self.observations.append(obs.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_obs.copy())
        self.dones.append(done)

        meta = metadata or {}
        meta["timestamp"] = int(time.time())
        meta["datetime"] = datetime.now(timezone.utc).isoformat()
        meta["transition_id"] = self.total_stored
        self.metadata.append(meta)

        self.total_stored += 1

    def add_trade(self, trade_result: Dict, obs_before: np.ndarray,
                  obs_after: np.ndarray, action: int):
        """
        Convert a closed trade into replay transitions.
        
        A single trade spanning N steps becomes N transitions:
        - reward = partial PnL at each step (interpolated)
        - final step gets the full PnL as reward
        """
        pnl = trade_result.get("pnl", 0)
        steps_held = trade_result.get("steps_held", 1)
        entry_price = trade_result.get("entry_price", 0.5)
        exit_price = trade_result.get("exit_price", 0.5)
        side = trade_result.get("side", "UP")

        # Create transitions for the trade
        # Step 0: obs_before, action → intermediate reward
        # Step N: last_obs, SELL → final reward
        
        # For a trade held N steps, we create min(3, N) transitions
        # to keep the buffer focused on decision-relevant moments
        n_transitions = min(3, max(1, steps_held))
        
        # Reward shaping: spread PnL across transitions, heavier at end
        # This teaches the model that the final outcome matters most
        if n_transitions == 1:
            rewards = [pnl]
        elif n_transitions == 2:
            rewards = [pnl * 0.2, pnl * 0.8]  # 20% early signal, 80% final
        else:
            rewards = [pnl * 0.1, pnl * 0.3, pnl * 0.6]  # escalating

        # Interpolate observations
        for i in range(n_transitions):
            # Blend between obs_before and obs_after
            alpha = (i + 1) / n_transitions
            obs = (1 - alpha) * obs_before + alpha * obs_after
            next_obs = obs_after if i == n_transitions - 1 else (1 - (alpha + 1/n_transitions)) * obs_before + (alpha + 1/n_transitions) * obs_after
            done = (i == n_transitions - 1)
            
            meta = {
                "trade_pnl": pnl,
                "trade_side": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "steps_held": steps_held,
                "transition_index": i,
                "n_transitions": n_transitions,
                "is_live": True,  # flag as live (not simulated)
            }
            self.add(obs, action, rewards[i], next_obs, done, meta)

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample n transitions uniformly."""
        n = min(n, len(self))
        indices = np.random.choice(len(self), size=n, replace=False)
        
        obs = np.array([self.observations[i] for i in indices])
        actions = np.array([self.actions[i] for i in indices])
        rewards = np.array([self.rewards[i] for i in indices])
        next_obs = np.array([self.next_observations[i] for i in indices])
        dones = np.array([self.dones[i] for i in indices])
        
        return obs, actions, rewards, next_obs, dones

    def get_recent(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get the N most recent transitions."""
        n = min(n, len(self))
        start = len(self) - n
        
        obs = np.array([self.observations[i] for i in range(start, len(self))])
        actions = np.array([self.actions[i] for i in range(start, len(self))])
        rewards = np.array([self.rewards[i] for i in range(start, len(self))])
        next_obs = np.array([self.next_observations[i] for i in range(start, len(self))])
        dones = np.array([self.dones[i] for i in range(start, len(self))])
        
        return obs, actions, rewards, next_obs, dones

    def get_stats(self) -> Dict:
        """Buffer statistics."""
        if len(self) == 0:
            return {"size": 0, "total_stored": self.total_stored}
        
        rewards = np.array(list(self.rewards))
        return {
            "size": len(self),
            "total_stored": self.total_stored,
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "positive_ratio": float(np.mean(rewards > 0)),
            "live_ratio": float(np.mean([1 for m in self.metadata if m.get("is_live")])),
        }

    def save(self):
        """Persist buffer to disk."""
        if not self.save_path:
            return
        
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Save arrays
        if len(self) > 0:
            np.savez_compressed(
                self.save_path / "buffer.npz",
                observations=np.array(list(self.observations)),
                actions=np.array(list(self.actions)),
                rewards=np.array(list(self.rewards)),
                next_observations=np.array(list(self.next_observations)),
                dones=np.array(list(self.dones)),
            )
        
        # Save metadata
        with open(self.save_path / "metadata.json", "w") as f:
            json.dump({
                "total_stored": self.total_stored,
                "current_size": len(self),
                "metadata": list(self.metadata),
            }, f, indent=2)
        
        self.last_save_time = time.time()

    def load(self):
        """Load buffer from disk."""
        if not self.save_path:
            return
        
        npz_path = self.save_path / "buffer.npz"
        meta_path = self.save_path / "metadata.json"
        
        if not npz_path.exists() or not meta_path.exists():
            return
        
        try:
            data = np.load(npz_path, allow_pickle=True)
            n = len(data["observations"])
            
            for i in range(n):
                self.observations.append(data["observations"][i])
                self.actions.append(int(data["actions"][i]))
                self.rewards.append(float(data["rewards"][i]))
                self.next_observations.append(data["next_observations"][i])
                self.dones.append(bool(data["dones"][i]))
            
            with open(meta_path) as f:
                meta = json.load(f)
                self.total_stored = meta.get("total_stored", n)
                for m in meta.get("metadata", []):
                    self.metadata.append(m)
            
            print(f"[ReplayBuffer] Loaded {len(self)} transitions from disk")
        except Exception as e:
            print(f"[ReplayBuffer] Load error: {e}")


# ─── Online PPO Trainer ──────────────────────────────────────────────────

class OnlinePPOTrainer:
    """
    Fine-tunes PPO on experience replay buffer.
    Uses lower learning rate and fewer steps than full training.
    """

    def __init__(self, production_model_path: str, config: Dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.production_model_path = Path(production_model_path)
        self.models_dir = Path(self.config["models_dir"])
        self.last_finetune_time = 0
        self.finetune_count = 0
        
        # Dynamically determine candidate_prefix from production model name
        model_name = self.production_model_path.stem  # e.g., "ppo_v10_btc_steps1000000"
        # Extract base name (before "_steps")
        if "_steps" in model_name:
            base_name = model_name.split("_steps")[0]  # e.g., "ppo_v10_btc"
        else:
            base_name = model_name.rsplit("_", 1)[0]  # fallback: remove last part
        self.config["candidate_prefix"] = f"{base_name}_online_"
        
    def should_finetune(self, buffer: ExperienceReplayBuffer) -> bool:
        """Check if it's time to fine-tune."""
        if len(buffer) < self.config["replay_min_for_train"]:
            return False
        
        interval_sec = self.config["finetune_interval_hours"] * 3600
        if time.time() - self.last_finetune_time < interval_sec:
            return False
        
        return True

    def finetune(self, buffer: ExperienceReplayBuffer,
                 env_factory=None) -> Optional[str]:
        """
        Fine-tune PPO on replay buffer. Returns path to candidate model, or None on failure.
        
        Args:
            buffer: Experience replay buffer with live transitions
            env_factory: callable that creates a Gymnasium env for PPO
        """
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        print(f"\n{'='*60}")
        print(f"  Online Fine-tune #{self.finetune_count + 1}")
        print(f"  Buffer: {len(buffer)} transitions")
        print(f"{'='*60}")
        
        try:
            # Create replay env wrapped for Gymnasium
            if env_factory:
                base_env = env_factory()
            else:
                base_env = ReplayEnv(buffer)
            
            # Wrap in DummyVecEnv (required by SB3)
            vec_env = DummyVecEnv([lambda: base_env])
            
            # Load production model
            print(f"[OnlineTrainer] Loading production model: {self.production_model_path}")
            model = PPO.load(
                str(self.production_model_path),
                env=vec_env,
            )
            
            # Override hyperparameters for fine-tuning
            model.learning_rate = self.config["finetune_lr"]
            model.n_epochs = self.config["finetune_n_epochs"]
            model.batch_size = self.config["finetune_batch_size"]
            
            # Fine-tune
            steps = self.config["finetune_steps"]
            print(f"[OnlineTrainer] Fine-tuning for {steps} steps (lr={self.config['finetune_lr']:.1e}, epochs={model.n_epochs}, batch={model.batch_size})...")
            
            model.learn(
                total_timesteps=steps,
                progress_bar=False,
            )
            
            # Save candidate model
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            candidate_name = f"{self.config['candidate_prefix']}{timestamp}.zip"
            candidate_path = self.models_dir / candidate_name
            self.models_dir.mkdir(parents=True, exist_ok=True)
            
            model.save(str(candidate_path))
            print(f"[OnlineTrainer] Candidate saved: {candidate_path}")
            
            # Cleanup
            self._cleanup_candidates()
            vec_env.close()
            
            self.last_finetune_time = time.time()
            self.finetune_count += 1
            
            return str(candidate_path)
            
        except Exception as e:
            print(f"[OnlineTrainer] Fine-tune failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _cleanup_candidates(self):
        """Remove old candidate models, keep last N."""
        candidates = sorted(
            self.models_dir.glob(f"{self.config['candidate_prefix']}*.zip"),
            key=lambda p: p.stat().st_mtime,
        )
        while len(candidates) > self.config["max_candidates"]:
            old = candidates.pop(0)
            old.unlink()
            print(f"[OnlineTrainer] Removed old candidate: {old.name}")


# ─── Replay Environment ──────────────────────────────────────────────────

class ReplayEnv(gym.Env):
    """
    A Gymnasium-compatible environment that replays transitions from the buffer.
    Each episode traverses the entire buffer to give PPO enough data to learn from.
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, buffer: ExperienceReplayBuffer):
        super().__init__()
        self.buffer = buffer
        self.current_idx = 0
        self.episode_length = len(buffer)  # one episode = full buffer pass
        self.n_features = len(buffer.observations[0]) if len(buffer) > 0 else 95
        
        # Define spaces (Gymnasium API)
        # Match the production model's observation space exactly
        self.observation_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(self.n_features,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)  # HOLD, BUY_UP, BUY_DOWN, SELL
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Start from beginning of buffer for each episode
        self.current_idx = 0
        obs = np.array(self.buffer.observations[0], dtype=np.float32)
        return obs, {}
    
    def step(self, action):
        # Reward from the transition at current_idx
        reward = float(self.buffer.rewards[self.current_idx])
        terminated = bool(self.buffer.dones[self.current_idx])
        self.current_idx += 1
        
        # Check if we've gone through the entire buffer
        if self.current_idx >= len(self.buffer):
            # Episode done — return last obs
            obs = np.array(self.buffer.next_observations[-1], dtype=np.float32)
            return obs, reward, True, False, {}
        
        obs = np.array(self.buffer.observations[self.current_idx], dtype=np.float32)
        return obs, reward, terminated, False, {}


# ─── A/B Validator ────────────────────────────────────────────────────────

class ABValidator:
    """
    Validates a candidate model against the production model on recent transitions.
    Candidate must be meaningfully better to be promoted.
    """

    def __init__(self, config: Dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.validation_history = []

    def validate(self, candidate_path: str, production_path: str,
                 buffer: ExperienceReplayBuffer) -> Dict:
        """
        Compare candidate vs production on recent buffer transitions.
        
        Returns dict with:
            passed: bool — candidate is better
            candidate_reward: float
            production_reward: float
            improvement_pct: float
        """
        from stable_baselines3 import PPO
        
        window = min(self.config["validation_window"], len(buffer))
        
        # Get recent transitions for validation
        obs, actions, rewards, next_obs, dones = buffer.get_recent(window)
        
        if len(obs) == 0:
            return {"passed": False, "reason": "no_data"}
        
        print(f"[ABValidator] Validating on {window} recent transitions...")
        
        try:
            candidate = PPO.load(candidate_path)
            production = PPO.load(production_path)
        except Exception as e:
            print(f"[ABValidator] Model load error: {e}")
            return {"passed": False, "reason": "load_error", "error": str(e)}
        
        # Evaluate both models: predict actions, compute would-be rewards
        cand_rewards = []
        prod_rewards = []
        
        for i in range(len(obs)):
            o = obs[i].reshape(1, -1)
            
            # Candidate action
            cand_action, _ = candidate.predict(o, deterministic=True)
            # Production action  
            prod_action, _ = production.predict(o, deterministic=True)
            
            # Reward = how well the model's action matches optimal
            # We use actual reward from buffer (from the action that was taken)
            # and adjust based on whether model would have taken same or better action
            
            # Simple metric: if model would HOLD when reward is negative → good
            # If model would ACT when reward is positive → good
            actual_reward = rewards[i]
            
            # For the action that was actually taken, reward is `actual_reward`
            # For other actions, we don't know the counterfactual
            # So we compare: would each model have taken the same action?
            # If yes, they get the same reward. If no, we penalize slightly.
            
            # Better approach: just evaluate model's predicted action quality
            # by running episodes in the replay env
            # For now: compare predicted action alignment with profitable actions
            
            if actual_reward > 0:
                # If actual trade was profitable:
                # candidate gets reward if it would take same action
                cand_r = actual_reward if cand_action == actions[i] else actual_reward * 0.3
                prod_r = actual_reward if prod_action == actions[i] else actual_reward * 0.3
            else:
                # If actual trade was unprofitable:
                # candidate gets reward if it would NOT take same action
                cand_r = 0 if cand_action != actions[i] else actual_reward
                prod_r = 0 if prod_action != actions[i] else actual_reward
            
            cand_rewards.append(cand_r)
            prod_rewards.append(prod_r)
        
        cand_avg = np.mean(cand_rewards)
        prod_avg = np.mean(prod_rewards)
        
        # Calculate Sharpe ratio (reward / std)
        cand_sharpe = cand_avg / (np.std(cand_rewards) + 1e-8)
        prod_sharpe = prod_avg / (np.std(prod_rewards) + 1e-8)
        
        # Calculate win rate
        cand_win_rate = np.mean([1 if r > 0 else 0 for r in cand_rewards])
        prod_win_rate = np.mean([1 if r > 0 else 0 for r in prod_rewards])
        
        # Combined score: 50% avg reward + 30% sharpe + 20% win rate
        cand_score = 0.5 * cand_avg + 0.3 * cand_sharpe + 0.2 * cand_win_rate
        prod_score = 0.5 * prod_avg + 0.3 * prod_sharpe + 0.2 * prod_win_rate
        
        improvement = (cand_score - prod_score) / (abs(prod_score) + 1e-8)
        
        min_improvement = self.config["validation_min_improvement"]
        passed = improvement >= min_improvement
        
        result = {
            "passed": passed,
            "candidate_path": candidate_path,
            "production_path": production_path,
            "candidate_avg_reward": float(cand_avg),
            "production_avg_reward": float(prod_avg),
            "improvement_pct": float(improvement * 100),
            "window_size": window,
            "min_improvement": float(min_improvement * 100),
            "timestamp": int(time.time()),
        }
        
        self.validation_history.append(result)
        
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"[ABValidator] {status}: candidate={cand_avg:.4f} vs production={prod_avg:.4f} (improvement: {improvement*100:+.1f}%)")
        
        return result


# ─── Model Swap Manager ──────────────────────────────────────────────────

class ModelSwapManager:
    """
    Safely swaps production model with validated candidate.
    Supports rollback on failure.
    """

    def __init__(self, config: Dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.models_dir = Path(self.config["models_dir"])
        self.current_model_path = None
        self.previous_model_path = None

    def swap(self, candidate_path: str, trader=None) -> bool:
        """
        Swap the trader's model with the candidate.
        
        Args:
            candidate_path: path to validated candidate model
            trader: RLTraderV2 instance to hot-swap
            
        Returns:
            True if swap succeeded
        """
        from stable_baselines3 import PPO
        
        try:
            # Backup current model path
            if trader:
                self.previous_model_path = str(trader.model_path) if hasattr(trader, 'model_path') else None
            
            # Load candidate
            print(f"[ModelSwap] Loading candidate: {candidate_path}")
            new_model = PPO.load(candidate_path)
            
            # Verify model works (predict on dummy obs)
            dummy = np.zeros(new_model.observation_space.shape, dtype=np.float32).reshape(1, -1)
            new_model.predict(dummy, deterministic=True)
            
            # Hot-swap in trader
            if trader and hasattr(trader, 'model'):
                old_model = trader.model
                trader.model = new_model
                if hasattr(trader, 'model_path'):
                    trader.model_path = candidate_path
                print(f"[ModelSwap] ✅ Model swapped successfully")
                print(f"[ModelSwap] New model: {Path(candidate_path).name}")
                return True
            else:
                print(f"[ModelSwap] ✅ Candidate model loaded (no trader to swap)")
                return True
                
        except Exception as e:
            print(f"[ModelSwap] ❌ Swap failed: {e}")
            
            # Rollback
            if self.config["rollback_on_failure"] and trader and self.previous_model_path:
                try:
                    old_model = PPO.load(self.previous_model_path)
                    trader.model = old_model
                    print(f"[ModelSwap] Rolled back to previous model")
                except:
                    print(f"[ModelSwap] Rollback also failed!")
            
            return False


# ─── Online Learning Controller ──────────────────────────────────────────

class OnlineLearningController:
    """
    Top-level controller that orchestrates the online learning loop.
    
    Usage:
        controller = OnlineLearningController(config)
        controller.initialize(production_model_path, trader)
        
        # On each trade close:
        controller.on_trade_closed(trade_result, obs_before, obs_after, action)
        
        # In main loop (periodic check):
        controller.check_and_finetune()
    """

    def __init__(self, config: Dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        self.buffer = ExperienceReplayBuffer(
            max_size=self.config["replay_max_size"],
            save_path=self.config["replay_save_path"],
        )
        self.trainer = OnlinePPOTrainer(
            production_model_path="",
            config=self.config,
        )
        self.validator = ABValidator(config=self.config)
        self.swapper = ModelSwapManager(config=self.config)
        
        self.trader = None
        self.initialized = False
        self.last_obs = None
        self.last_action = None

    def initialize(self, production_model_path: str, trader=None):
        """Initialize the controller with current model and trader."""
        self.trainer.production_model_path = Path(production_model_path)
        self.trader = trader
        
        # Load existing buffer
        self.buffer.load()
        
        self.initialized = True
        print(f"[OnlineLearning] Initialized")
        print(f"[OnlineLearning] Buffer: {len(self.buffer)} existing transitions")
        print(f"[OnlineLearning] Fine-tune interval: {self.config['finetune_interval_hours']}h")
        print(f"[OnlineLearning] Min buffer for train: {self.config['replay_min_for_train']}")

    def on_observation(self, obs: np.ndarray, action: int):
        """Store observation + action for later transition creation."""
        self.last_obs = obs.copy()
        self.last_action = action

    def on_trade_closed(self, trade_result: Dict, obs_after: np.ndarray):
        """
        Called when a trade closes. Creates replay transitions.
        
        Args:
            trade_result: dict from _close_position (with pnl, side, etc.)
            obs_after: observation after position close
        """
        if self.last_obs is None:
            return
        
        self.buffer.add_trade(
            trade_result=trade_result,
            obs_before=self.last_obs,
            obs_after=obs_after,
            action=self.last_action if self.last_action is not None else 0,
        )
        
        # Periodic save (every 10 trades)
        if self.buffer.total_stored % 10 == 0:
            self.buffer.save()
        
        stats = self.buffer.get_stats()
        print(f"[OnlineLearning] Trade stored | Buffer: {stats['size']} | Avg R: {stats.get('avg_reward', 0):.4f}")

    def check_and_finetune(self) -> Optional[str]:
        """
        Check if fine-tune is due, and if so, run the full pipeline:
        fine-tune → validate → swap
        
        Returns:
            Path to new model if swapped, else None
        """
        if not self.initialized:
            return None
        
        if not self.trainer.should_finetune(self.buffer):
            return None
        
        print(f"\n[OnlineLearning] 🔄 Fine-tune triggered!")
        
        # Step 1: Fine-tune
        candidate_path = self.trainer.finetune(self.buffer)
        if not candidate_path:
            print(f"[OnlineLearning] Fine-tune failed, skipping")
            return None
        
        # Step 2: Validate
        validation = self.validator.validate(
            candidate_path=candidate_path,
            production_path=str(self.trainer.production_model_path),
            buffer=self.buffer,
        )
        
        if not validation["passed"]:
            print(f"[OnlineLearning] Validation failed (improvement: {validation.get('improvement_pct', 0):+.1f}%)")
            print(f"[OnlineLearning] Keeping production model")
            return None
        
        # Step 3: Swap
        swapped = self.swapper.swap(candidate_path, self.trader)
        if swapped:
            # Update production model path for next fine-tune
            self.trainer.production_model_path = Path(candidate_path)
            print(f"[OnlineLearning] ✅ Model upgraded: {Path(candidate_path).name}")
            
            # Save buffer after successful swap
            self.buffer.save()
            
            return candidate_path
        
        return None

    def get_status(self) -> Dict:
        """Return current online learning status."""
        buffer_stats = self.buffer.get_stats()
        return {
            "initialized": self.initialized,
            "buffer_size": buffer_stats["size"],
            "buffer_total": buffer_stats["total_stored"],
            "buffer_avg_reward": buffer_stats.get("avg_reward", 0),
            "finetune_count": self.trainer.finetune_count,
            "last_finetune": datetime.fromtimestamp(
                self.trainer.last_finetune_time, tz=timezone.utc
            ).isoformat() if self.trainer.last_finetune_time else None,
            "next_finetune_in": max(0, 
                self.config["finetune_interval_hours"] * 3600 - 
                (time.time() - self.trainer.last_finetune_time)
            ) if self.trainer.last_finetune_time else "pending",
            "production_model": str(self.trainer.production_model_path),
            "validation_history": len(self.validator.validation_history),
        }


# ─── CLI for manual operations ────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Online Learning Controller")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--finetune", action="store_true", help="Force fine-tune")
    parser.add_argument("--validate", type=str, help="Validate candidate model path")
    parser.add_argument("--buffer-stats", action="store_true", help="Buffer statistics")
    parser.add_argument("--config", type=str, help="Config JSON file")
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG
    if args.config:
        with open(args.config) as f:
            config = {**config, **json.load(f)}
    
    controller = OnlineLearningController(config)
    controller.initialize(
        production_model_path=str(Path(config["models_dir"]) / config["production_model"]),
    )
    
    if args.status:
        print(json.dumps(controller.get_status(), indent=2))
    
    elif args.buffer_stats:
        print(json.dumps(controller.buffer.get_stats(), indent=2))
    
    elif args.finetune:
        result = controller.check_and_finetune()
        if result:
            print(f"Model upgraded: {result}")
        else:
            print("No model upgrade")
    
    elif args.validate:
        result = controller.validator.validate(
            candidate_path=args.validate,
            production_path=str(Path(config["models_dir"]) / config["production_model"]),
            buffer=controller.buffer,
        )
        print(json.dumps(result, indent=2))
