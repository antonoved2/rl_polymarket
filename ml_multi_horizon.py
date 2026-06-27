#!/usr/bin/env python3
"""
Multi-Horizon ML Forecast for RL Polymarket Bot.

Loads XGB+LGB models for multiple horizons (h=1,3,5,10) and provides
direction predictions at each horizon as RL features.

Additional features (12 total, per horizon):
  - ml_h{N}_prob_up: probability of up at horizon N
  - ml_h{N}_confidence: confidence at horizon N
  - ml_h{N}_edge: edge at horizon N

Total new features: 4 horizons × 3 features = 12
"""

import json
import os
import sys
import math
import time
import requests
import numpy as np
from pathlib import Path
from typing import Dict, Optional

WORKSPACE = Path("/home/antonov5/.openclaw/workspace")
BTC_FORECAST_MODELS = WORKSPACE / "btc_forecast" / "models"

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class MultiHorizonML:
    """
    Multi-horizon ML forecast using XGB+LGB ensemble.
    """

    def __init__(self, horizons: list = [1, 3, 5, 10], models_dir: str = None):
        self.horizons = horizons
        self.models_dir = Path(models_dir) if models_dir else BTC_FORECAST_MODELS

        # Models per horizon: {h: {'xgb': Booster, 'lgb': Booster, 'features': [...]}}
        self.models = {}
        self._load_all_models()

    def _load_all_models(self):
        """Load all XGB and LGB models for each horizon."""
        for h in self.horizons:
            self.models[h] = {'xgb': None, 'lgb': None, 'features': []}

            # XGB
            xgb_path = self.models_dir / f"xgb_final_5m_h{h}.json"
            if xgb_path.exists() and HAS_XGB:
                try:
                    booster = xgb.Booster()
                    booster.load_model(str(xgb_path))
                    self.models[h]['xgb'] = booster
                except Exception as e:
                    pass

            # LGB
            lgb_path = self.models_dir / f"lgb_final_5m_h{h}.json"
            if lgb_path.exists() and HAS_LGB:
                try:
                    booster = lgb.Booster(model_file=str(lgb_path))
                    self.models[h]['lgb'] = booster
                except Exception as e:
                    pass

            # Feature list
            sel_path = self.models_dir / f"selected_features_h{h}.json"
            if sel_path.exists():
                with open(sel_path) as f:
                    sel = json.load(f)
                if isinstance(sel, dict):
                    self.models[h]['features'] = sel.get("features", [])
                elif isinstance(sel, list):
                    self.models[h]['features'] = sel

        # Summary
        for h in self.horizons:
            m = self.models[h]
            status = f"XGB={'✓' if m['xgb'] else '✗'} LGB={'✓' if m['lgb'] else '✗'} feat={len(m['features'])}"
            print(f"  ML h={h}: {status}")

    def predict(self, features: Dict[str, float], horizon: int) -> Dict[str, float]:
        """Make prediction for a specific horizon."""
        m = self.models.get(horizon)
        if not m or (not m['xgb'] and not m['lgb']):
            return self._default_prediction(horizon)

        # Build feature vector
        feature_vector = []
        for feat_name in m['features']:
            val = features.get(feat_name, 0.0)
            feature_vector.append(float(val))

        X = np.array(feature_vector, dtype=np.float32).reshape(1, -1)

        # XGB
        xgb_prob = None
        if m['xgb']:
            try:
                dmatrix = xgb.DMatrix(X)
                pred = m['xgb'].predict(dmatrix)
                if isinstance(pred, np.ndarray) and len(pred) > 0:
                    xgb_prob = float(pred[0])
                    if xgb_prob < 0 or xgb_prob > 1:
                        xgb_prob = 1.0 / (1.0 + math.exp(-xgb_prob))
            except:
                pass

        # LGB
        lgb_prob = None
        if m['lgb']:
            try:
                pred = m['lgb'].predict(X)
                if isinstance(pred, np.ndarray) and len(pred) > 0:
                    lgb_prob = float(pred[0])
                    if lgb_prob < 0 or lgb_prob > 1:
                        lgb_prob = 1.0 / (1.0 + math.exp(-lgb_prob))
            except:
                pass

        # Ensemble
        probs = [p for p in [xgb_prob, lgb_prob] if p is not None]
        if not probs:
            return self._default_prediction(horizon)

        avg_prob = sum(probs) / len(probs)
        calibrated = 0.5 + (avg_prob - 0.5) * 0.9

        return {
            f"ml_h{horizon}_prob_up": calibrated,
            f"ml_h{horizon}_confidence": abs(calibrated - 0.5) * 2.0,
            f"ml_h{horizon}_edge": calibrated - 0.5,
        }

    def predict_all(self, features: Dict[str, float]) -> Dict[str, float]:
        """Make predictions for all horizons."""
        result = {}
        for h in self.horizons:
            pred = self.predict(features, h)
            result.update(pred)
        return result

    def predict_from_ta(self, ta_features: Dict[str, float]) -> Dict[str, float]:
        """
        Approximate ML prediction from TA features.
        Uses available TA features as proxy for the full 47 ML features.
        """
        # Compute a composite signal from TA
        rsi = ta_features.get("rsi", 0.5)
        macd_hist = ta_features.get("macd_hist", 0.0)
        bb_pct_b = ta_features.get("bb_pct_b", 0.5)
        momentum_5 = ta_features.get("momentum_5", 0.0)
        momentum_10 = ta_features.get("momentum_10", 0.0)
        ma_cross_5_20 = ta_features.get("ma_cross_5_20", 0.0)
        vol_ratio = ta_features.get("vol_ratio", 1.0)
        stoch_k = ta_features.get("stoch_k", 0.5)
        atr_pct = ta_features.get("atr_pct", 0.01)
        price_vs_sma20 = ta_features.get("price_vs_sma20", 0.0)

        # Composite score
        score = 0.0
        score += -(rsi - 0.5) * 0.4  # RSI signal (inverted)
        score += macd_hist * 2.0  # MACD
        score += -(bb_pct_b - 0.5) * 0.2  # BB (inverted)
        score += momentum_5 * 1.0 + momentum_10 * 0.5  # Momentum
        score += ma_cross_5_20 * 5.0  # MA cross
        score += -(stoch_k - 0.5) * 0.1  # Stochastic
        score += price_vs_sma20 * 2.0  # Price vs MA
        score *= min(vol_ratio, 2.0) ** 0.3  # Volume confirmation

        # Mean reversion for extreme RSI
        if rsi > 0.8:
            score -= 0.2
        elif rsi < 0.2:
            score += 0.2

        score = np.clip(score, -1.0, 1.0)
        prob_up = np.clip(0.5 + score * 0.3, 0.1, 0.9)

        result = {}
        for h in self.horizons:
            # Scale confidence by horizon (shorter = more confident)
            h_factor = {1: 1.2, 3: 1.1, 5: 1.0, 10: 0.9, 20: 0.8}.get(h, 1.0)
            h_prob = np.clip(0.5 + (prob_up - 0.5) * h_factor, 0.05, 0.95)
            result[f"ml_h{h}_prob_up"] = float(h_prob)
            result[f"ml_h{h}_confidence"] = float(abs(h_prob - 0.5) * 2.0)
            result[f"ml_h{h}_edge"] = float(h_prob - 0.5)

        return result

    def _default_prediction(self, horizon: int) -> Dict[str, float]:
        return {
            f"ml_h{horizon}_prob_up": 0.5,
            f"ml_h{horizon}_confidence": 0.0,
            f"ml_h{horizon}_edge": 0.0,
        }


# Singleton
_instance: Optional[MultiHorizonML] = None


def get_multi_horizon_ml() -> MultiHorizonML:
    global _instance
    if _instance is None:
        _instance = MultiHorizonML(horizons=[1, 3, 5, 10])
    return _instance


if __name__ == "__main__":
    print("=" * 60)
    print("  Multi-Horizon ML Test")
    print("=" * 60)

    ml = MultiHorizonML(horizons=[1, 3, 5, 10])

    # Test with sample TA features
    sample_ta = {
        "rsi": 0.65,
        "macd_hist": 0.001,
        "bb_pct_b": 0.7,
        "momentum_5": 0.01,
        "momentum_10": 0.005,
        "ma_cross_5_20": 0.002,
        "vol_ratio": 1.5,
        "stoch_k": 0.6,
        "atr_pct": 0.02,
        "price_vs_sma20": 0.01,
    }

    print("\nPrediction from TA features:")
    result = ml.predict_from_ta(sample_ta)
    for k, v in sorted(result.items()):
        print(f"  {k}: {v:.4f}")
