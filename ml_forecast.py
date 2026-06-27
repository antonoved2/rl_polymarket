#!/usr/bin/env python3
"""
ML Forecast integration for RL Polymarket Bot.

Loads XGB+LGB ensemble model from btc_forecast, makes direction predictions,
and provides them as additional features for the RL environment.

Features added to RL observation:
  - ml_prob_up: probability of price going up (from ML ensemble)
  - ml_prob_down: 1 - ml_prob_up
  - ml_confidence: |prob_up - 0.5| * 2 (how confident the model is)
  - ml_prediction: 1 if up, 0 if down, 0.5 if neutral
  - ml_edge: model's edge estimate (prob_up - fair_price)
  - ml_signal_strength: combined signal strength
"""

import json
import os
import sys
import math
import time
import requests
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

WORKSPACE = Path("/home/antonov5/.openclaw/workspace")
BTC_FORECAST_MODELS = WORKSPACE / "btc_forecast" / "models"

# Try to import xgboost and lightgbm
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[MLForecast] xgboost not installed, using fallback")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[MLForecast] lightgbm not installed")


class MLForecastModel:
    """
    Load and run XGB+LGB ensemble for price direction prediction.
    """

    def __init__(self, horizon: int = 10, models_dir: str = None):
        self.horizon = horizon
        self.models_dir = Path(models_dir) if models_dir else BTC_FORECAST_MODELS

        self.xgb_model = None
        self.lgb_model = None
        self.selected_features = []
        self.feature_stats = {}  # mean/std for normalization

        self._load_models()
        self._load_feature_config()

    def _load_models(self):
        """Load XGB and LGB models from JSON."""
        xgb_path = self.models_dir / f"xgb_final_5m_h{self.horizon}.json"
        lgb_path = self.models_dir / f"lgb_final_5m_h{self.horizon}.json"

        if xgb_path.exists() and HAS_XGB:
            try:
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(str(xgb_path))
                print(f"[MLForecast] Loaded XGB model: {xgb_path.name}")
            except Exception as e:
                print(f"[MLForecast] Failed to load XGB: {e}")

        if lgb_path.exists() and HAS_LGB:
            try:
                self.lgb_model = lgb.Booster(model_file=str(lgb_path))
                print(f"[MLForecast] Loaded LGB model: {lgb_path.name}")
            except Exception as e:
                print(f"[MLForecast] Failed to load LGB: {e}")

    def _load_feature_config(self):
        """Load selected features and normalization stats."""
        # Meta file has accuracy and edge info
        meta_path = self.models_dir / f"meta_final_5m_h{self.horizon}.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
            print(f"[MLForecast] Meta: acc={self.meta.get('accuracy', 0):.4f}, "
                  f"edge={self.meta.get('edge', 0):.4f}")
        else:
            self.meta = {}

        # Selected features list
        sel_path = self.models_dir / f"selected_features_h{self.horizon}.json"
        if sel_path.exists():
            with open(sel_path) as f:
                sel = json.load(f)
            if isinstance(sel, dict):
                self.selected_features = sel.get("features", [])
            elif isinstance(sel, list):
                self.selected_features = sel
            print(f"[MLForecast] Selected features: {len(self.selected_features)}")

    def _fetch_binance_klines(self, symbol: str = "BTCUSDT", interval: str = "5m", limit: int = 100) -> list:
        """Fetch klines from Binance for feature computation."""
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        return []

    def _fetch_binance_ticker(self, symbol: str = "BTCUSDT") -> Dict:
        """Fetch current ticker data."""
        try:
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        return {}

    def compute_features(self, klines: list, ticker: Dict = None) -> Dict[str, float]:
        """
        Compute the 47 ML features from Binance klines.
        This replicates the feature engineering from btc_forecast/write_features.py
        """
        if len(klines) < 50:
            return {}

        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        timestamps = [float(k[0]) for k in klines]

        n = len(closes)
        last = closes[-1]
        if last <= 0:
            last = 1.0

        features = {}

        # === Timeframe returns ===
        if n >= 12:
            features["tf15m_returns"] = (closes[-1] - closes[-12]) / closes[-12]
        if n >= 60:
            features["tf1h_returns"] = (closes[-1] - closes[-60]) / closes[-60]
        if n >= 240:
            features["tf4h_returns"] = (closes[-1] - closes[-240]) / closes[-240]

        # === SMA ratios ===
        for period in [5, 10, 20, 50]:
            if n >= period:
                sma = sum(closes[-period:]) / period
                features[f"sma_ratio_{period}"] = sma / last

        # === SMA slopes ===
        for period in [5, 10, 20, 50]:
            if n >= period * 2:
                sma_now = sum(closes[-period:]) / period
                sma_prev = sum(closes[-2*period:-period]) / period
                features[f"sma_slope_{period}"] = (sma_now - sma_prev) / sma_prev if sma_prev > 0 else 0.0

        # === RSI ===
        def _rsi(data, period=14):
            if len(data) < period + 1:
                return 50.0
            gains, losses = [], []
            for i in range(len(data) - period, len(data)):
                diff = data[i] - data[i-1]
                gains.append(max(diff, 0))
                losses.append(max(-diff, 0))
            ag = sum(gains) / period
            al = sum(losses) / period
            if al == 0:
                return 100.0
            return 100.0 - (100.0 / (1.0 + ag / al))

        features["rsi_14"] = _rsi(closes, 14)
        if n >= 240:
            features["tf4h_rsi_14"] = _rsi(closes[-240:], 14)

        # === Distance from high/low ===
        for period in [20, 50]:
            if n >= period:
                hh = max(highs[-period:])
                ll = min(lows[-period:])
                features[f"dist_from_high_{period}"] = (hh - last) / last if last > 0 else 0.0
                features[f"dist_from_low_{period}"] = (last - ll) / last if last > 0 else 0.0

        # === Return statistics ===
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, n) if closes[i-1] > 0]
        for period in [3, 5, 10, 20]:
            if len(returns) >= period:
                recent = returns[-period:]
                features[f"min_return_{period}"] = min(recent)
                features[f"max_return_{period}"] = max(recent)
                features[f"std_return_{period}"] = np.std(recent)

        # === Skewness ===
        for period in [3, 10, 20]:
            if len(returns) >= period:
                recent = returns[-period:]
                mean_r = np.mean(recent)
                std_r = np.std(recent)
                if std_r > 0:
                    features[f"skew_{period}"] = np.mean(((recent - mean_r) / std_r) ** 3)
                else:
                    features[f"skew_{period}"] = 0.0

        # === MACD ===
        def _ema(data, period):
            if len(data) < period:
                return data[-1] if data else 0.0
            k = 2.0 / (period + 1)
            e = sum(data[:period]) / period
            for p in data[period:]:
                e = p * k + e * (1 - k)
            return e

        if n >= 35:
            ema12 = _ema(closes, 12)
            ema26 = _ema(closes, 26)
            macd_line = ema12 - ema26
            macd_series = []
            for i in range(26, n + 1):
                macd_series.append(_ema(closes[:i], 12) - _ema(closes[:i], 26))
            signal = _ema(macd_series, 9) if len(macd_series) >= 9 else macd_line
            features["macd_diff"] = macd_line / last
            features["macd_hist_change"] = (macd_line - signal) / last

        # EMA cross
        if n >= 26:
            features["tf1h_ema_12_26_cross"] = (_ema(closes, 12) - _ema(closes, 26)) / last
        if n >= 240:
            features["tf4h_ema_12_26_cross"] = (_ema(closes[-240:], 12) - _ema(closes[-240:], 26)) / last

        # === Bollinger Bands ===
        if n >= 20:
            sma20 = sum(closes[-20:]) / 20
            std20 = np.std(closes[-20:])
            bb_width = 2 * std20 / sma20 if sma20 > 0 else 0.0
            features["bb_width"] = bb_width
            features["tf1h_bb_position"] = (closes[-1] - (sma20 - 2*std20)) / (4*std20) if std20 > 0 else 0.5

        # === ATR ===
        if n >= 14:
            trs = []
            for i in range(max(1, n-14), n):
                tr = max(highs[i] - lows[i],
                         abs(highs[i] - closes[i-1]),
                         abs(lows[i] - closes[i-1]))
                trs.append(tr)
            features["atr_14"] = sum(trs) / len(trs) / last if trs and last > 0 else 0.0

        # === Volume features ===
        if n >= 20:
            vol_sma20 = sum(volumes[-20:]) / 20
            features["volume_ratio"] = volumes[-1] / vol_sma20 if vol_sma20 > 0 else 1.0

            # Volume trend
            if n >= 40:
                vol_prev = sum(volumes[-40:-20]) / 20
                features["volume_trend"] = (vol_sma20 - vol_prev) / vol_prev if vol_prev > 0 else 0.0

            # OBV trend
            obv = 0.0
            for i in range(max(1, n-20), n):
                if closes[i] > closes[i-1]:
                    obv += volumes[i]
                elif closes[i] < closes[i-1]:
                    obv -= volumes[i]
            features["obv_trend"] = obv / (vol_sma20 * 20) if vol_sma20 > 0 else 0.0

        if n >= 240:
            vol_4h = sum(volumes[-240:]) / 240
            features["tf4h_volume_ratio"] = volumes[-1] / vol_4h if vol_4h > 0 else 1.0

        # === Range ===
        if n >= 10:
            recent_high = max(highs[-10:])
            recent_low = min(lows[-10:])
            features["range"] = (recent_high - recent_low) / last if last > 0 else 0.0

        # === Acceleration ===
        if n >= 10:
            ret_5 = (closes[-1] - closes[-6]) / closes[-6] if closes[-6] > 0 else 0.0
            ret_10 = (closes[-1] - closes[-11]) / closes[-11] if closes[-11] > 0 else 0.0
            features["acceleration"] = ret_5 - ret_10

        # === Trend strength ===
        if n >= 60:
            returns_1h = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(n-60, n) if closes[i-1] > 0]
            if returns_1h:
                features["tf1h_trend_strength"] = sum(returns_1h) / np.std(returns_1h) if np.std(returns_1h) > 0 else 0.0
                # Historical volatility
                features["tf1h_hvol_20"] = np.std(returns_1h[-20:]) if len(returns_1h) >= 20 else np.std(returns_1h)

        if n >= 240:
            returns_4h = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(n-240, n) if closes[i-1] > 0]
            if returns_4h:
                features["tf4h_trend_strength"] = sum(returns_4h) / np.std(returns_4h) if np.std(returns_4h) > 0 else 0.0

        # === Hour/Day features ===
        if timestamps:
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(timestamps[-1] / 1000, tz=timezone.utc)
            hour = dt.hour
            dow = dt.weekday()
            features["hour_sin"] = math.sin(2 * math.pi * hour / 24)
            features["hour_cos"] = math.cos(2 * math.pi * hour / 24)
            features["dow_cos"] = math.cos(2 * math.pi * dow / 7)

        return features

    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Make prediction using XGB+LGB ensemble.
        Returns probability of price going up.
        """
        if not self.xgb_model and not self.lgb_model:
            return self._default_prediction()

        # Build feature vector in correct order
        feature_vector = []
        for feat_name in self.selected_features:
            val = features.get(feat_name, 0.0)
            feature_vector.append(float(val))

        X = np.array(feature_vector, dtype=np.float32).reshape(1, -1)

        # XGB prediction
        xgb_prob = None
        if self.xgb_model:
            try:
                dmatrix = xgb.DMatrix(X)
                xgb_pred = self.xgb_model.predict(dmatrix)
                # XGBoost returns log-odds or probability depending on objective
                if isinstance(xgb_pred, np.ndarray) and len(xgb_pred) > 0:
                    xgb_prob = float(xgb_pred[0])
                    # If log-odds, convert to probability
                    if xgb_prob < 0 or xgb_prob > 1:
                        xgb_prob = 1.0 / (1.0 + math.exp(-xgb_prob))
            except Exception as e:
                pass

        # LGB prediction
        lgb_prob = None
        if self.lgb_model:
            try:
                lgb_pred = self.lgb_model.predict(X)
                if isinstance(lgb_pred, np.ndarray) and len(lgb_pred) > 0:
                    lgb_prob = float(lgb_pred[0])
                    if lgb_prob < 0 or lgb_prob > 1:
                        lgb_prob = 1.0 / (1.0 + math.exp(-lgb_prob))
            except Exception as e:
                pass

        # Ensemble (average)
        probs = [p for p in [xgb_prob, lgb_prob] if p is not None]
        if not probs:
            return self._default_prediction()

        avg_prob = sum(probs) / len(probs)

        # Calibration: the model tends to be overconfident
        # Shrink towards 0.5 slightly
        calibrated = 0.5 + (avg_prob - 0.5) * 0.9

        return {
            "ml_prob_up": calibrated,
            "ml_prob_down": 1.0 - calibrated,
            "ml_confidence": abs(calibrated - 0.5) * 2.0,
            "ml_prediction": 1.0 if calibrated > 0.5 else 0.0,
            "ml_edge": calibrated - 0.5,
            "ml_signal_strength": abs(calibrated - 0.5) * 2.0 * (1.0 if calibrated > 0.5 else -1.0),
            "ml_raw_xgb": xgb_prob if xgb_prob is not None else 0.5,
            "ml_raw_lgb": lgb_prob if lgb_prob is not None else 0.5,
        }

    def predict_from_binance(self, symbol: str = "BTCUSDT") -> Dict[str, float]:
        """Fetch data from Binance and make prediction."""
        klines = self._fetch_binance_klines(symbol, limit=100)
        if not klines:
            return self._default_prediction()

        ticker = self._fetch_binance_ticker(symbol)
        features = self.compute_features(klines, ticker)

        if not features:
            return self._default_prediction()

        result = self.predict(features)
        result["ml_features_count"] = len(features)
        return result

    def _default_prediction(self) -> Dict[str, float]:
        """Return neutral prediction when model unavailable."""
        return {
            "ml_prob_up": 0.5,
            "ml_prob_down": 0.5,
            "ml_confidence": 0.0,
            "ml_prediction": 0.5,
            "ml_edge": 0.0,
            "ml_signal_strength": 0.0,
            "ml_raw_xgb": 0.5,
            "ml_raw_lgb": 0.5,
            "ml_features_count": 0,
        }


# ═══════════════════════════════════════════════════════════════════
# Singleton instance for use in RL environment
# ═══════════════════════════════════════════════════════════════════

_instance: Optional[MLForecastModel] = None


def get_ml_model() -> MLForecastModel:
    """Get or create singleton ML model instance."""
    global _instance
    if _instance is None:
        _instance = MLForecastModel(horizon=10)
    return _instance


def get_ml_features() -> Dict[str, float]:
    """Get ML prediction features for current market state."""
    model = get_ml_model()
    return model.predict_from_binance()


if __name__ == "__main__":
    print("=" * 60)
    print("  ML Forecast Integration Test")
    print("=" * 60)

    model = MLForecastModel(horizon=10)

    print(f"\nSelected features: {len(model.selected_features)}")
    print(f"XGB loaded: {model.xgb_model is not None}")
    print(f"LGB loaded: {model.lgb_model is not None}")

    print("\nFetching Binance data and predicting...")
    result = model.predict_from_binance("BTCUSDT")

    print(f"\nPrediction results:")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
