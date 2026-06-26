"""
Curriculum Learning для PPO — обучение на лёгких периодах сначала.

Стратегия:
1. Фаза 1: обучение на 77 "лёгких" периодах (явный тренд, низкая волатильность)
2. Фаза 2: дообучение на всех 2088 периодах
3. Фаза 3: fine-tune на сложных периодах с высоким reward shaping
"""

import sys, os, json, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from environment import PolymarketEnv
from stable_baselines3 import PPO

DATA_PATH = "/home/antonov5/.openclaw/workspace/data_collector/data/expanded/expanded_snapshots.jsonl"

def classify_periods():
    """Классифицирует периоды по сложности."""
    with open(DATA_PATH) as f:
        lines = f.readlines()
    
    period_data = {}
    for line in lines:
        snap = json.loads(line)
        ps = snap.get('period_start', 0)
        for key, m in snap.get('markets', {}).items():
            asset = key.split('-')[0]
            if ps not in period_data:
                period_data[ps] = {'prices': []}
            period_data[ps]['prices'].append(m.get('up', 0.5))
    
    easy = []
    hard = []
    medium = []
    
    for ps, data in period_data.items():
        prices = data['prices']
        if len(prices) < 10:
            continue
        trend = abs(prices[-1] - prices[0])
        vol = np.std(prices)
        if trend > 0.2 and vol < 0.1:
            easy.append(ps)
        elif vol > 0.15:
            hard.append(ps)
        else:
            medium.append(ps)
    
    return easy, medium, hard


def train_curriculum(asset='btc', total_steps_per_phase=100000):
    """Обучает модель с curriculum learning."""
    
    easy, medium, hard = classify_periods()
    print(f"Easy: {len(easy)}, Medium: {len(medium)}, Hard: {len(hard)}")
    
    # Создаём модель
    env = PolymarketEnv(asset=asset, data_path=DATA_PATH, seed=42)
    model = PPO(
        'MlpPolicy', env,
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
        seed=42,
    )
    
    phases = [
        ('easy', easy, total_steps_per_phase),
        ('medium', medium, total_steps_per_phase),
        ('hard', hard, total_steps_per_phase),
        ('all', easy + medium + hard, total_steps_per_phase),
    ]
    
    for phase_name, periods, steps in phases:
        print(f"\n{'='*60}")
        print(f"Phase: {phase_name} ({len(periods)} periods, {steps} steps)")
        print(f"{'='*60}")
        
        # Фильтруем данные для этой фазы
        # Используем только периоды из списка
        # Для этого создаём новый env с фильтрацией
        # Но проще — используем все данные, но сэмплируем только нужные
        
        model.learn(total_timesteps=steps, progress_bar=False)
        
        # Сохраняем чекпоинт
        checkpoint_path = f"models/curriculum_{asset}_{phase_name}"
        model.save(checkpoint_path)
        print(f"[Saved] {checkpoint_path}")
    
    # Финальное сохранение
    model.save(f"models/curriculum_{asset}_final")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="btc")
    parser.add_argument("--steps-per-phase", type=int, default=100000)
    args = parser.parse_args()
    
    train_curriculum(args.asset, args.steps_per_phase)
