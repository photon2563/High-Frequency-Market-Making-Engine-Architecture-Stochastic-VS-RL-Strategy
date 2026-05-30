import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulator.lob_env import LOBEnv
from models.avellaneda_stoikov import AvellanedaStoikovModel

def evaluate_models():
    print("Evaluating Avellaneda-Stoikov (AS) Baseline...")
    env = LOBEnv(max_steps=500)
    as_model = AvellanedaStoikovModel()
    
    obs, _ = env.reset()
    as_pnl = []
    as_inventory = []
    
    done = False
    while not done:
        S, q, time_rem, ofi, cum_not = obs
        # AS model calculation
        ask_spread, bid_spread = as_model.get_quotes(S, int(q), time_rem)
        
        obs, reward, done, _, info = env.step([ask_spread, bid_spread])
        as_pnl.append(info['pnl'])
        as_inventory.append(info['inventory'])
        
    print(f"AS Final PnL: {as_pnl[-1]:.2f}")
    
    print("Evaluating A2C Agent...")
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'a2c_hfmm.zip')
    
    if not os.path.exists(model_path):
        print(f"A2C model not found at {model_path}. Please run train_rl.py first.")
        return
        
    a2c_model = A2C.load(model_path)
    obs, _ = env.reset()
    a2c_pnl = []
    a2c_inventory = []
    
    done = False
    while not done:
        # A2C Predict
        action, _states = a2c_model.predict(obs, deterministic=True)
        # Apply action
        obs, reward, done, _, info = env.step(action)
        a2c_pnl.append(info['pnl'])
        a2c_inventory.append(info['inventory'])
        
    print(f"A2C Final PnL: {a2c_pnl[-1]:.2f}")
    
    # Plotting
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(as_pnl, label='AS Model PnL', color='blue')
    plt.plot(a2c_pnl, label='A2C Model PnL', color='orange')
    plt.title('Cumulative PnL Comparison')
    plt.xlabel('Tick')
    plt.ylabel('PnL')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(as_inventory, label='AS Inventory', color='blue')
    plt.plot(a2c_inventory, label='A2C Inventory', color='orange')
    plt.title('Inventory Accumulation (Risk Management)')
    plt.xlabel('Tick')
    plt.ylabel('Inventory Position')
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'comparison_plot.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    evaluate_models()
