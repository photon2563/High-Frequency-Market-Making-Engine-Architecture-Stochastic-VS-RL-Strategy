import os
import sys
import torch
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Add parent directory to path to import simulator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulator.lob_env import LOBEnv

def train_and_export():
    print("Initializing Limit Order Book Environment...")
    # Wrap environment for stable-baselines3
    vec_env = make_vec_env(lambda: LOBEnv(), n_envs=4)

    print("Initializing Advantage Actor-Critic (A2C) Agent...")
    # Initialize A2C model with Continuous Action Space
    # Using a Multi-Layer Perceptron (MlpPolicy)
    model = A2C("MlpPolicy", vec_env, verbose=1, learning_rate=0.0007, ent_coef=0.01)

    print("Training A2C Agent for 50,000 timesteps...")
    # Train the model (Keep it short for the sake of the internship project demonstration)
    model.learn(total_timesteps=50000)
    
    # Save the Stable-Baselines3 model (for Python evaluation later)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'a2c_hfmm')
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    print("Exporting Actor network to TorchScript for C++ Inference Engine...")
    # Extract the PyTorch policy network (Actor) from the SB3 model
    # SB3's policy network takes observations and outputs action distribution parameters
    class ActorExporter(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.features_extractor = policy.features_extractor
            self.action_net = policy.action_net
            self.mlp_extractor = policy.mlp_extractor
            
        def forward(self, obs):
            # Forward pass to get deterministic action (mean of Gaussian)
            features = self.features_extractor(obs)
            latent_pi, _ = self.mlp_extractor(features)
            mean_actions = self.action_net(latent_pi)
            return mean_actions

    # Create exporter instance
    exporter = ActorExporter(model.policy)
    exporter.eval()
    
    # Trace the model with a dummy input tensor
    # Observation space is 5-dimensional: [Mid-price, Inventory, Time_rem, OFI, CumNotional]
    dummy_input = torch.randn(1, 5)
    
    with torch.no_grad():
        traced_actor = torch.jit.trace(exporter, dummy_input)
        
    # Save the traced model
    export_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'a2c_actor.pt')
    traced_actor.save(export_path)
    print(f"TorchScript model successfully exported to {export_path}")

if __name__ == "__main__":
    train_and_export()
