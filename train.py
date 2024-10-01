from ppo import PPO
import torch
import numpy as np

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cuda'
    rl = PPO(left = -500, right = 500, max_step=500, device = device)
    rl.train(mini_batch_size=8, val_data_size=64, epochs=10000, ppo_inner_epoch=2)