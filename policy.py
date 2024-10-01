import torch
import torch.nn as nn
import torch.nn.functional as F
from ops import gather_by_index

class Decoder(nn.Module):
    def __init__(self, state_dim=1, action_dim=3, hidden_dim=128):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),

            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.GELU(),

            nn.Linear(hidden_dim * 4, action_dim)
        )
    
    def forward(self, td):
        act_probs = self.decode(td['pos'])
        log_probs = F.log_softmax(act_probs, dim=-1)
        return log_probs

class Policy(nn.Module):
    def __init__(self, state_dim = 1, action_dim = 3, max_step=100):
        super(Policy, self).__init__()
        self.max_step = max_step
        self.decode_step = Decoder(state_dim = state_dim, action_dim = action_dim)

    def forward(self, td, env, decode_type=None, actions=None):
        collect_actions = []
        log_probs = []
        rewards = []
        step = 0
        while not td["done"].all():
            log_prob = self.decode_step(td)
            if actions is not None:
                td['current'] = actions[:, step:step+1]
            elif decode_type == 'sampling':
                td['current'] = torch.multinomial(log_prob.exp(), 1)
            else:
                td['current'] = torch.argmax(log_prob.exp(), dim=-1, keepdim=True)
                # print(decode_type, log_prob.exp())
                

            log_probs.append(gather_by_index(log_prob, td['current']))
            collect_actions.append(td['current'])
            td = env.step(td)

            rewards.append(td['reward'])

            step += 1
            if step >= self.max_step:
                break
        log_probs = torch.stack(log_probs, dim=1)
        collect_actions = torch.stack(collect_actions, dim=1).view(-1, step)
        # rewards = torch.concat(rewards, dim=-1)
        # gamma = torch.pow(0.99, torch.arange(rewards.shape[-1]-1, -1, -1))[None, ... ].cuda()
        # gamma = torch.pow(0.99, torch.arange(rewards.shape[-1]))[None, ... ].cuda()
        # rewards = (rewards * gamma * log_probs.exp()).sum(-1)
        
        rewards = torch.concat(rewards, dim=-1).sum(-1)
        return {
            'action': collect_actions,
            'log_probs': log_probs,
            'reward': rewards,
        }