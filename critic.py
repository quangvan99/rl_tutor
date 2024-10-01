import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim=1, value_dim=1, hidden_dim=128):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),

            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.GELU(),

            nn.Linear(hidden_dim * 4, value_dim)
        )
    def forward(self, td):
        valued = self.critic(td['pos'])
        return valued