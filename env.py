import torch
from tensordict.tensordict import TensorDict

class GridWorld:
    def __init__(self, left=-100, right=100):
        self.left = left
        self.right = right
        self.map_action = torch.tensor([-1, 0, 1], dtype=torch.float32).cuda()

    def reset(self, bs):
        return TensorDict({
            'pos': torch.zeros((bs, 1), dtype=torch.float32),
            # 'pos': torch.randint(0, 50, (bs, 1), dtype=torch.float32),
            'done': torch.zeros((bs, 1), dtype=torch.bool)
        }, batch_size=bs)

    def step(self, td):
        pos = td['pos'] + self.map_action[td['current']]
        pos = torch.clamp(pos, min=self.left, max=self.right)
        done = (pos == self.right)
        reward = self.map_action[td['current']]
    
        td.update(
            {
                'reward': reward,
                'done': done,
                'pos': pos
            }
        )
        return td

    def get_reward(self, td, actions):
        return self.map_action[actions].sum(-1).float()
