import torch
import torch.nn.functional as F
from env import GridWorld
from critic import Critic
from policy import Policy
from ops import print_metric

class PPO:
    def __init__(self, left = -100, right = 100, max_step=100, 
                 vf_lambda=0.2, max_grad_norm=0.5, 
                 entropy_lambda=0.2, clip_su = 0.2,
                 device='cuda'):
        
        self.env = GridWorld(left = left, right = right)
        self.policy = Policy(state_dim=1, action_dim=3, max_step=max_step).to(device)
        self.critic = Critic(state_dim=1, value_dim=1).to(device)
        self.device = device

        self.ppo_cfg = {
            "max_grad_norm": max_grad_norm,
            "vf_lambda": vf_lambda,
            "entropy_lambda": entropy_lambda,
            "clip_su": clip_su
        }

        self.key_metrics = ['epoch', 'reward/train', 'reward/val', 
                            'policy_l', 'value_l', 'loss', 'ratio', 
                            'entropy', 'lr']

        self.configure_optimizer()

    def configure_optimizer(self):
        self.parameters = list(self.policy.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.AdamW(self.parameters, lr=3e-4)  
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500)

    def train(self, mini_batch_size=8, val_data_size=64, epochs=100, ppo_inner_epoch=2):
        print(','.join(self.key_metrics))
        td_eval = self.env.reset(bs=val_data_size).to(self.device)
        for epoch in range(epochs):
            
            # train / reset each epoch
            td = self.env.reset(bs=mini_batch_size).to(self.device)
            with torch.no_grad():
                out = self.policy(td=td.clone(), env=self.env, decode_type='sampling')
            
            td['action'] = out['action']
            td['reward'] = out['reward'][..., None]
            td['log_probs'] = out['log_probs']
            
            # PPO updates
            for _ in range(ppo_inner_epoch):
                out = self.policy(td=td.clone(), env=self.env, actions=td['action'])
                value_pred = self.critic(td.clone())  # [batch, 1]
                ratio = torch.exp(out['log_probs'].sum(-1) - td['log_probs'].sum(-1))
                adv = td['reward'] - value_pred.detach()
                policy_loss = -torch.min(ratio * adv, 
                                        torch.clamp(ratio, 1 - self.ppo_cfg['clip_su'],
                                                    1 + self.ppo_cfg['clip_su']) * adv).mean()
                value_loss = F.huber_loss(value_pred, td['reward'])
                entropy = -(out['log_probs'] * torch.exp(out['log_probs'])).mean()
                loss = policy_loss + self.ppo_cfg['vf_lambda']*value_loss - self.ppo_cfg['entropy_lambda']*entropy
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters, self.ppo_cfg['max_grad_norm'])
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            self.scheduler.step()

            # eval
            with torch.no_grad():
                out = self.policy(td=td_eval.clone(), env=self.env)

            # print metrics
            metric = {
                'epoch': epoch,
                'reward/train': td['reward'].mean().item(),
                'reward/val': out['reward'].mean().item(),
                'policy_l': policy_loss.item(),
                'value_l': value_loss.item(),
                'loss': loss.item(),
                'ratio': ratio.mean().item(),
                'entropy': entropy.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            }
            print_metric(metric)

