import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_layers, num_uavs):
        super().__init__()
        self.num_layers = num_layers
        self.num_uavs = num_uavs*2
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_layers * self.num_uavs)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        logits = self.fc_out(h)
        return logits.view(-1, self.num_layers, self.num_uavs)

class PPOAgent:
    def __init__(self, state_dim, hidden_dim, num_layers, num_uavs, lr=1e-4):
        self.policy = PolicyNetwork(state_dim, hidden_dim, num_layers, num_uavs)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = 0.99
        self.real_num_uavs = num_uavs
        self.clip_epsilon = 0.2
        self.update_epochs = 5

    def get_action(self, state):
        logits = self.policy(state.unsqueeze(0))[0]
        probs = F.softmax(logits, dim=-1)
        dists = [torch.distributions.Categorical(probs[i]) for i in range(probs.size(0))]
        indices = [dist.sample() for dist in dists]
        log_probs = [dists[i].log_prob(indices[i]) for i in range(len(dists))]

        action_onehot = torch.zeros(self.policy.num_layers, self.policy.num_uavs)
        action_idxs=[]
        action_uavs=[]
        for i, idx in enumerate(indices):
            # only idx < 5, meaning to allocate the layer to uav
            if idx < self.real_num_uavs:
                action_onehot[i, idx] = 1.0
                action_idxs.append(i)
                action_uavs.append(idx.item())
        action_flat = action_onehot.view(-1)

        return action_flat, torch.stack(log_probs).sum(), probs.detach(),action_idxs, action_uavs

    def compute_returns(self, rewards):
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns)

    def update(self, states, actions, log_probs_old, rewards):
        returns = self.compute_returns(rewards)
        print('returns: ', returns)
        # 避免 returns 只有一个元素时产生 NaN
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            returns = returns  # 不做标准化

        for _ in range(self.update_epochs):
            logits = self.policy(states)
            probs = F.softmax(logits, dim=-1)
            log_probs = []
            for i in range(states.size(0)):
                sample_logits = probs[i]
                sample_action = actions[i].view(self.policy.num_layers, self.policy.num_uavs)
                lp = 0
                for l in range(self.policy.num_layers):
                    print('l:',l)
                    try:
                        dist = torch.distributions.Categorical(sample_logits[l])
                    except Exception as e:
                        print(e)
                    idx = torch.argmax(sample_action[l])
                    lp += dist.log_prob(idx)
                log_probs.append(lp)
            log_probs = torch.stack(log_probs)

            ratios = torch.exp(log_probs - log_probs_old.detach())
            advantages = returns
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('finish update')
