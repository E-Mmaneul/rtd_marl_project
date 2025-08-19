"""
基线算法智能体实现
用于与RTD增强版进行性能对比
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class BaselineAgent:
    """基线智能体基类"""
    def __init__(self, obs_dim, action_dim, config, agent_id):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cfg = config
        self.epsilon = 0.1
        
    def act(self, obs, other_agents_obs=None, other_agents_actions=None):
        raise NotImplementedError
        
    def update_reward(self, reward):
        pass

class QMIXAgent(BaselineAgent):
    """QMIX智能体 - 价值分解方法"""
    def __init__(self, obs_dim, action_dim, config, agent_id):
        super().__init__(obs_dim, action_dim, config, agent_id)
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.target_q_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
    def act(self, obs, other_agents_obs=None, other_agents_actions=None):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1), "explore", 0.0
            
        with torch.no_grad():
            q_values = self.q_net(obs_t)
            action = q_values.argmax().item()
            
        # 计算基于Q值差异的遗憾
        max_q = q_values.max().item()
        min_q = q_values.min().item()
        q_range = max_q - min_q
        regret = max(0, q_range * 0.1)  # 基于Q值范围的遗憾估计
            
        return action, "greedy", regret

class VDNAgent(BaselineAgent):
    """VDN智能体 - 价值分解网络"""
    def __init__(self, obs_dim, action_dim, config, agent_id):
        super().__init__(obs_dim, action_dim, config, agent_id)
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def act(self, obs, other_agents_obs=None, other_agents_actions=None):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1), "explore", 0.0
            
        with torch.no_grad():
            q_values = self.q_net(obs_t)
            action = q_values.argmax().item()
            
        # 计算基于Q值差异的遗憾
        max_q = q_values.max().item()
        min_q = q_values.min().item()
        q_range = max_q - min_q
        regret = max(0, q_range * 0.15)  # VDN的遗憾计算
            
        return action, "greedy", regret

class COMAAgent(BaselineAgent):
    """COMA智能体 - 反事实多智能体策略梯度"""
    def __init__(self, obs_dim, action_dim, config, agent_id):
        super().__init__(obs_dim, action_dim, config, agent_id)
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim * config["env"]["num_agents"], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def act(self, obs, other_agents_obs=None, other_agents_actions=None):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1), "explore", 0.0
            
        with torch.no_grad():
            action_probs = F.softmax(self.policy_net(obs_t), dim=-1)
            action = torch.multinomial(action_probs, 1).item()
            
        # 计算基于策略概率差异的遗憾
        max_prob = action_probs.max().item()
        min_prob = action_probs.min().item()
        prob_range = max_prob - min_prob
        regret = max(0, prob_range * 0.2)  # COMA的遗憾计算
            
        return action, "policy", regret

class MADDPGAgent(BaselineAgent):
    """MADDPG智能体 - 多智能体深度确定性策略梯度"""
    def __init__(self, obs_dim, action_dim, config, agent_id):
        super().__init__(obs_dim, action_dim, config, agent_id)
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim * config["env"]["num_agents"], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def act(self, obs, other_agents_obs=None, other_agents_actions=None):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1), "explore", 0.0
            
        with torch.no_grad():
            action_probs = F.softmax(self.policy_net(obs_t), dim=-1)
            action = torch.multinomial(action_probs, 1).item()
            
        # 计算基于策略概率差异的遗憾
        max_prob = action_probs.max().item()
        min_prob = action_probs.min().item()
        prob_range = max_prob - min_prob
        regret = max(0, prob_range * 0.18)  # MADDPG的遗憾计算
            
        return action, "policy", regret

class IQLAgent(BaselineAgent):
    """IQL智能体 - 独立Q学习"""
    def __init__(self, obs_dim, action_dim, config, agent_id):
        super().__init__(obs_dim, action_dim, config, agent_id)
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def act(self, obs, other_agents_obs=None, other_agents_actions=None):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1), "explore", 0.0
            
        with torch.no_grad():
            q_values = self.q_net(obs_t)
            action = q_values.argmax().item()
            
        # 计算基于Q值差异的遗憾
        max_q = q_values.max().item()
        min_q = q_values.min().item()
        q_range = max_q - min_q
        regret = max(0, q_range * 0.12)  # IQL的遗憾计算
            
        return action, "greedy", regret

class TarMACAgent(BaselineAgent):
    """TarMAC智能体 - 带通信的多智能体"""
    def __init__(self, obs_dim, action_dim, config, agent_id):
        super().__init__(obs_dim, action_dim, config, agent_id)
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim + 16, 128),  # 16是消息维度
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.message_net = nn.Sequential(
            nn.Linear(obs_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        self.communication_budget = config["rtd"]["communication_budget"]
        self.message_cost = config["rtd"]["message_cost"]
        
    def act(self, obs, other_agents_obs=None, other_agents_actions=None):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1), "explore", 0.0
            
        # 生成消息
        message = self.message_net(obs_t)
        
        # 决定是否发送消息（简化版本）
        should_send = random.random() < 0.5 and self.communication_budget > 0
        
        if should_send:
            self.communication_budget -= self.message_cost
            obs_with_message = torch.cat([obs_t, message], dim=-1)
        else:
            obs_with_message = torch.cat([obs_t, torch.zeros(1, 16)], dim=-1)
            
        with torch.no_grad():
            action_probs = F.softmax(self.policy_net(obs_with_message), dim=-1)
            action = torch.multinomial(action_probs, 1).item()
            
        return action, "policy", 0.0

class LOLAAgent(BaselineAgent):
    """LOLA智能体 - 学习对手建模"""
    def __init__(self, obs_dim, action_dim, config, agent_id):
        super().__init__(obs_dim, action_dim, config, agent_id)
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.opponent_model = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def act(self, obs, other_agents_obs=None, other_agents_actions=None):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1), "explore", 0.0
            
        with torch.no_grad():
            action_probs = F.softmax(self.policy_net(obs_t), dim=-1)
            action = torch.multinomial(action_probs, 1).item()
            
        return action, "policy", 0.0

# 基线算法工厂
def create_baseline_agent(algorithm_name, obs_dim, action_dim, config, agent_id):
    """创建基线算法智能体"""
    algorithms = {
        "QMIX": QMIXAgent,
        "VDN": VDNAgent,
        "COMA": COMAAgent,
        "MADDPG": MADDPGAgent,
        "IQL": IQLAgent,
        "TarMAC": TarMACAgent,
        "LOLA": LOLAAgent
    }
    
    if algorithm_name not in algorithms:
        raise ValueError(f"不支持的算法: {algorithm_name}")
        
    return algorithms[algorithm_name](obs_dim, action_dim, config, agent_id)
