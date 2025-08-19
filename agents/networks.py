"""
网络定义（策略网络、价值网络、集成Critic、集中式Critic、VOI估计器）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class EnsembleCritic(nn.Module):
    def __init__(self, input_dim, ensemble_size=3):
        super().__init__()
        self.critics = nn.ModuleList([MLP(input_dim, 1) for _ in range(ensemble_size)])

    def forward(self, x):
        values = torch.cat([c(x) for c in self.critics], dim=1)
        return values.mean(dim=1), values.std(dim=1)

class CentralizedCritic(nn.Module):
    """
    集中式Critic：Q_tot(s, a) - 用于计算团队后悔
    """
    def __init__(self, state_dim, action_dim, num_agents, hidden_dim=256):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 状态编码器 - 输入维度是 state_dim * num_agents
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim * num_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 动作编码器 - 输入维度是 action_dim * num_agents (one-hot编码后)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim * num_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 联合Q值网络
        self.q_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, states, actions):
        """
        Args:
            states: [batch_size, num_agents, state_dim]
            actions: [batch_size, num_agents] 或 [batch_size, num_agents, action_dim]
        """
        batch_size = states.shape[0]
        
        # 展平状态和动作
        states_flat = states.view(batch_size, self.num_agents * self.state_dim)  # [batch_size, num_agents * state_dim]
        
        if actions.dim() == 2:
            # 动作是离散索引，需要转换为one-hot
            actions_flat = F.one_hot(actions, num_classes=self.action_dim).float().view(batch_size, -1)
        else:
            # 动作已经是one-hot编码
            actions_flat = actions.view(batch_size, -1)
        
        # 编码
        state_features = self.state_encoder(states_flat)
        action_features = self.action_encoder(actions_flat)
        
        # 联合特征
        joint_features = torch.cat([state_features, action_features], dim=-1)
        
        # Q值
        q_value = self.q_net(joint_features)
        return q_value.squeeze(-1)

class VOIEstimator(nn.Module):
    """
    信息价值(VOI)估计器：E[U|m] - E[U|Ø] - cost(m)
    """
    def __init__(self, state_dim, message_dim, hidden_dim=128):
        super().__init__()
        
        # 有消息时的效用估计器
        self.utility_with_message = nn.Sequential(
            nn.Linear(state_dim + message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 无消息时的效用估计器
        self.utility_without_message = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 消息成本估计器
        self.message_cost_estimator = nn.Sequential(
            nn.Linear(message_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state, message=None):
        """
        Args:
            state: [batch_size, state_dim]
            message: [batch_size, message_dim] 或 None
        """
        if message is None:
            # 无消息情况
            utility_without = self.utility_without_message(state)
            return utility_without, torch.zeros_like(utility_without)
        else:
            # 有消息情况
            state_message = torch.cat([state, message], dim=-1)
            utility_with = self.utility_with_message(state_message)
            utility_without = self.utility_without_message(state)
            message_cost = self.message_cost_estimator(message)
            
            voi = utility_with - utility_without - message_cost
            return utility_with, voi

class CommunicationGate(nn.Module):
    """
    通信门控：决定是否发送/请求消息
    """
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state):
        """
        Args:
            state: [batch_size, state_dim]
        Returns:
            communication_prob: [batch_size, 1] - 通信概率
        """
        return self.gate_net(state)

class PolicyNetwork(nn.Module):
    """
    策略网络：输出动作概率分布
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=-1)
    
    def get_action_probs(self, x):
        """获取动作概率分布"""
        return self.forward(x)
    
    def sample_action(self, x):
        """采样动作"""
        probs = self.forward(x)
        action = torch.multinomial(probs, 1)
        return action.squeeze(-1), probs
