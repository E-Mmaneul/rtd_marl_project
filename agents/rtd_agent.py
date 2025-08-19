"""
RTD智能体（整合策略网络、后悔模块、延迟队列、通信机制）
增强版：支持策略非平稳性自适应、个体+团队后悔混合、合作/竞争检测
"""
import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from .networks import MLP, EnsembleCritic, CentralizedCritic, VOIEstimator, CommunicationGate, PolicyNetwork
from .regret_module import RegretModule
from utils.delay_queue import DelayQueue

class RTDAgent:
    def __init__(self, obs_dim, action_dim, config, agent_id):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cfg = config
        
        # 基础网络
        self.policy_net = PolicyNetwork(obs_dim, action_dim)
        self.critic = EnsembleCritic(obs_dim, config["rtd"]["ensemble_size"])
        
        # 集中式critic（用于团队后悔计算）
        # 注意：action_dim应该是one-hot编码后的维度，即5（支持0-4的动作）
        self.centralized_critic = CentralizedCritic(
            state_dim=obs_dim,  # 2
            action_dim=5,  # one-hot编码后的动作维度（0-4）
            num_agents=config["env"]["num_agents"]  # 2
        )
        
        # 后悔模块
        self.regret_module = RegretModule(config, agent_id)
        
        # 延迟队列
        self.delay_queue = DelayQueue(config["rtd"]["delay_queue_size"])
        
        # 通信相关
        self.voi_estimator = VOIEstimator(obs_dim, message_dim=16)
        self.communication_gate = CommunicationGate(obs_dim)
        self.communication_budget = config["rtd"]["communication_budget"]
        self.message_cost = config["rtd"]["message_cost"]
        self.voi_threshold = config["rtd"]["voi_threshold"]
        
        # 延迟域相关
        self.delay_domain_steps = config["rtd"]["delay_domain_steps"]
        self.guardian_action_prob = config["rtd"]["guardian_action_prob"]
        self.in_delay_domain = False
        self.delay_domain_counter = 0
        
        # 历史记录
        self.action_history = deque(maxlen=50)
        self.reward_history = deque(maxlen=50)
        self.policy_history = deque(maxlen=20)
        
        # 探索参数
        self.epsilon = 0.1
        
    def act(self, obs, other_agents_obs=None, other_agents_actions=None):
        """
        智能体动作选择
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # [1, 2]
        
        # 探索
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            return action, "explore", 0.0  # 探索时后悔为0
        
        # 获取动作概率分布
        action_probs = self.policy_net.get_action_probs(obs_t)
        action, _ = self.policy_net.sample_action(obs_t)
        action = action.item()
        
        # 记录策略历史
        self.policy_history.append(action_probs.detach())
        
        # 计算后悔
        blended_regret = self._compute_blended_regret(
            obs_t, action, other_agents_obs, other_agents_actions
        )
        
        # 三支决策
        decision = self.regret_module.decide(
            blended_regret=blended_regret,
            state=obs_t,
            action=action,
            other_actions=other_agents_actions,
            joint_action=[action] + (other_agents_actions or []),
            critic=self.centralized_critic,
            rewards=list(self.reward_history)[-5:] if self.reward_history else None,
            current_policy=action_probs.detach()
        )
        
        # 处理决策
        if decision == "accept":
            final_action = action
        elif decision == "delay":
            # 进入延迟域
            self.in_delay_domain = True
            self.delay_domain_counter = 0
            self.delay_queue.add(action)
            final_action = self._get_guardian_action()
        else:  # reject
            final_action = self._get_safe_action()
        
        # 记录历史
        self.action_history.append(action)
        
        # 返回后悔值（转换为Python标量）
        regret_value = blended_regret.item() if isinstance(blended_regret, torch.Tensor) else float(blended_regret)
        
        return final_action, decision, regret_value
    
    def _compute_blended_regret(self, obs_t, action, other_agents_obs, other_agents_actions):
        """
        计算混合后悔：保守优化版本 - 只计算个体后悔
        """
        # 只计算个体后悔，大幅减少前向传播
        individual_regret = self.regret_module.compute_individual_regret(
            obs_t, action, other_agents_actions, self.centralized_critic
        )
        
        # 直接返回个体后悔，跳过团队后悔计算
        return individual_regret
    
    def _get_guardian_action(self):
        """
        获取守护动作（安全动作）
        """
        if random.random() < self.guardian_action_prob:
            # 保守动作：向目标方向移动
            return random.choice([0, 1, 2, 3])  # 随机方向
        else:
            # 等待动作
            return 4  # 等待
    
    def _get_safe_action(self):
        """
        获取安全动作（拒绝时的备选）
        """
        return random.randint(0, self.action_dim - 1)
    
    def update_reward(self, reward):
        """
        更新奖励历史
        """
        self.reward_history.append(reward)
    
    def should_communicate(self, obs):
        """
        决定是否通信 - 保守优化版本：禁用通信机制
        """
        # 直接禁用通信，减少计算开销
        return False, None
    
    def process_delay_domain(self):
        """
        处理延迟域中的动作 - 保守优化版本：简化延迟域逻辑
        """
        if not self.in_delay_domain:
            return None
        
        # 简化：直接处理延迟队列，减少状态跟踪
        if self.delay_queue.queue:
            delayed_action = self.delay_queue.pop()
            self.in_delay_domain = False
            self.delay_domain_counter = 0
            return delayed_action
        
        return None
    
    def get_agent_info(self):
        """
        获取智能体信息用于调试
        """
        regret_info = self.regret_module.get_regret_info()
        
        return {
            'agent_id': self.agent_id,
            'in_delay_domain': self.in_delay_domain,
            'delay_domain_counter': self.delay_domain_counter,
            'communication_budget': self.communication_budget,
            'delay_queue_size': len(self.delay_queue.queue),
            'action_history_size': len(self.action_history),
            'reward_history_size': len(self.reward_history),
            **regret_info
        }
    
    def evaluate_action(self, obs):
        """
        评估动作（兼容性方法）
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        _, std_v = self.critic(obs_t)
        regret = std_v.item()
        return self.regret_module.decide(regret)
