"""
消融研究智能体实现
用于验证RTD增强版各个创新点的有效性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from .rtd_agent import RTDAgent
from .networks import PolicyNetwork, EnsembleCritic, CentralizedCritic

class AblationAgent(RTDAgent):
    """消融研究智能体基类"""
    def __init__(self, obs_dim, action_dim, config, agent_id, ablation_type):
        super().__init__(obs_dim, action_dim, config, agent_id)
        self.ablation_type = ablation_type
        self._apply_ablation()
        
    def _apply_ablation(self):
        """应用消融设置"""
        pass

class NoAdaptiveAgent(AblationAgent):
    """无自适应机制 - 固定λ参数"""
    def __init__(self, obs_dim, action_dim, config, agent_id):
        super().__init__(obs_dim, action_dim, config, agent_id, "no_adaptive")
        
    def _apply_ablation(self):
        """固定λ参数，禁用策略非平稳性自适应"""
        # 固定lambda_t为1.0，不进行动态调整
        self.regret_module.lambda_t = 1.0
        self.regret_module.alpha_t = self.regret_module.alpha_base
        self.regret_module.beta_t = self.regret_module.beta_base
        
    def act(self, obs, other_agents_obs=None, other_agents_actions=None):
        """重写act方法，禁用自适应更新"""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            return action, "explore", 0.0
        
        action_probs = self.policy_net.get_action_probs(obs_t)
        action, _ = self.policy_net.sample_action(obs_t)
        action = action.item()
        
        # 不更新策略历史（禁用自适应）
        # self.policy_history.append(action_probs.detach())
        
        blended_regret = self._compute_blended_regret(
            obs_t, action, other_agents_obs, other_agents_actions
        )
        
        # 使用固定阈值进行决策
        decision = self.regret_module.decide(
            blended_regret=blended_regret,
            state=obs_t,
            action=action,
            other_actions=other_agents_actions,
            joint_action=[action] + (other_agents_actions or []),
            critic=self.centralized_critic,
            rewards=None,  # 不传递奖励信息
            current_policy=None  # 不传递策略信息
        )
        
        if decision == "accept":
            final_action = action
        elif decision == "delay":
            final_action = self._get_guardian_action()
        else:  # reject
            final_action = self._get_safe_action()
            
        return final_action, decision, blended_regret

class IndividualRegretOnlyAgent(AblationAgent):
    """只有个体后悔 - w_individual = 1.0"""
    def __init__(self, obs_dim, action_dim, config, agent_id):
        super().__init__(obs_dim, action_dim, config, agent_id, "individual_regret_only")
        
    def _apply_ablation(self):
        """只使用个体后悔，禁用团队后悔"""
        self.regret_module.w_individual = 1.0
        self.regret_module.w_team = 0.0
        
    def _compute_blended_regret(self, obs_t, action, other_agents_obs, other_agents_actions):
        """只计算个体后悔"""
        individual_regret = self.regret_module.compute_individual_regret(
            obs_t, action, other_agents_actions, self.centralized_critic
        )
        # 不计算团队后悔，直接返回个体后悔
        return individual_regret

class TeamRegretOnlyAgent(AblationAgent):
    """只有团队后悔 - w_team = 1.0"""
    def __init__(self, obs_dim, action_dim, config, agent_id):
        super().__init__(obs_dim, action_dim, config, agent_id, "team_regret_only")
        
    def _apply_ablation(self):
        """只使用团队后悔，禁用个体后悔"""
        self.regret_module.w_individual = 0.0
        self.regret_module.w_team = 1.0
        
    def _compute_blended_regret(self, obs_t, action, other_agents_obs, other_agents_actions):
        """只计算团队后悔"""
        team_regret = self.regret_module.compute_team_regret(
            obs_t, [action] + (other_agents_actions or []), self.centralized_critic
        )
        # 不计算个体后悔，直接返回团队后悔
        return team_regret

class NoDelayDomainAgent(AblationAgent):
    """无延迟域 - 直接使用accept/reject决策"""
    def __init__(self, obs_dim, action_dim, config, agent_id):
        super().__init__(obs_dim, action_dim, config, agent_id, "no_delay_domain")
        
    def _apply_ablation(self):
        """禁用延迟域，只保留accept/reject"""
        self.delay_domain_steps = 0
        self.in_delay_domain = False
        
    def act(self, obs, other_agents_obs=None, other_agents_actions=None):
        """重写act方法，禁用延迟域"""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            return action, "explore", 0.0
        
        action_probs = self.policy_net.get_action_probs(obs_t)
        action, _ = self.policy_net.sample_action(obs_t)
        action = action.item()
        
        self.policy_history.append(action_probs.detach())
        
        blended_regret = self._compute_blended_regret(
            obs_t, action, other_agents_obs, other_agents_actions
        )
        
        # 修改决策逻辑：将delay转换为reject
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
        
        # 将delay转换为reject
        if decision == "delay":
            decision = "reject"
            
        if decision == "accept":
            final_action = action
        else:  # reject
            final_action = self._get_safe_action()
            
        return final_action, decision, blended_regret

class NoCentralizedCriticAgent(AblationAgent):
    """无集中式Critic - 完全去中心化"""
    def __init__(self, obs_dim, action_dim, config, agent_id):
        super().__init__(obs_dim, action_dim, config, agent_id, "no_centralized_critic")
        
    def _apply_ablation(self):
        """禁用集中式Critic"""
        # 移除集中式Critic
        self.centralized_critic = None
        
    def _compute_blended_regret(self, obs_t, action, other_agents_obs, other_agents_actions):
        """使用去中心化方法计算后悔"""
        # 使用个体Critic估计Q值
        current_q = self.critic(obs_t)[0].item()
        
        # 尝试所有可能的动作
        max_q = float('-inf')
        for alt_action in range(self.action_dim):
            # 这里需要重新构建观察（简化处理）
            alt_obs = obs_t.clone()
            # 假设动作会影响观察（简化）
            alt_q = self.critic(alt_obs)[0].item()
            max_q = max(max_q, alt_q)
        
        individual_regret = max_q - current_q
        
        # 团队后悔设为0（无法计算）
        team_regret = 0.0
        
        # 计算混合后悔
        blended_regret = (self.regret_module.w_individual * individual_regret + 
                         self.regret_module.w_team * team_regret)
        
        return blended_regret

class NoCommunicationAgent(AblationAgent):
    """无通信机制 - 禁用VOI估计和通信门控"""
    def __init__(self, obs_dim, action_dim, config, agent_id):
        super().__init__(obs_dim, action_dim, config, agent_id, "no_communication")
        
    def _apply_ablation(self):
        """禁用通信机制"""
        self.voi_estimator = None
        self.communication_gate = None
        self.communication_budget = 0
        
    def should_communicate(self, obs):
        """总是返回False，不进行通信"""
        return False
        
    def act(self, obs, other_agents_obs=None, other_agents_actions=None):
        """重写act方法，禁用通信"""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            return action, "explore", 0.0
        
        action_probs = self.policy_net.get_action_probs(obs_t)
        action, _ = self.policy_net.sample_action(obs_t)
        action = action.item()
        
        self.policy_history.append(action_probs.detach())
        
        blended_regret = self._compute_blended_regret(
            obs_t, action, other_agents_obs, other_agents_actions
        )
        
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
        
        if decision == "accept":
            final_action = action
        elif decision == "delay":
            final_action = self._get_guardian_action()
        else:  # reject
            final_action = self._get_safe_action()
            
        return final_action, decision, blended_regret

# 消融研究智能体工厂
def create_ablation_agent(ablation_type, obs_dim, action_dim, config, agent_id):
    """创建消融研究智能体"""
    ablation_types = {
        "no_adaptive": NoAdaptiveAgent,
        "individual_regret_only": IndividualRegretOnlyAgent,
        "team_regret_only": TeamRegretOnlyAgent,
        "no_delay_domain": NoDelayDomainAgent,
        "no_centralized_critic": NoCentralizedCriticAgent,
        "no_communication": NoCommunicationAgent
    }
    
    if ablation_type not in ablation_types:
        raise ValueError(f"不支持的消融类型: {ablation_type}")
        
    return ablation_types[ablation_type](obs_dim, action_dim, config, agent_id)
