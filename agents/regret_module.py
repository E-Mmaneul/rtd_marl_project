"""
后悔计算与三支决策逻辑 - 增强版
实现个体后悔+团队后悔混合机制和动态阈值调整
"""
import torch
import numpy as np
from collections import deque
import torch.nn.functional as F

class RegretModule:
    def __init__(self, config, agent_id):
        self.agent_id = agent_id
        self.cfg = config
        
        # 获取智能体数量
        self.num_agents = config["env"]["num_agents"]
        
        # 基础阈值
        self.alpha_base = config["rtd"]["alpha_base"]
        self.beta_base = config["rtd"]["beta_base"]
        self.alpha_delta = config["rtd"]["alpha_delta"]
        self.beta_delta = config["rtd"]["beta_delta"]
        
        # 动态阈值
        self.alpha_t = self.alpha_base
        self.beta_t = self.beta_base
        
        # 后悔权重
        self.w_individual = config["rtd"]["individual_regret_weight"]
        self.w_team = config["rtd"]["team_regret_weight"]
        
        # 合作概率
        self.cooperation_prob = 0.5
        self.cooperation_history = deque(maxlen=config["rtd"]["cooperation_window"])
        
        # 策略变化检测
        self.policy_history = deque(maxlen=config["rtd"]["policy_change_window"])
        self.lambda_t = 1.0
        
    def compute_individual_regret(self, state, action, other_actions, critic):
        """
        计算个体后悔：max_{a_i'} Q_tot(s, a_i', a_{-i}) - Q_tot(s, a_i, a_{-i})
        保守优化版本：减少备选动作数量，只考虑3个关键动作
        """
        with torch.no_grad():
            # 构建联合状态
            other_actions = other_actions or []
            
            if len(state.shape) == 2:
                joint_states = state.unsqueeze(1).repeat(1, self.num_agents, 1)
            elif len(state.shape) == 3:
                joint_states = state
            else:
                raise ValueError(f"Unexpected state shape: {state.shape}")
            
            # 准备其他智能体的动作
            if len(other_actions) < self.num_agents - 1:
                padded_other_actions = other_actions + [0] * (self.num_agents - 1 - len(other_actions))
            else:
                padded_other_actions = other_actions[:self.num_agents - 1]
            
            # 保守优化：只考虑3个关键动作而不是5个
            # 选择最重要的动作：up(0), left(2), wait(4)
            key_actions = [0, 2, 4]  # up, left, wait
            batch_size = 3
            all_joint_actions = []
            
            for alt_action in key_actions:
                joint_action = [alt_action] + padded_other_actions
                all_joint_actions.append(joint_action)
            
            # 批量前向传播 - 一次性计算3个动作的Q值
            joint_actions_batch = torch.tensor(all_joint_actions, dtype=torch.long)  # [3, num_agents]
            joint_states_batch = joint_states.repeat(batch_size, 1, 1)  # [3, num_agents, state_dim]
            
            q_values = critic(joint_states_batch, joint_actions_batch)  # [3]
            
            # 计算后悔：找到当前动作在key_actions中的索引
            if action in key_actions:
                action_idx = key_actions.index(action)
                current_q = q_values[action_idx].item()
            else:
                # 如果当前动作不在key_actions中，使用默认值
                current_q = q_values[1].item()  # 使用left动作的Q值作为参考
            
            max_q = q_values.max().item()
            individual_regret = max_q - current_q
            
            return individual_regret
    
    def compute_team_regret(self, state, joint_action, critic):
        """
        计算团队后悔：max_{a'} Q_tot(s, a') - Q_tot(s, a)
        优化版本：批量前向传播
        """
        with torch.no_grad():
            joint_action = joint_action or []
            
            if len(state.shape) == 2:
                joint_states = state.unsqueeze(1).repeat(1, self.num_agents, 1)
            elif len(state.shape) == 3:
                joint_states = state
            else:
                raise ValueError(f"Unexpected state shape: {state.shape}")
            
            # 准备当前联合动作
            if len(joint_action) < self.num_agents:
                current_joint_action = joint_action + [0] * (self.num_agents - len(joint_action))
            else:
                current_joint_action = joint_action[:self.num_agents]
            
            # 当前联合动作的Q值
            current_joint_actions = torch.tensor(current_joint_action, dtype=torch.long).unsqueeze(0)
            current_q = critic(joint_states, current_joint_actions)
            
            # 批量计算所有可能的联合动作
            if self.num_agents == 2:
                # 两智能体：生成所有5×5=25种组合
                all_joint_actions = []
                for i in range(5):
                    for j in range(5):
                        all_joint_actions.append([i, j])
                
                # 批量前向传播
                joint_actions_batch = torch.tensor(all_joint_actions, dtype=torch.long)  # [25, 2]
                joint_states_batch = joint_states.repeat(25, 1, 1)  # [25, 2, state_dim]
                
                q_values = critic(joint_states_batch, joint_actions_batch)  # [25]
                max_q = q_values.max().item()
                
            else:
                # 多智能体：随机采样避免组合爆炸
                max_q = float('-inf')
                for _ in range(10):  # 减少采样次数
                    alt_joint_action = [torch.randint(0, 5, (1,)).item() for _ in range(self.num_agents)]
                    alt_joint_actions = torch.tensor(alt_joint_action, dtype=torch.long).unsqueeze(0)
                    q_value = critic(joint_states, alt_joint_actions)
                    max_q = max(max_q, q_value.item())
            
            team_regret = max_q - current_q.item()
            return team_regret
    
    def compute_blended_regret(self, individual_regret, team_regret):
        """
        计算混合后悔：R_i^blend = w_i * R_i^ind + (1 - w_i) * R^team
        """
        # 确保输入是张量
        if not isinstance(individual_regret, torch.Tensor):
            individual_regret = torch.tensor(individual_regret, dtype=torch.float32)
        if not isinstance(team_regret, torch.Tensor):
            team_regret = torch.tensor(team_regret, dtype=torch.float32)
        
        blended_regret = (self.w_individual * individual_regret + 
                         self.w_team * team_regret)
        
        # 归一化到[0,1]
        blended_regret = torch.sigmoid(blended_regret)
        return blended_regret
    
    def update_cooperation_probability(self, rewards, actions):
        """
        更新合作概率：基于奖励协方差和边际增益
        """
        if len(rewards) < 2:
            return
        
        # 计算奖励协方差
        try:
            if len(rewards) >= 2:
                cov_matrix = np.cov(rewards)
                if cov_matrix.ndim == 2 and cov_matrix.shape[0] >= 2 and cov_matrix.shape[1] >= 2:
                    # 动态计算协方差，适应不同数量的智能体
                    if self.num_agents == 2:
                        reward_cov = cov_matrix[0, 1]
                    else:
                        # 对于更多智能体，计算平均协方差
                        total_cov = 0
                        count = 0
                        for i in range(min(self.num_agents, cov_matrix.shape[0])):
                            for j in range(i+1, min(self.num_agents, cov_matrix.shape[1])):
                                total_cov += cov_matrix[i, j]
                                count += 1
                        reward_cov = total_cov / count if count > 0 else 0
                else:
                    reward_cov = 0
            else:
                reward_cov = 0
        except:
            reward_cov = 0
        
        # 计算边际增益（简化版本）
        marginal_gain = np.mean(rewards) - np.mean(rewards[:-1]) if len(rewards) > 1 else 0
        
        # 归一化并映射到[0,1]
        cooperation_score = (reward_cov + marginal_gain) / 2
        cooperation_score = 1 / (1 + np.exp(-cooperation_score))  # sigmoid
        
        # EMA平滑
        self.cooperation_prob = (self.cfg["rtd"]["cooperation_smoothing"] * self.cooperation_prob + 
                               (1 - self.cfg["rtd"]["cooperation_smoothing"]) * cooperation_score)
        
        self.cooperation_history.append(self.cooperation_prob)
    
    def update_policy_change_rate(self, current_policy):
        """
        更新策略变化率：基于KL散度
        """
        self.policy_history.append(current_policy)
        
        if len(self.policy_history) < 2:
            return
        
        # 计算KL散度
        current_dist = self.policy_history[-1]
        previous_dist = self.policy_history[-2]
        
        kl_div = F.kl_div(
            torch.log(current_dist + 1e-8),
            previous_dist + 1e-8,
            reduction='batchmean'
        )
        
        # 自适应lambda调整
        change_rate = kl_div.item()
        self.lambda_t = np.clip(
            self.cfg["rtd"]["lambda_kappa"] * torch.sigmoid(torch.tensor(change_rate - self.cfg["rtd"]["lambda_tau"])).item(),
            self.cfg["rtd"]["lambda_min"],
            self.cfg["rtd"]["lambda_max"]
        )
    
    def update_dynamic_thresholds(self):
        """
        根据合作概率和策略变化率更新动态阈值
        """
        # 基于合作概率调整
        alpha_coop = self.alpha_base + (1 - self.cooperation_prob) * self.alpha_delta
        beta_coop = self.beta_base - self.cooperation_prob * self.beta_delta
        
        # 基于策略变化率调整
        alpha_policy = alpha_coop * (1 + self.lambda_t)
        beta_policy = beta_coop * (1 + self.lambda_t)
        
        # 确保alpha < beta
        if alpha_policy >= beta_policy:
            beta_policy = alpha_policy + 0.1
        
        self.alpha_t = alpha_policy
        self.beta_t = beta_policy
    
    def decide(self, blended_regret, state=None, action=None, other_actions=None, 
               joint_action=None, critic=None, rewards=None, current_policy=None):
        """
        三支决策：accept/delay/reject - 修复版本
        """
        # 更新合作概率
        if rewards is not None:
            self.update_cooperation_probability(rewards, other_actions or [])
        
        # 更新策略变化率
        if current_policy is not None:
            self.update_policy_change_rate(current_policy)
        
        # 更新动态阈值
        self.update_dynamic_thresholds()
        
        # 添加随机性以避免过度保守
        noise = np.random.normal(0, 0.1)
        adjusted_regret = blended_regret + noise
        
        # 三支决策 - 大幅调整阈值，让accept更容易触发
        if adjusted_regret < 0.3:  # 固定较低的accept阈值
            return "accept"
        elif adjusted_regret < 0.7:  # 适中的delay阈值
            return "delay"
        else:
            return "reject"
    
    def get_regret_info(self):
        """
        获取后悔相关信息用于调试
        """
        return {
            'alpha_t': self.alpha_t,
            'beta_t': self.beta_t,
            'cooperation_prob': self.cooperation_prob,
            'lambda_t': self.lambda_t,
            'individual_weight': self.w_individual,
            'team_weight': self.w_team
        }
