"""
通信工具类 - 实现VOI估计和通信决策
"""
import torch
import numpy as np
from collections import deque

class CommunicationManager:
    def __init__(self, config):
        self.cfg = config
        self.communication_budget = config["rtd"]["communication_budget"]
        self.message_cost = config["rtd"]["message_cost"]
        self.voi_threshold = config["rtd"]["voi_threshold"]
        
        # 通信历史
        self.message_history = deque(maxlen=100)
        self.voi_history = deque(maxlen=100)
        
        # 通信统计
        self.messages_sent = 0
        self.messages_received = 0
        self.total_voi = 0.0
    
    def estimate_voi(self, state, message, voi_estimator):
        """
        估计信息价值：VOI ≈ E[U|m] - E[U|Ø] - cost(m)
        """
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        message_t = torch.tensor(message, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            utility_with, voi = voi_estimator(state_t, message_t)
            
        return voi.item()
    
    def should_communicate(self, state, voi_estimator, communication_gate):
        """
        决定是否通信
        """
        # 检查通信预算
        if self.communication_budget <= 0:
            return False, None, 0.0
        
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        # 通信门控
        comm_prob = communication_gate(state_t)
        
        if comm_prob.item() > 0.5:
            # 生成示例消息
            message = np.random.randn(16)
            
            # 估计VOI
            voi = self.estimate_voi(state, message, voi_estimator)
            
            if voi > self.voi_threshold:
                # 发送消息
                self.communication_budget -= self.message_cost
                self.messages_sent += 1
                self.total_voi += voi
                
                # 记录历史
                self.message_history.append(message)
                self.voi_history.append(voi)
                
                return True, message, voi
        
        return False, None, 0.0
    
    def get_communication_stats(self):
        """
        获取通信统计信息
        """
        avg_voi = self.total_voi / max(self.messages_sent, 1)
        
        return {
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'communication_budget': self.communication_budget,
            'total_voi': self.total_voi,
            'avg_voi': avg_voi,
            'message_history_size': len(self.message_history),
            'voi_history_size': len(self.voi_history)
        }

class CooperationDetector:
    """
    合作/竞争检测器
    """
    def __init__(self, config):
        self.cfg = config
        self.cooperation_window = config["rtd"]["cooperation_window"]
        self.reward_covariance_threshold = config["rtd"]["reward_covariance_threshold"]
        self.marginal_gain_threshold = config["rtd"]["marginal_gain_threshold"]
        self.cooperation_smoothing = config["rtd"]["cooperation_smoothing"]
        
        # 历史记录
        self.reward_history = deque(maxlen=self.cooperation_window)
        self.action_history = deque(maxlen=self.cooperation_window)
        self.cooperation_prob = 0.5
    
    def update_cooperation_probability(self, rewards, actions):
        """
        更新合作概率：基于奖励协方差和边际增益
        """
        if len(rewards) < 2:
            return self.cooperation_prob
        
        # 计算奖励协方差
        try:
            reward_cov = np.cov(rewards)[0, 1] if len(rewards) > 1 else 0
        except:
            reward_cov = 0
        
        # 计算边际增益
        if len(self.reward_history) > 0:
            marginal_gain = np.mean(rewards) - np.mean(list(self.reward_history)[-5:])
        else:
            marginal_gain = 0
        
        # 归一化并映射到[0,1]
        cooperation_score = (reward_cov + marginal_gain) / 2
        cooperation_score = 1 / (1 + np.exp(-cooperation_score))  # sigmoid
        
        # EMA平滑
        self.cooperation_prob = (self.cooperation_smoothing * self.cooperation_prob + 
                               (1 - self.cooperation_smoothing) * cooperation_score)
        
        # 更新历史
        self.reward_history.append(rewards)
        self.action_history.append(actions)
        
        return self.cooperation_prob
    
    def get_cooperation_info(self):
        """
        获取合作信息
        """
        return {
            'cooperation_prob': self.cooperation_prob,
            'reward_history_size': len(self.reward_history),
            'action_history_size': len(self.action_history)
        }
