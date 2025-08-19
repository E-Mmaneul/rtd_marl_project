#!/usr/bin/env python3
"""
分析决策统计问题
"""
import sys
import os
import torch
import numpy as np
import yaml
from collections import defaultdict

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_decision_problem():
    """分析决策统计问题"""
    print("=== 分析决策统计全是1的问题 ===")
    
    # 加载配置
    with open("config/experiment_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print("1. 配置分析:")
    print(f"   alpha_base: {config['rtd']['alpha_base']}")
    print(f"   beta_base: {config['rtd']['beta_base']}")
    print(f"   alpha_delta: {config['rtd']['alpha_delta']}")
    print(f"   beta_delta: {config['rtd']['beta_delta']}")
    
    # 创建后悔模块
    from agents.regret_module import RegretModule
    regret_module = RegretModule(config, agent_id=0)
    
    print(f"\n2. 初始阈值:")
    print(f"   alpha_t: {regret_module.alpha_t}")
    print(f"   beta_t: {regret_module.beta_t}")
    
    # 测试后悔值范围
    print(f"\n3. 测试后悔值范围:")
    test_regrets = np.linspace(0.0, 1.0, 11)
    decision_distribution = defaultdict(int)
    
    for regret in test_regrets:
        decision = regret_module.decide(regret)
        decision_distribution[decision] += 1
        print(f"   后悔值 {regret:.2f} -> {decision}")
    
    print(f"\n决策分布: {dict(decision_distribution)}")
    
    # 分析问题原因
    print(f"\n4. 问题分析:")
    print("   问题1: 训练器中的统计记录方式")
    print("   - 原代码: self.episode_decisions[decision].append(1)")
    print("   - 问题: 每次都添加1，而不是计数")
    print("   - 修复: 应该先统计每个决策的次数，再记录")
    
    print(f"\n   问题2: 后悔值计算可能有问题")
    print("   - 如果后悔值总是很小，可能总是触发'accept'决策")
    print("   - 如果后悔值总是很大，可能总是触发'reject'决策")
    
    print(f"\n   问题3: 阈值设置可能不合理")
    print("   - alpha_t 和 beta_t 的值可能不适合当前环境")
    print("   - 需要根据实际后悔值分布调整阈值")
    
    # 提供修复建议
    print(f"\n5. 修复建议:")
    print("   ✓ 已修复训练器中的统计记录方式")
    print("   - 添加后悔值分布监控")
    print("   - 动态调整阈值")
    print("   - 添加决策调试信息")

def test_actual_regret_calculation():
    """测试实际的后悔计算"""
    print(f"\n=== 测试实际后悔计算 ===")
    
    # 加载配置
    with open("config/experiment_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 创建智能体
    from agents.rtd_agent import RTDAgent
    agent = RTDAgent(obs_dim=2, action_dim=5, config=config, agent_id=0)
    
    # 测试多个观察
    test_observations = [
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0]
    ]
    
    print("测试不同观察的决策:")
    for i, obs in enumerate(test_observations):
        try:
            action, decision, regret = agent.act(obs, other_agents_actions=[1])
            print(f"  观察 {obs} -> 动作: {action}, 决策: {decision}, 后悔: {regret:.4f}")
        except Exception as e:
            print(f"  观察 {obs} -> 错误: {e}")

def create_improved_config():
    """创建改进的配置"""
    print(f"\n=== 改进配置建议 ===")
    
    improved_config = {
        "env": {
            "name": "GridWorld",
            "size": 5,
            "num_agents": 2,
            "max_steps": 50
        },
        "training": {
            "episodes": 1000,
            "gamma": 0.95,
            "lr": 0.0001,
            "batch_size": 32
        },
        "rtd": {
            # 基础后悔阈值
            "regret_threshold": 1.0,
            "delay_queue_size": 10,
            "ensemble_size": 3,
            
            # 策略非平稳性自适应参数
            "policy_change_window": 10,
            "kl_threshold": 0.1,
            "lambda_min": 0.1,
            "lambda_max": 2.0,
            "lambda_kappa": 1.0,
            "lambda_tau": 0.5,
            
            # 个体后悔+团队后悔混合参数
            "individual_regret_weight": 0.6,
            "team_regret_weight": 0.4,
            "regret_learning_rate": 0.01,
            
            # 合作/竞争检测参数
            "cooperation_window": 20,
            "reward_covariance_threshold": 0.1,
            "marginal_gain_threshold": 0.05,
            "cooperation_smoothing": 0.9,
            
            # 改进的动态阈值调整参数
            "alpha_base": 0.2,  # 降低基础阈值
            "beta_base": 0.8,   # 提高基础阈值
            "alpha_delta": 0.1, # 减小调整幅度
            "beta_delta": 0.1,
            
            # 通信资源限制参数
            "communication_budget": 100,
            "message_cost": 1,
            "voi_threshold": 0.1,
            
            # 延迟域参数
            "delay_domain_steps": 3,
            "guardian_action_prob": 0.3
        }
    }
    
    print("改进的配置:")
    print(f"  alpha_base: {improved_config['rtd']['alpha_base']} (原: 0.3)")
    print(f"  beta_base: {improved_config['rtd']['beta_base']} (原: 0.7)")
    print(f"  alpha_delta: {improved_config['rtd']['alpha_delta']} (原: 0.2)")
    print(f"  beta_delta: {improved_config['rtd']['beta_delta']} (原: 0.2)")
    
    return improved_config

if __name__ == "__main__":
    analyze_decision_problem()
    test_actual_regret_calculation()
    create_improved_config()
