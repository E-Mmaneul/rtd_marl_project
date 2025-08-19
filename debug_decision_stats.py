#!/usr/bin/env python3
"""
调试决策统计全是1的问题
"""
import sys
import os
import torch
import numpy as np
import yaml

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_decision_stats():
    """调试决策统计问题"""
    print("=== 调试决策统计问题 ===")
    
    # 加载配置
    with open("config/experiment_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print(f"配置中的阈值设置:")
    print(f"  alpha_base: {config['rtd']['alpha_base']}")
    print(f"  beta_base: {config['rtd']['beta_base']}")
    print(f"  alpha_delta: {config['rtd']['alpha_delta']}")
    print(f"  beta_delta: {config['rtd']['beta_delta']}")
    
    # 创建后悔模块
    from agents.regret_module import RegretModule
    regret_module = RegretModule(config, agent_id=0)
    
    print(f"\n初始阈值:")
    print(f"  alpha_t: {regret_module.alpha_t}")
    print(f"  beta_t: {regret_module.beta_t}")
    
    # 测试不同的后悔值
    test_regrets = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    print(f"\n测试不同后悔值的决策:")
    for regret in test_regrets:
        decision = regret_module.decide(regret)
        print(f"  后悔值 {regret:.2f} -> 决策: {decision}")
    
    # 测试后悔计算
    print(f"\n测试后悔计算:")
    from agents.networks import CentralizedCritic
    
    # 创建模拟critic
    critic = CentralizedCritic(state_dim=2, action_dim=5, num_agents=2)
    
    # 测试状态和动作
    state = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    action = 2
    other_actions = [1]
    
    print(f"测试状态: {state}")
    print(f"测试动作: {action}")
    print(f"其他动作: {other_actions}")
    
    try:
        # 计算个体后悔
        individual_regret = regret_module.compute_individual_regret(
            state, action, other_actions, critic
        )
        print(f"个体后悔: {individual_regret}")
        
        # 计算团队后悔
        joint_action = [action] + other_actions
        team_regret = regret_module.compute_team_regret(
            state, joint_action, critic
        )
        print(f"团队后悔: {team_regret}")
        
        # 计算混合后悔
        blended_regret = regret_module.compute_blended_regret(
            individual_regret, team_regret
        )
        print(f"混合后悔: {blended_regret}")
        
        # 测试决策
        decision = regret_module.decide(blended_regret)
        print(f"最终决策: {decision}")
        
    except Exception as e:
        print(f"后悔计算失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 分析训练器中的统计记录
    print(f"\n分析训练器统计记录:")
    print("问题可能在于:")
    print("1. 后悔值计算不正确")
    print("2. 阈值设置不合理")
    print("3. 决策逻辑有问题")
    print("4. 统计记录方式有问题")
    
    # 检查训练器中的记录逻辑
    print(f"\n训练器记录逻辑分析:")
    print("在 _record_statistics 方法中:")
    print("  for agent_data in episode_data.values():")
    print("      for transition in agent_data:")
    print("          decision = transition.get('decision', 'unknown')")
    print("          self.episode_decisions[decision].append(1)  # 这里总是添加1")
    print("")
    print("问题：每次记录都是append(1)，而不是计数")
    print("正确的做法应该是计数，而不是每次都添加1")

def fix_decision_stats():
    """修复决策统计问题"""
    print(f"\n=== 修复建议 ===")
    
    print("1. 修改训练器中的统计记录方式:")
    print("   将 append(1) 改为计数方式")
    
    print("2. 检查后悔值计算:")
    print("   确保后悔值在合理范围内")
    
    print("3. 调整阈值设置:")
    print("   确保alpha_t < beta_t，且值合理")
    
    print("4. 添加调试信息:")
    print("   在决策过程中打印后悔值和阈值")

if __name__ == "__main__":
    debug_decision_stats()
    fix_decision_stats()
