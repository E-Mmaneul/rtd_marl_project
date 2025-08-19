#!/usr/bin/env python3
"""
基础功能测试脚本
用于验证环境、智能体等基本组件是否正常工作
"""
import sys
import os
import torch
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.gridworld import GridWorld
from agents.rtd_agent import RTDAgent
from agents.baseline_agents import QMIXAgent

def test_environment():
    """测试环境基本功能"""
    print("=" * 50)
    print("测试环境基本功能")
    print("=" * 50)
    
    try:
        # 创建环境
        env = GridWorld(size=3, num_agents=2, max_steps=10)
        print("✓ 环境创建成功")
        
        # 测试重置
        obs = env.reset()
        print(f"✓ 环境重置成功，观察值: {obs}")
        print(f"✓ 观察值类型: {type(obs)}, 长度: {len(obs) if obs else 'None'}")
        
        # 测试步进
        actions = [0, 1]  # 两个智能体的动作
        result = env.step(actions)
        print(f"✓ 环境步进成功，返回值: {result}")
        print(f"✓ 返回值类型: {type(result)}, 长度: {len(result)}")
        
        # 验证返回值
        if len(result) == 4:
            new_obs, rewards, done, info = result
            print(f"✓ 新观察值: {new_obs}")
            print(f"✓ 奖励: {rewards}")
            print(f"✓ 完成状态: {done}")
            print(f"✓ 信息: {info}")
        else:
            print(f"✗ 返回值长度不正确: {len(result)}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ 环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rtd_agent():
    """测试RTD智能体基本功能"""
    print("\n" + "=" * 50)
    print("测试RTD智能体基本功能")
    print("=" * 50)
    
    try:
        # 创建智能体
        config = {
            "env": {"num_agents": 2},
            "rtd": {
                "regret_threshold": 1.0,
                "delay_queue_size": 5,
                "ensemble_size": 2,
                "policy_change_window": 5,
                "kl_threshold": 0.1,
                "lambda_min": 0.1,
                "lambda_max": 2.0,
                "lambda_kappa": 1.0,
                "lambda_tau": 0.5,
                "individual_regret_weight": 0.6,
                "team_regret_weight": 0.4,
                "regret_learning_rate": 0.01,
                "cooperation_window": 10,
                "reward_covariance_threshold": 0.1,
                "marginal_gain_threshold": 0.05,
                "cooperation_smoothing": 0.9,
                "alpha_base": 0.2,
                "beta_base": 0.8,
                "alpha_delta": 0.1,
                "beta_delta": 0.1,
                "communication_budget": 50,
                "message_cost": 1,
                "voi_threshold": 0.1,
                "delay_domain_steps": 2,
                "guardian_action_prob": 0.3
            }
        }
        
        agent = RTDAgent(obs_dim=2, action_dim=5, config=config, agent_id=0)
        print("✓ RTD智能体创建成功")
        
        # 测试动作选择
        obs = [1, 2]  # 示例观察值
        other_agents_obs = [[0, 1]]
        other_agents_actions = [2]
        
        result = agent.act(obs, other_agents_obs, other_agents_actions)
        print(f"✓ 动作选择成功，返回值: {result}")
        print(f"✓ 返回值类型: {type(result)}, 长度: {len(result)}")
        
        # 验证返回值
        if len(result) == 3:
            action, decision, regret = result
            print(f"✓ 动作: {action}")
            print(f"✓ 决策: {decision}")
            print(f"✓ 后悔值: {regret}")
        else:
            print(f"✗ 返回值长度不正确: {len(result)}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ RTD智能体测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_baseline_agent():
    """测试基线智能体基本功能"""
    print("\n" + "=" * 50)
    print("测试基线智能体基本功能")
    print("=" * 50)
    
    try:
        # 创建智能体
        config = {"env": {"num_agents": 2}}
        agent = QMIXAgent(obs_dim=2, action_dim=5, config=config, agent_id=0)
        print("✓ QMIX智能体创建成功")
        
        # 测试动作选择
        obs = [1, 2]  # 示例观察值
        other_agents_obs = [[0, 1]]
        other_agents_actions = [2]
        
        result = agent.act(obs, other_agents_obs, other_agents_actions)
        print(f"✓ 动作选择成功，返回值: {result}")
        print(f"✓ 返回值类型: {type(result)}, 长度: {len(result)}")
        
        # 验证返回值
        if len(result) == 3:
            action, decision, regret = result
            print(f"✓ 动作: {action}")
            print(f"✓ 决策: {decision}")
            print(f"✓ 后悔值: {regret}")
        else:
            print(f"✗ 返回值长度不正确: {len(result)}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ 基线智能体测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_interaction():
    """测试简单的智能体-环境交互"""
    print("\n" + "=" * 50)
    print("测试简单的智能体-环境交互")
    print("=" * 50)
    
    try:
        # 创建环境和智能体
        env = GridWorld(size=3, num_agents=2, max_steps=5)
        config = {
            "env": {"num_agents": 2},
            "rtd": {
                "ensemble_size": 3,
                "delay_queue_size": 10,
                "communication_budget": 100,
                "message_cost": 1,
                "voi_threshold": 0.1,
                "delay_domain_steps": 5,
                "guardian_action_prob": 0.3,
                "alpha_base": 0.1,
                "beta_base": 0.2,
                "alpha_delta": 0.05,
                "beta_delta": 0.05,
                "individual_regret_weight": 0.6,
                "team_regret_weight": 0.4,
                "cooperation_window": 10,
                "policy_change_window": 20,
                "regret_threshold": 1.0,
                "kl_threshold": 0.1,
                "lambda_min": 0.1,
                "lambda_max": 2.0,
                "lambda_kappa": 1.0,
                "lambda_tau": 0.5,
                "regret_learning_rate": 0.01,
                "reward_covariance_threshold": 0.1,
                "marginal_gain_threshold": 0.05,
                "cooperation_smoothing": 0.9
            }
        }
        
        rtd_agent = RTDAgent(obs_dim=2, action_dim=5, config=config, agent_id=0)
        qmix_agent = QMIXAgent(obs_dim=2, action_dim=5, config=config, agent_id=1)
        
        agents = [rtd_agent, qmix_agent]
        print("✓ 环境和智能体创建成功")
        
        # 运行一个简单的episode
        obs = env.reset()
        print(f"✓ 环境重置，初始观察值: {obs}")
        
        for step in range(3):  # 只运行3步
            print(f"\n--- Step {step + 1} ---")
            
            # 收集动作
            actions = []
            for i, agent in enumerate(agents):
                other_agents_obs = [obs[j] for j in range(len(agents)) if j != i]
                # 过滤掉None值，只保留有效的动作
                other_agents_actions = [actions[j] for j in range(len(agents)) if j < len(actions) and actions[j] is not None]
                
                result = agent.act(obs[i], other_agents_obs, other_agents_actions)
                action, decision, regret = result
                actions.append(action)
                
                print(f"智能体 {i} ({type(agent).__name__}): 动作={action}, 决策={decision}, 后悔={regret}")
            
            # 环境步进
            new_obs, rewards, done, info = env.step(actions)
            print(f"环境步进: 奖励={rewards}, 完成={done}")
            
            # 更新观察值
            obs = new_obs
            
            if done:
                print("Episode完成")
                break
        
        print("✓ 简单交互测试成功")
        return True
        
    except Exception as e:
        print(f"✗ 简单交互测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始基础功能测试...")
    
    tests = [
        test_environment,
        test_rtd_agent,
        test_baseline_agent,
        test_simple_interaction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✓ 测试通过")
            else:
                print("✗ 测试失败")
        except Exception as e:
            print(f"✗ 测试异常: {e}")
    
    print(f"\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！基础功能正常")
        return True
    else:
        print("❌ 部分测试失败，需要检查代码")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
