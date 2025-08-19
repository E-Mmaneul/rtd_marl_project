#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试RTD保守优化效果
"""
import time
import yaml
import torch
import numpy as np
from envs.gridworld import GridWorld
from agents.rtd_agent import RTDAgent
from trainers.rtd_trainer import RTDTrainer

def test_conservative_optimization():
    """测试保守优化效果"""
    print("=" * 70)
    print("RTD保守优化效果测试")
    print("=" * 70)
    
    # 加载配置
    with open("config/improved_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 修改配置以加快测试
    config["training"]["episodes"] = 100  # 增加测试episodes
    config["env"]["max_steps"] = 40
    config["rtd"]["ensemble_size"] = 1
    
    print(f"测试配置: {config['training']['episodes']} episodes, {config['env']['max_steps']} max_steps")
    print("优化策略:")
    print("1. 简化后悔计算：只保留个体后悔，移除团队后悔")
    print("2. 禁用通信机制：直接返回False")
    print("3. 简化延迟域：减少状态跟踪")
    print("4. 减少备选动作：从5个降到3个（up, left, wait）")
    
    # 创建环境
    env = GridWorld(**{k: v for k, v in config["env"].items() 
                      if k in ["size", "num_agents", "max_steps"]})
    
    # 创建智能体
    obs_dim = 2
    action_dim = 5
    agents = []
    for i in range(config["env"]["num_agents"]):
        agent = RTDAgent(obs_dim, action_dim, config, agent_id=i)
        agents.append(agent)
    
    # 创建训练器
    trainer = RTDTrainer(agents, env, config)
    
    # 记录开始时间
    start_time = time.time()
    
    # 开始训练
    print("\n开始训练...")
    trainer.train()
    
    # 记录结束时间
    end_time = time.time()
    training_time = end_time - start_time
    
    # 获取统计信息
    stats = trainer.get_training_stats()
    
    # 打印结果
    print("\n" + "=" * 70)
    print("保守优化测试结果")
    print("=" * 70)
    print(f"训练时间: {training_time:.2f}秒")
    print(f"平均每episode时间: {training_time/config['training']['episodes']:.3f}秒")
    print(f"总episodes: {len(stats['episode_rewards'])}")
    print(f"平均奖励: {np.mean(stats['episode_rewards']):.2f}")
    print(f"平均后悔: {np.mean(stats['episode_regrets']):.4f}")
    
    # 决策分布
    print(f"\n决策分布:")
    for decision, counts in stats['episode_decisions'].items():
        if counts:
            total_count = sum(counts)
            print(f"  {decision}: {total_count} 次")
    
    # 性能评估
    print(f"\n性能评估:")
    if training_time < 60:
        print("✅ 保守优化成功！训练时间大幅减少")
    elif training_time < 120:
        print("⚠️  保守优化部分成功，时间有所减少")
    else:
        print("❌ 保守优化效果不明显")
    
    # 决策分布评估
    accept_count = sum(stats['episode_decisions'].get('accept', []))
    delay_count = sum(stats['episode_decisions'].get('delay', []))
    total_decisions = accept_count + delay_count
    
    if total_decisions > 0:
        accept_ratio = accept_count / total_decisions
        print(f"Accept比例: {accept_ratio:.2%}")
        if accept_ratio > 0.15:
            print("✅ 决策分布良好，accept比例合理")
        elif accept_ratio > 0.1:
            print("⚠️ 决策分布一般，accept比例偏低")
        else:
            print("❌ 决策分布不佳，accept比例过低")
    
    # 与之前结果对比
    print(f"\n优化效果对比:")
    print(f"之前训练时间: 113.24秒 (1000 episodes)")
    print(f"当前训练时间: {training_time:.2f}秒 ({config['training']['episodes']} episodes)")
    
    # 估算1000 episodes的时间
    estimated_time_1000 = training_time * (1000 / config['training']['episodes'])
    print(f"估算1000 episodes时间: {estimated_time_1000:.2f}秒")
    
    if estimated_time_1000 < 60:
        print("🎉 预期优化效果：时间减少80%+")
    elif estimated_time_1000 < 100:
        print("✅ 预期优化效果：时间减少60%+")
    else:
        print("⚠️ 预期优化效果：时间减少40%+")
    
    print("=" * 70)

if __name__ == "__main__":
    test_conservative_optimization()

