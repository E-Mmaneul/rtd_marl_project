#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试RTD优化效果
"""
import time
import yaml
import torch
import numpy as np
from envs.gridworld import GridWorld
from agents.rtd_agent import RTDAgent
from trainers.rtd_trainer import RTDTrainer

def test_optimization():
    """测试优化效果"""
    print("=" * 60)
    print("RTD优化效果测试")
    print("=" * 60)
    
    # 加载配置
    with open("config/improved_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 修改配置以加快测试
    config["training"]["episodes"] = 50  # 减少episodes
    config["env"]["max_steps"] = 30      # 减少最大步数
    config["rtd"]["ensemble_size"] = 2   # 减少集成大小
    
    print(f"测试配置: {config['training']['episodes']} episodes, {config['env']['max_steps']} max_steps")
    
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
    print("\n" + "=" * 60)
    print("优化测试结果")
    print("=" * 60)
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
    if training_time < 30:
        print("✅ 优化成功！训练时间大幅减少")
    elif training_time < 60:
        print("⚠️  优化部分成功，时间有所减少")
    else:
        print("❌ 优化效果不明显，需要进一步调整")
    
    # 决策分布评估
    accept_count = sum(stats['episode_decisions'].get('accept', []))
    delay_count = sum(stats['episode_decisions'].get('delay', []))
    total_decisions = accept_count + delay_count
    
    if total_decisions > 0:
        accept_ratio = accept_count / total_decisions
        print(f"Accept比例: {accept_ratio:.2%}")
        if accept_ratio > 0.1:
            print("✅ 决策分布改善，accept比例合理")
        else:
            print("⚠️ 决策分布仍需调整，accept比例过低")
    
    print("=" * 60)

if __name__ == "__main__":
    test_optimization()

