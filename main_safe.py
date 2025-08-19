#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编码安全的主入口文件
"""
import sys
import os

# 设置默认编码为UTF-8
if sys.platform.startswith('win'):
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except:
        pass

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """主函数"""
    try:
        # 导入必要的模块
        import yaml
        import torch
        from envs.gridworld import GridWorld
        from agents.rtd_agent import RTDAgent
        from trainers.rtd_trainer import RTDTrainer
        import numpy as np
        
        # 设置随机种子
        torch.manual_seed(42)
        
        # 加载改进的配置
        with open("config/improved_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 创建环境
        env_config = {k: v for k, v in config["env"].items() if k in ["size", "num_agents", "max_steps"]}
        env = GridWorld(**env_config)

        # 创建智能体
        obs_dim = 2  # (x, y)
        action_dim = 5  # 支持5个动作：up, down, left, right, wait
        agents = []
        for i in range(config["env"]["num_agents"]):
            agent = RTDAgent(obs_dim, action_dim, config, agent_id=i)
            agents.append(agent)
        
        # 创建训练器
        trainer = RTDTrainer(agents, env, config)
        
        # 开始训练
        print("=" * 50)
        print("RTD增强版训练开始（使用改进配置）")
        print("=" * 50)
        print(f"环境配置: {env_config}")
        print(f"智能体数量: {config['env']['num_agents']}")
        print(f"训练episodes: {config['training']['episodes']}")
        print("=" * 50)
        
        trainer.train()
        
        # 打印最终统计信息
        stats = trainer.get_training_stats()
        print("\n" + "=" * 50)
        print("训练完成！最终统计信息:")
        print("=" * 50)
        print(f"总episodes: {len(stats['episode_rewards'])}")
        print(f"平均奖励: {np.mean(stats['episode_rewards']):.2f}")
        print(f"平均后悔: {np.mean(stats['episode_regrets']):.4f}")
        print("=" * 50)
        
    except Exception as e:
        print(f"运行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
