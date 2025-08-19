"""
主入口 - RTD增强版
支持策略非平稳性自适应、个体+团队后悔混合、合作/竞争检测、VOI估计和通信门控
"""
import yaml
import torch
from envs.gridworld import GridWorld
from agents.rtd_agent import RTDAgent
from trainers.rtd_trainer import RTDTrainer
import numpy as np

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    
    # 加载配置
    with open("config/experiment_config.yaml", "r") as f:
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
    print("RTD增强版训练开始")
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
    print(f"决策统计: {stats['episode_decisions']}")
    print(f"通信统计: {stats['communication_stats']}")
    print("=" * 50)
