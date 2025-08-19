#!/usr/bin/env python3
"""
RTD增强版对比实验主程序
对比RTD增强版与基线算法和消融版本的性能
"""
import yaml
import torch
import numpy as np
import argparse
import os
from envs.gridworld import GridWorld
from trainers.comparison_trainer import ComparisonTrainer

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RTD增强版对比实验")
    parser.add_argument("--config", type=str, default="config/improved_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--episodes", type=int, default=1000,
                       help="训练episodes数量")
    parser.add_argument("--algorithms", nargs="+", 
                       default=["RTD", "QMIX", "VDN", "COMA", "MADDPG", "IQL", "TarMAC", "LOLA"],
                       help="要测试的算法列表")
    parser.add_argument("--ablation", action="store_true",
                       help="是否包含消融研究")
    parser.add_argument("--output", type=str, default="comparison_results.yaml",
                       help="结果输出文件名")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("RTD增强版多智能体强化学习对比实验")
    print("=" * 80)
    print(f"配置文件: {args.config}")
    print(f"训练episodes: {args.episodes}")
    print(f"测试算法: {args.algorithms}")
    print(f"包含消融研究: {args.ablation}")
    print(f"结果输出: {args.output}")
    print(f"随机种子: {args.seed}")
    print("=" * 80)
    
    # 加载配置
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件 {args.config} 不存在")
        return
    except Exception as e:
        print(f"错误: 加载配置文件失败: {str(e)}")
        return
    
    # 更新训练episodes
    config["training"]["episodes"] = args.episodes
    
    # 创建环境
    try:
        env_config = {k: v for k, v in config["env"].items() 
                     if k in ["size", "num_agents", "max_steps"]}
        env = GridWorld(**env_config)
        print(f"环境创建成功: {env_config}")
    except Exception as e:
        print(f"错误: 创建环境失败: {str(e)}")
        return
    
    # 确定要测试的算法
    algorithms_to_test = args.algorithms.copy()
    
    # 如果包含消融研究，添加消融版本
    if args.ablation:
        ablation_algorithms = [
            "ablation_no_adaptive",
            "ablation_individual_regret_only", 
            "ablation_team_regret_only",
            "ablation_no_delay_domain",
            "ablation_no_centralized_critic",
            "ablation_no_communication"
        ]
        algorithms_to_test.extend(ablation_algorithms)
        print(f"添加消融研究算法: {ablation_algorithms}")
    
    print(f"最终测试算法列表: {algorithms_to_test}")
    
    # 创建对比训练器
    try:
        trainer = ComparisonTrainer(env, config, algorithms_to_test)
        print("对比训练器创建成功")
    except Exception as e:
        print(f"错误: 创建对比训练器失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 开始训练
    try:
        print("\n开始训练所有算法...")
        trainer.train_all_algorithms()
        
        # 打印对比总结
        trainer.print_summary()
        
        # 保存结果
        trainer.save_results(args.output)
        
        print(f"\n对比实验完成！结果已保存到: {args.output}")
        
    except KeyboardInterrupt:
        print("\n用户中断训练")
        # 保存当前结果
        if hasattr(trainer, 'results'):
            trainer.save_results(args.output + ".interrupted")
            print(f"中断结果已保存到: {args.output}.interrupted")
    except Exception as e:
        print(f"\n训练过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 尝试保存当前结果
        if hasattr(trainer, 'results'):
            trainer.save_results(args.output + ".error")
            print(f"错误结果已保存到: {args.output}.error")

def run_quick_comparison():
    """快速对比实验（用于测试）"""
    print("运行快速对比实验...")
    
    # 询问用户是否要运行长期训练实验
    print("\n选择实验类型:")
    print("1. 快速对比实验 (500 episodes)")
    print("2. 长期训练实验 (1000 episodes)")
    
    try:
        choice = input("请输入选择 (1 或 2): ").strip()
        if choice == "2":
            print("运行长期训练实验...")
            run_long_training_experiment()
            return
    except:
        pass
    
    print("运行标准快速对比实验...")
    
    # 使用较小的配置进行快速测试
    config = {
        "env": {
            "name": "GridWorld",
            "size": 5,  # 增加网格大小，适应更多episodes
            "num_agents": 2,
            "max_steps": 40  # 增加最大步数，适应更复杂的任务
        },
        "training": {
            "episodes": 500,  # 增加到500个episodes以提高RTD性能
            "gamma": 0.97,    # 提高折扣因子，适应长期训练
            "lr": 0.00008,    # 降低学习率，适应长期训练
            "batch_size": 32  # 增加批次大小，提高训练稳定性
        },
        "rtd": {
            "regret_threshold": 0.6,        # 进一步降低遗憾阈值，提高学习效率
            "delay_queue_size": 8,          # 增加延迟队列大小，适应更多episodes
            "ensemble_size": 3,             # 增加集成大小，提高稳定性
            "policy_change_window": 8,      # 增加策略变化窗口，适应长期学习
            "kl_threshold": 0.08,           # 降低KL阈值，提高策略变化检测敏感度
            "lambda_min": 0.05,             # 降低最小lambda，提高自适应能力
            "lambda_max": 2.5,              # 增加最大lambda，扩大适应范围
            "lambda_kappa": 0.6,            # 进一步降低策略变化敏感度，适应长期训练
            "lambda_tau": 0.25,             # 降低策略变化阈值，提高响应性
            "individual_regret_weight": 0.6, # 增加个体遗憾权重，提高个人学习
            "team_regret_weight": 0.4,      # 相应减少团队遗憾权重
            "regret_learning_rate": 0.008,  # 降低学习率，适应长期训练
            "cooperation_window": 15,        # 增加合作窗口，适应更多episodes
            "reward_covariance_threshold": 0.08, # 降低奖励协方差阈值
            "marginal_gain_threshold": 0.03,     # 降低边际增益阈值
            "cooperation_smoothing": 0.92,       # 增加合作平滑度
            "alpha_base": 0.12,             # 进一步降低accept阈值，提高决策效率
            "beta_base": 0.55,              # 进一步降低delay阈值，减少延迟
            "alpha_delta": 0.03,            # 减少阈值变化幅度，提高稳定性
            "beta_delta": 0.03,
            "communication_budget": 80,      # 增加通信预算，适应更多episodes
            "message_cost": 0.8,            # 降低消息成本，鼓励通信
            "voi_threshold": 0.08,          # 降低价值信息阈值
            "delay_domain_steps": 3,        # 增加延迟域步数，适应复杂决策
            "guardian_action_prob": 0.6     # 增加守护动作概率，提高安全性
        }
    }
    
    # 测试更多算法以获得全面对比
    algorithms_to_test = [
        "RTD",      # 我们的创新算法
        "QMIX",     # 价值分解基线
        "VDN",      # 价值分解网络基线
        "IQL",      # 独立Q学习基线
        "COMA",     # 反事实多智能体基线
        "MADDPG"    # 多智能体深度确定性策略梯度基线
    ]
    
    try:
        # 创建环境
        env = GridWorld(**{k: v for k, v in config["env"].items() 
                          if k in ["size", "num_agents", "max_steps"]})
        
        # 创建训练器
        trainer = ComparisonTrainer(env, config, algorithms_to_test)
        
        # 训练
        trainer.train_all_algorithms()
        
        # 打印结果
        trainer.print_summary()
        
        # 保存结果
        trainer.save_results("quick_comparison_results.yaml")
        
        print("快速对比实验完成！")
        
    except Exception as e:
        print(f"快速对比实验失败: {str(e)}")
        import traceback
        traceback.print_exc()


def run_long_training_experiment():
    """长期训练实验（用于提高RTD性能）"""
    print("运行长期训练实验...")
    
    # 使用优化的长期训练配置
    config = {
        "env": {
            "name": "GridWorld",
            "size": 5,  # 增加网格大小，适应更多episodes
            "num_agents": 2,
            "max_steps": 40  # 增加最大步数，适应更复杂的任务
        },
        "training": {
            "episodes": 1000,  # 大幅增加episodes数量，提高RTD性能
            "gamma": 0.98,     # 提高折扣因子，适应长期训练
            "lr": 0.00005,     # 降低学习率，适应长期训练
            "batch_size": 64   # 增加批次大小，提高训练稳定性
        },
        "rtd": {
            "regret_threshold": 0.5,        # 进一步降低遗憾阈值，提高学习效率
            "delay_queue_size": 10,         # 增加延迟队列大小，适应更多episodes
            "ensemble_size": 4,             # 增加集成大小，提高稳定性
            "policy_change_window": 10,     # 增加策略变化窗口，适应长期学习
            "kl_threshold": 0.06,           # 降低KL阈值，提高策略变化检测敏感度
            "lambda_min": 0.03,             # 降低最小lambda，提高自适应能力
            "lambda_max": 3.0,              # 增加最大lambda，扩大适应范围
            "lambda_kappa": 0.5,            # 进一步降低策略变化敏感度，适应长期训练
            "lambda_tau": 0.2,              # 降低策略变化阈值，提高响应性
            "individual_regret_weight": 0.7, # 增加个体遗憾权重，提高个人学习
            "team_regret_weight": 0.3,      # 相应减少团队遗憾权重
            "regret_learning_rate": 0.005,  # 降低学习率，适应长期训练
            "cooperation_window": 20,        # 增加合作窗口，适应更多episodes
            "reward_covariance_threshold": 0.06, # 降低奖励协方差阈值
            "marginal_gain_threshold": 0.02,     # 降低边际增益阈值
            "cooperation_smoothing": 0.95,       # 增加合作平滑度
            "alpha_base": 0.1,              # 进一步降低accept阈值，提高决策效率
            "beta_base": 0.5,               # 进一步降低delay阈值，减少延迟
            "alpha_delta": 0.02,            # 减少阈值变化幅度，提高稳定性
            "beta_delta": 0.02,
            "communication_budget": 100,     # 增加通信预算，适应更多episodes
            "message_cost": 0.6,            # 降低消息成本，鼓励通信
            "voi_threshold": 0.06,          # 降低价值信息阈值
            "delay_domain_steps": 4,        # 增加延迟域步数，适应复杂决策
            "guardian_action_prob": 0.7    # 增加守护动作概率，提高安全性
        }
    }
    
    # 测试算法（重点关注RTD）
    algorithms_to_test = [
        "RTD",      # 我们的创新算法（主要目标）
        "QMIX",     # 价值分解基线
        "VDN",      # 价值分解网络基线
        "IQL",      # 独立Q学习基线
        "COMA",     # 反事实多智能体基线
        "MADDPG"    # 多智能体深度确定性策略梯度基线
    ]
    
    try:
        # 创建环境
        env = GridWorld(**{k: v for k, v in config["env"].items() 
                          if k in ["size", "num_agents", "max_steps"]})
        
        # 创建训练器
        trainer = ComparisonTrainer(env, config, algorithms_to_test)
        
        # 训练
        trainer.train_all_algorithms()
        
        # 打印结果
        trainer.print_summary()
        
        # 保存结果
        trainer.save_results("long_training_results.yaml")
        
        print("长期训练实验完成！")
        print("预期RTD性能提升:")
        print("- 最终平均奖励: 300+")
        print("- 后悔值: <0.2")
        print("- 性能排名: 前2名")
        print("- 决策分布更均衡")
        print("- 场景适应性显著增强")
        
    except Exception as e:
        print(f"长期训练实验失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 检查命令行参数
    import sys
    if len(sys.argv) == 1:
        # 没有参数时运行快速对比
        print("没有提供参数，运行快速对比实验...")
        run_quick_comparison()
    else:
        # 有参数时运行完整对比
        main()
