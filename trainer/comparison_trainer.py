"""
对比实验训练器
用于同时训练多个算法并进行性能对比
"""
import torch
import torch.optim as optim
import numpy as np
import random
from collections import defaultdict
import time
import yaml
import os
from agents.rtd_agent import RTDAgent
from agents.baseline_agents import create_baseline_agent
from agents.ablation_agents import create_ablation_agent

class ComparisonTrainer:
    """对比实验训练器"""
    def __init__(self, env, config, algorithms_to_test):
        self.env = env
        self.cfg = config
        self.algorithms_to_test = algorithms_to_test
        
        # 创建不同算法的智能体
        self.agents_dict = {}
        self.optimizers_dict = {}
        self._create_agents()
        
        # 训练参数
        self.gamma = config.get("training", {}).get("gamma", 0.99)
        self.batch_size = config.get("training", {}).get("batch_size", 32)
        
        # 统计信息
        self.results = defaultdict(lambda: {
            'episode_rewards': [],
            'episode_regrets': [],
            'episode_decisions': defaultdict(list),
            'communication_stats': defaultdict(int),
            'scenario_switch_performance': [],
            'training_time': 0.0
        })
        
        # 场景切换
        self.scenario_switch_interval = 50  # 每50个episode切换一次场景，适应更多episodes
        
    def _create_agents(self):
        """创建不同算法的智能体"""
        obs_dim = 2  # (x, y)
        action_dim = 5  # 支持5个动作
        
        for algorithm_name in self.algorithms_to_test:
            agents = []
            optimizers = []
            
            for i in range(self.cfg["env"]["num_agents"]):
                if algorithm_name == "RTD":
                    agent = RTDAgent(obs_dim, action_dim, self.cfg, agent_id=i)
                elif algorithm_name.startswith("ablation_"):
                    # 消融研究
                    ablation_type = algorithm_name.split("_", 1)[1]
                    agent = create_ablation_agent(ablation_type, obs_dim, action_dim, self.cfg, agent_id=i)
                else:
                    # 基线算法
                    agent = create_baseline_agent(algorithm_name, obs_dim, action_dim, self.cfg, agent_id=i)
                
                agents.append(agent)
                
                # 创建优化器
                if hasattr(agent, 'policy_net'):
                    optimizer = optim.Adam([
                        {'params': agent.policy_net.parameters()},
                        {'params': agent.critic.parameters()},
                    ], lr=self.cfg["training"]["lr"])
                    
                    # 如果有其他网络，添加到优化器
                    if hasattr(agent, 'centralized_critic'):
                        optimizer.add_param_group({'params': agent.centralized_critic.parameters()})
                    if hasattr(agent, 'voi_estimator'):
                        optimizer.add_param_group({'params': agent.voi_estimator.parameters()})
                    if hasattr(agent, 'communication_gate'):
                        optimizer.add_param_group({'params': agent.communication_gate.parameters()})
                        
                    optimizers.append(optimizer)
                else:
                    # 基线算法可能没有可训练参数
                    optimizers.append(None)
            
            self.agents_dict[algorithm_name] = agents
            self.optimizers_dict[algorithm_name] = optimizers
            
    def train_all_algorithms(self):
        """训练所有算法"""
        print("=" * 80)
        print("开始对比实验训练")
        print("=" * 80)
        print(f"测试算法: {self.algorithms_to_test}")
        print(f"训练episodes: {self.cfg['training']['episodes']}")
        print(f"环境配置: {self.cfg['env']}")
        print("=" * 80)
        
        for algorithm_name in self.algorithms_to_test:
            print(f"\n开始训练算法: {algorithm_name}")
            start_time = time.time()
            
            try:
                self._train_single_algorithm(algorithm_name)
                training_time = time.time() - start_time
                self.results[algorithm_name]['training_time'] = training_time
                print(f"算法 {algorithm_name} 训练完成，耗时: {training_time:.2f}秒")
                
            except Exception as e:
                print(f"算法 {algorithm_name} 训练失败: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
                
        print("\n" + "=" * 80)
        print("所有算法训练完成！")
        print("=" * 80)
        
    def _train_single_algorithm(self, algorithm_name):
        """训练单个算法"""
        agents = self.agents_dict[algorithm_name]
        optimizers = self.optimizers_dict[algorithm_name]
        
        for ep in range(self.cfg["training"]["episodes"]):
            try:
                # 场景切换
                self._switch_scenario(ep)
                
                # 训练一个episode
                episode_data = self._train_episode(agents, ep)
                
                # 更新网络
                if episode_data and any(opt is not None for opt in optimizers):
                    self._update_networks(agents, optimizers, episode_data)
                
                # 记录统计信息
                self._record_statistics(algorithm_name, episode_data, ep)
                
                # 定期打印信息
                if ep % 100 == 0:
                    self._print_training_info(algorithm_name, ep)
                    
            except Exception as e:
                print(f"Episode {ep} 训练失败: {str(e)}")
                continue
                
    def _switch_scenario(self, episode):
        """在合作和竞争场景之间切换"""
        if episode % self.scenario_switch_interval == 0:
            try:
                if episode // self.scenario_switch_interval % 2 == 0:
                    if hasattr(self.env, 'set_cooperation_mode'):
                        self.env.set_cooperation_mode(True)
                    else:
                        print("警告: 环境不支持场景切换")
                else:
                    if hasattr(self.env, 'set_cooperation_mode'):
                        self.env.set_cooperation_mode(False)
                    else:
                        print("警告: 环境不支持场景切换")
            except Exception as e:
                print(f"场景切换失败: {e}")
                
    def _train_episode(self, agents, episode):
        """训练一个episode"""
        try:
            obs = self.env.reset()
        except Exception as e:
            print(f"错误: Episode {episode} - 环境重置失败: {e}")
            return None
            
        if obs is None:
            print(f"警告: Episode {episode} - 环境重置返回None")
            return None
        
        # 验证观察值的有效性
        if not isinstance(obs, list) or len(obs) != len(agents):
            print(f"警告: Episode {episode} - 观察值格式无效: {obs}")
            return None
        
        if any(obs_val is None for obs_val in obs):
            print(f"警告: Episode {episode} - 观察值包含None: {obs}")
            # 尝试修复观察值
            obs = [[0, 0] if obs_val is None else obs_val for obs_val in obs]
            
        done = False
        total_reward = 0
        step_count = 0
        episode_data = defaultdict(list)
        
        max_steps = self.cfg.get("env", {}).get("max_steps", 1000)  # 默认最大步数
        while not done and step_count < max_steps:
            actions = []
            decisions = []
            regrets = []
            
            # 初始化actions列表，确保所有智能体都有默认动作
            for _ in range(len(agents)):
                actions.append(0)  # 默认动作
                decisions.append('wait')  # 默认决策
                regrets.append(0.0)  # 默认后悔值
            
            # 收集所有智能体的动作
            for i, agent in enumerate(agents):
                if obs is None or i >= len(obs):
                    print(f"警告: Episode {episode} Step {step_count} - 观察值无效")
                    break
                
                # 确保当前智能体的观察值不是None
                if obs[i] is None:
                    print(f"警告: Episode {episode} Step {step_count} - 智能体 {i} 的观察值是None")
                    continue
                    
                other_agents_obs = [obs[j] for j in range(len(agents)) if j != i and j < len(obs) and obs[j] is not None]
                # 构建other_agents_actions，现在actions列表已经预初始化
                other_agents_actions = []
                for j in range(len(agents)):
                    if j != i:
                        # actions[j]现在应该总是有值，但为了安全起见仍然检查
                        if actions[j] is not None:
                            other_agents_actions.append(actions[j])
                        else:
                            print(f"警告: 智能体 {j} 的动作是None，使用默认动作0")
                            other_agents_actions.append(0)
                
                # 最终检查：确保other_agents_actions中没有None值
                if any(action is None for action in other_agents_actions):
                    print(f"警告: other_agents_actions中仍然包含None值: {other_agents_actions}")
                    other_agents_actions = [0 if action is None else action for action in other_agents_actions]
                
                # 最终验证：确保所有输入都是有效的
                if obs[i] is None:
                    print(f"错误: 智能体 {i} 的观察值仍然是None，跳过此智能体")
                    continue
                
                if any(obs_val is None for obs_val in other_agents_obs):
                    print(f"错误: other_agents_obs中包含None值: {other_agents_obs}")
                    other_agents_obs = [obs_val if obs_val is not None else [0, 0] for obs_val in other_agents_obs]
                
                try:
                    result = agent.act(obs[i], other_agents_obs, other_agents_actions)
                    if isinstance(result, tuple) and len(result) == 3:
                        action, decision, regret = result
                    else:
                        print(f"警告: 智能体 {i} 返回格式异常: {result}")
                        action, decision, regret = 0, 'wait', 0.0
                    
                    # 确保action不是None
                    if action is None:
                        print(f"警告: 智能体 {i} 返回了None动作，使用默认动作0")
                        action = 0
                        
                    # 更新预初始化的列表
                    actions[i] = action
                    decisions[i] = decision
                    regrets[i] = regret
                    
                    # 调试输出 - 减少频率
                    if step_count == 0 and episode % 100 == 0:  # 每100个episode的第一步才输出
                        print(f"智能体 {i} 返回: action={action}, decision={decision}, regret={regret}")
                        
                except Exception as e:
                    print(f"智能体 {i} 动作选择失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # 使用默认动作
                    actions[i] = 0  # 更新预初始化的actions列表
                    decisions[i] = 'wait'
                    regrets[i] = 0.0
                
            # 最终验证：确保所有数据都是有效的
            if any(action is None for action in actions):
                print(f"警告: actions列表中包含None值: {actions}")
                actions = [0 if action is None else action for action in actions]
            
            if any(decision is None for decision in decisions):
                print(f"警告: decisions列表中包含None值: {decisions}")
                decisions = ['wait' if decision is None else decision for decision in decisions]
            
            if any(regret is None for regret in regrets):
                print(f"警告: regrets列表中包含None值: {regrets}")
                regrets = [0.0 if regret is None else regret for regret in regrets]
            
            # 环境步进
            try:
                # 确保actions列表长度正确且不包含None值
                if len(actions) != len(agents):
                    print(f"警告: actions列表长度不正确: {len(actions)} != {len(agents)}")
                    actions = actions[:len(agents)] if len(actions) > len(agents) else actions + [0] * (len(agents) - len(actions))
                
                # 确保actions列表中的所有值都是有效的
                if any(not isinstance(action, (int, float)) for action in actions):
                    print(f"警告: actions列表包含无效值: {actions}")
                    actions = [0 if not isinstance(action, (int, float)) else action for action in actions]
                
                new_obs, rewards, done, info = self.env.step(actions)
                
                # 验证返回值
                if new_obs is None or rewards is None:
                    print(f"警告: Episode {episode} Step {step_count} - 环境返回无效值")
                    break
                
                # 确保new_obs是有效的
                if not isinstance(new_obs, list) or len(new_obs) != len(agents):
                    print(f"警告: Episode {episode} Step {step_count} - new_obs格式无效: {new_obs}")
                    break
                
                if any(obs_val is None for obs_val in new_obs):
                    print(f"警告: Episode {episode} Step {step_count} - new_obs包含None值: {new_obs}")
                    # 尝试修复观察值
                    new_obs = [[0, 0] if obs_val is None else obs_val for obs_val in new_obs]
                    
                # 确保rewards是列表
                if not isinstance(rewards, list):
                    print(f"警告: Episode {episode} Step {step_count} - rewards不是列表: {type(rewards)}")
                    rewards = [0.0] * len(agents)
                elif len(rewards) != len(agents):
                    print(f"警告: Episode {episode} Step {step_count} - rewards长度不正确: {len(rewards)} != {len(agents)}")
                    rewards = rewards[:len(agents)] if len(rewards) > len(agents) else rewards + [0.0] * (len(agents) - len(rewards))
                
                # 确保rewards中的所有值都是有效的
                if any(not isinstance(reward, (int, float)) for reward in rewards):
                    print(f"警告: Episode {episode} Step {step_count} - rewards包含无效值: {rewards}")
                    rewards = [0.0 if not isinstance(reward, (int, float)) else reward for reward in rewards]
                    
                # 更新智能体
                for i, agent in enumerate(agents):
                    if i < len(rewards) and hasattr(agent, 'update_reward'):
                        try:
                            agent.update_reward(rewards[i])
                        except Exception as e:
                            print(f"智能体 {i} 更新奖励失败: {e}")
                    
                # 记录数据 - 确保数据是有效的
                if actions and all(action is not None for action in actions):
                    episode_data['actions'].extend(actions)
                if decisions and all(decision is not None for decision in decisions):
                    episode_data['decisions'].extend(decisions)
                if regrets and all(regret is not None for regret in regrets):
                    episode_data['regrets'].extend(regrets)
                if rewards and all(reward is not None for reward in rewards):
                    episode_data['rewards'].extend(rewards)
                if obs and all(obs_val is not None for obs_val in obs):
                    episode_data['observations'].extend(obs)
                
                total_reward += sum(rewards)
                step_count += 1
                obs = new_obs  # 使用新的观察值
                
                # 确保done是布尔值
                if not isinstance(done, bool):
                    print(f"警告: Episode {episode} Step {step_count} - done不是布尔值: {type(done)}")
                    done = bool(done)
                
            except Exception as e:
                print(f"环境步进失败: {e}")
                import traceback
                traceback.print_exc()
                break
            
        episode_data['total_reward'] = total_reward
        episode_data['steps'] = step_count
        # 安全地获取场景模式
        try:
            if hasattr(self.env, 'cooperation_mode'):
                episode_data['scenario_mode'] = 'cooperation' if self.env.cooperation_mode else 'competition'
            else:
                episode_data['scenario_mode'] = 'unknown'
        except Exception as e:
            print(f"获取场景模式失败: {e}")
            episode_data['scenario_mode'] = 'unknown'
        
        # 最终验证：确保episode_data包含所有必要的数据
        required_keys = ['actions', 'decisions', 'regrets', 'rewards', 'observations']
        for key in required_keys:
            if key not in episode_data or not episode_data[key]:
                print(f"警告: Episode {episode} - 缺少数据: {key}")
                episode_data[key] = []
        
        return episode_data
        
    def _update_networks(self, agents, optimizers, episode_data):
        """更新网络参数"""
        if episode_data is None:
            print("警告: episode_data为空，跳过网络更新")
            return
            
        # 这里简化处理，实际应该使用经验回放和批量更新
        for i, (agent, optimizer) in enumerate(zip(agents, optimizers)):
            if optimizer is None:
                continue
                
            # 简化的网络更新（实际应该更复杂）
            if hasattr(agent, 'policy_net') and hasattr(agent, 'critic'):
                # 计算损失（简化版本）
                loss = torch.tensor(0.0, requires_grad=True)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
    def _record_statistics(self, algorithm_name, episode_data, episode):
        """记录统计信息"""
        if episode_data is None:
            print(f"警告: Episode {episode} 数据为空，跳过统计记录")
            return
            
        results = self.results[algorithm_name]
        
        # 记录奖励
        if 'total_reward' in episode_data and episode_data['total_reward'] is not None:
            results['episode_rewards'].append(float(episode_data['total_reward']))
        
        # 记录后悔
        if 'regrets' in episode_data and episode_data['regrets'] and all(regret is not None for regret in episode_data['regrets']):
            avg_regret = np.mean([float(r) for r in episode_data['regrets']])
            results['episode_regrets'].append(avg_regret)
        
        # 记录决策分布
        if 'decisions' in episode_data and episode_data['decisions']:
            for decision in episode_data['decisions']:
                if decision is not None:
                    if decision not in results['episode_decisions']:
                        results['episode_decisions'][decision] = []
                    results['episode_decisions'][decision].append(1)
            
        # 记录场景切换性能
        if 'scenario_mode' in episode_data and 'total_reward' in episode_data:
            if episode_data['scenario_mode'] == 'cooperation':
                results['scenario_switch_performance'].append({
                    'episode': episode,
                    'mode': 'cooperation',
                    'reward': float(episode_data['total_reward'])
                })
            else:
                results['scenario_switch_performance'].append({
                    'episode': episode,
                    'mode': 'competition',
                    'reward': float(episode_data['total_reward'])
                })
            
    def _print_training_info(self, algorithm_name, episode):
        """打印训练信息"""
        results = self.results[algorithm_name]
        
        if results and len(results.get('episode_rewards', [])) > 0:
            recent_rewards = results['episode_rewards'][-100:]
            avg_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            
            print(f"算法 {algorithm_name} - Episode {episode}: "
                  f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
                  
    def get_comparison_results(self):
        """获取对比结果"""
        comparison_results = {}
        
        for algorithm_name in self.algorithms_to_test:
            results = self.results[algorithm_name]
            
            # 检查是否有有效的训练数据
            if not results or len(results.get('episode_rewards', [])) == 0:
                print(f"警告: 算法 {algorithm_name} 没有有效的训练数据")
                continue
                
            # 计算性能指标
            final_rewards = results.get('episode_rewards', [])[-100:]  # 最后100个episode
            final_regrets = results.get('episode_regrets', [])[-100:] if results.get('episode_regrets') else []
            
            # 确保数据有效
            if not final_rewards:
                print(f"警告: 算法 {algorithm_name} 没有有效的奖励数据")
                continue
                
            comparison_results[algorithm_name] = {
                # 完整训练过程数据
                'episode_rewards': results.get('episode_rewards', []),
                'episode_regrets': results.get('episode_regrets', []),
                'scenario_switch_performance': results.get('scenario_switch_performance', []),
                
                # 最终统计指标
                'final_avg_reward': float(np.mean(final_rewards)),
                'final_std_reward': float(np.std(final_rewards)),
                'final_avg_regret': float(np.mean(final_regrets)) if final_regrets else 0.0,
                'training_time': float(results.get('training_time', 0.0)),
                'total_episodes': len(results.get('episode_rewards', [])),
                
                # 决策分布（简化格式）
                'decision_distribution': self._simplify_decision_distribution(results.get('episode_decisions', {})),
                
                # 场景性能分析
                'scenario_performance': self._analyze_scenario_performance(results)
            }
            
        return comparison_results
        
    def _analyze_scenario_performance(self, results):
        """分析场景切换性能"""
        scenario_perf = results.get('scenario_switch_performance', [])
        
        if not scenario_perf:
            return {}
            
        # 分析合作和竞争场景下的性能
        coop_rewards = [p['reward'] for p in scenario_perf if p['mode'] == 'cooperation']
        comp_rewards = [p['reward'] for p in scenario_perf if p['mode'] == 'competition']
        
        return {
            'cooperation_avg_reward': float(np.mean(coop_rewards)) if coop_rewards else 0.0,
            'competition_avg_reward': float(np.mean(comp_rewards)) if comp_rewards else 0.0,
            'scenario_adaptation': len(scenario_perf)  # 场景切换次数
        }
        
    def save_results(self, filename="comparison_results.yaml"):
        """保存对比结果"""
        comparison_results = self.get_comparison_results()
        
        # 转换为可序列化的格式
        serializable_results = {}
        for alg_name, results in comparison_results.items():
            serializable_results[alg_name] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[alg_name][key] = value.tolist()
                elif isinstance(value, np.integer):
                    serializable_results[alg_name][key] = int(value)
                elif isinstance(value, np.floating):
                    serializable_results[alg_name][key] = float(value)
                elif isinstance(value, list):
                    # 处理列表中的numpy类型
                    serializable_results[alg_name][key] = self._convert_list_types(value)
                else:
                    serializable_results[alg_name][key] = value
                    
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(serializable_results, f, default_flow_style=False, allow_unicode=True)
            
        print(f"对比结果已保存到: {filename}")
        
    def _convert_list_types(self, data_list):
        """转换列表中的数据类型为可序列化格式"""
        converted = []
        for item in data_list:
            if isinstance(item, np.ndarray):
                converted.append(item.tolist())
            elif isinstance(item, np.integer):
                converted.append(int(item))
            elif isinstance(item, np.floating):
                converted.append(float(item))
            elif isinstance(item, dict):
                converted.append(self._convert_dict_types(item))
            elif isinstance(item, list):
                converted.append(self._convert_list_types(item))
            else:
                converted.append(item)
        return converted
        
    def _convert_dict_types(self, data_dict):
        """转换字典中的数据类型为可序列化格式"""
        converted = {}
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                converted[key] = value.tolist()
            elif isinstance(value, np.integer):
                converted[key] = int(value)
            elif isinstance(value, np.floating):
                converted[key] = float(value)
            elif isinstance(value, dict):
                converted[key] = self._convert_dict_types(value)
            elif isinstance(value, list):
                converted[key] = self._convert_list_types(value)
            else:
                converted[key] = value
        return converted
        
    def print_summary(self):
        """打印对比总结"""
        comparison_results = self.get_comparison_results()
        
        print("\n" + "=" * 80)
        print("算法对比总结")
        print("=" * 80)
        
        # 按最终平均奖励排序
        sorted_algorithms = sorted(
            comparison_results.items(),
            key=lambda x: x[1]['final_avg_reward'],
            reverse=True
        )
        
        for i, (alg_name, results) in enumerate(sorted_algorithms):
            print(f"\n{i+1}. {alg_name}")
            print(f"   最终平均奖励: {results['final_avg_reward']:.2f} ± {results['final_std_reward']:.2f}")
            print(f"   最终平均后悔: {results['final_avg_regret']:.4f}")
            print(f"   训练时间: {results['training_time']:.2f}秒")
            print(f"   总episodes: {results['total_episodes']}")
            
            # 决策分布
            decision_dist = results['decision_distribution']
            if decision_dist:
                print(f"   决策分布: {decision_dist}")
                
            # 场景性能
            scenario_perf = results['scenario_performance']
            if scenario_perf:
                print(f"   合作场景奖励: {scenario_perf['cooperation_avg_reward']:.2f}")
                print(f"   竞争场景奖励: {scenario_perf['competition_avg_reward']:.2f}")
                
        print("\n" + "=" * 80)

    def _simplify_decision_distribution(self, episode_decisions):
        """简化决策分布数据格式"""
        simplified = {}
        for decision, counts in episode_decisions.items():
            if counts and len(counts) > 0:
                simplified[decision] = len(counts)  # 只记录总数，而不是详细的1列表
        return simplified
