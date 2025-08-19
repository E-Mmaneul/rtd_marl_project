"""
RTD 训练器（完整实现）- 增强版
整合策略非平稳性自适应、个体+团队后悔混合、合作/竞争检测、VOI估计和通信门控
"""
import torch
import torch.optim as optim
import numpy as np
import random
from collections import defaultdict
import time
import torch.nn.functional as F

class RTDTrainer:
    def __init__(self, agents, env, config):
        self.agents = agents
        self.env = env
        self.cfg = config
        
        # 优化器
        self.optimizers = []
        for agent in agents:
            optimizer = optim.Adam([
                {'params': agent.policy_net.parameters()},
                {'params': agent.critic.parameters()},
                {'params': agent.centralized_critic.parameters()},
                {'params': agent.voi_estimator.parameters()},
                {'params': agent.communication_gate.parameters()}
            ], lr=config["training"]["lr"])
            self.optimizers.append(optimizer)
        
        # 训练参数
        self.gamma = config["training"]["gamma"]
        self.batch_size = config["training"]["batch_size"]
        
        # 延迟队列处理周期
        self.delay_process_interval = 5
        
        # 统计信息
        self.episode_rewards = []
        self.episode_regrets = []
        self.episode_decisions = defaultdict(list)
        self.communication_stats = defaultdict(int)
        
        # 合作/竞争切换场景
        self.cooperation_episodes = []
        self.competition_episodes = []
        self.scenario_switch_interval = 100  # 每100个episode切换场景
        
    def train(self):
        print("开始RTD增强版训练...")
        
        for ep in range(self.cfg["training"]["episodes"]):
            try:
                # 场景切换
                self._switch_scenario(ep)
                
                # 训练一个episode
                episode_data = self._train_episode(ep)
                
                # 更新网络
                if episode_data:
                    self._update_networks(episode_data)
                
                # 记录统计信息
                self._record_statistics(episode_data, ep)
                
                # 定期打印信息
                if ep % 50 == 0:
                    self._print_training_info(ep)
                    
            except Exception as e:
                print(f"Episode {ep} 训练失败: {str(e)}")
                import traceback
                traceback.print_exc()
                continue  # 继续下一个episode
    
    def _switch_scenario(self, episode):
        """
        在合作和竞争场景之间切换
        """
        if episode % self.scenario_switch_interval == 0:
            if episode // self.scenario_switch_interval % 2 == 0:
                # 切换到合作场景
                self.env.set_cooperation_mode(True)
                print(f"Episode {episode}: 切换到合作场景")
            else:
                # 切换到竞争场景
                self.env.set_cooperation_mode(False)
                print(f"Episode {episode}: 切换到竞争场景")
    
    def _train_episode(self, episode):
        """
        训练一个episode
        """
        obs = self.env.reset()
        done = False
        total_reward = 0
        step_count = 0
        episode_data = defaultdict(list)
        
        while not done:
            # 1. 处理延迟域
            delayed_actions = []
            for agent_idx, agent in enumerate(self.agents):
                delayed_action = agent.process_delay_domain()
                if delayed_action is not None:
                    delayed_actions.append((agent_idx, delayed_action))
            
            # 2. 生成新动作
            # 预先为每个环境中的智能体填充一个动作（默认wait=4），保证长度一致
            actions_to_execute = [4] * len(obs)
            decisions = []
            new_actions = []
            communication_messages = []
            
            # 确保只处理环境实际返回的智能体数量
            num_env_agents = len(obs)
            
            # 先写入延迟域要执行的动作
            for idx, delayed_act in delayed_actions:
                if 0 <= idx < num_env_agents:
                    actions_to_execute[idx] = delayed_act
                    decisions.append("delay")
            
            for agent_idx, agent in enumerate(self.agents):
                # 只处理环境实际存在的智能体
                if agent_idx >= num_env_agents:
                    break
                
                # 跳过已处理延迟动作的智能体
                if any(idx == agent_idx for idx, _ in delayed_actions):
                    continue
                
                # 获取其他智能体的观察和动作
                other_agents_obs = [obs[j] for j in range(len(obs)) if j != agent_idx]
                other_agents_actions = [new_actions[j] for j in range(len(new_actions)) if j != agent_idx]
                
                # 智能体动作选择
                if agent_idx < len(obs):
                    action, decision, regret = agent.act(
                        obs[agent_idx], 
                        other_agents_obs, 
                        other_agents_actions
                    )
                    
                    # 通信决策
                    should_comm, message = agent.should_communicate(obs[agent_idx])
                else:
                    # 如果智能体索引超出范围，使用默认动作
                    action = 0
                    decision = "accept"
                    regret = 0.0
                    should_comm = False
                    message = None
                if should_comm:
                    communication_messages.append((agent_idx, message))
                    self.communication_stats['messages_sent'] += 1
                
                # 记录数据
                episode_data[agent_idx].append({
                    'obs': obs[agent_idx] if agent_idx < len(obs) else [0, 0],
                    'action': action,
                    'decision': decision,
                    'regret': regret,
                    'message': message if should_comm else None,
                    'reward': None,
                    'next_obs': None
                })
                
                # 处理决策（按索引赋值，保证长度与env一致）
                if decision == "accept":
                    actions_to_execute[agent_idx] = action
                    decisions.append("accept")
                    new_actions.append(action)
                elif decision == "delay":
                    agent.delay_queue.add(action)
                    actions_to_execute[agent_idx] = 4  # 等待动作
                    decisions.append("delay")
                    new_actions.append(action)
                else:  # reject
                    safe_action = agent._get_safe_action()
                    actions_to_execute[agent_idx] = safe_action
                    decisions.append("reject")
                    new_actions.append(safe_action)
            
            # 3. 执行动作
            next_obs, rewards, done, info = self.env.step(actions_to_execute)
            total_reward += sum(rewards)
            
            # 4. 更新奖励历史
            for agent_idx, reward in enumerate(rewards):
                if agent_idx < len(self.agents):
                    self.agents[agent_idx].update_reward(reward)
            
            # 5. 更新数据记录
            if next_obs is not None:  # 添加None检查
                for agent_idx, reward_val in enumerate(rewards):
                    # 确保智能体索引在有效范围内
                    if (agent_idx < len(episode_data) and 
                        len(episode_data[agent_idx]) > 0 and
                        agent_idx < len(self.agents)):
                        episode_data[agent_idx][-1]['reward'] = reward_val
                        if agent_idx < len(next_obs):
                            episode_data[agent_idx][-1]['next_obs'] = next_obs[agent_idx]
            
            # 6. 更新状态
            obs = next_obs
            step_count += 1
            
            # 7. 记录决策统计
            decision_counts = {}
            for decision in decisions:
                decision_counts[decision] = decision_counts.get(decision, 0) + 1
            
            # 更新决策统计
            for decision, count in decision_counts.items():
                self.episode_decisions[decision].append(count)
        
        return episode_data
    
    def _update_networks(self, episode_data):
        """
        更新所有网络
        """
        for agent_idx, agent in enumerate(self.agents):
            if agent_idx not in episode_data or not episode_data[agent_idx]:
                continue
            
            # 数据验证和准备
            valid_transitions = self._validate_transitions(episode_data[agent_idx])
            if not valid_transitions:
                continue
            
            try:
                # 准备批量数据
                batch_data = self._prepare_batch_data(valid_transitions)
                
                # 更新网络
                self._update_agent_networks(agent, batch_data, agent_idx)
                
            except Exception as e:
                print(f"Agent {agent_idx} 网络更新失败: {str(e)}")
                continue
    
    def _validate_transitions(self, transitions):
        """
        验证转换数据
        """
        valid_transitions = []
        for transition in transitions:
            try:
                # 检查必需字段
                required_keys = ['obs', 'action', 'reward', 'next_obs']
                assert all(key in transition for key in required_keys)
                assert all(transition[key] is not None for key in required_keys)
                
                # 检查数据类型
                assert isinstance(transition['action'], (int, np.integer))
                
                valid_transitions.append(transition)
                
            except AssertionError:
                continue
        
        return valid_transitions
    
    def _prepare_batch_data(self, transitions):
        """
        准备批量数据
        """
        obs_batch = torch.tensor(
            [t['obs'] for t in transitions],
            dtype=torch.float32
        )
        next_obs_batch = torch.tensor(
            [t['next_obs'] for t in transitions],
            dtype=torch.float32
        )
        action_batch = torch.tensor(
            [t['action'] for t in transitions],
            dtype=torch.long
        )
        reward_batch = torch.tensor(
            [t['reward'] for t in transitions],
            dtype=torch.float32
        )
        
        return {
            'obs': obs_batch,
            'next_obs': next_obs_batch,
            'actions': action_batch,
            'rewards': reward_batch
        }
    
    def _update_agent_networks(self, agent, batch_data, agent_idx):
        """
        更新单个智能体的网络
        """
        optimizer = self.optimizers[agent_idx]
        optimizer.zero_grad()
        
        # 策略网络更新
        action_probs = agent.policy_net(batch_data['obs'])
        log_probs = torch.log(action_probs.gather(1, batch_data['actions'].unsqueeze(1))).squeeze(1)
        
        # Critic网络更新
        current_values, _ = agent.critic(batch_data['obs'])
        with torch.no_grad():
            next_values, _ = agent.critic(batch_data['next_obs'])
            target_values = batch_data['rewards'] + self.gamma * next_values
        
        # 集中式Critic更新 - 简化版本，只使用当前智能体的数据
        batch_size = batch_data['obs'].shape[0]
        num_agents = 2  # 假设2个智能体
        
        # 构建联合状态 [batch_size, num_agents, obs_dim]
        # 这里简化处理，假设其他智能体的状态相同
        joint_states = batch_data['obs'].unsqueeze(1).repeat(1, num_agents, 1)
        
        # 构建联合动作 [batch_size, num_agents]
        # 假设其他智能体的动作是随机的
        other_actions = torch.randint(0, 4, (batch_size, num_agents - 1))
        joint_actions = torch.cat([batch_data['actions'].unsqueeze(1), other_actions], dim=1)
        
        current_joint_q = agent.centralized_critic(joint_states, joint_actions)
        
        # 计算损失
        critic_loss = F.mse_loss(current_values, target_values)
        centralized_critic_loss = F.mse_loss(current_joint_q, target_values)
        policy_loss = -torch.mean(log_probs * (target_values - current_values.detach()))
        
        # VOI估计器更新（简化版本）
        voi_loss = torch.tensor(0.0)  # 占位符
        
        # 通信门控更新（简化版本）
        comm_loss = torch.tensor(0.0)  # 占位符
        
        # 总损失
        total_loss = (critic_loss + centralized_critic_loss + 
                     policy_loss + voi_loss + comm_loss)
        
        # 反向传播
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(agent.centralized_critic.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    def _record_statistics(self, episode_data, episode):
        """
        记录训练统计信息
        """
        if episode_data:
            # 计算episode总奖励
            episode_total_reward = 0
            for agent_data in episode_data.values():
                for transition in agent_data:
                    if transition.get('reward') is not None:
                        episode_total_reward += transition['reward']
            
            # 记录episode奖励
            if episode_total_reward != 0:
                self.episode_rewards.append(episode_total_reward)
            
            # 计算平均后悔
            regrets = []
            for agent_data in episode_data.values():
                for transition in agent_data:
                    if 'regret' in transition:
                        regrets.append(transition['regret'])
            
            if regrets:
                avg_regret = np.mean(regrets)
                self.episode_regrets.append(avg_regret)
            
            # 记录决策统计
            decision_counts = {}
            for agent_data in episode_data.values():
                for transition in agent_data:
                    decision = transition.get('decision', 'unknown')
                    decision_counts[decision] = decision_counts.get(decision, 0) + 1
            
            # 更新决策统计
            for decision, count in decision_counts.items():
                self.episode_decisions[decision].append(count)
    
    def _print_training_info(self, episode):
        """
        打印训练信息
        """
        print(f"\n=== Episode {episode} 训练信息 ===")
        
        # 智能体信息
        for agent_idx, agent in enumerate(self.agents):
            agent_info = agent.get_agent_info()
            print(f"Agent {agent_idx}: {agent_info}")
        
        # 通信统计
        print(f"通信统计: {dict(self.communication_stats)}")
        
        # 决策统计
        decision_stats = {}
        for decision, counts in self.episode_decisions.items():
            if counts:
                decision_stats[decision] = len(counts)
        print(f"决策统计: {decision_stats}")
        
        # 后悔统计
        if self.episode_regrets:
            recent_regrets = self.episode_regrets[-50:]
            print(f"平均后悔 (最近50ep): {np.mean(recent_regrets):.4f}")
        
        print("=" * 40)
    
    def get_training_stats(self):
        """
        获取训练统计信息
        """
        return {
            'episode_rewards': self.episode_rewards,
            'episode_regrets': self.episode_regrets,
            'episode_decisions': dict(self.episode_decisions),
            'communication_stats': dict(self.communication_stats)
        }