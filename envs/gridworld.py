import random
import numpy as np

class GridWorld:
    def __init__(self, size=5, num_agents=2, max_steps=50):
        self.size = size
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.cooperation_mode = True  # 默认合作模式
        self.reset()

    def reset(self):
        # 起点随机化
        self.agent_positions = [
            (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            for _ in range(self.num_agents)
        ]
        self.current_step = 0
        return self._get_obs()

    def set_cooperation_mode(self, is_cooperation):
        """
        设置合作/竞争模式
        """
        self.cooperation_mode = is_cooperation
        print(f"环境模式切换为: {'合作' if is_cooperation else '竞争'}")

    def _get_obs(self):
        # 返回每个智能体的观察列表
        return [list(pos) for pos in self.agent_positions]

    def _manhattan_distance(self, pos):
        goal = (self.size - 1, self.size - 1)
        return abs(goal[0] - pos[0]) + abs(goal[1] - pos[1])

    def _calculate_cooperation_reward(self, positions, actions):
        """
        计算合作模式下的奖励
        """
        rewards = []
        goal = (self.size - 1, self.size - 1)
        
        # 计算团队协作奖励
        team_progress = 0
        for pos in positions:
            dist = self._manhattan_distance(pos)
            team_progress += (self.size * 2 - dist)  # 越接近目标，进度越高
        
        # 协作奖励：团队整体进度
        cooperation_bonus = team_progress / (self.num_agents * self.size * 2)
        
        for i, (pos, action) in enumerate(zip(positions, actions)):
            old_pos = self.agent_positions[i]
            old_dist = self._manhattan_distance(old_pos)
            new_dist = self._manhattan_distance(pos)
            
            # 基础奖励
            if pos == goal:
                reward = 50.0 + cooperation_bonus * 10  # 大奖励 + 协作奖励
            elif new_dist < old_dist:
                reward = 1.0 + cooperation_bonus  # 靠近终点 + 协作奖励
            elif new_dist > old_dist:
                reward = -0.5 + cooperation_bonus * 0.5  # 远离终点但仍有协作奖励
            else:
                reward = -0.2 + cooperation_bonus * 0.3  # 原地但仍有协作奖励
            
            rewards.append(reward)
        
        return rewards

    def _calculate_competition_reward(self, positions, actions):
        """
        计算竞争模式下的奖励 - 修复版本
        """
        rewards = []
        goal = (self.size - 1, self.size - 1)
        
        # 计算竞争奖励：只有第一个到达目标的智能体获得大奖励
        first_to_goal = None
        for i, pos in enumerate(positions):
            if pos == goal and first_to_goal is None:
                first_to_goal = i
        
        for i, (pos, action) in enumerate(zip(positions, actions)):
            old_pos = self.agent_positions[i]
            old_dist = self._manhattan_distance(old_pos)
            new_dist = self._manhattan_distance(pos)
            
            # 基础奖励：基于距离变化
            if new_dist < old_dist:
                base_reward = 3.0  # 靠近终点
            elif new_dist > old_dist:
                base_reward = -1.0  # 远离终点
            else:
                base_reward = 0.0  # 原地
            
            # 目标到达奖励
            if pos == goal:
                if i == first_to_goal:
                    goal_bonus = 50.0  # 第一个到达目标的智能体
                else:
                    goal_bonus = 20.0   # 后续到达的智能体
            else:
                goal_bonus = 0.0
            
            # 进度奖励：基于当前距离
            progress_reward = max(0, (self.size * 2 - new_dist) * 0.5)
            
            # 竞争惩罚：如果其他智能体更接近目标
            other_agents_closer = 0
            for j, other_pos in enumerate(positions):
                if j != i:
                    other_dist = self._manhattan_distance(other_pos)
                    if other_dist < new_dist:
                        other_agents_closer += 1
            
            competition_penalty = other_agents_closer * 0.3
            
            # 总奖励
            total_reward = base_reward + goal_bonus + progress_reward - competition_penalty
            
            # 确保奖励不为负值（避免过度惩罚）
            total_reward = max(total_reward, -2.0)
            
            rewards.append(total_reward)
        
        return rewards

    def step(self, actions):
        if actions is None:
            print(f"[ERROR] 动作为None")
            return None, [0.0] * self.num_agents, True, {}
        
        # 确保动作数量正确
        if len(actions) != self.num_agents:
            print(f"[WARNING] 动作数量不匹配: {len(actions)} != {self.num_agents}, 动作: {actions}")
            # 补充缺失的动作
            if len(actions) < self.num_agents:
                actions = actions + [0] * (self.num_agents - len(actions))
            else:
                actions = actions[:self.num_agents]
            
        rewards = []
        done = False
        self.current_step += 1
        goal = (self.size - 1, self.size - 1)

        new_positions = []
        for idx, action in enumerate(actions):
            x, y = self.agent_positions[idx]
            old_dist = self._manhattan_distance((x, y))

            # 执行动作
            if action == 0:    # up
                y = max(y - 1, 0)
            elif action == 1:  # down
                y = min(y + 1, self.size - 1)
            elif action == 2:  # left
                x = max(x - 1, 0)
            elif action == 3:  # right
                x = min(x + 1, self.size - 1)
            elif action == 4:  # wait
                pass  # 保持原位置

            new_pos = (x, y)
            new_positions.append(new_pos)

        # 根据模式计算奖励
        if self.cooperation_mode:
            rewards = self._calculate_cooperation_reward(new_positions, actions)
        else:
            rewards = self._calculate_competition_reward(new_positions, actions)

        self.agent_positions = new_positions

        # 检查终止条件
        if self.cooperation_mode:
            # 合作模式：至少有一个智能体到达目标即可
            any_at_goal = any(pos == goal for pos in new_positions)
            if any_at_goal:
                done = True
        else:
            # 竞争模式：所有智能体都到达目标
            all_at_goal = all(pos == goal for pos in new_positions)
            if all_at_goal:
                done = True

        if self.current_step >= self.max_steps:
            done = True

        obs = self._get_obs()
        info = {
            'cooperation_mode': self.cooperation_mode,
            'team_progress': sum(self._manhattan_distance(pos) for pos in new_positions),
            'agents_at_goal': sum(1 for pos in new_positions if pos == goal)
        }

        # 调试输出（大幅减少频率）
        if self.current_step % 50 == 0 and self.current_step > 0:  # 每50步输出一次，且不是第一步
            mode_str = "合作" if self.cooperation_mode else "竞争"
            print(f"[DEBUG] {mode_str}模式 Step={self.current_step}, "
                  f"Positions={self.agent_positions}, Actions={actions}, "
                  f"Rewards={rewards}, Total={sum(rewards)}, Done={done}")
        
        # 验证返回值
        if obs is None or rewards is None:
            print(f"[ERROR] 环境返回无效值: obs={obs}, rewards={rewards}")
            # 提供默认值
            obs = self._get_obs()
            rewards = [0.0] * self.num_agents
        
        # 确保rewards是列表
        if not isinstance(rewards, list):
            print(f"[ERROR] rewards不是列表: {type(rewards)}")
            rewards = [0.0] * self.num_agents
            
        # 确保obs是列表
        if not isinstance(obs, list):
            print(f"[ERROR] obs不是列表: {type(obs)}")
            obs = self._get_obs()

        return obs, rewards, done, info
