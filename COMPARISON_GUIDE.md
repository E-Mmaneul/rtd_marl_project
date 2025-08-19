# RTD增强版对比实验使用指南

## 🎯 概述

本指南介绍如何运行RTD增强版与基线算法的对比实验，以验证创新点的有效性。

## 📋 对比算法列表

### 基线算法
- **QMIX**: 价值分解方法
- **VDN**: 价值分解网络
- **COMA**: 反事实多智能体策略梯度
- **MADDPG**: 多智能体深度确定性策略梯度
- **IQL**: 独立Q学习
- **TarMAC**: 目标导向多智能体通信
- **LOLA**: 学习对手建模

### 消融研究版本
- **ablation_no_adaptive**: 无自适应机制
- **ablation_individual_regret_only**: 仅个体后悔
- **ablation_team_regret_only**: 仅团队后悔
- **ablation_no_delay_domain**: 无延迟域机制
- **ablation_no_centralized_critic**: 无集中式Critic
- **ablation_no_communication**: 无通信机制

## 🚀 快速开始

### 1. 运行快速对比实验
```bash
python run_comparison_experiment.py
```
这将运行一个简化的对比实验，使用较小的环境和较少的训练episodes。

### 2. 运行完整对比实验
```bash
python run_comparison_experiment.py --episodes 2000 --ablation
```

### 3. 自定义算法列表
```bash
python run_comparison_experiment.py --algorithms RTD QMIX VDN IQL --episodes 1500
```

## ⚙️ 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `config/improved_config.yaml` |
| `--episodes` | 训练episodes数量 | `1000` |
| `--algorithms` | 要测试的算法列表 | 所有基线算法 |
| `--ablation` | 是否包含消融研究 | `False` |
| `--output` | 结果输出文件名 | `comparison_results.yaml` |
| `--seed` | 随机种子 | `42` |

## 📊 结果分析

### 1. 查看对比结果
```bash
python visualize_comparison_results.py
```

### 2. 生成分析报告
```bash
python visualize_comparison_results.py --type all
```

### 3. 生成特定类型图表
```bash
# 性能对比图
python visualize_comparison_results.py --type performance

# 消融研究图
python visualize_comparison_results.py --type ablation
```

## 📈 性能指标

### 主要指标
- **最终平均奖励**: 最后100个episode的平均奖励
- **最终平均后悔**: 最后100个episode的平均后悔值
- **训练时间**: 完整训练所需时间
- **决策分布**: 各种决策类型的分布情况

### 场景适应性指标
- **合作场景性能**: 合作模式下的平均奖励
- **竞争场景性能**: 竞争模式下的平均奖励
- **场景切换次数**: 环境模式切换的次数

## 🔬 实验设计

### 环境设置
- **GridWorld**: 3x3网格环境
- **智能体数量**: 2个
- **最大步数**: 20步
- **场景切换**: 每100个episode切换一次

### 训练配置
- **学习率**: 0.0001
- **折扣因子**: 0.95
- **批量大小**: 16
- **随机种子**: 42

## 📁 输出文件

### 结果文件
- `comparison_results.yaml`: 详细的对比结果数据
- `comparison_analysis/`: 分析报告目录
  - `performance_comparison.png`: 性能对比图
  - `ablation_study.png`: 消融研究图
  - `analysis_report.txt`: 文本分析报告

### 中断/错误处理
- `*.interrupted`: 用户中断时的结果
- `*.error`: 训练错误时的结果

## 🎨 自定义配置

### 1. 修改配置文件
编辑 `config/improved_config.yaml` 来调整：
- 环境参数
- 训练参数
- RTD算法参数

### 2. 添加新算法
在 `agents/baseline_agents.py` 中添加新的基线算法：
```python
class NewAlgorithm(BaselineAgent):
    def __init__(self, obs_dim, action_dim, config, agent_id):
        super().__init__(obs_dim, action_dim, config, agent_id)
        # 实现算法逻辑
    
    def act(self, obs, other_agents_obs=None, other_agents_actions=None):
        # 实现动作选择逻辑
        pass
```

### 3. 添加新的消融研究
在 `agents/ablation_agents.py` 中添加新的消融版本：
```python
class NewAblationAgent(AblationAgent):
    def __init__(self, obs_dim, action_dim, config, agent_id):
        super().__init__(obs_dim, action_dim, config, agent_id, "new_ablation")
    
    def _apply_ablation(self):
        # 实现消融设置
        pass
```

## 🔍 故障排除

### 常见问题

#### 1. 内存不足
- 减少 `batch_size`
- 减少 `episodes` 数量
- 使用较小的环境

#### 2. 训练时间过长
- 减少 `episodes` 数量
- 使用较少的算法进行对比
- 关闭消融研究

#### 3. 结果文件损坏
- 检查磁盘空间
- 验证文件权限
- 重新运行实验

#### 4. 算法训练失败
- 检查依赖库版本
- 验证配置文件格式
- 查看错误日志

## 📚 扩展阅读

### 相关论文
- QMIX: [Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- COMA: [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- MADDPG: [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)

### 技术文档
- [PyTorch官方文档](https://pytorch.org/docs/)
- [多智能体强化学习教程](https://github.com/oxwhirl/marl)

## 🤝 贡献指南

欢迎提交问题和改进建议！

### 提交问题
1. 检查现有问题列表
2. 提供详细的错误信息
3. 包含环境配置和代码版本

### 贡献代码
1. Fork项目仓库
2. 创建功能分支
3. 提交Pull Request

## 📞 联系方式

如有问题，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者
- 参与项目讨论

---

**注意**: 本对比实验需要足够的计算资源和时间。建议先在小型配置上测试，确认无误后再运行完整实验。
