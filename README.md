# RTD增强版多智能体强化学习项目

## 📋 项目概述

本项目实现了基于RTD（Regret-based Three-way Decision）增强版多智能体强化学习系统，支持以下核心功能：

### 🎯 核心特性

1. **策略非平稳性自适应机制**
   - 基于KL散度检测其他智能体策略变化率
   - 动态调整接受/拒绝阈值
   - 自适应λ参数调整

2. **个体后悔+团队后悔混合驱动学习**
   - 个体后悔：`R_i^ind(s, a) = max_{a_i'} Q_tot(s, a_i', a_{-i}) - Q_tot(s, a_i, a_{-i})`
   - 团队后悔：`R^team(s, a) = max_{a'} Q_tot(s, a') - Q_tot(s, a)`
   - 混合后悔：`R_i^blend = w_i * R_i^ind + (1 - w_i) * R^team`

3. **合作/竞争动态切换**
   - 基于奖励协方差和边际增益检测合作概率
   - 动态阈值调整：合作时放宽拒绝条件，竞争时收紧接受条件
   - EMA平滑避免频繁抖动

4. **通信资源受限的延迟域机制**
   - VOI（Value of Information）估计：`VOI ≈ E[U|m] - E[U|Ø] - cost(m)`
   - 通信门控决策
   - 延迟域中的守护动作执行

## 📁 项目结构

```
rtd_marl_project/
├── agents/                          # 智能体相关模块
│   ├── __init__.py
│   ├── networks.py                  # 网络定义（策略、Critic、VOI估计器等）
│   ├── regret_module.py            # 后悔计算与三支决策逻辑
│   └── rtd_agent.py                # RTD智能体（整合所有功能）
├── config/                          # 配置文件
│   ├── __init__.py
│   ├── experiment_config.yaml      # 实验配置文件
│   └── improved_config.yaml        # 改进的配置文件
├── envs/                           # 环境模块
│   ├── __init__.py
│   └── gridworld.py               # GridWorld环境（支持合作/竞争模式）
├── trainers/                       # 训练器模块
│   ├── __init__.py
│   └── rtd_trainer.py             # RTD训练器
├── utils/                          # 工具模块
│   ├── __init__.py
│   ├── delay_queue.py             # 延迟队列实现
│   └── communication_utils.py     # 通信工具类
├── main.py                         # 主入口文件
├── main_improved.py               # 改进版主入口
├── run_project.py                 # 项目运行脚本（含错误处理）
├── visualize_results.py           # 结果可视化脚本
├── analyze_decision_stats.py      # 决策统计分析
├── debug_decision_stats.py        # 决策统计调试
├── install_dependencies.py        # 依赖安装脚本
├── install_guide.md               # 安装指南
├── conda_install_guide.md         # Conda安装指南
├── requirements.txt               # 依赖列表
└── README.md                      # 项目说明
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.9.0+
- NumPy 1.21.0+
- PyYAML 5.4.0+

### 安装依赖

#### 方案1：使用pip（推荐）

```bash
# 安装基础依赖
pip install -r requirements.txt

# 或者手动安装
pip install torch>=1.9.0 numpy>=1.21.0 pyyaml>=5.4.0
```

#### 方案2：使用conda（稳定）

```bash
# 创建新环境
conda create -n rtd_env python=3.8
conda activate rtd_env

# 安装PyTorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 安装其他依赖
conda install numpy pyyaml
```

#### 方案3：使用项目安装脚本

```bash
# 运行安装脚本
python install_dependencies.py
```

### 运行项目

#### 基础运行

```bash
# 使用默认配置运行
python main.py

# 使用改进配置运行
python main_improved.py

# 使用项目运行脚本（推荐，含错误处理）
python run_project.py
```

#### 可视化结果

```bash
# 生成训练结果可视化
python visualize_results.py
```

## ⚙️ 配置说明

### 基础配置

```yaml
env:
  name: GridWorld
  size: 5                    # 网格世界大小
  num_agents: 2              # 智能体数量
  max_steps: 50              # 最大步数

training:
  episodes: 1000             # 训练episodes
  gamma: 0.95                # 折扣因子
  lr: 0.0001                 # 学习率
  batch_size: 32             # 批次大小
```

### RTD增强功能配置

```yaml
rtd:
  # 基础后悔阈值
  regret_threshold: 1.0
  delay_queue_size: 10
  ensemble_size: 3
  
  # 策略非平稳性自适应参数
  policy_change_window: 10   # 策略变化检测窗口
  kl_threshold: 0.1          # KL散度阈值
  lambda_min: 0.1
  lambda_max: 2.0
  lambda_kappa: 1.0
  lambda_tau: 0.5
  
  # 个体后悔+团队后悔混合参数
  individual_regret_weight: 0.6
  team_regret_weight: 0.4
  regret_learning_rate: 0.01
  
  # 合作/竞争检测参数
  cooperation_window: 20
  reward_covariance_threshold: 0.1
  marginal_gain_threshold: 0.05
  cooperation_smoothing: 0.9  # EMA参数
  
  # 动态阈值调整参数
  alpha_base: 0.3            # 基础接受阈值
  beta_base: 0.7             # 基础拒绝阈值
  alpha_delta: 0.2           # 接受阈值调整幅度
  beta_delta: 0.2            # 拒绝阈值调整幅度
  
  # 通信资源限制参数
  communication_budget: 100
  message_cost: 1
  voi_threshold: 0.1
  
  # 延迟域参数
  delay_domain_steps: 3
  guardian_action_prob: 0.3
```

## 🔧 核心组件详解

### 1. 后悔模块 (RegretModule)

实现了后悔计算机制：

- **个体后悔计算**：基于反事实推理
- **团队后悔计算**：基于联合动作优化
- **混合后悔**：加权组合个体和团队后悔
- **动态阈值调整**：根据合作概率和策略变化率

### 2. 网络架构

- **PolicyNetwork**：策略网络，输出动作概率分布
- **EnsembleCritic**：集成Critic，提供价值估计和不确定性
- **CentralizedCritic**：集中式Critic，用于团队后悔计算
- **VOIEstimator**：信息价值估计器
- **CommunicationGate**：通信门控网络

### 3. 环境增强

GridWorld环境支持：

- **合作模式**：团队协作奖励，鼓励共同到达目标
- **竞争模式**：零和奖励，只有第一个到达目标的智能体获得大奖励
- **动态切换**：训练过程中自动在合作和竞争模式间切换

### 4. 通信机制

- **VOI估计**：评估消息的信息价值
- **通信预算管理**：限制通信成本
- **通信门控**：决定是否发送消息

## 📊 训练监控

训练过程中会输出以下信息：

- **智能体状态**：后悔值、合作概率、策略变化率等
- **决策统计**：accept/delay/reject决策分布
- **通信统计**：消息发送数量、VOI平均值等
- **场景切换**：合作/竞争模式切换信息

## 🎯 实验设计

### 消融研究

可以进行以下消融研究：

1. **无自适应机制**：固定λ参数
2. **只有个体后悔**：w_individual = 1.0
3. **只有团队后悔**：w_team = 1.0
4. **无延迟域**：直接使用accept/reject决策
5. **无集中式Critic**：完全去中心化

### 基线算法

- QMIX / VDN / COMA / MADDPG / IQL
- TarMAC / DIAL / IC3Net（带通信）
- LOLA / Opponent Modeling（非平稳性）

## 📈 性能指标

- **团队平均回报**：episodic return
- **平均/分位后悔**：`𝔼[R^blend]`曲线
- **策略稳定性**：验证期内return方差与最大回撤
- **通信代价**：消息条数/bits
- **切换响应能力**：合作→竞争切换时策略恢复速度
- **样本效率**：达到某阈值所需steps

## 🔍 调试和故障排除

### 常见问题

1. **依赖安装失败**
   - 参考 `install_guide.md` 和 `conda_install_guide.md`
   - 运行 `python install_dependencies.py`

2. **导入错误**
   - 确保在项目根目录运行
   - 检查Python路径设置

3. **训练失败**
   - 检查配置文件格式
   - 查看错误日志

### 调试工具

```bash
# 检查项目结构
python run_project.py

# 分析决策统计
python analyze_decision_stats.py

# 调试决策统计
python debug_decision_stats.py
```

## 📊 结果可视化

项目提供了完整的结果可视化功能：

```bash
# 生成所有可视化图表
python visualize_results.py
```

可视化内容包括：
- 训练进度图表
- 决策分析图表
- 性能指标对比
- 通信统计图表

## ⚠️ 主要风险点与对策

1. **计算复杂度**：使用样本近似、价值分解、稀疏计算
2. **误判合作/竞争**：平滑、滞后窗口、置信区间检验
3. **信号噪声**：归一化后悔、裁剪、EMA
4. **通信预算不足**：RL学习通信决策、启发式阈值
5. **论文说服力**：理论引导 + 多场景实证 + 消融研究 + 代码发布

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

### 贡献步骤

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

MIT License

## 📞 联系方式

如有问题，请通过以下方式联系：

- 提交Issue
- 发送邮件
- 项目讨论区

## 🔄 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 实现基础RTD功能
- 支持合作/竞争环境

### v1.1.0 (2024-01-15)
- 添加策略非平稳性自适应
- 实现个体+团队后悔混合
- 增强通信机制

### v1.2.0 (2024-01-30)
- 添加结果可视化
- 改进配置系统
- 优化训练稳定性
