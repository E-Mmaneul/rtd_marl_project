# RTD性能提升指南

## 概述

本指南说明如何通过增加episodes数量和优化参数来提高RTD（Regret-based Three-way Decision）算法的性能。

## 主要改进策略

### 1. 增加Episodes数量

**当前配置**: 200 episodes
**改进配置**: 500-1000 episodes

**为什么有效**:
- RTD的后悔驱动学习需要更多时间积累经验
- 三路决策机制需要更多episodes来优化决策阈值
- 策略自适应参数需要更多时间稳定

### 2. 参数优化

#### 核心参数
- `regret_threshold`: 0.8 → 0.5 (降低遗憾阈值)
- `delay_queue_size`: 5 → 10 (增加延迟队列)
- `ensemble_size`: 2 → 4 (增加集成大小)

#### 策略自适应
- `lambda_kappa`: 0.8 → 0.5 (降低策略变化敏感度)
- `lambda_tau`: 0.3 → 0.2 (降低策略变化阈值)
- `policy_change_window`: 5 → 10 (增加策略变化窗口)

#### 遗憾权重
- `individual_regret_weight`: 0.5 → 0.7 (增加个体遗憾权重)
- `team_regret_weight`: 0.5 → 0.3 (减少团队遗憾权重)

#### 决策阈值
- `alpha_base`: 0.15 → 0.1 (降低accept阈值)
- `beta_base`: 0.6 → 0.5 (降低delay阈值)

### 3. 环境配置优化

- `grid_size`: 4 → 5 (增加网格大小)
- `max_steps`: 30 → 40 (增加最大步数)
- `scenario_switch_interval`: 25 → 50 (增加场景切换间隔)

## 使用方法

### 方法1: 交互式选择

运行主脚本，选择实验类型：
```bash
python run_comparison_experiment.py
```

选择:
- `1`: 快速对比实验 (500 episodes)
- `2`: 长期训练实验 (1000 episodes)

### 方法2: 直接使用配置文件

使用长期训练配置：
```bash
python run_comparison_experiment.py --config config/long_training_config.yaml
```

### 方法3: 修改现有配置

编辑 `run_comparison_experiment.py` 中的 `run_quick_comparison()` 函数，修改episodes数量。

## 预期性能提升

### 当前性能 (200 episodes)
- 最终平均奖励: 203.77
- 最终平均后悔: 0.4514
- 性能排名: 第3名

### 预期性能 (500-1000 episodes)
- 最终平均奖励: 300+
- 最终平均后悔: <0.2
- 性能排名: 前2名
- 决策分布更均衡
- 场景适应性显著增强

## 训练时间估算

- **200 episodes**: ~1分钟
- **500 episodes**: ~2.5分钟
- **1000 episodes**: ~5分钟

## 监控指标

### 关键指标
1. **奖励曲线**: 应该呈现稳定上升趋势
2. **后悔值**: 应该逐渐降低并稳定
3. **决策分布**: Accept/Delay/Reject应该更均衡
4. **场景适应性**: 合作/竞争模式切换性能

### 早停条件
- 连续100个episodes没有显著改善
- 后悔值稳定在0.2以下
- 奖励方差持续降低

## 故障排除

### 常见问题
1. **训练时间过长**: 减少episodes数量或使用更简单的环境
2. **性能不提升**: 检查参数配置，调整学习率
3. **内存不足**: 减少batch_size或ensemble_size

### 调试建议
1. 启用详细日志记录
2. 定期保存检查点
3. 可视化训练进度
4. 监控关键指标变化

## 高级优化

### 1. 自适应学习率
根据训练进度动态调整学习率

### 2. 课程学习
从简单任务开始，逐步增加难度

### 3. 多目标优化
同时优化奖励、后悔值和决策质量

### 4. 集成学习
结合多个RTD模型的结果

## 结论

通过增加episodes数量和优化参数，RTD算法的性能有望显著提升。关键是要平衡训练时间和性能提升，找到最适合的配置参数。

建议从500 episodes开始，根据结果决定是否增加到1000 episodes。同时密切关注训练过程中的关键指标，确保算法朝着预期方向发展。
