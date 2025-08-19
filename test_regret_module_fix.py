#!/usr/bin/env python3
"""
测试RegretModule修复后的功能
"""

def test_regret_module_import():
    """测试RegretModule是否可以正常导入"""
    try:
        from agents.regret_module import RegretModule
        print("✓ RegretModule导入成功")
        return True
    except Exception as e:
        print(f"✗ RegretModule导入失败: {e}")
        return False

def test_regret_module_initialization():
    """测试RegretModule是否可以正常初始化"""
    try:
        from agents.regret_module import RegretModule
        
        # 创建配置
        config = {
            "env": {"num_agents": 3},  # 测试3个智能体
            "rtd": {
                "alpha_base": 0.1,
                "beta_base": 0.2,
                "alpha_delta": 0.05,
                "beta_delta": 0.05,
                "individual_regret_weight": 0.6,
                "team_regret_weight": 0.4,
                "cooperation_window": 10,
                "policy_change_window": 20,
                "cooperation_smoothing": 0.9
            }
        }
        
        # 初始化RegretModule
        regret_module = RegretModule(config, agent_id=0)
        
        # 验证智能体数量
        assert regret_module.num_agents == 3, f"期望智能体数量为3，实际为{regret_module.num_agents}"
        
        print("✓ RegretModule初始化成功，智能体数量正确")
        return True
        
    except Exception as e:
        print(f"✗ RegretModule初始化失败: {e}")
        return False

def test_joint_actions_construction():
    """测试联合动作构建逻辑"""
    try:
        from agents.regret_module import RegretModule
        import torch
        
        # 创建配置
        config = {
            "env": {"num_agents": 4},  # 测试4个智能体
            "rtd": {
                "alpha_base": 0.1,
                "beta_base": 0.2,
                "alpha_delta": 0.05,
                "beta_delta": 0.05,
                "individual_regret_weight": 0.6,
                "team_regret_weight": 0.4,
                "cooperation_window": 10,
                "policy_change_window": 20,
                "cooperation_smoothing": 0.9
            }
        }
        
        # 初始化RegretModule
        regret_module = RegretModule(config, agent_id=0)
        
        # 测试不同情况下的联合动作构建
        test_cases = [
            ([], 1),  # 没有其他动作
            ([2], 1),  # 1个其他动作
            ([2, 3], 1),  # 2个其他动作
            ([2, 3, 4], 1),  # 3个其他动作
        ]
        
        for other_actions, action in test_cases:
            # 模拟状态
            state = torch.randn(1, 2)  # [batch_size, state_dim]
            
            # 模拟critic（返回随机值）
            class MockCritic:
                def __call__(self, states, actions):
                    return torch.randn(1)
            
            critic = MockCritic()
            
            # 测试个体后悔计算
            try:
                regret = regret_module.compute_individual_regret(state, action, other_actions, critic)
                print(f"✓ 联合动作构建测试通过: other_actions={other_actions}, action={action}")
            except Exception as e:
                print(f"✗ 联合动作构建测试失败: other_actions={other_actions}, action={action}, 错误: {e}")
                return False
        
        print("✓ 所有联合动作构建测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 联合动作构建测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试RegretModule修复...")
    print("=" * 50)
    
    tests = [
        test_regret_module_import,
        test_regret_module_initialization,
        test_joint_actions_construction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！RegretModule修复成功")
    else:
        print("❌ 部分测试失败，需要进一步检查")
    
    return passed == total

if __name__ == "__main__":
    main()
