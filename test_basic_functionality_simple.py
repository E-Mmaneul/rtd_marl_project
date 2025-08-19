#!/usr/bin/env python3
"""
简化版基础功能测试脚本
不依赖PyTorch，只测试环境基本功能
"""
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_environment_only():
    """只测试环境基本功能（不依赖PyTorch）"""
    print("=" * 50)
    print("测试环境基本功能（简化版）")
    print("=" * 50)
    
    try:
        # 测试环境导入
        from envs.gridworld import GridWorld
        print("✓ 环境模块导入成功")
        
        # 创建环境
        env = GridWorld(size=3, num_agents=2, max_steps=10)
        print("✓ 环境创建成功")
        
        # 测试重置
        obs = env.reset()
        print(f"✓ 环境重置成功，观察值: {obs}")
        print(f"✓ 观察值类型: {type(obs)}, 长度: {len(obs) if obs else 'None'}")
        
        # 测试步进
        actions = [0, 1]  # 两个智能体的动作
        result = env.step(actions)
        print(f"✓ 环境步进成功，返回值: {result}")
        print(f"✓ 返回值类型: {type(result)}, 长度: {len(result)}")
        
        # 验证返回值
        if len(result) == 4:
            new_obs, rewards, done, info = result
            print(f"✓ 新观察值: {new_obs}")
            print(f"✓ 奖励: {rewards}")
            print(f"✓ 完成状态: {done}")
            print(f"✓ 信息: {info}")
        else:
            print(f"✗ 返回值长度不正确: {len(result)}")
            return False
            
        # 测试多步交互
        print("\n--- 测试多步交互 ---")
        obs = env.reset()
        for step in range(3):
            actions = [step % 5, (step + 1) % 5]  # 简单的动作序列
            new_obs, rewards, done, info = env.step(actions)
            print(f"Step {step + 1}: 动作={actions}, 奖励={rewards}, 完成={done}")
            obs = new_obs
            if done:
                break
        
        print("✓ 多步交互测试成功")
        return True
        
    except Exception as e:
        print(f"✗ 环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """测试各个模块的导入"""
    print("\n" + "=" * 50)
    print("测试模块导入")
    print("=" * 50)
    
    modules_to_test = [
        ("envs.gridworld", "GridWorld环境"),
        ("agents.rtd_agent", "RTD智能体"),
        ("agents.baseline_agents", "基线智能体"),
        ("trainers.comparison_trainer", "对比训练器"),
        ("utils.delay_queue", "延迟队列工具"),
        ("utils.regret_utils", "后悔计算工具")
    ]
    
    passed = 0
    total = len(modules_to_test)
    
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {description} 导入成功")
            passed += 1
        except ImportError as e:
            print(f"✗ {description} 导入失败: {e}")
        except Exception as e:
            print(f"✗ {description} 导入异常: {e}")
    
    print(f"\n导入测试结果: {passed}/{total} 成功")
    return passed == total

def main():
    """主测试函数"""
    print("开始简化版基础功能测试...")
    print("注意：此版本不测试需要PyTorch的智能体功能")
    
    tests = [
        test_imports,
        test_environment_only
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✓ 测试通过")
            else:
                print("✗ 测试失败")
        except Exception as e:
            print(f"✗ 测试异常: {e}")
    
    print(f"\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 基础测试通过！环境功能正常")
        print("💡 要测试智能体功能，需要安装PyTorch: pip install torch")
        return True
    else:
        print("❌ 部分测试失败，需要检查代码")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
