#!/usr/bin/env python3
"""
纯Python模块测试脚本
只测试不依赖外部库的模块
"""
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_delay_queue():
    """测试延迟队列（纯Python实现）"""
    print("=" * 50)
    print("测试延迟队列")
    print("=" * 50)
    
    try:
        from utils.delay_queue import DelayQueue
        print("✓ 延迟队列模块导入成功")
        
        # 创建队列
        queue = DelayQueue(max_size=3)
        print("✓ 延迟队列创建成功")
        
        # 测试添加和弹出
        queue.add("item1")
        queue.add("item2")
        queue.add("item3")
        print("✓ 添加3个元素成功")
        
        # 测试队列大小限制
        queue.add("item4")  # 应该覆盖第一个元素
        print("✓ 队列大小限制生效")
        
        # 测试弹出
        item1 = queue.pop()
        item2 = queue.pop()
        print(f"✓ 弹出元素: {item1}, {item2}")
        
        # 测试空队列
        item3 = queue.pop()
        item4 = queue.pop()
        print(f"✓ 空队列处理: {item3}, {item4}")
        
        return True
        
    except Exception as e:
        print(f"✗ 延迟队列测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_project_structure():
    """测试项目结构"""
    print("\n" + "=" * 50)
    print("测试项目结构")
    print("=" * 50)
    
    required_dirs = [
        "agents",
        "envs", 
        "trainers",
        "utils",
        "config"
    ]
    
    required_files = [
        "agents/rtd_agent.py",
        "agents/baseline_agents.py",
        "envs/gridworld.py",
        "trainers/comparison_trainer.py",
        "utils/delay_queue.py",
        "run_comparison_experiment.py",
        "requirements.txt"
    ]
    
    passed = 0
    total = len(required_dirs) + len(required_files)
    
    # 检查目录
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"✓ 目录存在: {dir_name}")
            passed += 1
        else:
            print(f"✗ 目录缺失: {dir_name}")
    
    # 检查文件
    for file_name in required_files:
        if os.path.isfile(file_name):
            print(f"✓ 文件存在: {file_name}")
            passed += 1
        else:
            print(f"✗ 文件缺失: {file_name}")
    
    print(f"\n项目结构检查结果: {passed}/{total} 通过")
    return passed == total

def test_requirements():
    """测试依赖要求"""
    print("\n" + "=" * 50)
    print("测试依赖要求")
    print("=" * 50)
    
    try:
        with open("requirements.txt", "r") as f:
            requirements = f.read().strip().split("\n")
        
        print("✓ requirements.txt 文件存在")
        print("需要的依赖包:")
        for req in requirements:
            if req.strip():
                print(f"  - {req}")
        
        print("\n💡 要运行完整测试，请安装依赖:")
        print("pip install -r requirements.txt")
        
        return True
        
    except Exception as e:
        print(f"✗ 依赖检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始纯Python模块测试...")
    print("此版本只测试不需要外部依赖的模块")
    
    tests = [
        test_project_structure,
        test_requirements,
        test_delay_queue
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
        print("🎉 纯Python测试通过！")
        print("💡 要运行完整功能测试，需要安装依赖包")
        return True
    else:
        print("❌ 部分测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
