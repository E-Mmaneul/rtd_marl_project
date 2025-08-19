#!/usr/bin/env python3
"""
RTD项目运行脚本 - 包含错误处理和调试信息
"""
import sys
import os
import traceback

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """检查依赖"""
    print("=== 检查依赖 ===")
    
    missing_deps = []
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        missing_deps.append("torch")
        print("❌ PyTorch 未安装")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        missing_deps.append("numpy")
        print("❌ NumPy 未安装")
    
    try:
        import yaml
        print("✓ PyYAML")
    except ImportError:
        missing_deps.append("pyyaml")
        print("❌ PyYAML 未安装")
    
    if missing_deps:
        print(f"\n❌ 缺少依赖: {missing_deps}")
        print("请运行: python install_dependencies.py")
        return False
    
    print("✅ 所有依赖检查通过")
    return True

def check_project_structure():
    """检查项目结构"""
    print("\n=== 检查项目结构 ===")
    
    required_files = [
        "main.py",
        "config/experiment_config.yaml",
        "envs/gridworld.py",
        "agents/rtd_agent.py",
        "agents/regret_module.py",
        "agents/networks.py",
        "trainers/rtd_trainer.py",
        "utils/delay_queue.py",
        "utils/communication_utils.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path}")
    
    if missing_files:
        print(f"\n❌ 缺少文件: {missing_files}")
        return False
    
    print("✅ 项目结构检查通过")
    return True

def test_imports():
    """测试导入"""
    print("\n=== 测试导入 ===")
    
    try:
        from envs.gridworld import GridWorld
        print("✓ GridWorld 导入成功")
        
        from agents.rtd_agent import RTDAgent
        print("✓ RTDAgent 导入成功")
        
        from trainers.rtd_trainer import RTDTrainer
        print("✓ RTDTrainer 导入成功")
        
        print("✅ 所有模块导入成功")
        return True
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        traceback.print_exc()
        return False

def run_main():
    """运行主程序"""
    print("\n=== 运行主程序 ===")
    
    try:
        # 导入主程序
        import main
        
        print("✅ 主程序运行成功")
        return True
        
    except Exception as e:
        print(f"❌ 主程序运行失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("RTD增强版多智能体强化学习项目")
    print("=" * 50)
    
    # 1. 检查依赖
    if not check_dependencies():
        return False
    
    # 2. 检查项目结构
    if not check_project_structure():
        return False
    
    # 3. 测试导入
    if not test_imports():
        return False
    
    # 4. 运行主程序
    if not run_main():
        return False
    
    print("\n🎉 项目运行成功！")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ 项目运行失败，请检查上述错误信息")
        sys.exit(1)
