#!/usr/bin/env python3
"""
RTD Enhanced Training Results Visualization Script
RTD增强版训练结果可视化脚本
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import yaml
import matplotlib.font_manager as fm

# Add current directory to Python path
# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Global variable to control whether to use Chinese
# 全局变量，用于控制是否使用中文
USE_CHINESE = True

def setup_chinese_font():
    """Setup Chinese font support"""
    global USE_CHINESE
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'STSong', 'NSimSun']
    font_found = False
    
    for font_name in chinese_fonts:
        try:
            # Check if font is available
            # 检查字体是否可用
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path != fm.rcParams['font.sans-serif']:
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                font_found = True
                print(f"✅ Using font: {font_name}")
                break
        except:
            continue
    
    if not font_found:
        # If no Chinese font is found, use default font and disable Chinese display
        # 如果没有找到中文字体，使用默认字体并禁用中文显示
        print("⚠️  No Chinese font found, will use English display")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        # Set global variable to control whether to use Chinese
        # 设置全局变量来控制是否使用中文
        USE_CHINESE = False
    else:
        USE_CHINESE = True
    
    plt.rcParams['axes.unicode_minus'] = False

# Set Chinese font
setup_chinese_font()

def get_title(title_chinese, title_english=None):
    """Return title based on font support"""
    if title_english is None:
        title_english = title_chinese
    return title_chinese if USE_CHINESE else title_english

def load_training_results():
    """Load training results"""
    try:
        # Try to load results from file, if not return sample data
        if os.path.exists('training_results.npy'):
            results = np.load('training_results.npy', allow_pickle=True).item()
            return results
        else:
            # Generate sample data for demonstration
            return generate_sample_results()
    except Exception as e:
        print(f"Failed to load training results: {e}")
        return generate_sample_results()

def generate_sample_results():
    """Generate sample training results"""
    episodes = 1000
    results = {
        'episode_rewards': np.random.normal(50, 10, episodes) + np.linspace(0, 20, episodes),
        'episode_regrets': np.random.exponential(0.5, episodes) * np.exp(-np.linspace(0, 3, episodes)),
        'episode_decisions': {
            'accept': np.random.poisson(15, episodes),
            'delay': np.random.poisson(8, episodes),
            'reject': np.random.poisson(5, episodes),
            'explore': np.random.poisson(3, episodes)
        },
        'communication_stats': {
            'messages_sent': np.cumsum(np.random.poisson(2, episodes)),
            'total_voi': np.cumsum(np.random.exponential(0.3, episodes))
        },
        'agent_info': {
            'cooperation_prob': np.random.beta(2, 2, episodes),
            'lambda_t': np.random.uniform(0.1, 2.0, episodes)
        }
    }
    return results

def plot_training_progress(results):
    """Plot training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(get_title('RTD Enhanced Training Progress', 'RTD Enhanced Training Progress'), fontsize=16, fontweight='bold')
    
    episodes = len(results['episode_rewards'])
    episode_range = range(1, episodes + 1)
    
    # 1. Reward curve
    ax1 = axes[0, 0]
    ax1.plot(episode_range, results['episode_rewards'], 'b-', alpha=0.7, linewidth=1)
    ax1.set_title(get_title('Episode Reward Change', 'Episode Reward Change'))
    ax1.set_xlabel(get_title('Episode', 'Episode'))
    ax1.set_ylabel(get_title('Reward', 'Reward'))
    ax1.grid(True, alpha=0.3)
    
    # Add moving average line
    window = min(50, episodes // 10)
    if window > 1:
        moving_avg = np.convolve(results['episode_rewards'], np.ones(window)/window, mode='valid')
        ax1.plot(range(window, episodes + 1), moving_avg, 'r-', linewidth=2, label=f'{window}{get_title(" Period Moving Average", " Period Moving Average")}')
        ax1.legend()
    
    # 2. Regret value change
    ax2 = axes[0, 1]
    ax2.plot(episode_range, results['episode_regrets'], 'g-', alpha=0.7, linewidth=1)
    ax2.set_title(get_title('Episode Regret Change', 'Episode Regret Change'))
    ax2.set_xlabel(get_title('Episode', 'Episode'))
    ax2.set_ylabel(get_title('Regret', 'Regret'))
    ax2.grid(True, alpha=0.3)
    
    # Add moving average line
    if window > 1:
        moving_avg_regret = np.convolve(results['episode_regrets'], np.ones(window)/window, mode='valid')
        ax2.plot(range(window, episodes + 1), moving_avg_regret, 'r-', linewidth=2, label=f'{window}{get_title(" Period Moving Average", " Period Moving Average")}')
        ax2.legend()
    
    # 3. Decision distribution
    ax3 = axes[1, 0]
    decision_data = results['episode_decisions']
    decision_names = list(decision_data.keys())
    decision_values = [np.sum(decision_data[name]) for name in decision_names]
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db']
    bars = ax3.bar(decision_names, decision_values, color=colors, alpha=0.8)
    ax3.set_title(get_title('Decision Distribution Statistics', 'Decision Distribution Statistics'))
    ax3.set_ylabel(get_title('Decision Count', 'Decision Count'))
    
    # Add value labels
    for bar, value in zip(bars, decision_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(decision_values)*0.01,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Communication statistics
    ax4 = axes[1, 1]
    comm_data = results['communication_stats']
    if 'messages_sent' in comm_data and 'total_voi' in comm_data:
        ax4.plot(episode_range, comm_data['messages_sent'], 'purple', label=get_title('Messages Sent', 'Messages Sent'), linewidth=2)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(episode_range, comm_data['total_voi'], 'orange', label=get_title('Total VOI', 'Total VOI'), linewidth=2)
        
        ax4.set_title(get_title('Communication Statistics', 'Communication Statistics'))
        ax4.set_xlabel(get_title('Episode', 'Episode'))
        ax4.set_ylabel(get_title('Message Count', 'Message Count'), color='purple')
        ax4_twin.set_ylabel(get_title('VOI Value', 'VOI Value'), color='orange')
        
        # Merge legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax4.text(0.5, 0.5, get_title('No Communication Data', 'No Communication Data'), ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title(get_title('Communication Statistics', 'Communication Statistics'))
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_decision_analysis(results):
    """Plot decision analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(get_title('RTD Decision Analysis', 'RTD Decision Analysis'), fontsize=16, fontweight='bold')
    
    episodes = len(results['episode_rewards'])
    episode_range = range(1, episodes + 1)
    
    # 1. Decision time series
    ax1 = axes[0, 0]
    decision_data = results['episode_decisions']
    
    for i, (decision, values) in enumerate(decision_data.items()):
        color = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db'][i % 4]
        ax1.plot(episode_range, values, color=color, label=decision, alpha=0.7, linewidth=1)
    
    ax1.set_title(get_title('Decision Time Series', 'Decision Time Series'))
    ax1.set_xlabel(get_title('Episode', 'Episode'))
    ax1.set_ylabel(get_title('Decision Count', 'Decision Count'))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Decision proportion pie chart
    ax2 = axes[0, 1]
    total_decisions = {k: np.sum(v) for k, v in decision_data.items()}
    labels = list(total_decisions.keys())
    sizes = list(total_decisions.values())
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db']
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=90)
    ax2.set_title(get_title('Decision Distribution Proportion', 'Decision Distribution Proportion'))
    
    # 3. Cooperation probability change
    ax3 = axes[1, 0]
    if 'cooperation_prob' in results['agent_info']:
        ax3.plot(episode_range, results['agent_info']['cooperation_prob'], 'b-', linewidth=2)
        ax3.set_title(get_title('Cooperation Probability Change', 'Cooperation Probability Change'))
        ax3.set_xlabel(get_title('Episode', 'Episode'))
        ax3.set_ylabel(get_title('Cooperation Probability', 'Cooperation Probability'))
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
    
    # 4. Lambda parameter change
    ax4 = axes[1, 1]
    if 'lambda_t' in results['agent_info']:
        ax4.plot(episode_range, results['agent_info']['lambda_t'], 'r-', linewidth=2)
        ax4.set_title(get_title('Lambda Parameter Change', 'Lambda Parameter Change'))
        ax4.set_xlabel(get_title('Episode', 'Episode'))
        ax4.set_ylabel(get_title('Lambda Value', 'Lambda Value'))
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('decision_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_metrics(results):
    """Plot performance metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(get_title('RTD Performance Metrics Analysis', 'RTD Performance Metrics Analysis'), fontsize=16, fontweight='bold')
    
    episodes = len(results['episode_rewards'])
    episode_range = range(1, episodes + 1)
    
    # 1. Reward distribution histogram
    ax1 = axes[0, 0]
    ax1.hist(results['episode_rewards'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title(get_title('Reward Distribution', 'Reward Distribution'))
    ax1.set_xlabel(get_title('Reward Value', 'Reward Value'))
    ax1.set_ylabel(get_title('Frequency', 'Frequency'))
    ax1.axvline(np.mean(results['episode_rewards']), color='red', linestyle='--', 
                label=f'{get_title("Average", "Average")}: {np.mean(results["episode_rewards"]):.2f}')
    ax1.legend()
    
    # 2. Regret distribution histogram
    ax2 = axes[0, 1]
    ax2.hist(results['episode_regrets'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_title(get_title('Regret Distribution', 'Regret Distribution'))
    ax2.set_xlabel(get_title('Regret', 'Regret'))
    ax2.set_ylabel(get_title('Frequency', 'Frequency'))
    ax2.axvline(np.mean(results['episode_regrets']), color='red', linestyle='--',
                label=f'{get_title("Average", "Average")}: {np.mean(results["episode_regrets"]):.4f}')
    ax2.legend()
    
    # 3. Reward vs Regret scatter plot
    ax3 = axes[1, 0]
    ax3.scatter(results['episode_rewards'], results['episode_regrets'], alpha=0.6, s=20)
    ax3.set_title(get_title('Reward vs Regret', 'Reward vs Regret'))
    ax3.set_xlabel(get_title('Reward Value', 'Reward Value'))
    ax3.set_ylabel(get_title('Regret', 'Regret'))
    
    # Add trend line
    z = np.polyfit(results['episode_rewards'], results['episode_regrets'], 1)
    p = np.poly1d(z)
    ax3.plot(results['episode_rewards'], p(results['episode_rewards']), "r--", alpha=0.8)
    
    # 4. Cumulative reward
    ax4 = axes[1, 1]
    cumulative_rewards = np.cumsum(results['episode_rewards'])
    ax4.plot(episode_range, cumulative_rewards, 'b-', linewidth=2)
    ax4.set_title(get_title('Cumulative Reward', 'Cumulative Reward'))
    ax4.set_xlabel(get_title('Episode', 'Episode'))
    ax4.set_ylabel(get_title('Cumulative Reward', 'Cumulative Reward'))
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_comparison_charts(results):
    """Plot comparison charts"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(get_title('RTD System Comparison Analysis', 'RTD System Comparison Analysis'), fontsize=16, fontweight='bold')
    
    episodes = len(results['episode_rewards'])
    episode_range = range(1, episodes + 1)
    
    # 1. Decision efficiency analysis
    ax1 = axes[0, 0]
    decision_data = results['episode_decisions']
    decision_efficiency = {}
    
    for decision, values in decision_data.items():
        if decision == 'accept':
            try:
                reject_values = decision_data.get('reject', [0]*episodes)
                if len(values) == len(reject_values):
                    efficiency = np.array(values) / (np.array(values) + np.array(reject_values) + 1e-6)
                    decision_efficiency[decision] = efficiency
            except Exception as e:
                print(f"Decision efficiency calculation failed: {e}")
    
    if 'accept' in decision_efficiency:
        ax1.plot(episode_range, decision_efficiency['accept'], 'g-', linewidth=2, label=get_title('Accept Efficiency', 'Accept Efficiency'))
        ax1.set_title(get_title('Decision Efficiency Analysis', 'Decision Efficiency Analysis'))
        ax1.set_xlabel(get_title('Episode', 'Episode'))
        ax1.set_ylabel(get_title('Accept Efficiency', 'Accept Efficiency'))
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, get_title('Decision Efficiency Calculation Failed', 'Decision Efficiency Calculation Failed'), ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(get_title('Decision Efficiency Analysis', 'Decision Efficiency Analysis'))
    
    # 2. Communication efficiency
    ax2 = axes[0, 1]
    comm_data = results['communication_stats']
    if 'messages_sent' in comm_data and 'total_voi' in comm_data and len(comm_data['messages_sent']) > 0:
        try:
            communication_efficiency = np.array(comm_data['total_voi']) / (np.array(comm_data['messages_sent']) + 1e-6)
            ax2.plot(episode_range, communication_efficiency, 'purple', linewidth=2)
            ax2.set_title(get_title('Communication Efficiency (VOI/Message Count)', 'Communication Efficiency (VOI/Message Count)'))
            ax2.set_xlabel(get_title('Episode', 'Episode'))
            ax2.set_ylabel(get_title('Communication Efficiency', 'Communication Efficiency'))
            ax2.grid(True, alpha=0.3)
        except Exception as e:
            ax2.text(0.5, 0.5, f'{get_title("Communication Efficiency Calculation Failed", "Communication Efficiency Calculation Failed")}\n{str(e)}', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(get_title('Communication Efficiency', 'Communication Efficiency'))
    else:
        ax2.text(0.5, 0.5, get_title('No Communication Data', 'No Communication Data'), ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(get_title('Communication Efficiency', 'Communication Efficiency'))
    
    # 3. Learning curve
    ax3 = axes[1, 0]
    window = min(50, episodes // 10)
    if window > 1:
        moving_avg_rewards = np.convolve(results['episode_rewards'], np.ones(window)/window, mode='valid')
        moving_avg_regrets = np.convolve(results['episode_regrets'], np.ones(window)/window, mode='valid')
        
        ax3_twin = ax3.twinx()
        line1 = ax3.plot(range(window, episodes + 1), moving_avg_rewards, 'b-', label=get_title('Reward', 'Reward'), linewidth=2)
        line2 = ax3_twin.plot(range(window, episodes + 1), moving_avg_regrets, 'r-', label=get_title('Regret', 'Regret'), linewidth=2)
        
        ax3.set_title(get_title('Learning Curve', 'Learning Curve'))
        ax3.set_xlabel(get_title('Episode', 'Episode'))
        ax3.set_ylabel(get_title('Reward', 'Reward'), color='b')
        ax3_twin.set_ylabel(get_title('Regret', 'Regret'), color='r')
        
        # Merge legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')
    
    # 4. Decision stability
    ax4 = axes[1, 1]
    decision_stability = {}
    window = 20
    
    for decision, values in decision_data.items():
        if len(values) >= window:
            stability = []
            for i in range(window, len(values)):
                window_std = np.std(values[i-window:i])
                stability.append(window_std)
            decision_stability[decision] = stability
    
    for decision, stability in decision_stability.items():
        color = {'accept': 'green', 'delay': 'orange', 'reject': 'red', 'explore': 'blue'}.get(decision, 'gray')
        ax4.plot(range(window, len(stability) + window), stability, color=color, label=decision, linewidth=2)
    
    ax4.set_title(get_title('Decision Stability (Rolling Standard Deviation)', 'Decision Stability (Rolling Standard Deviation)'))
    ax4.set_xlabel(get_title('Episode', 'Episode'))
    ax4.set_ylabel(get_title('Standard Deviation', 'Standard Deviation'))
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_report(results):
    """Create summary report"""
    print("=" * 60)
    print(get_title('RTD Enhanced Training Results Summary Report', 'RTD Enhanced Training Results Summary Report'))
    print("=" * 60)
    
    episodes = len(results['episode_rewards'])
    
    # Basic statistics
    print(f"\n📊 {get_title('Basic Statistics', 'Basic Statistics')}:")
    print(f"  {get_title('Total Training Episodes', 'Total Training Episodes')}: {episodes}")
    print(f"  {get_title('Average Reward', 'Average Reward')}: {np.mean(results['episode_rewards']):.2f} ± {np.std(results['episode_rewards']):.2f}")
    print(f"  {get_title('Average Regret', 'Average Regret')}: {np.mean(results['episode_regrets']):.4f} ± {np.std(results['episode_regrets']):.4f}")
    print(f"  {get_title('Max Reward', 'Max Reward')}: {np.max(results['episode_rewards']):.2f}")
    print(f"  {get_title('Min Reward', 'Min Reward')}: {np.min(results['episode_rewards']):.2f}")
    
    # Decision statistics
    print(f"\n🎯 {get_title('Decision Statistics', 'Decision Statistics')}:")
    decision_data = results['episode_decisions']
    total_decisions = sum(np.sum(values) for values in decision_data.values())
    
    for decision, values in decision_data.items():
        count = np.sum(values)
        percentage = (count / total_decisions) * 100 if total_decisions > 0 else 0
        print(f"  {decision}: {count} {get_title('times', 'times')} ({percentage:.1f}%)")
    
    # Communication statistics
    print(f"\n📡 {get_title('Communication Statistics', 'Communication Statistics')}:")
    comm_data = results['communication_stats']
    if 'messages_sent' in comm_data and 'total_voi' in comm_data:
        try:
            # Process messages_sent
            if isinstance(comm_data['messages_sent'], (list, np.ndarray)) and len(comm_data['messages_sent']) > 0:
                total_messages = comm_data['messages_sent'][-1]
            else:
                total_messages = comm_data['messages_sent'] if comm_data['messages_sent'] else 0
            
            # Process total_voi
            if isinstance(comm_data['total_voi'], (list, np.ndarray)) and len(comm_data['total_voi']) > 0:
                total_voi = comm_data['total_voi'][-1]
            else:
                total_voi = comm_data['total_voi'] if comm_data['total_voi'] else 0
            
            avg_voi = total_voi / max(total_messages, 1)
            print(f"  {get_title('Total Messages Sent', 'Total Messages Sent')}: {total_messages}")
            print(f"  {get_title('Total VOI Value', 'Total VOI Value')}: {total_voi:.4f}")
            print(f"  {get_title('Average VOI', 'Average VOI')}: {avg_voi:.4f}")
        except Exception as e:
            print(f"  {get_title('Communication Data Processing Failed', 'Communication Data Processing Failed')}: {e}")
    else:
        print(f"  {get_title('No Communication Data', 'No Communication Data')}")
    
    # Performance metrics
    print(f"\n📈 {get_title('Performance Metrics', 'Performance Metrics')}:")
    reward_trend = np.polyfit(range(episodes), results['episode_rewards'], 1)[0]
    regret_trend = np.polyfit(range(episodes), results['episode_regrets'], 1)[0]
    
    print(f"  {get_title('Reward Trend', 'Reward Trend')}: {'Increasing' if reward_trend > 0 else 'Decreasing'} ({reward_trend:.4f})")
    print(f"  {get_title('Regret Trend', 'Regret Trend')}: {'Increasing' if regret_trend > 0 else 'Decreasing'} ({regret_trend:.4f})")
    
    # Suggestions
    print(f"\n💡 {get_title('Suggestions', 'Suggestions')}:")
    if reward_trend > 0:
        print(f"  ✅ {get_title('Reward is increasing, training is effective', 'Reward is increasing, training is effective')}")
    else:
        print(f"  ⚠️  {get_title('Reward is decreasing, suggest adjusting hyperparameters', 'Reward is decreasing, suggest adjusting hyperparameters')}")
    
    if regret_trend < 0:
        print(f"  ✅ {get_title('Regret is decreasing, decision quality is improving', 'Regret is decreasing, decision quality is improving')}")
    else:
        print(f"  ⚠️  {get_title('Regret is increasing, suggest optimizing decision strategy', 'Regret is increasing, suggest optimizing decision strategy')}")
    
    print("=" * 60)

def main():
    """Main function"""
    print(get_title('🎨 RTD Enhanced Training Results Visualization', '🎨 RTD Enhanced Training Results Visualization'))
    print("=" * 40)
    
    # Load results
    results = load_training_results()
    
    # Create visualizations
    print(f"📊 {get_title('Generating Training Progress Chart...', 'Generating Training Progress Chart...')}")
    plot_training_progress(results)
    
    print(f"🎯 {get_title('Generating Decision Analysis Chart...', 'Generating Decision Analysis Chart...')}")
    plot_decision_analysis(results)
    
    print(f"📈 {get_title('Generating Performance Metrics Chart...', 'Generating Performance Metrics Chart...')}")
    plot_performance_metrics(results)
    
    print(f"🔍 {get_title('Generating Comparison Analysis Chart...', 'Generating Comparison Analysis Chart...')}")
    plot_comparison_charts(results)
    
    # Create summary report
    print(f"📝 {get_title('Generating Summary Report...', 'Generating Summary Report...')}")
    create_summary_report(results)
    
    print(f"\n✅ {get_title('Visualization Complete!', 'Visualization Complete!')}")
    print(f"📁 {get_title('Generated Image Files', 'Generated Image Files')}:")
    print(f"  - {get_title('training_progress.png', 'training_progress.png')}")
    print(f"  - {get_title('decision_analysis.png', 'decision_analysis.png')}")
    print(f"  - {get_title('performance_metrics.png', 'performance_metrics.png')}")
    print(f"  - {get_title('comparison_analysis.png', 'comparison_analysis.png')}")

if __name__ == "__main__":
    main()
