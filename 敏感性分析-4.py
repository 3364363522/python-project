import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格，符合学术论文要求
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
# 尝试设置字体，如果系统中没有Arial可能会回退到默认
try:
    plt.rcParams['font.sans-serif'] = ['Arial']
except:
    pass

def calculate_gini(shares):
    """计算基尼系数，用于衡量不平等程度"""
    # 确保输入是numpy数组
    x = np.array(shares, dtype=np.float64)
    if np.any(x < 0): x = np.abs(x) # 防御性编程
    
    # 平均绝对误差
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    
    return diffsum / (len(x)**2 * np.mean(x))

def run_sensitivity_analysis_real():
    # --- 1. 读取数据与筛选案例 (Data Loading & Case Selection) ---
    try:
        df = pd.read_csv('/Users/garytchois/Desktop/w_0p5_修改版1/posterior_vote_share_summary.csv')
    except FileNotFoundError:
        print("Error: 文件未找到，请确保已上传 posterior_vote_share_summary.csv")
        return

    # 找出每一周的数据
    weeks_data = []
    
    # 按赛季和周分组
    grouped = df.groupby(['season', 'week'])
    
    max_gini = -1
    target_data = None
    target_label = ""
    
    for (season, week), group in grouped:
        # 获取该周所有选手的粉丝得票比例 (p_mean)
        shares = group['p_mean'].values
        names = group['celebrity_name'].values
        
        # 简单归一化 (以防原始数据和不为1)
        shares = shares / np.sum(shares)
        
        # 计算基尼系数
        gini = calculate_gini(shares)
        
        if gini > max_gini:
            max_gini = gini
            # 按得票比例从高到低排序，方便画图
            sorted_indices = np.argsort(shares)[::-1]
            target_data = {
                'shares': shares[sorted_indices],
                'names': names[sorted_indices],
                'season': season,
                'week': week
            }
            target_label = f"Season {season} Week {week}"

    print(f"Selected Case: {target_label} (Max Gini: {max_gini:.4f})")
    print(f"Top 1 Share: {target_data['shares'][0]:.2%}")
    print(f"Bottom 1 Share: {target_data['shares'][-1]:.2%}")

    # --- 2. 敏感性计算 (Sensitivity Calculation) ---
    raw_shares = target_data['shares']
    contestants = target_data['names']
    
    epsilon = 1e-5
    gamma_range = np.linspace(0.1, 1.2, 50)
    
    top1_shares = []
    bottom1_shares = []
    gini_scores = []
    
    for gamma in gamma_range:
        # 核心变换公式: g(f) = (f)^gamma  (因为输入已经是比例，epsilon可以很小或省略)
        # 注意：这里直接对比例做幂变换，然后重新归一化
        transformed = np.power(raw_shares, gamma)
        normalized = transformed / np.sum(transformed)
        
        top1_shares.append(normalized[0])
        bottom1_shares.append(normalized[-1])
        gini_scores.append(calculate_gini(normalized))

    # --- 3. 绘图 (Visualization) ---
    fig = plt.figure(figsize=(14, 6))
    
    # 图 1: Gamma 对不平等程度的影响 (Line Chart)
    ax1 = fig.add_subplot(1, 2, 1)
    
    # 绘制 Top 1 和 Bottom 1 的份额变化
    ax1.plot(gamma_range, top1_shares, label=f'Top 1 ({contestants[0]})', color='#d62728', linewidth=2.5)
    ax1.plot(gamma_range, bottom1_shares, label=f'Bottom 1 ({contestants[-1]})', color='#2ca02c', linewidth=2.5)
    
    # 绘制 Gini 系数变化 (用虚线，次坐标轴)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(gamma_range, gini_scores, label='Gini Coefficient (Inequality)', color='gray', linestyle='--', alpha=0.6)
    ax1_twin.set_ylabel('Gini Coefficient', color='gray')
    ax1_twin.grid(False) # 避免网格混乱

    # 标注 gamma = 0.5 (Proposed)
    idx_05 = np.abs(gamma_range - 0.5).argmin()
    ax1.scatter(gamma_range[idx_05], top1_shares[idx_05], color='black', zorder=5)
    ax1.annotate('Proposed $\gamma=0.5$\n(More Balanced)', 
                 xy=(gamma_range[idx_05], top1_shares[idx_05]), 
                 xytext=(0.3, 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    # 标注 gamma = 1.0 (Original)
    idx_10 = np.abs(gamma_range - 1.0).argmin()
    ax1.scatter(gamma_range[idx_10], top1_shares[idx_10], color='black', zorder=5)
    ax1.annotate('Original $\gamma=1.0$\n(High Inequality)', 
                 xy=(gamma_range[idx_10], top1_shares[idx_10]), 
                 xytext=(0.8, 0.7),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    ax1.set_xlabel(r'Gamma Parameter ($\gamma$)', fontsize=12)
    ax1.set_ylabel('Normalized Fan Score Share', fontsize=12)
    ax1.set_title(f'Effect of $\gamma$ on Vote Inequality\n(Data: {target_label})', fontsize=14, fontweight='bold')
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

    # 图 2: 不同 Gamma 下的分布对比 (Bar Chart)
    ax2 = fig.add_subplot(1, 2, 2)
    
    selected_gammas = [1.0, 0.5, 0.2] 
    bar_width = 0.25
    # 只展示前5名和最后3名，避免X轴太挤，或者展示全部但旋转标签
    # 这里为了清晰，展示所有选手（如果少于10人）或 Top3 + Bottom3
    
    display_indices = np.arange(len(raw_shares))
    if len(raw_shares) > 10:
        # 如果人数太多，只取 Top 3 和 Bottom 3
        keep_mask = np.concatenate([np.arange(3), np.arange(len(raw_shares)-3, len(raw_shares))])
        display_raw = raw_shares[keep_mask]
        display_names = [contestants[i] for i in keep_mask]
        display_names.insert(3, "...")
        # 这种截断绘图比较复杂，为了稳健，我们只画 Top 5 和 Bottom 2
        # 简单起见，画全部，但旋转标签
        display_names = contestants
    else:
        display_names = contestants
        
    index = np.arange(len(display_names))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels_g = ['Original ($\gamma=1.0$)', 'Proposed ($\gamma=0.5$)', 'Extreme ($\gamma=0.2$)']
    
    for i, g_val in enumerate(selected_gammas):
        t_votes = np.power(raw_shares, g_val)
        n_scores = t_votes / np.sum(t_votes)
        ax2.bar(index + i*bar_width, n_scores, bar_width, label=labels_g[i], color=colors[i], alpha=0.8)

    ax2.set_xlabel('Contestants (Ranked by MCMC Vote Share)', fontsize=12)
    ax2.set_ylabel('Final Fan Score Proportion', fontsize=12)
    ax2.set_title(f'Score Distribution Adjustment\n(Data: {target_label})', fontsize=14, fontweight='bold')
    ax2.set_xticks(index + bar_width)
    ax2.set_xticklabels(display_names, rotation=45, ha='right', fontsize=9)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('gamma_sensitivity_real_data.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_sensitivity_analysis_real()