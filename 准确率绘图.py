import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置绘图风格，使用seaborn的高级样式
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # 确保兼容性，如果系统有中文字体可替换
plt.rcParams['axes.unicode_minus'] = False

def create_beautiful_accuracy_chart(file_path):
    # 1. 读取数据
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return

    # 确保 season 是数值型并排序
    df['season'] = pd.to_numeric(df['season'])
    df = df.sort_values('season')
    
    seasons = df['season']
    accuracy = df['survival_accuracy']

    # 2. 创建画布
    fig, ax = plt.subplots(figsize=(14, 8))

    # 3. 颜色映射逻辑
    # 如果准确率为1.0，使用金色，否则使用深青色
    colors = ['#FFD700' if x == 1.0 else '#4A90E2' for x in accuracy]
    sizes = [150 if x == 1.0 else 80 for x in accuracy] # 满分点稍微大一点

    # 4. 绘制棒棒糖图 (Lollipop Chart)
    # 绘制垂直线 (Stems)
    ax.vlines(x=seasons, ymin=0, ymax=accuracy, color=colors, alpha=0.6, linewidth=2)
    
    # 绘制圆点 (Heads)
    scatter = ax.scatter(seasons, accuracy, s=sizes, c=colors, alpha=1, zorder=3, edgecolors='white', linewidth=1.5)

    # 5. 添加趋势线 (使用滚动平均，窗口为5)
    df['ma'] = df['survival_accuracy'].rolling(window=5, center=True, min_periods=1).mean()
    ax.plot(seasons, df['ma'], color='#FF6B6B', linewidth=2.5, linestyle='-', alpha=0.8, label='Trend (5-Season Moving Avg)', zorder=2)

    # 6. 添加平均线
    mean_acc = accuracy.mean()
    ax.axhline(y=mean_acc, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label=f'Average Accuracy: {mean_acc:.2%}')

    # 7. 图表装饰
    # 标题和标签
    ax.set_title('Survival Prediction Accuracy by Season', fontsize=20, fontweight='bold', pad=20, color='#333333')
    ax.set_xlabel('Season', fontsize=14, labelpad=10)
    ax.set_ylabel('Accuracy (0.0 - 1.0)', fontsize=14, labelpad=10)

    # 坐标轴设置
    ax.set_xticks(seasons)
    ax.set_xticklabels(seasons, rotation=45, fontsize=9)
    ax.set_ylim(0, 1.1) # 留出一点顶部空间
    
    # 移除顶部和右侧的边框
    sns.despine(left=True, bottom=True)

    # 添加图例
    # 创建自定义图例句柄
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Perfect Accuracy (100%)', markerfacecolor='#FFD700', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Standard Accuracy', markerfacecolor='#4A90E2', markersize=8),
        Line2D([0], [0], color='#FF6B6B', lw=2.5, label='Trend Line'),
        Line2D([0], [0], color='gray', lw=1.5, linestyle='--', label=f'Overall Avg: {mean_acc:.2f}')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, framealpha=0.9, shadow=True)

    # 8. 为100%准确率的点添加数值标签
    for x, y in zip(seasons, accuracy):
        if y == 1.0:
            ax.text(x, y + 0.02, '100%', ha='center', va='bottom', fontsize=9, color='#D4AF37', fontweight='bold')
        elif y < 0.5: # 标记异常低分
            ax.text(x, y + 0.02, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='#333333')

    # 添加网格线 (仅Y轴)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.grid(False)

    plt.tight_layout()
    
    # 保存图片
    output_filename = 'survival_accuracy_chart.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"图表已生成并保存为: {output_filename}")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    file_path = 'final_season_survival_accuracy.csv'
    create_beautiful_accuracy_chart(file_path)