import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

# Load data
df_summary = pd.read_csv('mechanism_eval_summary.csv')
df_sims = pd.read_csv('mechanism_eval_sims.csv')

# Define color palette (Proposed stands out)
colors = {
    'rank_sum': '#95a5a6',      # Gray
    'percent_sum': '#3498db',   # Blue (Baseline)
    'judge_save': '#2ecc71',    # Green
    'proposed': '#e74c3c'       # Red (Highlight)
}

mechanism_order = ['rank_sum', 'percent_sum', 'judge_save', 'proposed']
metric_labels = {
    'gap': 'Vote Gap (Lower is Better)',
    'upset_rate': 'Upset Rate (Higher is Better)',
    'merit_loss_rate': 'Merit Loss Rate (Lower is Better)',
    'controversy_rate': 'Controversy Rate (Lower is Better)',
    'precision': 'Precision (vs History)'
}

# ==========================================
# Figure 1: Indicator Forest Plot (Mean + 95% CI)
# ==========================================
def plot_forest():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['gap', 'upset_rate', 'merit_loss_rate', 'controversy_rate']
    
    for ax, metric in zip(axes.flatten(), metrics):
        # Filter data for this metric
        subset = df_summary[df_summary['metric'] == metric].copy()
        
        # Special handling for GAP: remove rank_sum as scale is different
        if metric == 'gap':
            subset = subset[subset['mechanism'] != 'rank_sum']
            order = ['percent_sum', 'judge_save', 'proposed']
        else:
            order = mechanism_order
            
        # Reorder subset
        subset['mechanism'] = pd.Categorical(subset['mechanism'], categories=order, ordered=True)
        subset = subset.sort_values('mechanism')
        
        # Plotting
        y_pos = range(len(subset))
        ax.errorbar(subset['mean'], y_pos, 
                    xerr=[subset['mean'] - subset['q025'], subset['q975'] - subset['mean']], 
                    fmt='o', capsize=5, color='black', alpha=0.3)
        
        for i, (idx, row) in enumerate(subset.iterrows()):
            ax.plot(row['mean'], i, 'o', markersize=10, 
                    color=colors[row['mechanism']], label=row['mechanism'])
            
        ax.set_yticks(y_pos)
        ax.set_yticklabels(subset['mechanism'])
        ax.set_xlabel('Mean Value (with 95% CI)')
        ax.set_title(metric_labels.get(metric, metric), fontweight='bold')
        
        # Grid layout adjustment
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    plt.suptitle('Figure 1: Comparison of Mean Indicators with 95% Confidence Intervals', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('figure1_forest_plot.png', bbox_inches='tight', dpi=300)
    print("Figure 1 saved.")

# ==========================================
# Figure 2: Delta Difference Distribution
# ==========================================
def plot_deltas():
    # Pivot data to get side-by-side comparison per sim
    df_pivot = df_sims.pivot_table(index='sim', columns='mechanism', values=['gap', 'upset_rate', 'merit_loss_rate', 'controversy_rate'])
    
    metrics_to_plot = ['gap', 'merit_loss_rate', 'controversy_rate', 'upset_rate']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for ax, metric in zip(axes.flatten(), metrics_to_plot):
        # Calculate Delta: Proposed - Percent_Sum (Baseline)
        delta = df_pivot[(metric, 'proposed')] - df_pivot[(metric, 'percent_sum')]
        
        # Determine "Better" direction
        if metric in ['upset_rate']:
            better_mask = delta > 0
            better_label = "Proposed > Baseline"
            color_better = 'green'
            color_worse = 'red'
            xlabel = f"Δ {metric}\n(>0 means Proposed is better)"
        else:
            better_mask = delta < 0
            better_label = "Proposed < Baseline"
            color_better = 'green'
            color_worse = 'red'
            xlabel = f"Δ {metric}\n(<0 means Proposed is better)"
            
        prob_better = better_mask.mean()
        
        # Plot Histogram / KDE
        sns.histplot(delta, ax=ax, kde=True, color='gray', alpha=0.4, bins=30, stat='density')
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax.axvline(delta.mean(), color='blue', linestyle='-', linewidth=1.5, label=f'Mean Δ: {delta.mean():.4f}')
        
        # Annotation
        stats_text = (f"P(Better) = {prob_better:.1%}\n"
                      f"Mean Δ = {delta.mean():.4f}\n"
                      f"95% CI: [{delta.quantile(0.025):.4f}, {delta.quantile(0.975):.4f}]")
        
        # Place text box
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f"Improvement Distribution: {metric}", fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.legend()

    plt.suptitle('Figure 2: Distribution of Differences (Proposed vs. Baseline) across 300 Simulations', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('figure2_delta_distribution.png', bbox_inches='tight', dpi=300)
    print("Figure 2 saved.")

# ==========================================
# Figure 3: Fairness - Excitement Pareto Front
# ==========================================
def plot_pareto():
    # Prepare data
    # Fairness = 1 - merit_loss_rate
    # Excitement = upset_rate
    
    pareto_df = df_sims.copy()
    pareto_df['fairness'] = 1 - pareto_df['merit_loss_rate']
    pareto_df['excitement'] = pareto_df['upset_rate']
    
    plt.figure(figsize=(10, 8))
    
    # 1. Plot individual simulation points (Cloud)
    for mech in mechanism_order:
        subset = pareto_df[pareto_df['mechanism'] == mech]
        plt.scatter(subset['fairness'], subset['excitement'], 
                    alpha=0.15, s=30, color=colors[mech], label=None) # No label for cloud
        
    # 2. Plot Mean points (Centroids)
    means = pareto_df.groupby('mechanism')[['fairness', 'excitement']].mean()
    
    for mech in mechanism_order:
        plt.scatter(means.loc[mech, 'fairness'], means.loc[mech, 'excitement'], 
                    s=200, color=colors[mech], edgecolors='black', linewidth=1.5, label=mech, zorder=10)
        
        # Add text label near centroid
        plt.text(means.loc[mech, 'fairness'] + 0.002, means.loc[mech, 'excitement'] + 0.002, 
                 mech, fontsize=12, fontweight='bold', color=colors[mech])

    # Aesthetics
    plt.xlabel('Fairness (1 - Merit Loss Rate)', fontsize=14)
    plt.ylabel('Excitement (Upset Rate)', fontsize=14)
    plt.title('Figure 3: Fairness vs. Excitement Trade-off (Pareto Frontier)', fontsize=16)
    
    # Add arrow pointing to ideal corner
    plt.arrow(0.92, 0.1, 0.05, 0.1, head_width=0.01, head_length=0.01, fc='black', ec='black')
    plt.text(0.93, 0.21, "Better (More Fair & More Exciting)", fontsize=10, ha='center')
    
    plt.legend(title='Mechanism', loc='lower left')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('figure3_pareto_front.png', bbox_inches='tight', dpi=300)
    print("Figure 3 saved.")

# ==========================================
# Figure 4: Boxplots for Robustness
# ==========================================
def plot_boxplots():
    metrics = ['merit_loss_rate', 'controversy_rate', 'upset_rate', 'precision']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for ax, metric in zip(axes.flatten(), metrics):
        sns.boxplot(data=df_sims, x='mechanism', y=metric, order=mechanism_order, 
                    palette=colors, ax=ax, width=0.6, linewidth=1.2, hue='mechanism', legend=False)
        
        ax.set_title(metric_labels.get(metric, metric), fontweight='bold')
        ax.set_xlabel('')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.suptitle('Figure 4: Distribution Robustness across Metrics', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('figure4_boxplots.png', bbox_inches='tight', dpi=300)
    print("Figure 4 saved.")

if __name__ == "__main__":
    plot_forest()
    plot_deltas()
    plot_pareto()
    plot_boxplots()
    print("All figures generated successfully.")