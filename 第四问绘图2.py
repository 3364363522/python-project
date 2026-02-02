import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.transforms as transforms

# ==========================================
# 0. Global Style & Configuration
# ==========================================

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-white') # Base style (clean white background)

# Okabe-Ito Color Palette (Colorblind friendly)
# Mapping: rank_sum=Gray, percent_sum=Blue, judge_save=Green, proposed=Vermilion
COLORS = {
    'rank_sum': '#7F7F7F',      
    'percent_sum': '#0072B2',   
    'judge_save': '#009E73',    
    'proposed': '#D55E00'       
}

# Configuration params
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'axes.linewidth': 0.8,         # Spines thickness
    'grid.linewidth': 0.5,
    'grid.alpha': 0.2,             # Subtle grid
    'legend.frameon': False,       # Clean legend
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'figure.dpi': 300
})

# Load data
df_summary = pd.read_csv('mechanism_eval_summary.csv')
df_sims = pd.read_csv('mechanism_eval_sims.csv')

mechanism_order = ['rank_sum', 'percent_sum', 'judge_save', 'proposed']

# Labels map
metric_labels = {
    'gap': 'Vote Gap',
    'upset_rate': 'Upset Rate',
    'merit_loss_rate': 'Merit Loss Rate',
    'controversy_rate': 'Controversy Rate',
    'precision': 'Precision (vs History)'
}

# ==========================================
# Figure 1: Indicator Forest Plot (Mean + 95% CI)
# ==========================================
def plot_forest():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metrics = ['gap', 'upset_rate', 'merit_loss_rate', 'controversy_rate']
    
    # Calculate effect sizes for text annotation
    means = df_summary.pivot(index='mechanism', columns='metric', values='mean')
    gap_improvement = (means.loc['proposed', 'gap'] - means.loc['percent_sum', 'gap']) / means.loc['percent_sum', 'gap']
    upset_improvement_abs = means.loc['proposed', 'upset_rate'] - means.loc['percent_sum', 'upset_rate']
    upset_improvement_rel = upset_improvement_abs / means.loc['percent_sum', 'upset_rate']

    for ax, metric in zip(axes.flatten(), metrics):
        # Prepare data subset
        subset = df_summary[df_summary['metric'] == metric].copy()
        
        # Specific handling for GAP
        if metric == 'gap':
            subset = subset[subset['mechanism'] != 'rank_sum']
            current_order = ['percent_sum', 'judge_save', 'proposed']
            # Add note about rank_sum exclusion
            ax.text(0.95, 0.05, "Note: rank_sum excluded\n(ordinal scale not comparable)", 
                    transform=ax.transAxes, fontsize=9, ha='right', va='bottom', 
                    color='#555555', style='italic', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
            # Add Effect Size annotation for Gap
            ax.set_title(f"{metric_labels[metric]}\n(Proposed vs Percent: {gap_improvement:.1%})", fontweight='bold')
        else:
            current_order = mechanism_order
            if metric == 'upset_rate':
                ax.set_title(f"{metric_labels[metric]}\n(Increase: +{upset_improvement_abs:.3f} / +{upset_improvement_rel:.0%})", fontweight='bold')
            else:
                ax.set_title(metric_labels[metric], fontweight='bold')

        # Reorder
        subset['mechanism'] = pd.Categorical(subset['mechanism'], categories=current_order, ordered=True)
        subset = subset.sort_values('mechanism', ascending=False) # Plot bottom-up for logical y-axis reading
        
        y_pos = range(len(subset))
        
        # 1. Error bars (Neutral Gray)
        ax.errorbar(subset['mean'], y_pos, 
                    xerr=[subset['mean'] - subset['q025'], subset['q975'] - subset['mean']], 
                    fmt='none', ecolor='#555555', capsize=4, elinewidth=1.2, capthick=1.2, zorder=1)
        
        # 2. Points (Colored with edge)
        for i, (idx, row) in enumerate(subset.iterrows()):
            ax.plot(row['mean'], i, 'o', markersize=9, 
                    markerfacecolor=COLORS[row['mechanism']], markeredgecolor='black', markeredgewidth=0.8, zorder=2)

        # Styling
        ax.set_yticks(y_pos)
        ax.set_yticklabels(subset['mechanism'])
        ax.set_xlabel('Mean (95% CI)')
        
        # Remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Grid
        ax.grid(True, axis='x', linestyle='-', color='#cccccc', alpha=0.3)

    plt.tight_layout()
    plt.savefig('figure1_forest_plot.png', dpi=300, bbox_inches='tight')
    print("Figure 1 saved.")

# ==========================================
# Figure 2: Delta Difference Distribution
# ==========================================
def plot_deltas():
    # Pivot for delta calculation
    df_pivot = df_sims.pivot_table(index='sim', columns='mechanism', values=['gap', 'upset_rate', 'merit_loss_rate', 'controversy_rate'])
    
    metrics = ['gap', 'upset_rate', 'merit_loss_rate', 'controversy_rate']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for ax, metric in zip(axes.flatten(), metrics):
        # Delta = Proposed - Baseline (Percent_Sum)
        delta = df_pivot[(metric, 'proposed')] - df_pivot[(metric, 'percent_sum')]
        
        # Direction Logic
        if metric == 'upset_rate':
            is_good = delta > 0
            direction_text = "Positive is better (>0)"
            color_better = 'green' # conceptual logic, but we use grayscale for histogram
        else:
            is_good = delta < 0
            direction_text = "Negative is better (<0)"
            color_better = 'green'
            
        prob_better = is_good.mean()
        mean_delta = delta.mean()
        ci_lower, ci_upper = delta.quantile(0.025), delta.quantile(0.975)
        
        # Plot Histogram + KDE
        sns.histplot(delta, ax=ax, kde=True, color='#cccccc', edgecolor='#555555', 
                     linewidth=0.5, stat='density', alpha=0.5, bins=25)
        
        # Reference Lines
        ax.axvline(0, color='#999999', linestyle='--', linewidth=1.2, label='Zero (No Diff)')
        ax.axvline(mean_delta, color='#0072B2', linestyle='-', linewidth=1.5, label='Mean Δ') # Using Blue for Mean
        
        # Annotation Box
        stats_text = (f"P(Proposed Better): {prob_better:.1%}\n"
                      f"Mean Δ: {mean_delta:.4f}\n"
                      f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        ax.text(0.04, 0.96, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='square,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#e0e0e0'))
        
        # Titles & Labels
        ax.set_title(metric_labels[metric], fontweight='bold')
        ax.set_xlabel(f"Δ (Proposed − Percent Sum)\n{direction_text}", fontsize=10, style='italic', color='#444444')
        ax.set_ylabel("Density")
        
        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('figure2_delta_distribution.png', dpi=300, bbox_inches='tight')
    print("Figure 2 saved.")

# ==========================================
# Figure 3: Pareto Frontier (Fairness vs Excitement)
# ==========================================
def plot_pareto():
    pareto_df = df_sims.copy()
    # Definitions
    pareto_df['fairness'] = 1 - pareto_df['merit_loss_rate'] # X axis
    pareto_df['excitement'] = pareto_df['upset_rate']        # Y axis
    
    plt.figure(figsize=(9, 7))
    ax = plt.gca()
    
    # 1. Cloud points (Small, low alpha)
    for mech in mechanism_order:
        subset = pareto_df[pareto_df['mechanism'] == mech]
        plt.scatter(subset['fairness'], subset['excitement'], 
                    alpha=0.12, s=12, color=COLORS[mech], label=None, rasterized=True) # Rasterized for lighter PDF
        
    # 2. Centroids (Large, bordered)
    means = pareto_df.groupby('mechanism')[['fairness', 'excitement']].mean()
    
    # Manual offsets to prevent overlap (dx, dy)
    offsets = {
        'rank_sum':    (-0.005, -0.015), # Move down-left
        'percent_sum': (0.002, -0.015),  # Move down
        'judge_save':  (0.005, 0.005),   # Move up-right
        'proposed':    (0.000, 0.015)    # Move up
    }
    
    for mech in mechanism_order:
        mx, my = means.loc[mech, 'fairness'], means.loc[mech, 'excitement']
        
        # Plot Mean Point
        plt.scatter(mx, my, s=180, facecolor=COLORS[mech], edgecolor='black', linewidth=1.5, zorder=10)
        
        # Add Label with Offset & BBox
        dx, dy = offsets.get(mech, (0.005, 0.005))
        plt.text(mx + dx, my + dy, mech, fontsize=11, fontweight='bold', color=COLORS[mech],
                 ha='center' if abs(dx)<0.003 else ('left' if dx>0 else 'right'),
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    # 3. Guidance text (Top Left)
    plt.text(0.02, 0.98, "Better = Up & Right\n(More Fair + More Exciting)", 
             transform=ax.transAxes, va='top', ha='left', fontsize=10, 
             style='italic', color='#444444', bbox=dict(facecolor='white', alpha=0.8))

    # Axes & Style
    plt.xlabel('Fairness (1 − Merit Loss Rate)', fontsize=12)
    plt.ylabel('Excitement (Upset Rate)', fontsize=12)
    plt.title('Fairness vs. Excitement Trade-off', fontweight='bold', fontsize=13)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(True, linestyle='-', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('figure3_pareto_front.png', dpi=300, bbox_inches='tight')
    print("Figure 3 saved.")

# ==========================================
# Figure 4: Distribution Boxplots
# ==========================================
def plot_boxplots():
    metrics = ['merit_loss_rate', 'controversy_rate', 'upset_rate', 'precision']
    direction_labels = {
        'merit_loss_rate': 'Lower is Better',
        'controversy_rate': 'Lower is Better',
        'upset_rate': 'Higher is Better',
        'precision': 'Higher is Better'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for ax, metric in zip(axes.flatten(), metrics):
        # Create boxplot
        sns.boxplot(data=df_sims, x='mechanism', y=metric, order=mechanism_order, 
                    ax=ax, width=0.5, linewidth=1.2,
                    showfliers=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.3, markerfacecolor='gray', markeredgecolor='none'),
                    medianprops=dict(color='white', linewidth=1.5),
                    whiskerprops=dict(color='#555555'),
                    capprops=dict(color='#555555'),
                    boxprops=dict(edgecolor='none')) # We will color boxes manually below for transparency
        
        # Custom coloring for transparency + dark borders
        # sns.boxplot doesn't support alpha well directly on the artist, so we iterate
        for i, artist in enumerate(ax.artists):
            col = COLORS[mechanism_order[i]]
            # Fill with alpha
            artist.set_facecolor(col)
            artist.set_alpha(0.45)
            # Edge color (solid)
            artist.set_edgecolor(col)
            artist.set_linewidth(1.5)

        # Titles
        ax.set_title(f"{metric_labels[metric]}\n({direction_labels[metric]})", fontweight='bold')
        ax.set_xlabel('')
        ax.set_xticklabels(mechanism_order) # Keep labels
        
        # Spines & Grid
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='-', alpha=0.2, color='#cccccc')
        ax.xaxis.grid(False)

    plt.tight_layout()
    plt.savefig('figure4_boxplots.png', dpi=300, bbox_inches='tight')
    print("Figure 4 saved.")

if __name__ == "__main__":
    print("Generating publication-quality figures...")
    plot_forest()
    plot_deltas()
    plot_pareto()
    plot_boxplots()
    print("All tasks completed.")