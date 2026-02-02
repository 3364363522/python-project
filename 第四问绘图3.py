import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors  # Added for precise color manipulation

# ==========================================
# 0. Global Style & Configuration
# ==========================================

# Set style for publication-quality plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

# Okabe-Ito Color Palette (Colorblind friendly)
COLORS = {
    'rank_sum': '#7F7F7F',      # Gray
    'percent_sum': '#56B4E9',   # Sky Blue
    'judge_save': '#009E73',    # Bluish Green
    'proposed': '#E69F00'       # Orange
}

# Display Names Mapping
NAME_MAPPING = {
    'rank_sum': 'Rank Sum',
    'percent_sum': 'Percent Sum',
    'judge_save': 'Judge Save',
    'proposed': 'Proposed'
}

# Configuration params
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.15,          # Subtle grid
    'legend.frameon': False,
    'figure.dpi': 300,
    'savefig.bbox': 'tight'
})

# Load data
df_summary = pd.read_csv('mechanism_eval_summary.csv')
df_sims = pd.read_csv('mechanism_eval_sims.csv')

# Fixed order
mechanism_order = ['rank_sum', 'percent_sum', 'judge_save', 'proposed']
display_order = [NAME_MAPPING[m] for m in mechanism_order]

# Metrics config
METRICS_CONFIG = {
    'gap': {'label': 'Vote Gap', 'better': 'lower'},
    'upset_rate': {'label': 'Upset Rate', 'better': 'higher'},
    'merit_loss_rate': {'label': 'Merit Loss Rate', 'better': 'lower'},
    'controversy_rate': {'label': 'Controversy Rate', 'better': 'lower'},
    'precision': {'label': 'Precision', 'better': 'higher'}
}

# Helper to apply name mapping
def apply_naming(df, col='mechanism'):
    df_out = df.copy()
    df_out[col] = df_out[col].map(NAME_MAPPING)
    return df_out

df_summary = apply_naming(df_summary)
df_sims = apply_naming(df_sims)

# ==========================================
# Figure 1: Forest Plot
# ==========================================
def plot_forest_panel(save_path='figure1_forest_plot.png'):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics = ['gap', 'upset_rate', 'merit_loss_rate', 'controversy_rate']
    
    means = df_summary.set_index(['mechanism', 'metric'])['mean']
    def get_mean(mech, met):
        return means.get((NAME_MAPPING[mech], met), np.nan)

    for ax, metric in zip(axes.flatten(), metrics):
        config = METRICS_CONFIG[metric]
        subset = df_summary[df_summary['metric'] == metric].copy()
        
        if metric == 'gap':
            subset = subset[subset['mechanism'] != NAME_MAPPING['rank_sum']]
            current_order = [NAME_MAPPING['percent_sum'], NAME_MAPPING['judge_save'], NAME_MAPPING['proposed']]
            
            p_mean = get_mean('proposed', metric)
            b_mean = get_mean('percent_sum', metric)
            if not np.isnan(p_mean) and not np.isnan(b_mean):
                pct_change = (p_mean - b_mean) / b_mean
                ax.set_title(f"{config['label']}\n(Proposed vs Percent: {pct_change:.1%})", fontweight='bold')
            else:
                ax.set_title(config['label'], fontweight='bold')
            
            ax.text(0.98, 0.02, "Note: Rank Sum excluded\n(ordinal scale not comparable)", 
                    transform=ax.transAxes, fontsize=9, ha='right', va='bottom', 
                    color='#666666', style='italic')
        else:
            current_order = display_order
            p_mean = get_mean('proposed', metric)
            b_mean = get_mean('percent_sum', metric)
            if metric == 'upset_rate' and not np.isnan(p_mean) and not np.isnan(b_mean):
                diff = p_mean - b_mean
                ax.set_title(f"{config['label']}\n(Increase: +{diff:.3f})", fontweight='bold')
            else:
                ax.set_title(config['label'], fontweight='bold')

        subset['mechanism'] = pd.Categorical(subset['mechanism'], categories=current_order, ordered=True)
        subset = subset.sort_values('mechanism', ascending=False) 
        subset = subset.set_index('mechanism').reindex(current_order)
        subset = subset.iloc[::-1] 
        
        y_pos = range(len(subset))
        mechanisms = subset.index.tolist()
        
        xerr_low = subset['mean'] - subset['q025']
        xerr_high = subset['q975'] - subset['mean']
        
        ax.errorbar(subset['mean'], y_pos, xerr=[xerr_low, xerr_high], 
                    fmt='none', ecolor='#444444', capsize=4, elinewidth=1.5, zorder=1)
        
        for i, mech in enumerate(mechanisms):
            orig_key = [k for k, v in NAME_MAPPING.items() if v == mech][0]
            col = COLORS[orig_key]
            
            ax.plot(subset.loc[mech, 'mean'], i, 'o', markersize=10, 
                    markerfacecolor=col, markeredgecolor='black', markeredgewidth=1.0, zorder=2)
            
            mean_val = subset.loc[mech, 'mean']
            low_val = subset.loc[mech, 'q025']
            high_val = subset.loc[mech, 'q975']
            label_text = f"{mean_val:.3f} [{low_val:.3f}, {high_val:.3f}]"
            x_range = subset['q975'].max() - subset['q025'].min()
            text_offset = x_range * 0.05
            ax.text(high_val + text_offset, i, label_text, va='center', fontsize=9, color='#333333')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(mechanisms)
        ax.set_xlabel('Mean (95% CI)')
        sns.despine(ax=ax, left=False, bottom=False, top=True, right=True)
        ax.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {save_path}")

# ==========================================
# Figure 2: Delta Distribution
# ==========================================
def plot_delta_panel(save_path='figure2_delta_distribution.png'):
    df_pivot = df_sims.pivot_table(index='sim', columns='mechanism', values=['gap', 'upset_rate', 'merit_loss_rate', 'controversy_rate'])
    metrics = ['gap', 'upset_rate', 'merit_loss_rate', 'controversy_rate']
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    
    baseline_name = NAME_MAPPING['percent_sum']
    proposed_name = NAME_MAPPING['proposed']
    
    for ax, metric in zip(axes.flatten(), metrics):
        config = METRICS_CONFIG[metric]
        delta = df_pivot[(metric, proposed_name)] - df_pivot[(metric, baseline_name)]
        
        if config['better'] == 'higher': 
            is_better = delta > 0
            direction_str = "Positive > 0 is Better"
        else: 
            is_better = delta < 0
            direction_str = "Negative < 0 is Better"
            
        prob_better = is_better.mean()
        mean_delta = delta.mean()
        ci_low, ci_high = delta.quantile(0.025), delta.quantile(0.975)
        
        sns.histplot(delta, ax=ax, kde=True, color='#DDDDDD', edgecolor='#666666', 
                     linewidth=0.8, stat='density', alpha=0.6, line_kws={'linewidth': 2, 'color': '#444444'})
        
        ax.axvline(0, color='#666666', linestyle='--', linewidth=1.5, alpha=0.8, label='No Difference')
        ax.axvline(mean_delta, color=COLORS['proposed'], linestyle='-', linewidth=2, label='Mean Δ')
        ax.axvspan(ci_low, ci_high, color=COLORS['proposed'], alpha=0.15, label='95% CI')
        
        stats_text = (f"P(Proposed Better): {prob_better:.1%}\n"
                      f"Mean Δ: {mean_delta:.4f}\n"
                      f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10, family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#CCCCCC'))
        
        data_min, data_max = delta.min(), delta.max()
        view_min = min(data_min, 0)
        view_max = max(data_max, 0)
        margin = (view_max - view_min) * 0.15
        ax.set_xlim(view_min - margin, view_max + margin)
        
        ax.set_title(config['label'], fontweight='bold')
        ax.set_xlabel(f"Δ (Proposed − Percent Sum)\n{direction_str}", style='italic', color='#555555')
        ax.set_ylabel("Density")
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {save_path}")

# ==========================================
# Figure 3: Fairness vs Excitement (No Arrow)
# ==========================================
def plot_tradeoff(save_path='figure3_pareto_front.png'):
    df_points = df_sims.copy()
    df_points['fairness'] = 1 - df_points['merit_loss_rate']
    df_points['excitement'] = df_points['upset_rate']
    
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    for mech in mechanism_order:
        disp_name = NAME_MAPPING[mech]
        subset = df_points[df_points['mechanism'] == disp_name]
        plt.scatter(subset['fairness'], subset['excitement'], 
                    alpha=0.15, s=15, color=COLORS[mech], label=None, 
                    edgecolor='none', rasterized=True)
        
    means = df_points.groupby('mechanism')[['fairness', 'excitement']].mean()
    
    for mech in mechanism_order:
        disp_name = NAME_MAPPING[mech]
        if disp_name not in means.index: continue
        
        mx, my = means.loc[disp_name, 'fairness'], means.loc[disp_name, 'excitement']
        col = COLORS[mech]
        
        plt.scatter(mx, my, s=280, facecolor=col, edgecolor='black', linewidth=1.8, zorder=10)
        
        dx, dy = 0.003, 0.003
        ha, va = 'left', 'bottom'
        if mech == 'rank_sum':
            dx, dy = -0.004, -0.004
            ha, va = 'right', 'top'
        elif mech == 'percent_sum':
            dx, dy = 0.004, -0.004
            ha, va = 'left', 'top'
        elif mech == 'proposed':
            dx, dy = 0.0, 0.008
            ha, va = 'center', 'bottom'
            
        plt.text(mx + dx, my + dy, disp_name, fontsize=12, fontweight='bold', color=col,
                 ha=ha, va=va,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax.text(0.03, 0.97, "Better = Up & Right\n(More Fair + More Exciting)", 
            transform=ax.transAxes, va='top', ha='left', fontsize=11, 
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='#DDDDDD'))
    
    plt.xlabel('Fairness (1 − Merit Loss Rate)', fontsize=13)
    plt.ylabel('Excitement (Upset Rate)', fontsize=13)
    plt.title('Fairness vs. Excitement Trade-off', fontweight='bold', fontsize=15)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {save_path}")

# ==========================================
# Figure 4: Boxplot Panel (Improved Colors)
# ==========================================
def plot_metric_boxpanel(save_path='figure4_boxplots.png'):
    metrics = ['merit_loss_rate', 'controversy_rate', 'upset_rate', 'precision']
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    
    for ax, metric in zip(axes.flatten(), metrics):
        config = METRICS_CONFIG[metric]
        
        # Draw Boxplot with clean outliers (no swarmplot inside)
        sns.boxplot(data=df_sims, x='mechanism', y=metric, order=display_order,
                    ax=ax, width=0.55, linewidth=1.5, 
                    showfliers=True, # Keep outliers
                    zorder=1,
                    # Initialize with explicit props, colors will be overridden below
                    boxprops=dict(facecolor='white', edgecolor='black'),
                    whiskerprops=dict(color='#444444'),
                    capprops=dict(color='#444444'),
                    medianprops=dict(color='#333333', linewidth=2),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5, 
                                    markerfacecolor='none', markeredgecolor='#777777'))
        
        # ---------------------------------------------------------
        # CRITICAL COLOR ADJUSTMENT: 
        # Separate Fill (Transparent) and Edge (Solid)
        # ---------------------------------------------------------
        for i, patch in enumerate(ax.patches):
            if i < len(mechanism_order):
                orig_key = mechanism_order[i]
                col = COLORS[orig_key]
                
                # 1. Fill: Color with Alpha (Soft look)
                rgba_fill = mcolors.to_rgba(col, alpha=0.3)
                patch.set_facecolor(rgba_fill)
                
                # 2. Edge: Solid Color (Sharp look)
                patch.set_edgecolor(col)
                
                # Ensure linewidth is consistent
                patch.set_linewidth(1.5)
        
        better_text = "Lower is better" if config['better'] == 'lower' else "Higher is better"
        ax.set_title(f"{config['label']}", fontweight='bold')
        
        ax.text(0.98, 0.98, better_text, transform=ax.transAxes, 
                ha='right', va='top', fontsize=10, style='italic', color='#555555',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=0)
        
        sns.despine(ax=ax)
        ax.yaxis.grid(True, linestyle='--', alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    print("Generating publication-quality figures...")
    plot_forest_panel()
    plot_delta_panel()
    plot_tradeoff()
    plot_metric_boxpanel()
    print("All tasks completed.")