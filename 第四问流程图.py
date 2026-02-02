import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart():
    # 1. 设置画布
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 130)
    ax.axis('off')  # 关闭坐标轴

    # --- 定义绘图辅助函数 ---
    def draw_box(x, y, w, h, text, color='#E6F3FF', edgecolor='black', shape='rect'):
        if shape == 'rect':
            p = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1,rounding_size=0.5", 
                                       linewidth=1.5, edgecolor=edgecolor, facecolor=color)
        elif shape == 'diamond':
            p = patches.Polygon([[x+w/2, y], [x+w, y+h/2], [x+w/2, y+h], [x, y+h/2]], 
                                closed=True, linewidth=1.5, edgecolor=edgecolor, facecolor=color)
        elif shape == 'circle':
            p = patches.Ellipse((x+w/2, y+h/2), w, h, linewidth=1.5, edgecolor=edgecolor, facecolor=color)
        
        ax.add_patch(p)
        
        # 添加文本
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, 
                fontweight='bold', fontname='Arial', color='black', wrap=True)
        return p

    def draw_arrow(x1, y1, x2, y2, text=None):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='#555555'))
        if text:
            mid_x = (x1 + x2) / 2
            mid_y = (x1 + x2) / 2 # 简单取中点可能会重叠，根据具体情况调整
            # 这里简单处理，针对垂直线
            ax.text(x1 + 0.5, (y1+y2)/2, text, fontsize=9, color='#333333', ha='left')

    # --- 2. 绘制节点 (从上到下) ---

    # Level 0: Start
    draw_box(40, 122, 20, 6, "Start: Week w", color='#D3D3D3', shape='circle') # Top

    # Level 1: Inputs
    draw_box(10, 110, 25, 6, "Raw Judge Scores (J)", color='#FFFACD')
    draw_box(40, 110, 20, 6, "Week Number (w)", color='#F0F8FF')
    draw_box(65, 110, 25, 6, "Raw Fan Votes (F)", color='#FFFACD')

    # Level 2: Processing
    draw_box(10, 98, 25, 6, "Standardization\n(Z-Score)", color='#E0FFFF')
    draw_box(40, 98, 20, 6, "Calc Dynamic Weight\n α(w)", color='#E0FFFF')
    draw_box(65, 98, 25, 6, "Concave Transform\ng(f) = (f+ε)^γ", color='#FFDAB9') # 你的亮点

    # Level 3: Synthesis
    draw_box(30, 84, 40, 8, "Calculate Safety Score (S_iw)\nS = α*J_norm + (1-α)*g(f)", color='#98FB98')

    # Level 4: Ranking & Bottom-3
    draw_box(30, 72, 40, 6, "Identify Bottom-3 Set (B_w)", color='#FFB6C1')

    # Level 5: Fan Save (Stage 1)
    draw_box(30, 58, 40, 8, "Stage 1: Fan Save\nSelect Max g(f) in B_w", color='#FFD700')
    
    # Branch out for Fan Save
    draw_box(75, 58, 20, 6, "Contestant Safe", color='#90EE90')

    # Level 6: Remaining 2
    draw_box(30, 46, 40, 6, "2 Contestants Remain\n(Risk Set)", color='#D3D3D3')

    # Level 7: Judge Save Decision (Stage 2)
    # Diamond shape manually approximated by placement logic or using specific patch
    draw_box(30, 30, 40, 10, "Decision:\nJudges use Save Token?", shape='diamond', color='#F0E68C')

    # Level 8: Outcomes
    # No Save (Left path)
    draw_box(5, 15, 30, 8, "Natural Elimination\n(Lowest S_iw removed)", color='#FF6347')
    
    # Yes Save (Right path)
    draw_box(65, 15, 30, 8, "Judge Intervention\n(Vote to save 1, other out)\n(Token -1)", color='#87CEFA')

    # Level 9: End
    draw_box(40, 2, 20, 6, "Update Roster\nNext Week", color='#D3D3D3', shape='circle')

    # --- 3. 绘制连接线 (Arrows) ---
    
    # Inputs -> Processing
    draw_arrow(50, 122, 50, 118) # Start down (approx)
    
    # Connect Inputs to Process
    draw_arrow(22.5, 110, 22.5, 106) # J -> Std
    draw_arrow(50, 110, 50, 106)     # w -> Weight
    draw_arrow(77.5, 110, 77.5, 106) # F -> Concave

    # Process -> Synthesis
    draw_arrow(22.5, 96, 40, 93)
    draw_arrow(50, 96, 50, 93)
    draw_arrow(77.5, 96, 60, 93)

    # Synthesis -> Rank
    draw_arrow(50, 82, 50, 80)

    # Rank -> Fan Save
    draw_arrow(50, 70, 50, 68)

    # Fan Save -> Safe
    draw_arrow(72, 62, 74, 62, "Top Fan\nPick")
    
    # Fan Save -> Remaining
    draw_arrow(50, 56, 50, 54, "Others")

    # Remaining -> Judge Decision
    draw_arrow(50, 44, 50, 41)

    # Decision -> No (Left)
    # Start from left point of diamond (30, 35) to top of box (20, 23)
    # Approximating positions based on box logic
    ax.annotate('No / Tokens Used Up', xy=(20, 24), xytext=(30, 35),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='red'), fontsize=9, ha='right')

    # Decision -> Yes (Right)
    ax.annotate('Yes (Uses Token)', xy=(80, 24), xytext=(70, 35),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='green'), fontsize=9, ha='left')

    # Outcomes -> End
    draw_arrow(20, 14, 45, 9)
    draw_arrow(80, 14, 55, 9)
    
    # Safe -> End (Long arrow)
    # Draw a line from Safe box down to End
    ax.plot([85, 85, 65], [58, 5, 5], color='#555555', lw=1.5, linestyle='--')
    ax.annotate('', xy=(62, 5), xytext=(65, 5), arrowprops=dict(arrowstyle='->', lw=1.5, color='#555555'))

    plt.tight_layout()
    plt.savefig('mechanism_flowchart_matplotlib.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    draw_flowchart()