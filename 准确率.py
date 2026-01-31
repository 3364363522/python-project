import pandas as pd
import numpy as np

def calculate_accuracy():
    # ---------------------------------------------------------
    # 1. 读取并预处理数据
    # ---------------------------------------------------------
    
    # 读取原始数据 (真实情况)
    raw_df = pd.read_csv('/Users/garytchois/Desktop/美赛/2026_MCM_Problem_C_Data.csv')
    
    # 读取预测数据 (后验分布)
    pred_df = pd.read_csv('/Users/garytchois/Desktop/mcmc_outputs_12000/w_0p5/posterior_vote_share_summary.csv')
    
    # 清理列名，方便处理
    raw_df.columns = [c.lower().strip() for c in raw_df.columns]
    
    # ---------------------------------------------------------
    # 2. 从原始数据中提取每周的真实评委分数和淘汰情况
    # ---------------------------------------------------------
    
    # 创建一个字典来存储真实情况: actual_status[(season, week)] = { 'eliminated': [names], 'active': [names], 'scores': {name: score} }
    ground_truth = {}
    
    # 获取最大周数 (根据列名推断，假设最大到20周左右，遍历查找)
    week_cols = [c for c in raw_df.columns if 'week' in c and 'judge' in c]
    # 提取唯一的周数序号
    weeks = sorted(list(set([int(c.split('week')[1].split('_')[0]) for c in week_cols])))
    
    # 辅助函数：获取某选手某周的总分
    def get_week_score(row, w):
        # 找到该周的所有评委打分列
        cols = [c for c in raw_df.columns if f'week{w}_' in c and 'judge' in c]
        total = 0
        valid_scores = 0
        for c in cols:
            val = row[c]
            # 处理 N/A 或空值
            if pd.isna(val) or str(val).strip().upper() == 'N/A' or str(val).strip() == '':
                continue
            try:
                score = float(val)
                total += score
                valid_scores += 1
            except:
                continue
        return total if valid_scores > 0 else 0

    # 遍历每个赛季和选手，构建时间线
    # 我们按赛季分组处理
    seasons = raw_df['season'].unique()
    
    # 存储处理后的真实数据行，方便后续merge
    truth_rows = []

    for season in seasons:
        season_df = raw_df[raw_df['season'] == season]
        
        for w in weeks:
            # 下一周 (用于判断淘汰)
            next_w = w + 1
            
            eliminated_this_week = []
            active_contestants = []
            contestant_scores = {}
            
            for idx, row in season_df.iterrows():
                name = row['celebrity_name']
                
                # 获取当前周分数
                current_score = get_week_score(row, w)
                
                # 如果当前周没有分数，说明这周他已经不在比赛了(或者还没开始)，跳过
                if current_score == 0:
                    continue
                
                # 记录该选手本周在场
                active_contestants.append(name)
                contestant_scores[name] = current_score
                
                # 判断是否本周被淘汰
                # 逻辑：本周有分，下一周分数为0 (或没有下一周的数据)
                # 注意：如果是决赛周(最后一周)，通常不会有next_w的分数，但不算"淘汰"。
                # 我们通过检查该赛季是否有任何人在next_w有分数来判断next_w是否存在。
                
                next_score = get_week_score(row, next_w)
                
                # 检查该赛季该周次是否整体存在（如果所有人在next_w都是0，说明本周是决赛周，没有淘汰）
                season_next_week_active = False
                for _, r_check in season_df.iterrows():
                    if get_week_score(r_check, next_w) > 0:
                        season_next_week_active = True
                        break
                
                if season_next_week_active and next_score == 0:
                    eliminated_this_week.append(name)
            
            # 如果本周有活跃选手，记录数据
            if active_contestants:
                # 记录每一位活跃选手的状态
                for name in active_contestants:
                    is_elim = name in eliminated_this_week
                    truth_rows.append({
                        'season': season,
                        'week': w,
                        'celebrity_name': name,
                        'actual_judge_total': contestant_scores[name],
                        'is_eliminated': is_elim,
                        'total_eliminated_count': len(eliminated_this_week) # 本周共淘汰几人
                    })

    truth_df = pd.DataFrame(truth_rows)
    
    # ---------------------------------------------------------
    # 3. 合并预测数据与真实数据
    # ---------------------------------------------------------
    
    # 确保列名一致以便合并
    # 预测数据列: season, week, celebrity_name, judge_total, p_mean
    
    # 合并 (Inner Join，只分析我们在真实数据中确认活跃的周次)
    # 注意：名字可能需要strip处理以防空格
    pred_df['celebrity_name'] = pred_df['celebrity_name'].str.strip()
    truth_df['celebrity_name'] = truth_df['celebrity_name'].str.strip()
    
    merged_df = pd.merge(truth_df, pred_df[['season', 'week', 'celebrity_name', 'p_mean']], 
                         on=['season', 'week', 'celebrity_name'], 
                         how='inner')
    
    # ---------------------------------------------------------
    # 4. 应用淘汰规则并计算准确率
    # ---------------------------------------------------------
    
    results = []
    
    # 按赛季和周分组进行模拟
    grouped = merged_df.groupby(['season', 'week'])
    
    for (season, week), group in grouped:
        # 获取本周实际淘汰的人数
        n_elim = group['total_eliminated_count'].iloc[0]
        
        # 如果本周没人淘汰 (n_elim = 0)，通常是决赛或者是特殊周，跳过或记录为N/A
        if n_elim == 0:
            continue
            
        # 复制一份组数据避免警告
        df_week = group.copy()
        
        # --- 规则分支 ---
        
        predicted_eliminated_names = []
        
        if season <= 2:
            # === 排名法 (Ranking Method) ===
            # 1. 评委排名: 分数越高，排名越靠前 (Rank 1 is best). method='min' means ties share top rank.
            # Ascending=False means High Score -> Rank 1.
            df_week['rank_judge'] = df_week['actual_judge_total'].rank(ascending=False, method='min')
            
            # 2. 粉丝排名: p_mean 越高，排名越靠前
            df_week['rank_fan'] = df_week['p_mean'].rank(ascending=False, method='min')
            
            # 3. 总排名 = 评委排名 + 粉丝排名
            df_week['total_metric'] = df_week['rank_judge'] + df_week['rank_fan']
            
            # 4. 淘汰规则: 最大的几个人淘汰 (Rank Sum 越大表示表现越差)
            # 我们按照 total_metric 降序排列
            df_week = df_week.sort_values(by=['total_metric', 'p_mean'], ascending=[False, True]) # p_mean secondary sort just in case
            
        else:
            # === 百分比法 (Percentage Method) ===
            # 1. 评委百分比
            total_judge_score = df_week['actual_judge_total'].sum()
            df_week['pct_judge'] = df_week['actual_judge_total'] / total_judge_score
            
            # 2. 粉丝百分比 (p_mean 已经是归一化的)
            # 但为了严谨，确保当前组内的 p_mean sum 为 1 (因为模型可能是对全员预测，而这里只剩下未淘汰的)
            # 用户提示说 p_mean 是归一化后的，我们假设是在全部存活选手里归一化。
            # 这里我们重新归一化一下当前活跃选手的 p_mean 以防万一，或者直接用。
            # 按照题目百分比法逻辑: (Score/Total) + (Votes/Total). 
            current_p_sum = df_week['p_mean'].sum()
            df_week['pct_fan'] = df_week['p_mean'] / current_p_sum
            
            # 3. 总分
            df_week['total_metric'] = df_week['pct_judge'] + df_week['pct_fan']
            
            # 4. 淘汰规则: 分数最小的几个人淘汰
            # 按照 total_metric 升序排列 (分数越低越差)
            df_week = df_week.sort_values(by=['total_metric'], ascending=[True])
            
        # --- 判定预测结果 ---
        
        # 选取前 n_elim 个人作为预测淘汰者
        predicted_eliminated = df_week.iloc[:n_elim]['celebrity_name'].tolist()
        
        # 实际淘汰者
        actual_eliminated = df_week[df_week['is_eliminated'] == True]['celebrity_name'].tolist()
        
        # --- 记录结果 ---
        
        # 1. 完全准确 (Exact Match): 集合相同
        is_exact_match = set(predicted_eliminated) == set(actual_eliminated)
        
        # 2. 软准确 (Soft Match): 预测淘汰的人里，至少有一个是真的被淘汰了
        # (这在淘汰2人预测中猜对1个时很有用，或者衡量"危险区"预测能力)
        intersection = set(predicted_eliminated).intersection(set(actual_eliminated))
        correct_count = len(intersection)
        
        results.append({
            'season': season,
            'week': week,
            'n_to_eliminate': n_elim,
            'actual_eliminated': ", ".join(actual_eliminated),
            'predicted_eliminated': ", ".join(predicted_eliminated),
            'is_exact_match': is_exact_match,
            'correct_count': correct_count,
            'total_eliminated': n_elim
        })

    # ---------------------------------------------------------
    # 5. 生成统计报告
    # ---------------------------------------------------------
    
    results_df = pd.DataFrame(results)
    
    # 总预测次数 (周次数量)
    total_weeks = len(results_df)
    
    # 完全正确的周次
    total_exact_matches = results_df['is_exact_match'].sum()
    
    # 准确率
    accuracy = total_exact_matches / total_weeks if total_weeks > 0 else 0
    
    # 另一种准确率：按人头算 (预测了100个人次被淘汰，其中有多少是对的)
    total_people_predicted = results_df['n_to_eliminate'].sum()
    total_people_correct = results_df['correct_count'].sum()
    person_accuracy = total_people_correct / total_people_predicted if total_people_predicted > 0 else 0
    
    print("=== Prediction Accuracy Analysis Report ===")
    print(f"Total Prediction Events (Weeks analyzed): {total_weeks}")
    print(f"Seasons Covered: {results_df['season'].unique()}")
    print(f"Exact Match Accuracy (Week-level): {accuracy:.2%} ({total_exact_matches}/{total_weeks})")
    print(f"Individual Elimination Accuracy (Person-level): {person_accuracy:.2%} ({total_people_correct}/{total_people_predicted})")
    print("\nSample Results (First 10 rows):")
    print(results_df[['season', 'week', 'actual_eliminated', 'predicted_eliminated', 'is_exact_match']].head(10).to_string())
    
    # 保存详细结果到CSV
    results_df.to_csv('prediction_accuracy_detailed_report.csv', index=False)
    print("\nDetailed report saved to 'prediction_accuracy_detailed_report.csv'")

if __name__ == "__main__":
    calculate_accuracy()