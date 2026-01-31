import pandas as pd
import numpy as np

# Load datasets
raw_df = pd.read_csv('/Users/garytchois/Desktop/美赛/2026_MCM_Problem_C_Data_副本.csv')
report_df = pd.read_csv('prediction_accuracy_detailed_report.csv')

# Helper function to identify the active roster for a specific week
def get_active_roster(raw_df, season, week):
    season_df = raw_df[raw_df['season'] == season]
    active_names = []
    # Find columns for that week
    cols = [c for c in season_df.columns if c.startswith(f"week{week}_judge") and c.endswith("score")]
    if not cols: return set()
    
    for _, row in season_df.iterrows():
        # A contestant is active if they have valid scores > 0
        if row[cols].sum(min_count=0) > 0:
            active_names.append(row['celebrity_name'].strip())
    return set(active_names)

# Clean data names
report_df['actual_eliminated_list'] = report_df['actual_eliminated'].astype(str).apply(lambda x: [n.strip() for n in x.split(',')])
report_df['predicted_eliminated_list'] = report_df['predicted_eliminated'].astype(str).apply(lambda x: [n.strip() for n in x.split(',')])

results = []
for _, row in report_df.iterrows():
    s, w = row['season'], row['week']
    roster = get_active_roster(raw_df, s, w)
    
    actual_elim = set(row['actual_eliminated_list'])
    pred_elim = set(row['predicted_eliminated_list'])
    
    # Calculate Survivors
    # Intersection with roster ensures we handle name matching somewhat safely
    actual_survivors = roster - actual_elim
    predicted_survivors = roster - pred_elim
    
    # Calculate Accuracy
    intersection = actual_survivors.intersection(predicted_survivors)
    
    if len(actual_survivors) > 0:
        acc = len(intersection) / len(actual_survivors)
    else:
        acc = np.nan # Should not happen unless empty week
        
    results.append({
        'season': s, 'week': w,
        'roster_size': len(roster),
        'actual_survivors_count': len(actual_survivors),
        'predicted_survivors_count': len(predicted_survivors),
        'intersection_count': len(intersection),
        'survival_accuracy': acc
    })

results_df = pd.DataFrame(results)
final_df = pd.concat([report_df, results_df.drop(columns=['season', 'week'])], axis=1)

print(f"Overall Survival Accuracy: {final_df['survival_accuracy'].mean():.4f}")
final_df.to_csv('survival_accuracy_report.csv', index=False)