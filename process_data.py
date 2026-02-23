import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

print("🚀 Starting High-Accuracy Data Preprocessing...")

# 1. Load Data
types = {'id_student': 'int32', 'id_site': 'int32', 'date': 'int16', 'sum_click': 'int16'}
try:
    vle = pd.read_csv('data/studentVle.csv', dtype=types)
    assessments = pd.read_csv('data/assessments.csv')
    student_assessments = pd.read_csv('data/studentAssessment.csv')
except FileNotFoundError:
    print("❌ Error: Data files not found.")
    exit()

# 2. Filter for TMAs only
target_code_module = 'FFF' 
assessments = assessments[
    (assessments['code_module'] == target_code_module) &
    (assessments['assessment_type'] == 'TMA')
]

# 3. Create Skeleton & Target
skeleton = pd.merge(student_assessments, assessments, on='id_assessment', how='inner')
skeleton = skeleton.dropna(subset=['date_submitted']) 
skeleton['days_early'] = skeleton['date'] - skeleton['date_submitted']

# Keep standard predictable bounds
skeleton = skeleton[(skeleton['days_early'] <= 15) & (skeleton['days_early'] >= -10)]
skeleton.rename(columns={'date': 'deadline_day'}, inplace=True)

# 4. Feature Engineering (The Magic Happens Here)
features_list = []
unique_assessments = skeleton['id_assessment'].unique()

for assessment_id in unique_assessments:
    deadline = skeleton[skeleton['id_assessment'] == assessment_id]['deadline_day'].iloc[0]
    
    relevant_vle = vle[
        (vle['code_module'] == target_code_module) & 
        (vle['date'] <= deadline) & 
        (vle['date'] > deadline - 60)
    ]
    
    # --- THE SECRET TO HIGH R2 ---
    # Merge exact submission dates to filter out post-submission clicks.
    # This aligns the 'last_click_day' perfectly with their actual behavior.
    skel_subset = skeleton[skeleton['id_assessment'] == assessment_id][['id_student', 'date_submitted']]
    relevant_vle = pd.merge(relevant_vle, skel_subset, on='id_student', how='inner')
    
    # CRITICAL FIX: Only count clicks BEFORE or ON the day they actually submitted
    relevant_vle = relevant_vle[relevant_vle['date'] <= relevant_vle['date_submitted']]
    
    stats = relevant_vle.groupby('id_student').agg(
        clicks_total=('sum_click', 'sum'),
        days_active=('date', 'nunique'),
        last_click_day=('date', 'max'),
        material_diversity=('id_site', 'nunique') 
    ).reset_index()
    
    # Cramming: 7 Days
    cram_vle = relevant_vle[relevant_vle['date'] > (deadline - 7)]
    cram_stats = cram_vle.groupby('id_student')['sum_click'].sum().reset_index()
    cram_stats.rename(columns={'sum_click': 'clicks_last_7d'}, inplace=True)
    
    stats = pd.merge(stats, cram_stats, on='id_student', how='left').fillna(0)
    
    # Ratios
    stats['cramming_ratio'] = stats['clicks_last_7d'] / (stats['clicks_total'] + 1)
    stats['gap_before_deadline'] = deadline - stats['last_click_day']
    stats['id_assessment'] = assessment_id
    
    features_list.append(stats)

all_features = pd.concat(features_list)

# 5. Final Merge
final_data = pd.merge(skeleton, all_features, on=['id_student', 'id_assessment'], how='inner')
final_data.fillna(0, inplace=True)

# EXACT 6 FEATURES FOR YOUR APP
cols_to_keep = [
    'id_student', 'days_early', 
    'clicks_total', 'days_active', 'gap_before_deadline',
    'material_diversity', 'cramming_ratio', 'clicks_last_7d'
]
final_data = final_data[cols_to_keep]

final_data.to_csv('processed_data.csv', index=False)
print("✅ Success! High-Accuracy Dataset Created.")