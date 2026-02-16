import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

print("🚀 Starting Advanced Data Preprocessing...")

# 1. Load Data with optimized types to save RAM
types = {'id_student': 'int32', 'id_site': 'int32', 'date': 'int16', 'sum_click': 'int16'}
try:
    vle = pd.read_csv('data/studentVle.csv', dtype=types)
    assessments = pd.read_csv('data/assessments.csv')
    student_assessments = pd.read_csv('data/studentAssessment.csv')
except FileNotFoundError:
    print("❌ Error: Data files not found in 'data/' folder.")
    exit()

print(f"Loaded {len(vle)} click records.")

# 2. Filter Scope: Focus on STEM course 'FFF'
target_code_module = 'FFF' 
assessments = assessments[assessments['code_module'] == target_code_module]
print(f"Filtering for Module: {target_code_module}")

# 3. Create the 'Skeleton' (Target Variable)
skeleton = pd.merge(student_assessments, assessments, on='id_assessment', how='inner')
skeleton = skeleton.dropna(subset=['date_submitted']) 

# Calculate Target: Days Early
skeleton['days_early'] = skeleton['date'] - skeleton['date_submitted']

# Remove outliers (Submitted >40 days early or >20 days late is usually an error)
skeleton = skeleton[(skeleton['days_early'] < 40) & (skeleton['days_early'] > -20)]

# Keep necessary columns
skeleton = skeleton[['id_student', 'id_assessment', 'date', 'days_early', 'weight']]
skeleton.rename(columns={'date': 'deadline_day'}, inplace=True)

print(f"Found {len(skeleton)} valid submissions to analyze.")

# 4. ADVANCED Feature Engineering
print("Starting Behavioral Analysis (This loop takes 1-2 mins)...")

features_list = []
unique_assessments = skeleton['id_assessment'].unique()

for assessment_id in unique_assessments:
    # Get deadline
    deadline = skeleton[skeleton['id_assessment'] == assessment_id]['deadline_day'].iloc[0]
    
    # FILTER 1: Broad Study Window (Last 60 Days)
    # We look further back to see if they are consistent or just cramming
    relevant_vle = vle[
        (vle['code_module'] == target_code_module) & 
        (vle['date'] <= deadline) & 
        (vle['date'] > deadline - 60)
    ]
    
    # AGGREGATION: Basic Stats
    stats = relevant_vle.groupby('id_student').agg(
        clicks_total=('sum_click', 'sum'),
        days_active=('date', 'nunique'),
        last_click_day=('date', 'max'),
        material_diversity=('id_site', 'nunique') # NEW: How many diff things did they open?
    ).reset_index()
    
    # FILTER 2: Cramming Window (Last 7 Days)
    cram_vle = relevant_vle[relevant_vle['date'] > (deadline - 7)]
    cram_stats = cram_vle.groupby('id_student')['sum_click'].sum().reset_index()
    cram_stats.rename(columns={'sum_click': 'clicks_last_7d'}, inplace=True)
    
    # Merge Cramming Stats
    stats = pd.merge(stats, cram_stats, on='id_student', how='left').fillna(0)
    
    # CALCULATE RATIOS
    # Cramming Ratio: (Clicks in last 7 days) / (Total Clicks + 1)
    stats['cramming_ratio'] = stats['clicks_last_7d'] / (stats['clicks_total'] + 1)
    
    # Gap: Days since last login
    stats['gap_before_deadline'] = deadline - stats['last_click_day']
    stats['id_assessment'] = assessment_id
    
    features_list.append(stats)

# Combine all chunks
all_features = pd.concat(features_list)

# 5. Final Merge
# Use LEFT join so students with 0 clicks are kept (as 0s), not dropped
final_data = pd.merge(skeleton, all_features, on=['id_student', 'id_assessment'], how='left')

# Fill NaNs
final_data.fillna(0, inplace=True)

# Select final columns
cols_to_keep = [
    'id_student', 'days_early', 
    'clicks_total', 'days_active', 'gap_before_deadline',
    'material_diversity', 'cramming_ratio', 'clicks_last_7d'
]
final_data = final_data[cols_to_keep]

# 6. Save
final_data.to_csv('processed_data.csv', index=False)
print("✅ Success! 'processed_data.csv' created with Advanced Features.")
print(final_data.head())