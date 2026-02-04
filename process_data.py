import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

print("Loading OULAD Datasets... (This might take 30 seconds)")

# 1. Load Data with optimized types to save RAM
types = {'id_student': 'int32', 'id_site': 'int32', 'date': 'int16', 'sum_click': 'int16'}
vle = pd.read_csv('data/studentVle.csv', dtype=types)
assessments = pd.read_csv('data/assessments.csv')
student_assessments = pd.read_csv('data/studentAssessment.csv')

print(f"Loaded {len(vle)} click records.")

# 2. Filter Scope: Let's focus on one specific course to start (e.g., 'FFF' - A STEM course)
# Mixing courses can confuse the model initially due to different structures.
target_code_module = 'FFF' 
assessments = assessments[assessments['code_module'] == target_code_module]
print(f"Filtering for Module: {target_code_module}")

# 3. Create the 'Skeleton' (Students + Assessments + Deadlines)
# Merge Student Assessments with Assessment Info to get the 'date' (deadline)
skeleton = pd.merge(student_assessments, assessments, on='id_assessment', how='inner')

# Calculate Target Variable: Days Early (Positive = Good, Negative = Late)
# We handle NaNs in 'date_submitted' by assuming they dropped out (or filter them)
skeleton = skeleton.dropna(subset=['date_submitted']) 
skeleton['days_early'] = skeleton['date'] - skeleton['date_submitted']

# Keep only necessary columns
skeleton = skeleton[['id_student', 'id_assessment', 'date', 'days_early', 'weight']]
skeleton.rename(columns={'date': 'deadline_day'}, inplace=True)

print(f"Found {len(skeleton)} valid submissions to analyze.")

# 4. Feature Engineering: The 'Time-Travel' Aggregation
# We must count clicks for EACH student for EACH assignment, but ONLY clicks before that deadline.

print("Starting Feature Engineering (This loop is slow, grab a coffee)...")

features_list = []

# We iterate through unique assessments to save time (Vectorization is hard here due to variable windows)
unique_assessments = skeleton['id_assessment'].unique()

for assessment_id in unique_assessments:
    # Get the deadline for this specific assessment
    deadline = skeleton[skeleton['id_assessment'] == assessment_id]['deadline_day'].iloc[0]
    
    # 1. Filter VLE for clicks ONLY related to this module and BEFORE the deadline
    # We add a 'buffer' (e.g., only look at clicks in the 14 days before deadline) to be specific
    relevant_vle = vle[
        (vle['code_module'] == target_code_module) & 
        (vle['date'] <= deadline) & 
        (vle['date'] > deadline - 14)  # Look at 2-week window
    ]
    
    # 2. Group by student to get features
    stats = relevant_vle.groupby('id_student').agg(
        clicks_total=('sum_click', 'sum'),
        days_active=('date', 'nunique'),
        last_click_day=('date', 'max')
    ).reset_index()
    
    # 3. Calculate "Procrastination Gap" (Deadline - Last Click)
    stats['gap_before_deadline'] = deadline - stats['last_click_day']
    stats['id_assessment'] = assessment_id
    
    features_list.append(stats)

# Combine all chunks
all_features = pd.concat(features_list)

# 5. Final Merge
# Join our Features onto our Skeleton
final_data = pd.merge(skeleton, all_features, on=['id_student', 'id_assessment'], how='left')

# Fill NaNs (Students who had 0 clicks get 0s, not NaNs)
final_data.fillna(0, inplace=True)

# 6. Save for Phase 2
final_data.to_csv('processed_data.csv', index=False)
print("Success! 'processed_data.csv' created.")
print(final_data.head())