import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import xgboost as xgb

# 1. Load the "Brain" (Model) and Data
print("Loading Model and Data...")
model = pickle.load(open("procrastination_model.pkl", "rb"))
df = pd.read_csv('processed_data.csv')

# 2. Simulate a Classroom
# Let's grab a random sample of 40 students to form 10 teams of 4
classroom = df.sample(40, random_state=42).copy()

# 3. Predict Risk Scores
# We use the same features the model was trained on
features = ['clicks_total', 'days_active', 'gap_before_deadline']
classroom['predicted_early'] = model.predict(classroom[features])

# Convert 'Days Early' to a 'Risk Penalty' for easier math
# If they are 5 days early -> Risk is Low. If -2 days (late) -> Risk is High.
# We invert it: Higher number = Higher Risk
classroom['risk_score'] = classroom['predicted_early'] * -1 

print("\n--- Student Risk Profiles ---")
print(classroom[['id_student', 'risk_score']].head())

# ==========================================
# ALGORITHM 1: The "Old Way" (Random)
# ==========================================
classroom = classroom.sample(frac=1, random_state=5) # Shuffle
classroom['team_random'] = np.repeat(range(1, 11), 4) # Assign Team 1-10

# Calculate Total Risk per Team
random_teams = classroom.groupby('team_random')['risk_score'].sum()
print(f"\nRandom Assignment Variance: {random_teams.var():.2f}")

# ==========================================
# ALGORITHM 2: The "Acadelo Way" (Snake Draft)
# ==========================================
# Logic: Sort students by Risk. 
# Team 1 gets Best, Team 2 gets 2nd Best... 
# Then wrap around: Team 10 gets Worst, Team 9 gets 2nd Worst...
# This balances the "heavy lifters" with the "risky" students.

sorted_students = classroom.sort_values('risk_score', ascending=False).reset_index(drop=True)
num_teams = 10
teams = [[] for _ in range(num_teams)]

for i, row in sorted_students.iterrows():
    # Snake Draft Logic
    # 0, 1, 2... 9, 9, 8... 0
    cycle = i // num_teams
    index = i % num_teams
    
    if cycle % 2 == 1: # On odd cycles, reverse direction
        actual_team_idx = (num_teams - 1) - index
    else:
        actual_team_idx = index
        
    teams[actual_team_idx].append(row['risk_score'])

# Sum up risk for smart teams
smart_team_risks = [sum(t) for t in teams]
print(f"Smart Assignment Variance: {np.var(smart_team_risks):.2f}")

# ==========================================
# VISUALIZATION (The Proof)
# ==========================================
plt.figure(figsize=(10, 6))

# Plot Random Teams (Red)
plt.plot(range(1, 11), random_teams, marker='o', color='red', linewidth=2, label='Random Assignment')

# Plot Smart Teams (Green)
plt.plot(range(1, 11), smart_team_risks, marker='o', color='green', linewidth=2, label='Acadelo Smart-Sort')

plt.axhline(y=0, color='black', linestyle='--')
plt.title("Team Viability: Random vs. Smart Allocation")
plt.xlabel("Team ID")
plt.ylabel("Total Procrastination Risk (Lower is Better)")
plt.legend()
plt.grid(True, alpha=0.3)

print("Displaying Graph... Close the window to finish.")
plt.show()