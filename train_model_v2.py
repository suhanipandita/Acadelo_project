import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt

# 1. Load the Advanced Data
# Make sure you have run 'process_data.py' first!
print("Loading Advanced Data...")
try:
    df = pd.read_csv('processed_data.csv')
except FileNotFoundError:
    print("Error: 'processed_data.csv' not found. Run process_data.py first.")
    exit()

# 2. Define the NEW Feature Set (6 Features)
features = [
    'clicks_total', 
    'days_active', 
    'gap_before_deadline', 
    'material_diversity', 
    'cramming_ratio', 
    'clicks_last_7d'
]

print(f"Training on {len(features)} features: {features}")

X = df[features]
y = df['days_early']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=150,  # More trees for more features
    learning_rate=0.05,
    max_depth=6
)

model.fit(X_train, y_train)

# 5. Evaluate
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"\n--- SUCCESS ---")
print(f"New RMSE: {rmse:.4f} days")
print(f"New R2: {r2:.4f}")

# 6. Save the upgraded model
pickle.dump(model, open("procrastination_model.pkl", "wb"))
print("Saved new model to 'procrastination_model.pkl'")