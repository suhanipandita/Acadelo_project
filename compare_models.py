import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load Data
print("📊 Loading processed data for model showdown...")
try:
    df = pd.read_csv('processed_data.csv')
except FileNotFoundError:
    print("❌ Error: 'processed_data.csv' not found. Run process_data.py first.")
    exit()

# 2. Define Features
features = [
    'clicks_total', 
    'days_active', 
    'gap_before_deadline', 
    'material_diversity', 
    'cramming_ratio', 
    'clicks_last_7d'
]
X = df[features]
y = df['days_early']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define the Competitors
models = {
    "Linear Regression": LinearRegression(),
    
    "Random Forest": RandomForestRegressor(
        n_estimators=200, 
        max_depth=5, 
        random_state=42
    ),
    
    "XGBoost": xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
}

# 4. Train and Evaluate
results = []

print("\nStarting Model Comparison...\n")

for name, model in models.items():
    print(f"Training {name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    preds = model.predict(X_test)
    
    # Score
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    results.append({
        "Model": name,
        "RMSE (Days Error)": round(rmse, 2),
        "R2 Score (Accuracy)": round(r2, 4)
    })

# 5. Display Results
results_df = pd.DataFrame(results).sort_values(by="R2 Score (Accuracy)", ascending=False)

print("\n🏆 FINAL LEADERBOARD:")
print("="*50)
print(results_df.to_string(index=False))
print("="*50)

# 6. Plot the Comparison
plt.figure(figsize=(10, 5))
plt.bar(results_df['Model'], results_df['R2 Score (Accuracy)'], color=['gold', 'silver', 'brown'])
plt.title('Model Accuracy (R2 Score) Comparison')
plt.ylabel('R2 Score (Higher is Better)')
plt.ylim(0, max(results_df['R2 Score (Accuracy)']) + 0.1)
plt.show()