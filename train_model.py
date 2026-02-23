import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# 1. Load the Processed Data
print("Loading high-accuracy processed data...")
df = pd.read_csv('processed_data.csv')

# 2. Define Features (Exact 6)
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

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model (Restrictions Removed)
print("Training XGBoost Regressor...")
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,        
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse) 
r2 = r2_score(y_test, predictions)

print(f"\n--- Model Results ---")
print(f"RMSE: {rmse:.2f} days")
print(f"🏆 R2 Score: {r2:.2f}")

# 6. Save
pickle.dump(model, open("procrastination_model.pkl", "wb"))
print("Model saved. Ready for App.")