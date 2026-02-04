import pandas as pd
import numpy as np # Added numpy for the fix
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

# 1. Load the Processed Data
print("Loading processed data...")
df = pd.read_csv('processed_data.csv')

# 2. Define Features (X) and Target (Y)
features = ['clicks_total', 'days_active', 'gap_before_deadline']
X = df[features]
y = df['days_early']

# 3. Split Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train XGBoost
print("Training XGBoost Regressor...")
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5
)

model.fit(X_train, y_train)

# 5. Evaluate Performance (FIXED THIS PART)
predictions = model.predict(X_test)

# Calculate MSE first, then take square root manually to avoid version errors
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse) 
r2 = r2_score(y_test, predictions)

print(f"\n--- Model Results ---")
print(f"RMSE: {rmse:.2f} days (Average error in prediction)")
print(f"R2 Score: {r2:.2f} (How well we explain the variance)")

# 6. Feature Importance
print("\nGenerating Feature Importance Plot...")
xgb.plot_importance(model)
plt.title("What Drives Procrastination?")
plt.show()

# 7. Save the Model
pickle.dump(model, open("procrastination_model.pkl", "wb"))
print("Model saved as 'procrastination_model.pkl'. Ready for App.")