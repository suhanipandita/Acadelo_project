# This file lists the steps taken to complete this ML project.

## Version 1
- relational dataset created by merging required fields from OULAD dataset.
- basic UI with streamlit
- functionality :

### Tab1
Insertion of a single record for a student.

### Tab2
Generate teams and graphs (showing before/after XGBoost model implementation on the relational merged dataset) from records present in the dataset present in supabase.

### Tab3
Insert selected number of records in bulk in the database.

- Accuracy Check (RMSE): 62.9335 days
- Model Fit (R2 Score):  0.1440 


## Version 2
### Features:
- cramming ratio : percentage of work done in past few days =  input(number of clicks in the past week) / total clicks
- material diversity: estimation using a heuristic (usually ~10% of clicks are unique pages), so manual counting is not required.

- Model Accuracy (RMSE): 7.9114 days
- R2 Score: 0.3629

## Version 3
### Features:
- SHAP (SHapley Additive exPlanations) : Explainable AI model added to reason as to which factor(clicks/cramming ratio) contributes to late submission.
    - Base Value: The average prediction for all students (e.g., 0.5 days early).
    - Red Bars: Features dragging the student down (e.g., gap_before_deadline = 10 might push the score down by -2.0).
    - Blue Bars: Features helping the student (e.g., clicks_total = 500 might push the score up by +1.5).
    - Final Value: The sum of everything, which equals the predicted days early/late. 
