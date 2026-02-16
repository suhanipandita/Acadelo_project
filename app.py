import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, r2_score
from supabase import create_client, Client

# Set Page Config
st.set_page_config(page_title="Acadelo-Pro (Cloud)", layout="wide")

# ===================================================
# 1. CONFIGURATION & CREDENTIALS
# ===================================================
try:
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
except FileNotFoundError:
    st.error("Secrets file not found. Please create .streamlit/secrets.toml")
    st.stop()

MODEL_FILE = 'procrastination_model.pkl'

# ===================================================
# 2. HELPER FUNCTIONS
# ===================================================

@st.cache_resource
def init_supabase():
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {e}")
        return None

@st.cache_resource
def load_model():
    try:
        return pickle.load(open(MODEL_FILE, "rb"))
    except:
        st.error(f"Model file '{MODEL_FILE}' not found. Please run 'train_model.py' first.")
        return None

def load_db(supabase):
    try:
        response = supabase.table('students').select("*").execute()
        data = response.data
        if data:
            return pd.DataFrame(data)
        else:
            # Return empty DF with ALL columns
            return pd.DataFrame(columns=[
                'student_id', 'clicks_total', 'days_active', 'gap_before_deadline',
                'material_diversity', 'cramming_ratio', 'clicks_last_7d'
            ])
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def save_single_student(supabase, row_data):
    """Save one student with all 6 features"""
    try:
        supabase.table('students').insert(row_data).execute()
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False

def bulk_insert_advanced(supabase, num_students=50):
    """
    Generates students with Advanced Behavioral Patterns.
    """
    new_rows = []
    
    # Deterministic Counts
    count_anchor = int(num_students * 0.33)
    count_risk = int(num_students * 0.17)
    count_member = num_students - (count_anchor + count_risk)
    
    # 1. Anchors (Consistent, Low Cramming, High Diversity)
    for _ in range(count_anchor):
        clicks = int(np.random.randint(400, 800))
        cram_ratio = np.random.uniform(0.1, 0.3) # Only 10-30% work in last week
        
        new_rows.append({
            "student_id": f"22310{np.random.randint(100, 999)}",
            "clicks_total": clicks,
            "days_active": int(np.random.randint(15, 30)),
            "gap_before_deadline": int(np.random.randint(0, 2)),
            "material_diversity": int(np.random.randint(10, 25)), # High diversity
            "cramming_ratio": round(cram_ratio, 2),
            "clicks_last_7d": int(clicks * cram_ratio)
        })

    # 2. Members (Average)
    for _ in range(count_member):
        clicks = int(np.random.randint(100, 300))
        cram_ratio = np.random.uniform(0.3, 0.6) # 30-60% work in last week
        
        new_rows.append({
            "student_id": f"22310{np.random.randint(100, 999)}",
            "clicks_total": clicks,
            "days_active": int(np.random.randint(5, 14)),
            "gap_before_deadline": int(np.random.randint(3, 8)),
            "material_diversity": int(np.random.randint(4, 12)),
            "cramming_ratio": round(cram_ratio, 2),
            "clicks_last_7d": int(clicks * cram_ratio)
        })

    # 3. Risks (Panic Workers or Ghosts)
    for _ in range(count_risk):
        clicks = int(np.random.randint(0, 50))
        cram_ratio = np.random.uniform(0.8, 1.0) # 80-100% work in last week (Panic)
        
        new_rows.append({
            "student_id": f"22310{np.random.randint(100, 999)}",
            "clicks_total": clicks,
            "days_active": int(np.random.randint(0, 3)),
            "gap_before_deadline": int(np.random.randint(7, 30)),
            "material_diversity": int(np.random.randint(0, 3)), # Low diversity
            "cramming_ratio": round(cram_ratio, 2),
            "clicks_last_7d": int(clicks * cram_ratio)
        })
    
    import random
    random.shuffle(new_rows)
    
    try:
        supabase.table('students').insert(new_rows).execute()
        return True, len(new_rows)
    except Exception as e:
        st.error(f"Bulk insert failed: {e}")
        return False, 0

# ===================================================
# 3. INITIALIZATION & ACCURACY CHECK
# ===================================================

supabase = init_supabase()
model = load_model()

# --- TERMINAL OUTPUT ---
if model:
    print("\n" + "="*40)
    print(f"🤖 ACADELO SYSTEM ONLINE")
    try:
        # Check against local processed data for accuracy report
        if os.path.exists('processed_data.csv'):
            test_data = pd.read_csv('processed_data.csv')
            # Updated Feature List
            feature_cols = ['clicks_total', 'days_active', 'gap_before_deadline', 
                           'material_diversity', 'cramming_ratio', 'clicks_last_7d']
            
            # Ensure columns exist before testing (backward compatibility check)
            if all(col in test_data.columns for col in feature_cols):
                X_test = test_data[feature_cols]
                y_test = test_data['days_early']
                preds = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)
                print(f"📊 Model Accuracy (RMSE): {rmse:.4f} days")
                print(f"📈 R2 Score: {r2:.4f}")
            else:
                print("⚠️  Data mismatch: CSV missing new columns.")
    except Exception as e:
        print(f"⚠️  Accuracy check skipped: {e}")
    print("="*40 + "\n")

# ===================================================
# 4. UI TABS
# ===================================================

st.title("🎓 Acadelo-Pro")
st.markdown("**Smart Team Formation System** | *Advanced Behavioral Analytics*")
st.write("---")

if not supabase or not model:
    st.stop()

tab1, tab2, tab3 = st.tabs(["📝 Data Entry", "⚖️ Auto-Team Balancer", "💾 Database"])

# --- TAB 1: DATA ENTRY (SMART VERSION) ---
with tab1:
    st.subheader("Add Student Profile")
    
    col1, col2 = st.columns(2)
    with col1:
        s_id = st.text_input("Student ID", value="", placeholder="e.g. 22310884")
        clicks = st.number_input("Total Clicks (Semester)", 0, 2000, 150)
        active = st.number_input("Active Days", 0, 60, 5)
    
    with col2:
        gap = st.number_input("Gap (Days since login)", 0, 60, 2)
        # We ask for "Recent Clicks" to auto-calculate Cramming Ratio
        clicks_recent = st.number_input("Clicks in Last 7 Days", 0, 500, 30, help="Used to calculate Cramming Ratio")

    # --- AUTO-CALCULATION LOGIC ---
    # 1. Cramming Ratio = (Recent / Total)
    if clicks > 0:
        cram_ratio = clicks_recent / clicks
    else:
        cram_ratio = 0.0
        
    # 2. Material Diversity Estimation
    # Logic: Students usually visit 1 unique page for every 15 clicks, capped at 25 unique pages.
    # This is a heuristic to save you manual entry.
    diversity_estimate = min(25, int(clicks / 15)) + 1
    
    # Show the calculated values so the user knows what's happening
    st.info(f"📊 Auto-Calculated Metrics: Cramming Ratio: **{cram_ratio:.2f}** | Diversity Score: **{diversity_estimate}**")

    # Predict Button
    if st.button("☁️ Save & Predict"):
        if s_id == "":
            st.warning("Enter Student ID")
        else:
            # Prepare the row with ALL 6 features
            row_data = {
                "student_id": s_id,
                "clicks_total": clicks,
                "days_active": active,
                "gap_before_deadline": gap,
                "material_diversity": diversity_estimate, # Auto-filled
                "cramming_ratio": cram_ratio,             # Auto-filled
                "clicks_last_7d": clicks_recent
            }
            
            # Save to Supabase
            if save_single_student(supabase, row_data):
                st.success(f"Saved **{s_id}** successfully!")
                
                # --- PREDICTION & EXPLANATION ---
                
                # 1. Prepare Input for Model
                # Create a DataFrame with the EXACT column order the model expects
                feature_names = ['clicks_total', 'days_active', 'gap_before_deadline', 
                               'material_diversity', 'cramming_ratio', 'clicks_last_7d']
                
                input_df = pd.DataFrame([row_data])
                # Filter to keep only feature columns (drop student_id)
                input_df = input_df[feature_names]
                
                # 2. Make Prediction
                pred = model.predict(input_df)[0]
                
                st.write("---")
                col_res, col_why = st.columns([1, 2])
                
                with col_res:
                    st.subheader("Prediction")
                    if pred < 0:
                        st.error(f"⚠️ **{abs(pred):.1f} Days Late**")
                        st.caption("High Risk of Procrastination")
                    else:
                        st.success(f"✅ **{pred:.1f} Days Early**")
                        st.caption("On Track")

                with col_why:
                    st.subheader("💡 Why this result?")
                    
                    # 3. SHAP Explanation
                    import shap
                    
                    # Initialize the explainer with your model
                    # (Ideally, initialize this once at the top of app.py to save time, but here is fine for now)
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(input_df) # Calculate SHAP values for this specific student
                    
                    # Visualize
                    # We use a waterfall plot to show how features push the prediction
                    fig_shap, ax = plt.subplots(figsize=(8, 5))
                    shap.plots.waterfall(shap_values[0], show=False) # [0] because we only have 1 row
                    st.pyplot(fig_shap)
                    
                    st.info("**How to read this:** Red bars push the prediction lower (Late). Blue bars push it higher (Early).")

# --- TAB 2: TEAM FORMATION ---
with tab2:
    st.subheader("Optimized Class Partitioning")
    classroom = load_db(supabase)
    
    if classroom.empty:
        st.warning("Database is empty.")
    else:
        st.write(f"Loaded **{len(classroom)}** students.")
        
        if st.button("🚀 Generate Teams"):
            if len(classroom) < 4:
                st.error("Need 4+ students.")
            else:
                # 1. Predict using ALL 6 features
                features = ['clicks_total', 'days_active', 'gap_before_deadline', 
                           'material_diversity', 'cramming_ratio', 'clicks_last_7d']
                
                # Fill NaNs just in case to prevent crash
                classroom[features] = classroom[features].fillna(0)
                
                classroom['predicted_early'] = model.predict(classroom[features])
                classroom['risk_score'] = classroom['predicted_early'] * -1 # Invert so High = Risky
                
                # Calibration Hack (Override model if strictly necessary)
                classroom.loc[classroom['clicks_total'] < 30, 'risk_score'] = 5.0
                classroom.loc[classroom['clicks_total'] > 500, 'risk_score'] = -5.0
                
                # 2. Smart Logic (Snake Draft)
                num_teams = max(1, len(classroom) // 4)
                sorted_students = classroom.sort_values('risk_score', ascending=False).reset_index(drop=True)
                smart_risks = [0] * num_teams
                final_roster = []
                
                for i, row in sorted_students.iterrows():
                    cycle = i // num_teams
                    idx = i % num_teams
                    
                    # Snake Draft Logic
                    if cycle % 2 == 1:
                        team_idx = (num_teams - 1) - idx
                    else:
                        team_idx = idx
                    
                    if team_idx < num_teams:
                        smart_risks[team_idx] += row['risk_score']
                        
                        # Role Assignment
                        role = "👤 Member"
                        if row['risk_score'] >= 1.5: role = "⚠️ Risk Factor"
                        elif row['risk_score'] <= -2.0: role = "🛡️ Anchor"
                            
                        final_roster.append({
                            "Team ID": team_idx + 1,
                            "Student ID": row['student_id'],
                            "Role": role,
                            "Risk Score": row['risk_score']
                        })
                
                # 3. Graphs
                # Calculate Random for comparison
                classroom_rnd = classroom.copy().sample(frac=1).reset_index(drop=True)
                t_ids = np.array_split(range(len(classroom_rnd)), num_teams)
                classroom_rnd['team_rnd'] = 0
                for t, idxs in enumerate(t_ids): classroom_rnd.loc[idxs, 'team_rnd'] = t+1
                random_risk = classroom_rnd.groupby('team_rnd')['risk_score'].sum()

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🔴 Random")
                    fig1, ax1 = plt.subplots()
                    ax1.plot(random_risk.values, marker='o', color='red', linestyle='--')
                    ax1.set_ylabel("Total Risk Score")
                    ax1.set_title(f"Variance: {random_risk.var():.2f}")
                    st.pyplot(fig1)
                with col2:
                    st.markdown("### 🟢 Acadelo")
                    fig2, ax2 = plt.subplots()
                    ax2.plot(smart_risks, marker='o', color='green')
                    ax2.set_ylabel("Total Risk Score")
                    ax2.set_title(f"Variance: {np.var(smart_risks):.2f}")
                    st.pyplot(fig2)

                # 4. Roster
                st.write("---")
                st.subheader("📋 Final Teams Roster")
                roster_df = pd.DataFrame(final_roster).sort_values(['Team ID', 'Risk Score'])
                st.dataframe(
                    roster_df,
                    column_config={
                        "Team ID": st.column_config.NumberColumn("Team", format="%d"),
                        "Risk Score": st.column_config.ProgressColumn(
                            "Risk", format="%.2f", min_value=-5, max_value=5
                        ),
                    },
                    hide_index=True,
                    use_container_width=True
                )

# --- TAB 3: DB MANAGEMENT ---
with tab3:
    st.subheader("Manage Database")
    current_db = load_db(supabase)
    st.dataframe(current_db, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎲 Bulk Generate")
        num = st.number_input("Count", 1, 200, 60)
        if st.button("Generate & Insert"):
            success, count = bulk_insert_advanced(supabase, num)
            if success:
                st.success(f"Added {count} profiles!")
                st.rerun()
    
    with col2:
        st.subheader("⚠️ Cleanup")
        if st.button("🗑️ DELETE ALL"):
            supabase.table('students').delete().gt('id', 0).execute()
            st.warning("Cleared!")
            st.rerun()