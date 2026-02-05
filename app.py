import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import xgboost as xgb

# Set Page Config
st.set_page_config(page_title="Acadelo-Pro", layout="wide")

# --- HEADER ---
st.title("🎓 Acadelo-Pro")
st.markdown("""
**Smart Team Formation System** | *Powered by XGBoost & Behavioral Analytics*
""")
st.write("---")

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    model = pickle.load(open("procrastination_model.pkl", "rb"))
    data = pd.read_csv('processed_data.csv')
    return model, data

try:
    model, df = load_resources()
    st.success("System Ready: Model Loaded & Data Connected")
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# --- TABS FOR DIFFERENT MODES ---
tab1, tab2 = st.tabs(["🕵️ Single Student Profiler", "⚖️ Auto-Team Balancer"])

# ===================================================
# TAB 1: PREDICT ONE STUDENT (The "Micro" View)
# ===================================================
with tab1:
    st.subheader("Predict Procrastination Risk")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        clicks = st.number_input("Total Clicks (Last 2 Weeks)", min_value=0, max_value=500, value=25)
    with col2:
        active_days = st.number_input("Active Days", min_value=0, max_value=14, value=3)
    with col3:
        gap = st.number_input("Days Since Last Login", min_value=0, max_value=30, value=5)
        
    # Prediction Logic
    if st.button("Analyze Student Risk"):
        # Create input dataframe matching training shape
        input_data = pd.DataFrame([[clicks, active_days, gap]], 
                                columns=['clicks_total', 'days_active', 'gap_before_deadline'])
        
        prediction = model.predict(input_data)[0]
        
        st.write("---")
        st.markdown(f"### Predicted Behavior: **{prediction:.2f} Days Early**")
        
        if prediction < 0:
            st.error(f"⚠️ HIGH RISK: Student is predicted to submit {abs(prediction):.1f} days LATE.")
            st.info("Recommendation: Pair with a 'Tier A' Anchor student.")
        else:
            st.success(f"✅ SAFE: Student is predicted to submit {prediction:.1f} days EARLY.")

# ===================================================
# TAB 2: TEAM FORMATION (The "Macro" View)
# ===================================================
with tab2:
    st.subheader("Optimized Class Partitioning")
    
    # Sliders
    num_students = st.slider("Class Size", 20, 100, 40)
    
    if st.button("Generate Teams"):
        # 1. Simulate Class
        classroom = df.sample(num_students, replace=True).copy() # Use replace=True if dataset is small
        features = ['clicks_total', 'days_active', 'gap_before_deadline']
        
        # 2. Score Students
        classroom['predicted_early'] = model.predict(classroom[features])
        classroom['risk_score'] = classroom['predicted_early'] * -1 # Invert so High = Risky
        
        # 3. Random Assignment (The Control)
        classroom = classroom.sample(frac=1).reset_index(drop=True)
        classroom['team_random'] = np.repeat(range(1, (num_students//4)+1), 4)[:num_students]
        random_risk = classroom.groupby('team_random')['risk_score'].sum()
        
        # 4. Smart Assignment (Snake Draft)
        sorted_students = classroom.sort_values('risk_score', ascending=False).reset_index(drop=True)
        num_teams = num_students // 4
        smart_risks = [0] * num_teams
        
        for i, row in sorted_students.iterrows():
            cycle = i // num_teams
            idx = i % num_teams
            # Snake Logic
            if cycle % 2 == 1:
                team_idx = (num_teams - 1) - idx
            else:
                team_idx = idx
            
            if team_idx < num_teams: # Safety check
                smart_risks[team_idx] += row['risk_score']
                
        # 5. Visual Comparison
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### 🔴 Random Assignment")
            st.write("Notice the spikes? Those are teams destined to fail.")
            fig1, ax1 = plt.subplots()
            ax1.plot(random_risk.values, marker='o', color='red', linestyle='--')
            ax1.set_ylabel("Total Risk Score")
            ax1.set_title(f"Variance: {random_risk.var():.2f}")
            st.pyplot(fig1)
            
        with col_right:
            st.markdown("### 🟢 Acadelo Optimization")
            st.write("Notice the flat line? Risk is shared equally.")
            fig2, ax2 = plt.subplots()
            ax2.plot(smart_risks, marker='o', color='green')
            ax2.set_ylabel("Total Risk Score")
            ax2.set_title(f"Variance: {np.var(smart_risks):.2f}")
            st.pyplot(fig2)

        st.success(f"Optimization Complete! Variance reduced by {((random_risk.var() - np.var(smart_risks))/random_risk.var())*100:.1f}%")