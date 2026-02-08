import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# Set Page Config
st.set_page_config(page_title="Acadelo-Pro", layout="wide")

# --- FILE PATHS ---
DB_FILE = 'classroom_database.csv'
MODEL_FILE = 'procrastination_model.pkl'
DATA_FILE = 'processed_data.csv'

# --- HEADER ---
st.title("🎓 Acadelo-Pro")
st.markdown("""
**Smart Team Formation System** | *Powered by XGBoost & Behavioral Analytics*
""")
st.write("---")

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_resources():
    # Load Model and Historical Data (for simulation)
    try:
        model = pickle.load(open(MODEL_FILE, "rb"))
        data = pd.read_csv(DATA_FILE)
        return model, data
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

def init_db():
    # Initialize the CSV database if it doesn't exist
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=['student_name', 'clicks_total', 'days_active', 'gap_before_deadline'])
        df.to_csv(DB_FILE, index=False)

def load_db():
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE)
    return pd.DataFrame()

def save_to_db(name, clicks, active, gap):
    new_data = pd.DataFrame([[name, clicks, active, gap]], 
                           columns=['student_name', 'clicks_total', 'days_active', 'gap_before_deadline'])
    # Append to CSV without loading the whole thing (efficient)
    new_data.to_csv(DB_FILE, mode='a', header=not os.path.exists(DB_FILE), index=False)

# --- INITIALIZATION ---
model, historical_data = load_resources()
init_db()

if model is None:
    st.stop()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Data Entry (Profiler)", "Auto-Team Balancer", "Database View"])

# ===================================================
# TAB 1: DATA ENTRY & SAVING (FIXED)
# ===================================================
with tab1:
    st.subheader("Add Student to Database")
    
    col1, col2 = st.columns(2)
    with col1:
        # FIX: Removed np.random from 'value'. Now it defaults to empty.
        # You can type whatever you want (e.g., "John Doe", "10234")
        s_name = st.text_input("Student Name / ID", value="", placeholder="e.g. Student_01")
        
    with col2:
        clicks = st.number_input("Total Clicks", 0, 1000, 25)
        
    col3, col4 = st.columns(2)
    with col3:
        active_days = st.number_input("Active Days", 0, 30, 3)
    with col4:
        gap = st.number_input("Gap (Days since login)", 0, 60, 5)

    # 1. Predict Live
    input_data = pd.DataFrame([[clicks, active_days, gap]], 
                            columns=['clicks_total', 'days_active', 'gap_before_deadline'])
    
    # Simple check to prevent errors if model acts up
    if model:
        prediction = model.predict(input_data)[0]
        st.info(f"Predicted Submission: **{prediction:.2f} Days Early**")

    # 2. Save Button
    if st.button("💾 Save to Database"):
        if s_name == "":
            st.error("Please enter a Student Name or ID first.")
        else:
            save_to_db(s_name, clicks, active_days, gap)
            st.success(f"Saved **{s_name}** to the class roster!")
            
            # This makes the button disappear and updates the database view immediately
            st.rerun()

            
# ===================================================
# TAB 2: TEAM FORMATION
# ===================================================
with tab2:
    st.subheader("Optimized Class Partitioning")
    
    # Selection: Use Fake Data or Real Database?
    data_source = st.radio("Select Data Source:", 
                          ["Simulate Random Class (Demo)", "Use My Saved Database (Real)"], 
                          horizontal=True)
    
    classroom = pd.DataFrame()

    if "Saved Database" in data_source:
        classroom = load_db()
        if len(classroom) < 4:
            st.warning(f"Not enough students in database! (Count: {len(classroom)}). Please add at least 4 students in Tab 1.")
            st.stop()
        st.write(f"Loaded **{len(classroom)}** students from your custom database.")
    else:
        num_students = st.slider("Simulation Size", 20, 100, 40)
        classroom = historical_data.sample(num_students, replace=True).copy()
        st.write(f"Simulating **{len(classroom)}** random students.")

    if st.button("🚀 Generate Teams"):
        features = ['clicks_total', 'days_active', 'gap_before_deadline']
        
        # 1. Predict Risk for the selected pool
        # Ensure columns match what model expects
        classroom['predicted_early'] = model.predict(classroom[features])
        classroom['risk_score'] = classroom['predicted_early'] * -1 
        
        # 2. Random Assignment (Red Line)
        classroom = classroom.sample(frac=1).reset_index(drop=True)
        # Calculate number of teams
        num_teams = max(1, len(classroom) // 4)
        
        # Create 'Team IDs'
        team_ids = np.array_split(range(len(classroom)), num_teams)
        # Assign team IDs effectively (Flatten logic)
        classroom['team_random'] = 0
        for t_id, indices in enumerate(team_ids):
             classroom.loc[indices, 'team_random'] = t_id + 1

        random_risk = classroom.groupby('team_random')['risk_score'].sum()
        
        # 3. Smart Assignment (Green Line - Snake Draft)
        sorted_students = classroom.sort_values('risk_score', ascending=False).reset_index(drop=True)
        smart_risks = [0] * num_teams
        smart_teams_assignments = [[] for _ in range(num_teams)] # To store names if needed
        
        for i, row in sorted_students.iterrows():
            cycle = i // num_teams
            idx = i % num_teams
            if cycle % 2 == 1:
                team_idx = (num_teams - 1) - idx
            else:
                team_idx = idx
            
            if team_idx < num_teams:
                smart_risks[team_idx] += row['risk_score']
                
        # 4. Visualization
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### 🔴 Random Assignment")
            fig1, ax1 = plt.subplots()
            ax1.plot(random_risk.values, marker='o', color='red', linestyle='--')
            ax1.set_xlabel("Team ID")
            ax1.set_ylabel("Total Risk Score")
            ax1.set_title(f"Variance: {random_risk.var():.2f}")
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            
        with col_right:
            st.markdown("### 🟢 Acadelo Optimization")
            fig2, ax2 = plt.subplots()
            ax2.plot(smart_risks, marker='o', color='green')
            ax2.set_xlabel("Team ID")
            ax2.set_ylabel("Total Risk Score")
            ax2.set_title(f"Variance: {np.var(smart_risks):.2f}")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

        improvement = 0
        if random_risk.var() > 0:
            improvement = ((random_risk.var() - np.var(smart_risks))/random_risk.var())*100
            
        st.success(f"Optimization Complete! Variance reduced by {improvement:.1f}%")

# ===================================================
# TAB 3: DATABASE MANAGEMENT
# ===================================================
with tab3:
    st.subheader("Manage Database")
    current_db = load_db()
    st.dataframe(current_db)
    
    if st.button("🗑️ Clear Database"):
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
            init_db()
            st.warning("Database cleared!")
            st.rerun()