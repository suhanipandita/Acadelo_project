import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
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
    """Connect to Supabase Cloud DB"""
    try:
        if "PASTE_YOUR" in SUPABASE_URL:
            st.error("⚠️ PLEASE UPDATE SUPABASE KEYS IN THE CODE (Line 14-15)")
            return None
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {e}")
        return None

@st.cache_resource
def load_model():
    """Load the trained XGBoost model"""
    try:
        return pickle.load(open(MODEL_FILE, "rb"))
    except:
        st.error(f"Model file '{MODEL_FILE}' not found. Please run 'train_model.py' first.")
        return None

def load_db(supabase):
    """Fetch all students from the cloud"""
    try:
        response = supabase.table('students').select("*").execute()
        data = response.data
        if data:
            return pd.DataFrame(data)
        else:
            # Return empty dataframe with correct columns if DB is empty
            return pd.DataFrame(columns=['student_name', 'clicks_total', 'days_active', 'gap_before_deadline'])
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def save_single_student(supabase, name, clicks, active, gap):
    """Save one student to Supabase"""
    data = {
        "student_name": name, 
        "clicks_total": int(clicks), 
        "days_active": int(active), 
        "gap_before_deadline": int(gap)
    }
    try:
        supabase.table('students').insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False

def bulk_insert_random_students(supabase, num_students=50):
    """Generate random students with IDs like 22310xxx"""
    new_rows = []
    
    for _ in range(num_students):
        # Generate ID format: 22310 + 3 random digits
        student_id = f"22310{np.random.randint(100, 999)}"
        
        row = {
            "student_name": student_id,
            "clicks_total": int(np.random.randint(0, 600)),      # Random clicks 0-600
            "days_active": int(np.random.randint(1, 20)),        # Random active days 1-20
            "gap_before_deadline": int(np.random.randint(0, 30)) # Random gap 0-30 days
        }
        new_rows.append(row)
    
    try:
        # Bulk insert
        supabase.table('students').insert(new_rows).execute()
        return True, len(new_rows)
    except Exception as e:
        st.error(f"Bulk insert failed: {e}")
        return False, 0

# ===================================================
# 3. MAIN APPLICATION UI
# ===================================================

# Initialize Resources
supabase = init_supabase()
model = load_model()

# Header
st.title("🎓 Acadelo-Pro")
st.markdown("""
**Smart Team Formation System** | *Powered by XGBoost & Supabase Cloud DB*
""")
st.write("---")

if not supabase or not model:
    st.stop() # Stop execution if setup fails

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Data Entry (Profiler)", "⚖️ Auto-Team Balancer", "💾 Database Management"])

# --- TAB 1: DATA ENTRY ---
with tab1:
    st.subheader("Add Single Student to Cloud DB")
    
    col1, col2 = st.columns(2)
    with col1:
        # Value is empty string so it doesn't reset while typing
        s_name = st.text_input("Student Name / ID", value="", placeholder="e.g. 22310884")
    with col2:
        clicks = st.number_input("Total Clicks", 0, 1000, 150)
        
    col3, col4 = st.columns(2)
    with col3:
        active_days = st.number_input("Active Days", 0, 30, 5)
    with col4:
        gap = st.number_input("Gap (Days since login)", 0, 60, 2)

    # Live Prediction
    if st.button("🔍 Analyze Risk"):
        input_data = pd.DataFrame([[clicks, active_days, gap]], 
                                columns=['clicks_total', 'days_active', 'gap_before_deadline'])
        prediction = model.predict(input_data)[0]
        
        st.write("---")
        if prediction < 0:
            st.error(f"⚠️ HIGH RISK: Predicted to submit **{abs(prediction):.1f} days LATE**.")
        else:
            st.success(f"✅ SAFE: Predicted to submit **{prediction:.1f} days EARLY**.")

    # Save Button
    if st.button("☁️ Save to Cloud"):
        if s_name == "":
            st.warning("Please enter a Student Name/ID.")
        else:
            success = save_single_student(supabase, s_name, clicks, active_days, gap)
            if success:
                st.success(f"Saved **{s_name}** to Supabase!")
                # Optional: st.rerun() to refresh, but might reset inputs

# --- TAB 2: TEAM FORMATION ---
with tab2:
    st.subheader("Optimized Class Partitioning")
    
    # Load Live Data
    classroom = load_db(supabase)
    
    if classroom.empty:
        st.warning("Database is empty. Go to Tab 3 to generate random students.")
    else:
        st.write(f"Loaded **{len(classroom)}** students from Supabase.")
        
        if st.button("🚀 Generate Teams"):
            if len(classroom) < 4:
                st.error("Need at least 4 students to form teams.")
            else:
                # 1. Predict Risk Scores
                features = ['clicks_total', 'days_active', 'gap_before_deadline']
                # Ensure input columns match model expectations
                classroom['predicted_early'] = model.predict(classroom[features])
                classroom['risk_score'] = classroom['predicted_early'] * -1 # Invert: High score = High risk
                
                # 2. Random Logic (Control Group)
                classroom = classroom.sample(frac=1).reset_index(drop=True)
                num_teams = max(1, len(classroom) // 4)
                
                # Assign Team IDs cyclically for random distribution
                team_ids = np.array_split(range(len(classroom)), num_teams)
                classroom['team_random'] = 0
                for t_id, indices in enumerate(team_ids):
                     classroom.loc[indices, 'team_random'] = t_id + 1
                     
                random_risk = classroom.groupby('team_random')['risk_score'].sum()
                
                # 3. Smart Logic (Snake Draft Algorithm)
                sorted_students = classroom.sort_values('risk_score', ascending=False).reset_index(drop=True)
                smart_risks = [0] * num_teams
                
                # Snake Draft Implementation
                for i, row in sorted_students.iterrows():
                    cycle = i // num_teams
                    idx = i % num_teams
                    if cycle % 2 == 1:
                        team_idx = (num_teams - 1) - idx # Reverse direction
                    else:
                        team_idx = idx # Normal direction
                    
                    if team_idx < num_teams:
                        smart_risks[team_idx] += row['risk_score']
                
                # 4. Graphs
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("### 🔴 Random Assignment")
                    st.caption("High variance = Some teams will fail.")
                    fig1, ax1 = plt.subplots()
                    ax1.plot(random_risk.values, marker='o', color='red', linestyle='--', linewidth=2)
                    ax1.set_xlabel("Team ID")
                    ax1.set_ylabel("Total Risk Score")
                    ax1.set_title(f"Variance: {random_risk.var():.2f}")
                    ax1.grid(True, alpha=0.3)
                    st.pyplot(fig1)
                    
                with col_right:
                    st.markdown("### 🟢 Acadelo Optimization")
                    st.caption("Low variance = Fair workload distribution.")
                    fig2, ax2 = plt.subplots()
                    ax2.plot(smart_risks, marker='o', color='green', linewidth=2)
                    ax2.set_xlabel("Team ID")
                    ax2.set_ylabel("Total Risk Score")
                    ax2.set_title(f"Variance: {np.var(smart_risks):.2f}")
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig2)

                # Improvement Metric
                imp = ((random_risk.var() - np.var(smart_risks))/random_risk.var())*100
                st.success(f"Optimization Complete! Team Stability Improved by **{imp:.1f}%**")

# --- TAB 3: DATABASE MANAGEMENT ---
with tab3:
    st.subheader("Manage Live Database")
    
    # 1. Show Current Data
    current_db = load_db(supabase)
    st.dataframe(current_db, height=300, use_container_width=True)
    st.write(f"**Total Records:** {len(current_db)}")
    
    st.write("---")
    
    col1, col2 = st.columns(2)
    
    # --- BULK GENERATE BUTTON ---
    with col1:
        st.subheader("🎲 Generate Random Class")
        num_to_gen = st.number_input("Number of Students", min_value=1, max_value=200, value=50)
        
        if st.button("Generate & Insert Records"):
            with st.spinner("Generating synthetic student data..."):
                success, count = bulk_insert_random_students(supabase, num_to_gen)
                if success:
                    st.success(f"Successfully added {count} new students (IDs starting with 22310)!")
                    st.rerun()

    # --- DELETE BUTTON ---
    with col2:
        st.subheader("⚠️ Danger Zone")
        if st.button("🗑️ DELETE ALL RECORDS"):
            try:
                # Delete all rows where ID > 0
                supabase.table('students').delete().gt('id', 0).execute()
                st.warning("Database Cleared Successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing DB: {e}")