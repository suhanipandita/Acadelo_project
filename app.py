import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from supabase import create_client, Client

# Set Page Config
st.set_page_config(page_title="Acadelo-Pro (Cloud)", layout="wide")

# --- 1. SUPABASE CREDENTIALS ---
try:
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
except FileNotFoundError:
    st.error("Secrets file not found. Please create .streamlit/secrets.toml")
    st.stop()

# --- HEADER ---
st.title("🎓 Acadelo-Pro")
st.markdown("""
**Smart Team Formation System** | *Powered by XGBoost & Supabase Cloud DB*
""")
st.write("---")

# --- 2. HELPER FUNCTIONS (UPDATED FOR CLOUD) ---
@st.cache_resource
def init_supabase():
    # Connect to the cloud database
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {e}")
        return None

@st.cache_resource
def load_model():
    # Load Model (Local File)
    try:
        return pickle.load(open('procrastination_model.pkl', "rb"))
    except:
        st.error("Model file not found. Please run 'train_model.py' first.")
        return None

def load_db(supabase):
    # Fetch all rows from Supabase 'students' table
    try:
        response = supabase.table('students').select("*").execute()
        data = response.data
        if data:
            return pd.DataFrame(data)
        else:
            return pd.DataFrame(columns=['student_name', 'clicks_total', 'days_active', 'gap_before_deadline'])
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def save_to_db(supabase, name, clicks, active, gap):
    # Insert a new row into Supabase
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

# --- INITIALIZATION ---
supabase = init_supabase()
model = load_model()

if not supabase or not model:
    st.stop()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Data Entry (Cloud)", "Auto-Team Balancer", "Database View"])

# ===================================================
# TAB 1: DATA ENTRY & SAVING
# ===================================================
with tab1:
    st.subheader("Add Student to Cloud DB")
    
    col1, col2 = st.columns(2)
    with col1:
        s_name = st.text_input("Student Name / ID", value="", placeholder="e.g. Student_01")
    with col2:
        clicks = st.number_input("Total Clicks", 0, 1000, 25)
        
    col3, col4 = st.columns(2)
    with col3:
        active_days = st.number_input("Active Days", 0, 30, 3)
    with col4:
        gap = st.number_input("Gap (Days since login)", 0, 60, 5)

    # Predict Live
    input_data = pd.DataFrame([[clicks, active_days, gap]], 
                            columns=['clicks_total', 'days_active', 'gap_before_deadline'])
    prediction = model.predict(input_data)[0]
    st.info(f"Predicted Submission: **{prediction:.2f} Days Early**")

    # Save to Cloud
    if st.button("☁️ Save to Cloud"):
        if s_name == "":
            st.error("Enter a name first.")
        else:
            success = save_to_db(supabase, s_name, clicks, active_days, gap)
            if success:
                st.success(f"Saved **{s_name}** to Supabase!")
                st.rerun()


# ===================================================
# TAB 2: TEAM FORMATION (UPDATED WITH ROSTER)
# ===================================================
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
                classroom['predicted_early'] = model.predict(classroom[features])
                classroom['risk_score'] = classroom['predicted_early'] * -1 # Invert so High = Risky
                
                # 2. Random Logic (Control Group)
                classroom = classroom.sample(frac=1).reset_index(drop=True)
                num_teams = max(1, len(classroom) // 4)
                
                # Random Assignment
                team_ids = np.array_split(range(len(classroom)), num_teams)
                classroom['team_random'] = 0
                for t_id, indices in enumerate(team_ids):
                     classroom.loc[indices, 'team_random'] = t_id + 1
                random_risk = classroom.groupby('team_random')['risk_score'].sum()
                
                # 3. Smart Logic (Snake Draft Algorithm)
                # We need to assign 'team_smart' ID to every student row
                sorted_students = classroom.sort_values('risk_score', ascending=False).reset_index(drop=True)
                smart_risks = [0] * num_teams
                classroom['team_smart'] = 0 # Initialize column
                
                # Create a list to store the new order with team assignments
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
                        
                        # Assign Role based on Risk Score
                        role = "👤 Member"
                        if row['risk_score'] > 2:  # Late submitter
                            role = "⚠️ Risk Factor"
                        elif row['risk_score'] < -2: # Early submitter
                            role = "🛡️ Anchor"
                            
                        # Add to roster list
                        final_roster.append({
                            "Team ID": team_idx + 1,
                            "Student Name": row['student_name'],
                            "Risk Score": round(row['risk_score'], 2),
                            "Predicted Days Late": round(row['predicted_early'] * -1, 1),
                            "Role": role
                        })
                
                # Convert roster to DataFrame for display
                roster_df = pd.DataFrame(final_roster).sort_values(['Team ID', 'Risk Score'])
                
                # 4. Visualization (Graphs)
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("### 🔴 Random Assignment")
                    fig1, ax1 = plt.subplots()
                    ax1.plot(random_risk.values, marker='o', color='red', linestyle='--')
                    ax1.set_ylabel("Total Risk Score")
                    ax1.set_title(f"Variance: {random_risk.var():.2f}")
                    st.pyplot(fig1)
                    
                with col_right:
                    st.markdown("### 🟢 Acadelo Optimization")
                    fig2, ax2 = plt.subplots()
                    ax2.plot(smart_risks, marker='o', color='green')
                    ax2.set_ylabel("Total Risk Score")
                    ax2.set_title(f"Variance: {np.var(smart_risks):.2f}")
                    st.pyplot(fig2)

                # 5. Display The Teams Table
                st.write("---")
                st.subheader("📋 Final Balanced Teams Roster")
                
                # Create a clean display where we group by Team ID
                # Streamlit dataframe with column configuration for better UI
                st.dataframe(
                    roster_df,
                    column_config={
                        "Team ID": st.column_config.NumberColumn("Team #", format="%d"),
                        "Risk Score": st.column_config.ProgressColumn(
                            "Risk Level", 
                            help="Higher score = Higher Procrastination Risk",
                            format="%.2f",
                            min_value=-10,
                            max_value=10,
                        ),
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
# ===================================================
# TAB 3: DATABASE MANAGEMENT
# ===================================================
with tab3:
    st.subheader("Live Database View")
    current_db = load_db(supabase)
    st.dataframe(current_db)
    
    if st.button("🗑️ DELETE ALL RECORDS"):
        # Supabase requires a 'where' clause for delete. 
        # This deletes everything where ID > 0
        supabase.table('students').delete().gt('id', 0).execute()
        st.warning("Database Cleared.")
        st.rerun()