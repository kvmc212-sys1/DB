# =============================================================================
# DRIFTBREAKER V2: AI-NATIVE RISK COMMAND CENTER
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import os
import json
from datetime import datetime

# =============================================================================
# 1. VISUAL CONFIGURATION (THE "SLEEK" LOOK)
# =============================================================================

st.set_page_config(
    page_title="DriftBreaker AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Dark Fintech" Aesthetic
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        background-image: radial-gradient(#1c2230 1px, transparent 1px);
        background-size: 20px 20px;
    }
    
    /* Card Styling */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: #161b26;
        border: 1px solid #262d3d;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #161b26;
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.85rem; }
    div[data-testid="stMetricValue"] { color: #f8fafc; font-size: 1.6rem; font-weight: 600; }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0e1117;
        border-right: 1px solid #262d3d;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        border: none;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 0 15px rgba(37, 99, 235, 0.5);
    }
    
    /* AI Chat Message Styling */
    .user-message {
        background-color: #262d3d;
        padding: 15px;
        border-radius: 12px 12px 0 12px;
        margin: 10px 0;
        text-align: right;
    }
    .ai-message {
        background-color: #1e293b;
        border-left: 3px solid #8b5cf6;
        padding: 15px;
        border-radius: 0 12px 12px 12px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. DATA & UTILS (PRESERVED FROM ORIGINAL)
# =============================================================================

DATA_PATH = "lending_club_loan_two.csv"
DEFAULT_STATUSES = ['charged off', 'default', 'late (31-120 days)']
GOOD_STATUSES = ['fully paid', 'current', 'in grace period']

@st.cache_data
def load_and_prep_data():
    """Load and prepare data for the dashboard"""
    # Try multiple possible file paths
    possible_paths = [
        DATA_PATH,
        "loan.csv",
        "data/loan.csv",
        "lending_club_loan_two.csv"
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                st.sidebar.success(f"Loaded: {path}")
                break
            except Exception as e:
                continue
    
    # If no CSV found, create demo data
    if df is None:
        st.sidebar.warning("Using demo data - no CSV file found")
        dates = pd.date_range(start='2018-01-01', periods=1000, freq='D')
        df = pd.DataFrame({
            'loan_amnt_numeric': np.random.randint(5000, 35000, 1000),
            'default': np.random.choice([0, 1], 1000, p=[0.9, 0.1]),
            'issue_year': dates.year,
            'risk_segment': np.random.choice(['A', 'B', 'C', 'D'], 1000),
            'vintage': dates.to_period('Y').astype(str)
        })
        return df
    
    # Standardize column names and prepare data
    # Map common column name variations
    column_mapping = {
        'loan_amnt': 'loan_amnt_numeric',
        'loan_amount': 'loan_amnt_numeric',
        'funded_amnt': 'loan_amnt_numeric',
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Ensure required columns exist
    if 'loan_amnt_numeric' not in df.columns:
        if 'loan_amnt' in df.columns:
            df['loan_amnt_numeric'] = pd.to_numeric(df['loan_amnt'], errors='coerce').fillna(0)
        else:
            df['loan_amnt_numeric'] = 10000  # Default
    
    if 'default' not in df.columns:
        # Try to infer from loan_status
        if 'loan_status' in df.columns:
            df['default'] = df['loan_status'].isin(DEFAULT_STATUSES).astype(int)
        elif 'post_event_loan_status' in df.columns:
            df['default'] = (df['post_event_loan_status'] == 'Charged Off').astype(int)
        else:
            df['default'] = 0
    
    # Ensure risk_segment exists
    if 'risk_segment' not in df.columns:
        if 'grade' in df.columns:
            df['risk_segment'] = df['grade'].str[0] if df['grade'].dtype == 'object' else 'A'
        else:
            df['risk_segment'] = np.random.choice(['A', 'B', 'C', 'D'], len(df))
    
    # Ensure vintage exists
    if 'vintage' not in df.columns:
        if 'issue_d' in df.columns:
            try:
                issue_dates = pd.to_datetime(df['issue_d'], errors='coerce')
                df['vintage'] = issue_dates.dt.to_period('Y').astype(str)
                df['issue_year'] = issue_dates.dt.year
            except:
                df['vintage'] = '2018'
                df['issue_year'] = 2018
        elif 'issue_year' in df.columns:
            df['vintage'] = df['issue_year'].astype(str)
        else:
            df['vintage'] = '2018'
            df['issue_year'] = 2018
    
    return df

df = load_and_prep_data()
df_resolved = df.copy()

def apply_dark_theme(fig):
    """Apply dark theme to plotly figures"""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e2e8f0',
        xaxis=dict(gridcolor='#334155', showline=True, linecolor='#334155'),
        yaxis=dict(gridcolor='#334155', showline=True, linecolor='#334155'),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified"
    )
    return fig

# =============================================================================
# 3. SIDEBAR & NAVIGATION
# =============================================================================

with st.sidebar:
    st.markdown("### 🛡️ **DriftBreaker**")
    st.caption("AI-Driven Credit Risk Supervision")
    
    st.markdown("---")
    
    # Navigation with Icons
    nav = st.radio("Navigation", [
        "📊 Dashboard Main",
        "🧠 AI Risk Analyst", 
        "⚙️ Configuration"
    ], index=0)
    
    st.markdown("---")
    
    # Global Filters (persist across tabs)
    st.markdown("#### **Global Filters**")
    selected_segments = st.multiselect(
        "Risk Segments", 
        options=sorted(df['risk_segment'].unique()),
        default=sorted(df['risk_segment'].unique())
    )
    
    unemployment_shock = st.slider("Macro Stress (Unemployment)", 0.03, 0.15, 0.04, 0.01, format="%.0%")

# Apply Filters
mask = df['risk_segment'].isin(selected_segments)
df_f = df[mask].copy()

# =============================================================================
# 4. DASHBOARD LOGIC (SIMPLIFIED FOR VISUALS)
# =============================================================================

if nav == "📊 Dashboard Main":
    
    # -- Header KPIs --
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate Metrics
    exposure = df_f['loan_amnt_numeric'].sum()
    default_rate = df_f['default'].mean()
    stressed_pd = default_rate * (1 + (unemployment_shock - 0.04)*4) # Simple stress formula
    
    col1.metric("Total Exposure", f"${exposure/1e6:.1f}M", delta="1.2% vs last month")
    col2.metric("Portfolio PD", f"{default_rate:.2%}", delta="-0.05%", delta_color="inverse")
    col3.metric("Stressed PD", f"{stressed_pd:.2%}", delta="Macro Impact")
    col4.metric("Active Loans", f"{len(df_f):,}")
    
    st.markdown("---")
    
    # -- Row 1: Visuals --
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Default Evolution by Vintage")
        # Aggregation
        vin_data = df_f.groupby('vintage')['default'].mean().reset_index()
        vin_data.columns = ['vintage', 'default_rate']
        fig = px.area(vin_data, x='vintage', y='default_rate', 
                      color_discrete_sequence=['#3b82f6'],
                      labels={'default_rate': 'Default Rate', 'vintage': 'Vintage'})
        fig.add_hline(y=default_rate, line_dash="dash", line_color="gray", 
                     annotation_text=f"Avg ({default_rate:.2%})")
        st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
        
    with c2:
        st.subheader("Risk Mix")
        mix_data = df_f['risk_segment'].value_counts().reset_index()
        mix_data.columns = ['risk_segment', 'count']
        fig2 = px.pie(mix_data, values='count', names='risk_segment', hole=0.6,
                      color_discrete_sequence=px.colors.qualitative.Pastel1)
        fig2.update_layout(showlegend=False)
        st.plotly_chart(apply_dark_theme(fig2), use_container_width=True)
    
    # -- Row 2: Additional Visuals --
    st.markdown("---")
    c3, c4 = st.columns(2)
    
    with c3:
        st.subheader("Exposure by Segment")
        exp_data = df_f.groupby('risk_segment')['loan_amnt_numeric'].sum().reset_index()
        exp_data.columns = ['risk_segment', 'exposure']
        fig3 = px.bar(exp_data, x='risk_segment', y='exposure',
                      color='risk_segment',
                      labels={'exposure': 'Exposure ($)', 'risk_segment': 'Risk Segment'},
                      color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(apply_dark_theme(fig3), use_container_width=True)
    
    with c4:
        st.subheader("Default Rate by Segment")
        def_data = df_f.groupby('risk_segment')['default'].mean().reset_index()
        def_data.columns = ['risk_segment', 'default_rate']
        fig4 = px.bar(def_data, x='risk_segment', y='default_rate',
                      color='risk_segment',
                      labels={'default_rate': 'Default Rate', 'risk_segment': 'Risk Segment'},
                      color_discrete_sequence=px.colors.sequential.Reds_r)
        fig4.update_layout(yaxis_tickformat='.2%')
        st.plotly_chart(apply_dark_theme(fig4), use_container_width=True)

# =============================================================================
# 5. THE AI RISK ANALYST (GEMINI INTEGRATION)
# =============================================================================

elif nav == "🧠 AI Risk Analyst":
    st.title("🧠 Gemini Risk Analyst")
    
    # 1. API Configuration
    # Check for API key in secrets or environment
    api_key = None
    if 'GEMINI_API_KEY' in os.environ:
        api_key = os.environ['GEMINI_API_KEY']
        st.success("✓ API key found in environment")
    elif 'gemini_api_key' in st.secrets:
        api_key = st.secrets['gemini_api_key']
        st.success("✓ API key found in Streamlit secrets")
    else:
        api_key = st.text_input("Enter Gemini API Key (or set in env vars)", type="password")
    
    # 2. Context Engine
    # We serialize the current dashboard state into a context dictionary
    context_data = {
        "summary": {
            "total_exposure": float(exposure),
            "base_pd": float(default_rate),
            "stressed_pd": float(stressed_pd),
            "stress_scenario_ue": float(unemployment_shock),
            "loan_count": int(len(df_f))
        },
        "segment_breakdown": df_f.groupby('risk_segment')['default'].mean().to_dict(),
        "vintage_trend": df_f.groupby('vintage')['default'].mean().tail(5).to_dict() # Last 5 periods
    }
    
    # 3. UI Layout for AI
    col_chat, col_context = st.columns([2, 1])
    
    with col_context:
        st.markdown("### **Live Context**")
        st.info("This data is injected into Gemini automatically.")
        st.json(context_data)
    
    with col_chat:
        st.markdown("### **Analysis Stream**")
        
        # Pre-canned prompts for quick interaction
        q_options = [
            "Summarize the key risks in this portfolio.",
            "Explain the difference between Base PD and Stressed PD.",
            "Draft a memo to the CRO about the recent vintage performance."
        ]
        selected_prompt = st.selectbox("Quick Prompts:", [""] + q_options)
        
        user_input = st.text_area("Or ask a specific question:", value=selected_prompt, height=100)
        
        if st.button("Generate Analysis ✨"):
            if not api_key:
                st.error("Please provide a Gemini API Key. You can set it as an environment variable GEMINI_API_KEY or in Streamlit secrets.")
            else:
                try:
                    # -- THE GEMINI CALL --
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-1.5-pro')
                    
                    # Construct the System Prompt
                    system_prompt = f"""
                    You are the Chief Risk Officer's AI assistant. 
                    You are analyzing a credit portfolio.
                    
                    CURRENT DATA CONTEXT:
                    {json.dumps(context_data, indent=2)}
                    
                    USER QUESTION:
                    {user_input}
                    
                    INSTRUCTIONS:
                    1. Be concise, professional, and direct (Wall Street tone).
                    2. Cite specific numbers from the context.
                    3. If the Stressed PD is > 20% higher than Base PD, flag it as a "Material Risk".
                    4. Provide actionable insights where possible.
                    """
                    
                    with st.spinner("Consulting Gemini..."):
                        response = model.generate_content(system_prompt)
                        
                        # Display
                        st.markdown("---")
                        st.markdown(f'<div class="ai-message">{response.text}</div>', unsafe_allow_html=True)
                        
                        # Option to copy response
                        st.code(response.text, language=None)
                        
                except Exception as e:
                    st.error(f"AI Error: {str(e)}")
                    st.info("Make sure you have installed: pip install google-generativeai")

elif nav == "⚙️ Configuration":
    st.title("System Configuration")
    
    st.markdown("### Data Configuration")
    st.text(f"Current data file: {DATA_PATH}")
    st.text(f"Total records loaded: {len(df):,}")
    st.text(f"Date columns available: {', '.join(df.columns[df.dtypes == 'object'].tolist()[:5])}")
    
    st.markdown("---")
    st.markdown("### Column Mapping")
    with st.expander("View Data Schema"):
        st.dataframe(df.dtypes.reset_index().rename(columns={0: 'dtype', 'index': 'column'}))
    
    st.markdown("---")
    st.markdown("### Macro Scenarios")
    st.slider("Default Unemployment Shock", 0.03, 0.15, 0.04, 0.01, key="config_ue")
    st.text("Configure additional stress scenarios here...")
    
    st.markdown("---")
    st.markdown("### Model Thresholds")
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Low Risk Threshold", 0.0, 1.0, 0.05, 0.01, key="low_thresh")
        st.number_input("Medium Risk Threshold", 0.0, 1.0, 0.15, 0.01, key="med_thresh")
    with col2:
        st.number_input("High Risk Threshold", 0.0, 1.0, 0.30, 0.01, key="high_thresh")
        st.number_input("Maximum Acceptable PD", 0.0, 1.0, 0.50, 0.01, key="max_pd")
