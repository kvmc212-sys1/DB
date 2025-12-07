import streamlit as st

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

import numpy as np

import json



# --- PAGE CONFIGURATION ---

st.set_page_config(

    page_title="DriftBreaker v3",

    page_icon="🛡️",

    layout="wide",

    initial_sidebar_state="expanded"

)



# --- CSS FOR "EXPENSIVE" LOOK ---

st.markdown("""

<style>

    .stMetric {background-color: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #e9ecef;}

    div[data-testid="stSidebarNav"] {display: none;} /* Hide default nav */

    .layer-header {font-size: 12px; font-weight: bold; color: #6c757d; margin-top: 10px;}

</style>

""", unsafe_allow_html=True)



# --- MOCK DATA GENERATOR ---

@st.cache_data

def load_mock_data():

    # Layer 1: BI Data

    bi_data = pd.DataFrame({

        'Metric': ['Revenue', 'Losses', 'Net Income', 'ROI'],

        'Value': [12.5, 4.2, 8.3, 0.18], # Millions or %

        'Delta': [1.2, -0.5, 1.7, 0.02]

    })

    

    # Layer 2: Attribution Data (Micro vs Macro)

    attr_data = pd.DataFrame({

        'Driver': ['Underwriting (Micro)', 'Economic (Macro)', 'Unexplained'],

        'Contribution': [0.65, 0.25, 0.10]

    })

    

    # Layer 3: Counterfactuals (The "What If")

    strategies = ['Status Quo', 'DriftBreaker (Aggressive)', 'Conservative']

    cf_data = pd.DataFrame({

        'Strategy': strategies,

        'Net_Profit': [4.1, 4.76, 3.8],

        'Approval_Rate': [0.64, 0.748, 0.55]

    })

    

    # Layer 4: Drift Data (Risk Curve)

    months = np.arange(1, 37)

    hazard = 0.005 + 0.025 * np.exp(-((months - 7)**2) / 20) 

    risk_curve = pd.DataFrame({'month': months, 'hazard': hazard})

    

    # Layer 5: LLM Context (The JSON)

    llm_context = {

        "meta": {"framework": "DriftBreaker v3", "generated_at": "2025-10-27T14:30:00Z"},

        "executive_summary": {

            "status": "ACTION_REQUIRED",

            "message": "Aggressive strategy yields +$660k. Macro sensitivity stable."

        },

        "attribution": {"primary_driver": "MICRO", "micro_contribution": 0.65},

        "warnings": [{"type": "DRIFT", "severity": "LOW", "feature": "income"}]

    }

    

    return bi_data, attr_data, cf_data, risk_curve, llm_context



bi_df, attr_df, cf_df, risk_df, llm_json = load_mock_data()



# --- SIDEBAR NAVIGATION (THE "LAYERS") ---

with st.sidebar:

    st.title("🛡️ DriftBreaker v3")

    st.caption("Complete Risk Framework")

    st.divider()

    

    # The Vertical Stack Layout

    st.markdown('<p class="layer-header">LAYER 1: BUSINESS INTELLIGENCE</p>', unsafe_allow_html=True)

    l1 = st.button("📊 Portfolio Health", use_container_width=True)

    

    st.markdown('<p class="layer-header">LAYER 2: ATTRIBUTION</p>', unsafe_allow_html=True)

    l2 = st.button("🔍 Micro vs Macro", use_container_width=True)

    

    st.markdown('<p class="layer-header">LAYER 3: COUNTERFACTUALS</p>', unsafe_allow_html=True)

    l3 = st.button("🔮 Strategy Simulation", use_container_width=True)

    

    st.markdown('<p class="layer-header">LAYER 4: DRIFT WARNINGS</p>', unsafe_allow_html=True)

    l4 = st.button("🚨 Early Warning System", use_container_width=True)

    

    st.markdown('<p class="layer-header">LAYER 5: SEMANTIC BRIDGE</p>', unsafe_allow_html=True)

    l5 = st.button("🤖 LLM Context Output", use_container_width=True)

    
    st.divider()

    st.caption("System Status: ● Online")



# --- MAIN CONTENT LOGIC ---

# Default to Layer 1 if nothing clicked, or handle state

if 'active_layer' not in st.session_state:

    st.session_state.active_layer = 'L1'



if l1: st.session_state.active_layer = 'L1'

if l2: st.session_state.active_layer = 'L2'

if l3: st.session_state.active_layer = 'L3'

if l4: st.session_state.active_layer = 'L4'

if l5: st.session_state.active_layer = 'L5'



layer = st.session_state.active_layer



# --- RENDER LAYERS ---



if layer == 'L1':

    st.header("Layer 1: Business Intelligence")

    st.markdown("#### *WHAT happened?*")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Net Income", "$8.3M", "+$1.7M")

    col2.metric("ROI", "18.2%", "+0.2%")

    col3.metric("Default Rate (PD12)", "4.2%", "-0.5%")

    col4.metric("Active Loans", "12,450", "+8%")

    

    st.divider()

    st.subheader("Portfolio Composition")

    # Mock Chart

    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["A", "B", "C"])

    st.line_chart(chart_data)



elif layer == 'L2':

    st.header("Layer 2: Attribution Analysis")

    st.markdown("#### *WHY did it happen?*")

    

    c1, c2 = st.columns([1, 2])

    with c1:

        st.info("**Primary Driver: MICRO**\n\nDefaults are currently driven by underwriting selection, not economic headwinds.")

    with c2:

        fig = px.pie(attr_df, values='Contribution', names='Driver', hole=0.4, 

                     title="Variance Decomposition", color_discrete_sequence=px.colors.sequential.RdBu)

        st.plotly_chart(fig, use_container_width=True)



elif layer == 'L3':

    st.header("Layer 3: Counterfactual Analysis")

    st.markdown("#### *WHAT IF we acted differently?*")

    

    st.plotly_chart(px.bar(cf_df, x='Strategy', y='Net_Profit', color='Strategy', 

                           title="Net Profit by Strategy (Projected)", text_auto=True), use_container_width=True)

    

    st.success("Recommendation: The **DriftBreaker (Aggressive)** strategy captures an additional **$660k** in value vs Status Quo.")



elif layer == 'L4':

    st.header("Layer 4: Drift & Early Warning")

    st.markdown("#### *WHAT is coming?*")

    

    c1, c2 = st.columns(2)

    c1.metric("Feature PSI", "0.089", "Stable")

    c2.metric("Concept Drift", "Month 7 Peak", "Confirmed")

    

    st.subheader("The 'Month 7' Hazard Peak")

    fig = px.line(risk_df, x='month', y='hazard', markers=True, title="Monthly Hazard Rate",

                  labels={'hazard': 'Probability of Default'})

    fig.add_vline(x=7, line_dash="dash", line_color="red", annotation_text="Structural Peak")

    st.plotly_chart(fig, use_container_width=True)



elif layer == 'L5':

    st.header("Layer 5: LLM-Ready Output")

    st.markdown("#### *How do we explain it?*")

    st.markdown("This is the exact JSON payload sent to the LLM to generate the Executive Summary.")

    

    st.json(llm_json)

    

    st.download_button("Download JSON Payload", data=json.dumps(llm_json), file_name="llm_context.json")

