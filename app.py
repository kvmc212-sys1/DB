import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import google.generativeai as genai
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DriftBreaker AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS FOR MODERN "FINTECH" LOOK ---
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp { background-color: #f8f9fa; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; color: #1e293b; }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    .nav-header {
        font-size: 11px; font-weight: 700; text-transform: uppercase; 
        color: #94a3b8; margin-top: 20px; margin-bottom: 5px; letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# --- GEMINI AI INTEGRATION ---
def get_gemini_summary(context_json):
    if "GEMINI_API_KEY" not in st.secrets:
        return "⚠️ API Key missing. Set GEMINI_API_KEY in .streamlit/secrets.toml"
    
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.0-flash') # Or gemini-pro
        
        prompt = f"""
        You are the Chief Risk Officer. Write a 3-sentence executive summary based on this risk context.
        Focus on the 2014 H2 deterioration and the Month 7 Peak signal.
        Context: {json.dumps(context_json)}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Connection Error: {str(e)}"

# --- HYBRID DATA LOADER ---
@st.cache_data
def load_data():
    data_source = "LIVE PIPELINE"
    
    # 1. BI TRENDS (Always Hardcoded from your MD files for the story)
    bi_trends = pd.DataFrame({
        'Period': ['2013 H1', '2013 H2', '2014 H1', '2014 H2'],
        'Net_Income': [3900000, 5200000, 5600000, -900000],
        'ROI': [0.051, 0.051, 0.041, -0.005],
        'Default_Rate': [0.139, 0.147, 0.156, 0.180],
        'Active_Loans': [6025, 7850, 9200, 11500]
    })
    
    # 2. ATTEMP TO LOAD REAL ARTIFACTS
    try:
        risk_curve = pd.read_csv("dashboard_risk_curve.csv")
        finance_df = pd.read_csv("dashboard_finance.csv")
        with open("dashboard_vitals.json", "r") as f:
            vitals = json.load(f)
            
    except Exception as e:
        # FALLBACK TO MOCK GENERATOR (The Safety Net)
        data_source = "DEMO MODE (Synthetic)"
        
        # Mock Risk Curve
        months = np.arange(1, 37)
        hazard = 0.005 + 0.025 * np.exp(-((months - 7)**2) / 20) 
        risk_curve = pd.DataFrame({'month': months, 'hazard': hazard})
        
        # Mock Finance
        finance_df = pd.DataFrame([
            {"Strategy": "DriftBreaker AI", "Scenario": "Baseline (Current)", "Net_Profit": 4.76, "Color": "#10b981"},
            {"Strategy": "DriftBreaker AI", "Scenario": "Mild Recession (+2% UE)", "Net_Profit": 3.95, "Color": "#f59e0b"},
            {"Strategy": "DriftBreaker AI", "Scenario": "Severe Shock (2008 Style)", "Net_Profit": 2.10, "Color": "#ef4444"}
        ])
        
        vitals = {"peak_month": 7, "drift_status": "Stable"}

    return bi_trends, risk_curve, finance_df, vitals, data_source

trends_df, risk_curve, finance_df, vitals, source_status = load_data()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=40)
    st.title("DriftBreaker")
    st.caption(f"v3.4 | {source_status}")
    
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    st.markdown('<p class="nav-header">INTELLIGENCE</p>', unsafe_allow_html=True)
    nav_bi = st.button("📊 Portfolio Health", use_container_width=True)
    nav_attr = st.button("🔍 Attribution", use_container_width=True)
    
    st.markdown('<p class="nav-header">STRESS TESTING</p>', unsafe_allow_html=True)
    nav_sim = st.button("📉 Strategy Lab", use_container_width=True)
    nav_risk = st.button("🚨 Drift Monitor", use_container_width=True)
    
    st.markdown('<p class="nav-header">OUTPUT</p>', unsafe_allow_html=True)
    nav_llm = st.button("🤖 AI Context Bridge", use_container_width=True)
    
    st.divider()
    scenario = st.selectbox("Macro Overlay", ["Baseline (Current)", "Mild Recession", "Severe Shock"])

# --- NAVIGATION STATE ---
if 'page' not in st.session_state: st.session_state.page = 'BI'

if nav_bi: st.session_state.page = 'BI'
if nav_attr: st.session_state.page = 'ATTR'
if nav_sim: st.session_state.page = 'SIM'
if nav_risk: st.session_state.page = 'RISK'
if nav_llm: st.session_state.page = 'LLM'

page = st.session_state.page

# --- MAIN CONTENT ---

if page == 'BI':
    st.title("Portfolio Health")
    st.markdown("### Layer 1: Business Intelligence (Historical)")
    
    latest = trends_df.iloc[-1]
    prev = trends_df.iloc[-2]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Income (H2)", f"${latest['Net_Income']/1e6:.1f}M", f"{(latest['Net_Income']-prev['Net_Income'])/1e6:.1f}M", delta_color="inverse")
    c2.metric("ROI", f"{latest['ROI']:.1%}", f"{(latest['ROI']-prev['ROI']):.1%}", delta_color="inverse")
    c3.metric("Default Rate", f"{latest['Default_Rate']:.1%}", f"{(latest['Default_Rate']-prev['Default_Rate']):.1%}", delta_color="inverse")
    c4.metric("Active Loans", f"{int(latest['Active_Loans']):,}", f"{int(latest['Active_Loans']-prev['Active_Loans']):,}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Profitability Trend")
        fig = px.bar(trends_df, x='Period', y='Net_Income', color='Net_Income', color_continuous_scale='RdBu', title="Net Income Collapse (2014)")
        fig.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Risk Deterioration")
        fig = px.line(trends_df, x='Period', y='Default_Rate', markers=True, title="Rising Defaults")
        fig.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

elif page == 'ATTR':
    st.title("Attribution Analysis")
    st.markdown("### Layer 2: Variance Decomposition")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.info("**Insight:** 2014 H2 volatility is 40% driven by Macro factors, indicating high sensitivity to economic stress.")
    with c2:
        attr_data = pd.DataFrame({'Driver': ['Micro', 'Macro', 'Noise'], 'Contribution': [0.45, 0.40, 0.15]})
        fig = px.pie(attr_data, values='Contribution', names='Driver', hole=0.5, color_discrete_sequence=px.colors.qualitative.Prism)
        st.plotly_chart(fig, use_container_width=True)

elif page == 'SIM':
    st.title("Strategy & Macro Lab")
    st.markdown(f"### Layer 3: Portfolio Stress Test ({scenario})")
    
    # Filter data based on dropdown match
    # Matches "Baseline" to "Baseline", etc.
    subset = finance_df[finance_df['Scenario'].str.contains(scenario.split()[0])]
    
    if not subset.empty:
        profit = subset.iloc[0]['Net_Profit']
        st.metric("Projected Profit (DriftBreaker)", f"${profit}M")
        
        # Compare to Status Quo (Hardcoded baseline for context)
        sq_profit = 4.09 if "Baseline" in scenario else 3.5 if "Mild" in scenario else 1.5
        
        chart_data = pd.DataFrame({
            'Strategy': ['Status Quo', 'DriftBreaker AI'],
            'Profit': [sq_profit, profit],
            'Color': ['#94a3b8', subset.iloc[0]['Color']]
        })
        
        fig = px.bar(chart_data, x='Strategy', y='Profit', color='Color', text_auto=True, color_discrete_map="identity")
        fig.update_layout(plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Save for AI
        st.session_state.llm_ctx = {"scenario": scenario, "ai_profit": profit, "sq_profit": sq_profit}

elif page == 'RISK':
    st.title("Drift & Early Warning")
    st.markdown("### Layer 4: Structural Drift")
    c1, c2 = st.columns(2)
    c1.metric("Peak Hazard Timing", f"Month {vitals['peak_month']}", "Confirmed")
    c2.metric("Drift Status", vitals['drift_status'], "Stable")
    
    fig = px.line(risk_curve, x='month', y='hazard', markers=True, title="Monthly Hazard Rate")
    fig.add_vline(x=vitals['peak_month'], line_dash="dash", line_color="red")
    fig.update_traces(line_color='#ef4444', line_width=3)
    st.plotly_chart(fig, use_container_width=True)

elif page == 'LLM':
    st.title("AI Context Bridge")
    ctx = st.session_state.get('llm_ctx', {"meta": "Run Simulation First"})
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("JSON Payload")
        st.json(ctx)
    with c2:
        st.subheader("Gemini Assessment")
        if st.button("Generate Summary", type="primary"):
            with st.spinner("Analyzing..."):
                st.info(get_gemini_summary(ctx), icon="🤖")
