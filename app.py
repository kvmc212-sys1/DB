import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import google.generativeai as genai

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
    .stApp {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #1e293b;
    }
    
    /* Clean Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    .nav-header {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        color: #94a3b8;
        margin-top: 20px;
        margin-bottom: 5px;
        letter-spacing: 1px;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #64748b;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #0f172a;
    }
    
    /* Buttons */
    div.stButton > button:first-child {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        color: #334155;
        font-weight: 500;
    }
    div.stButton > button:hover {
        border-color: #10b981;
        color: #10b981;
    }
</style>
""", unsafe_allow_html=True)

# --- GEMINI AI INTEGRATION ---
def get_gemini_summary(context_json):
    """
    Sends Layer 5 context to Gemini and gets a risk assessment.
    """
    # Check for secrets
    if "GEMINI_API_KEY" not in st.secrets:
        return "⚠️ API Key missing. Please set GEMINI_API_KEY in .streamlit/secrets.toml."
    
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""
        You are the Chief Risk Officer. Analyze this JSON risk context and write a 
        concise 3-sentence executive summary.
        
        Focus on:
        1. The specific deterioration in 2014 H2 (ROI/Net Income).
        2. The shift from Micro to Macro drivers.
        3. A recommendation on the Aggressive Strategy.
        
        Context: {json.dumps(context_json)}
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error connecting to AI: {str(e)}"

# --- DATA LOADER (MOCK & BI DATA) ---
@st.cache_data
def load_data():
    # 1. BI TREND DATA (The 2013-2014 Crash)
    bi_trends = pd.DataFrame({
        'Period': ['2013 H1', '2013 H2', '2014 H1', '2014 H2'],
        'Net_Income': [3900000, 5200000, 5600000, -900000],  # The Crash
        'ROI': [0.051, 0.051, 0.041, -0.005],                # Negative ROI
        'Default_Rate': [0.139, 0.147, 0.156, 0.180],        # Rising Risk
        'Active_Loans': [6025, 7850, 9200, 11500]
    })
    
    # 2. ATTRIBUTION (The "Why")
    attr_data = pd.DataFrame({
        'Driver': ['Underwriting (Micro)', 'Economic (Macro)', 'Unexplained'],
        'Contribution': [0.45, 0.40, 0.15] # Macro rising significantly
    })
    
    # 3. STRATEGY SIMULATION (The "What If")
    cf_data = pd.DataFrame({
        'Strategy': ['Status Quo (Baseline)', 'DriftBreaker AI (Aggressive)', 'Conservative'],
        'Net_Profit': [4090000, 4760000, 3800000],
        'Approval_Rate': [0.639, 0.748, 0.550],
        'Color': ['#94a3b8', '#10b981', '#64748b']
    })
    
    # 4. RISK CURVE (The "Month 7" Peak)
    months = np.arange(1, 37)
    # Formula creates a peak at x=7
    hazard = 0.005 + 0.025 * np.exp(-((months - 7)**2) / 20) 
    risk_curve = pd.DataFrame({'month': months, 'hazard': hazard})
    
    # 5. LLM JSON (Layer 5 Context)
    llm_json = {
        "meta": {"model": "DriftBreaker v3", "run_date": "2025-12-07"},
        "alert": {
            "status": "CRITICAL", 
            "message": "2014 H2 ROI deterioration detected (-0.5%). Macro factors dominant."
        },
        "attribution": {"micro": 0.45, "macro": 0.40},
        "financials": {"current_roi": -0.005, "proj_profit_lift": 670000},
        "recommendation": "Switch to Aggressive Strategy to capture $4.76M vs $4.09M baseline."
    }
    
    return bi_trends, attr_data, cf_data, risk_curve, llm_json

trends_df, attr_df, cf_df, risk_df, llm_json = load_data()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=40)
    st.title("DriftBreaker")
    st.caption("v3.1 | Lifecycle Engine")
    
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    # Navigation Groups
    st.markdown('<p class="nav-header">INTELLIGENCE</p>', unsafe_allow_html=True)
    nav_bi = st.button("📊 Portfolio Health", use_container_width=True)
    nav_attr = st.button("🔍 Attribution", use_container_width=True)
    
    st.markdown('<p class="nav-header">SIMULATION</p>', unsafe_allow_html=True)
    nav_sim = st.button("🔮 Strategy Lab", use_container_width=True)
    nav_risk = st.button("🚨 Drift Monitor", use_container_width=True)
    
    st.markdown('<p class="nav-header">OUTPUT & API</p>', unsafe_allow_html=True)
    nav_llm = st.button("🤖 AI Context Bridge", use_container_width=True)
    nav_api = st.button("⚡ Loan Simulator", use_container_width=True)
    
    st.divider()
    
    # Scenario Selector
    st.selectbox("Economic Overlay", ["Baseline (Historical)", "Stress Test (2008)", "Forward Rate Shock"])
    st.caption("System Status: ● Online")

# --- NAVIGATION STATE ---
if 'page' not in st.session_state: st.session_state.page = 'BI'

if nav_bi: st.session_state.page = 'BI'
if nav_attr: st.session_state.page = 'ATTR'
if nav_sim: st.session_state.page = 'SIM'
if nav_risk: st.session_state.page = 'RISK'
if nav_llm: st.session_state.page = 'LLM'
if nav_api: st.session_state.page = 'API'

page = st.session_state.page

# --- MAIN CONTENT ---

if page == 'BI':
    st.title("Portfolio Health")
    st.markdown("### Layer 1: Business Intelligence (What happened?)")
    
    # Get latest data point (2014 H2)
    latest = trends_df.iloc[-1]
    prev = trends_df.iloc[-2]
    
    # Top Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Income (H2)", f"${latest['Net_Income']/1e6:.1f}M", f"{(latest['Net_Income']-prev['Net_Income'])/1e6:.1f}M", delta_color="inverse")
    c2.metric("ROI", f"{latest['ROI']:.1%}", f"{(latest['ROI']-prev['ROI']):.1%}", delta_color="inverse")
    c3.metric("Default Rate", f"{latest['Default_Rate']:.1%}", f"{(latest['Default_Rate']-prev['Default_Rate']):.1%}", delta_color="inverse")
    c4.metric("Active Loans", f"{int(latest['Active_Loans']):,}", f"{int(latest['Active_Loans']-prev['Active_Loans']):,}")

    st.divider()
    
    # Charts Row
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Profitability Collapse")
        fig_prof = px.bar(trends_df, x='Period', y='Net_Income', 
                          title="Net Income by Half-Year", text_auto='.2s',
                          color='Net_Income', color_continuous_scale='RdBu')
        fig_prof.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_prof, use_container_width=True)
        
    with col_right:
        st.subheader("Risk Deterioration")
        fig_risk = px.line(trends_df, x='Period', y='Default_Rate', 
                           title="Default Rate vs. ROI Trend", markers=True)
        fig_risk.add_scatter(x=trends_df['Period'], y=trends_df['ROI'], mode='lines+markers', name='ROI', yaxis='y2', line=dict(color='orange'))
        
        # Dual Axis layout
        fig_risk.update_layout(
            yaxis2=dict(overlaying='y', side='right', title='ROI', showgrid=False),
            plot_bgcolor="white",
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9')
        )
        st.plotly_chart(fig_risk, use_container_width=True)

elif page == 'ATTR':
    st.title("Attribution Analysis")
    st.markdown("### Layer 2: Variance Decomposition (Why did it happen?)")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.info("""
        **Executive Insight:** While 2013 performance was driven by **Micro** factors (selection), 2014 H2 shows a spike in **Macro** contribution. 
        
        This indicates the portfolio has crossed a "fragility threshold" regarding economic stress.
        """)
        st.metric("Macro Contribution", "40%", "+15% YoY")
        st.metric("Micro Contribution", "45%", "-5% YoY")
    
    with c2:
        fig = px.pie(attr_df, values='Contribution', names='Driver', hole=0.5, 
                     title="Variance Decomposition (2014 H2)", 
                     color_discrete_sequence=px.colors.qualitative.Prism)
        st.plotly_chart(fig, use_container_width=True)

elif page == 'SIM':
    st.title("Strategy Lab")
    st.markdown("### Layer 3: Counterfactuals (What If?)")
    
    # Financial Impact Chart
    fig = px.bar(cf_df, x='Strategy', y='Net_Profit', color='Color', 
                 title="Projected Net Profit by Strategy", text_auto='.3s')
    fig.update_layout(showlegend=False, plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
    
    # Delta Metric
    delta = cf_df.loc[1, 'Net_Profit'] - cf_df.loc[0, 'Net_Profit']
    
    c1, c2 = st.columns(2)
    c1.success(f"**Recommendation:** The Aggressive Strategy yields **+${delta/1e6:.2f}M** in additional profit.")
    c2.warning("**Constraint:** Requires acceptance of 11% higher approval volume (Operational Impact).")

elif page == 'RISK':
    st.title("Drift & Early Warning")
    st.markdown("### Layer 4: Structural Drift (What's Coming?)")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Feature PSI", "0.089", "Stable (<0.1)")
    c2.metric("Peak Hazard Timing", "Month 7", "Confirmed")
    c3.metric("Concept Drift", "Detected", "Recalibrate")
    
    st.subheader("The 'Month 7' Hazard Peak")
    fig = px.line(risk_df, x='month', y='hazard', markers=True, title="Monthly Hazard Rate (Probability of Default)",
                  labels={'hazard': 'Hazard Rate'})
    fig.update_traces(line_color='#ef4444', line_width=3)
    
    # Add annotation for the peak
    fig.add_vline(x=7, line_dash="dash", line_color="gray")
    fig.add_annotation(x=7, y=max(risk_df['hazard']), text="Peak Risk (Month 7)", showarrow=True, arrowhead=1)
    
    fig.update_layout(plot_bgcolor="white", xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f1f5f9'))
    st.plotly_chart(fig, use_container_width=True)

elif page == 'LLM':
    st.title("AI Context Bridge")
    st.markdown("### Layer 5: Semantic Payload & Interpretation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Raw JSON Payload")
        st.caption("Serialized context sent to LLM.")
        st.code(json.dumps(llm_json, indent=2), language="json")
        st.download_button("Download JSON", data=json.dumps(llm_json), file_name="driftbreaker_context.json")
    
    with col2:
        st.subheader("Gemini Interpretation")
        st.caption("Live response from Google Gemini model.")
        
        if st.button("Generate Executive Summary", type="primary"):
            with st.spinner("Consulting Chief Risk Officer (AI)..."):
                summary = get_gemini_summary(llm_json)
                st.info(summary, icon="🤖")
        else:
            st.info("Click to generate a fresh analysis based on the JSON context.")

elif page == 'API':
    st.title("Live Loan Simulator")
    st.markdown("Test the **DriftBreaker API** in real-time with custom inputs.")
    
    # Input Form
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            inc = st.number_input("Annual Income ($)", value=75000, step=5000)
        with col2:
            dti = st.number_input("DTI (%)", value=15, step=1)
        with col3:
            amt = st.number_input("Loan Amount ($)", value=12000, step=1000)
            
        if st.button("Score Application via API Logic", type="primary", use_container_width=True):
            # SIMULATE THE API LOGIC LOCALLY
            base_pd = 0.05 + (0.08 if dti > 20 else 0) + (0.05 if inc < 60000 else 0)
            decision = "APPROVE" if base_pd < 0.22 else "REJECT"
            
            st.divider()
            
            c1, c2 = st.columns(2)
            c1.metric("Decision", decision, "Aggressive Strategy" if decision=="APPROVE" else "Risk Cutoff")
            c2.metric("Probability of Default", f"{base_pd:.1%}", "Micro Driven" if dti>20 else "Macro Base")
            
            st.subheader("Layer 5 Generated Context")
            st.json({
                "narrative": f"Applicant (Inc: ${inc}, DTI: {dti}%) presents {base_pd:.1%} risk.",
                "attribution": {"micro": base_pd - 0.02, "macro": 0.02},
                "recommendation": "Approve based on DriftBreaker EV model."
            })


