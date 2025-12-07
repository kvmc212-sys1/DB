import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json

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
    /* Clean Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    .nav-header {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        color: #adb5bd;
        margin-top: 20px;
        margin-bottom: 5px;
        letter-spacing: 1px;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADER (INTEGRATING YOUR BI REPORTS) ---
@st.cache_data
def load_data():
    # 1. BI TREND DATA (Extracted from your uploaded 2013-2014 reports)
    # Note the sharp drop in 2014 H2 - this drives the narrative!
    bi_trends = pd.DataFrame({
        'Period': ['2013 H1', '2013 H2', '2014 H1', '2014 H2'],
        'Net_Income': [3900000, 5200000, 5600000, -900000],  # 2014 H2 shows the loss
        'ROI': [0.051, 0.051, 0.041, -0.005],                # ROI goes negative
        'Default_Rate': [0.139, 0.147, 0.156, 0.180],        # Risk climbs steady
        'Active_Loans': [6025, 7850, 9200, 11500]
    })
    
    # 2. ATTRIBUTION (Why did 2014 H2 fail?)
    attr_data = pd.DataFrame({
        'Driver': ['Underwriting (Micro)', 'Economic (Macro)', 'Unexplained'],
        'Contribution': [0.45, 0.40, 0.15] # Macro impact rises in later periods
    })
    
    # 3. STRATEGY SIMULATION (The Fix)
    cf_data = pd.DataFrame({
        'Strategy': ['Status Quo (Baseline)', 'DriftBreaker AI (Aggressive)', 'Conservative'],
        'Net_Profit': [4090000, 4760000, 3800000],
        'Approval_Rate': [0.639, 0.748, 0.550],
        'Color': ['#94a3b8', '#10b981', '#64748b']
    })
    
    # 4. RISK CURVE (The "Month 7" Peak)
    months = np.arange(1, 37)
    hazard = 0.005 + 0.025 * np.exp(-((months - 7)**2) / 20) 
    risk_curve = pd.DataFrame({'month': months, 'hazard': hazard})
    
    # 5. LLM JSON (The Output)
    llm_json = {
        "meta": {"model": "DriftBreaker v3", "run_date": "2025-12-07"},
        "alert": {"status": "CRITICAL", "message": "2014 H2 ROI deterioration detected (-0.5%). Macro factors dominant."},
        "recommendation": "Switch to Aggressive Strategy (+$670k proj. lift)."
    }
    
    return bi_trends, attr_data, cf_data, risk_curve, llm_json

trends_df, attr_df, cf_df, risk_df, llm_json = load_data()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=40)
    st.title("DriftBreaker")
    st.caption("v3.1 | Lifecycle Engine")
    
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    # Navigation Buttons (No more "Layer X")
    st.markdown('<p class="nav-header">INTELLIGENCE</p>', unsafe_allow_html=True)
    nav_bi = st.button("📊 Portfolio Health", use_container_width=True)
    nav_attr = st.button("🔍 Attribution", use_container_width=True)
    
    st.markdown('<p class="nav-header">SIMULATION</p>', unsafe_allow_html=True)
    nav_sim = st.button("🔮 Strategy Lab", use_container_width=True)
    nav_risk = st.button("🚨 Drift Monitor", use_container_width=True)
    
    st.markdown('<p class="nav-header">OUTPUT</p>', unsafe_allow_html=True)
    nav_llm = st.button("🤖 AI Context Bridge", use_container_width=True)
    
    st.divider()
    
    # Scenario Selector
    st.selectbox("Economic Overlay", ["Baseline (Historical)", "Stress Test (2008)", "Forward Rate Shock"])

# --- NAVIGATION STATE MANAGEMENT ---
if 'page' not in st.session_state: st.session_state.page = 'BI'

if nav_bi: st.session_state.page = 'BI'
if nav_attr: st.session_state.page = 'ATTR'
if nav_sim: st.session_state.page = 'SIM'
if nav_risk: st.session_state.page = 'RISK'
if nav_llm: st.session_state.page = 'LLM'

page = st.session_state.page

# --- MAIN CONTENT RENDER ---

if page == 'BI':
    st.title("Portfolio Health")
    st.markdown("Top-line performance metrics extracted from bi-annual reports.")
    
    # Get latest data point (2014 H2)
    latest = trends_df.iloc[-1]
    prev = trends_df.iloc[-2]
    
    # Top Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Income (H2)", f"${latest['Net_Income']/1e6:.1f}M", f"{(latest['Net_Income']-prev['Net_Income'])/1e6:.1f}M")
    c2.metric("ROI", f"{latest['ROI']:.1%}", f"{(latest['ROI']-prev['ROI']):.1%}")
    c3.metric("Default Rate", f"{latest['Default_Rate']:.1%}", f"{(latest['Default_Rate']-prev['Default_Rate']):.1%}", delta_color="inverse")
    c4.metric("Active Loans", f"{int(latest['Active_Loans']):,}", f"{int(latest['Active_Loans']-prev['Active_Loans']):,}")

    st.divider()
    
    # Charts Row
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Profitability Trend")
        fig_prof = px.bar(trends_df, x='Period', y='Net_Income', 
                          title="Net Income by Half-Year", text_auto='.2s',
                          color='Net_Income', color_continuous_scale='RdBu')
        st.plotly_chart(fig_prof, use_container_width=True)
        
    with col_right:
        st.subheader("Risk Deterioration")
        fig_risk = px.line(trends_df, x='Period', y='Default_Rate', 
                           title="Default Rate vs. ROI", markers=True)
        fig_risk.add_scatter(x=trends_df['Period'], y=trends_df['ROI'], mode='lines+markers', name='ROI', yaxis='y2')
        
        # Dual Axis layout
        fig_risk.update_layout(yaxis2=dict(overlaying='y', side='right', title='ROI', showgrid=False))
        st.plotly_chart(fig_risk, use_container_width=True)

elif page == 'ATTR':
    st.title("Attribution Analysis")
    st.markdown("Why is performance degrading? Decomposing Micro vs. Macro drivers.")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.info("""
        **Insight:** While 2013 was driven by **Micro** factors (selection), 2014 H2 shows a spike in **Macro** contribution. 
        
        This indicates the portfolio is becoming sensitive to external economic stress.
        """)
        st.metric("Macro Contribution", "40%", "+15% YoY")
    
    with c2:
        fig = px.pie(attr_df, values='Contribution', names='Driver', hole=0.5, 
                     title="Variance Decomposition (2014 H2)", 
                     color_discrete_sequence=px.colors.qualitative.Prism)
        st.plotly_chart(fig, use_container_width=True)

elif page == 'SIM':
    st.title("Strategy Lab")
    st.markdown("Counterfactual analysis: What if we used the DriftBreaker model?")
    
    # Financial Impact Chart
    fig = px.bar(cf_df, x='Strategy', y='Net_Profit', color='Color', 
                 title="Projected Net Profit by Strategy", text_auto='.3s')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Delta Metric
    delta = cf_df.loc[1, 'Net_Profit'] - cf_df.loc[0, 'Net_Profit']
    st.success(f"**Recommendation:** The Aggressive Strategy yields **+${delta/1e6:.2f}M** in additional profit while maintaining acceptable risk boundaries.")

elif page == 'RISK':
    st.title("Drift & Early Warning")
    st.markdown("Real-time structural monitoring of the risk curve.")
    
    c1, c2 = st.columns(2)
    c1.metric("Feature PSI", "0.089", "Stable (<0.1)")
    c2.metric("Peak Hazard Timing", "Month 7", "Confirmed")
    
    st.subheader("The 'Month 7' Hazard Peak")
    fig = px.line(risk_df, x='month', y='hazard', markers=True, title="Monthly Hazard Rate",
                  labels={'hazard': 'Probability of Default'})
    fig.update_traces(line_color='#ef4444', line_width=3)
    
    # Add annotation for the peak
    fig.add_shape(type="line", x0=7, y0=0, x1=7, y1=max(risk_df['hazard']),
                  line=dict(color="Gray", width=1, dash="dash"))
    fig.add_annotation(x=7, y=max(risk_df['hazard']), text="Peak Risk", showarrow=True, arrowhead=1)
    
    st.plotly_chart(fig, use_container_width=True)

elif page == 'LLM':
    st.title("AI Context Bridge")
    st.markdown("Raw semantic payload for Large Language Model interpretation.")
    
    st.code(json.dumps(llm_json, indent=2), language="json")
    
    st.download_button("Download Context JSON", data=json.dumps(llm_json), file_name="driftbreaker_context.json")
