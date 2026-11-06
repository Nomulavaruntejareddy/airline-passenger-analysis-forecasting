import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import DataLoader
from src.forecast import Forecaster
from src.evaluate import Evaluator

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------

st.set_page_config(
    page_title="Airline Passenger Forecaster",
    page_icon="✈️",
    layout="wide"
)

# ------------------------------------------------
# Color Palette
# ------------------------------------------------
DEEP_TEAL   = "#006078"   # primary dark accent - headings, borders, text
DUSTY_BLUE  = "#82BAC4"   # secondary accent - sidebar, muted elements
BLUSH       = "#FAEFED"   # near-white - page background
PEACH       = "#FFD4D1"   # soft accent - panels, hover states
ROSE_CORAL  = "#E37C78"   # primary CTA / highlight accent
INK         = "#173C46"   # dark teal-gray for body text readability

# ------------------------------------------------
# Custom CSS — Soft Premium Light Aesthetic
# ------------------------------------------------
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif;
    }}

    .mono {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* Overall app background — soft blush to peach wash */
    .stApp {{
        background: linear-gradient(160deg, {BLUSH} 0%, #FFF6F4 45%, {PEACH} 100%);
        color: {INK};
    }}

    /* Headings */
    h1, h2, h3 {{
        font-family: 'Space Grotesk', sans-serif !important;
        color: {DEEP_TEAL} !important;
        letter-spacing: 0.3px;
    }}

    h1 {{
        background: linear-gradient(90deg, {DEEP_TEAL} 0%, {ROSE_CORAL} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }}

    p, span, label, .stCaption, .stMarkdown {{
        color: {INK} !important;
    }}

    /* ---------- Signature: Flight Path header ---------- */
    .flight-header {{
        display: flex;
        align-items: center;
        gap: 18px;
        margin: 4px 0 2px 0;
    }}
    .airport-code {{
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 3px;
        color: {DEEP_TEAL};
        border: 1.5px solid rgba(0, 96, 120, 0.35);
        border-radius: 6px;
        padding: 6px 12px;
        background: rgba(0, 96, 120, 0.06);
        white-space: nowrap;
    }}
    .flight-path {{
        flex: 1;
        display: flex;
        align-items: center;
        min-width: 80px;
    }}
    .flight-path .seg {{
        flex: 1;
        height: 0;
        border-top: 2px dashed rgba(0, 96, 120, 0.3);
    }}
    .flight-path .dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: {DUSTY_BLUE};
        flex-shrink: 0;
    }}
    .flight-path .dot.live {{
        background: {ROSE_CORAL};
        box-shadow: 0 0 0 rgba(227, 124, 120, 0.6);
        animation: pulse-dot 2.2s ease-out infinite;
    }}
    @keyframes pulse-dot {{
        0%   {{ box-shadow: 0 0 0 0 rgba(227, 124, 120, 0.55); }}
        70%  {{ box-shadow: 0 0 0 9px rgba(227, 124, 120, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(227, 124, 120, 0); }}
    }}
    .flight-path .plane {{
        flex-shrink: 0;
        margin: 0 8px;
        font-size: 1.1rem;
        transform: rotate(90deg);
        opacity: 0.9;
    }}
    .eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 3px;
        color: {ROSE_CORAL};
        opacity: 0.9;
        font-weight: 600;
        margin-bottom: 4px;
        text-transform: uppercase;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {DUSTY_BLUE} 0%, {DEEP_TEAL} 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.15);
    }}
    section[data-testid="stSidebar"] * {{
        color: {BLUSH} !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {BLUSH} !important;
        -webkit-text-fill-color: {BLUSH} !important;
        background: none !important;
    }}

    /* Slider accent */
    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background-color: {ROSE_CORAL} !important;
        box-shadow: 0 0 10px rgba(227, 124, 120, 0.6);
    }}
    .stSlider [data-baseweb="slider"] > div > div {{
        background: {ROSE_CORAL} !important;
    }}

    /* Info box */
    .stAlert {{
        background-color: rgba(255, 255, 255, 0.35) !important;
        border-left: 4px solid {ROSE_CORAL} !important;
        border-radius: 10px !important;
        color: {BLUSH} !important;
    }}

    /* ---------- Metric cards — boarding-pass stub ---------- */
    div[data-testid="stMetric"] {{
        position: relative;
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 96, 120, 0.18);
        padding: 20px 20px 18px 20px;
        border-radius: 14px;
        box-shadow: 0 8px 24px rgba(0, 96, 120, 0.10);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        overflow: visible;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(227, 124, 120, 0.25);
    }}
    /* tear-line under the label, like a ticket stub perforation */
    div[data-testid="stMetricLabel"] {{
        color: {DEEP_TEAL} !important;
        opacity: 0.8;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        border-bottom: 2px dashed rgba(0, 96, 120, 0.25);
        padding-bottom: 12px;
        margin-bottom: 10px;
        display: block;
    }}
    /* punch-hole notches at the tear line */
    div[data-testid="stMetric"]::before,
    div[data-testid="stMetric"]::after {{
        content: "";
        position: absolute;
        top: 44px;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: {BLUSH};
        z-index: 2;
    }}
    div[data-testid="stMetric"]::before {{ left: -9px; }}
    div[data-testid="stMetric"]::after  {{ right: -9px; }}
    div[data-testid="stMetricValue"] {{
        color: {ROSE_CORAL} !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        font-variant-numeric: tabular-nums;
    }}

    /* Tabs */
    button[data-baseweb="tab"] {{
        background-color: transparent !important;
        color: {INK} !important;
        opacity: 0.55;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.5px;
        border-radius: 10px 10px 0 0 !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        opacity: 1;
        color: {ROSE_CORAL} !important;
        border-bottom: 3px solid {ROSE_CORAL} !important;
    }}

    /* Buttons */
    div.stButton > button:first-child {{
        background: linear-gradient(90deg, {ROSE_CORAL} 0%, #f0968f 100%);
        color: {BLUSH};
        width: 100%;
        border: none;
        border-radius: 10px;
        height: 3.2em;
        font-weight: 600;
        font-size: 1.05em;
        letter-spacing: 0.5px;
        box-shadow: 0 6px 18px rgba(227, 124, 120, 0.35);
        transition: all 0.25s ease;
    }}
    div.stButton > button:first-child:hover {{
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 10px 24px rgba(227, 124, 120, 0.5);
    }}

    /* Download button */
    div.stDownloadButton > button:first-child {{
        background: transparent;
        color: {DEEP_TEAL} !important;
        border: 2px solid {DEEP_TEAL};
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
    }}
    div.stDownloadButton > button:first-child:hover {{
        background: {DEEP_TEAL};
        color: {BLUSH} !important;
    }}

    /* Dataframes */
    div[data-testid="stDataFrame"] {{
        background: rgba(255, 255, 255, 0.5);
        border-radius: 12px;
        border: 1px solid rgba(0, 96, 120, 0.15);
        padding: 4px;
    }}

    /* Divider */
    hr {{
        border-color: rgba(227, 124, 120, 0.35) !important;
    }}

    /* Success message */
    div[data-testid="stSuccess"] {{
        background-color: rgba(130, 186, 196, 0.35) !important;
        border-left: 4px solid {DEEP_TEAL} !important;
        border-radius: 10px !important;
        color: {DEEP_TEAL} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# Shared Plotly template matching the palette
PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Poppins, sans-serif", color=INK),
        xaxis=dict(gridcolor="rgba(0,96,120,0.12)", zerolinecolor="rgba(0,96,120,0.25)"),
        yaxis=dict(gridcolor="rgba(0,96,120,0.12)", zerolinecolor="rgba(0,96,120,0.25)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
)

# ------------------------------------------------
# Sidebar & Logic
# ------------------------------------------------

with st.sidebar:
    import os
    logo_path = "assets/201623.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=100)
    else:
        st.markdown("### ✈️")
    st.title("Settings")
    future_months = st.slider("Forecast Horizon (Months)", 1, 24, 12)
    st.info("Adjust the slider to change the prediction window for the RNN model.")

# ------------------------------------------------
# Data & Header
# ------------------------------------------------

loader = DataLoader("data/airline_passengers.csv")
df = loader.load_data()

st.markdown(f"""
    <div class="flight-header">
        <div class="airport-code">HIST</div>
        <div class="flight-path">
            <span class="dot"></span>
            <span class="seg"></span>
            <span class="plane">✈️</span>
            <span class="seg"></span>
            <span class="dot live"></span>
        </div>
        <div class="airport-code">FCST</div>
    </div>
    """, unsafe_allow_html=True)

st.title("Airline Passenger Analysis & Forecasting")
st.markdown(
    '<p class="mono" style="opacity:0.65; margin-top:-10px; font-size:0.85rem; '
    'letter-spacing:1px;">RNN · LSTM TIME SERIES MODEL — HISTORICAL DATA IN, FUTURE DEMAND OUT</p>',
    unsafe_allow_html=True
)

# ------------------------------------------------
# Metrics & Overview Tabs
# ------------------------------------------------

tab1, tab2 = st.tabs(["🚀 Model Performance", "🔎 Exploratory Data Analysis"])

with tab1:
    st.markdown('<div class="eyebrow">Boarding Pass · Model Metrics</div>', unsafe_allow_html=True)
    st.subheader("Model Accuracy Metrics")
    mae, mse, rmse = Evaluator().evaluate()

    m1, m2, m3 = st.columns(3)
    m1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}", delta_color="inverse")
    m2.metric("Mean Squared Error (MSE)", f"{mse:.2f}", delta_color="inverse")
    m3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}", delta_color="inverse")

with tab2:
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown('<div class="eyebrow">Manifest</div>', unsafe_allow_html=True)
        st.subheader("Raw Data")
        st.dataframe(df, height=350)

    with col_b:
        st.markdown('<div class="eyebrow">Flight Recorder</div>', unsafe_allow_html=True)
        st.subheader("Historical Trend")
        fig = px.line(df, x=df.index, y="Passengers",
                      template=PLOTLY_TEMPLATE,
                      color_discrete_sequence=[DEEP_TEAL])
        fig.update_traces(line=dict(width=2.5))
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# Forecasting Section
# ------------------------------------------------

st.markdown("---")
st.markdown('<div class="eyebrow">Now Boarding · Departure FCST</div>', unsafe_allow_html=True)
st.header("🔮 Generate Future Forecast")

if st.button("Run RNN Model"):
    with st.spinner("Analyzing temporal patterns..."):
        forecaster = Forecaster()
        future = forecaster.forecast(future_months)

        last_date = df.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=future_months,
            freq="MS"
        )

        forecast_df = pd.DataFrame({
            "Month": future_dates,
            "Predicted Passengers": future.flatten()
        })

    st.success(f"Successfully generated forecast for {future_months} months!")

    # Layout for Results
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.markdown('<div class="eyebrow">Ticket Stub</div>', unsafe_allow_html=True)
        st.subheader("Forecasted Values")
        st.dataframe(forecast_df, use_container_width=True)

        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="forecast_results.csv",
            mime="text/csv"
        )

    with res_col2:
        st.markdown('<div class="eyebrow">Full Route</div>', unsafe_allow_html=True)
        st.subheader("Combined Projection")

        # Create a combined chart with Plotly
        fig_combined = go.Figure()

        # Historical Data
        fig_combined.add_trace(go.Scatter(
            x=df.index, y=df["Passengers"],
            name="Historical", line=dict(color=DEEP_TEAL, width=2)
        ))

        # Forecasted Data
        fig_combined.add_trace(go.Scatter(
            x=forecast_df["Month"], y=forecast_df["Predicted Passengers"],
            name="Forecast", line=dict(color=ROSE_CORAL, width=3, dash='dot')
        ))

        fig_combined.update_layout(
            template=PLOTLY_TEMPLATE,
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_combined, use_container_width=True)