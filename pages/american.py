import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from main import Airline  # your backend function

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="American Airlines Forecast ✈️",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- STYLING ---
st.markdown("""
<style>
/* GLOBAL */
.stApp {
    background: linear-gradient(145deg, #e8edf2, #f8fafc);
    font-family: 'Poppins', sans-serif;
    color: #0d1b2a;
}

/* HEADER */
.header {
    background: radial-gradient(circle at top left, #003566, #001d3d 80%);
    color: #ffffff;
    text-align: center;
    padding: 45px 0;
    border-radius: 22px;
    box-shadow: 0 10px 35px rgba(0, 21, 41, 0.4);
    margin-bottom: 50px;
    transition: transform 0.3s ease;
}
.header:hover { transform: translateY(-3px); }
.header h1 {
    font-weight: 700;
    font-size: 2.5rem;
    letter-spacing: 0.5px;
    text-shadow: 0px 0px 12px rgba(255,255,255,0.25);
}
.header p {
    font-size: 1.05rem;
    color: #cfe2ff;
    margin-top: 10px;
}

/* INPUT FIELD */
div[data-baseweb="input"] > input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1.8px solid #003566 !important;
    border-radius: 10px !important;
    padding: 0.7em 1em !important;
    font-size: 1rem !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    transition: 0.3s ease;
}
div[data-baseweb="input"] > input:focus {
    box-shadow: 0 0 10px #0077b6 !important;
    border: 1.8px solid #0077b6 !important;
}
label[data-testid="stWidgetLabel"] {
    color: #000000 !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}

/* BUTTONS */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #001d3d, #003566);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    font-size: 17px;
    padding: 0.8em 2.2em;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
    transition: all 0.3s ease-in-out;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #000814, #003566);
    color: #00b4d8;
    transform: translateY(-3px) scale(1.05);
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

/* CARD CONTAINERS */
.card {
    background: linear-gradient(145deg, #ffffff, #f3f4f6);
    border-radius: 20px;
    padding: 35px;
    margin: 30px auto;
    width: 90%;
    box-shadow: 0 12px 24px rgba(0,0,0,0.08),
                inset 0 0 10px rgba(255,255,255,0.4);
    transition: 0.3s ease;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 18px 35px rgba(0,0,0,0.12),
                0 0 15px rgba(0, 119, 182, 0.2);
}

/* SECTION HEADINGS */
h3 {
    color: #001d3d !important;
    font-weight: 700;
    text-align: center;
    letter-spacing: 0.3px;
    margin-bottom: 20px;
}

/* METRICS */
[data-testid="metric-container"] {
    background: linear-gradient(180deg, #ffffff 60%, #f8f9fa 100%) !important;
    border: 1px solid rgba(0, 53, 102, 0.2);
    border-radius: 16px !important;
    padding: 1.5rem !important;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.12);
    transition: transform 0.25s ease-in-out;
}
[data-testid="metric-container"]:hover {
    transform: scale(1.03);
    box-shadow: 0px 8px 28px rgba(0,0,0,0.15);
}
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="metric-container"] * {
    color: #000000 !important;
    font-weight: 700 !important;
}
[data-testid="stMetricValue"] {
    font-size: 2.5rem !important;
    letter-spacing: 0.3px;
}

/* FOOTER */
.footer {
    background: linear-gradient(90deg, #001d3d, #000814);
    color: #ffffff;
    text-align: center;
    padding: 18px;
    border-radius: 14px;
    margin-top: 60px;
    font-size: 15px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.25);
}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class='header'>
    <h1>✈️ American Airlines Forecast Dashboard</h1>
    <p>AI-powered forecasting using PCA & SARIMAX with time series decomposition</p>
</div>
""", unsafe_allow_html=True)

# --- INPUT SECTION ---
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h3>📅 Enter Forecast Period</h3>", unsafe_allow_html=True)
steps = st.number_input(
    "Enter number of months to forecast ahead:",
    min_value=1,
    max_value=36,
    value=8,
    step=1,
    help="Specify how many future months you want to forecast."
)
run_button = st.button("🚀 Run Forecast")
st.markdown("</div>", unsafe_allow_html=True)

# --- PROCESSING ---
if run_button:
    with st.spinner("Running time series decomposition and forecast... Please wait ⏳"):
        try:
            # Backend returns 6 values
            top3_df, explained_variance_ratio, fig_forecast, fig_decomp, metrics, forecast_summary = Airline("american", steps)

            mae = metrics.get("MAE", 0)
            rmse = metrics.get("RMSE", 0)
            mape = metrics.get("MAPE", 0)

            st.success("✅ Forecast & Decomposition Complete!")
        except Exception as e:
            st.error(f"⚠️ Error during forecast: {e}")
            st.stop()

    # PCA Factors
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>🔍 Top 3 PCA Sensitivity Factors</h3>", unsafe_allow_html=True)
    st.dataframe(top3_df.style.background_gradient(cmap="Blues").format(precision=3))
    st.markdown("</div>", unsafe_allow_html=True)

    # PCA Explained Variance
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>📈 PCA Explained Variance Ratio</h3>", unsafe_allow_html=True)
    fig_var, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, color="#003566")
    ax.set_title("Explained Variance by PCA Components")
    ax.set_xlabel("Principal Components")
    ax.set_ylabel("Variance Ratio")
    st.pyplot(fig_var)
    st.markdown("</div>", unsafe_allow_html=True)

    # Model Metrics
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>📊 Model Performance Metrics</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("MAE", f"{mae:.3f}")
    with col2: st.metric("RMSE", f"{rmse:.3f}")
    with col3: st.metric("MAPE", f"{mape:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    # Time Series Decomposition
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>🧩 Time Series Decomposition</h3>", unsafe_allow_html=True)
    st.markdown("Decomposing the original data into **Trend**, **Seasonal**, and **Residual** components helps reveal hidden patterns before forecasting.")
    st.pyplot(fig_decomp)
    st.markdown("</div>", unsafe_allow_html=True)

    # Forecast Visualization
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>📈 Forecast Visualization</h3>", unsafe_allow_html=True)
    st.pyplot(fig_forecast)
    st.markdown("</div>", unsafe_allow_html=True)

    # Forecasted Data
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>🧾 Forecasted Price Data</h3>", unsafe_allow_html=True)
    st.dataframe(forecast_summary.style.highlight_max(color="lightgreen", axis=0))
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("💡 Enter a forecast period and click **Run Forecast** to start analysis.")

# --- FOOTER ---
st.markdown("""
<div class='footer'>
    © 2025 Aviation Stock Forecast
</div>
""", unsafe_allow_html=True)