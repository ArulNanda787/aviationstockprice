# pages/compare_models.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# ✅ reuse your existing logic (no changes to those files)
from main import Airline, sensitivity_index
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------- UI SETUP ----------------
st.set_page_config(layout="wide")
st.title("📊 Compare Models")
st.markdown("Compare the forecasting performance of two models.")

# --- Initialize session state ---
if "selected_models" not in st.session_state:
    st.session_state.selected_models = []

# --- Styling ---
st.markdown("""
    <style>
    div[data-testid="stButton"] > button {
        width: 100%;
        padding: 14px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.1);
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-3px);
        box-shadow: 0px 6px 14px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Inputs (keep airline here) ---
with st.sidebar:
    st.header("⚙️ Settings")
    airline = st.selectbox("✈️ Select Airline", ["American", "Delta", "Southwest", "United"])
    # ⛔️ removed forecast_periods from sidebar

# --- Model Selection ---
models = ["SARIMAX", "Prophet", "Exponential Smoothing"]
st.markdown("### Select two models to compare:")

cols = st.columns(len(models))
for i, model in enumerate(models):
    with cols[i]:
        selected = model in st.session_state.selected_models
        button_type = "primary" if selected else "secondary"
        if st.button(model, key=f"pick_{model}", use_container_width=True, type=button_type):
            if selected:
                st.session_state.selected_models.remove(model)
            elif len(st.session_state.selected_models) < 2:
                st.session_state.selected_models.append(model)
            else:
                st.warning("⚠️ You can select up to 2 models only.")
            st.rerun()

st.markdown("---")

# 👉 NEW: forecast periods input in the main area (same as one_model.py)
forecast_periods = st.number_input(
    "Enter forecast periods (months):",
    min_value=1, max_value=24,
    value=st.session_state.get("cmp_periods", 8),
    key="cmp_periods"
)

selected_models = st.session_state.selected_models

# ✅ Run button right under selections
if len(selected_models) == 2:
    st.success(f"✅ Selected: {selected_models[0]} and {selected_models[1]}")
    run_btn = st.button("🚀 Run Comparison", use_container_width=True)
else:
    st.info("Please select exactly two models to compare.")
    run_btn = False

# ---------------- HELPERS ----------------
def _normalize_airline_label(label: str) -> str:
    """Match your CSV naming (american.csv, delta.csv, southwest.csv, united.csv)."""
    return str(label).strip().lower()

def _get_df_from_res(res: dict, airline_key: str) -> pd.DataFrame:
    """Use DF from Airline() if present; otherwise rebuild from sensitivity_index()."""
    if isinstance(res, dict) and "DF" in res and isinstance(res["DF"], pd.DataFrame):
        return res["DF"].copy()
    df, _, _ = sensitivity_index(airline_key)
    return df

def _render_sarimax(airline_key: str, periods: int):
    res = Airline(airline_key, periods, model="SARIMAX")
    df = _get_df_from_res(res, airline_key)
    forecast_summary = res["Forecast"].copy()
    metrics = res.get("Metrics", {})
    fitted_values = res.get("Fitted", None)

    df["Date"] = pd.to_datetime(df["Date"])
    forecast_summary["Date"] = pd.to_datetime(forecast_summary["Date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Price"],
                             mode='lines+markers', name='Actual Price',
                             line=dict(color='#2E86AB', width=2)))
    if fitted_values is not None:
        fig.add_trace(go.Scatter(x=df["Date"], y=fitted_values,
                                 mode='lines', name='Fitted Values',
                                 line=dict(color='#82E0AA', dash='dash')))
    fig.add_trace(go.Scatter(x=forecast_summary["Date"], y=forecast_summary["Forecasted Price"],
                             mode='lines+markers', name='Forecasted Price',
                             line=dict(color='#E74C3C', width=2)))

    forecast_start_date = forecast_summary["Date"].iloc[0]
    fig.add_shape(type="line", x0=forecast_start_date, x1=forecast_start_date,
                  y0=0, y1=1, yref="paper", line=dict(color="red", width=2, dash="dash"))
    fig.add_annotation(x=forecast_start_date, y=1, yref="paper",
                       text="Forecast Start", showarrow=False, yshift=10, font=dict(color="red"))

    fig.update_layout(title="SARIMAX: Actual vs Fitted vs Forecasted Price",
                      xaxis_title="Date", yaxis_title="Price",
                      hovermode="x unified", template="plotly_white",
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                      height=500)

    table_df = forecast_summary.copy()
    table_df["Date"] = table_df["Date"].dt.date
    return fig, metrics, table_df

def _render_prophet(airline_key: str, periods: int):
    res = Airline(airline_key, periods, model="PROPHET")
    df_hw = _get_df_from_res(res, airline_key)
    forecast_summary = res["Forecast"].copy()
    metrics = res["Metrics"]

    df, _, _ = sensitivity_index(airline_key)
    df["Date"] = pd.to_datetime(df["Date"])
    forecast_summary["Date"] = pd.to_datetime(forecast_summary["Date"])

    fitted_map = forecast_summary.set_index("Date")["yhat"]
    fitted_values = df["Date"].map(fitted_map)

    future_mask = forecast_summary["Date"] > df["Date"].max()
    fc_future = forecast_summary.loc[future_mask].iloc[:periods].copy()
    fc_future["Forecasted Price"] = fc_future["yhat"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Price"],
                             mode='lines+markers', name='Actual Price',
                             line=dict(color='#2E86AB', width=2)))
    fig.add_trace(go.Scatter(x=df["Date"], y=fitted_values,
                             mode='lines', name='Fitted Values',
                             line=dict(color='#82E0AA', dash='dash')))
    fig.add_trace(go.Scatter(x=fc_future["Date"], y=fc_future["Forecasted Price"],
                             mode='lines+markers', name='Forecasted Price',
                             line=dict(color='#E74C3C', width=2)))

    forecast_start_date = fc_future["Date"].iloc[0]
    fig.add_shape(type="line", x0=forecast_start_date, x1=forecast_start_date,
                  y0=0, y1=1, yref="paper", line=dict(color="red", width=2, dash="dash"))
    fig.add_annotation(x=forecast_start_date, y=1, yref="paper",
                       text="Forecast Start", showarrow=False, yshift=10, font=dict(color="red"))

    fig.update_layout(title="Prophet: Actual vs Fitted vs Forecasted Price",
                      xaxis_title="Date", yaxis_title="Price",
                      hovermode="x unified", template="plotly_white",
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                      height=500)

    table_df = fc_future[["Date", "Forecasted Price"]].copy()
    table_df["Date"] = table_df["Date"].dt.date
    return fig, metrics, table_df

def _render_holt_winters(airline_key: str, periods: int):
    res = Airline(airline_key, periods, model="HOLT-WINTERS")
    df_hw = _get_df_from_res(res, airline_key)
    forecast_summary = res["Forecast"].copy()
    metrics = res["Metrics"]
    trend_type = res.get("Trend", "add")
    seasonal_type = res.get("Seasonal", "add")

    df_hw["Date"] = pd.to_datetime(df_hw["Date"])
    forecast_summary["Date"] = pd.to_datetime(forecast_summary["Date"])

    ts = pd.Series(df_hw["Price"].values, index=df_hw["Date"])
    hw_fit = ExponentialSmoothing(ts, trend=trend_type, seasonal=seasonal_type, seasonal_periods=12).fit(optimized=True)
    fitted_values = hw_fit.fittedvalues

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hw["Date"], y=df_hw["Price"],
                             mode='lines+markers', name='Actual Price',
                             line=dict(color='#2E86AB', width=2)))
    fig.add_trace(go.Scatter(x=df_hw["Date"], y=fitted_values,
                             mode='lines', name='Fitted Values',
                             line=dict(color='#82E0AA', dash='dash')))
    fig.add_trace(go.Scatter(x=forecast_summary["Date"], y=forecast_summary["Forecasted Price"],
                             mode='lines+markers', name='Forecasted Price',
                             line=dict(color='#E74C3C', width=2)))

    forecast_start_date = forecast_summary["Date"].iloc[0]
    fig.add_shape(type="line", x0=forecast_start_date, x1=forecast_start_date,
                  y0=0, y1=1, yref="paper", line=dict(color="red", width=2, dash="dash"))
    fig.add_annotation(x=forecast_start_date, y=1, yref="paper",
                       text=f"Forecast Start ({trend_type}/{seasonal_type})",
                       showarrow=False, yshift=10, font=dict(color="red"))

    fig.update_layout(title="Holt–Winters: Actual vs Fitted vs Forecasted Price",
                      xaxis_title="Date", yaxis_title="Price",
                      hovermode="x unified", template="plotly_white",
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                      height=500)

    table_df = forecast_summary.copy()
    table_df["Date"] = table_df["Date"].dt.date
    return fig, metrics, table_df

def render_model_block(model_name: str, airline_key: str, periods: int):
    if model_name == "SARIMAX":
        return _render_sarimax(airline_key, periods)
    elif model_name == "Prophet":
        return _render_prophet(airline_key, periods)
    elif model_name == "Exponential Smoothing":
        return _render_holt_winters(airline_key, periods)
    else:
        raise ValueError("Unknown model selected")

# ---------------- RUN COMPARISON ----------------
if len(selected_models) == 2 and run_btn:
    airline_key = _normalize_airline_label(airline)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(selected_models[0])
        with st.spinner(f"Running {selected_models[0]}..."):
            fig1, metrics1, table1 = render_model_block(selected_models[0], airline_key, forecast_periods)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>📊 Model Performance Metrics</h3>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("MAE", f"${metrics1.get('MAE', 0):.3f}")
        with c2: st.metric("RMSE", f"${metrics1.get('RMSE', 0):.3f}")
        with c3: st.metric("MAPE", f"{metrics1.get('MAPE', 0):.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)
        st.subheader("📅 Forecasted Values")
        st.dataframe(table1, hide_index=True)

    with col2:
        st.subheader(selected_models[1])
        with st.spinner(f"Running {selected_models[1]}..."):
            fig2, metrics2, table2 = render_model_block(selected_models[1], airline_key, forecast_periods)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>📊 Model Performance Metrics</h3>", unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1: st.metric("MAE", f"${metrics2.get('MAE', 0):.3f}")
        with d2: st.metric("RMSE", f"${metrics2.get('RMSE', 0):.3f}")
        with d3: st.metric("MAPE", f"{metrics2.get('MAPE', 0):.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)
        st.subheader("📅 Forecasted Values")
        st.dataframe(table2, hide_index=True)
elif len(selected_models) == 2 and not run_btn:
    st.warning("Click **🚀 Run Comparison** to generate results.")
