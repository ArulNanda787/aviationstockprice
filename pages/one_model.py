import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from main import Airline  # ✅ Import your SARIMAX backend function
from main import Airline, sensitivity_index, detect_model_type
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go


st.title("🔍 Use One Model")
st.markdown("Select a forecasting model for your chosen airline.")

# Get selected airline from session state
airline = st.session_state.get("selected_airline", "No airline selected")
st.markdown(f"### ✈️ Airline: **{airline}**")

models = ["SARIMAX", "Prophet", "Exponential Smoothing"]

# Define colors
model_colors = {
    "SARIMAX": "#28a745",  # Green
    "Prophet": "#007bff",  # Blue
    "Exponential Smoothing": "#dc3545"  # Red
}

# --- CSS Styling ---
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

# --- Model selection buttons ---
cols = st.columns(3)
for i, model in enumerate(models):
    with cols[i]:
        selected = st.session_state.get("selected_model") == model
        button_type = "primary" if selected else "secondary"

        if st.button(f"{model}", key=model, use_container_width=True, type=button_type):
            st.session_state["selected_model"] = model
            st.rerun()

# --- Highlight selected model ---
selected_model = st.session_state.get("selected_model")
if selected_model:
    model_index = models.index(selected_model)
    button_style = f"""
        <style>
        div[data-testid="stHorizontalBlock"] > div:nth-child({model_index + 1}) div[data-testid="stButton"] > button {{
            background-color: {model_colors[selected_model]} !important;
            color: white !important;
            border: none !important;
        }}
        </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)

# --- Run model section ---
if selected_model:
    st.success(f"✅ You selected: {selected_model}")
    st.markdown("### 📈 Run forecast and visualize results")

    forecast_periods = st.number_input("Enter forecast periods (months):", min_value=1, max_value=24, value=8)

    if st.button("🚀 Run Forecast"):
        if airline == "No airline selected":
            st.error("Please select an airline first on the main page.")
        else:
                        airline_key = str(airline).strip().lower()

        if selected_model == "SARIMAX":
                with st.spinner("Running SARIMAX model..."):
                    ts, loadings_df, explained_variance_ratio, fig_forecast, metrics, forecast_summary, df, fitted_values = Airline(Airline(str(airline).strip().lower(), forecast_periods, model="SARIMAX")
, forecast_periods)

                    st.subheader("📈 Forecast Visualization")
                    # Ensure datetime format for Plotly
                    df["Date"] = pd.to_datetime(df["Date"])
                    forecast_summary["Date"] = pd.to_datetime(forecast_summary["Date"])

                    fig = go.Figure()

                    # Actual prices
                    fig.add_trace(go.Scatter(
                        x=df["Date"], y=df["Price"],
                        mode='lines+markers',
                        name='Actual Price',
                        line=dict(color='#2E86AB', width=2)
                    ))

                    # Fitted values
                    fig.add_trace(go.Scatter(
                        x=df["Date"], y=fitted_values,
                        mode='lines',
                        name='Fitted Values',
                        line=dict(color='#82E0AA', dash='dash')
                    ))

                    # Forecasted values
                    fig.add_trace(go.Scatter(
                        x=forecast_summary["Date"], y=forecast_summary["Forecasted Price"],
                        mode='lines+markers',
                        name='Forecasted Price',
                        line=dict(color='#E74C3C', width=2)
                    ))

                    # Add vertical line marking forecast start using add_shape instead
                    forecast_start_date = forecast_summary["Date"].iloc[0]
                    fig.add_shape(
                        type="line",
                        x0=forecast_start_date,
                        x1=forecast_start_date,
                        y0=0,
                        y1=1,
                        yref="paper",
                        line=dict(color="red", width=2, dash="dash")
                    )
                    
                    # Add annotation for the vertical line
                    fig.add_annotation(
                        x=forecast_start_date,
                        y=1,
                        yref="paper",
                        text="Forecast Start",
                        showarrow=False,
                        yshift=10,
                        font=dict(color="red")
                    )


                    fig.update_layout(
                        title="SARIMAX: Actual vs Fitted vs Forecasted Price",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        hovermode="x unified",
                        template="plotly_white",
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)


                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<h3>📊 Model Performance Metrics</h3>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    mae = metrics.get("MAE", 0)
                    rmse = metrics.get("RMSE", 0)
                    mape = metrics.get("MAPE", 0)
                    with col1: st.metric("MAE", f"${mae:.3f}")
                    with col2: st.metric("RMSE", f"${rmse:.3f}")
                    with col3: st.metric("MAPE", f"{mape:.2f}%")
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.subheader("📅 Forecasted Values")
                    forecast_summary["Date"] = forecast_summary["Date"].dt.date
                    st.dataframe(forecast_summary, hide_index = True)


        elif selected_model == "Prophet":
                    with st.spinner("Running Prophet model..."):
                          
                          airline_key = str(airline).strip().lower()

                    # Run unified gateway
                    res = Airline(airline_key, forecast_periods, model="PROPHET")
                    forecast_summary = res["Forecast"].copy()     # columns: Date, yhat, yhat_lower, yhat_upper
                    metrics = res["Metrics"]

                    # Actual data (for plotting)
                    df, _, _ = sensitivity_index(airline_key)
                    df["Date"] = pd.to_datetime(df["Date"])
                    forecast_summary["Date"] = pd.to_datetime(forecast_summary["Date"])

                    # Fitted values = in-sample Prophet yhat aligned to history dates
                    fitted_map = forecast_summary.set_index("Date")["yhat"]
                    fitted_values = df["Date"].map(fitted_map)

                    # Only future rows for the forecast line
                    future_mask = forecast_summary["Date"] > df["Date"].max()
                    fc_future = forecast_summary.loc[future_mask].iloc[:forecast_periods].copy()
                    fc_future.rename(columns={"yhat": "Forecasted Price"}, inplace=True)

                    st.subheader("📈 Forecast Visualization")
                    fig = go.Figure()

                    # Actual prices
                    fig.add_trace(go.Scatter(
                        x=df["Date"], y=df["Price"],
                        mode='lines+markers',
                        name='Actual Price',
                        line=dict(color='#2E86AB', width=2)
                    ))

                    # Fitted values (in-sample)
                    fig.add_trace(go.Scatter(
                        x=df["Date"], y=fitted_values,
                        mode='lines',
                        name='Fitted Values',
                        line=dict(color='#82E0AA', dash='dash')
                    ))

                    # Forecasted values (future yhat)
                    fig.add_trace(go.Scatter(
                        x=fc_future["Date"], y=fc_future["Forecasted Price"],
                        mode='lines+markers',
                        name='Forecasted Price',
                        line=dict(color='#E74C3C', width=2)
                    ))

                    # Vertical line at forecast start
                    forecast_start_date = fc_future["Date"].iloc[0]
                    fig.add_shape(
                        type="line",
                        x0=forecast_start_date,
                        x1=forecast_start_date,
                        y0=0, y1=1, yref="paper",
                        line=dict(color="red", width=2, dash="dash")
                    )
                    fig.add_annotation(
                        x=forecast_start_date, y=1, yref="paper",
                        text="Forecast Start", showarrow=False, yshift=10,
                        font=dict(color="red")
                    )

                    fig.update_layout(
                        title="Prophet: Actual vs Fitted vs Forecasted Price",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        hovermode="x unified",
                        template="plotly_white",
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Metrics card
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<h3>📊 Model Performance Metrics</h3>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    mae = metrics.get("MAE", 0)
                    rmse = metrics.get("RMSE", 0)
                    mape = metrics.get("MAPE", 0)
                    with col1: st.metric("MAE", f"${mae:.3f}")
                    with col2: st.metric("RMSE", f"${rmse:.3f}")
                    with col3: st.metric("MAPE", f"{mape:.2f}%")
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Table in the same style
                    st.subheader("📅 Forecasted Values")
                    out = fc_future[["Date", "Forecasted Price"]].copy()
                    out["Date"] = out["Date"].dt.date
                    st.dataframe(out, hide_index=True)


        elif selected_model == "Exponential Smoothing":
                with st.spinner("Running Holt–Winters (Exponential Smoothing) model..."):
                    airline_key = str(airline).strip().lower()

                    # Unified gateway (returns forecast & metrics; we’ll compute fitted here for plotting)
                    res = Airline(airline_key, forecast_periods, model="HOLT-WINTERS")
                    forecast_summary = res["Forecast"].copy()  # columns: Date, Forecasted Price
                    metrics = res["Metrics"]
                    trend_type = res.get("Trend", "add")
                    seasonal_type = res.get("Seasonal", "add")

                    # Actual data + fitted
                    df, _, _ = sensitivity_index(airline_key)
                    df["Date"] = pd.to_datetime(df["Date"])
                    forecast_summary["Date"] = pd.to_datetime(forecast_summary["Date"])

                    # Refit quickly to get fitted values (same detected types)
                    ts = pd.Series(df["Price"].values, index=df["Date"])
                    hw_fit = ExponentialSmoothing(
                        ts, trend=trend_type, seasonal=seasonal_type, seasonal_periods=12
                    ).fit(optimized=True)
                    fitted_values = hw_fit.fittedvalues

                    st.subheader("📈 Forecast Visualization")
                    fig = go.Figure()

                    # Actual prices
                    fig.add_trace(go.Scatter(
                        x=df["Date"], y=df["Price"],
                        mode='lines+markers',
                        name='Actual Price',
                        line=dict(color='#2E86AB', width=2)
                    ))

                    # Fitted values
                    fig.add_trace(go.Scatter(
                        x=df["Date"], y=fitted_values,
                        mode='lines',
                        name='Fitted Values',
                        line=dict(color='#82E0AA', dash='dash')
                    ))

                    # Forecasted values
                    fig.add_trace(go.Scatter(
                        x=forecast_summary["Date"], y=forecast_summary["Forecasted Price"],
                        mode='lines+markers',
                        name='Forecasted Price',
                        line=dict(color='#E74C3C', width=2)
                    ))

                    # Vertical line at forecast start
                    forecast_start_date = forecast_summary["Date"].iloc[0]
                    fig.add_shape(
                        type="line",
                        x0=forecast_start_date,
                        x1=forecast_start_date,
                        y0=0, y1=1, yref="paper",
                        line=dict(color="red", width=2, dash="dash")
                    )
                    fig.add_annotation(
                        x=forecast_start_date, y=1, yref="paper",
                        text=f"Forecast Start ({trend_type}/{seasonal_type})",
                        showarrow=False, yshift=10, font=dict(color="red")
                    )

                    fig.update_layout(
                        title="Holt–Winters: Actual vs Fitted vs Forecasted Price",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        hovermode="x unified",
                        template="plotly_white",
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Metrics card
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<h3>📊 Model Performance Metrics</h3>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    mae = metrics.get("MAE", 0)
                    rmse = metrics.get("RMSE", 0)
                    mape = metrics.get("MAPE", 0)
                    with col1: st.metric("MAE", f"${mae:.3f}")
                    with col2: st.metric("RMSE", f"${rmse:.3f}")
                    with col3: st.metric("MAPE", f"{mape:.2f}%")
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Table in the same style
                    st.subheader("📅 Forecasted Values")
                    out = forecast_summary.copy()
                    out["Date"] = out["Date"].dt.date
                    st.dataframe(out, hide_index=True)
