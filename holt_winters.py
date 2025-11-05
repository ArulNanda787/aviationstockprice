from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

#STEP 1: Data Import + PCA 
def import_and_clean_data(airline):
    df = pd.read_csv(f"{airline}.csv")
    numeric_cols = df.columns[2:]
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(",", "")
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Year'] = pd.to_datetime(df['Year'], format="%Y")
    df['Month'] = pd.to_datetime(df['Month'], format="%m")
    x = df.iloc[:, 3:]
    standard = StandardScaler()
    x = standard.fit_transform(x)
    return x, df

def sensitivity_index(airline):
    x, df = import_and_clean_data(airline)
    pca = PCA(0.9)
    pca.fit(x)
    pca_data = pca.transform(x)
    explained_variance_ratio = pca.explained_variance_ratio_
    loadings = np.sqrt(pca.explained_variance_) * pca.components_.T
    loadings_df = pd.DataFrame(
        loadings,
        columns=["PC" + f"{i+1}" for i in range(len(explained_variance_ratio))],
        index=df.columns[3:]
    )
    loadings_df = loadings_df.round(3)
    top3_df = loadings_df.nlargest(3, 'PC1')
    pc_df = pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(pca_data.shape[1])])
    df_out = pd.concat([df[['Year', 'Price']].reset_index(drop=True), pc_df], axis=1)
    df_out['Year'] = pd.to_datetime(df_out['Year'])
    df_out = df_out.sort_values('Year').reset_index(drop=True)
    df_out['seq_index'] = range(len(df_out))
    return df_out, top3_df, explained_variance_ratio

#STEP 2: Seasonal Decomposition
def plot_seasonal_decomposition(ts, period=12):
    result = seasonal_decompose(ts, model='multiplicative', period=period)
    fig, axes = plt.subplots(2, 2, figsize=(15, 7))
    axes[0, 0].plot(ts); axes[0, 0].set_title("Original Series")
    axes[0, 1].plot(result.trend); axes[0, 1].set_title("Trend")
    axes[1, 0].plot(result.seasonal); axes[1, 0].set_title("Seasonal")
    axes[1, 1].plot(result.resid); axes[1, 1].set_title("Residual")
    plt.tight_layout()
    plt.show()

#STEP 3: Automatic Additive/Multiplicative Detection 
def detect_model_type(series):
    """Automatically decide between additive and multiplicative based on variance behavior."""
    mean_val = series.mean()
    std_val = series.std()
    ratio = std_val / mean_val

    if ratio > 0.3:
        trend_type = "mul"
        seasonal_type = "mul"
    else:
        trend_type = "add"
        seasonal_type = "add"

    print(f"Detected model type: trend='{trend_type}', seasonal='{seasonal_type}' (ratio={ratio:.2f})")
    return trend_type, seasonal_type

#STEP 4: Model Evaluation 
def model_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

#STEP 5: Holt-Winters Forecasting 
def holt_winters_forecast(airline, forecast_periods=8):
    df, top3_df, explained_variance_ratio = sensitivity_index(airline)
    ts = pd.Series(df['Price'].values, index=df['seq_index'])

    # Plot decomposition
    if len(ts) >= 24:
        plot_seasonal_decomposition(ts)

    # Auto detect model type
    trend_type, seasonal_type = detect_model_type(ts)

    # Fit Holt-Winters model
    model = ExponentialSmoothing(
        ts,
        trend=trend_type,
        seasonal=seasonal_type,
        seasonal_periods=12
    ).fit(optimized=True)

    fitted_values = model.fittedvalues
    forecast = model.forecast(forecast_periods)

    # Create forecast index
    forecast_index = np.arange(len(ts), len(ts) + forecast_periods)

    # Plot results
    plt.figure(figsize=(15, 7))
    plt.plot(ts.index, ts, label='Actual', linewidth=2, marker='o')
    plt.plot(ts.index, fitted_values, label='Fitted', linestyle='--', linewidth=2)
    plt.plot(forecast_index, forecast, label='Forecast', color='red', marker='s', linewidth=2)
    plt.axvline(x=len(ts)-0.5, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
    plt.title("Holt-Winters Forecast (Auto Add/Mul Detection)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    metrics = model_metrics(ts, fitted_values)

    # Create monthly forecast results
    forecast_df = pd.DataFrame({
        "Month": [f"Month {i+1}" for i in range(forecast_periods)],
        "Predicted Price": forecast.values
    })

    # Summary with all predictions + final one
    forecast_summary = {
        "Forecasted for next months": forecast_periods,
        "Predicted Prices": forecast_df,
    }

    return top3_df, explained_variance_ratio, model, forecast_summary, metrics, trend_type, seasonal_type


if __name__ == "__main__":
    top3_df, explained_variance_ratio, model, forecast_summary, metrics, trend_type, seasonal_type = holt_winters_forecast("american", 8)
    print("\nTop 3 PCA Loadings:\n", top3_df)
    print("\nExplained Variance Ratio:\n", explained_variance_ratio)
    print("\nForecast Summary:\n", forecast_summary)
    print("\nModel Metrics:\n", metrics)
    print(f"\nModel Type Used → Trend: {trend_type.upper()}, Seasonal: {seasonal_type.upper()}")
