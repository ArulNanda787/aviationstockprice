from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
import itertools
from tqdm.notebook import tqdm


def import_and_clean_data(airline):
    df = pd.read_csv(f"{airline}.csv")
    # Convert everything to string, remove commas
    numeric_cols = df.columns[2:]  # Price + features
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(",", "")  # remove commas
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Year'] = pd.to_datetime(df['Year'], format="%Y")
    df['Month'] = pd.to_datetime(df['Month'], format="%m")
    x = df.iloc[:, 3:]
    # standardize
    standard = StandardScaler()
    x = standard.fit_transform(x)
    return x, df


def sensitivity_index(airline):
    x, df = import_and_clean_data(airline)
    # PCA for Sensitivity Index
    pca = PCA(n_components=1)
    pca.fit(x)
    sensitivity_index = pca.transform(x)
    explained_variance_ratio = pca.explained_variance_ratio_
    loadings = np.sqrt(pca.explained_variance_) * pca.components_.T
    loadings_df = pd.DataFrame(loadings, columns=["PC1"], index=df.columns[3:])
    loadings_df = loadings_df.round(3)
    top3_df = loadings_df.nlargest(3, 'PC1')
    df_out = pd.DataFrame({
        "Year": df['Year'],
        "Price": df['Price'],
        "SI": np.array(sensitivity_index).ravel()
    })
    df_out['Year'] = pd.to_datetime(df_out['Year'])
    df_out = df_out.sort_values('Year').reset_index(drop=True)
    df_out['seq_index'] = range(len(df_out))
    return df_out, top3_df,explained_variance_ratio


def plot_seasonal_decomposition(ts, period=12):
    # ts: pd.Series (index can be numeric); period = season length
    result = seasonal_decompose(ts, model='multiplicative', period=period)
    fig, axes = plt.subplots(2, 2, figsize=(20, 8))
    axes[0, 0].plot(ts)
    axes[0, 0].set_title('Time series')
    axes[1, 0].plot(result.seasonal)
    axes[1, 0].set_title('Seasonal component')
    axes[0, 1].plot(result.trend)
    axes[0, 1].set_title('Trend component')
    axes[1, 1].plot(result.resid)
    axes[1, 1].set_title('Random component')
    plt.tight_layout()
    plt.show()


def detect_seasonality(series, period=12, threshold=0.1):
    # Guard: seasonal_decompose needs at least 2 * period observations
    if len(series) < 2 * period:
        return False
    result = seasonal_decompose(series, model='additive', period=period)
    # seasonal strength metric
    sev = result.seasonal.var()
    resv = result.resid.var()
    seasonal_strength = sev / (sev + resv) if (sev + resv) != 0 else 0
    return seasonal_strength > threshold


def differencing(series, seasonal=False):
    """
    Returns (series_after_diff, d, D)
    series input is raw (not logged) and must be a pd.Series.
    """
    series = np.log(series)
    d = 0
    D = 0

    # Regular differencing up to 2 times
    for i in range(2):
        diff_series = series.diff().dropna()
        d = i + 1
        if adfuller(diff_series)[1] < 0.05:
            return diff_series, d, D
        series = diff_series

    # Seasonal differencing if requested
    if seasonal:
        diff_series = series.diff(12).dropna()
        D = 1
        if adfuller(diff_series)[1] < 0.05:
            return diff_series, d, D
        return diff_series, d, D

    # Return what we have if not stationary
    return series, d, D


def stationarity_check_conversion(series, seasonal=False):
    p_val = adfuller(series)[1]
    if p_val > 0.05:
        series_out, d, D = differencing(series, seasonal)
        return series_out, d, D
    else:
        return series, 0, 0


def optimize_SARIMAX(parameters_list, d, D, s, endog, exog=None):
    results = []
    for param in tqdm(parameters_list):
        try:
            model = SARIMAX(
                endog,
                exog=exog,
                order=(param[0], d, param[1]),
                seasonal_order=(param[2], D, param[3], s),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            results.append({
                '(p,q)x(P,Q)': param,
                'AIC': model.aic
            })
        except Exception:
            continue
    result_df = pd.DataFrame(results)
    if result_df.empty:
        return result_df
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    return result_df


def forecasting_seasonality_index(df, steps=8):
    # Use pmdarima to auto-select ARIMA for SI, then refit a SARIMAX and forecast SI
    # auto_arima returns an object with .order and .seasonal_order
    si_arima = auto_arima(df['SI'], seasonal=True, m=12, trace=False, error_action='ignore', suppress_warnings=True)
    order = si_arima.order
    seasonal_order = si_arima.seasonal_order

    si_model = SARIMAX(df['SI'], order=order, seasonal_order=seasonal_order,
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

    si_forecast = si_model.get_forecast(steps=steps)
    si_forecast_values = si_forecast.predicted_mean.values.reshape(-1, 1)
    return si_forecast_values


def plot_price(df, forecast_index, forecast_mean, fitted_values):
    plt.figure(figsize=(15, 7.5))
    plt.plot(df.index, df['Price'], label='Actual Price', linewidth=2, marker='o', markersize=4)
    plt.plot(df.index, fitted_values, label='Fitted Values', linewidth=2)
    plt.plot(forecast_index, forecast_mean, label='Forecasted Price', linewidth=2, marker='s', markersize=6)
    plt.axvspan(len(df) - 0.5, len(df) + len(forecast_mean) - 0.5, alpha=0.3, color='lightgrey', label='Forecast Period')
    plt.axvline(x=len(df) - 0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('SARIMAX: Actual vs Fitted vs Forecasted Price Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def model_metrics(df,best_model):
    y_true = df['Price'][best_model.loglikelihood_burn:]  # skip initial NaNs
    y_pred = best_model.fittedvalues[best_model.loglikelihood_burn:]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred)/y_true)) * 100

    print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.2f}%")


def time_series(airline):
    df, _,_ = sensitivity_index(airline)
    ts = pd.Series(df['Price'].values, index=df['seq_index'])

    # Decomposition plot (only if long enough)
    if len(ts) >= 24:
        plot_seasonal_decomposition(ts)
    seasonality_present = detect_seasonality(ts, period=12)

    # Stationarity + chosen d, D
    ts_stationary, d, D = stationarity_check_conversion(ts, seasonal=seasonality_present)

    # Grid search for (p,q,P,Q)
    p = P = q = Q = range(0, 3)
    parameters = list(itertools.product(p, q, P, Q))
    result_table = optimize_SARIMAX(parameters, d=d, D=D, s=12, endog=df['Price'], exog=df[['SI']])

    if result_table.empty:
        raise ValueError("No SARIMAX models converged in optimize_SARIMAX. Try smaller search space or change data.")

    bestvals = result_table.iloc[0, 0]
    p, q, P, Q = bestvals

    best_model = SARIMAX(endog=df['Price'], exog=df[['SI']],
                         order=(p, d, q),
                         seasonal_order=(P, D, Q, 12),
                         enforce_stationarity=False,
                         enforce_invertibility=False).fit(disp=False)

    si_forecast_values = forecasting_seasonality_index(df, steps=8)
    fitted_values = best_model.fittedvalues
    # Set a few initial values to NaN for alignment if needed
    fitted_values.iloc[:max(5, int(len(fitted_values) * 0.05))] = np.NaN

    forecast = best_model.get_forecast(steps=8, exog=si_forecast_values)
    forecast_mean = forecast.predicted_mean

    forecast_index = np.arange(len(df), len(df) + len(forecast_mean))
    plot_price(df, forecast_index, forecast_mean, fitted_values)
    # Print summary
    print("\n=== Price Forecast Summary ===")
    print(f"Last actual price: {df['Price'].iloc[-1]:.2f}")
    print(f"First forecasted price: {forecast_mean.iloc[0]:.2f}")
    print(f"Last forecasted price: {forecast_mean.iloc[-1]:.2f}")
    print(f"\nForecasted prices:")
    for i, val in enumerate(forecast_mean, 1):
        print(f"  Step {i}: {val:.2f}")
    model_metrics(df,best_model)


def Airline(airline):
    df, top3_df,explained_variance_ratio = sensitivity_index(airline)
    print("Explained variance ratio:\n", explained_variance_ratio)
    print("Top 3 loadings:\n", top3_df)
    time_series(airline)

if __name__ == "__main__":
    Airline("american")
