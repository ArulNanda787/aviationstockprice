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

import warnings
warnings.filterwarnings("ignore")

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
    pca = PCA(0.9)
    pca.fit(x)
    pca_data = pca.transform(x)
    explained_variance_ratio = pca.explained_variance_ratio_
    loadings = np.sqrt(pca.explained_variance_) * pca.components_.T
    loadings_df = pd.DataFrame(loadings, columns=["PC"+f"{i+1}" for i in range(len(explained_variance_ratio))], index=df.columns[3:])
    loadings_df = loadings_df.round(3)
    top3_df = loadings_df.nlargest(3, 'PC1')
    pc_df = pd.DataFrame(
    pca_data,
    columns=[f"PC{i+1}" for i in range(pca_data.shape[1])]
    )
    df_out = pd.concat([df[['Year', 'Price']].reset_index(drop=True), pc_df], axis=1)
    df_out['Year'] = pd.to_datetime(df_out['Year'])
    df_out = df_out.sort_values('Year').reset_index(drop=True)
    df_out['seq_index'] = range(len(df_out))
    #df_out : year, price, pc1,pc2,...,seq_index
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


def forecast_pca_exog(df, steps=8):
    """
    Forecast each PCA component (exogenous variable) separately using auto_arima.
    Returns np.array of shape (steps, num_PCs).
    """
    exog = df.iloc[:, 2:-1]  # all PC columns
    future_pcs = []

    for col in exog.columns:
        model_pc = auto_arima(
            exog[col],
            seasonal=True,
            m=12,
            trace=False,
            error_action='ignore',
            suppress_warnings=True
        )
        pc_forecast = model_pc.predict(n_periods=steps)
        future_pcs.append(pc_forecast)

    # shape (steps, n_PCs)
    exog_future = np.column_stack(future_pcs)
    return exog_future


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

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def time_series(airline,forecast_periods):
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
    result_table = optimize_SARIMAX(parameters, d=d, D=D, s=12, endog=df['Price'], exog=df.iloc[:,2:-1])

    if result_table.empty:
        raise ValueError("No SARIMAX models converged in optimize_SARIMAX. Try smaller search space or change data.")

    bestvals = result_table.iloc[0, 0]
    p, q, P, Q = bestvals

    best_model = SARIMAX(endog=df['Price'], exog=df.iloc[:,2:-1],
                         order=(p, d, q),
                         seasonal_order=(P, D, Q, 12),
                         enforce_stationarity=False,
                         enforce_invertibility=False).fit(disp=False)

 # Forecast each PCA exog for the next n months
    exog_future = forecast_pca_exog(df, steps=forecast_periods)

    fitted_values = best_model.fittedvalues
    fitted_values.iloc[:max(5, int(len(fitted_values) * 0.05))] = np.NaN

    forecast = best_model.get_forecast(steps=forecast_periods, exog=exog_future)
    forecast_mean = forecast.predicted_mean

    forecast_index = np.arange(len(df), len(df) + len(forecast_mean))
    plot_price(df, forecast_index, forecast_mean, fitted_values)
    # Print summary
    forecast_summary = {
    "Last actual price": df['Price'].iloc[-1],
    "First forecasted price": forecast_mean.iloc[0],
    "Last forecasted price": forecast_mean.iloc[-1],
    "Forecasted values": forecast_mean.tolist()
    }
    metrics = model_metrics(df,best_model)
    return forecast_summary, forecast_index, forecast_mean, fitted_values, df, best_model, metrics


def Airline(airline, forecast_periods):
    df, top3_df, explained_variance_ratio = sensitivity_index(airline)
    forecast_summary, forecast_index, forecast_mean, fitted_values, df, best_model, metrics = time_series(airline, forecast_periods)

    # Create Figures for Streamlit
    fig_forecast, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df.index, df['Price'], label='Actual Price', linewidth=2)
    ax1.plot(df.index, fitted_values, label='Fitted', linestyle='--')
    ax1.plot(forecast_index, forecast_mean, label='Forecast', color='red', marker='o')
    ax1.legend()
    ax1.set_title('Forecasted vs Actual Prices')
    plt.tight_layout()

    # Decomposition Plot
    ts = pd.Series(df['Price'].values, index=df['seq_index'])
    if len(ts) >= 24:
        result = seasonal_decompose(ts, model='multiplicative', period=12)
        fig_decomp, axes = plt.subplots(2, 2, figsize=(12, 6))
        axes[0, 0].plot(ts); axes[0, 0].set_title("Original")
        axes[0, 1].plot(result.trend); axes[0, 1].set_title("Trend")
        axes[1, 0].plot(result.seasonal); axes[1, 0].set_title("Seasonal")
        axes[1, 1].plot(result.resid); axes[1, 1].set_title("Residual")
        plt.tight_layout()
    else:
        fig_decomp = None

    return top3_df, explained_variance_ratio, fig_forecast, fig_decomp, metrics, pd.DataFrame(forecast_summary)


if __name__ == "__main__":
    Airline("american",8)
