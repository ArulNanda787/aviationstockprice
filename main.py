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
    df['Date'] = pd.to_datetime(df['Year'].dt.year.astype(str) + '-' + df['Month'].dt.month.astype(str) + '-01')

    # FIX: Only select numeric feature columns for PCA (exclude Year, Month, Date, Price)
    x = df.iloc[:, 3:].select_dtypes(include=[np.number])
    
    # standardize
    standard = StandardScaler()
    x = standard.fit_transform(x)
    return x, df


def sensitivity_index(airline):
    x, df = import_and_clean_data(airline)
    
    # Store the feature column names BEFORE PCA
    feature_cols = df.iloc[:, 3:].select_dtypes(include=[np.number]).columns
    
    # PCA for Sensitivity Index
    pca = PCA(0.9)
    pca.fit(x)
    pca_data = pca.transform(x)
    explained_variance_ratio = pca.explained_variance_ratio_
    loadings = np.sqrt(pca.explained_variance_) * pca.components_.T
    
    # FIX: Use actual feature_cols instead of df.columns[3:]
    loadings_df = pd.DataFrame(loadings, columns=["PC"+f"{i+1}" for i in range(len(explained_variance_ratio))], index=feature_cols)
    loadings_df = loadings_df.round(3)
    pc_df = pd.DataFrame(
        pca_data,
        columns=[f"PC{i+1}" for i in range(pca_data.shape[1])]
    )
    
    # Create df_out with Date FIRST, then numeric columns
    df_out = pd.DataFrame()
    df_out['Date'] = df['Date'].reset_index(drop=True)
    df_out['Price'] = df['Price'].reset_index(drop=True)
    
    # Add PC columns
    for col in pc_df.columns:
        df_out[col] = pc_df[col].values
    
    df_out = df_out.sort_values('Date').reset_index(drop=True)

    return df_out, loadings_df, explained_variance_ratio

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
    # FIX: Explicitly select only PC columns
    exog = df[[col for col in df.columns if col.startswith('PC')]]
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
    import matplotlib.dates as mdates
    
    plt.figure(figsize=(15, 7.5))
    plt.plot(df['Date'], df['Price'], label='Actual Price', linewidth=2, marker='o', markersize=4)
    plt.plot(df['Date'], fitted_values, label='Fitted Values', linewidth=2)
    
    # CRITICAL: Connect forecast to last actual data point
    last_historical_date = df['Date'].iloc[-1]
    last_historical_price = df['Price'].iloc[-1]
    
    complete_forecast_dates = [last_historical_date] + list(forecast_index)
    complete_forecast_values = [last_historical_price] + list(forecast_mean)
    
    plt.plot(complete_forecast_dates, complete_forecast_values, 
             label='Forecasted Price', linewidth=2, marker='s', markersize=6, color='red')
    
    last_forecast_date = forecast_index[-1]
    
    plt.axvspan(last_historical_date, last_forecast_date, alpha=0.3, color='lightgrey', label='Forecast Period')
    plt.axvline(x=last_historical_date, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Format x-axis to show year and month
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Show every 6 months
    plt.xticks(rotation=45, ha='right')
    
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
    ts = pd.Series(df['Price'].values, index=df['Date'])
    
    seasonality_present = detect_seasonality(ts, period=12)

    # Stationarity + chosen d, D
    ts_stationary, d, D = stationarity_check_conversion(ts, seasonal=seasonality_present)

    # Grid search for (p,q,P,Q)
    p = P = q = Q = range(0, 3)
    parameters = list(itertools.product(p, q, P, Q))
    
    # FIX: Explicitly select only PC columns, exclude Date
    exog_vars = df[[col for col in df.columns if col.startswith('PC')]]
    
    result_table = optimize_SARIMAX(parameters, d=d, D=D, s=12, endog=df['Price'], exog=exog_vars)

    if result_table.empty:
        raise ValueError("No SARIMAX models converged in optimize_SARIMAX. Try smaller search space or change data.")

    bestvals = result_table.iloc[0, 0]
    p, q, P, Q = bestvals

    best_model = SARIMAX(endog=df['Price'], exog=exog_vars,
                         order=(p, d, q),
                         seasonal_order=(P, D, Q, 12),
                         enforce_stationarity=False,
                         enforce_invertibility=False).fit(disp=False)

    # Forecast each PCA exog for the next n months
    exog_future = forecast_pca_exog(df, steps=forecast_periods)

    fitted_values = best_model.fittedvalues.copy()
    fitted_values.iloc[:max(5, int(len(fitted_values) * 0.05))] = np.NaN

    forecast = best_model.get_forecast(steps=forecast_periods, exog=exog_future)
    forecast_mean = forecast.predicted_mean

    forecast_index = pd.date_range(start=df['Date'].iloc[-1] + pd.DateOffset(months=1), periods=len(forecast_mean), freq='MS')

    plot_price(df, forecast_index, forecast_mean, fitted_values)
    # Print summary
    forecast_summary = pd.DataFrame({
        "Date": forecast_index,
        "Forecasted Price": forecast_mean
    })

    metrics = model_metrics(df,best_model)
    return ts, forecast_summary, forecast_index, forecast_mean, fitted_values, df, best_model, metrics


def Airline(airline, forecast_periods):
    df, loadings_df, explained_variance_ratio = sensitivity_index(airline)
    df['seq_index'] = range(len(df))
    ts, forecast_summary, forecast_index, forecast_mean, fitted_values, df, best_model, metrics = time_series(airline, forecast_periods)

    # Create Figures for Streamlit
    fig_forecast, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df['Date'], df['Price'], label='Actual Price', linewidth=2)  # Changed from df.index
    ax1.plot(df['Date'], fitted_values, label='Fitted', linestyle='--')  # Changed from df.index
    ax1.plot(forecast_index, forecast_mean, label='Forecast', color='red', marker='o')
    ax1.legend()
    ax1.set_title('Forecasted vs Actual Prices')
    plt.tight_layout()
    

    return ts, loadings_df, explained_variance_ratio, fig_forecast, metrics, forecast_summary


if __name__ == "__main__":
    Airline("american",8)
