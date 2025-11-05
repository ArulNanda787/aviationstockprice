from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")

def import_and_clean_data(airline):
    df = pd.read_csv(f"{airline}.csv")
    numeric_cols = df.columns[2:]  
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(",", "")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df['Year'] = pd.to_datetime(df['Year'], format="%Y")
    df['Month'] = pd.to_datetime(df['Month'], format="%m")
    df['Date'] = pd.to_datetime(df['Year'].dt.year.astype(str) + '-' + df['Month'].dt.month.astype(str) + '-01')
    
    x = df.iloc[:, 3:].select_dtypes(include=[np.number])
    standard = StandardScaler()
    x = standard.fit_transform(x)
    return x, df

def sensitivity_index(airline):
    x, df = import_and_clean_data(airline)
    feature_cols = df.iloc[:, 3:].select_dtypes(include=[np.number]).columns
    
    pca = PCA(0.9)
    pca.fit(x)
    pca_data = pca.transform(x)
    explained_variance_ratio = pca.explained_variance_ratio_
    loadings = np.sqrt(pca.explained_variance_) * pca.components_.T
    
    loadings_df = pd.DataFrame(
        loadings, 
        columns=["PC" + str(i + 1) for i in range(len(explained_variance_ratio))],
        index=feature_cols
    )
    
    pc_df = pd.DataFrame(pca_data, columns=["PC" + str(i + 1) for i in range(pca_data.shape[1])])
    df_out = pd.DataFrame({
        "Date": df["Date"],
        "Price": df["Price"]
    })
    for col in pc_df.columns:
        df_out[col] = pc_df[col].values

    df_out = df_out.sort_values("Date").reset_index(drop=True)
    return df_out, loadings_df, explained_variance_ratio

def forecast_pca_exog(df, steps=8):
    exog = df[[col for col in df.columns if col.startswith("PC")]]
    future_pcs = []
    for col in exog.columns:
        model_pc = auto_arima(
            exog[col],
            seasonal=True,
            m=12,
            trace=False,
            error_action="ignore",
            suppress_warnings=True
        )
        pc_forecast = model_pc.predict(n_periods=steps)
        future_pcs.append(pc_forecast)
    exog_future = np.column_stack(future_pcs)
    return exog_future


def prophet_forecast(df, forecast_periods):
    prophet_df = df[["Date", "Price"]].rename(columns={"Date": "ds", "Price": "y"})
    pcs = df[[col for col in df.columns if col.startswith("PC")]]
    
    #Add PCs as regressors
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    for col in pcs.columns:
        m.add_regressor(col)

    m.fit(pd.concat([prophet_df, pcs], axis=1))
    
    future = m.make_future_dataframe(periods=forecast_periods, freq="MS")
    
    #Forecast exogenous regressors (PCs)
    exog_future = forecast_pca_exog(df, forecast_periods)
    future_pcs = pd.DataFrame(exog_future, columns=pcs.columns)
    future_pcs_full = pd.concat([pcs, future_pcs], ignore_index=True)
    
    future_full = pd.concat([future, future_pcs_full], axis=1)
    
    forecast = m.predict(future_full)
    
    fig1 = m.plot(forecast)
    plt.title("Prophet Forecast with PCA Regressors")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()
    
    y_true = prophet_df["y"]
    y_pred = forecast["yhat"][:len(y_true)]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    
    forecast_summary = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_periods)
    forecast_summary.rename(columns={"ds": "Date", "yhat": "Forecasted Price"}, inplace=True)
    
    return m, forecast, metrics, forecast_summary, fig1


def Airline_Prophet(airline, forecast_periods=8):
    df, loadings_df, explained_variance_ratio = sensitivity_index(airline)
    model, forecast, metrics, forecast_summary, fig = prophet_forecast(df, forecast_periods)
    return df, loadings_df, explained_variance_ratio, model, forecast, metrics, forecast_summary, fig

if __name__ == "__main__":
    df, loadings_df, explained_variance_ratio, model, forecast, metrics, forecast_summary, fig = Airline_Prophet("american", 8)
    print("Metrics:", metrics)
    print(forecast_summary)
