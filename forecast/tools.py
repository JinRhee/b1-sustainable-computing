import os
import numpy as np
import pandas as pd
import time
import sys
import select
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from river import time_series
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from pmdarima.arima import auto_arima

# Check stationarity (from https://www.geeksforgeeks.org/sarima-seasonal-autoregressive-integrated-moving-average/)
def check_stationarity(timeseries):
    # Perform the Dickey-Fuller test
    result = adfuller(timeseries, autolag='AIC')
    p_value = result[1]
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {p_value}')
    print('Stationary' if p_value < 0.05 else 'Non-Stationary')

def diff(timeseries, num):
    diff_timeseries = timeseries.diff(periods=num)
    diff_timeseries.dropna(inplace=True)
    return diff_timeseries

def diff_n(timeseries, num, n):
    # carry out diff() n times
    for _ in range(n):
        diff_timeseries = diff(timeseries, num)
    return diff_timeseries

def split_data(data: pd.DataFrame, timescale: str):
    timescale_carbon = data.resample(timescale).mean()
    train_data = timescale_carbon.truncate(after=pd.Timestamp('2023-12-31 23:59:59').tz_localize('UTC'))
    test_data = timescale_carbon.truncate(before=pd.Timestamp('2024-01-01 00:00:00').tz_localize('UTC'))
    return train_data, test_data

def forecast_SARIMA_river(train_data, forecast_periods, params, seasonal_params):
    (p, d, q) = params
    (P, D, Q, m) = seasonal_params
    model = time_series.SNARIMAX(p, d, q, m, P, D, Q)
    for intensity in train_data.to_numpy():
        model.learn_one(intensity)
    forecast = model.forecast(horizon = forecast_periods)

    return forecast

def add_dates(data, start_date, timescale):
    date_range = pd.date_range(start=start_date, periods=len(data), freq=timescale)
    return pd.DataFrame(data, index=date_range)

def forecast_SARIMA(train_data, forecast_periods, params, seasonal_params):
    ## Define SARIMA parameters
    # SARIMA(0,1,1)(0,1,1)12 model
    #p, d, q = 0, 1, 1  
    #P, D, Q, s = 0, 1, 1, 12

    # Fit the SARIMA model
    model = SARIMAX(train_data, order=params, seasonal_order=seasonal_params)
    results = model.fit()

    forecast = results.get_forecast(steps=forecast_periods)
    return forecast

def forecast_SARIMA_auto(train_data, forecast_periods, params, seasonal_params):
    model = auto_arima(train_data, start_p=0, d=1, start_q=0, max_p=5, max_d=5, max_q=5,
    start_P=0, D=1, start_Q=0, max_P=5, max_D=5,
    max_Q=5, m=365, seasonal=True,
    stationary=False,
    error_action='warn', trace=True,
    suppress_warnings=True, stepwise=True,
    random_state=20, n_fits=50)
    print(model.aic())
    return

def calculate_errors(observed, forecast_mean):
    mae = mean_absolute_error(observed, forecast_mean)
    mse = mean_squared_error(observed, forecast_mean)
    rmse = root_mean_squared_error(observed, forecast_mean)
    return mae, mse, rmse

def plot_forecast(train_data, test_data, forecast, scale, forecast_periods, m, method):
    # Plot the forecast
    plt.figure(figsize=(20, 7))
    plt.plot(train_data, 'bo-', label='Observed')
    plt.plot(forecast, 'rx-', label='Forecast')
    plt.plot(test_data, 'ko--', label='Actual')
    #plt.fill_between(forecast.conf_int().index, forecast.conf_int().iloc[:, 0], forecast.conf_int().iloc[:, 1], color='pink')
    plt.title(f"Carbon Intensity Forecast for 2024 in {scale}")
    plt.xlabel("Date")
    plt.ylabel("Carbon Intensity")
    plt.legend()
    plt.savefig(f"plots/{scale}_{forecast_periods}_{m}_{method}.png")
    #plt.show()

def main_single():
    print(os.getpid())
    
    # Dictionary for timescales
    timescale_dict = {
        'year' : ('YE', 1, 5)#,
        #'month' : ('ME', 12, 12),
        #'week' : ('W', 50, 52),
        #'day' : ('D', 348, 365),
        #'hour' : ('h', 8338, 24*7*31)
    }
    (scale, forecast_periods, m) = timescale_dict['hour']
    
    # Data
    file_name = 'df_fuel_ckan.csv'
    file_path = os.path.join(os.getcwd(), 'data', file_name)
    df = pd.read_csv(file_path)
    carbon_data = pd.DataFrame(df['CARBON_INTENSITY'].values, index=pd.to_datetime(df['DATETIME']))
    #carbon_data = carbon_data.truncate(before=pd.Timestamp('2016-01-01 00:00:00').tz_localize('UTC'))

    train_data, test_data = split_data(carbon_data, scale)
    print('data read okay...')
    print('train data:\n', train_data.head(), '\n')
    print('test_data:\n', test_data.head())

    params = (0, 0, 0)                  # p, d, q
    seasonal_params = (0, 1, 0, 24*7*365)      # P, D, Q, m

    results = []
    forecasts = []
    states = (0, 1)
    grid_search = False
    if (grid_search):
        for p in states:
            for d in states:
                for q in states:
                    for P in states:
                        for D in states:
                            for Q in states:
                                params = (p, d, q)
                                seasonal_params = (P, D, Q, m)
                                forecast = forecast_SARIMA_river(train_data, forecast_periods, params, seasonal_params)
                                forecast = add_dates(forecast, pd.Timestamp('2024-01-01 00:00:00').tz_localize('UTC'), scale)
                                mae, mse, rmse = calculate_errors(test_data, forecast)
                                results.append((mae, params, seasonal_params))
                                print(params, seasonal_params, f'MAE: {mae:3.2f} MSE: {mse:3.2f} RMSE: {rmse:3.2f}')
                                plot_forecast(test_data, test_data, forecast)
                                forecasts.append(forecast)
    else:
        #forecast = forecast_SARIMA_river(train_data, forecast_periods, params, seasonal_params)
        forecast = forecast_expo_smooth(train_data, forecast_periods, 24*7*31)
        forecast = add_dates(forecast, pd.Timestamp('2024-01-01 00:00:00').tz_localize('UTC'), scale)
        forecast.dropna(inplace=True)
        print(forecast)
        mae, mse, rmse = calculate_errors(test_data, forecast)
        results.append((mae, params, seasonal_params))
        print(params, seasonal_params, f'MAE: {mae:3.2f} MSE: {mse:3.2f} RMSE: {rmse:3.2f}')
        forecasts.append(forecast)

    for result in results:
        print(result[0])
        print()
    
    # Print results
    print(min(results, key=lambda x: x[0]))
    plot_forecast(train_data, test_data, forecasts[results.index(min(results, key=lambda x: x[0]))])

def holt_winter(train_data, scale, forecast_periods, m):
    # Fit the Exponential Smoothing model
    model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=m)
    model_fit = model.fit()

    # Forecast for 2024 and add dates
    forecast = model_fit.forecast(steps=forecast_periods)
    forecast = add_dates(forecast, pd.Timestamp('2024-01-01 00:00:00').tz_localize('UTC'), scale)
    forecast.dropna(inplace=True)

    return forecast

def sarima(train_data, scale, forecast_periods, m, params, seasonal_params):
    # Unpack arguments
    (p, d, q) = params
    (P, D, Q) = seasonal_params

    # Fit the SNARIMAX model
    model = time_series.SNARIMAX(p, d, q, m, P, D, Q)
    for intensity in train_data.to_numpy():
        model.learn_one(intensity)
    
    # Forecast for 2024 and add dates
    forecast = model.forecast(horizon = forecast_periods)
    forecast = add_dates(forecast, pd.Timestamp('2024-01-01 00:00:00').tz_localize('UTC'), scale)
    forecast.dropna(inplace=True)

    return forecast

def log_results(forecast, scale, forecast_periods, m, method):
    forecast.to_csv(f'results/{scale}_{forecast_periods}_{m}_{method}.csv', header=['FORECAST_INTENSITY'], index_label=['DATETIME'])

