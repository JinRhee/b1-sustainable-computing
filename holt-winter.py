import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

def split_data(data: pd.DataFrame, timescale: str):
    timescale_carbon = data.resample(timescale).mean()
    train_data = timescale_carbon.truncate(after=pd.Timestamp('2023-12-31 23:59:59').tz_localize('UTC'))
    test_data = timescale_carbon.truncate(before=pd.Timestamp('2024-01-01 00:00:00').tz_localize('UTC'))
    return train_data, test_data

def forecast_holt_winter(data: pd.DataFrame, forecast_periods: int):
    # Create an instance of ExponentialSmoothing class
    model_triple = ExponentialSmoothing(
        data, seasonal_periods=365, trend='add', damped_trend=True, seasonal='add')

    # Fit the model to the data
    model_triple_fit = model_triple.fit()

    return model_triple_fit.forecast(forecast_periods)

def calculate_errors(observed, forecast_mean):
    mae = mean_absolute_error(observed, forecast_mean)
    mse = mean_squared_error(observed, forecast_mean)
    rmse = root_mean_squared_error(observed, forecast_mean)
    return mae, mse, rmse

def plot_forecast(train_data, test_data, forecast):
    # Plot the forecast
    plt.figure(figsize=(20, 7))
    plt.plot(train_data, 'bo-', label='Observed')
    plt.plot(forecast, 'rx-', label='Forecast')
    plt.plot(test_data, 'ko--', label='Actual')
    plt.title("Carbon Intensity Forecast for 2024")
    plt.xlabel("Date")
    plt.ylabel("Carbon Intensity")
    plt.legend()
    plt.show()

def main():
    # Dictionary for timescales
    timescale_dict = {
        'year' : ('YE', 1, 2),
        'month' : ('ME', 12, 12),
        'day' : ('D', 348, 365),
        'hour' : ('h', 24*7, 24*7)
    }
    (scale, forecast_periods, m) = timescale_dict['day']
    
    # Data
    file_name = 'df_fuel_ckan.csv'
    file_path = os.path.join(os.getcwd(), 'data', file_name)
    df = pd.read_csv(file_path)
    carbon_data = pd.DataFrame(df['CARBON_INTENSITY'].values, index=pd.to_datetime(df['DATETIME']))
    #carbon_data = carbon_data.truncate(before=pd.Timestamp('2015-01-01 00:00:00').tz_localize('UTC'))

    train_data, test_data = split_data(carbon_data, scale)
    print('data read okay...')
    print('train data:\n', train_data.head(), '\n')
    print('test_data:\n', test_data.head())

    # Forecasting from Holt-Winter
    print('\nforecasting...')
    forecast = forecast_holt_winter(train_data, forecast_periods)
    print('\nforecasting okay...')

    mae, mse, rmse = calculate_errors(test_data, forecast)
    print('\nerror calculation okay...')

    results = []
    forecasts = []
    results.append(rmse)
    forecasts.append(forecast)
    
    # Print results
    plot_forecast(train_data, test_data, forecasts[results.index(min(results))])

if __name__ == '__main__':
    main()