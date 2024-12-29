from forecast.tools import *

def main():
    f = open('log/forecast_log.txt', 'w')
    while True:
        if select.select([sys.stdin], [], [], 0.1)[0]:
            line = sys.stdin.readline().strip()
            if line:
                f.write(str(time.time_ns()))
                f.write('\n')
                f.flush()
                break
        else:
            print("No input yet, still waiting...")
            time.sleep(.1)

    time.sleep(5)
    # Dictionary for timescales
    timescale_dict = {
        'year' : ('YE', 1, 3),
        'month' : ('ME', 12, 12),
        'week' : ('W', 52, 52),
        'day' : ('D', 300, 365),
        'hour' : ('h', 8338, 24*7*31)
    }

    # Data
    f.write('PID: '+str(os.getpid()))
    f.write('\n')
    f.flush()
    
    file_name = 'df_fuel_ckan.csv'
    file_path = os.path.join(os.getcwd(), 'data', file_name)
    df = pd.read_csv(file_path)
    carbon_data = pd.DataFrame(df['CARBON_INTENSITY'].values, index=pd.to_datetime(df['DATETIME']))
    #carbon_data = carbon_data.truncate(before=pd.Timestamp('2015-01-01 00:00:00').tz_localize('UTC'))
    
    models = ('holt-winter', 'sarima')
    
    for model in models:
        results = []
        for key in timescale_dict:
            (scale, forecast_periods, m) = timescale_dict[key]
            train_data, test_data = split_data(carbon_data, scale)
            forecast_periods = test_data.shape[0]

            #print('data read okay...\n')
            #print('train data:\n', train_data.head(), '\n')
            #print('test_data:\n', test_data.head())

            f.write('========================\n')
            f.write('data read okay...\n')
            f.write('------------------------\n')
            f.write('TRAIN DATA:\n'+str(train_data.head())+'\n')
            f.write('TEST DATA:\n'+str(test_data.head())+'\n')
            f.write('------------------------\n')
            f.flush()

            t0 = time.process_time_ns()
            if (model == 'holt-winter'):
                forecast = holt_winter(train_data, scale, forecast_periods, m)

            elif (model == 'sarima'):
                params = (0, 0, 0)
                seasonal_params = (0, 1, 0)
                forecast = sarima(train_data, scale, forecast_periods, m, params, seasonal_params)
            t1 = time.process_time_ns()
            log_results(forecast, scale, forecast_periods, m, model)
            t2 = time.process_time_ns()
            plot_forecast(train_data, test_data, forecast, scale, forecast_periods, m, model)
            t3 = time.process_time_ns()

            # Get process times
            model_time = t1 - t0
            log_time = t2 - t1
            plot_time = t3 - t2
            
            process_time = model_time + log_time #+ plot_time
            
            mae, mse, rmse = calculate_errors(test_data, forecast)
            results.append((mae, mse, rmse, process_time))
            f.flush()
        
        f.write(str(model)+'\n')
        f.flush()
        for (mae, mse, rmse, process_time) in results:
            f.write(f'MAE: {mae:7.2f}, MSE: {mse:7.2f}, RMSE: {rmse:7.2f}, time: {process_time*1e-6: 9.2f} ms\n')
            f.flush()

if __name__ == '__main__':
    main()
    