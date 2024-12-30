from forecast.tools import *
import os
import argparse

parser = argparse.ArgumentParser(description='forecasting tool')
parser.add_argument('--model', type=str, nargs='+', default=['holt-winter'],
                    help='Forecasting model to use')
parser.add_argument('--timescale', type=str, nargs='+', default=['hour'],
                    help='Forecasting timescale to use')
parser.add_argument('--wait', type=int, default=0,
                    help='Wait for stdin input (default=0)')
parser.add_argument('--verbose', type=int, default=0,
                    help='Enable verbose timing (default=0)')

args = parser.parse_args()

def main():
    t_start = time.process_time_ns()
    print('starting with...')
    print(args.model, args.timescale, args.wait)
    f = open(f'log/forecast_log{os.getpid()}.txt', 'w')
    f.write(f'{args.model}, {args.timescale}, {args.wait}')

    if args.wait:
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
    
    for model in args.model:                                            # Remove if just calling one model
        results = []
        for key in args.timescale:
            t0 = time.process_time_ns()
            (scale, forecast_periods, m) = timescale_dict[key]
            train_data, test_data = split_data(carbon_data, scale)      # Data spltting happens too many times..?
            forecast_periods = test_data.shape[0]

            f.write('========================\n')
            f.write('data read okay...\n')
            f.write('------------------------\n')
            f.write('TRAIN DATA:\n'+str(train_data.head())+'\n')
            f.write('TEST DATA:\n'+str(test_data.head())+'\n')
            f.write('------------------------\n')
            f.flush()

            t1 = time.process_time_ns()
            if (model == 'holt-winter'):
                forecast = holt_winter(train_data, scale, forecast_periods, m)

            elif (model == 'sarima'):
                params = (0, 0, 0)
                seasonal_params = (0, 1, 0)
                forecast = sarima(train_data, scale, forecast_periods, m, params, seasonal_params)
                
            t2 = time.process_time_ns()
            log_results(forecast, scale, forecast_periods, m, model)
            t3 = time.process_time_ns()
            plot_forecast(train_data, test_data, forecast, scale, forecast_periods, m, model)
            t4 = time.process_time_ns()
            mae, mse, rmse = calculate_errors(test_data, forecast)
            t5 = time.process_time_ns()

            # Get process times
            setup_time = t1 - t0
            model_time = t2 - t1
            log_time = t3 - t2
            plot_time = t4 - t3
            error_time = t5 - t4

            results.append((mae, mse, rmse, setup_time, model_time, log_time, plot_time, error_time))
            f.flush()
        
        f.write(str(model)+'\n')
        f.flush()

        for (mae, mse, rmse, setup_time, model_time, log_time, plot_time, error_time) in results:
            f.write(f'MAE: {mae:7.2f}, MSE: {mse:7.2f}, RMSE: {rmse:7.2f}')
            if args.verbose:
                f.write('\n')
                f.write(f'setup: {setup_time*1e-6: 11.4f} ms\n')
                f.write(f'model: {model_time*1e-6: 11.4f} ms\n')
                f.write(f'log  : {log_time*1e-6: 11.4f} ms\n')
                f.write(f'plot : {plot_time*1e-6: 11.4f} ms\n')
                f.write(f'error: {error_time*1e-6: 11.4f} ms\n')
                f.write(f'total: {(setup_time+model_time+log_time+plot_time+error_time)*1e-6: 11.4f} ms\n')
            else:
                f.write(f', time: {(setup_time+model_time+log_time+plot_time+error_time)*1e-6: 11.4f} ms\n')

            f.flush()

    t_end = time.process_time_ns()
    t_total = t_end - t_start
    return t_total

if __name__ == '__main__':
    total_time = main()
    print(f'total: {total_time*1e-6: 9.2f} ms')
    