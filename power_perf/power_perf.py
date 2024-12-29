import time
import argparse
from collections import deque
from utils import *
import sys
import select
import signal
import json

parser = argparse.ArgumentParser(
    description='asitop: Performance monitoring CLI tool for Apple Silicon')
parser.add_argument('--interval', type=float, default=100,
                    help='Display interval and sampling interval for powermetrics (ms)')
parser.add_argument('--color', type=int, default=2,
                    help='Choose display color (0~8)')
parser.add_argument('--avg', type=int, default=30000,
                    help='Interval for averaged values (ms)')
parser.add_argument('--show_cores', type=bool, default=False,
                    help='Choose show cores mode')
parser.add_argument('--max_count', type=int, default=0,
                    help='Max show count to restart powermetrics')
parser.add_argument('--process_pid', type=int, default=0,
                    help='Process PID to monitor')
parser.add_argument('--process-gpid', type=int, default=0,
                    help='Process PID to monitor')
parser.add_argument('--wait', type=int, default=0,
                    help='Wait for stdin input (default=0)')

args = parser.parse_args()

def main():
    #print(os.getpid())
    f = open('log/log.txt', 'w')
    flog = open('log/flog.txt', 'w')
    f.write("[1/3] Waiting for process PID\n")
    f.write("Waiting for input from stdin...\n")
    f.flush()

    if args.wait:
        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                line = sys.stdin.readline().strip()
                if line:
                    args.process_pid = int(line)#+1
                    break
            else:
                print("No input yet, still waiting...")
                time.sleep(.1)
                
    #args.process_pid = 389
    f.write(f"\n Searching process: {args.process_pid}")
    f.write("\n[2/3] Starting powermetrics process\n")
    timecode = str(int(time.time()))

    #powermetrics_process = run_powermetrics_process(timecode,
    #                                                interval=args.interval)

    powermetrics_process = run_powermetrics_process_manual(timecode,
                                                    interval=args.interval)
    
    f.write("\n[3/3] Waiting for first reading...\n")
    def get_reading(wait=1e-3):
        powermetrics_process.send_signal(signal.SIGINFO)
        ready = parse_powermetrics(timecode=timecode)
        while not ready:
            time.sleep(wait)
            ready = parse_powermetrics(timecode=timecode)
        return ready

    ready = get_reading()
    last_timestamp = ready[-1]
    last_timestamp_ns = time.time_ns()
    
    cumulative_intensity = 0

    count = 0
    try:
        # Write starting time
        f.write(str(time.time_ns())+'\n')
        f.flush()
        
        # Get energy consumption and carbon emission (valid for 30 mins)
        carbon_intensity = fetch_carbon_intensity()     # gCO2/kWh
        actual_carbon_intensity = carbon_intensity['data'][0]['intensity']['actual']
        
        while True:
            t0 = time.time_ns()
            # Force sample and parse
            powermetrics_process.send_signal(signal.SIGINFO)
            ready = parse_powermetrics(timecode=timecode)
            count += 1
            if ready:
                cpu_metrics_dict, gpu_metrics_dict, thermal_pressure, bandwidth_metrics, process_dicts, timestamp = ready
                timestamp_ns = time.time_ns()
                if timestamp_ns > last_timestamp_ns:
                    # Timing
                    delta_t = (timestamp_ns - last_timestamp_ns) / 1e6
                    last_timestamp = timestamp
                    last_timestamp_ns = timestamp_ns

                    f.write('--------------------------\n')
                    f.write(str(timestamp)+' | '+str(delta_t)+' | '+str(count)+'\n')
                    f.flush()
                    print(str(timestamp) + ' | +' + str(delta_t) + ' | ' + str(count))

                    # Power calculations
                    cpu_power_W = cpu_metrics_dict["cpu_W"] / args.interval
                    gpu_power_W = cpu_metrics_dict["gpu_W"] / args.interval

                    all_process_power = get_power(process_dicts, "-2")
                    process_power = get_power(process_dicts, args.process_pid)

                    # If process is not found
                    if (process_power == all_process_power):
                        print(f"could not find process with PID: {args.process_pid}\n")
                        f.write(f"could not find process with PID: {args.process_pid}\n")
                        f.flush()

                    else:
                        # Calculations
                        process_power_estimate = float(cpu_power_W * (process_power / all_process_power))
                        process_energy_estimate = float((delta_t) / (1000*60*60) * process_power_estimate / 1e3)

                        process_carbon_intensity_estimate = float(process_energy_estimate * actual_carbon_intensity)
                        cumulative_intensity += process_carbon_intensity_estimate

                        print(f"CPU: {cpu_power_W:5.3f}[W] | GPU: {gpu_power_W:5.3f}[W]")
                        print(f"all_process: {all_process_power:9.3f}")
                        print(f"{args.process_pid:11}: {process_power:9.3f}")
                        print(f"process_P  : {process_power_estimate:9.3e} [W]")
                        print(f"delta_t    : {delta_t:9.3e} [ms]")
                        print(f"process_E  : {process_energy_estimate:9.3e} [kWh]")
                        print(f"carbon_int : {actual_carbon_intensity:9.3e} [gCO2/kWh]")
                        print(f"process_int: {process_carbon_intensity_estimate:9.3e} [gCO2]")
                        print(f"cumulative : {cumulative_intensity:9.3e} [gCO2]")
                        
                        f.write(f"CPU: {cpu_power_W:5.3f}[W] | GPU: {gpu_power_W:5.3f}[W]\n")
                        f.write(f"all_process: {all_process_power:9.3f}\n")
                        f.write(f"{args.process_pid:11}: {process_power:9.3f}\n")
                        f.write(f"process_P  : {process_power_estimate:9.3e} [W]\n")
                        f.write(f"delta_t    : {delta_t:9.3e} [ms]\n")
                        f.write(f"process_E  : {process_energy_estimate:9.3e} [kWh]\n")
                        f.write(f"carbon_int : {actual_carbon_intensity:9.3e} [gCO2/kWh]\n")
                        f.write(f"process_int: {process_carbon_intensity_estimate:9.3e} [gCO2]\n")
                        f.write(f"cumulative : {cumulative_intensity:9.3e} [gCO2]\n")
                        f.flush()
                        
            t_end = time.time_ns()
            #print((t_end-t0)/1e6)
            time.sleep(args.interval / 1e3)

    except KeyboardInterrupt:
        f.close()
        flog.close()
        print("Stopping...")
        print("\033[?25h")
    
    return powermetrics_process

if __name__ == '__main__':
    powermetrics_process = main()
    try:
        powermetrics_process.terminate()
        print("Successfully terminated powermetrics process")
    except Exception as e:
        print(e)
        powermetrics_process.terminate()
        print("Successfully terminated powermetrics process")