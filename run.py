import subprocess
import time
import psutil

def run_a(stderr_file):
    process_a = subprocess.Popen(['sudo', 'python3', 'forecast.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_file, shell=False)
    return process_a

# Function to run b.py
def run_b(stderr_file):
    process_b = subprocess.Popen(['sudo', 'python3', 'power_perf/power_perf.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_file)
    return process_b

# Open the error log file
with open('log/error.txt', 'w') as error_file:
    # Run a.py and get its PID
    print("Forecast ready...")
    process_a = run_a(error_file)

    # Run b.py
    print("Monitor ready...")
    process_b = run_b(error_file)

    # Get the child processes of process_a
    parent = psutil.Process(process_a.pid)
    time.sleep(1)
    children = parent.children()
    child = children[0]
    print(f"monitoring process: {child.pid}")

    process_b.stdin.write(f"{child.pid}\n".encode())
    process_b.stdin.flush()
    print('Monitor STARTED')

    time.sleep(1)

    process_a.stdin.write(f"execute\n".encode())
    process_a.stdin.flush()
    print('Forecast STARTED')
    
    process_a.wait()
    process_b.kill()
    #process_b.wait()