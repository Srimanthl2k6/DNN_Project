import os
import sys
import time
import subprocess

def watch_directory():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoints_dir = os.path.join(root, 'checkpoints')
    eval_script = os.path.join(root, 'src', 'evaluate_auto.py')
    
    evaluated_checkpoints = set()
    print(f"Watching for checkpoints in {checkpoints_dir}...")
    
    while True:
        if os.path.exists(checkpoints_dir):
            for file in os.listdir(checkpoints_dir):
                if file.endswith('.pth') and file not in evaluated_checkpoints:
                    print(f"\nNew checkpoint detected: {file}")
                    try:
                        # evaluate_auto.py will evaluate un-evaluated checkpoints and append to results.csv
                        subprocess.run([sys.executable, eval_script], check=True)
                        evaluated_checkpoints.add(file)
                        print(f"Successfully processed {file}.")
                    except subprocess.CalledProcessError as e:
                        print(f"Error evaluating {file}: {e}")
        time.sleep(5)

if __name__ == '__main__':
    watch_directory()
