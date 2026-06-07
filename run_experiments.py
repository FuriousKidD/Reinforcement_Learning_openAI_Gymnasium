import subprocess
import sys
import os
from multiprocessing import Pool

# Define seeds for each run
SEEDS = [42, 123, 7, 99, 256, 13, 17, 37, 77, 200]  # 10 seeds
NUM_EPISODES = 1000
def run_training(seed: int):
    print(f"\n{'='*50}")
    print(f"Starting training run with seed: {seed}")
    print(f"{'='*50}")
    env = os.environ.copy()
    env["TRAINING_SEED"] = str(seed)
    env["NUM_EPISODES"] = str(NUM_EPISODES)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    agent_path = os.path.join(script_dir, "Lander_Environment", "Agent.py")
    subprocess.run([sys.executable, agent_path], env=env, 
                   cwd=os.path.join(script_dir, "Lander_Environment"), check=True)

if __name__ == "__main__":
    with Pool(processes=8) as pool:
        pool.map(run_training, SEEDS)

    print("\nAll runs complete. Aggregator will generate the variance plots.")