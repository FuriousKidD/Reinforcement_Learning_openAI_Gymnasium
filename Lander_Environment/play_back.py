import numpy as np
import gymnasium as gym
import pickle
import os
import sys

sys.path.append(os.path.dirname(__file__))
from Neural_Network_package import modular_mlp as mlp


def load_model(filepath: str):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def play(model_path: str, env_name: str = "LunarLander-v3", num_episodes: int = 10, render: bool = True):
    network = load_model(model_path)

    render_mode = "human" if render else "rgb_array"
    env = gym.make(env_name, render_mode=render_mode)

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Greedy action selection — no exploration
            state_input = np.array(state).reshape(1, -1)
            q_values = network.forward_pass(state_input)
            action = int(np.argmax(q_values))

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated

        print(f"Episode {episode + 1}: Reward = {total_reward:.2f} | Steps = {steps}")

    env.close()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":
    models = {
        "0": ("Untrained (random)", os.path.join(BASE_DIR, "Data_lr0001/untrained_model.pkl")),
        "1": ("Seed 99 (best model)", os.path.join(BASE_DIR, "Data_lr0001/best_model_Iteration_20260525_seed99.pkl")),
        "2": ("Seed 17 (best model)", os.path.join(BASE_DIR, "Data_lr0001/best_model_Iteration_20260525_seed17.pkl")),
        "3": ("Seed 42 (best model)", os.path.join(BASE_DIR, "Data_lr0001/best_model_Iteration_20260525_seed42.pkl")),
        "4": ("Seed 7 (best model)", os.path.join(BASE_DIR, "Data_lr0001/best_model_Iteration_20260525_seed7.pkl")),
        "5": ("Seed 77 (best model)", os.path.join(BASE_DIR, "Data_lr0001/best_model_Iteration_20260525_seed77.pkl")),
        
    }

    print("\nSelect model to demo:")
    for key, (label, _) in models.items():
        print(f"  [{key}] {label}")
    print("\n Q: Quit" )
    
    quit = False
    while not quit:
        choice = input("\nEnter choice: ").strip()
        if choice.lower() == "q":
            quit = True
        elif choice not in models:
            print("Invalid choice.")
        else:
            label, path = models[choice]
            print(f"\nRunning: {label}")
            play(path, env_name="LunarLander-v3", num_episodes=10, render=True)