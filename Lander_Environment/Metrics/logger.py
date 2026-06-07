import os
import csv
from datetime import datetime
import numpy as np

class AgentLogger:
    def __init__(self, run_name: str, save_dir: str, seed: int):
        """Logs and saves DQNAgent training and evaluation metrics """
        self.run_name = run_name
        self.save_dir = save_dir
        self.seed = seed

        # list of dicts, one per episode
        self.training_logs = [] 

        # list of dicts, one per evaluation
        self.evaluation_logs = []

        # Unique identifier for this run's files
        timestamp = datetime.now().strftime("%Y%m%d")
        self.run_id =  f"{run_name}_{timestamp}_seed{seed}"

        os.makedirs(save_dir, exist_ok=True)

    def log_episode(self, episode, total_reward, steps, epsilon, learning_rate, episode_losses, action_percentages):
        """ """
        average_loss = np.mean(episode_losses) if episode_losses else 0.0
        entry = {"Episode": episode, "Total_Reward": total_reward, 
                                   "Steps": steps, "Epsilon": epsilon,
                                     "Learning_Rate": learning_rate, "Average_Loss": average_loss}
        
        for i, percentage in enumerate(action_percentages):
            entry[f"Action_{i}_Percentage"] = percentage
        self.training_logs.append(entry)

    def log_evaluation(self, episode, average_reward, max_reward, min_reward):
        self.evaluation_logs.append({"Episode":episode, "Average_Reward": average_reward,
                                     "Maximum_Reward": max_reward, "Minimum_Reward": min_reward})

    def save(self):
        if self.training_logs:
            filepath = os.path.join(self.save_dir, f"{self.run_id}_training.csv")
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(self.training_logs[0].keys()))
                writer.writeheader()
                writer.writerows(self.training_logs)

        if self.evaluation_logs:
            filepath = os.path.join(self.save_dir, f"{self.run_id}_evaluation.csv")
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(self.evaluation_logs[0].keys()))
                writer.writeheader()
                writer.writerows(self.evaluation_logs) 
