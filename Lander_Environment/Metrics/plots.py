import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

def load_run(data_dir: str, run_id: str):
    training_path = os.path.join(data_dir, f"{run_id}_training.csv")
    evaluation_path = os.path.join(data_dir, f"{run_id}_evaluation.csv")
    training_df = pd.read_csv(training_path)
    evaluation_df = pd.read_csv(evaluation_path)

    return training_df, evaluation_df

def smooth(series, window: int):
    return series.rolling(window=window, min_periods=1).mean()

def plot_performance_vs_exploration(training_df, evaluation_df, run_id, save_dir, smooth_window=20):
    fig, ax1 = plt.subplots(figsize=(12,5))

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.plot(training_df["Episode"], smooth(training_df["Total_Reward"], smooth_window), label = "Training Reward (Smoothed)",
             color = "steelblue", alpha=0.9)
    ax1.plot(evaluation_df["Episode"], evaluation_df["Average_Reward"],
         label="Evaluation Reward", color="darkorange", linewidth=2, marker='o', markersize=4)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Epsilon", color="green")
    ax2.plot(training_df["Episode"], training_df["Epsilon"], label="Epsilon",
             color="green", alpha=0.4, linestyle="--")
    ax2.tick_params(axis='y', labelcolor='green')

    plt.title(f"Performance vs Exploration - {run_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{run_id}_performance.png"), dpi=150)
    plt.close()

def plot_policy_stability(training_df, run_id, save_dir, smooth_window=20):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(training_df["Episode"], smooth(training_df["Steps"], smooth_window),
            label="Episode Length (smoothed)", color="purple")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title(f"Policy Stability (Episode Length) — {run_id}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{run_id}_stability.png"), dpi=150)
    plt.close()


def plot_loss_curve(training_df, run_id, save_dir, smooth_window=20):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(training_df["Episode"], smooth(training_df["Average_Loss"], smooth_window),
            label="Avg TD Loss (smoothed)", color="crimson")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.set_title(f"TD Loss Over Training — {run_id}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{run_id}_loss.png"), dpi=150)
    plt.close()


def generate_all_plots(data_dir: str, run_id: str, smooth_window=20):
    training_df, eval_df = load_run(data_dir, run_id)
    plot_performance_vs_exploration(training_df, eval_df, run_id, data_dir, smooth_window)
    plot_policy_stability(training_df, run_id, data_dir, smooth_window)
    plot_loss_curve(training_df, run_id, data_dir, smooth_window)
    print(f"Plots saved to {data_dir}/plot_imsges")