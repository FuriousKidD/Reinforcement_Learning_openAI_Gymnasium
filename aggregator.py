import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def load_all_runs(data_dir: str, metric: str = "training"):
    """Loads all CSVs matching the metric type from data_dir into a list of DataFrames."""
    pattern = os.path.join(data_dir, f"*_{metric}.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No {metric} CSVs found in {data_dir}")
    return [pd.read_csv(f) for f in files], files


def align_and_aggregate(dataframes: list, column: str):
    """
    Aligns multiple run DataFrames by episode and computes
    mean and standard deviation across runs at each episode.
    Returns a DataFrame with columns: Episode, Mean, Std.
    """
    # Use the shortest run as the common length — avoids NaN at the edges
    min_len = min(len(df) for df in dataframes)
    aligned = pd.concat(
        [df[column].iloc[:min_len].reset_index(drop=True) for df in dataframes],
        axis=1
    )
    result = pd.DataFrame()
    result["Episode"] = dataframes[0]["Episode"].iloc[:min_len].reset_index(drop=True)
    result["Mean"] = aligned.mean(axis=1)
    result["Std"] = aligned.std(axis=1)
    return result


def smooth(series, window: int):
    return series.rolling(window=window, min_periods=1).mean()


def plot_variance_band(data_dir: str, column: str, title: str,
                       ylabel: str, filename: str, smooth_window: int = 20):
    """
    Loads all training runs, aggregates them, and plots a
    mean line with ±1 std deviation band.
    """
    dataframes, files = load_all_runs(data_dir, metric="training")
    print(f"Aggregating {len(files)} runs for column: {column}")

    aggregated = align_and_aggregate(dataframes, column)

    smoothed_mean = smooth(aggregated["Mean"], smooth_window)
    smoothed_std = smooth(aggregated["Std"], smooth_window)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(aggregated["Episode"], smoothed_mean, label=f"Mean {ylabel}", color="steelblue")
    ax.fill_between(
        aggregated["Episode"],
        smoothed_mean - smoothed_std,
        smoothed_mean + smoothed_std,
        alpha=0.3, color="steelblue", label="±1 Std Dev"
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def generate_aggregated_plots(data_dir: str, smooth_window: int = 20):
    """Entry point — generates all variance-band plots for a set of runs."""
    plot_variance_band(data_dir, "Total_Reward", "Mean Training Reward Across Seeds",
                       "Reward", "aggregated_reward.png", smooth_window)
    plot_variance_band(data_dir, "Steps", "Mean Episode Length Across Seeds",
                       "Steps", "aggregated_steps.png", smooth_window)
    plot_variance_band(data_dir, "Average_Loss", "Mean TD Loss Across Seeds",
                       "Loss", "aggregated_loss.png", smooth_window)
    


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "Lander_Environment", "Data")
    generate_aggregated_plots(data_dir, smooth_window=20)