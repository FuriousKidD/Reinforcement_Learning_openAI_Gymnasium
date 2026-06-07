"""
DQN LunarLander Pipeline
========================
This file is to execute the full pipeline in the correct order.
Each step will prompt the user before proceeding.
"""

import subprocess
import sys
import os
import pickle


def prompt(message):
    response = input("\n" + message + " [y/n]: ").strip().lower()
    return response == 'y'


def run(label, command, cwd=None):
    print("\n" + "="*55)
    print("  " + label)
    print("="*55)
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        print("\n[ERROR] Step failed with exit code " + str(result.returncode))
        sys.exit(1)
    print("\n[DONE] " + label)


def generate_untrained_model(save_dir):
    """Creates a freshly initialised (untrained) model and saves it."""
    lander_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Lander_Environment")
    sys.path.insert(0, lander_dir)
    from Neural_Network_package import modular_mlp as mlp

    config = [
        (8, 128, 'relu', 'xavier'),
        (128, 128, 'relu', 'xavier'),
        (128, 4, 'identity', 'xavier')
    ]
    network = mlp.MLP(config)
    network.layer_assembly()

    save_path = os.path.join(save_dir, "untrained_model.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(network, f)
    print("[SAVED] Untrained model -> " + save_path)
    return save_path


def pick_model(data_dir, label):
    """Lists all PKL models in data_dir and lets the user pick one."""
    candidates = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".pkl")
    ])
    if not candidates:
        print("[ERROR] No .pkl models found in " + data_dir)
        sys.exit(1)
    print("\nAvailable models for " + label + ":")
    for i, c in enumerate(candidates):
        print("  [" + str(i) + "] " + c)
    choice = input("Enter model number: ").strip()
    return os.path.join(data_dir, candidates[int(choice)])


def run_demo(model_path, lander_dir):
    """Launches play_back.py with the given model path."""
    env = os.environ.copy()
    env["DEMO_MODEL_PATH"] = model_path
    result = subprocess.run(
        [sys.executable, os.path.join(lander_dir, "play_back.py")],
        cwd=lander_dir,
        env=env
    )
    if result.returncode != 0:
        print("[ERROR] Demo failed.")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lander_dir = os.path.join(script_dir, "Lander_Environment")

    # Ask user which Data folder to use
    print("\n" + "="*55)
    print("  DQN LUNARLANDER - FULL PIPELINE")
    print("="*55)
    print("""
Steps:
  1. Train  - run_experiments.py  (all seeds, parallel)
  2. Plots  - aggregator.py       (variance band plots)
  3. Demo   - play_back.py        (render agent)

You will be prompted before each step.
""")

    print("Which Data folder should this pipeline use?")
    print("  [0] Lander_Environment/Data         (current/new runs)")
    print("  [1] Lander_Environment/Data_lr0005  (saved lr=0.005 runs)")
    folder_choice = input("\nEnter choice [0/1]: ").strip()
    if folder_choice == "1":
        data_dir = os.path.join(lander_dir, "Data_lr0005")
    else:
        data_dir = os.path.join(lander_dir, "Data")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("[INFO] Created folder: " + data_dir)

    print("\n[USING] " + data_dir)

    # Step 1: Training
    if prompt("Step 1: Run training across all seeds? (takes ~70 min with 8 cores)"):
        run(
            "Step 1 - Training all seeds",
            [sys.executable, os.path.join(script_dir, "run_experiments.py")],
            cwd=script_dir
        )
    else:
        print("[SKIPPED] Training.")

    # Step 2: Aggregation
    if prompt("Step 2: Generate aggregated variance band plots?"):
        run(
            "Step 2 - Aggregating results and generating plots",
            [sys.executable, os.path.join(script_dir, "aggregator.py")],
            cwd=script_dir
        )
    else:
        print("[SKIPPED] Aggregation.")

    # Step 3: Demo
    if prompt("Step 3: Launch agent demo?"):

        print("\nDemo mode:")
        print("  [1] Single model demo")
        print("  [2] Three-model comparison (untrained vs worst vs best)")
        demo_mode = input("\nEnter choice [1/2]: ").strip()

        if demo_mode == "2":

            # Generate untrained model if it doesn't exist
            untrained_path = os.path.join(data_dir, "untrained_model.pkl")
            if not os.path.exists(untrained_path):
                print("\n[INFO] No untrained model found. Generating one now...")
                untrained_path = generate_untrained_model(data_dir)
            else:
                print("\n[FOUND] Untrained model: " + untrained_path)

            print("\n--- Select your WORST seed model ---")
            worst_path = pick_model(data_dir, "worst seed")

            print("\n--- Select your BEST seed model ---")
            best_path = pick_model(data_dir, "best seed")

            demos = [
                ("Model 1 of 3 - Untrained agent (random behaviour)", untrained_path),
                ("Model 2 of 3 - Worst seed best model", worst_path),
                ("Model 3 of 3 - Best seed best model", best_path),
            ]

            for label, path in demos:
                input("\nPress Enter to start: " + label)
                print("[RUNNING] " + path)
                run_demo(path, lander_dir)

        else:
            model_path = pick_model(data_dir, "demo")
            run_demo(model_path, lander_dir)

    else:
        print("[SKIPPED] Demo.")

    print("\n" + "="*55)
    print("  PIPELINE COMPLETE")
    print("  Results folder: " + data_dir)
    print("="*55 + "\n")


if __name__ == "__main__":
    main()