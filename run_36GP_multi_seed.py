import os
import csv
import time
import shutil
import importlib.util
from pathlib import Path
from typing import Optional
import numpy as np
from scipy.stats.qmc import LatinHypercube


THIS_DIR = Path(__file__).resolve().parent
GEN_FILE = THIS_DIR / "36GP_generate_initial_data_snap.py"
BO_FILE = THIS_DIR / "36GP_bayesian_optimization_main_discrete_snap.py"


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_seeded_lhs_function(seed: int):
    def generate_lhs_samples(num_samples, bounds):
        dimensions = len(bounds)
        sampler = LatinHypercube(d=dimensions, seed=seed)
        lhs_points = sampler.random(n=num_samples)

        bounds_arr = np.array(bounds)
        lower_bounds = bounds_arr[:, 0]
        upper_bounds = bounds_arr[:, 1]

        scaled_points = lower_bounds + (upper_bounds - lower_bounds) * lhs_points
        return scaled_points

    return generate_lhs_samples


def move_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))


def run_one_seed(seed: int, n_init: int = 4000, n_iter: int = 100, results_root: Optional[Path] = None):
    if results_root is None:
        results_root = THIS_DIR / "multi_seed_results"

    run_dir = results_root / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(THIS_DIR)

    initial_mod = load_module(f"init_mod_seed_{seed}", GEN_FILE)
    bo_mod = load_module(f"bo_mod_seed_{seed}", BO_FILE)

    np.random.seed(seed)
    initial_mod.np.random.seed(seed)
    bo_mod.np.random.seed(seed)

    initial_mod.generate_lhs_samples = make_seeded_lhs_function(seed)

    original_de = bo_mod.differential_evolution

    def seeded_differential_evolution(*args, **kwargs):
        kwargs.setdefault("seed", seed)
        return original_de(*args, **kwargs)

    bo_mod.differential_evolution = seeded_differential_evolution

    captured_history = {"best_y_history": None}
    original_plot = bo_mod.plt.plot

    def capturing_plot(*args, **kwargs):
        if args:
            y = None
            if len(args) >= 2:
                y = args[1]
            elif len(args) == 1:
                y = args[0]
            if y is not None:
                try:
                    arr = np.asarray(y, dtype=float).reshape(-1)
                    if arr.size > 0:
                        captured_history["best_y_history"] = arr.copy()
                except Exception:
                    pass
        return original_plot(*args, **kwargs)

    bo_mod.plt.plot = capturing_plot

    initial_data_path = run_dir / f"initial_data_seed_{seed}.npz"
    plot_src = THIS_DIR / "best_weight_vs_iteration_36GP.png"
    plot_dst = run_dir / f"best_weight_vs_iteration_seed_{seed}.png"
    log_path = run_dir / f"run_seed_{seed}.txt"

    t0 = time.time()

    with open(log_path, "w", encoding="utf-8") as log_file:
        def log(msg=""):
            print(msg)
            print(msg, file=log_file)
            log_file.flush()

        log(f"=== Seed {seed} ===")
        log(f"Generating initial data: n_init = {n_init}")
        initial_mod.generate_initial_data(n_init=n_init, save_path=str(initial_data_path))

        log("")
        log(f"Running Bayesian optimization: n_iter = {n_iter}")
        bo_mod.start_time = time.time()
        best_x, best_weight = bo_mod.bayesian_optimization(
            n_iter=n_iter,
            data_path=str(initial_data_path),
        )

        elapsed = time.time() - t0
        log("")
        log("Finished run")
        log(f"best_x = {best_x}")
        log(f"best_weight = {best_weight}")
        log(f"elapsed_seconds = {elapsed:.2f}")

    move_if_exists(plot_src, plot_dst)

    best_y_history = captured_history["best_y_history"]
    if best_y_history is None:
        raise RuntimeError(
            "Failed to capture best_y_history from the BO script. "
            "Please check whether the plotting section still uses plt.plot(best_y_history, ...)."
        )

    return {
        "seed": seed,
        "best_weight": float(best_weight),
        "elapsed_seconds": float(time.time() - t0),
        "initial_data_path": str(initial_data_path),
        "plot_path": str(plot_dst) if plot_dst.exists() else "",
        "log_path": str(log_path),
        "best_x": np.array(best_x, dtype=float),
        "best_y_history": np.array(best_y_history, dtype=float),
    }


def run_all_seeds(
    seeds=None,
    n_init: int = 4000,
    n_iter: int = 100,
    results_root: Optional[Path] = None,
):
    if seeds is None:
        seeds = list(range(10))

    if results_root is None:
        results_root = THIS_DIR / "multi_seed_results"
    results_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    all_history_rows = []

    for seed in seeds:
        result = run_one_seed(seed=seed, n_init=n_init, n_iter=n_iter, results_root=results_root)
        summary_rows.append(result)

        for i, value in enumerate(result["best_y_history"], start=1):
            all_history_rows.append({
                "seed": seed,
                "iteration": i,
                "best_weight_so_far": float(value),
            })    

    csv_path = results_root / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "iteration", "best_weight_so_far"])
        for row in all_history_rows:
            writer.writerow([row["seed"], row["iteration"], row["best_weight_so_far"]])

    all_weights = np.array([row["best_weight"] for row in summary_rows], dtype=float)
    np.savez(
        results_root / "summary.npz",
        seeds=np.array([row["seed"] for row in summary_rows], dtype=int),
        best_weights=all_weights,
        best_x=np.vstack([row["best_x"] for row in summary_rows]),
        mean_best_weight=np.mean(all_weights),
        std_best_weight=np.std(all_weights, ddof=1) if len(all_weights) > 1 else 0.0,
    )

    print("\n=== All runs finished ===")
    print(f"Results folder: {results_root}")
    print(f"Mean best weight: {np.mean(all_weights):.6f}")
    print(f"Std best weight : {np.std(all_weights, ddof=1) if len(all_weights) > 1 else 0.0:.6f}")
    print(f"Summary CSV     : {csv_path}")



if __name__ == "__main__":
    run_all_seeds(seeds=[11, 23, 35, 42, 56, 78, 99, 123], n_init=4000, n_iter=100)
