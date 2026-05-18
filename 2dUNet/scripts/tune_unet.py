"""
Hyperparameter tuning script for the 2D-UNet (3-Year Dataset).

Phase 1: Architecture & Regularization Grid
Phase 2: Physics Loss Calibration Grid

Usage:
    cd d:\\Mlpr_project\\2dUNet
    python scripts/tune_unet.py --phase 1
    python scripts/tune_unet.py --phase 2
"""
import os
import sys
import json
import csv
import yaml
import copy
import itertools
import time
import argparse

# Dynamically resolve project root and config path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

TUNE_EPOCHS = 5  # fast mode epochs
BASE_CONFIG = os.path.join(PROJECT_ROOT, "configs", "default.yaml")


def get_grid(phase):
    if phase == 1:
        # Phase 1: Regularization (fixed capacity)
        return {
            "dropout_rate": [0.0, 0.1, 0.2],
            "weight_decay": [0.0, 1e-4, 1e-3],
        }
    elif phase == 2:
        # Phase 2: Physics Constraints (Highly Targeted)
        return {
            "lambda_stability": [0.0, 0.05, 0.1, 0.5],
            "lambda_surface": [0.5, 1.0, 2.0],
        }
    else:
        raise ValueError("Phase must be 1 or 2")


def build_run_config(base_cfg, phase, params, run_name):
    """Create a modified config dict for a single tuning run."""
    cfg = copy.deepcopy(base_cfg)

    # Universally fixed tuning settings
    cfg['data']['batch_size'] = 16 # Use 16 as it's the standard for our 3 year model
    cfg['training']['learning_rate'] = 0.001
    cfg['training']['epochs'] = TUNE_EPOCHS
    cfg['training']['fast_mode'] = True  # skip backup/checkpoint, eval only last epoch
    cfg['training']['use_argo_loss'] = False # Completely removed from training

    if phase == 1:
        # Fixed architecture for Phase 1
        cfg['model']['filters'] = [32, 64, 128, 256]
        cfg['training']['lambda_stability'] = 0.1
        cfg['training']['lambda_surface'] = 1.0
        
        # Grid parameters
        cfg['model']['dropout_rate'] = params['dropout_rate']
        cfg['model']['weight_decay'] = params['weight_decay']
    
    elif phase == 2:
        # Architecture remains whatever is naturally in default.yaml
        # Grid parameters
        cfg['training']['lambda_stability'] = params['lambda_stability']
        cfg['training']['lambda_surface'] = params['lambda_surface']

    # Per-run output directories (avoid overwriting)
    cfg['model']['name'] = run_name
    cfg['training']['checkpoint_dir'] = f"checkpoints/tuning/phase{phase}/{run_name}"
    cfg['training']['results_dir'] = f"results/tuning/phase{phase}/{run_name}"
    cfg['training']['backup_dir'] = f"backups/tuning/phase{phase}/{run_name}"

    return cfg


def run_single(cfg, run_name):
    """Train one configuration and return the final test RMSE and R² summary."""
    # Write temp config to disk
    temp_config_path = f"configs/_tune_{run_name}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    try:
        import subprocess
        result = subprocess.run(
            ["python", "-m", "src.train", "--config", temp_config_path],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            err_lines = result.stderr.strip().split('\n')[-5:]
            print(f"  [ERROR] Run {run_name} failed:")
            for line in err_lines:
                print(f"    {line}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Run {run_name} timed out (15 min)")
        return None
    except Exception as e:
        print(f"  [ERROR] Run {run_name} failed: {e}")
        return None
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    # Read the saved metrics
    metrics_path = os.path.join(cfg['training']['results_dir'], f"{run_name}_metrics.json")
    if not os.path.exists(metrics_path):
        print(f"  [WARNING] No metrics file found for {run_name}")
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)

    # Compute average test RMSE and R² across all 10 depths (last epoch)
    avg_rmse_T, avg_rmse_S = 0.0, 0.0
    avg_r2_T, avg_r2_S = 0.0, 0.0
    n = len(metrics['depth_labels'])
    for label in metrics['depth_labels']:
        avg_rmse_T += metrics['rmse_T'][label]['test'][-1]
        avg_rmse_S += metrics['rmse_S'][label]['test'][-1]
        avg_r2_T += metrics['r2_T'][label]['test'][-1]
        avg_r2_S += metrics['r2_S'][label]['test'][-1]

    return {
        'avg_test_rmse_T': avg_rmse_T / n,
        'avg_test_rmse_S': avg_rmse_S / n,
        'avg_test_r2_T': avg_r2_T / n,
        'avg_test_r2_S': avg_r2_S / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for 2D-UNet")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2],
                        help="Tuning phase (1: Regularization, 2: Physics Loss)")
    args = parser.parse_args()
    phase = args.phase

    print("=" * 60)
    print(f"  2D-UNet Hyperparameter Tuning - Phase {phase}")
    print("=" * 60)

    with open(BASE_CONFIG) as f:
        base_cfg = yaml.safe_load(f)

    grid = get_grid(phase)
    keys = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))
    total = len(combos)

    print(f"Grid: {' x '.join(f'{k}({len(v)})' for k, v in grid.items())}")
    print(f"Total runs: {total}, Epochs per run: {TUNE_EPOCHS}")
    if phase == 1:
        print(f"Fixed: filters=[32, 64, 128, 256], lambda_stability=0.1, lambda_surface=1.0")
    else:
        print(f"Fixed: Architecture from {BASE_CONFIG}")
    print()

    # Prepare CSV log
    results_dir = f"results/tuning/phase{phase}"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "tuning_results.csv")
    csv_fields = ['run'] + keys + [
        'avg_test_rmse_T', 'avg_test_rmse_S', 'avg_test_r2_T', 'avg_test_r2_S', 'time_min'
    ]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        
        # Build a safe run name based on parameters
        param_str = "_".join([f"{k.split('_')[-1]}{v}" for k, v in params.items()])
        run_name = f"p{phase}_{i+1:03d}_{param_str}"

        print(f"\n{'='*60}")
        print(f"  Run {i+1}/{total}: {params}")
        print(f"{'='*60}")

        cfg = build_run_config(base_cfg, phase, params, run_name)

        t0 = time.time()
        result = run_single(cfg, run_name)
        elapsed = (time.time() - t0) / 60.0

        row = {
            'run': run_name,
            **params,
            'time_min': f"{elapsed:.1f}",
        }
        if result:
            row.update(result)
            results.append(row)
        else:
            row.update({k: 'FAILED' for k in csv_fields if k not in row})

        # Append to CSV immediately
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
            writer.writerow(row)

        print(f"  Finished in {elapsed:.1f} min")
        if result:
            print(f"  Test RMSE T={result['avg_test_rmse_T']:.4f}, "
                  f"S={result['avg_test_rmse_S']:.4f}")
            print(f"  Test R²   T={result['avg_test_r2_T']:.4f}, "
                  f"S={result['avg_test_r2_S']:.4f}")

    # Final ranking
    print(f"\n{'='*60}")
    print("  FINAL RANKING (by avg Test R² Temperature)")
    print(f"{'='*60}")
    results.sort(key=lambda r: r.get('avg_test_r2_T', -999) if isinstance(r.get('avg_test_r2_T'), float) else -999, reverse=True)
    for rank, r in enumerate(results[:10], 1):
        # Format the parameters nicely for the ranking printout
        p_str = ", ".join([f"{k}={r.get(k)}" for k in keys])
        print(f"  #{rank}: {p_str} → "
              f"R²_T={r.get('avg_test_r2_T', 0):.4f}, RMSE_T={r.get('avg_test_rmse_T', 0):.4f}")

    print(f"\nFull results saved to: {csv_path}")


if __name__ == "__main__":
    main()
