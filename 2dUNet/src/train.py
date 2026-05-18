"""
Training script for the 2D-UNet ocean reconstruction model.

Handles model construction, training loop with pause/resume support,
and per-epoch RMSE + R² evaluation at 10 strategic depth levels.
"""
import tensorflow as tf
from src.models.unet2d import build_unet2d
from src.datasets import create_dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
import json
import os


# ---------------------------------------------------------------------------
# Custom Keras Model with optional Argo loss
# ---------------------------------------------------------------------------

class OceanUNet(tf.keras.Model):
    """Wraps the base UNet with a custom train/test step supporting physics constraints."""

    def __init__(self, model, lambda_surface=1.0, lambda_stability=0.1, **kwargs):
        super(OceanUNet, self).__init__(**kwargs)
        self.model = model
        self.lambda_surface = lambda_surface
        self.lambda_stability = lambda_stability
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.dense_loss_tracker = tf.keras.metrics.Mean(name="dense_loss")
        self.surface_loss_tracker = tf.keras.metrics.Mean(name="surface_loss")
        self.stability_loss_tracker = tf.keras.metrics.Mean(name="stability_loss")

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def _compute_loss(self, x, y_pred, dense_target):
        """Shared loss computation for train and test steps."""
        dense_loss = tf.reduce_mean(tf.square(dense_target - y_pred))

        # Surface Consistency: 
        # SST (x[..., 0:1]) should match Temp at 0.5m (y_pred[..., 0:1])
        # SSS (x[..., 1:2]) should match Salt at 0.5m (y_pred[..., 26:27])
        sst_input = x[..., 0:1]
        temp_surface_pred = y_pred[..., 0:1]
        temp_surface_loss = tf.reduce_mean(tf.square(temp_surface_pred - sst_input))

        sss_input = x[..., 1:2]
        salt_surface_pred = y_pred[..., 26:27]
        salt_surface_loss = tf.reduce_mean(tf.square(salt_surface_pred - sss_input))
        
        surface_loss = temp_surface_loss + salt_surface_loss

        # Thermal Stability: Temperature should generally decrease with depth.
        # temp_diff > 0 means temp increased with depth (unstable). We penalize positive diffs.
        temp_pred = y_pred[..., :26]
        temp_diff = temp_pred[..., 1:] - temp_pred[..., :-1]
        stability_loss = tf.reduce_mean(tf.nn.relu(temp_diff))

        total_loss = dense_loss + self.lambda_surface * surface_loss + self.lambda_stability * stability_loss
        return total_loss, dense_loss, surface_loss, stability_loss

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            total_loss, dense_loss, surface_loss, stability_loss = self._compute_loss(
                x, y_pred, y["dense_target"])

        self.optimizer.apply_gradients(
            zip(tape.gradient(total_loss, self.trainable_variables),
                self.trainable_variables))

        self.loss_tracker.update_state(total_loss)
        self.dense_loss_tracker.update_state(dense_loss)
        self.surface_loss_tracker.update_state(surface_loss)
        self.stability_loss_tracker.update_state(stability_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self.model(x, training=False)
        total_loss, dense_loss, surface_loss, stability_loss = self._compute_loss(
            x, y_pred, y["dense_target"])

        self.loss_tracker.update_state(total_loss)
        self.dense_loss_tracker.update_state(dense_loss)
        self.surface_loss_tracker.update_state(surface_loss)
        self.stability_loss_tracker.update_state(stability_loss)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.loss_tracker, self.dense_loss_tracker, 
                self.surface_loss_tracker, self.stability_loss_tracker]


# ---------------------------------------------------------------------------
# Per-epoch RMSE & R² callback
# ---------------------------------------------------------------------------

class MetricsCallback(tf.keras.callbacks.Callback):
    """
    Computes RMSE and R² at 10 strategic ocean depth levels every epoch.
    Crops padded pixels before computing metrics to avoid inflated accuracy.
    Generates train-vs-test curves and saves a JSON history at the end.
    """
    # 10 strategic depth indices spanning surface (0.5m) to deep (186m)
    DEPTH_INDICES = [0, 3, 6, 9, 13, 16, 19, 22, 24, 25]
    DEPTH_LABELS = ["0.5m", "3.8m", "7.9m", "13.5m", "25.2m",
                    "40.3m", "65.8m", "109.7m", "155.9m", "186.1m"]

    def __init__(self, train_dataset, val_dataset, results_dir, model_name,
                 original_hw=None, n_depths=26,
                 train_eval_steps=100, val_eval_steps=100, eval_frequency=1):
        """
        Args:
            original_hw: (H, W) tuple of unpadded spatial dims. If provided,
                         predictions are cropped before metric computation.
            n_depths: Number of depth levels (default 26).
            model_name: Used for output file naming.
            eval_frequency: Compute metrics every N epochs (1=every epoch).
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.results_dir = results_dir
        self.model_name = model_name
        self.original_hw = original_hw
        self.n_depths = n_depths
        self.train_eval_steps = train_eval_steps
        self.val_eval_steps = val_eval_steps
        self.eval_frequency = eval_frequency

        # Per-depth RMSE and R² history — load from disk if resuming
        self._history_path = os.path.join(results_dir, f'{model_name}_metrics.json')
        if os.path.exists(self._history_path):
            print(f"Resuming metrics history from {self._history_path}")
            with open(self._history_path) as f:
                saved = json.load(f)
            self.rmse_T = saved.get('rmse_T', {d: {"train": [], "test": []} for d in self.DEPTH_LABELS})
            self.rmse_S = saved.get('rmse_S', {d: {"train": [], "test": []} for d in self.DEPTH_LABELS})
            self.r2_T = saved.get('r2_T', {d: {"train": [], "test": []} for d in self.DEPTH_LABELS})
            self.r2_S = saved.get('r2_S', {d: {"train": [], "test": []} for d in self.DEPTH_LABELS})
        else:
            self.rmse_T = {d: {"train": [], "test": []} for d in self.DEPTH_LABELS}
            self.rmse_S = {d: {"train": [], "test": []} for d in self.DEPTH_LABELS}
            self.r2_T = {d: {"train": [], "test": []} for d in self.DEPTH_LABELS}
            self.r2_S = {d: {"train": [], "test": []} for d in self.DEPTH_LABELS}

    def _compute_metrics(self, dataset, eval_steps):
        """
        Compute per-channel RMSE and R² over the dataset.
        Returns (rmse_array, r2_array) each of shape (out_channels,).
        """
        sum_sq_err = None   # SS_res accumulator
        sum_y = None         # for computing global mean
        sum_y2 = None        # for SS_tot
        n_pixels = 0

        for x, y in dataset.take(eval_steps):
            y_pred = self.model(x, training=False)
            y_true = y["dense_target"]

            # Crop padding before computing metrics
            if self.original_hw:
                h, w = self.original_hw
                y_pred = y_pred[:, :h, :w, :]
                y_true = y_true[:, :h, :w, :]

            n = tf.cast(tf.reduce_prod(tf.shape(y_true)[:3]), tf.float32)
            sq_err = tf.reduce_sum(tf.square(y_true - y_pred), axis=[0, 1, 2])
            y_sum = tf.reduce_sum(y_true, axis=[0, 1, 2])
            y2_sum = tf.reduce_sum(tf.square(y_true), axis=[0, 1, 2])

            if sum_sq_err is None:
                sum_sq_err, sum_y, sum_y2 = sq_err, y_sum, y2_sum
            else:
                sum_sq_err += sq_err
                sum_y += y_sum
                sum_y2 += y2_sum
            n_pixels += n.numpy()

        if sum_sq_err is None:
            return None, None

        mse = (sum_sq_err / n_pixels).numpy()
        rmse = np.sqrt(mse)

        # R² = 1 - SS_res / SS_tot
        ss_res = sum_sq_err.numpy()
        ss_tot = sum_y2.numpy() - (sum_y.numpy() ** 2) / n_pixels
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

        return rmse, r2

    def on_epoch_end(self, epoch, logs=None):
        # Skip expensive eval on non-target epochs (for tuning speed)
        total_epochs = self.params.get('epochs', 1)
        is_last = (epoch + 1) >= total_epochs
        if not is_last and (epoch + 1) % self.eval_frequency != 0:
            return

        rmse_train, r2_train = self._compute_metrics(self.train_dataset, self.train_eval_steps)
        rmse_test, r2_test = self._compute_metrics(self.val_dataset, self.val_eval_steps)
        if rmse_train is None or rmse_test is None:
            return

        nd = self.n_depths
        for idx, label in zip(self.DEPTH_INDICES, self.DEPTH_LABELS):
            self.rmse_T[label]["train"].append(float(rmse_train[idx]))
            self.rmse_T[label]["test"].append(float(rmse_test[idx]))
            self.rmse_S[label]["train"].append(float(rmse_train[nd + idx]))
            self.rmse_S[label]["test"].append(float(rmse_test[nd + idx]))
            self.r2_T[label]["train"].append(float(r2_train[idx]))
            self.r2_T[label]["test"].append(float(r2_test[idx]))
            self.r2_S[label]["train"].append(float(r2_train[nd + idx]))
            self.r2_S[label]["test"].append(float(r2_test[nd + idx]))

        # Save to disk after every epoch so history survives termination
        self._save_history()

    def on_train_end(self, logs=None):
        os.makedirs(self.results_dir, exist_ok=True)
        n_epochs = len(self.rmse_T[self.DEPTH_LABELS[0]]["train"])
        if n_epochs == 0:
            return
        epochs = list(range(1, n_epochs + 1))

        # --- Plot RMSE over epochs ---
        print("Generating RMSE over Epochs plots...")
        for var, history in [("Temperature", self.rmse_T), ("Salinity", self.rmse_S)]:
            fig, axes = plt.subplots(2, 5, figsize=(22, 8))
            fig.suptitle(f'{self.model_name} — {var} RMSE: Train vs Test', fontsize=14)
            for i, label in enumerate(self.DEPTH_LABELS):
                ax = axes[i // 5][i % 5]
                ax.plot(epochs, history[label]["train"], label='Train', color='tab:blue')
                ax.plot(epochs, history[label]["test"], label='Test', color='tab:orange')
                ax.set_title(f'Depth {label}', fontsize=10)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('RMSE')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                # Auto-scale: skip first 2 warmup epochs
                vals = history[label]["train"][2:] + history[label]["test"][2:]
                if vals:
                    lo, hi = min(vals), max(vals)
                    margin = (hi - lo) * 0.15 + 1e-6
                    ax.set_ylim(lo - margin, hi + margin)
            plt.tight_layout()
            suffix = "T" if var == "Temperature" else "S"
            plt.savefig(os.path.join(self.results_dir,
                        f'{self.model_name}_rmse_epochs_{suffix}.png'), dpi=300)
            plt.close()

        self._save_history()

    def _save_history(self):
        """Persist metrics history to JSON on disk."""
        os.makedirs(self.results_dir, exist_ok=True)
        n_epochs = len(self.rmse_T[self.DEPTH_LABELS[0]]["train"])
        history_data = {
            "model_name": self.model_name,
            "total_epochs": n_epochs,
            "depth_labels": self.DEPTH_LABELS,
            "rmse_T": self.rmse_T, "rmse_S": self.rmse_S,
            "r2_T": self.r2_T, "r2_S": self.r2_S,
        }
        with open(self._history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        print(f"Metrics saved to {self._history_path} ({n_epochs} epochs)")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Propagate in_channels to data config for generator TensorSpec
    config['data']['in_channels'] = config['model']['in_channels']
    
    out_channels = config['model']['out_channels']
    print("Setting up data pipeline...")
    train_dataset = create_dataset(config['data'], out_channels, is_train=True)
    val_dataset = create_dataset(config['data'], out_channels, is_train=False)

    # Read unpadded spatial dimensions for metric cropping
    ds_tmp = xr.open_dataset(config['data']['glorys_path'])
    original_hw = (ds_tmp.sizes['latitude'], ds_tmp.sizes['longitude'])
    n_depths = ds_tmp.sizes['depth']
    ds_tmp.close()
    print(f"Original spatial dims: {original_hw[0]}x{original_hw[1]}, Depths: {n_depths}")

    print("Building model...")
    patch_size = config['data']['patch_size']
    in_channels = config['model']['in_channels']
    base_model = build_unet2d(
        input_shape=(patch_size, patch_size, in_channels),
        out_channels=out_channels,
        filters=config['model']['filters'],
        dropout_rate=config['model'].get('dropout_rate', 0.0),
        weight_decay=config['model'].get('weight_decay', 0.0)
    )

    model = OceanUNet(base_model,
                      lambda_surface=config['training'].get('lambda_surface', 1.0),
                      lambda_stability=config['training'].get('lambda_stability', 0.1))
    model.build(input_shape=(None, patch_size, patch_size, in_channels))
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=config['training']['learning_rate']))

    # --- Callbacks ---
    results_dir = config['training'].get('results_dir', 'results')
    model_name = config['model']['name']
    fast_mode = config['training'].get('fast_mode', False)
    os.makedirs(results_dir, exist_ok=True)

    callbacks = []

    if not fast_mode:
        checkpoint_dir = config['training']['checkpoint_dir']
        backup_dir = config['training'].get('backup_dir', 'backups')
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "model_latest.weights.h5"),
            save_weights_only=True, save_best_only=False, verbose=1))
        callbacks.append(tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir))

    train_steps = config['data'].get('train_samples', 200000) // config['data']['batch_size']
    val_steps = config['data'].get('val_samples', 10000) // config['data']['batch_size']
    eval_freq = config['training']['epochs'] if fast_mode else 1  # only last epoch in fast mode

    metrics_cb = MetricsCallback(
        train_dataset, val_dataset, results_dir,
        model_name=model_name,
        original_hw=original_hw if config['data'].get('full_map') else None,
        n_depths=n_depths,
        train_eval_steps=min(train_steps, 200),
        val_eval_steps=min(val_steps, 200),
        eval_frequency=eval_freq)
    callbacks.append(metrics_cb)

    print(f"Starting training... {'(fast mode)' if fast_mode else ''}")
    model.fit(
        train_dataset.repeat(),
        epochs=config['training']['epochs'],
        steps_per_epoch=train_steps,
        validation_data=val_dataset.repeat(),
        validation_steps=val_steps,
        callbacks=callbacks)

    print("Training finished.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train 2D-UNet ocean model")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()
    train(args.config)
