import os
import sys
import yaml
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unet2d import build_unet2d
from src.train import OceanUNet
from src.datasets import OceanDatasetGenerator

def main():
    print("=== Independent Argo Validation ===")
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    if not config['data'].get('argo_path'):
        print("Error: No argo_path specified in config. Cannot evaluate Argo.")
        return

    # 1. Build Model
    patch_size = config['data']['patch_size']
    in_channels = config['model']['in_channels']
    out_channels = config['model']['out_channels']
    
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
    
    # Needs to be built to load weights
    model.build(input_shape=(None, patch_size, patch_size, in_channels))
    
    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', config['training']['checkpoint_dir'])
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    
    if latest_ckpt:
        print(f"Loading weights from {latest_ckpt}...")
        model.load_weights(latest_ckpt).expect_partial()
    else:
        print("WARNING: No checkpoint found! Evaluating with random untrained weights.")

    # 2. Setup Test Generator
    print(f"\nEvaluating on Test Set ({config['data']['time_test'][0]} to {config['data']['time_test'][1]})...")
    generator = OceanDatasetGenerator(config['data'], is_train=False)
    
    all_preds = []
    all_targets = []
    
    # 3. Iterate over the entire test set
    num_test_samples = config['data'].get('val_samples', 365)
    
    for i in range(num_test_samples):
        # The generator yields one raw sample at a time
        x, y = next(generator())
        
        # Add batch dimension
        x_batch = np.expand_dims(x, axis=0)
        
        # Predict
        y_pred = model.model(x_batch, training=False).numpy()
        y_pred = y_pred[0] # remove batch dim
        
        # We only care about temperature (first 26 channels) for Argo
        temp_pred = y_pred[..., :26]
        argo_target = y["argo_target"]  # (H, W, 26)
        argo_mask = y["argo_mask"]      # (H, W, 1)
        
        # Extract pixels where argo mask is 1
        # argo_mask shape is (H, W, 1), we can squeeze it to (H, W) for boolean indexing
        mask_2d = argo_mask.squeeze(axis=-1) > 0.5
        
        if np.any(mask_2d):
            # Extract the actual valid pixels
            valid_preds = temp_pred[mask_2d]     # shape: (N_valid_pixels, 26)
            valid_targets = argo_target[mask_2d] # shape: (N_valid_pixels, 26)
            
            all_preds.append(valid_preds)
            all_targets.append(valid_targets)
            
        sys.stdout.write(f"\rProcessed {i+1}/{num_test_samples} days")
        sys.stdout.flush()

    print("\n\n=== Final Argo Evaluation Results ===")
    if not all_preds:
        print("No Argo data found in the test set period!")
        return
        
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate RMSE and R2 per depth
    depth_labels = ["0.5m", "3.8m", "7.9m", "13.5m", "25.2m", "40.3m", "65.8m", "109.7m", "155.9m", "186.1m"]
    # Assuming the first 10 depths correspond to the labels above
    
    print(f"Total valid Argo profiles evaluated: {all_preds.shape[0]}")
    print(f"\n{'Depth':<10} | {'RMSE':<10} | {'R²':<10}")
    print("-" * 35)
    
    for d in range(10):
        preds_d = all_preds[:, d]
        targets_d = all_targets[:, d]
        
        rmse = np.sqrt(mean_squared_error(targets_d, preds_d))
        r2 = r2_score(targets_d, preds_d)
        
        print(f"{depth_labels[d]:<10} | {rmse:<10.4f} | {r2:<10.4f}")
        
    # Overall metrics across all depths
    overall_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    overall_r2 = r2_score(all_targets, all_preds)
    
    print("-" * 35)
    print(f"{'OVERALL':<10} | {overall_rmse:<10.4f} | {overall_r2:<10.4f}")

if __name__ == "__main__":
    main()
