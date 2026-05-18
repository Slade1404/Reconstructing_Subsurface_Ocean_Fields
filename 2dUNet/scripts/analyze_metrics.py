import json

d = json.load(open(r'results\bob_1_4\unet2d_bob_1_4_metrics.json'))
labels = d['depth_labels']

print(f"Epochs: {d['total_epochs']}\n")
print(f"{'Depth':>8} | {'Train RMSE':>10} {'Test RMSE':>10} {'Gap':>6} | {'Train R2':>8} {'Test R2':>8}")
print("-" * 72)
for l in labels:
    tr = d['rmse_T'][l]['train'][-1]
    te = d['rmse_T'][l]['test'][-1]
    r2_tr = d['r2_T'][l]['train'][-1]
    r2_te = d['r2_T'][l]['test'][-1]
    print(f"{l:>8} | {tr:10.4f} {te:10.4f} {te - tr:6.2f} | {r2_tr:8.4f} {r2_te:8.4f}")

print("\n--- Salinity ---")
print(f"{'Depth':>8} | {'Train RMSE':>10} {'Test RMSE':>10} {'Gap':>6} | {'Train R2':>8} {'Test R2':>8}")
print("-" * 72)
for l in labels:
    tr = d['rmse_S'][l]['train'][-1]
    te = d['rmse_S'][l]['test'][-1]
    r2_tr = d['r2_S'][l]['train'][-1]
    r2_te = d['r2_S'][l]['test'][-1]
    print(f"{l:>8} | {tr:10.4f} {te:10.4f} {te - tr:6.2f} | {r2_tr:8.4f} {r2_te:8.4f}")

# Check convergence: are last 5 epochs still improving?
print("\n--- Convergence Check (last 5 epochs Train RMSE at 0.5m) ---")
vals = d['rmse_T']['0.5m']['train'][-5:]
for i, v in enumerate(vals):
    print(f"  Epoch {d['total_epochs'] - 4 + i}: {v:.4f}")
