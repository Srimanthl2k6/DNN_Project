import os
import pandas as pd

def select_best_model(root_dir):
    res_path = os.path.join(root_dir, 'results.csv')
    if not os.path.exists(res_path):
        print("No results.csv found. Cannot select best model.")
        return
        
    df = pd.read_csv(res_path)
    if df.empty:
        print("results.csv is empty. Training checkpoints have not completed evaluating.")
        return
        
    print("--- Best Model Selection ---")
    
    # Selection Criteria:
    # 1. Lowest KL
    # 2. Strong robustness (requires benchmark metrics)
    # 3. Calibration (Lowest ECE)
    
    best_kl = df.loc[df['kl'].idxmin()]
    best_ece = df.loc[df['ece'].idxmin()]
    
    print(f"Lowest KL Divergence Model: {best_kl['experiment_name']} (KL: {best_kl['kl']:.4f})")
    print(f"Best Calibrated Model (ECE): {best_ece['experiment_name']} (ECE: {best_ece['ece']:.4f})")
    
    # Save selection matrix with the best KL model first for downstream reporting/audit.
    df = df.sort_values('kl').reset_index(drop=True)
    df.to_csv(os.path.join(root_dir, 'FINAL_COMPARISON_MATRIX.csv'), index=False)
    print("Exported FINAL_COMPARISON_MATRIX.csv")

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    select_best_model(root)
