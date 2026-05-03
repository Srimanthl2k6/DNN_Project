import os
import pandas as pd

def consolidate_results():
    if os.path.exists('results.csv'):
        pd.read_csv('results.csv').to_csv('results_final.csv', index=False)

def generate_report():
    try:
        final_res = pd.read_csv('results_final.csv')
        best = pd.read_csv('FINAL_COMPARISON_MATRIX.csv') if os.path.exists('FINAL_COMPARISON_MATRIX.csv') else final_res
        metrics = best.iloc[0].to_dict() if not best.empty else {}
    except Exception:
        metrics = {'kl': 'N/A', 'ece': 'N/A', 'experiment_name': 'N/A'}
        
    try:
        rob_df = pd.read_csv('robustness_results.csv')
        fgsm_drop_high = rob_df[(rob_df['entropy_group'] == 'high') & (rob_df['attack'] == 'fgsm')].iloc[0]['acc_drop']
        pgd_drop_high = rob_df[(rob_df['entropy_group'] == 'high') & (rob_df['attack'] == 'pgd')].iloc[0]['acc_drop']
        fgsm_drop_low = rob_df[(rob_df['entropy_group'] == 'low') & (rob_df['attack'] == 'fgsm')].iloc[0]['acc_drop']
        pgd_drop_low = rob_df[(rob_df['entropy_group'] == 'low') & (rob_df['attack'] == 'pgd')].iloc[0]['acc_drop']
        rob_text = f"\n### Robustness Integration\n- FGSM High Entropy Acc Drop: {fgsm_drop_high:.4f}\n- PGD High Entropy Acc Drop: {pgd_drop_high:.4f}\n- FGSM Low Entropy Acc Drop: {fgsm_drop_low:.4f}\n- PGD Low Entropy Acc Drop: {pgd_drop_low:.4f}\n"
    except:
        rob_text = ""
        
    with open('FINAL_REPORT.md', 'w') as f:
        f.write("# Final Project Report\n\n")
        f.write(f"Best Experiment: {metrics.get('experiment_name', 'N/A')}\n")
        f.write(f"- KL Divergence: {metrics.get('kl', 'N/A')}\n")
        f.write(f"- Calibration ECE: {metrics.get('ece', 'N/A')}\n")
        f.write(rob_text)
        
if __name__ == "__main__":
    consolidate_results()
    generate_report()
