import os
import pandas as pd
import re
issues=[]
# File existence
required=['results.csv','results_final.csv','FINAL_COMPARISON_MATRIX.csv','robustness_results.csv','FINAL_REPORT.md']
for f in required:
    if not os.path.exists(os.path.join('..', f)) and not os.path.exists(f):
        issues.append({'File':f,'Problem':'Missing file','Fix':'Regenerate or restore file'})
# plots
plots_path='plots'
if os.path.isdir(plots_path):
    nonempty=[p for p in os.listdir(plots_path) if os.path.getsize(os.path.join(plots_path,p))>0]
    if len(nonempty)==0:
        issues.append({'File':'plots/','Problem':'Empty plots directory','Fix':'Run visualize.py to regenerate plots'})
else:
    issues.append({'File':'plots/','Problem':'Missing plots directory','Fix':'Run visualize.py to generate visualizations'})
# Load CSVs
csvs={}
for name in ['results.csv','results_final.csv','FINAL_COMPARISON_MATRIX.csv']:
    if os.path.exists(name):
        try:
            csvs[name]=pd.read_csv(name)
        except Exception as e:
            issues.append({'File':name,'Problem':f'Failed to read CSV: {e}','Fix':'Fix CSV format'})
# Consistency
if 'results_final.csv' in csvs and 'FINAL_COMPARISON_MATRIX.csv' in csvs:
    rf=set(csvs['results_final.csv']['experiment_name'].astype(str))
    fc=set(csvs['FINAL_COMPARISON_MATRIX.csv']['experiment_name'].astype(str))
    if rf!=fc:
        missing_final=list(fc-rf)+list(rf-fc)
        issues.append({'File':'results_final.csv / FINAL_COMPARISON_MATRIX.csv','Problem':f'Mismatch experiment sets: {missing_final}','Fix':'Ensure both files contain identical experiment rows'})
# Duplicates
for k,df in csvs.items():
    if 'experiment_name' in df.columns:
        dup=df['experiment_name'].duplicated().any()
        if dup:
            dup_names=df['experiment_name'][df['experiment_name'].duplicated()].unique().tolist()
            issues.append({'File':k,'Problem':f'Duplicate experiments: {dup_names}','Fix':'Remove duplicates and regenerate results_final.csv'})
# Best model integrity
best_exp=None
if 'FINAL_COMPARISON_MATRIX.csv' in csvs:
    df_fc=csvs['FINAL_COMPARISON_MATRIX.csv']
    if len(df_fc)>0:
        best_exp=df_fc.iloc[0]['experiment_name']
        if 'results_final.csv' in csvs:
            df_rf=csvs['results_final.csv']
            if 'kl' in df_rf.columns:
                minkl=df_rf['kl'].min()
                minrow=df_rf[df_rf['kl']==minkl]
                if len(minrow)==0 or minrow.iloc[0]['experiment_name']!=best_exp:
                    issues.append({'File':'FINAL_COMPARISON_MATRIX.csv','Problem':'Top row experiment does not match lowest KL in results_final.csv','Fix':'Recompute FINAL_COMPARISON_MATRIX.csv with correct ordering'})
        if os.path.exists('FINAL_REPORT.md'):
            txt=open('FINAL_REPORT.md').read()
            if best_exp not in txt:
                issues.append({'File':'FINAL_REPORT.md','Problem':'Best experiment name not present or mismatched','Fix':'Regenerate FINAL_REPORT.md'})
# Metric validity
numeric_checks={'kl':(0,None),'tvd':(0,1),'ece':(0,1),'sba':(0,1),'p@100':(0,1),'p@200':(0,1),'p@500':(0,1)}
if 'results_final.csv' in csvs:
    df=csvs['results_final.csv']
    for col,(low,high) in numeric_checks.items():
        if col in df.columns:
            vals=df[col].astype(float)
            if (vals==0).all():
                issues.append({'File':'results_final.csv','Problem':f'All values zero for {col}','Fix':'Verify evaluation pipeline for {col}'})
            if (vals<low).any():
                issues.append({'File':'results_final.csv','Problem':f'Some values for {col} below {low}','Fix':'Check metric computation'})
            if high is not None and (vals>high).any():
                issues.append({'File':'results_final.csv','Problem':f'Some values for {col} above {high}','Fix':'Check metric computation'})
# Robustness checks
if os.path.exists('robustness_results.csv'):
    try:
        rpd=pd.read_csv('robustness_results.csv')
        for grp in ['high','low']:
            for atk in ['fgsm','pgd']:
                mask=(rpd['entropy_group']==grp)&(rpd['attack']==atk)
                if mask.sum()==0:
                    issues.append({'File':'robustness_results.csv','Problem':f'Missing entry for {grp} {atk}','Fix':'Re-run benchmark_robustness.py'})
                else:
                    row=rpd[mask].iloc[0]
                    if 'kl_shift' in row and row['kl_shift']<=0:
                        issues.append({'File':'robustness_results.csv','Problem':f'KL did not increase for {grp} {atk}','Fix':'Check robustness computations'})
                    if 'acc_drop' in row and row['acc_drop']<=0:
                        issues.append({'File':'robustness_results.csv','Problem':f'Accuracy did not decrease for {grp} {atk}','Fix':'Check robustness computations'})
    except Exception as e:
        issues.append({'File':'robustness_results.csv','Problem':f'Failed to read or validate: {e}','Fix':'Fix CSV format'})
else:
    issues.append({'File':'robustness_results.csv','Problem':'Missing robustness_results.csv','Fix':'Run benchmark_robustness.py to generate it'})
# Robustness in report
if os.path.exists('robustness_results.csv') and os.path.exists('FINAL_REPORT.md'):
    txt=open('FINAL_REPORT.md').read()
    if 'Robustness Integration' not in txt:
        issues.append({'File':'FINAL_REPORT.md','Problem':'Robustness section missing','Fix':'Regenerate FINAL_REPORT.md'})
# Report authenticity
if os.path.exists('FINAL_REPORT.md'):
    txt=open('FINAL_REPORT.md').read()
    if 'N/A' in txt or 'placeholder' in txt.lower():
        issues.append({'File':'FINAL_REPORT.md','Problem':'Contains placeholder text or N/A','Fix':'Regenerate with real metrics'})
    nums=re.findall(r"\d+\.\d+", txt)
    if len(nums)==0:
        issues.append({'File':'FINAL_REPORT.md','Problem':'No numeric metrics present','Fix':'Regenerate with CSV values'})
# Final output
if len(issues)==0:
    print('### PASS')
else:
    print('### FAIL')
    print('\n### Issues (if any)')
    for it in issues:
        print(f"- [{it['Problem']}]\n  - File: {it['File']}\n  - Problem: {it['Problem']}\n  - Fix: {it['Fix']}\n")
