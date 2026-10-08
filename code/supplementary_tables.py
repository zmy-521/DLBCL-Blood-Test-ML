from config import O, prepare_run
from tables import three_line, ptext
import pandas as pd, json
R=O/'results_v2'
def main():
 prepare_run()
 spec=json.loads((O/'models_v2/frozen_model_specification.json').read_text(encoding='utf8'))['models']['Final procedure']
 q=pd.read_csv(R/'paired_model_comparisons.csv');rows=[[r.comparator,f'{r.AUROC_difference:.4f}',f'{r.CI_low:.4f}–{r.CI_high:.4f}',ptext(r.p_value),ptext(r.p_BH)] for r in q.itertuples()]
 three_line('model_comparisons.docx','Table S5. Paired development OOF AUROC comparisons',['Comparator','Final minus comparator','95% CI','P value','BH-adjusted P'],rows,['Final adaptive procedure versus each independently nested comparator. 2000 stratified paired patient bootstrap samples; centered bootstrap two-sided P values with finite-sample correction. CIs are conditional on the fitted OOF predictions. No external model selection.'],True)
 c=pd.read_csv(R/'feature_number_performance.csv');rows=[[int(r.n_features),f'{r.mean_auc:.4f}',f'{r.se_auc:.4f}','Yes' if r.n_features==len(spec['features']) else ''] for r in c.itertuples()]
 three_line('feature_number_performance.docx','Table S6. Training-only feature-count selection',['Feature count','Mean inner-CV AUROC','SE','Selected'],rows,['Full-development inner 3-fold grouped CV. Features ranked within each training fold by random forest importance. Smallest k within one SE of the best mean AUROC is selected. This is a selection curve, not an unbiased performance estimate; external data are never used.'])
 f=pd.read_csv(R/'final_procedure_fold_performance.csv');choices=json.loads((R/'outer_selected_models.json').read_text(encoding='utf8'));rows=[]
 for r,x in zip(f.itertuples(),choices):rows.append([r.fold,x['model'],x['n_features'],f'{r.AUROC:.3f}',f'{r.AUPRC:.3f}',f'{r.sensitivity:.3f}',f'{r.specificity:.3f}',f'{r.brier:.4f}'])
 three_line('final_procedure_folds.docx','Table S7. Final adaptive procedure by held-out development fold',['Fold','Selected algorithm','Features','AUROC','AUPRC','Sensitivity','Specificity','Brier'],rows,['Each held-out patient occurs in exactly one outer fold. Algorithm, parameters, feature ranking and feature count were selected using only that outer training set. The whole-development frozen model is RF with 13 features; the OOF procedure is adaptive across folds.'],True)
if __name__=='__main__':main()
