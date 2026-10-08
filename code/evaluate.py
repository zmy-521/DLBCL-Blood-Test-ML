from pathlib import Path
import sys,json
from config import O, F, prepare_run
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix,brier_score_loss
from scipy.optimize import minimize
from scipy.special import expit,logit
R=O/'results_v2';SEED=20261007
def calibration(y,p):
 z=logit(np.clip(p,1e-6,1-1e-6));y=np.asarray(y,float)
 def f(b):eta=b[0]+b[1]*z;return np.logaddexp(0,eta).sum()-y@eta
 fit=minimize(f,[0.,1.],method='BFGS');a,b=fit.x
 one=minimize(lambda v:np.logaddexp(0,v[0]+z).sum()-y@(v[0]+z),[0.],method='BFGS')
 return dict(calibration_intercept=float(a),calibration_slope=float(b),calibration_in_the_large=float(one.x[0]),calibration_converged=bool(fit.success or np.linalg.norm(fit.jac)<1e-3),brier=brier_score_loss(y,p))
def metrics(d,ci=True):
 y=d.y.to_numpy(int);p=d.p.to_numpy(float);cut=d.cutoff.to_numpy(float);pred=p>=cut;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 div=lambda a,b:float(a/b) if b else np.nan
 r=dict(n=len(y),cases=int(sum(y)),controls=int(sum(y==0)),AUROC=roc_auc_score(y,p),AUPRC=average_precision_score(y,p),sensitivity=div(tp,tp+fn),specificity=div(tn,tn+fp),accuracy=div(tp+tn,len(y)),F1=div(2*tp,2*tp+fp+fn),PPV=div(tp,tp+fp),NPV=div(tn,tn+fn),TP=int(tp),TN=int(tn),FP=int(fp),FN=int(fn),**calibration(y,p))
 if ci:
  rng=np.random.default_rng(SEED);i0=np.where(y==0)[0];i1=np.where(y==1)[0];au=[];ap=[]
  for _ in range(2000):
   ix=np.r_[rng.choice(i0,len(i0),True),rng.choice(i1,len(i1),True)];au.append(roc_auc_score(y[ix],p[ix]));ap.append(average_precision_score(y[ix],p[ix]))
  r.update(AUROC_low=np.quantile(au,.025),AUROC_high=np.quantile(au,.975),AUPRC_low=np.quantile(ap,.025),AUPRC_high=np.quantile(ap,.975))
 return r
def main():
 allpatients=[];results=[];records=[]
 for cohort,fn in [('Development OOF','development_oof_record_predictions.csv'),('External','external_record_predictions.csv')]:
  d=pd.read_csv(R/fn)
  for model,g in d.groupby('model',sort=False):
   assert g.groupby('group_key').y.nunique().max()==1 and g.groupby('group_key').cutoff.nunique().max()==1
   z=g.groupby('group_key').agg(y=('y','first'),p=('p','mean'),cutoff=('cutoff','first'),records=('p','size')).reset_index();z['cohort']=cohort;z['model']=model;allpatients.append(z)
   results.append(dict(cohort=cohort,model=model,**metrics(z)));records.append(dict(cohort=cohort,model=model,**metrics(g,ci=False)))
   print('Evaluated',cohort,model,flush=True)
 patient=pd.concat(allpatients);patient.to_csv(R/'patient_predictions.csv',index=False);pd.DataFrame(results).to_csv(R/'patient_level_performance.csv',index=False);pd.DataFrame(records).to_csv(R/'record_level_performance_NO_independent_CI.csv',index=False)
 # Paired patient bootstrap comparisons against final procedure, all dev-only.
 d=patient[patient.cohort.eq('Development OOF')&~patient.model.eq('Strict baseline sensitivity')];piv=d.pivot(index='group_key',columns='model',values='p');y=d.drop_duplicates('group_key').set_index('group_key').loc[piv.index,'y'].to_numpy();i0=np.where(y==0)[0];i1=np.where(y==1)[0];out=[]
 for name in ['Logistic Regression','Random Forest','XGBoost','SVM','KNN','Naive Bayes','Decision Tree']:
  a=piv['Final procedure'].to_numpy();b=piv[name].to_numpy();delta=roc_auc_score(y,a)-roc_auc_score(y,b);rng=np.random.default_rng(SEED);ds=[]
  for _ in range(2000):
   ix=np.r_[rng.choice(i0,len(i0),True),rng.choice(i1,len(i1),True)];ds.append(roc_auc_score(y[ix],a[ix])-roc_auc_score(y[ix],b[ix]))
  # Center bootstrap differences to approximate the null, with finite-sample correction.
  pv=(1+sum(abs(np.asarray(ds)-delta)>=abs(delta)))/(len(ds)+1)
  out.append(dict(comparator=name,AUROC_difference=delta,CI_low=np.quantile(ds,.025),CI_high=np.quantile(ds,.975),p_value=pv))
 q=pd.DataFrame(out);order=np.argsort(q.p_value);adj=np.minimum.accumulate((q.p_value.to_numpy()[order]*len(q)/np.arange(1,len(q)+1))[::-1])[::-1];q['p_BH']=np.nan;q.loc[order,'p_BH']=np.minimum(adj,1);q.to_csv(R/'paired_model_comparisons.csv',index=False)
 folds=pd.read_csv(R/'development_oof_record_predictions.csv');fr=[]
 for fold,g in folds[folds.model.eq('Final procedure')].groupby('fold'):
  z=g.groupby('group_key').agg(y=('y','first'),p=('p','mean'),cutoff=('cutoff','first')).reset_index();fr.append(dict(fold=int(fold)+1,**metrics(z,ci=False)))
 pd.DataFrame(fr).to_csv(R/'final_procedure_fold_performance.csv',index=False)
if __name__=='__main__':main()
