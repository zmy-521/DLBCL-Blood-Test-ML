"""Nested group-aware development; external access only after model freeze."""
from pathlib import Path
import os,sys,json,hashlib,time,warnings
from config import O, F, prepare_run
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve
from xgboost import XGBClassifier
warnings.filterwarnings('ignore',message='X does not have valid feature names')
warnings.filterwarnings('ignore',category=FutureWarning)
SEED=20261007;R=O/'results_v2';M=O/'models_v2'
# Candidate order is loaded from public metadata by config.
PARAMS={'Logistic Regression':[{'C':.1},{'C':1.}], 'Random Forest':[{'min_samples_leaf':2},{'min_samples_leaf':5}], 'XGBoost':[{'max_depth':2},{'max_depth':3}], 'SVM':[{'C':.5},{'C':2.}], 'KNN':[{'n_neighbors':5},{'n_neighbors':15}], 'Naive Bayes':[{'var_smoothing':1e-9},{'var_smoothing':.01}], 'Decision Tree':[{'max_depth':3},{'max_depth':5}]}
def splits(y,g,n=3,seed=SEED):
 out=list(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,g))
 for a,b in out:assert not set(g[a])&set(g[b]) and len(set(y[b]))==2
 return out
def estimator(name,params,y,g):
 if name=='Logistic Regression':mod=LogisticRegression(max_iter=2000,random_state=SEED,**params)
 elif name=='Random Forest':mod=RandomForestClassifier(n_estimators=120,n_jobs=2,random_state=SEED,**params)
 elif name=='XGBoost':mod=XGBClassifier(n_estimators=120,learning_rate=.05,subsample=.9,colsample_bytree=.9,min_child_weight=3,n_jobs=2,random_state=SEED,eval_metric='logloss',**params)
 elif name=='SVM':mod=SVC(kernel='rbf',probability=False,random_state=SEED,**params)
 elif name=='KNN':mod=KNeighborsClassifier(**params)
 elif name=='Naive Bayes':mod=GaussianNB(**params)
 else:mod=DecisionTreeClassifier(min_samples_leaf=3,random_state=SEED,**params)
 pipe=Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler() if name in ['Logistic Regression','SVM','KNN'] else 'passthrough'),('model',mod)])
 if name=='SVM':return CalibratedClassifierCV(pipe,method='sigmoid',cv=splits(y,g,3),ensemble=False,n_jobs=1)
 return pipe
def aggregate(y,p,g):
 z=pd.DataFrame({'y':y,'p':p,'group_key':g});assert z.groupby('group_key').y.nunique().max()==1
 return z.groupby('group_key',sort=True).agg(y=('y','first'),p=('p','mean')).reset_index()
def score(y,p,g):
 a=aggregate(y,p,g);return roc_auc_score(a.y,a.p)
def threshold(y,p,g):
 a=aggregate(y,p,g);f,t,c=roc_curve(a.y,a.p);j=t-f;j[~np.isfinite(c)]=-np.inf
 return float(c[np.argmax(j)])
def cvfit(name,param,X,y,g,ss):
 p=np.full(len(y),np.nan);scores=[]
 for a,b in ss:
  mod=estimator(name,param,y[a],g[a]);mod.fit(X[a],y[a]);p[b]=mod.predict_proba(X[b])[:,1];scores.append(score(y[b],p[b],g[b]))
 return p,scores
def tune(X,y,g,label):
 ss=splits(y,g);scores=[];best={}
 for name,params in PARAMS.items():
  candidates=[]
  for par in params:
   p,s=cvfit(name,par,X,y,g,ss);candidates.append((float(np.mean(s)),par,p,s));scores.append(dict(stage=label,model=name,parameters=json.dumps(par),mean_auc=float(np.mean(s)),fold_aucs=json.dumps(s)))
  z=max(candidates,key=lambda v:v[0]);best[name]={'auc':z[0],'params':z[1],'oof':z[2],'cutoff':threshold(y,z[2],g)}
 return best,scores
def rank(X,y):
 imp=SimpleImputer(strategy='median',keep_empty_features=True);z=imp.fit_transform(X)
 rf=RandomForestClassifier(n_estimators=120,max_depth=5,min_samples_leaf=3,n_jobs=2,random_state=SEED);rf.fit(z,y)
 return np.argsort(-rf.feature_importances_,kind='stable')
def feature_select(name,param,X,y,g,label):
 ss=splits(y,g);ranks=[rank(X[a],y[a]) for a,b in ss];curve=[];preds={}
 for k in range(1,len(F)+1):
  pp=np.full(len(y),np.nan);sc=[]
  for (a,b),rr in zip(ss,ranks):
   ix=rr[:k];mod=estimator(name,param,y[a],g[a]);mod.fit(X[a][:,ix],y[a]);pp[b]=mod.predict_proba(X[b][:,ix])[:,1];sc.append(score(y[b],pp[b],g[b]))
  curve.append(dict(stage=label,model=name,n_features=k,mean_auc=float(np.mean(sc)),se_auc=float(np.std(sc,ddof=1)/np.sqrt(len(sc))),fold_aucs=json.dumps(sc)));preds[k]=pp
  if k%10==0:print(label,'feature curve',k,flush=True)
 best=max(curve,key=lambda z:z['mean_auc']);limit=best['mean_auc']-best['se_auc'];k=min(z['n_features'] for z in curve if z['mean_auc']>=limit)
 return k,rank(X,y)[:k],curve,preds[k]
def writej(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def main():
 """Run the unchanged nested procedure; freeze before opening external tables."""
 prepare_run()
 from validate_inputs import validate_development
 validate_development()
 if (R/'external_record_predictions.csv').exists():
  raise RuntimeError('Completed frozen run exists. Reproduce in a separate output copy; existing frozen models and external predictions will not be overwritten.')
 dev=pd.read_csv(O/'data_v2/development_grouped_anonymous.csv');strict=pd.read_csv(O/'data_v2/development_strict_baseline_anonymous.csv')
 X=dev[F].to_numpy(float);y=dev.group.to_numpy(int);g=dev.group_key.to_numpy();outer=splits(y,g,5)
 foldmap={pid:i for i,(a,b) in enumerate(outer) for pid in set(g[b])};assert len(foldmap)==dev.group_key.nunique()
 writej(R/'development_fold_assignment.json',foldmap)
 # Once-per-patient controls and grouped cases remain wholly within each split.
 for fold,(tr,va) in enumerate(outer):
  ck=R/f'outer_fold_{fold}.joblib'
  if ck.exists():print('Reuse completed outer fold',fold,flush=True);continue
  print('OUTER FOLD',fold+1,'of 5',flush=True)
  best,tuning=tune(X[tr],y[tr],g[tr],f'outer_{fold}');rows=[]
  for name,v in best.items():
   mod=estimator(name,v['params'],y[tr],g[tr]);mod.fit(X[tr],y[tr]);pr=mod.predict_proba(X[va])[:,1]
   for j,p in zip(va,pr):rows.append(dict(model=name,index=int(j),fold=fold,group_key=g[j],y=int(y[j]),p=float(p),cutoff=v['cutoff']))
  chosen=max(best,key=lambda k:best[k]['auc']);par=best[chosen]['params'];k,ix,curve,innerp=feature_select(chosen,par,X[tr],y[tr],g[tr],f'outer_{fold}')
  mod=estimator(chosen,par,y[tr],g[tr]);mod.fit(X[tr][:,ix],y[tr]);pp=mod.predict_proba(X[va][:,ix])[:,1];cut=threshold(y[tr],innerp,g[tr])
  for j,p in zip(va,pp):rows.append(dict(model='Final procedure',index=int(j),fold=fold,group_key=g[j],y=int(y[j]),p=float(p),cutoff=cut))
  # Fixed single-variable comparators evaluated independently in outer folds.
  for fi,f in enumerate(F):
   sm=estimator('Logistic Regression',{'C':1.},y[tr],g[tr]);sm.fit(X[tr][:,[fi]],y[tr]);sp=sm.predict_proba(X[va][:,[fi]])[:,1]
   ip,_=cvfit('Logistic Regression',{'C':1.},X[tr][:,[fi]],y[tr],g[tr],splits(y[tr],g[tr]));sc=threshold(y[tr],ip,g[tr])
   for j,p in zip(va,sp):rows.append(dict(model='Single: '+f,index=int(j),fold=fold,group_key=g[j],y=int(y[j]),p=float(p),cutoff=sc))
  # Baseline-only fit; selection above never saw held-out patients.
  sb=strict[strict.group_key.map(foldmap).eq(fold)];sa=strict[~strict.group_key.map(foldmap).eq(fold)];sx=sa[F].to_numpy(float);sy=sa.group.to_numpy(int);sg=sa.group_key.to_numpy()
  bm=estimator(chosen,par,sy,sg);bm.fit(sx[:,ix],sy);bp=bm.predict_proba(sb[F].to_numpy(float)[:,ix])[:,1]
  bip,_=cvfit(chosen,par,sx[:,ix],sy,sg,splits(sy,sg));bcut=threshold(sy,bip,sg)
  baseline=[dict(model='Strict baseline sensitivity',index=-1,fold=fold,group_key=r.group_key,y=int(r.group),p=float(p),cutoff=bcut) for r,p in zip(sb.itertuples(),bp)]
  joblib.dump(dict(rows=rows,baseline=baseline,tuning=tuning,curve=curve,selected=dict(fold=fold,model=chosen,params=par,n_features=k,features=[F[i] for i in ix],cutoff=cut)),ck)
  print('Finished fold',fold+1,'selected',chosen,k,'features',flush=True)
 runs=[joblib.load(R/f'outer_fold_{i}.joblib') for i in range(5)];oof=pd.DataFrame([r for z in runs for r in z['rows']+z['baseline']]);oof.to_csv(R/'development_oof_record_predictions.csv',index=False)
 pd.DataFrame([r for z in runs for r in z['curve']]).to_csv(R/'nested_feature_number_curves.csv',index=False);writej(R/'outer_selected_models.json',[z['selected'] for z in runs])
 fullck=R/'full_development_selection.joblib'
 if fullck.exists():full=joblib.load(fullck);best,tuning,chosen,par,k,ix,curve= [full[x] for x in ['best','tuning','chosen','par','k','ix','curve']]
 else:
  print('Full development selection',flush=True);best,tuning=tune(X,y,g,'full_development');chosen=max(best,key=lambda n:best[n]['auc']);par=best[chosen]['params'];k,ix,curve,_=feature_select(chosen,par,X,y,g,'full_development');joblib.dump(dict(best=best,tuning=tuning,chosen=chosen,par=par,k=k,ix=ix,curve=curve),fullck)
 pd.DataFrame(curve).to_csv(R/'feature_number_performance.csv',index=False);pd.DataFrame([r for z in runs for r in z['tuning']]+tuning).to_csv(R/'hyperparameter_search.csv',index=False)
 frozen={};meta={}
 for name,v in best.items():
  mo=estimator(name,v['params'],y,g);mo.fit(X,y);frozen[name]=(mo,list(range(40)));op=oof[oof.model.eq(name)];meta[name]=dict(params=v['params'],features=F,cutoff=threshold(op.y,op.p,op.group_key))
 mo=estimator(chosen,par,y,g);mo.fit(X[:,ix],y);frozen['Final procedure']=(mo,ix.tolist());op=oof[oof.model.eq('Final procedure')];cut=threshold(op.y,op.p,op.group_key);meta['Final procedure']=dict(algorithm=chosen,params=par,features=[F[i] for i in ix],cutoff=cut,feature_count_rule='smallest k within one SE of best inner-CV AUROC')
 sx=strict[F].to_numpy(float);sy=strict.group.to_numpy(int);sg=strict.group_key.to_numpy();bm=estimator(chosen,par,sy,sg);bm.fit(sx[:,ix],sy);frozen['Strict baseline sensitivity']=(bm,ix.tolist());op=oof[oof.model.eq('Strict baseline sensitivity')];meta['Strict baseline sensitivity']=dict(algorithm=chosen,params=par,features=[F[i] for i in ix],cutoff=threshold(op.y,op.p,op.group_key))
 for i,f in enumerate(F):
  mo=estimator('Logistic Regression',{'C':1.},y,g);mo.fit(X[:,[i]],y);name='Single: '+f;frozen[name]=(mo,[i]);op=oof[oof.model.eq(name)];meta[name]=dict(params={'C':1.},features=[f],cutoff=threshold(op.y,op.p,op.group_key))
 joblib.dump(frozen,M/'frozen_models.joblib');writej(M/'frozen_model_specification.json',dict(seed=SEED,models=meta,source_hashes={fn:hashlib.sha256((O/'data_v2'/fn).read_bytes()).hexdigest() for fn in ['development_grouped_anonymous.csv','development_strict_baseline_anonymous.csv']},freeze_utc=pd.Timestamp.now(tz='UTC').isoformat()))
 print('MODEL FROZEN:',chosen,k,'features',meta['Final procedure']['features'],flush=True)
 # This is the single final external prediction pass for all prespecified models.
 ep=R/'external_record_predictions.csv'
 if ep.exists():raise RuntimeError('External prediction output already exists; not re-evaluating.')
 from validate_inputs import validate_external
 validate_external()
 ext=pd.read_csv(O/'data_v2/external_grouped_anonymous.csv');eb=pd.read_csv(O/'data_v2/external_strict_baseline_anonymous.csv');rows=[]
 for name,(mo,cols) in frozen.items():
  data=eb if name=='Strict baseline sensitivity' else ext;pr=mo.predict_proba(data[F].to_numpy(float)[:,cols])[:,1]
  for r,p in zip(data.itertuples(),pr):rows.append(dict(model=name,group_key=r.group_key,y=int(r.group),p=float(p),cutoff=meta[name]['cutoff']))
 pd.DataFrame(rows).to_csv(ep,index=False);writej(R/'external_evaluation_receipt.json',dict(prediction_passes=1,models_frozen_before_read=True,frozen_model_sha256=hashlib.sha256((M/'frozen_models.joblib').read_bytes()).hexdigest(),timestamp_utc=pd.Timestamp.now(tz='UTC').isoformat()))
 print('Completed external prediction pass.',flush=True)
if __name__=='__main__':main()
