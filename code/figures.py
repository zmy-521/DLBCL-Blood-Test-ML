"""Frozen v2 figure and permutation-SHAP workflow; explicit execution only."""
def main():
    from pathlib import Path
    import sys,json,warnings
    from config import O, F, prepare_run
    import numpy as np,pandas as pd,joblib,shap
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch,Rectangle
    from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
    from scipy.stats import norm
    warnings.filterwarnings('ignore',category=FutureWarning)
    G=O/'figures_v2';S=O/'supplementary_figures_v2';R=O/'results_v2';G.mkdir(exist_ok=True);S.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.labelsize':10,'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'legend.frameon':False,'figure.facecolor':'white','savefig.facecolor':'white'})
    COL=['#3A678B','#B56459','#64977D','#92749E','#CAAB55','#699DA6','#777777'];MODELS=['Logistic Regression','Random Forest','XGBoost','SVM','KNN','Naive Bayes','Decision Tree']
    def save(fig,name,folder=G):
     fig.savefig(folder/(name+'.png'),dpi=400,bbox_inches='tight');fig.savefig(folder/(name+'.tiff'),dpi=400,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'});plt.close(fig)
    def loadpred(fn):
     d=pd.read_csv(R/fn);return d.groupby(['model','group_key']).agg(y=('y','first'),p=('p','mean')).reset_index()
    dev=loadpred('development_oof_record_predictions.csv');ext=loadpred('external_record_predictions.csv')
    spec=json.loads((O/'models_v2/frozen_model_specification.json').read_text(encoding='utf8'))['models']['Final procedure'];features=spec['features']
    # Figure 1: schematic-led study design, with cohort selection and validation paths.
    fig,ax=plt.subplots(figsize=(9,6));ax.set(xlim=(0,10),ylim=(0,8));ax.axis('off')
    def box(x,y,w,h,txt,color):
     ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor=color,edgecolor='#71808B',linewidth=.7));ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=10)
    def arrow(x,y,a,b):ax.annotate('',xy=(a,b),xytext=(x,y),arrowprops={'arrowstyle':'->','color':'#52616C','lw':1.2})
    box(.2,6.2,4.2,1.1,'Wanbei development center\nRaw LIS reconstruction + stable identity linkage\nInvestigator-confirmed pretreatment specimens','#EDF2F5')
    box(5.6,6.2,4.2,1.1,'Anhui Medical University external center\nIndependent raw LIS reconstruction\nInvestigator-confirmed pretreatment specimens','#EFF3ED')
    box(.2,4.3,4.2,1.2,'Primary: 65 DLBCL + 430 healthy controls\n285 case panels + 430 control panels\nSame-patient, unique same-day panel pairing','#EDF2F5')
    box(5.6,4.3,4.2,1.2,'Primary: 131 DLBCL + 248 healthy controls\n240 case panels + 248 control panels\nSampling date preferred; test-date proxy flagged','#EFF3ED')
    arrow(2.3,6.15,2.3,5.6);arrow(7.7,6.15,7.7,5.6)
    box(.2,2.25,4.2,1.3,'Nested stratified group CV: outer 5 / inner 3\n7 algorithms; training-only preprocessing\nTraining-only algorithm / feature-count selection\nNo patient crosses folds','#EDF2F5')
    box(5.6,2.25,4.2,1.3,'One external prediction pass after freezing\nNo tuning, feature selection or cutoff selection\nPatient metric = mean panel probability\nAll external comparators frozen in development','#EFF3ED')
    arrow(2.3,4.2,2.3,3.65);arrow(7.7,4.2,7.7,3.65);arrow(4.6,2.9,5.4,2.9);ax.text(5,3.2,'Freeze',ha='center',fontsize=9)
    box(.2,.35,4.2,1.1,'Strict-baseline sensitivity\n65 DLBCL + 430 controls; one panel/person\nEarliest eligible paired sampling episode','#F5F4F0')
    box(5.6,.35,4.2,1.1,'Strict-baseline sensitivity\n41 DLBCL + 248 controls; one panel/person\nReliable sampling chronology required','#F5F4F0')
    arrow(2.3,2.15,2.3,1.55);arrow(7.7,2.15,7.7,1.55)
    fig.suptitle('DLBCL versus healthy-control discrimination',fontsize=14,y=.98);save(fig,'Figure_1')
    # Figure 2: seven prespecified models, patient-aggregated predictions.
    fig,axs=plt.subplots(1,2,figsize=(9,4.4))
    for ax,data,title in zip(axs,[dev,ext],['A  Development nested OOF (65 cases / 430 controls)','B  Independent external (131 cases / 248 controls)']):
     for model,color in zip(MODELS,COL):
      z=data[data.model.eq(model)];f,t,_=roc_curve(z.y,z.p);ax.plot(f,t,label=f'{model}: {roc_auc_score(z.y,z.p):.3f}',color=color,lw=1.35)
     ax.plot([0,1],[0,1],':',color='#999999');ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='1 − Specificity',ylabel='Sensitivity',title=title);ax.legend(loc='lower right',fontsize=8)
    fig.tight_layout();save(fig,'Figure_2')
    # Final frozen RF SHAP, distinct development patients, no external explanation fitting.
    models=joblib.load(O/'models_v2/frozen_models.joblib');model,cols=models['Final procedure'];allb=pd.read_csv(O/'data_v2/development_strict_baseline_anonymous.csv');rng=np.random.default_rng(20261007);bg=allb[features].to_numpy(float)[rng.choice(len(allb),40,replace=False)];b=allb.iloc[np.sort(rng.choice(len(allb),200,replace=False))].reset_index(drop=True);xx=b[features].to_numpy(float);imputed=model.named_steps['imputer'].transform(xx)
    model.named_steps['model'].set_params(n_jobs=1) # execution parallelism only, no fitted-state change
    svpath=R/'final_permutation_shap_values.npz'
    if svpath.exists():z=np.load(svpath);sv=z['values'];base=z['base'];probs=z['probabilities']
    else:
     ex=shap.Explainer(model.predict_proba,shap.maskers.Independent(bg),algorithm='permutation',feature_names=features,seed=20261007);v=ex(xx,max_evals=5*(2*len(features)+1),silent=True);sv=v.values[:,:,1];base=v.base_values[:,1];probs=model.predict_proba(xx)[:,1];assert np.max(abs(base+sv.sum(axis=1)-probs))<1e-8;np.savez(svpath,values=sv,base=base,probabilities=probs)
    assert np.max(abs(base+sv.sum(axis=1)-probs))<1e-8
    rank=np.argsort(-np.abs(sv).mean(axis=0));pd.DataFrame({'feature':features,'mean_absolute_SHAP':np.abs(sv).mean(axis=0)}).sort_values('mean_absolute_SHAP',ascending=False).to_csv(R/'final_shap_importance.csv',index=False)
    # Figure 3: explanation and training-only reduction are explicitly distinguished.
    fig,axs=plt.subplots(1,3,figsize=(12,4.7),gridspec_kw={'width_ratios':[1.3,1,1.3]})
    for yy,j in enumerate(rank[::-1]):
     values=imputed[:,j];lo,hi=np.quantile(values,[.05,.95]);c=np.clip((values-lo)/(hi-lo+1e-8),0,1);axs[0].scatter(sv[:,j],yy+rng.uniform(-.25,.25,len(b)),c=c,cmap='coolwarm',s=5,alpha=.65,rasterized=True)
    axs[0].axvline(0,color='#888888',lw=.6);axs[0].set(yticks=range(len(features)),yticklabels=np.array(features)[rank[::-1]],xlabel='SHAP value (probability contribution)\nBlue: low value; red: high value',title='A  Frozen RF SHAP summary')
    axs[1].barh(range(len(features)),np.abs(sv).mean(axis=0)[rank[::-1]],color='#6D8CA4');axs[1].set(yticks=range(len(features)),yticklabels=np.array(features)[rank[::-1]],xlabel='Mean |SHAP value|',title='B  Explanatory importance')
    cv=pd.read_csv(R/'feature_number_performance.csv');axs[2].plot(cv.n_features,cv.mean_auc,color='#3A678B',lw=1.5);axs[2].fill_between(cv.n_features,cv.mean_auc-cv.se_auc,cv.mean_auc+cv.se_auc,color='#3A678B',alpha=.15);axs[2].axvline(len(features),ls='--',color='#B56459',label=f'Selected k={len(features)}');best=cv.loc[cv.mean_auc.idxmax()];axs[2].axhline(best.mean_auc-best.se_auc,ls=':',color='#777777',label='One-SE boundary');axs[2].set(xlabel='Number of features',ylabel='Inner-CV patient AUROC',title='C  Development feature reduction',xlim=(1,40));axs[2].legend(fontsize=8,loc='lower right');fig.tight_layout();save(fig,'Figure_3')
    fig,axs=plt.subplots(2,3,figsize=(10,6.5))
    from tables import units
    for ax,j,letter in zip(axs.flat,rank[:6],'ABCDEF'):
     ax.scatter(imputed[:,j],sv[:,j],c=b.group,cmap=matplotlib.colors.ListedColormap(['#7299B0','#BD786B']),s=12,alpha=.7,edgecolor='none');ax.axhline(0,color='#999999',lw=.7);ax.set(xlabel=f'{features[j]} ({units[features[j]]})',ylabel='SHAP value',title=f'{letter}  {features[j]}')
    fig.suptitle('Frozen RF dependence: 200 sampled development patients\nBlue: healthy controls; red: DLBCL',fontsize=12);fig.tight_layout();save(fig,'Figure_4')
    def calibration(ax,data,title):
     z=data[data.model.eq('Final procedure')].copy();z['bin']=pd.qcut(z.p,5,duplicates='drop');cal=[]
     for _,v in z.groupby('bin',observed=True):
      n=len(v);ph=v.y.mean();zz=norm.ppf(.975);mid=(ph+zz**2/(2*n))/(1+zz**2/n);half=zz*np.sqrt(ph*(1-ph)/n+zz**2/(4*n*n))/(1+zz**2/n);cal.append((v.p.mean(),ph,mid-half,mid+half,n))
     cc=np.array(cal);ax.plot([0,1],[0,1],':',c='#777777',label='Ideal');ax.errorbar(cc[:,0],cc[:,1],yerr=np.vstack([cc[:,1]-cc[:,2],cc[:,3]-cc[:,1]]),color='#3A678B',marker='o',lw=1.2,capsize=3,label='5 quantile bins; Wilson 95% CI');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed DLBCL fraction',title=title);ax.legend(fontsize=7,loc='upper left');return cc
    def waterfall(ax,i,title):
     vals=sv[i];order=np.argsort(-np.abs(vals));top=order[:8];contrib=list(vals[top]);labels=[f'{features[j]} = {imputed[i,j]:.3g}' for j in top]
     if len(order)>8:contrib.append(float(vals[order[8:]].sum()));labels.append(f'Other {len(order)-8} features')
     pos=float(base[i]);lo=pos;hi=pos
     for yy,(v,lab) in enumerate(zip(contrib,labels)):
      nxt=pos+v;ax.barh(yy,abs(v),left=min(pos,nxt),height=.65,color='#B56459' if v>=0 else '#608CA5');ax.text(max(pos,nxt)+.008,yy,f'{v:+.3f}',va='center',fontsize=7);lo=min(lo,nxt);hi=max(hi,nxt);pos=nxt
     ax.axvline(base[i],c='#777777',ls=':',lw=.8);ax.axvline(probs[i],c='#333333',ls='--',lw=.8);ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel=f'Base {base[i]:.3f} → predicted {probs[i]:.3f}',title=title,xlim=(min(-.02,lo-.08),max(1.02,hi+.12)));ax.invert_yaxis()
    fig,axs=plt.subplots(2,2,figsize=(10,8))
    # Deterministic representative closest to median frozen risk within each class.
    for ax,gr,title in zip(axs[0],[1,0],['A  Representative DLBCL patient','B  Representative healthy control']):
     ii=np.where(b.group.to_numpy()==gr)[0];i=ii[np.argmin(abs(probs[ii]-np.median(probs[ii])))];waterfall(ax,i,title)
    c=calibration(axs[1,0],dev,'C  Development nested OOF (n=495)');e=calibration(axs[1,1],ext,'D  Independent external (n=379)');fig.tight_layout();save(fig,'Figure_5')
    pd.DataFrame(c,columns=['predicted','observed','low','high','n']).to_csv(R/'development_calibration_bins.csv',index=False);pd.DataFrame(e,columns=['predicted','observed','low','high','n']).to_csv(R/'external_calibration_bins.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(9,4.4))
    for ax,data,title in zip(axs,[dev,ext],['A  Development nested OOF (n=495)','B  Independent external (n=379)']):
     for name,col in zip(MODELS,COL):
      z=data[data.model.eq(name)];pr,re,_=precision_recall_curve(z.y,z.p);ax.plot(re,pr,color=col,label=f'{name}: {average_precision_score(z.y,z.p):.3f}')
     ax.axhline(z.y.mean(),ls=':',color='#888888');ax.set(xlabel='Recall',ylabel='Precision',title=title,xlim=(0,1),ylim=(0,1.02));ax.legend(fontsize=8)
    fig.tight_layout();save(fig,'Figure_S1_PR',S)
    fig,axs=plt.subplots(1,2,figsize=(9,4.3))
    for data,cohort,col in [(dev,'Development OOF','#3A678B'),(ext,'External','#B56459')]:
     for name,ls in [('Final procedure','-'),('Strict baseline sensitivity','--')]:
      z=data[data.model.eq(name)];f,t,_=roc_curve(z.y,z.p);pr,re,_=precision_recall_curve(z.y,z.p);axs[0].plot(f,t,color=col,ls=ls,label=f'{cohort}, {name}: {roc_auc_score(z.y,z.p):.3f}');axs[1].plot(re,pr,color=col,ls=ls,label=f'{cohort}, {name}: {average_precision_score(z.y,z.p):.3f}')
    axs[0].set(xlabel='1 − Specificity',ylabel='Sensitivity',title='A  Final procedure and baseline sensitivity');axs[1].set(xlabel='Recall',ylabel='Precision',title='B  Final procedure and baseline sensitivity')
    for ax in axs:ax.legend(fontsize=7,loc='lower left');ax.set(xlim=(0,1),ylim=(0,1.02))
    fig.tight_layout();save(fig,'Figure_S2_final_and_baseline',S)
    legends='''# Figure legends (statistical deliverable only)

    Figure 1. Reconstruction and validation flow for DLBCL versus healthy-control discrimination. Primary development: 65 DLBCL/430 controls, 715 panels; external: 131/248, 488 panels. Five outer and three inner stratified group folds prevent patients crossing folds. Strict baseline cohorts: development 65/430 and external 41/248. Test-date episode pairing is distinguished from reliable sampling chronology. No centers were pooled for splitting.

    Figure 2. Patient-level ROC curves for seven prespecified algorithms. A: development nested OOF, 495 patients. B: independent external, 379 patients. Each patient's eligible panel probabilities were averaged. External comparators were all frozen in development; their results were not used to change the final model.

    Figure 3. A–B: probability-scale SHAP explanations of the frozen 13-feature RF on one strict-baseline panel from each of 495 development patients. The background comprises 100 development patients. SHAP explains the frozen fit and is not independent validation. C: full-development training-only three-fold grouped CV feature-number curve, mean ± one SE. RF importance was recalculated within each training fold for reduction; SHAP ranking was not used for selection. The one-SE rule selected 13 features. This selection curve is not an unbiased performance estimate.

    Figure 4. The six highest mean-absolute-SHAP features in the frozen RF, using 495 distinct development patients. Blue: healthy controls; red: DLBCL. Probability-scale SHAP contributions are associational explanations, not causal effects. Final model trained on the full development repeated-measure dataset.

    Figure 5. A–B: deterministic representative development case and control nearest the median predicted probability within their class; probability-scale waterfall contributions. C: development nested OOF patient calibration (65/430). D: independent external patient calibration (131/248). Points are five quantile bins; bars are Wilson 95% intervals. Development OOF evaluates the adaptive selection procedure (algorithms/features may differ across folds); external evaluates the single frozen RF. No DCA or external recalibration was performed.

    Figure S1. Precision–recall curves for seven models using development nested OOF (495 patients) and independent external predictions (379 patients). Dashed baseline represents the case fraction in each cohort. Average precision is reported as AUPRC; it depends on case/control composition.

    Figure S2. Final adaptive procedure versus strict-baseline sensitivity. Primary: development 65/430 and external 131/248. Strict baseline: development 65/430 and external 41/248, one panel/person. The external populations differ; curve differences cannot be attributed solely to removing repeat panels. The strict model was fitted on development baseline panels only, with training-only selection in internal validation.
    '''
    legends=legends.replace('one strict-baseline panel from each of 495 development patients','one strict-baseline panel from each of 200 randomly sampled distinct development patients').replace('background comprises 100 development patients','background comprises 40 development patients; model-agnostic permutation SHAP uses five antithetic permutation cycles with seed 20261007').replace('using 495 distinct development patients','using the same 200 sampled distinct development patients').replace('representative development case and control','representative sampled development case and control')
    (O/'Figure_legends_v2.md').write_text(legends,encoding='utf8');print('Figures generated; SHAP additivity verified.',flush=True)

if __name__=='__main__':main()
