"""Repeated splits: model and threshold chosen without looking at test labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, precision_score

def main():
 data=load_breast_cancer(as_frame=True); X=data.data; y=(data.target==0).astype(int); records=[]
 for seed in range(2026,2036):
  train,test,ytrain,ytest=train_test_split(X,y,test_size=.2,stratify=y,random_state=seed)
  fit,val,yfit,yval=train_test_split(train,ytrain,test_size=.25,stratify=ytrain,random_state=seed)
  models={'logistic':make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced',random_state=seed)), 'forest':RandomForestClassifier(n_estimators=200,min_samples_leaf=2,class_weight='balanced',random_state=seed,n_jobs=1)}
  scores={}
  for name,model in models.items():
   model.fit(fit,yfit); scores[name]=average_precision_score(yval,model.predict_proba(val)[:,1])
  chosen=max(scores,key=scores.get); calibrated=CalibratedClassifierCV(models[chosen],method='sigmoid',cv=3).fit(fit,yfit)
  pv=calibrated.predict_proba(val)[:,1]; candidates=[]
  for threshold in np.linspace(.001,.999,999):
   flag=pv>=threshold
   if 0<flag.mean()<=.3:candidates.append((100*int(yval[flag].sum())-5*int(flag.sum()),threshold))
  _,t=max(candidates,key=lambda row:row[0]); pt=calibrated.predict_proba(test)[:,1]; pred=pt>=t
  records.append({'seed':seed,'model':chosen,'threshold':t,'roc_auc':roc_auc_score(ytest,pt),'average_precision':average_precision_score(ytest,pt),'precision':precision_score(ytest,pred,zero_division=0),'recall':recall_score(ytest,pred,zero_division=0),'test_flag_rate':pred.mean()})
  print(f'Completed {len(records)}/10: seed={seed}, model={chosen}, AUROC={records[-1]["roc_auc"]:.4f}',flush=True)
 out=Path(__file__).resolve().parents[1]/'outputs'/'robustness';out.mkdir(parents=True,exist_ok=True)
 frame=pd.DataFrame(records);frame.to_csv(out/'repeated_splits.csv',index=False)
 summary={'n_splits':len(frame),'description':'Repeated stratified holdout; test sets overlap, not independent external validation','metrics':{c:{'mean':float(frame[c].mean()),'min':float(frame[c].min()),'max':float(frame[c].max())} for c in ['roc_auc','average_precision','precision','recall','test_flag_rate']},'test_budget_exceeded_count':int((frame.test_flag_rate>.3).sum())}
 (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
