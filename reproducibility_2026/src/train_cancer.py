"""Offline reproducible experiment on sklearn's Wisconsin diagnostic dataset."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

OUT=Path(__file__).resolve().parents[1]/'outputs'/'corrected'; SEED=2026
def metric(y,p,t):
 q=p>=t
 return {'roc_auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p)),'precision':float(precision_score(y,q)),'recall':float(recall_score(y,q)),'flag_rate':float(q.mean())}
def main():
 OUT.mkdir(parents=True,exist_ok=True); ds=load_breast_cancer(as_frame=True); X=ds.data; y=(ds.target==0).astype(int) # malignant=1
 xa,xt,ya,yt=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED); xf,xv,yf,yv=train_test_split(xa,ya,test_size=.25,stratify=ya,random_state=SEED)
 models={'logistic':make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced',random_state=SEED)),'random_forest':RandomForestClassifier(n_estimators=500,min_samples_leaf=2,class_weight='balanced',random_state=SEED)}
 rows=[]; fitted={}
 for n,m in models.items():
  m.fit(xf,yf); p=m.predict_proba(xv)[:,1]; rows.append({'model':n,**metric(yv,p,.5)}); fitted[n]=m
 board=pd.DataFrame(rows).sort_values('pr_auc',ascending=False); name=board.iloc[0].model
 model=CalibratedClassifierCV(fitted[name],method='sigmoid',cv=3).fit(xf,yf); pv=model.predict_proba(xv)[:,1]
 cand=[]
 for t in np.linspace(.001,.999,999):
  q=pv>=t
  if q.mean()<=.30 and q.sum(): cand.append((t,q.mean(),100*yv[q].sum()-5*q.sum()))
 curve=pd.DataFrame(cand,columns=['threshold','flag_rate','utility']); t=float(curve.loc[curve.utility.idxmax(),'threshold']); p=model.predict_proba(xt)[:,1]
 imp=permutation_importance(model,xt,yt,n_repeats=20,scoring='average_precision',random_state=SEED)
 importance=pd.DataFrame({'feature':X.columns,'mean_drop_pr_auc':imp.importances_mean}).sort_values('mean_drop_pr_auc',ascending=False)
 group=pd.qcut(xt['mean radius'],q=3,duplicates='drop'); seg=[]
 for g in group.cat.categories:
  mask=group==g; seg.append({'mean_radius_group':str(g),'n':int(mask.sum()),**metric(yt[mask],p[mask],t)})
 board.to_csv(OUT/'leaderboard.csv',index=False);curve.to_csv(OUT/'threshold_curve.csv',index=False);importance.to_csv(OUT/'permutation_importance.csv',index=False);pd.DataFrame(seg).to_csv(OUT/'segment_metrics.csv',index=False)
 print('Validation model comparison:');print(board.to_string(index=False));print('Threshold:',t);print('Test:',metric(yt,p,t))
 summary={'dataset':'Wisconsin Diagnostic Breast Cancer (sklearn built-in copy)','rows':int(len(X)),'features':int(X.shape[1]),'positive_class':'malignant','positive_rate':float(y.mean()),'seed':SEED,'winner':name,'threshold':t,'budget_constraint':.30,'utility_assumption':'100 units per correctly prioritized malignant case; 5 per flagged case','test_metrics':metric(yt,p,t),'note':'Educational decision-support experiment only; not a clinical device.'}
 (OUT/'run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
