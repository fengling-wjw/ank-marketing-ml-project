"""Leakage-aware, calibrated Bank Marketing experiment."""
from __future__ import annotations
import argparse, json, time, zipfile
from pathlib import Path
from urllib.request import urlretrieve
import joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/raw'; OUT=ROOT/'outputs'; SEED=2026
def get_data(url):
    DATA.mkdir(parents=True,exist_ok=True); csv=DATA/'bank-additional-full.csv'
    if not csv.exists():
        z=DATA/'bank.zip'; print('[data] downloading'); urlretrieve(url,z)
        with zipfile.ZipFile(z) as f, f.open(next(x for x in f.namelist() if x.endswith('bank-additional-full.csv'))) as i, csv.open('wb') as o:o.write(i.read())
    return csv
def prep(X):
    n=X.select_dtypes(include=np.number).columns; c=X.columns.difference(n)
    return ColumnTransformer([('n',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),n),('c',Pipeline([('i',SimpleImputer(strategy='most_frequent')),('o',OneHotEncoder(handle_unknown='ignore'))]),c)])
def metrics(y,p,t):
    q=p>=t
    return {'roc_auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p)),'precision':float(precision_score(y,q,zero_division=0)),'recall':float(recall_score(y,q,zero_division=0)),'contact_rate':float(q.mean())}
def pick_threshold(y,p,budget=.2):
    z=[]
    for t in np.linspace(.02,.9,177):
        q=p>=t
        if 0<q.mean()<=budget:z.append((t,q.mean(),50*y[q].sum()-q.sum()))
    a=pd.DataFrame(z,columns=['threshold','contact_rate','utility']);return float(a.loc[a.utility.idxmax(),'threshold']),a
def main(url):
    OUT.mkdir(exist_ok=True); start=time.time(); d=pd.read_csv(get_data(url),sep=';').drop_duplicates(); y=(d.pop('y')=='yes').astype(int)
    d['never_previously_contacted']=(d.pdays==999).astype(int);d['pdays']=d.pdays.replace(999,np.nan);X=d.drop(columns='duration') # duration is post-call leakage
    xa,xt,ya,yt=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED);xf,xv,yf,yv=train_test_split(xa,ya,test_size=.25,stratify=ya,random_state=SEED)
    candidates={'logistic':LogisticRegression(max_iter=1500,class_weight='balanced',random_state=SEED),'random_forest':RandomForestClassifier(n_estimators=350,min_samples_leaf=3,class_weight='balanced_subsample',n_jobs=-1,random_state=SEED)}
    rows=[];models={}
    for name,m in candidates.items():
        pipe=Pipeline([('prep',prep(xf)),('model',m)]).fit(xf,yf);p=pipe.predict_proba(xv)[:,1];rows.append({'model':name,**metrics(yv,p,.5)});models[name]=pipe;print(name,rows[-1])
    board=pd.DataFrame(rows).sort_values('pr_auc',ascending=False);winner=board.iloc[0].model;model=CalibratedClassifierCV(models[winner],method='sigmoid',cv=3).fit(xa,ya);pv=model.predict_proba(xv)[:,1];t,curve=pick_threshold(yv.reset_index(drop=True),pv);p=model.predict_proba(xt)[:,1]
    imp=permutation_importance(model,xt,yt,n_repeats=8,scoring='average_precision',random_state=SEED,n_jobs=-1)
    importance=pd.DataFrame({'feature':xt.columns,'mean_drop_pr_auc':imp.importances_mean}).sort_values('mean_drop_pr_auc',ascending=False)
    # Robustness audit: the same operating threshold is assessed in age groups.
    groups=pd.cut(xt.age,bins=[0,30,45,60,120],labels=['<=30','31-45','46-60','>60'])
    segments=[]
    for g in groups.cat.categories:
        mask=groups==g
        if mask.sum()>20: segments.append({'age_group':str(g),'n':int(mask.sum()),'observed_rate':float(yt[mask].mean()),**metrics(yt[mask],p[mask],t)})
    fig,ax=plt.subplots(1,2,figsize=(11,4));ax[0].plot(curve.contact_rate,curve.utility);ax[0].set(xlabel='Contact rate',ylabel='Utility',title='Budget threshold search');f,o=calibration_curve(yt,p,n_bins=10);ax[1].plot([0,1],[0,1],'--');ax[1].plot(o,f,'o-');ax[1].set(xlabel='Predicted probability',ylabel='Observed rate',title='Calibration');fig.tight_layout();fig.savefig(OUT/'evaluation.png',dpi=180)
    board.to_csv(OUT/'leaderboard.csv',index=False);importance.to_csv(OUT/'permutation_importance.csv',index=False);curve.to_csv(OUT/'threshold_curve.csv',index=False);pd.DataFrame(segments).to_csv(OUT/'age_segment_metrics.csv',index=False);joblib.dump(model,OUT/'model.joblib')
    summary={'dataset_rows':len(X),'positive_rate':float(y.mean()),'leakage_feature_removed':'duration','winner':winner,'threshold':t,'budget':.2,'test_metrics':metrics(yt,p,t),'runtime_seconds':round(time.time()-start,1),'utility_assumption':'50 utility units per success; 1 per contact'}
    (OUT/'run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data-url',required=True);main(a.parse_args().data_url)
