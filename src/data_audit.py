"""Produce reproducible dataset-level descriptive statistics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

data=load_breast_cancer(as_frame=True)
X=data.data
y=pd.Series(np.where(data.target==0,'malignant','benign'),index=X.index,name='diagnosis')
out=Path(__file__).resolve().parents[1]/'outputs'/'data_audit'
out.mkdir(parents=True,exist_ok=True)
summary={'rows':len(X),'features':X.shape[1],'missing_cells':int(X.isna().sum().sum()),'duplicate_feature_rows':int(X.duplicated().sum()),'diagnosis_counts':{str(k):int(v) for k,v in y.value_counts().items()}}
(out/'data_quality.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
X.describe().T.to_csv(out/'feature_summary.csv',index_label='feature')
X.groupby(y).mean().T.to_csv(out/'class_feature_means.csv',index_label='feature')
print(json.dumps(summary,indent=2))
