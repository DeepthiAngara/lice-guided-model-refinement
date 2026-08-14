#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, brier_score_loss, cohen_kappa_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score)

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'predictions/test/eight_configuration_test_predictions_wide.csv.gz'

old=pd.read_csv(OLD)
sources={
 'Baseline':(old.Source_Row_Index,old.y_true,old.baseline_prediction,old.baseline_probability),
 'M3-Balanced':(old.Source_Row_Index,old.y_true,old.m3_balanced_prediction,old.m3_balanced_probability),
 'M3-High':(old.Source_Row_Index,old.y_true,old.m3_high_prediction,old.m3_high_probability),
}
for model,file in [('Jose-LightGBM','jose_lightgbm_test_predictions.csv.gz'),('Pang-MARS','pang_mars_test_predictions.csv.gz')]:
 d=pd.read_csv(ROOT/'predictions/recent_comparators'/file).sort_values('Test_Position')
 sources[model]=(d.Source_Row_Index,d.y_true,d.Predicted_Class,d.Predicted_Probability)

base_ids=np.asarray(sources['Baseline'][0],int); base_y=np.asarray(sources['Baseline'][1],int)
checks=[]
def ck(n,c,o,e): checks.append({'Check':n,'Observed':str(o),'Expected':str(e),'Status':'PASS' if c else 'FAIL'})
def metric(y,p,q):
 tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
 return {'TN':tn,'FP':fp,'FN':fn,'TP':tp,'Accuracy':accuracy_score(y,p),'AUC':roc_auc_score(y,q),
 'Recall':recall_score(y,p),'Precision':precision_score(y,p),'F1':f1_score(y,p),
 'Kappa':cohen_kappa_score(y,p),'MCC':matthews_corrcoef(y,p),'Brier':brier_score_loss(y,q)}

rows=[]; aligned={}
for model,(ids,y,p,q) in sources.items():
 frame=pd.DataFrame({'id':np.asarray(ids,int),'y':np.asarray(y,int),'p':np.asarray(p,int),'q':np.asarray(q,float)})
 frame=frame.set_index('id').loc[base_ids].reset_index()
 ck(model+' source-row linkage',np.array_equal(frame.id,base_ids),int(np.sum(frame.id!=base_ids)),0)
 ck(model+' label linkage',np.array_equal(frame.y,base_y),int(np.sum(frame.y!=base_y)),0)
 ck(model+' threshold agreement',np.array_equal(frame.p,(frame.q>=.5).astype(int)),int(np.sum(frame.p!=(frame.q>=.5))),0)
 aligned[model]=(frame.p.to_numpy(),frame.q.to_numpy())
 rows.append({'Model':model,**metric(base_y,*aligned[model])})

# Exact paired McNemar; Holm family = six comparator-reference comparisons.
def holm(vals):
 vals=np.asarray(vals); order=np.argsort(vals); z=np.maximum.accumulate((len(vals)-np.arange(len(vals)))*vals[order]); z=np.minimum(z,1); out=np.empty_like(z); out[order]=z; return out
mr=[]
for comp in ['Jose-LightGBM','Pang-MARS']:
 for ref in ['Baseline','M3-Balanced','M3-High']:
  rp=aligned[ref][0]; cp=aligned[comp][0]; rc=rp==base_y; cc=cp==base_y
  b=int(np.sum(rc&~cc)); c=int(np.sum(~rc&cc)); n=b+c
  pv=binomtest(min(b,c),n,.5).pvalue if n else 1.
  mr.append({'Comparison':f'{comp} vs {ref}','Reference_Correct_Comparator_Wrong':b,
   'Reference_Wrong_Comparator_Correct':c,'Discordant_Total':n,'Exact_Two_Sided_P_Value':pv})
m=pd.DataFrame(mr);m['Holm_Adjusted_P_Value']=holm(m.Exact_Two_Sided_P_Value);m['Significant_After_Holm_0.05']=m.Holm_Adjusted_P_Value<.05

reported=pd.read_csv(ROOT/'results/recent_comparators/recent_comparator_performance_full_precision.csv').query("Scope=='Test'")
calc=pd.DataFrame(rows).set_index('Model')
for _,x in reported.iterrows():
 dif=max(abs(calc.loc[x.Model,k]-x[k]) for k in ['TN','FP','FN','TP','Accuracy','AUC','Recall','Precision','F1','Kappa','MCC','Brier'])
 ck(x.Model+' reported metric reproduction',dif<=1e-12,dif,'<=1e-12')
v=pd.DataFrame(checks)
summary={'checks':len(v),'passed':int((v.Status=='PASS').sum()),'failed':int((v.Status!='PASS').sum())}
print(pd.DataFrame(rows).to_string(index=False));print('\n',m.to_string(index=False));print('\n',summary)
print('VERIFICATION_SUMMARY ' + json.dumps(summary))
if summary['failed']: raise SystemExit(1)
