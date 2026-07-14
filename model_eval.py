import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

preds = pd.read_csv('test_predictions.csv')
y = preds['Target_Direction'].values

print('=== MODEL PERFORMANCE ON TEST SET (2025-05 to 2026-07) ===')
print(f'Test set size: {len(preds)} rows\n')

threshold = preds['Ensemble_Prob'].mean()
for name, prob_col, thr in [
    ('CatBoost',  'Cat_Prob',      0.5),
    ('XGBoost',   'XGB_Prob',      0.5),
    ('LightGBM',  'LGB_Prob',      0.5),
    ('Ensemble',  'Ensemble_Prob', threshold),
]:
    prob = preds[prob_col].values
    p = (prob >= thr).astype(int)
    acc  = accuracy_score(y, p)
    auc  = roc_auc_score(y, prob)
    f1   = f1_score(y, p, zero_division=0)
    prec = precision_score(y, p, zero_division=0)
    rec  = recall_score(y, p, zero_division=0)
    print(f'  {name:10s}  Acc={acc:.3f}  AUC={auc:.3f}  F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}')

print()
print('=== SIGNAL DISTRIBUTION ===')
print(preds['Signal'].value_counts().to_string())

print()
print('=== ENSEMBLE_PROB STATS ===')
print(preds[['Ensemble_Prob','Cat_Prob','XGB_Prob','LGB_Prob']].describe().round(4).to_string())
