"""
Generate full-period predictions (2022-2026) using production models.
Writes test_predictions_full.csv with the same schema as test_predictions.csv
so that step9_backtest_strategy.py can be pointed at it for a proper
multi-year evaluation with enough signal instances.
"""
import os, sys, json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib

if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')

DIR = os.path.dirname(os.path.abspath(__file__))

# -- Load models ---------------------------------------------------------------
print("Loading production models...")
cat_model = CatBoostClassifier()
cat_model.load_model(os.path.join(DIR, "catboost_prod.cbm"))

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(DIR, "xgb_prod.json"))

lgb_model = lgb.Booster(model_file=os.path.join(DIR, "lgb_prod.txt"))

scaler = joblib.load(os.path.join(DIR, "scaler.pkl"))

with open(os.path.join(DIR, "model_threshold.json")) as f:
    threshold_data = json.load(f)
threshold = threshold_data.get("threshold", 0.5)

# -- Load full dataset ---------------------------------------------------------
print("Loading full master features dataset...")
df = pd.read_csv(os.path.join(DIR, "master_features.csv"))
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
print(f"  Total rows: {len(df):,}  ({df['Date'].min().date()} to {df['Date'].max().date()})")

# -- Build target --------------------------------------------------------------
df['Close_Return'] = np.log(df['Close'] / df['Close'].shift(1))
df['Target_Direction'] = (df['Close_Return'].shift(-1) > 0).astype(int)
df = df.dropna(subset=['Close_Return', 'Target_Direction']).reset_index(drop=True)

# -- Identify feature columns (same as training) -------------------------------
exclude = ['Date', 'Target_Direction', 'Signal', 'Close_Return',
           'Open', 'High', 'Low', 'Close', 'Tick_Volume',
           'POC_20', 'VAH_20', 'VAL_20', 'POC_60', 'VAH_60', 'VAL_60',
           'POC_252', 'VAH_252', 'VAL_252', 'VWAP_20', 'VWAP_60', 'VWAP_252',
           'CVD_20', 'CVD_60', 'CVD_252']
feature_cols = [c for c in df.columns if c not in exclude
                and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
print(f"  Features: {len(feature_cols)}")

X = df[feature_cols].fillna(0)

# -- Scale with proper feature alignment (same logic as step10_live_inference) -
print("Aligning features with model scaler...")
if hasattr(scaler, 'feature_names_in_'):
    expected = list(scaler.feature_names_in_)
    extra    = [f for f in X.columns if f not in expected]
    missing  = [f for f in expected if f not in X.columns]
    if extra or missing:
        print(f"  Dropping {len(extra)} unseen features, zero-filling {len(missing)} missing.")
        X = X.drop(columns=[c for c in extra if c in X.columns], errors='ignore')
        for col in missing:
            X[col] = 0.0
        X = X[expected]
    print(f"  Feature alignment complete: {len(X.columns)} features")

X_scaled = scaler.transform(X)

cat_prob = cat_model.predict_proba(X_scaled)[:, 1]
xgb_prob = xgb_model.predict_proba(X_scaled)[:, 1]
lgb_prob  = lgb_model.predict(X_scaled)

ensemble_prob = (cat_prob + xgb_prob + lgb_prob) / 3.0

# -- Signal label --------------------------------------------------------------
signal = np.where(ensemble_prob >= 0.65, 'LONG',
          np.where(ensemble_prob <= 0.35, 'SHORT', 'NEUTRAL'))

# -- Build output DataFrame ----------------------------------------------------
out = pd.DataFrame({
    'Date':             df['Date'].dt.strftime('%Y-%m-%d'),
    'Cat_Prob':         cat_prob,
    'XGB_Prob':         xgb_prob,
    'LGB_Prob':         lgb_prob,
    'Ensemble_Prob':    ensemble_prob,
    'Signal':           signal,
    'Target_Direction': df['Target_Direction'].astype(int),
})

out_path = os.path.join(DIR, "test_predictions_full.csv")
out.to_csv(out_path, index=False)
print(f"\nSaved {len(out):,} rows to {out_path}")
print(f"  Date range: {out['Date'].min()} to {out['Date'].max()}")
print(f"  LONG: {(out['Signal']=='LONG').sum()}  SHORT: {(out['Signal']=='SHORT').sum()}  NEUTRAL: {(out['Signal']=='NEUTRAL').sum()}")
print(f"  Ensemble_Prob range: {ensemble_prob.min():.4f} to {ensemble_prob.max():.4f}")
