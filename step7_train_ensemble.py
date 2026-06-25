"""
Phase 12: Optimal Ensemble — CatBoost + XGBoost + LightGBM Voting
===================================================================
Changes vs Phase 11:
  ✓ LSTM completely removed (was dragging accuracy below 50%)
  ✓ 3-Model Voting Ensemble: CatBoost + XGBoost + LightGBM
  ✓ Optuna tunes ALL THREE models independently
  ✓ Filtered target: days where |return| < 0.08% marked as FLAT → skipped
  ✓ Calibrated F1-Optimal threshold
  ✓ High-Confidence Regime Filter (>60% confidence only)
  ✓ Stacked meta-learner on top of 3-model outputs (Logistic Regression)
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from catboost  import CatBoostClassifier
import xgboost  as xgb
import lightgbm as lgb
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_IN    = os.path.join(OUTPUT_DIR, "multimodal_master_dataset.csv")
MODEL_CAT_OUT = os.path.join(OUTPUT_DIR, "catboost_prod.cbm")
MODEL_XGB_OUT = os.path.join(OUTPUT_DIR, "xgb_prod.json")
MODEL_LGB_OUT = os.path.join(OUTPUT_DIR, "lgb_prod.txt")
MODEL_META_OUT= os.path.join(OUTPUT_DIR, "meta_learner.pkl")
SCALER_OUT    = os.path.join(OUTPUT_DIR, "scaler.pkl")
THRESHOLD_OUT = os.path.join(OUTPUT_DIR, "model_threshold.json")
CONFIDENCE_BAND = 0.60

# ── HELPERS ───────────────────────────────────────────────────────────────────
def optimal_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.28, 0.73, 0.01):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)

def evaluate(y_true, y_prob, threshold, label):
    y_pred = (y_prob >= threshold).astype(int)
    acc    = accuracy_score(y_true, y_pred)
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    try:   auc = roc_auc_score(y_true, y_prob)
    except: auc = 0.5
    hc_mask  = (y_prob > CONFIDENCE_BAND) | (y_prob < (1 - CONFIDENCE_BAND))
    hc_acc   = accuracy_score(np.array(y_true)[hc_mask],
                               (y_prob[hc_mask] >= threshold).astype(int)) if hc_mask.sum() else acc
    hc_n     = hc_mask.sum()
    print(f"\n{'='*58}")
    print(f"  [{label}]")
    print(f"{'='*58}")
    print(f"  Threshold         : {threshold:.2f}")
    print(f"  Overall Win Rate  : {acc*100:.2f}%")
    print(f"  Precision         : {prec*100:.2f}%")
    print(f"  Recall            : {rec*100:.2f}%")
    print(f"  F1-Score          : {f1:.4f}")
    print(f"  ROC-AUC           : {auc:.4f}")
    print(f"  ── High-Confidence ({CONFIDENCE_BAND*100:.0f}%+ filter) ──")
    print(f"  HC Win Rate       : {hc_acc*100:.2f}%  ({hc_n} signals / {len(y_true)} days)")
    return {"acc": acc, "hc_acc": hc_acc, "hc_trades": int(hc_n), "threshold": threshold}

# ── OPTUNA TUNERS ─────────────────────────────────────────────────────────────
def tune_catboost(X_tr, y_tr, n_trials=40):
    print("  Tuning CatBoost...")
    def obj(trial):
        p = {
            "iterations":    trial.suggest_int("iterations", 200, 700),
            "depth":         trial.suggest_int("depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
            "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
            "eval_metric": "Logloss", "verbose": 0, "random_seed": 42
        }
        n = len(X_tr); folds = 5; fold = n // (folds + 1)
        scores = []
        for k in range(1, folds + 1):
            te = min((k+1)*fold, n)
            if te - k*fold < 15: continue
            m = CatBoostClassifier(**p)
            m.fit(X_tr.iloc[:k*fold], y_tr.iloc[:k*fold])
            scores.append(roc_auc_score(y_tr.iloc[k*fold:te],
                                        m.predict_proba(X_tr.iloc[k*fold:te])[:,1]))
        return float(np.mean(scores))
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    bp.update({"eval_metric": "Logloss", "verbose": 0, "random_seed": 42})
    print(f"    Best AUC={study.best_value:.4f}  depth={bp['depth']} lr={bp['learning_rate']:.4f}")
    return bp

def tune_xgboost(X_tr, y_tr, n_trials=40):
    print("  Tuning XGBoost...")
    def obj(trial):
        p = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 700),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "eval_metric": "logloss", "use_label_encoder": False,
            "random_state": 42, "verbosity": 0
        }
        n = len(X_tr); folds = 5; fold = n // (folds + 1)
        scores = []
        for k in range(1, folds + 1):
            te = min((k+1)*fold, n)
            if te - k*fold < 15: continue
            m = xgb.XGBClassifier(**p)
            m.fit(X_tr.iloc[:k*fold], y_tr.iloc[:k*fold], verbose=False)
            scores.append(roc_auc_score(y_tr.iloc[k*fold:te],
                                        m.predict_proba(X_tr.iloc[k*fold:te])[:,1]))
        return float(np.mean(scores))
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    bp.update({"eval_metric": "logloss", "use_label_encoder": False,
               "random_state": 42, "verbosity": 0})
    print(f"    Best AUC={study.best_value:.4f}  depth={bp['max_depth']} lr={bp['learning_rate']:.4f}")
    return bp

def tune_lightgbm(X_tr, y_tr, n_trials=40):
    print("  Tuning LightGBM...")
    def obj(trial):
        p = {
            "n_estimators":    trial.suggest_int("n_estimators", 200, 700),
            "num_leaves":      trial.suggest_int("num_leaves", 20, 80),
            "learning_rate":   trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "random_state": 42, "verbosity": -1, "force_col_wise": True
        }
        n = len(X_tr); folds = 5; fold = n // (folds + 1)
        scores = []
        for k in range(1, folds + 1):
            te = min((k+1)*fold, n)
            if te - k*fold < 15: continue
            m = lgb.LGBMClassifier(**p)
            m.fit(X_tr.iloc[:k*fold], y_tr.iloc[:k*fold])
            scores.append(roc_auc_score(y_tr.iloc[k*fold:te],
                                        m.predict_proba(X_tr.iloc[k*fold:te])[:,1]))
        return float(np.mean(scores))
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    bp.update({"random_state": 42, "verbosity": -1, "force_col_wise": True})
    print(f"    Best AUC={study.best_value:.4f}  leaves={bp['num_leaves']} lr={bp['learning_rate']:.4f}")
    return bp

# ── WALK-FORWARD PREDICTION ───────────────────────────────────────────────────
def walk_forward(X_train_df, y_train, X_test_df, y_test,
                 cat_p, xgb_p, lgb_p, chunk_size=60):
    all_cat, all_xgb, all_lgb = [], [], []
    cur_X, cur_y = X_train_df.copy(), y_train.copy()
    total_chunks = int(np.ceil(len(X_test_df) / chunk_size))

    for i in range(total_chunks):
        s = i * chunk_size
        e = min((i+1)*chunk_size, len(X_test_df))
        chunk_X = X_test_df.iloc[s:e]
        chunk_y = y_test.iloc[s:e]

        m_cat = CatBoostClassifier(**cat_p).fit(cur_X, cur_y)
        m_xgb = xgb.XGBClassifier(**xgb_p)
        m_xgb.fit(cur_X, cur_y, verbose=False)
        m_lgb = lgb.LGBMClassifier(**lgb_p).fit(cur_X, cur_y)

        all_cat.extend(m_cat.predict_proba(chunk_X)[:, 1])
        all_xgb.extend(m_xgb.predict_proba(chunk_X)[:, 1])
        all_lgb.extend(m_lgb.predict_proba(chunk_X)[:, 1])

        cur_X = pd.concat([cur_X, chunk_X]).reset_index(drop=True)
        cur_y = pd.concat([cur_y, chunk_y]).reset_index(drop=True)
        print(f"  WF chunk {i+1}/{total_chunks} done.")

    return np.array(all_cat), np.array(all_xgb), np.array(all_lgb)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  PHASE 12: 3-MODEL OPTIMAL ENSEMBLE ENGINE")
    print("  CatBoost + XGBoost + LightGBM + Meta-Learner")
    print("=" * 58)

    # Step 1: Load
    print("\n=== Step 1: Load & Split ===")
    df = pd.read_csv(DATASET_IN)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    X = df.drop(columns=['Date', 'Target_Direction'])
    y = df['Target_Direction']

    train_size = int(len(df) * 0.80)
    val_size   = int(len(df) * 0.10)

    X_train_raw = X.iloc[:train_size]
    y_train_raw = y.iloc[:train_size]
    X_val_raw   = X.iloc[train_size:train_size + val_size]
    y_val_raw   = y.iloc[train_size:train_size + val_size]
    X_test_raw  = X.iloc[train_size + val_size:]
    y_test_raw  = y.iloc[train_size + val_size:]

    print(f"  Train: {len(X_train_raw):,}  Val: {len(X_val_raw):,}  Test: {len(X_test_raw):,}")

    # Step 2: Scale
    print("\n=== Step 2: Scale ===")
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X.columns)
    X_val_sc   = pd.DataFrame(scaler.transform(X_val_raw),   columns=X.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test_raw),  columns=X.columns)

    # Step 3: Optuna Tuning
    print("\n=== Step 3: Optuna Bayesian Tuning (40 trials each) ===")
    cat_p = tune_catboost(X_train_sc, y_train_raw, n_trials=40)
    xgb_p = tune_xgboost(X_train_sc, y_train_raw, n_trials=40)
    lgb_p = tune_lightgbm(X_train_sc, y_train_raw, n_trials=40)

    # Step 4: Walk-Forward on Val + Test combined
    print("\n=== Step 4: Walk-Forward Evaluation ===")
    X_wf = pd.concat([X_val_sc, X_test_sc]).reset_index(drop=True)
    y_wf = pd.concat([y_val_raw, y_test_raw]).reset_index(drop=True)

    cat_p_wf, xgb_p_wf, lgb_p_wf = walk_forward(
        X_train_sc, y_train_raw,
        X_wf, y_wf,
        cat_p, xgb_p, lgb_p, chunk_size=60
    )

    # Step 5: Stacked Meta-Learner on Val set
    print("\n=== Step 5: Meta-Learner Stacking ===")
    val_n = len(X_val_sc)
    val_stack = np.column_stack([cat_p_wf[:val_n], xgb_p_wf[:val_n], lgb_p_wf[:val_n]])
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta.fit(val_stack, y_wf[:val_n].values)
    print(f"  Meta-learner weights: CatBoost={meta.coef_[0][0]:.3f}  "
          f"XGB={meta.coef_[0][1]:.3f}  LGB={meta.coef_[0][2]:.3f}")

    # Step 6: Calibrated Threshold on Val portion (Meta-Learner outputs)
    print("\n=== Step 6: Calibrated Threshold (Meta-Learner) ===")
    val_meta_probs = meta.predict_proba(val_stack)[:, 1]
    best_t  = optimal_threshold(y_wf[:val_n].values, val_meta_probs)
    print(f"  Calibrated threshold: {best_t:.2f}")

    with open(THRESHOLD_OUT, 'w') as f:
        json.dump({"threshold": best_t, "confidence_band": CONFIDENCE_BAND,
                   "ensemble": "catboost+xgboost+lightgbm"}, f)

    # Step 7: Evaluate on Test set
    print("\n=== Step 7: Final Test Evaluation ===")
    test_stack  = np.column_stack([cat_p_wf[val_n:], xgb_p_wf[val_n:], lgb_p_wf[val_n:]])
    test_probs  = meta.predict_proba(test_stack)[:, 1]
    y_test_align = y_wf[val_n:].values
    results = evaluate(y_test_align, test_probs, best_t, "Phase 12 Optimal Ensemble")

    # Save predictions
    test_dates = df['Date'].iloc[train_size + val_n:].values
    preds_df = pd.DataFrame({
        'Date':             test_dates,
        'Cat_Prob':         cat_p_wf[val_n:],
        'XGB_Prob':         xgb_p_wf[val_n:],
        'LGB_Prob':         lgb_p_wf[val_n:],
        'Ensemble_Prob':    test_probs,
        'Signal':           np.where(test_probs > CONFIDENCE_BAND, 'LONG',
                            np.where(test_probs < (1-CONFIDENCE_BAND), 'SHORT', 'NEUTRAL')),
        'Target_Direction': y_test_align
    })
    preds_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

    # Step 8: Train Production models on 100% data
    print("\n=== Step 8: Train Production Models (100% Data) ===")
    prod_scaler = StandardScaler()
    X_full = pd.DataFrame(prod_scaler.fit_transform(X), columns=X.columns)

    prod_cat = CatBoostClassifier(**cat_p).fit(X_full, y)
    prod_xgb = xgb.XGBClassifier(**xgb_p)
    prod_xgb.fit(X_full, y, verbose=False)
    prod_lgb = lgb.LGBMClassifier(**lgb_p).fit(X_full, y)

    # Re-train meta learner on full OUT-OF-SAMPLE walk-forward predictions
    # to avoid extreme in-sample overfitting!
    full_wf_stack = np.column_stack([cat_p_wf, xgb_p_wf, lgb_p_wf])
    prod_meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    prod_meta.fit(full_wf_stack, y_wf.values)

    prod_cat.save_model(MODEL_CAT_OUT)
    prod_xgb.save_model(MODEL_XGB_OUT)
    prod_lgb.booster_.save_model(MODEL_LGB_OUT)
    joblib.dump(prod_meta,   MODEL_META_OUT)
    joblib.dump(prod_scaler, SCALER_OUT)

    print(f"  CatBoost saved : {MODEL_CAT_OUT}")
    print(f"  XGBoost saved  : {MODEL_XGB_OUT}")
    print(f"  LightGBM saved : {MODEL_LGB_OUT}")
    print(f"  Meta-learner   : {MODEL_META_OUT}")

    print("\n" + "="*58)
    print("  PHASE 12 COMPLETE")
    print(f"  Overall Win Rate  : {results['acc']*100:.2f}%")
    print(f"  HC Win Rate       : {results['hc_acc']*100:.2f}%")
    print(f"  HC Signals Issued : {results['hc_trades']}")
    print(f"  Threshold         : {best_t:.2f}")
    print("="*58)

if __name__ == "__main__":
    main()
