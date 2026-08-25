"""
Phase 12 Enhanced v2: Optimal Ensemble — CatBoost + XGBoost + LightGBM
========================================================================
Changes in v2 (paper-driven win-rate upgrades):
  ✓ 5-day purge embargo between train/test chunks (Chai et al. macro digestion)
  ✓ Rolling-origin expanding TSCV with per-fold MCC/DA table (Arif et al.)
  ✓ MCC + Directional Accuracy (DA) in all evaluation outputs (Dakalbab et al.)
  ✓ Consensus + Compulsory-Agent signal aggregation replaces flat majority vote
    (Hernes et al. A-Trader — MyStrategy was consistently worst performer)
  ✓ Volatility regime made COMPULSORY veto (not just a vote)
  ✓ RSI regime also compulsory veto on conflicting signals
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef)
from catboost  import CatBoostClassifier
import xgboost  as xgb
import lightgbm as lgb
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── WIN-RATE UPGRADE CONSTANTS ────────────────────────────────────────────────
EMBARGO_DAYS   = 5    # Chai et al.: macro shocks absorbed within ~5 trading days
N_TSCV_FOLDS   = 5    # Arif et al.: 5-fold rolling-origin expanding TSCV

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

# ── TUNABLE CONSTANTS ─────────────────────────────────────────────────────────
CONFIDENCE_BAND = 0.65    # Raised from 0.60 → higher win rate, fewer trades
N_OPTUNA_TRIALS = 15      # 15 trials per model for fast efficient optimization

# ── HELPERS ───────────────────────────────────────────────────────────────────
def optimal_threshold(y_true, y_prob):
    """Find F1-optimal threshold by grid search."""
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
    # MCC: far more informative than accuracy on imbalanced classes (Dakalbab et al.)
    mcc    = matthews_corrcoef(y_true, y_pred)
    # DA (Directional Accuracy): fraction where predicted sign matches actual sign
    # For binary classification this equals accuracy, reported separately for clarity
    da     = acc
    try:   auc = roc_auc_score(y_true, y_prob)
    except: auc = 0.5

    # High-confidence accuracy: only look at predictions > CONFIDENCE_BAND or < (1-CONFIDENCE_BAND)
    hc_mask = (y_prob > CONFIDENCE_BAND) | (y_prob < (1 - CONFIDENCE_BAND))
    hc_acc  = accuracy_score(np.array(y_true)[hc_mask],
                              (y_prob[hc_mask] >= threshold).astype(int)) if hc_mask.sum() else acc
    hc_n    = hc_mask.sum()

    print(f"\n{'='*60}")
    print(f"  [{label}]")
    print(f"{'='*60}")
    print(f"  Threshold               : {threshold:.2f}")
    print(f"  Overall Win Rate (DA)   : {acc*100:.2f}%")
    print(f"  Directional Accuracy    : {da*100:.2f}%")
    print(f"  Matthews Corr Coeff     : {mcc:.4f}  (>0.10=signal, >0.20=good)")
    print(f"  Precision               : {prec*100:.2f}%")
    print(f"  Recall                  : {rec*100:.2f}%")
    print(f"  F1-Score                : {f1:.4f}")
    print(f"  ROC-AUC                 : {auc:.4f}")
    print(f"  ── High-Confidence ({CONFIDENCE_BAND*100:.0f}%+ filter) ──")
    print(f"  HC Win Rate             : {hc_acc*100:.2f}%  ({hc_n} signals / {len(y_true)} days)")
    return {"acc": acc, "da": da, "mcc": mcc, "hc_acc": hc_acc,
            "hc_trades": int(hc_n), "threshold": threshold}

# ── OPTUNA TUNERS ─────────────────────────────────────────────────────────────
def tune_catboost(X_tr, y_tr, n_trials=N_OPTUNA_TRIALS):
    print("  Tuning CatBoost...")
    def obj(trial):
        p = {
            "iterations":    trial.suggest_int("iterations", 300, 900),
            "depth":         trial.suggest_int("depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.12, log=True),
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

def tune_xgboost(X_tr, y_tr, n_trials=N_OPTUNA_TRIALS):
    print("  Tuning XGBoost...")
    def obj(trial):
        p = {
            "n_estimators":     trial.suggest_int("n_estimators", 300, 900),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.003, 0.12, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "eval_metric": "logloss",
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
    bp.update({"eval_metric": "logloss", "random_state": 42, "verbosity": 0})
    print(f"    Best AUC={study.best_value:.4f}  depth={bp['max_depth']} lr={bp['learning_rate']:.4f}")
    return bp

def tune_lightgbm(X_tr, y_tr, n_trials=N_OPTUNA_TRIALS):
    print("  Tuning LightGBM...")
    def obj(trial):
        p = {
            "n_estimators":    trial.suggest_int("n_estimators", 300, 900),
            "num_leaves":      trial.suggest_int("num_leaves", 20, 100),
            "learning_rate":   trial.suggest_float("learning_rate", 0.003, 0.12, log=True),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 50),
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

# ── ROLLING-ORIGIN EXPANDING TSCV (Arif et al.) ──────────────────────────────
def rolling_origin_tscv(X_df, y_series, cat_p, xgb_p, lgb_p,
                         n_folds=N_TSCV_FOLDS, embargo_days=EMBARGO_DAYS):
    """
    5-fold rolling-origin expanding-window Time-Series Cross-Validation.
    Each fold grows the training window by one fold-size increment.
    A purge embargo of `embargo_days` rows is removed from the tail of each
    training window to prevent feature-leakage from overlapping rolling windows
    (Chai et al. macro digestion window = 5 trading days).

    Returns:
        all_cat, all_xgb, all_lgb  — OOF probability arrays over val+test window
        fold_results               — per-fold dict with acc, mcc metrics
    """
    n = len(X_df)
    # Split into n_folds+1 equal blocks; first block = initial training seed
    fold_size = n // (n_folds + 1)

    all_cat, all_xgb, all_lgb = [], [], []
    fold_results = []

    print(f"  [TSCV] {n_folds}-fold rolling-origin  |  "
          f"fold_size={fold_size}  |  embargo={embargo_days}d")

    for k in range(1, n_folds + 1):
        # Training: all data up to fold boundary minus embargo
        train_end  = k * fold_size - embargo_days
        # Test: next fold window
        test_start = k * fold_size
        test_end   = min((k + 1) * fold_size, n)

        if train_end < 30 or test_start >= n:
            print(f"  [TSCV] Fold {k}: skipped (insufficient data).")
            continue

        X_tr = X_df.iloc[:train_end]
        y_tr = y_series.iloc[:train_end]
        X_te = X_df.iloc[test_start:test_end]
        y_te = y_series.iloc[test_start:test_end]

        # Fit all 3 base models on expanding training window
        m_cat = CatBoostClassifier(**cat_p).fit(X_tr, y_tr)
        m_xgb = xgb.XGBClassifier(**xgb_p)
        m_xgb.fit(X_tr, y_tr, verbose=False)
        m_lgb = lgb.LGBMClassifier(**lgb_p).fit(X_tr, y_tr)

        p_cat = m_cat.predict_proba(X_te)[:, 1]
        p_xgb = m_xgb.predict_proba(X_te)[:, 1]
        p_lgb = m_lgb.predict_proba(X_te)[:, 1]

        all_cat.extend(p_cat)
        all_xgb.extend(p_xgb)
        all_lgb.extend(p_lgb)

        # Per-fold metrics
        fold_avg = (p_cat + p_xgb + p_lgb) / 3
        fold_pred = (fold_avg >= 0.5).astype(int)
        fold_acc  = accuracy_score(y_te, fold_pred)
        fold_mcc  = matthews_corrcoef(y_te, fold_pred)
        fold_results.append({
            'fold': k, 'train_n': len(X_tr), 'test_n': len(X_te),
            'accuracy': fold_acc, 'mcc': fold_mcc
        })
        print(f"  [TSCV] Fold {k}/{n_folds}  "
              f"train={len(X_tr):>4}  test={len(X_te):>4}  "
              f"acc={fold_acc*100:.1f}%  MCC={fold_mcc:.3f}  done.")

    # Print per-fold summary table (Arif et al. recommendation)
    print("\n  Per-Fold TSCV Summary:")
    print("  ┌─────┬───────────┬──────────┬──────────┬──────────┐")
    print("  │Fold │ Train N   │ Test N   │  Acc %   │  MCC     │")
    print("  ├─────┼───────────┼──────────┼──────────┼──────────┤")
    for r in fold_results:
        print(f"  │  {r['fold']}  │ {r['train_n']:>8,} │"
              f" {r['test_n']:>7,}  │ {r['accuracy']*100:>6.2f}%  │"
              f" {r['mcc']:>7.4f}  │")
    if fold_results:
        avg_acc = np.mean([r['accuracy'] for r in fold_results])
        avg_mcc = np.mean([r['mcc'] for r in fold_results])
        std_acc = np.std( [r['accuracy'] for r in fold_results])
        print("  ├─────┼───────────┼──────────┼──────────┼──────────┤")
        print(f"  │ Avg │           │          │ {avg_acc*100:>6.2f}%  │ {avg_mcc:>7.4f}  │")
        print(f"  │ Std │           │          │ {std_acc*100:>6.2f}%  │          │")
        print("  └─────┴───────────┴──────────┴──────────┴──────────┘")
        fold_var = std_acc * 100
        if fold_var < 5:
            print(f"  Stability: GOOD  (fold-to-fold σ={fold_var:.1f}% < 5%)")
        elif fold_var < 10:
            print(f"  Stability: MODERATE  (fold-to-fold σ={fold_var:.1f}%, watch regime shifts)")
        else:
            print(f"  Stability: POOR  (fold-to-fold σ={fold_var:.1f}%, high regime sensitivity)")

    return np.array(all_cat), np.array(all_xgb), np.array(all_lgb), fold_results


# ── LEGACY walk_forward kept for backward compatibility ──────────────────────
def walk_forward(X_train_df, y_train, X_test_df, y_test,
                 cat_p, xgb_p, lgb_p, chunk_size=60,
                 embargo_days=EMBARGO_DAYS):
    """
    Fixed-chunk walk-forward with purge embargo.
    Used for the final val+test combined evaluation pass.
    """
    all_cat, all_xgb, all_lgb = [], [], []
    cur_X, cur_y = X_train_df.copy(), y_train.copy()
    total_chunks = int(np.ceil(len(X_test_df) / chunk_size))

    for i in range(total_chunks):
        s = i * chunk_size
        e = min((i+1)*chunk_size, len(X_test_df))
        chunk_X = X_test_df.iloc[s:e]
        chunk_y = y_test.iloc[s:e]

        # Purge embargo: exclude last `embargo_days` rows from training window
        emb_X = cur_X.iloc[:-embargo_days] if len(cur_X) > embargo_days else cur_X
        emb_y = cur_y.iloc[:-embargo_days] if len(cur_y) > embargo_days else cur_y

        m_cat = CatBoostClassifier(**cat_p).fit(emb_X, emb_y)
        m_xgb = xgb.XGBClassifier(**xgb_p)
        m_xgb.fit(emb_X, emb_y, verbose=False)
        m_lgb = lgb.LGBMClassifier(**lgb_p).fit(emb_X, emb_y)

        all_cat.extend(m_cat.predict_proba(chunk_X)[:, 1])
        all_xgb.extend(m_xgb.predict_proba(chunk_X)[:, 1])
        all_lgb.extend(m_lgb.predict_proba(chunk_X)[:, 1])

        cur_X = pd.concat([cur_X, chunk_X]).reset_index(drop=True)
        cur_y = pd.concat([cur_y, chunk_y]).reset_index(drop=True)
        print(f"  WF chunk {i+1}/{total_chunks} done.")

    return np.array(all_cat), np.array(all_xgb), np.array(all_lgb)

# ── SIGNAL QUALIFICATION — CONSENSUS + COMPULSORY-AGENT (Hernes et al.) ───────
def qualify_signal_consensus(prob_up, p_cat, p_xgb, p_lgb, long_thresh, short_thresh,
                             rsi_regime=None, high_vol=None):
    """
    Hernes et al. A-Trader Consensus + Compulsory-Agent strategy.

    Improvements over flat majority-vote (MyStrategy — worst in their comparison):
      1. COMPULSORY VETO: volatility & RSI regime can unconditionally block signals
         regardless of model agreement (Evolution-style agent compulsion).
      2. CONSENSUS: use median-sorted model probability as confirmation gate
         rather than simple average/vote — more robust to individual model outliers.

    Returns: "LONG" | "SHORT" | "NEUTRAL"
    """
    # ── GATE 1 (COMPULSORY): High-volatility veto ─────────────────────────────
    # Volatility regime agent is COMPULSORY — it vetoes regardless of model votes.
    # Hernes et al.: certain agents in Evolution strategy have compulsory=True flag.
    if high_vol is not None and high_vol == 1:
        return "NEUTRAL"

    # ── GATE 2: Adaptive percentile thresholds ────────────────────────────────
    if prob_up >= long_thresh:
        raw_signal = "LONG"
    elif prob_up <= short_thresh:
        raw_signal = "SHORT"
    else:
        return "NEUTRAL"

    # ── GATE 3: Consensus (median-sorted probability, Hernes et al.) ──────────
    # Compare median of sorted model probabilities against mid_thresh baseline
    # rather than hardcoded 0.50 (since probabilities cluster around distribution mean).
    mid_thresh = (long_thresh + short_thresh) / 2.0
    sorted_probs = sorted([p_cat, p_xgb, p_lgb])
    consensus_prob = sorted_probs[1]  # median
    if raw_signal == "LONG"  and consensus_prob < mid_thresh:
        return "NEUTRAL"   # Meta-learner says LONG but consensus is below baseline
    if raw_signal == "SHORT" and consensus_prob > mid_thresh:
        return "NEUTRAL"   # Meta-learner says SHORT but consensus is above baseline

    # ── GATE 4 (COMPULSORY): RSI Regime veto ─────────────────────────────────
    # RSI regime is also compulsory — structural overbought/oversold vetoes signal.
    if rsi_regime is not None:
        if raw_signal == "LONG"  and rsi_regime == 1:  return "NEUTRAL"  # Overbought
        if raw_signal == "SHORT" and rsi_regime == -1: return "NEUTRAL"  # Oversold

    return raw_signal


# Alias kept for any external callers
def qualify_signal(prob_up, p_cat, p_xgb, p_lgb, long_thresh, short_thresh,
                   rsi_regime=None, high_vol=None):
    """Backward-compatible alias — delegates to consensus implementation."""
    return qualify_signal_consensus(
        prob_up, p_cat, p_xgb, p_lgb, long_thresh, short_thresh,
        rsi_regime, high_vol
    )

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PHASE 12 Enhanced v2: 3-MODEL OPTIMAL ENSEMBLE ENGINE")
    print("  CatBoost + XGBoost + LightGBM + Meta-Learner")
    print(f"  Confidence Band : {CONFIDENCE_BAND*100:.0f}%  |  Optuna Trials: {N_OPTUNA_TRIALS}")
    print(f"  Embargo Days    : {EMBARGO_DAYS}  |  TSCV Folds   : {N_TSCV_FOLDS}")
    print(f"  Signal Logic    : Consensus + Compulsory-Agent (Hernes et al.)")
    print("=" * 60)

    # Step 1: Load
    print("\n=== Step 1: Load & Split ===")
    df = pd.read_csv(DATASET_IN)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # ── Feature matrix: explicitly exclude all target-derived columns ────────────
    # SHAP WARNING: 'Target_SD_3class' was flagged as 13x dominant feature
    # because it is a target-derived label, NOT a predictive feature.
    # These columns must NEVER appear in X.
    EXCLUDE_COLS = [
        'Date',
        'Target_Direction',     # primary target
        'Target_SD_Binary',     # SD-based label variant — derived from target return
        'Target_SD_3class',     # SD-based 3-class label — derived from target return
        'Next_Day_Return',      # raw next-day return — direct future lookahead
    ]
    X = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])
    y = df['Target_Direction']

    # Extract RSI_Regime and High_Vol_Regime columns for signal qualification (not used as features)
    rsi_regime_col  = df['RSI_Regime'].values    if 'RSI_Regime'      in df.columns else None
    high_vol_col    = df['High_Vol_Regime'].values if 'High_Vol_Regime' in df.columns else None

    train_size = int(len(df) * 0.80)
    val_size   = int(len(df) * 0.10)
    # BUG FIX: use val_size consistently (not val_n from downstream)
    test_start = train_size + val_size

    X_train_raw = X.iloc[:train_size]
    y_train_raw = y.iloc[:train_size]
    X_val_raw   = X.iloc[train_size:test_start]
    y_val_raw   = y.iloc[train_size:test_start]
    X_test_raw  = X.iloc[test_start:]
    y_test_raw  = y.iloc[test_start:]

    print(f"  Train: {len(X_train_raw):,}  Val: {len(X_val_raw):,}  Test: {len(X_test_raw):,}")

    # Step 2: Scale
    print("\n=== Step 2: Scale ===")
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X.columns)
    X_val_sc   = pd.DataFrame(scaler.transform(X_val_raw),   columns=X.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test_raw),  columns=X.columns)

    # Step 3: Optuna Tuning
    print(f"\n=== Step 3: Optuna Bayesian Tuning ({N_OPTUNA_TRIALS} trials each) ===")
    cat_p = tune_catboost(X_train_sc, y_train_raw, n_trials=N_OPTUNA_TRIALS)
    xgb_p = tune_xgboost(X_train_sc, y_train_raw, n_trials=N_OPTUNA_TRIALS)
    lgb_p = tune_lightgbm(X_train_sc, y_train_raw, n_trials=N_OPTUNA_TRIALS)

    # Step 3b: Rolling-Origin Expanding TSCV — diagnostic pass (Arif et al.)
    print(f"\n=== Step 3b: Rolling-Origin TSCV ({N_TSCV_FOLDS}-fold, embargo={EMBARGO_DAYS}d) ===")
    X_all_sc = pd.concat([X_train_sc, X_val_sc, X_test_sc]).reset_index(drop=True)
    y_all    = pd.concat([y_train_raw, y_val_raw, y_test_raw]).reset_index(drop=True)
    _, _, _, tscv_fold_results = rolling_origin_tscv(
        X_all_sc, y_all, cat_p, xgb_p, lgb_p,
        n_folds=N_TSCV_FOLDS, embargo_days=EMBARGO_DAYS
    )

    # Step 4: Walk-Forward on Val + Test combined (with embargo)
    print(f"\n=== Step 4: Walk-Forward Evaluation (chunk=60, embargo={EMBARGO_DAYS}d) ===")
    X_wf = pd.concat([X_val_sc, X_test_sc]).reset_index(drop=True)
    y_wf = pd.concat([y_val_raw, y_test_raw]).reset_index(drop=True)

    cat_p_wf, xgb_p_wf, lgb_p_wf = walk_forward(
        X_train_sc, y_train_raw,
        X_wf, y_wf,
        cat_p, xgb_p, lgb_p, chunk_size=60, embargo_days=EMBARGO_DAYS
    )

    # Step 5: Ensemble Stacking (Average of Base Model Probabilities)
    # Note: Fitting LogisticRegression on tiny validation sample (26 rows)
    # caused coefficient compression (probs squashed to 0.57-0.61).
    # Average weighting preserves the full dynamic range [0.10 → 0.90] of base models.
    print("\n=== Step 5: Ensemble Aggregation (Equal Weighting) ===")
    val_n = len(X_val_sc)
    val_meta_probs = (cat_p_wf[:val_n] + xgb_p_wf[:val_n] + lgb_p_wf[:val_n]) / 3.0

    # Step 6: Calibrated Threshold on Val portion
    print("\n=== Step 6: Calibrated Threshold ===")
    best_t  = optimal_threshold(y_wf[:val_n].values, val_meta_probs)
    print(f"  Calibrated threshold: {best_t:.2f}")

    with open(THRESHOLD_OUT, 'w') as f:
        json.dump({"threshold": best_t, "confidence_band": CONFIDENCE_BAND,
                   "ensemble": "catboost+xgboost+lightgbm_mean"}, f)

    # Step 7: Evaluate on Test set
    print("\n=== Step 7: Final Test Evaluation ===")
    test_probs   = (cat_p_wf[val_n:] + xgb_p_wf[val_n:] + lgb_p_wf[val_n:]) / 3.0
    y_test_align = y_wf[val_n:].values
    results = evaluate(y_test_align, test_probs, best_t, "Phase 12 Enhanced Ensemble")

    # Build enhanced signals with multi-gate filtering
    # Use adaptive thresholds from test predictions distribution
    long_thresh  = float(np.percentile(test_probs, 70))
    short_thresh = float(np.percentile(test_probs, 30))
    print(f"  Adaptive signal thresholds: LONG >= {long_thresh:.4f}, SHORT <= {short_thresh:.4f}")

    test_rsi_regime = rsi_regime_col[test_start:] if rsi_regime_col is not None else [None]*len(test_probs)
    test_high_vol   = high_vol_col[test_start:]   if high_vol_col   is not None else [None]*len(test_probs)

    n_long = n_short = n_neutral = 0
    signals = []
    for i, (prob, pcat, pxgb, plgb, rsi, hvol) in enumerate(zip(
            test_probs, cat_p_wf[val_n:], xgb_p_wf[val_n:], lgb_p_wf[val_n:],
            test_rsi_regime, test_high_vol)):
        # Use consensus+compulsory signal aggregation (Hernes et al.)
        sig = qualify_signal_consensus(prob, pcat, pxgb, plgb,
                                       long_thresh, short_thresh, rsi, hvol)
        signals.append(sig)
        if sig == "LONG":    n_long    += 1
        elif sig == "SHORT": n_short   += 1
        else:                n_neutral += 1

    # Save predictions — BUG FIX: use test_start not (train_size + val_n)
    test_dates = df['Date'].iloc[test_start:].values
    preds_df = pd.DataFrame({
        'Date':             test_dates,
        'Cat_Prob':         cat_p_wf[val_n:],
        'XGB_Prob':         xgb_p_wf[val_n:],
        'LGB_Prob':         lgb_p_wf[val_n:],
        'Ensemble_Prob':    test_probs,
        'Signal':           signals,
        'Target_Direction': y_test_align
    })
    preds_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

    # Signal quality summary
    sig_counts = pd.Series(signals).value_counts()
    print(f"\n  Signal distribution — LONG: {sig_counts.get('LONG',0)}  "
          f"SHORT: {sig_counts.get('SHORT',0)}  "
          f"NEUTRAL: {sig_counts.get('NEUTRAL',0)}")

    # Step 8: Train Production models on 100% data
    print("\n=== Step 8: Train Production Models (100% Data) ===")
    prod_scaler = StandardScaler()
    X_full = pd.DataFrame(prod_scaler.fit_transform(X), columns=X.columns)

    prod_cat = CatBoostClassifier(**cat_p).fit(X_full, y)
    prod_xgb = xgb.XGBClassifier(**xgb_p)
    prod_xgb.fit(X_full, y, verbose=False)
    prod_lgb = lgb.LGBMClassifier(**lgb_p).fit(X_full, y)

    # Re-train meta on full walk-forward predictions
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

    # Print TSCV stability summary
    if tscv_fold_results:
        tscv_accs = [r['accuracy'] for r in tscv_fold_results]
        tscv_mccs = [r['mcc'] for r in tscv_fold_results]
        print(f"\n  TSCV Cross-Validation Summary ({N_TSCV_FOLDS} folds):")
        print(f"    Avg Acc  : {np.mean(tscv_accs)*100:.2f}%  "
              f"(σ={np.std(tscv_accs)*100:.2f}%)")
        print(f"    Avg MCC  : {np.mean(tscv_mccs):.4f}  "
              f"(σ={np.std(tscv_mccs):.4f})")

    print("\n" + "="*60)
    print("  PHASE 12 v2 COMPLETE — Consensus+Compulsory Signal Logic")
    print(f"  Overall Win Rate (DA): {results['acc']*100:.2f}%")
    print(f"  MCC                  : {results['mcc']:.4f}")
    print(f"  HC Win Rate ({CONFIDENCE_BAND*100:.0f}%+) : {results['hc_acc']*100:.2f}%")
    print(f"  HC Signals Issued    : {results['hc_trades']}")
    print(f"  Threshold            : {best_t:.2f}")
    print(f"  Embargo Days         : {EMBARGO_DAYS}")
    print(f"  TSCV Folds           : {N_TSCV_FOLDS}")
    print(f"  Signal Logic         : Consensus + Compulsory-Agent")
    print("="*60)

if __name__ == "__main__":
    main()
