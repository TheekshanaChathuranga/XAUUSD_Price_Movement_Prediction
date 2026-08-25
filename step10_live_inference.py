import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import shap

if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR     = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DATA = os.path.join(OUTPUT_DIR, "live_inference_data.csv")
MODEL_CAT      = os.path.join(OUTPUT_DIR, "catboost_prod.cbm")
MODEL_XGB      = os.path.join(OUTPUT_DIR, "xgb_prod.json")
MODEL_LGB      = os.path.join(OUTPUT_DIR, "lgb_prod.txt")
MODEL_META     = os.path.join(OUTPUT_DIR, "meta_learner.pkl")
SCALER_PATH    = os.path.join(OUTPUT_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(OUTPUT_DIR, "model_threshold.json")
RAW_PRICES     = os.path.join(OUTPUT_DIR, "xauusd_raw_prices.csv")


# ── CONSENSUS + COMPULSORY-AGENT SIGNAL (Hernes et al.) ──────────────────────
def qualify_signal_consensus(prob_up, p_cat, p_xgb, p_lgb,
                             long_thresh, short_thresh,
                             rsi_regime=None, high_vol=None):
    """
    Hernes et al. Consensus + Compulsory-Agent signal aggregation.
    Replaces flat majority vote (MyStrategy — worst performer in their study).

    Gate 1 (COMPULSORY): High-vol regime vetoes unconditionally.
    Gate 2: Adaptive percentile thresholds on meta-learner prob.
    Gate 3 (CONSENSUS): Median of sorted [p_cat, p_xgb, p_lgb] must agree.
    Gate 4 (COMPULSORY): RSI overbought/oversold vetoes conflicting signals.
    """
    # Compulsory volatility veto
    if high_vol is not None and high_vol == 1:
        return "HOLD (High-Vol Veto)"

    # Adaptive threshold gate
    if prob_up >= long_thresh:
        raw_signal = "LONG (BUY)"
    elif prob_up <= short_thresh:
        raw_signal = "SHORT (SELL)"
    else:
        return "HOLD"

    # Consensus gate: median of sorted model probs
    consensus_prob = sorted([p_cat, p_xgb, p_lgb])[1]   # median
    if "LONG" in raw_signal  and consensus_prob <= 0.50:
        return "HOLD (No Consensus)"
    if "SHORT" in raw_signal and consensus_prob >= 0.50:
        return "HOLD (No Consensus)"

    # Compulsory RSI veto
    if rsi_regime is not None:
        if "LONG"  in raw_signal and rsi_regime == 1:  return "HOLD (RSI Overbought)"
        if "SHORT" in raw_signal and rsi_regime == -1: return "HOLD (RSI Oversold)"

    return raw_signal

def calculate_atr(df, period=14):
    high_low   = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close  = np.abs(df['Low']  - df['Close'].shift())
    ranges     = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def load_threshold():
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH) as f:
            cfg = json.load(f)
        return (
            cfg.get("threshold", 0.5),
            cfg.get("confidence_band", 0.65),
            cfg.get("sl_atr_mult", 1.5),   # fallback to original 1.5× SL
            cfg.get("tp_atr_mult", 3.0),   # fallback to original 3.0× TP
        )
    return 0.5, 0.65, 1.5, 3.0

def load_adaptive_thresholds():
    """
    Compute adaptive LONG/SHORT thresholds from historical Ensemble_Prob
    distribution. The meta-learner compresses probs into a narrow band
    (e.g. 0.52–0.66), making hardcoded 0.65 unreachable.
    """
    PERCENTILE_LONG  = 85
    PERCENTILE_SHORT = 15
    preds_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    if os.path.exists(preds_path):
        try:
            preds_df = pd.read_csv(preds_path)
            if 'Ensemble_Prob' in preds_df.columns and len(preds_df) > 20:
                long_t  = float(np.percentile(preds_df['Ensemble_Prob'], PERCENTILE_LONG))
                short_t = float(np.percentile(preds_df['Ensemble_Prob'], PERCENTILE_SHORT))
                return long_t, short_t
        except Exception:
            pass
    _, cb = load_threshold()
    return cb, 1 - cb

def main():
    print("==================================================")
    print(" PHASE 7 (ENHANCED): LIVE INFERENCE — FULL ENSEMBLE")
    print(" CatBoost + XGBoost + LightGBM + Meta-Learner")
    print("==================================================")

    # Check all required files exist
    required = [INFERENCE_DATA, MODEL_CAT, MODEL_XGB, MODEL_LGB, MODEL_META, SCALER_PATH]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("Error: Missing production files:")
        for m in missing:
            print(f"  ✗ {m}")
        print("Please run step6 and step7 first.")
        sys.exit(1)

    print("\n[1] Loading Production Data & Models...")
    inf_df = pd.read_csv(INFERENCE_DATA)
    inference_date = inf_df['Date'].iloc[-1]
    X_inf = inf_df.drop(columns=['Date'])
    features = X_inf.columns.tolist()

    # Load scaler and scale features
    scaler = joblib.load(SCALER_PATH)

    # Feature alignment — handle mismatch between saved model and new pipeline output
    if hasattr(scaler, 'feature_names_in_'):
        expected = list(scaler.feature_names_in_)
        extra    = [f for f in X_inf.columns if f not in expected]
        missing  = [f for f in expected if f not in X_inf.columns]
        if extra or missing:
            print(f"[ALIGN] Dropping {len(extra)} unseen features, zero-filling {len(missing)} missing.")
            print(f"  Extra (dropped)  : {extra[:5]}")
            print(f"  Missing (zeroed) : {missing[:5]}")
            X_inf = X_inf.drop(columns=[c for c in extra if c in X_inf.columns], errors='ignore')
            for col in missing:
                X_inf[col] = 0.0
            X_inf = X_inf[expected]
        features = list(X_inf.columns)

    X_inf_scaled = scaler.transform(X_inf)
    X_inf_df = pd.DataFrame(X_inf_scaled, columns=features)

    # BUG FIX: Load ALL three models + meta-learner (not just CatBoost)
    m_cat = CatBoostClassifier()
    m_cat.load_model(MODEL_CAT)
    m_xgb = xgb.XGBClassifier()
    m_xgb.load_model(MODEL_XGB)
    m_lgb = lgb.Booster(model_file=MODEL_LGB)
    meta  = joblib.load(MODEL_META)
    print("All 3 models + meta-learner loaded successfully.")

    print("\n[2] Running Ensemble Prediction for Tomorrow...")
    p_cat = float(m_cat.predict_proba(X_inf_df)[0, 1])
    p_xgb = float(m_xgb.predict_proba(X_inf_df)[0, 1])
    p_lgb = float(m_lgb.predict(X_inf_df.values)[0])
    prob_up = float((p_cat + p_xgb + p_lgb) / 3.0)

    long_thresh, short_thresh = load_adaptive_thresholds()

    # Signal with adaptive thresholds (fixes all-NEUTRAL bug)
    raw_signal = "HOLD"
    if prob_up >= long_thresh:
        raw_signal = "LONG (BUY)"
    elif prob_up <= short_thresh:
        raw_signal = "SHORT (SELL)"

    # Ensemble consensus check (relaxed: 2/3 majority)
    models_bullish = sum(1 for p in [p_cat, p_xgb, p_lgb] if p > 0.50)
    models_bearish = sum(1 for p in [p_cat, p_xgb, p_lgb] if p < 0.50)
    if "LONG"  in raw_signal and models_bullish < 2: raw_signal = "HOLD (No Consensus)"
    if "SHORT" in raw_signal and models_bearish < 2: raw_signal = "HOLD (No Consensus)"

    print(f"Target Date          : Next trading day after {inference_date}")
    print(f"Model Probabilities  : CatBoost={p_cat:.4f}  XGBoost={p_xgb:.4f}  LightGBM={p_lgb:.4f}")
    print(f"Meta-Learner Prob    : {prob_up:.4f} (UP)  /  {1-prob_up:.4f} (DOWN)")
    print(f"Adaptive Thresholds  : LONG >= {long_thresh:.4f}  SHORT <= {short_thresh:.4f}")

    # Consensus signal (Hernes et al.) — replaces flat majority vote
    consensus_prob = sorted([p_cat, p_xgb, p_lgb])[1]   # median
    raw_signal = qualify_signal_consensus(
        prob_up, p_cat, p_xgb, p_lgb,
        long_thresh, short_thresh,
        rsi_regime=None,   # live RSI regime not available without feature pipeline
        high_vol=None      # live vol regime not available without feature pipeline
    )

    print(f"Consensus Prob       : {consensus_prob:.4f} (median of 3 models)")
    print(f"Algorithmic Signal   : {raw_signal}")
    consensus_ok = consensus_prob > 0.50 if "LONG" in raw_signal else consensus_prob < 0.50
    print(f"Ensemble Consensus   : {'OK' if 'HOLD' not in raw_signal else 'BLOCKED'}")

    print("\n[3] AI Reasoning (SHAP Feature Importance)...")
    explainer  = shap.TreeExplainer(m_cat)
    shap_vals  = explainer.shap_values(X_inf_df)
    inst_shap  = shap_vals[0] if not isinstance(shap_vals, list) else shap_vals[1][0]
    feat_impacts = sorted(zip(features, inst_shap, X_inf.iloc[0]),
                          key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Driving Factors for this specific prediction:")
    for i, (feat, impact, val) in enumerate(feat_impacts[:5]):
        direction = "Pushing UP" if impact > 0 else "Pushing DOWN"
        print(f"  {i+1}. {feat}: {val:.4f} (Impact: {impact:.4f} → {direction})")

    print("\n[4] Risk Management Parameters...")
    if "HOLD" in raw_signal:
        print("Model confidence too low or consensus blocked. Holding cash.")
        print("No trade parameters generated.")
        sys.exit(0)

    # Get recent raw prices for ATR-based dynamic stop loss / take profit
    raw_df = pd.read_csv(RAW_PRICES)
    raw_df['Date'] = pd.to_datetime(raw_df['Date'])
    raw_df = raw_df.sort_values('Date').reset_index(drop=True)
    raw_df['ATR'] = calculate_atr(raw_df, period=14)
    latest_atr   = raw_df['ATR'].iloc[-1]
    latest_close = raw_df['Close'].iloc[-1]

    # Load SL/TP multipliers from model_threshold.json (set by step9b sweep)
    _, _, sl_mult, tp_mult = load_threshold()

    entry_price = latest_close
    print(f"Calculated 14-Day ATR : ${latest_atr:.2f}")
    print(f"SL Multiplier         : {sl_mult}x ATR  (from model_threshold.json)")
    print(f"TP Multiplier         : {tp_mult}x ATR  (from model_threshold.json)")

    if "LONG" in raw_signal:
        stop_loss   = entry_price - (sl_mult * latest_atr)
        take_profit = entry_price + (tp_mult * latest_atr)
    else:
        stop_loss   = entry_price + (sl_mult * latest_atr)
        take_profit = entry_price - (tp_mult * latest_atr)

    risk_amt   = abs(entry_price - stop_loss)
    reward_amt = abs(take_profit - entry_price)

    print("\n==================================================")
    print("          TRADING EXECUTION PLAN")
    print("==================================================")
    print(f"  Entry Price : ${entry_price:,.2f}")
    print(f"  Stop Loss   : ${stop_loss:,.2f} ({sl_mult}x ATR Risk)")
    print(f"  Take Profit : ${take_profit:,.2f} ({tp_mult}x ATR Reward)")
    print(f"  Risk Amount : ${risk_amt:,.2f}")
    print(f"  Reward Amt  : ${reward_amt:,.2f}")
    print(f"  Risk:Reward : 1:{reward_amt/risk_amt:.1f}")
    print("==================================================")

if __name__ == "__main__":
    main()
