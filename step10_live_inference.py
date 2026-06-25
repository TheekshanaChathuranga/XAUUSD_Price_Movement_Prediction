import os
import sys
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier
import shap

if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DATA = os.path.join(OUTPUT_DIR, "live_inference_data.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "catboost_prod.cbm")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")
RAW_PRICES = os.path.join(OUTPUT_DIR, "xauusd_raw_prices.csv")

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    return atr

def main():
    print("==================================================")
    print(" PHASE 7: LIVE INFERENCE & RISK MANAGEMENT ENGINE")
    print("==================================================")
    
    if not os.path.exists(INFERENCE_DATA) or not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: Missing production files. Please run step6 and step7 first.")
        sys.exit(1)
        
    print("\n[1] Loading Production Data & Models...")
    inf_df = pd.read_csv(INFERENCE_DATA)
    inference_date = inf_df['Date'].iloc[-1]
    
    X_inf = inf_df.drop(columns=['Date'])
    features = X_inf.columns.tolist()
    
    scaler = joblib.load(SCALER_PATH)
    X_inf_scaled = scaler.transform(X_inf)
    X_inf_df = pd.DataFrame(X_inf_scaled, columns=features)
    
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    print("CatBoost Production Model & Scaler loaded successfully.")
    
    print("\n[2] Generating Prediction for Tomorrow...")
    prob_up = model.predict_proba(X_inf_df)[0, 1]
    
    signal = "HOLD"
    if prob_up > 0.55:
        signal = "LONG (BUY)"
    elif prob_up < 0.45:
        signal = "SHORT (SELL)"
        
    print(f"Target Date         : Next trading day after {inference_date}")
    print(f"Probability (UP)    : {prob_up:.4f}")
    print(f"Algorithmic Signal  : {signal}")
    
    print("\n[3] AI Reasoning (SHAP Feature Importance)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_inf_df)
    
    # SHAP values for binary CatBoost is usually just a 1D array per instance if output is raw margin
    if isinstance(shap_values, list):
        instance_shap = shap_values[1][0] # class 1
    else:
        # CatBoost returns shape (1, n_features)
        instance_shap = shap_values[0]
        
    feature_impacts = list(zip(features, instance_shap, X_inf.iloc[0]))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("Top 3 Driving Factors for this specific prediction:")
    for i in range(3):
        feat, impact, val = feature_impacts[i]
        direction = "Pushing UP" if impact > 0 else "Pushing DOWN"
        print(f"  {i+1}. {feat}: {val:.4f} (Impact: {impact:.4f} -> {direction})")
        
    print("\n[4] Risk Management Parameters...")
    if signal == "HOLD":
        print("Model confidence is too low. Holding cash to protect capital.")
        print("No trade parameters generated.")
        sys.exit(0)
        
    # Get recent raw prices to calculate ATR for dynamic Stop Loss
    raw_df = pd.read_csv(RAW_PRICES)
    raw_df['Date'] = pd.to_datetime(raw_df['Date'])
    raw_df = raw_df.sort_values('Date').reset_index(drop=True)
    
    raw_df['ATR'] = calculate_atr(raw_df, period=14)
    latest_atr = raw_df['ATR'].iloc[-1]
    latest_close = raw_df['Close'].iloc[-1]
    
    print(f"Calculated 14-Day ATR : ${latest_atr:.2f}")
    
    entry_price = latest_close
    
    if "LONG" in signal:
        stop_loss = entry_price - (1.5 * latest_atr)
        take_profit = entry_price + (3.0 * latest_atr)
    else:
        stop_loss = entry_price + (1.5 * latest_atr)
        take_profit = entry_price - (3.0 * latest_atr)
        
    print("\n==================================================")
    print("          TRADING EXECUTION PLAN")
    print("==================================================")
    print(f"  Entry Price : ${entry_price:,.2f}")
    print(f"  Stop Loss   : ${stop_loss:,.2f} (1.5x ATR Risk)")
    print(f"  Take Profit : ${take_profit:,.2f} (3.0x ATR Reward)")
    print(f"  Risk:Reward : 1:2")
    print("==================================================")

if __name__ == "__main__":
    main()
