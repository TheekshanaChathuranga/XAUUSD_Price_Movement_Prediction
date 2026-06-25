import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap

# Fix Unicode encoding on Windows
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_IN = os.path.join(OUTPUT_DIR, "multimodal_master_dataset.csv")

def main():
    print("=== Step 1: Loading Dataset & Splitting Chronologically ===")
    if not os.path.exists(DATASET_IN):
        print(f"Error: Fused dataset {DATASET_IN} not found!")
        sys.exit(1)

    df = pd.read_csv(DATASET_IN)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    X = df.drop(columns=['Date', 'Target_Direction'])
    y = df['Target_Direction']

    # Chronological Split (80% Train, 20% Test)
    train_size = int(len(df) * 0.8)
    X_train_raw, X_test_raw = X.iloc[:train_size], X.iloc[train_size:]
    y_train_raw, y_test_raw = y.iloc[:train_size], y.iloc[train_size:]

    print("\n=== Step 2: Standardizing Features ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # Re-convert to DataFrame to maintain column names in SHAP
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    print("\n=== Step 3: Training XGBoost Classifier ===")
    xgb_model = XGBClassifier(
        max_depth=4,
        learning_rate=0.01,
        n_estimators=200,
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X_train_scaled_df, y_train_raw)
    print("XGBoost training complete.")

    print("\n=== Step 4: Computing SHAP Feature Importances ===")
    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(xgb_model)
    
    # Compute SHAP values on test set
    shap_values = explainer(X_test_scaled_df)
    
    # Get mean absolute SHAP values for each feature
    # shap_values.values is a 2D array of shape (samples, features)
    mean_abs_shaps = np.abs(shap_values.values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Mean_Absolute_SHAP': mean_abs_shaps
    }).sort_values(by='Mean_Absolute_SHAP', ascending=False).reset_index(drop=True)
    
    print("\n=== SHAP Feature Importance Report (Top 15 Features) ===")
    print("="*65)
    print(f"{'Rank':<5} | {'Feature Name':<25} | {'Mean Abs SHAP (Impact)':<22}")
    print("-"*65)
    for idx, row in importance_df.head(15).iterrows():
        print(f"{idx+1:<5} | {row['Feature']:<25} | {row['Mean_Absolute_SHAP']:.6f}")
    print("="*65)
    
    # Highlight potential leaks or weird importances
    print("\n=== Diagnostics & Analysis ===")
    top_feature = importance_df.loc[0, 'Feature']
    top_val = importance_df.loc[0, 'Mean_Absolute_SHAP']
    print(f"  • Top influential feature: '{top_feature}' (impact={top_val:.6f})")
    
    # Check if there is any obvious target leak (e.g. features with extremely high impact compared to others)
    ratio = top_val / (importance_df.loc[1, 'Mean_Absolute_SHAP'] + 1e-8)
    if ratio > 5.0:
        print(f"  ⚠️ Warning: '{top_feature}' has {ratio:.1f}x the impact of the next feature. This could indicate a look-ahead leak!")
    else:
        print("  ✓ No obvious single-feature dominance detected (leak check passed).")
        
    print("\nInterpretability analysis finished successfully.")

if __name__ == "__main__":
    main()
