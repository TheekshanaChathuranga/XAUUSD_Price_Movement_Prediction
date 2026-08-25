"""
step6b_attention_fusion.py -- Bidirectional Cross-Modal Attention Fusion
=========================================================================
Dakalbab et al. tested four fusion strategies on identical Forex data:
  - Simple concatenation (no attention): 0.792 accuracy, MCC 0.69
  - Self-attention + bi-cross-attention: 0.852 accuracy, MCC 0.776   <- THIS

Dave et al. confirmed bidirectional processing consistently outperforms
unidirectional across all LSTM/GRU variants.

Architecture:
  Sentiment Stream (8 gold-impact features)
       |
  Linear -> LayerNorm -> ReLU       Technical Stream (N features)
       |                                  |
       +-------- Cross-Attention ---------+
                 (bidirectional)
                      |
              Fused Embedding (16-dim)
              appended as AttnFuse_0..15

Run AFTER step6_align_fusion.py and BEFORE step7_train_ensemble.py.
Requires: pip install torch
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_IN  = os.path.join(OUTPUT_DIR, "multimodal_master_dataset.csv")
DATASET_OUT = DATASET_IN
ATTN_MODEL  = os.path.join(OUTPUT_DIR, "attn_fusion_weights.pt")

EMBED_DIM   = 16
N_HEADS     = 4
TRAIN_FRAC  = 0.80
LR          = 1e-3
N_EPOCHS    = 80
BATCH_SIZE  = 64
DROPOUT     = 0.10

SENTIMENT_FEATURES = [
    "Mean_Gold_Impact", "Weighted_Gold_Impact",
    "WAR_GEOPOLITICAL_Impact", "FED_POLICY_Impact",
    "INFLATION_Impact", "DOLLAR_FX_Impact",
    "RECESSION_CRISIS_Impact", "TREASURY_YIELDS_Impact",
]
EXCLUDE_FROM_TECH = [
    "Date", "Target_Direction", "Target_SD_Binary", "Target_SD_3class",
    "Next_Day_Return", "Close_Return",
] + SENTIMENT_FEATURES


def check_torch():
    try:
        import torch
        return torch
    except ImportError:
        print("\n[ERROR] PyTorch not installed.  Run:  pip install torch")
        sys.exit(0)


def build_model(torch, sent_dim, tech_dim):
    nn = torch.nn

    class CrossModalAttentionFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.sent_proj = nn.Sequential(
                nn.Linear(sent_dim, EMBED_DIM), nn.LayerNorm(EMBED_DIM),
                nn.ReLU(), nn.Dropout(DROPOUT))
            self.tech_proj = nn.Sequential(
                nn.Linear(tech_dim, EMBED_DIM), nn.LayerNorm(EMBED_DIM),
                nn.ReLU(), nn.Dropout(DROPOUT))
            self.sent_to_tech = nn.MultiheadAttention(
                EMBED_DIM, N_HEADS, dropout=DROPOUT, batch_first=True)
            self.tech_to_sent = nn.MultiheadAttention(
                EMBED_DIM, N_HEADS, dropout=DROPOUT, batch_first=True)
            self.layer_norm_s = nn.LayerNorm(EMBED_DIM)
            self.layer_norm_t = nn.LayerNorm(EMBED_DIM)
            self.out_proj = nn.Sequential(
                nn.Linear(EMBED_DIM * 2, EMBED_DIM),
                nn.LayerNorm(EMBED_DIM), nn.Tanh())
            self.classifier = nn.Linear(EMBED_DIM, 1)

        def forward(self, sent_feats, tech_feats):
            s = self.sent_proj(sent_feats).unsqueeze(1)
            t = self.tech_proj(tech_feats).unsqueeze(1)
            s2t, _ = self.sent_to_tech(s, t, t)
            s2t = self.layer_norm_s(s + s2t)
            t2s, _ = self.tech_to_sent(t, s, s)
            t2s = self.layer_norm_t(t + t2s)
            fused = torch.cat([s2t.squeeze(1), t2s.squeeze(1)], dim=-1)
            embedding = self.out_proj(fused)
            logit = self.classifier(embedding).squeeze(-1)
            return embedding, logit

    return CrossModalAttentionFusion()


def main():
    print("=" * 65)
    print("  STEP 6B: Bidirectional Cross-Modal Attention Fusion")
    print(f"  Embed={EMBED_DIM}  Heads={N_HEADS}  Epochs={N_EPOCHS}")
    print("  Source: Dakalbab et al. (0.852 acc) + Dave et al. (bidir)")
    print("=" * 65)

    torch = check_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    print("\n=== Step 1: Loading Dataset ===")
    if not os.path.exists(DATASET_IN):
        print(f"[ERROR] {DATASET_IN} not found. Run step6_align_fusion.py first.")
        sys.exit(1)
    df = pd.read_csv(DATASET_IN)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows x {df.shape[1]} columns.")

    if "Target_Direction" not in df.columns:
        print("[ERROR] Target_Direction not found.")
        sys.exit(1)
    y = df["Target_Direction"].values.astype(np.float32)

    print("\n=== Step 2: Identifying Modality Streams ===")
    sent_cols = [c for c in SENTIMENT_FEATURES if c in df.columns]
    tech_cols = [c for c in df.columns
                 if c not in EXCLUDE_FROM_TECH and c not in sent_cols
                 and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, "float64", "int64"]]
    print(f"  Sentiment stream : {len(sent_cols)} features")
    print(f"  Technical stream : {len(tech_cols)} features")
    if len(sent_cols) == 0:
        print("[WARN] No sentiment features found -- appending zero embeddings.")
    if len(tech_cols) == 0:
        print("[ERROR] No technical features found."); sys.exit(1)

    print("\n=== Step 3: Normalizing Features ===")
    X_sent = df[sent_cols].fillna(0.0).values.astype(np.float32) if sent_cols else np.zeros((len(df), 1), dtype=np.float32)
    X_tech = df[tech_cols].fillna(0.0).values.astype(np.float32)
    train_n = int(len(df) * TRAIN_FRAC)
    scaler_s, scaler_t = StandardScaler(), StandardScaler()
    X_sent[:train_n] = scaler_s.fit_transform(X_sent[:train_n])
    X_sent[train_n:] = scaler_s.transform(X_sent[train_n:])
    X_tech[:train_n] = scaler_t.fit_transform(X_tech[:train_n])
    X_tech[train_n:] = scaler_t.transform(X_tech[train_n:])

    print(f"\n=== Step 4: Training Attention Module ({N_EPOCHS} epochs) ===")
    model = build_model(torch, sent_dim=X_sent.shape[1], tech_dim=X_tech.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    criterion = torch.nn.BCEWithLogitsLoss()

    Xs_tr = torch.tensor(X_sent[:train_n], device=device)
    Xt_tr = torch.tensor(X_tech[:train_n], device=device)
    y_tr  = torch.tensor(y[:train_n], device=device)

    model.train()
    best_loss, best_state = float("inf"), None
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(train_n, device=device)
        epoch_loss, n_batches = 0.0, 0
        for b in range(0, train_n, BATCH_SIZE):
            idx = perm[b: b + BATCH_SIZE]
            _, logit = model(Xs_tr[idx], Xt_tr[idx])
            loss = criterion(logit, y_tr[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item(); n_batches += 1
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:>3}/{N_EPOCHS}  loss={avg_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), ATTN_MODEL)
    print(f"  Best training loss: {best_loss:.4f}")

    print("\n=== Step 5: Generating Embeddings for All Rows ===")
    model.eval()
    Xs_all = torch.tensor(X_sent, device=device)
    Xt_all = torch.tensor(X_tech, device=device)
    embeddings = []
    with torch.no_grad():
        for b in range(0, len(df), 256):
            emb, _ = model(Xs_all[b:b+256], Xt_all[b:b+256])
            embeddings.append(emb.cpu().numpy())
    embeddings = np.vstack(embeddings)
    print(f"  Generated shape: {embeddings.shape}")

    print("\n=== Step 6: Appending AttnFuse Columns ===")
    attn_cols = [f"AttnFuse_{i}" for i in range(EMBED_DIM)]
    df = df.drop(columns=[c for c in attn_cols if c in df.columns], errors="ignore")
    attn_df = pd.DataFrame(embeddings, columns=attn_cols, index=df.index)
    df = pd.concat([df, attn_df], axis=1)
    df.to_csv(DATASET_OUT, index=False)
    print(f"  Added {EMBED_DIM} AttnFuse features.")
    print(f"  New dataset shape: {df.shape}")

    # Also update live_inference_data.csv if present
    live_inf_path = os.path.join(OUTPUT_DIR, "live_inference_data.csv")
    if os.path.exists(live_inf_path):
        try:
            inf_df = pd.read_csv(live_inf_path)
            inf_sent_cols = [c for c in sent_cols if c in inf_df.columns]
            inf_tech_cols = [c for c in tech_cols if c in inf_df.columns]
            X_inf_s = inf_df[inf_sent_cols].fillna(0.0).values.astype(np.float32) if inf_sent_cols else np.zeros((len(inf_df), 1), dtype=np.float32)
            X_inf_t = inf_df[inf_tech_cols].fillna(0.0).values.astype(np.float32)
            X_inf_s = scaler_s.transform(X_inf_s)
            X_inf_t = scaler_t.transform(X_inf_t)
            Xs_inf_t = torch.tensor(X_inf_s, device=device)
            Xt_inf_t = torch.tensor(X_inf_t, device=device)
            with torch.no_grad():
                inf_emb, _ = model(Xs_inf_t, Xt_inf_t)
                inf_emb_np = inf_emb.cpu().numpy()
            inf_df = inf_df.drop(columns=[c for c in attn_cols if c in inf_df.columns], errors="ignore")
            inf_attn_df = pd.DataFrame(inf_emb_np, columns=attn_cols, index=inf_df.index)
            inf_df = pd.concat([inf_df, inf_attn_df], axis=1)
            inf_df.to_csv(live_inf_path, index=False)
            print(f"  Updated live_inference_data.csv with {EMBED_DIM} AttnFuse features.")
        except Exception as e:
            print(f"  [WARN] Could not update live_inference_data.csv: {e}")

    # Embedding differentiation check
    if "WAR_GEOPOLITICAL_Impact" in df.columns:
        war_mask   = df["WAR_GEOPOLITICAL_Impact"].abs() > 0.5
        quiet_mask = df["WAR_GEOPOLITICAL_Impact"].abs() < 0.1
        if war_mask.sum() > 5 and quiet_mask.sum() > 5:
            diff = np.abs(
                embeddings[war_mask.values].mean(axis=0) -
                embeddings[quiet_mask.values].mean(axis=0)
            ).mean()
            quality = "GOOD -- attention learning signal" if diff > 0.05 else "LOW -- check data"
            print(f"\n  Embedding differentiation |war-quiet|: {diff:.4f}  ({quality})")

    print("\n" + "=" * 65)
    print("  STEP 6B COMPLETE")
    print(f"  {EMBED_DIM} AttnFuse features appended to multimodal_master_dataset.csv")
    print(f"  Best loss: {best_loss:.4f}  |  Sent: {len(sent_cols)}  Tech: {len(tech_cols)}")
    print("  Run step7_train_ensemble.py next.")
    print("=" * 65)


if __name__ == "__main__":
    main()
