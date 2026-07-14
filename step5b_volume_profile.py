"""
STEP 5B: Gold Futures Volume Profile Feature Engineering
=========================================================
Computes per-day Volume Profile features from XAUUSD OHLCV data
(Tick_Volume from GC=F Gold Futures via Yahoo Finance).

Volume Profile reveals WHERE volume traded across price levels — not just
how much volume occurred over time. For Gold Futures, these levels act as
powerful support/resistance magnets that the ML ensemble currently lacks.

Features produced per trading day:
──────────────────────────────────
  POC              — Point of Control (price level with most volume)
  VAH              — Value Area High  (top of 70% volume zone)
  VAL              — Value Area Low   (bottom of 70% volume zone)
  POC_Distance     — (Close - POC) / POC  — normalised distance from fair value
  VArea_Width      — (VAH - VAL) / POC    — width of value area (volatility proxy)
  Price_vs_VAH     — (Close - VAH) / POC  — positive = above VAH (breakout zone)
  Price_vs_VAL     — (Close - VAL) / POC  — negative = below VAL (breakdown zone)
  In_HVN           — 1 if Close is inside a High Volume Node (sticky level)
  In_LVN           — 1 if Close is inside a Low Volume Node  (fast-move zone)
  Vol_Imbalance    — |POC_Distance| weighted by volume skew direction
  VP_Long_Bias     — 1 if price below POC (undervalued vs. market consensus)
  VP_Short_Bias    — 1 if price above POC (overvalued vs. market consensus)

Win-Rate Mechanism:
───────────────────
  1. POC Distance Filter  → only trade in direction consistent with POC location
  2. LVN Zone Detection   → flag unstable price zones (reduce position size)
  3. VAH/VAL Breakout     → confirm or suppress signals at key boundaries

Lookback windows computed:
  VP_20  — short-term (1-month) profile
  VP_60  — medium-term (quarterly) profile  ← PRIMARY
  VP_252 — long-term (yearly) profile
"""
import os
import sys
import numpy as np
import pandas as pd

if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__))
PRICES_IN   = os.path.join(OUTPUT_DIR, "xauusd_raw_prices.csv")
VP_OUT      = os.path.join(OUTPUT_DIR, "volume_profile_features.csv")

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
N_BINS          = 100     # price buckets for volume histogram
VALUE_AREA_PCT  = 0.70    # standard 70% value area
HVN_THRESHOLD   = 1.50    # bucket volume > 1.5× mean → HVN
LVN_THRESHOLD   = 0.30    # bucket volume < 0.3× mean → LVN
LOOKBACK_SHORT  = 20      # short-term VP (1 month)
LOOKBACK_MID    = 60      # medium-term VP (quarter) — primary signals
LOOKBACK_LONG   = 252     # long-term VP (year) — structural levels


# ══════════════════════════════════════════════════════════════════════════════
# CORE VOLUME PROFILE CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_volume_profile(ohlcv_window: pd.DataFrame, n_bins: int = N_BINS) -> dict:
    """
    Compute Volume Profile for a window of OHLCV bars.

    Volume Distribution Method:
        Each bar distributes its Tick_Volume uniformly across the price range
        [Low, High]. This approximates intrabar volume distribution without
        tick-level data (unavailable at daily resolution).

    Args:
        ohlcv_window:  DataFrame with columns [Open, High, Low, Close, Tick_Volume]
        n_bins:        Number of price buckets in the histogram

    Returns:
        dict with POC, VAH, VAL, bin_edges, bin_volumes, mean_bin_vol
    """
    if len(ohlcv_window) < 5:
        return None

    low_price  = ohlcv_window['Low'].min()
    high_price = ohlcv_window['High'].max()

    if high_price <= low_price:
        return None

    # Create price bins
    bin_edges  = np.linspace(low_price, high_price, n_bins + 1)
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    bin_vols   = np.zeros(n_bins, dtype=np.float64)

    for _, bar in ohlcv_window.iterrows():
        bar_range = bar['High'] - bar['Low']
        if bar_range < 1e-9:
            # Zero-range bar: all volume at Close
            idx = np.searchsorted(bin_edges[1:], bar['Close'], side='left')
            idx = min(idx, n_bins - 1)
            bin_vols[idx] += bar['Tick_Volume']
            continue

        # Distribute volume across bins proportional to overlap with [Low, High]
        bar_low  = bar['Low']
        bar_high = bar['High']
        vol      = bar['Tick_Volume']

        for b in range(n_bins):
            overlap_low  = max(bin_edges[b], bar_low)
            overlap_high = min(bin_edges[b + 1], bar_high)
            if overlap_high > overlap_low:
                fraction = (overlap_high - overlap_low) / bar_range
                bin_vols[b] += vol * fraction

    total_vol   = bin_vols.sum()
    if total_vol < 1e-9:
        return None

    # Point of Control: bin with maximum volume
    poc_bin = np.argmax(bin_vols)
    poc     = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2.0

    # Value Area: accumulate 70% of total volume starting from POC bin,
    # expanding outward to higher/lower bins
    target_vol  = total_vol * VALUE_AREA_PCT
    sorted_bins = np.argsort(bin_vols)[::-1]  # descending by volume

    va_vol  = 0.0
    va_bins = []
    for b in sorted_bins:
        va_vol += bin_vols[b]
        va_bins.append(b)
        if va_vol >= target_vol:
            break

    va_bins_arr = np.array(va_bins)
    vah_bin = va_bins_arr.max()
    val_bin = va_bins_arr.min()
    vah     = bin_edges[vah_bin + 1]  # top edge of highest VA bin
    val     = bin_edges[val_bin]      # bottom edge of lowest VA bin

    mean_bin_vol = bin_vols.mean()

    return {
        'poc':          poc,
        'vah':          vah,
        'val':          val,
        'bin_edges':    bin_edges,
        'bin_vols':     bin_vols,
        'mean_bin_vol': mean_bin_vol,
        'total_vol':    total_vol,
    }


def classify_price_zone(close: float, bin_edges: np.ndarray,
                         bin_vols: np.ndarray, mean_bin_vol: float) -> tuple[int, int]:
    """
    Classify current Close price as inside HVN or LVN zone.

    Returns:
        (in_hvn, in_lvn): binary flags
    """
    # Find which bin the close falls into
    bin_idx = np.searchsorted(bin_edges[1:], close, side='left')
    bin_idx = min(bin_idx, len(bin_vols) - 1)

    vol_at_close = bin_vols[bin_idx]
    in_hvn = int(vol_at_close > HVN_THRESHOLD * mean_bin_vol)
    in_lvn = int(vol_at_close < LVN_THRESHOLD * mean_bin_vol)
    return in_hvn, in_lvn


def extract_vp_features(close: float, vp: dict) -> dict:
    """
    Extract scalar features from a computed Volume Profile for a given Close price.

    Args:
        close: Current day's closing price
        vp:    Output dict from compute_volume_profile()

    Returns:
        dict of normalised scalar features
    """
    poc = vp['poc']
    vah = vp['vah']
    val = vp['val']

    poc_ref = poc if poc > 1e-9 else 1.0

    poc_distance  = (close - poc) / poc_ref          # + = above POC, - = below
    varea_width   = (vah - val) / poc_ref            # value area width normalised
    price_vs_vah  = (close - vah) / poc_ref          # + = above VAH (breakout)
    price_vs_val  = (close - val) / poc_ref          # - = below VAL (breakdown)

    in_hvn, in_lvn = classify_price_zone(
        close, vp['bin_edges'], vp['bin_vols'], vp['mean_bin_vol']
    )

    # Volume Imbalance: asymmetry of volume above/below POC
    poc_bin      = np.argmax(vp['bin_vols'])
    vol_above    = vp['bin_vols'][poc_bin:].sum()
    vol_below    = vp['bin_vols'][:poc_bin + 1].sum()
    vol_total    = vol_above + vol_below
    vol_imbalance = (vol_above - vol_below) / (vol_total + 1e-9)  # -1 to +1

    # Directional bias flags
    vp_long_bias  = int(close < poc)   # price below fair value → mean-reversion long
    vp_short_bias = int(close > poc)   # price above fair value → mean-reversion short

    return {
        'POC':            round(poc, 4),
        'VAH':            round(vah, 4),
        'VAL':            round(val, 4),
        'POC_Distance':   round(poc_distance, 6),
        'VArea_Width':    round(varea_width, 6),
        'Price_vs_VAH':   round(price_vs_vah, 6),
        'Price_vs_VAL':   round(price_vs_val, 6),
        'In_HVN':         in_hvn,
        'In_LVN':         in_lvn,
        'Vol_Imbalance':  round(vol_imbalance, 6),
        'VP_Long_Bias':   vp_long_bias,
        'VP_Short_Bias':  vp_short_bias,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ROLLING VP COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_rolling_vp(prices: pd.DataFrame, lookback: int, suffix: str) -> pd.DataFrame:
    """
    Compute rolling Volume Profile features for every trading day.

    Args:
        prices:   DataFrame with [Date, Open, High, Low, Close, Tick_Volume]
        lookback: Rolling window size in trading days
        suffix:   Column name suffix (e.g. '_60' for 60-day profile)

    Returns:
        DataFrame with Date + suffixed VP feature columns
    """
    print(f"  Computing {lookback}-day rolling Volume Profile ({len(prices)} bars)...")

    records = []
    n = len(prices)

    for i in range(n):
        date  = prices['Date'].iloc[i]
        close = prices['Close'].iloc[i]

        # Require at least half the lookback window before computing
        start = max(0, i - lookback + 1)
        window = prices.iloc[start:i + 1]

        vp = compute_volume_profile(window)
        if vp is None:
            records.append({'Date': date})
            continue

        features = extract_vp_features(close, vp)
        features['Date'] = date
        records.append(features)

    df = pd.DataFrame(records)

    # Rename feature columns with suffix
    feature_cols = [c for c in df.columns if c != 'Date']
    df.rename(columns={c: f'{c}{suffix}' for c in feature_cols}, inplace=True)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def build_volume_profile_features(prices_path: str = PRICES_IN,
                                   output_path: str = VP_OUT) -> pd.DataFrame:
    """
    Full pipeline: load prices → compute multi-window VP → save CSV.

    Returns the merged VP feature DataFrame.
    """
    print("=" * 65)
    print("  STEP 5B: Gold Futures Volume Profile Feature Engineering")
    print("=" * 65)

    # ── Load prices ────────────────────────────────────────────────────────
    if not os.path.exists(prices_path):
        print(f"[ERROR] Price file not found: {prices_path}")
        sys.exit(1)

    prices = pd.read_csv(prices_path)
    prices['Date']        = pd.to_datetime(prices['Date'])
    prices['Tick_Volume'] = pd.to_numeric(prices['Tick_Volume'], errors='coerce').fillna(0.0)
    prices = prices.sort_values('Date').reset_index(drop=True)

    required = ['Open', 'High', 'Low', 'Close', 'Tick_Volume']
    missing  = [c for c in required if c not in prices.columns]
    if missing:
        print(f"[ERROR] Missing columns in price file: {missing}")
        sys.exit(1)

    # Replace any zero volume with 1 to avoid degenerate profiles
    prices['Tick_Volume'] = prices['Tick_Volume'].replace(0, 1.0)

    print(f"\n  Loaded {len(prices):,} bars of price data.")
    print(f"  Date range : {prices['Date'].min().date()} → {prices['Date'].max().date()}")
    print(f"  Tick_Volume: min={prices['Tick_Volume'].min():.0f}  "
          f"max={prices['Tick_Volume'].max():.0f}  "
          f"mean={prices['Tick_Volume'].mean():.1f}")
    print()

    # ── Compute rolling VP for each window ────────────────────────────────
    vp_short = compute_rolling_vp(prices, LOOKBACK_SHORT, f'_{LOOKBACK_SHORT}')
    print(f"    Done. Columns: {[c for c in vp_short.columns if c != 'Date']}\n")

    vp_mid   = compute_rolling_vp(prices, LOOKBACK_MID,   f'_{LOOKBACK_MID}')
    print(f"    Done.\n")

    vp_long  = compute_rolling_vp(prices, LOOKBACK_LONG,  f'_{LOOKBACK_LONG}')
    print(f"    Done.\n")

    # ── Merge all windows ─────────────────────────────────────────────────
    vp_all = vp_short.merge(vp_mid,  on='Date', how='inner')
    vp_all = vp_all.merge(vp_long, on='Date', how='inner')

    # ── Cross-window features ─────────────────────────────────────────────
    # POC Confluence: all three windows agree on direction
    mid_bias  = vp_all[f'VP_Long_Bias_{LOOKBACK_MID}']
    long_bias = vp_all[f'VP_Long_Bias_{LOOKBACK_LONG}']
    short_bias_mid  = vp_all[f'VP_Short_Bias_{LOOKBACK_MID}']
    short_bias_long = vp_all[f'VP_Short_Bias_{LOOKBACK_LONG}']

    vp_all['VP_Confluence_Long']  = ((mid_bias == 1) & (long_bias == 1)).astype(int)
    vp_all['VP_Confluence_Short'] = ((short_bias_mid == 1) & (short_bias_long == 1)).astype(int)

    # POC Distance Spread (medium minus long): shows if price is trending toward
    # or away from the long-term fair value
    vp_all['POC_Distance_Spread'] = (
        vp_all[f'POC_Distance_{LOOKBACK_MID}'] - vp_all[f'POC_Distance_{LOOKBACK_LONG}']
    )

    # In LVN on ANY window → unreliable signal zone
    vp_all['In_Any_LVN'] = (
        (vp_all[f'In_LVN_{LOOKBACK_SHORT}'] == 1) |
        (vp_all[f'In_LVN_{LOOKBACK_MID}']   == 1)
    ).astype(int)

    # In HVN on multiple windows → very sticky level
    vp_all['In_Multi_HVN'] = (
        (vp_all[f'In_HVN_{LOOKBACK_SHORT}'] == 1) &
        (vp_all[f'In_HVN_{LOOKBACK_MID}']   == 1)
    ).astype(int)

    # VAH Breakout Strength: price above both short and mid VAH → strong long
    vp_all['VAH_Breakout_Strength'] = (
        (vp_all[f'Price_vs_VAH_{LOOKBACK_SHORT}'] > 0) &
        (vp_all[f'Price_vs_VAH_{LOOKBACK_MID}']   > 0)
    ).astype(int)

    # VAL Breakdown Strength: price below both short and mid VAL → strong short
    vp_all['VAL_Breakdown_Strength'] = (
        (vp_all[f'Price_vs_VAL_{LOOKBACK_SHORT}'] < 0) &
        (vp_all[f'Price_vs_VAL_{LOOKBACK_MID}']   < 0)
    ).astype(int)

    # Fill any NaNs (from early window where lookback > available data)
    vp_all = vp_all.fillna(0.0)

    print(f"\n  Volume Profile features shape : {vp_all.shape}")
    print(f"  Feature columns ({len(vp_all.columns) - 1} total):")
    for col in sorted(vp_all.columns):
        if col != 'Date':
            print(f"    {col}")

    # ── Validation ────────────────────────────────────────────────────────
    print("\n  Validation:")
    null_counts = vp_all.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts) == 0:
        print("    No missing values. ✓")
    else:
        print(f"    [WARN] Columns with NaNs:\n{null_counts}")

    poc_mid = vp_all[f'POC_{LOOKBACK_MID}']
    valid_poc = poc_mid[poc_mid > 0]
    print(f"    POC_60 range: ${valid_poc.min():.2f} → ${valid_poc.max():.2f}  (n={len(valid_poc)})")

    long_bias_pct  = vp_all['VP_Confluence_Long'].mean() * 100
    short_bias_pct = vp_all['VP_Confluence_Short'].mean() * 100
    lvn_pct        = vp_all['In_Any_LVN'].mean() * 100
    hvn_pct        = vp_all['In_Multi_HVN'].mean() * 100
    print(f"    VP Confluence Long  : {long_bias_pct:.1f}% of days")
    print(f"    VP Confluence Short : {short_bias_pct:.1f}% of days")
    print(f"    In LVN (any window) : {lvn_pct:.1f}% of days")
    print(f"    In HVN (multi-win)  : {hvn_pct:.1f}% of days")

    # ── Save ─────────────────────────────────────────────────────────────
    vp_all.to_csv(output_path, index=False)
    print(f"\n[SAVED] {output_path}")
    print(f"        {len(vp_all):,} rows × {vp_all.shape[1]} columns")
    print("\n✅ Step 5B: Volume Profile feature engineering complete.")

    return vp_all


if __name__ == "__main__":
    build_volume_profile_features()
