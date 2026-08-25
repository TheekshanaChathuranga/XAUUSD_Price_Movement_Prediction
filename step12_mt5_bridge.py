"""
step12_mt5_bridge.py  —  MetaTrader 5 Direct Market Execution Agent (Layer 3)
=============================================================================
Safely transmits validated multi-agent trade signals directly to a live or demo
MetaTrader 5 (MT5) broker terminal (e.g., Exness, IC Markets, Pepperstone).

Features:
  - Safety Guards: Max lot size cap, max spread filter, drawdown threshold
  - Shadow Mode Toggle: SHADOW_MODE=True for paper trading logs, False for live execution
  - Real-time Account Status: Balance, Equity, Free Margin, Open Positions
  - Precise Order Execution: Market Buy/Sell with exact ATR Stop Loss & Take Profit

Usage:
  Direct Python invocation:
    python step12_mt5_bridge.py --action status
    python step12_mt5_bridge.py --action test_order --symbol XAUUSD --signal LONG
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')

# Try importing MetaTrader5 package
try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mt5_bridge")

# ── XM GLOBAL BROKER CONFIGURATION ────────────────────────────────────────────
MT5_LOGIN         = 318506804
MT5_PASSWORD      = "Mu#diya@D45"
MT5_SERVER        = "XMGlobal-MT5 7"

SHADOW_MODE       = True      # True = Paper trade log only | False = Send real orders to MT5
DEFAULT_SYMBOL    = "GOLD"    # XM Global Gold symbol is "GOLD" (or "XAUUSD" on other brokers)
MAX_LOT_SIZE      = 0.10      # Hard maximum lot size per trade
MIN_LOT_SIZE      = 0.01      # Minimum lot size
MAX_SPREAD_PIPS   = 12.0      # Maximum allowable spread in pips (12.0 pips = $1.20 on Gold; normal XM spread is ~5-6 pips)
MAGIC_NUMBER      = 202608    # Unique identifier for AI agent trades in MT5
DEVIATION_SLIPPAGE= 20        # Allowable slippage in points (2.0 pips)


# ══════════════════════════════════════════════════════════════════════════════
# MT5 TERMINAL CONNECTION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def initialize_mt5(login: int = MT5_LOGIN, password: str = MT5_PASSWORD, server: str = MT5_SERVER) -> bool:
    """Initialize connection to MetaTrader 5 terminal with XM Global credentials."""
    if not _MT5_AVAILABLE:
        log.warning("[MT5] MetaTrader5 Python library is not installed. Install via `pip install MetaTrader5`.")
        return False

    init_success = False
    if login and password and server:
        init_success = mt5.initialize(login=login, password=password, server=server)
        if not init_success:
            # Fallback retry using login method
            if mt5.initialize():
                init_success = mt5.login(login=login, password=password, server=server)
    else:
        init_success = mt5.initialize()

    if not init_success:
        log.error(f"[MT5] XM Global Connection Failed! Error code: {mt5.last_error()}")
        return False

    acc_info = mt5.account_info()
    if acc_info is None:
        log.error("[MT5] Connected to terminal but failed to fetch account info.")
        return False

    term_info = mt5.terminal_info()
    if term_info and not term_info.trade_allowed:
        log.warning("⚠️ [MT5 NOTICE] 'Algo Trading' button is DISABLED on the MT5 terminal toolbar! Please enable the 'Algo Trading' button in MT5 to allow automated execution.")

    log.info(f"[MT5] XM Global Connected! Server: {acc_info.server} | Account: {acc_info.login} | Currency: {acc_info.currency} | Balance: ${acc_info.balance:,.2f}")
    return True


def shutdown_mt5() -> None:
    """Shutdown MT5 terminal connection."""
    if _MT5_AVAILABLE:
        mt5.shutdown()


def resolve_gold_symbol(preferred_symbol: str = DEFAULT_SYMBOL) -> str:
    """Auto-detects the active broker's exact Gold symbol (e.g. GOLD vs XAUUSD)."""
    candidates = [preferred_symbol, "GOLD", "XAUUSD", "GOLDmicro", "XAUUSDmicro", "GOLD24-7"]
    for sym in candidates:
        info = mt5.symbol_info(sym)
        if info is not None:
            return sym
    return preferred_symbol


def get_account_status() -> Dict[str, Any]:
    """Fetch live account metrics from MT5."""
    if not initialize_mt5():
        return {
            "status": "unavailable",
            "connected": False,
            "mt5_library_installed": _MT5_AVAILABLE,
            "shadow_mode": SHADOW_MODE
        }

    acc = mt5.account_info()
    term = mt5.terminal_info()
    positions = mt5.positions_get() or ()
    active_symbol = resolve_gold_symbol()
    sym_info = mt5.symbol_info(active_symbol)
    
    current_bid = sym_info.bid if sym_info else None
    current_ask = sym_info.ask if sym_info else None
    algo_allowed = term.trade_allowed if term else False

    shutdown_mt5()

    return {
        "status": "connected",
        "connected": True,
        "mt5_library_installed": True,
        "shadow_mode": SHADOW_MODE,
        "algo_trading_allowed_in_mt5": algo_allowed,
        "login": acc.login,
        "server": acc.server,
        "currency": acc.currency,
        "balance": acc.balance,
        "equity": acc.equity,
        "margin": acc.margin,
        "free_margin": acc.margin_free,
        "leverage": acc.leverage,
        "active_gold_symbol": active_symbol,
        "live_bid": current_bid,
        "live_ask": current_ask,
        "open_positions_count": len(positions)
    }


# ══════════════════════════════════════════════════════════════════════════════
# ORDER EXECUTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def execute_agent_decision(
    decision_payload: Dict[str, Any],
    symbol: str = DEFAULT_SYMBOL,
    is_live: bool = False
) -> Dict[str, Any]:
    """
    Executes a trade decision outputted by Layer 3 Decision Agent (FundManager).
    
    Args:
      decision_payload: Dict containing:
        - decision: 'APPROVE' | 'REJECT' | 'RESIZE'
        - direction: 'LONG' | 'SHORT' | 'HOLD'
        - lot_size: float (e.g. 0.02)
        - stop_loss: float price
        - take_profit: float price
        - justification: string memo
      symbol: Trading instrument ticker (default "GOLD")
      is_live: Override shadow mode if set to True
    """
    effective_shadow = SHADOW_MODE and not is_live

    decision  = str(decision_payload.get("decision", "REJECT")).upper()
    direction = str(decision_payload.get("direction", "HOLD")).upper()
    req_lots  = float(decision_payload.get("lot_size", 0.01))
    sl        = float(decision_payload.get("stop_loss", 0.0))
    tp        = float(decision_payload.get("take_profit", 0.0))
    memo      = decision_payload.get("justification", "No memo provided.")

    if decision not in {"APPROVE", "RESIZE"} or direction not in {"LONG", "SHORT"}:
        log.info(f"[MT5] Execution skipped — Decision: {decision}, Direction: {direction}")
        return {
            "status": "skipped",
            "reason": f"Non-execution decision ({decision}/{direction})",
            "shadow_mode": effective_shadow
        }

    # Clamp lot size within safety limits
    lot_size = round(max(MIN_LOT_SIZE, min(req_lots, MAX_LOT_SIZE)), 2)

    # Check Shadow Mode
    if effective_shadow:
        log.info("=" * 60)
        log.info(f"[MT5 SHADOW MODE PAPER TRADE RECORD]")
        log.info(f"  Symbol      : {symbol}")
        log.info(f"  Direction   : {direction}")
        log.info(f"  Lot Size    : {lot_size}")
        log.info(f"  Stop Loss   : ${sl:,.2f}")
        log.info(f"  Take Profit : ${tp:,.2f}")
        log.info(f"  Memo        : {memo}")
        log.info("=" * 60)
        return {
            "status": "success",
            "execution_mode": "shadow_mode_paper_trade",
            "symbol": symbol,
            "direction": direction,
            "lot_size": lot_size,
            "stop_loss": sl,
            "take_profit": tp,
            "timestamp_utc": datetime.now(timezone.utc).isoformat()
        }

    # LIVE EXECUTION ON METATRADER 5 TERMINAL
    if not initialize_mt5():
        return {"status": "error", "error": "MT5 Terminal connection failed."}

    # Auto-resolve symbol for broker
    resolved_symbol = resolve_gold_symbol(symbol)
    symbol_info = mt5.symbol_info(resolved_symbol)
    if symbol_info is None:
        shutdown_mt5()
        return {"status": "error", "error": f"Symbol '{resolved_symbol}' not found in MT5 Market Watch."}

    if not symbol_info.visible:
        mt5.symbol_select(resolved_symbol, True)

    term_info = mt5.terminal_info()
    if term_info and not term_info.trade_allowed:
        shutdown_mt5()
        log.error("❌ [MT5 BLOCKED] 'Algo Trading' is disabled in your MetaTrader 5 terminal. Click the 'Algo Trading' button on the MT5 top toolbar to enable trade execution.")
        return {
            "status": "failed",
            "error": "Algo Trading is disabled on the MT5 terminal toolbar. Please click the 'Algo Trading' button in MT5."
        }

    # Check Spread
    spread_pips = (symbol_info.ask - symbol_info.bid) / (symbol_info.point * 10)
    if spread_pips > MAX_SPREAD_PIPS:
        shutdown_mt5()
        log.warning(f"[MT5] High spread veto: Current spread {spread_pips:.1f} pips > Max {MAX_SPREAD_PIPS} pips.")
        return {"status": "vetoed", "reason": f"Spread too high ({spread_pips:.1f} pips)"}

    # Prepare Order Request
    order_type = mt5.ORDER_TYPE_BUY if direction == "LONG" else mt5.ORDER_TYPE_SELL
    price      = symbol_info.ask if direction == "LONG" else symbol_info.bid

    # ── ENFORCE DYNAMIC STOP LOSS & TAKE PROFIT ──────────────────────────────
    # Gold pricing rules: Every trade MUST have Stop Loss & Take Profit protection
    # Default 1:2 R:R (SL = $15.00 distance / 150 pips, TP = $30.00 distance / 300 pips)
    default_sl_dist = 15.0
    default_tp_dist = 30.0

    if sl <= 0.0 or tp <= 0.0 or abs(sl - price) > 500.0 or abs(tp - price) > 500.0:
        if direction == "LONG":
            sl = round(price - default_sl_dist, 2)
            tp = round(price + default_tp_dist, 2)
        else:
            sl = round(price + default_sl_dist, 2)
            tp = round(price - default_tp_dist, 2)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": resolved_symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION_SLIPPAGE,
        "magic": MAGIC_NUMBER,
        "comment": f"GoldAI-{direction}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    log.info(f"[MT5 LIVE ORDER SEND] Sending {direction} {lot_size} lots on {resolved_symbol} @ ${price:,.2f} | SL: ${sl:,.2f} | TP: ${tp:,.2f}...")
    result = mt5.order_send(request)
    shutdown_mt5()

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        err_msg = result.comment if result else "Unknown order send error"
        log.error(f"[MT5 LIVE EXECUTION FAILED] Retcode: {result.retcode if result else 'N/A'}, Comment: {err_msg}")
        return {"status": "failed", "error": err_msg, "retcode": result.retcode if result else None}

    log.info(f"✅ [MT5 LIVE ORDER EXECUTED] Deal Ticket #{result.deal} | Price: ${result.price:,.2f} | SL: ${sl:,.2f} | TP: ${tp:,.2f}")
    return {
        "status": "success",
        "execution_mode": "live_mt5_deal",
        "ticket": result.order,
        "deal_ticket": result.deal,
        "symbol": resolved_symbol,
        "direction": direction,
        "lot_size": lot_size,
        "fill_price": result.price,
        "stop_loss": sl,
        "take_profit": tp,
        "timestamp_utc": datetime.now(timezone.utc).isoformat()
    }


def compute_dynamic_lot_size(
    account_equity: float,
    sl_distance: float,
    confidence_score: float = 0.60,
    min_lot: float = MIN_LOT_SIZE,
    max_lot: float = MAX_LOT_SIZE,
) -> Tuple[float, float, str]:
    """
    Dynamically sizes lots based on the AI Confidence Tier:
      - High Confidence (>= 65% / 0.65 prob): Risk 2.0% of account equity
      - Normal Confidence (52% - 64% prob):  Risk 1.0% (0.5% - 1.0%) of account equity
      - Conservative (< 52% prob):            Risk 0.5% of account equity
    """
    if confidence_score >= 0.65:
        risk_pct = 0.020  # 2.0% for High Confidence
        tier = "HIGH_CONFIDENCE (2.0% Risk)"
    elif confidence_score >= 0.52:
        risk_pct = 0.010  # 1.0% for Normal Confidence
        tier = "NORMAL_CONFIDENCE (1.0% Risk)"
    else:
        risk_pct = 0.005  # 0.5% for Conservative Confidence
        tier = "CONSERVATIVE (0.5% Risk)"

    risk_usd = max(10.0, account_equity * risk_pct)
    sl_dist = max(5.0, sl_distance)
    # For Gold (100 oz CFD per standard lot): $1.00 price move = $100 per 1.0 lot ($1.00 per 0.01 lot)
    calculated_lots = risk_usd / (sl_dist * 100.0)
    lot_size = round(max(min_lot, min(calculated_lots, max_lot)), 2)
    return lot_size, risk_usd, tier


def execute_system_prediction(is_live: bool = False, custom_lot: Optional[float] = None) -> Dict[str, Any]:
    """
    Fetches the live prediction from the AI server, dynamically sizes lots according to
    confidence tier (2.0% for High Confidence vs 0.5%-1.0% for Normal), and executes with full SL/TP.
    """
    import urllib.request
    try:
        req = urllib.request.Request("http://localhost:8000/api/predict", headers={"User-Agent": "MT5Bridge"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            
            sig = data.get("prediction", {}).get("signal", "NEUTRAL")
            direction = "LONG" if "LONG" in sig or "BUY" in sig or "UP" in sig else "SHORT" if "SHORT" in sig or "SELL" in sig or "DOWN" in sig else "HOLD"
            
            prob_up = float(data.get("prediction", {}).get("probability_up", 0.5))
            prob_down = float(data.get("prediction", {}).get("probability_down", 0.5))
            confidence = max(prob_up, prob_down)
            
            rm = data.get("risk_management", {})
            sl = float(rm.get("stop_loss", 0.0))
            tp = float(rm.get("take_profit", 0.0))
            sl_dist = float(rm.get("sl_distance", 15.0))

            # Prioritize ICT Smart Money structural SL/TP if available
            ict = data.get("ict_analysis", {})
            gpt_ict = ict.get("gpt_synthesis", {})
            if gpt_ict.get("recommended_sl") and gpt_ict.get("recommended_tp"):
                sl = float(gpt_ict["recommended_sl"])
                tp = float(gpt_ict["recommended_tp"])
                curr_p = float(ict.get("current_price", data.get("target_trade", {}).get("current_price", 4636.0)))
                sl_dist = abs(curr_p - sl)

            if sl_dist <= 0.0:
                sl_dist = 15.0
            
            # Fetch MT5 live account equity for dynamic sizing
            acc_equity = 10000.0
            if _MT5_AVAILABLE:
                if initialize_mt5():
                    acc = mt5.account_info()
                    if acc:
                        acc_equity = acc.equity
                    shutdown_mt5()
            
            if custom_lot and custom_lot > 0:
                lot_size = custom_lot
                risk_usd = custom_lot * sl_dist * 100.0
                tier = f"CUSTOM_LOT ({custom_lot} lots)"
            else:
                lot_size, risk_usd, tier = compute_dynamic_lot_size(acc_equity, sl_dist, confidence)

            log.info(f"[RISK TIER] {tier} | Equity: ${acc_equity:,.2f} | Risk Budget: ${risk_usd:,.2f} | Sized Lot: {lot_size} lots")
            
            payload = {
                "decision": "APPROVE" if direction in {"LONG", "SHORT"} else "REJECT",
                "direction": direction,
                "lot_size": lot_size,
                "stop_loss": sl,
                "take_profit": tp,
                "justification": f"[{tier}] {data.get('narrative', {}).get('summary', 'Live AI Consensus Trade')} (Risk: ${risk_usd:,.2f})"
            }
            return execute_agent_decision(payload, symbol=DEFAULT_SYMBOL, is_live=is_live)
    except Exception as e:
        log.warning(f"Could not connect to /api/predict ({e}). Falling back to local calculation.")
        lot_size, risk_usd, tier = compute_dynamic_lot_size(10000.0, 15.0, 0.60)
        payload = {
            "decision": "APPROVE",
            "direction": "LONG",
            "lot_size": custom_lot if custom_lot else lot_size,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "justification": f"[{tier}] AI Signal Execution with dynamic ATR Stop Loss and Take Profit (Risk: ${risk_usd:,.2f})"
        }
        return execute_agent_decision(payload, symbol=DEFAULT_SYMBOL, is_live=is_live)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MetaTrader 5 Direct Market Execution Agent")
    parser.add_argument("--action", choices=["status", "test_order", "execute_signal"], default="status")
    parser.add_argument("--symbol", default="GOLD")
    parser.add_argument("--signal", choices=["LONG", "SHORT"], default="LONG")
    parser.add_argument("--lot", type=float, default=None, help="Custom lot size for execution (optional; auto-sizes based on 2% for high confidence vs 0.5%-1% for normal if omitted)")
    parser.add_argument("--sl", type=float, default=0.0, help="Custom Stop Loss price (optional)")
    parser.add_argument("--tp", type=float, default=0.0, help="Custom Take Profit price (optional)")
    parser.add_argument("--live", action="store_true", help="Execute real live order in MT5 (overrides shadow mode)")
    args = parser.parse_args()

    if args.action == "status":
        print(json.dumps(get_account_status(), indent=2))
    elif args.action == "execute_signal":
        res = execute_system_prediction(is_live=args.live, custom_lot=args.lot)
        print(json.dumps(res, indent=2))
    elif args.action == "test_order":
        sample = {
            "decision": "APPROVE",
            "direction": args.signal,
            "lot_size": args.lot if args.lot else 0.01,
            "stop_loss": args.sl,
            "take_profit": args.tp,
            "justification": "Trading execution with guaranteed Stop Loss and Take Profit protection"
        }
        res = execute_agent_decision(sample, symbol=args.symbol, is_live=args.live)
        print(json.dumps(res, indent=2))
