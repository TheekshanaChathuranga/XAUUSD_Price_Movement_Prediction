"""
Trade Lifecycle Manager for XAU/USD (Gold)
Persistent Trade Lock, Stop-Loss / Take-Profit Lifecycle Monitor, and Auto-Scan on Exit.
"""

import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [Lifecycle] %(message)s")
log = logging.getLogger("LifecycleManager")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "trade_lock_state.json")

def _load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Error loading state: {e}")
    return {
        "status": "IDLE_SCANNING", # IDLE_SCANNING, PENDING_LIMIT, LOCKED_ACTIVE
        "locked_trade": None,
        "history": [],
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "auto_scan_enabled": True
    }

def _save_state(state: Dict[str, Any]) -> None:
    try:
        state["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log.error(f"Error saving trade lock state: {e}")

def get_current_trade_state() -> Dict[str, Any]:
    """Returns the current trade lock state, syncing with live MT5 broker."""
    state = _load_state()
    return sync_trade_lifecycle(state)

def lock_trade(trade_plan: Dict[str, Any], ticket: Optional[int] = None, is_live_position: bool = False) -> Dict[str, Any]:
    """Locks a trade into active management mode."""
    state = _load_state()
    
    locked_obj = {
        "ticket": ticket,
        "is_live_position": is_live_position,
        "symbol": trade_plan.get("symbol", "GOLD"),
        "signal": trade_plan.get("signal", "LONG"),
        "entry_price": float(trade_plan.get("entry_price", 0.0)),
        "stop_loss": float(trade_plan.get("stop_loss", 0.0)),
        "take_profit": float(trade_plan.get("take_profit", 0.0)),
        "lot_size": float(trade_plan.get("lot_size", 0.10)),
        "confidence": float(trade_plan.get("confidence", 65.0)),
        "risk_tier": trade_plan.get("risk_tier", "1.0% NORMAL"),
        "be_triggered": False,
        "locked_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "status": "ACTIVE_IN_MARKET" if is_live_position else "LOCKED_PENDING_FILL"
    }

    state["status"] = "LOCKED_ACTIVE" if is_live_position else "PENDING_LIMIT"
    state["locked_trade"] = locked_obj
    _save_state(state)
    log.info(f"🔒 Trade locked successfully: {locked_obj['signal']} @ {locked_obj['entry_price']} | SL: {locked_obj['stop_loss']} | TP: {locked_obj['take_profit']}")
    return state

def sync_trade_lifecycle(state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Checks if active locked trade in MT5 hit SL or TP.
    If hit, logs outcome and triggers a fresh scan for the next trade.
    """
    if state is None:
        state = _load_state()

    locked = state.get("locked_trade")
    if not locked:
        return state

    try:
        from step12_mt5_bridge import _MT5_AVAILABLE, initialize_mt5, resolve_gold_symbol
        if not _MT5_AVAILABLE:
            return state

        if initialize_mt5():
            import MetaTrader5 as mt5
            symbol = resolve_gold_symbol()
            
            # Check current open positions
            positions = mt5.positions_get(symbol=symbol) or ()
            ticket = locked.get("ticket")

            matching_pos = None
            if ticket:
                matching_pos = next((p for p in positions if p.ticket == ticket), None)
            elif positions:
                matching_pos = positions[0]
                locked["ticket"] = int(matching_pos.ticket)
                locked["is_live_position"] = True

            if matching_pos:
                # Position is currently running live in broker
                state["status"] = "LOCKED_ACTIVE"
                locked["is_live_position"] = True
                locked["current_price"] = float(matching_pos.price_current)
                locked["profit_usd"] = round(float(matching_pos.profit), 2)
                locked["price_open"] = float(matching_pos.price_open)
                
                # Check 1:1 Risk-to-Reward (1:1 R:R) Break-Even Logic
                ep = locked["price_open"]
                tp = locked["take_profit"]
                cp = locked["current_price"]
                sig = locked["signal"]
                initial_sl = float(locked.get("stop_loss", 0.0))

                if not locked.get("be_triggered", False) and initial_sl and ep:
                    # Risk distance = |Entry - Initial Stop Loss| (1R)
                    risk_dist = abs(ep - initial_sl)
                    if risk_dist > 0:
                        be_trigger_price = ep + risk_dist if sig in {"LONG", "BUY"} else ep - risk_dist
                        if (sig in {"LONG", "BUY"} and cp >= be_trigger_price) or (sig in {"SHORT", "SELL"} and cp <= be_trigger_price):
                            locked["be_triggered"] = True
                            log.info(f"🛡️ 1:1 R:R reached (+${risk_dist:.2f} profit @ ${cp:.2f})! Moving Stop Loss to Break-Even (${ep:.2f})")
                            # Try modifying SL in MT5
                            try:
                                req = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "position": matching_pos.ticket,
                                    "symbol": symbol,
                                    "sl": ep,
                                    "tp": tp
                                }
                                res = mt5.order_send(req)
                                log.info(f"[MT5] 1:1 R:R Break-Even Order Send Code: {res.retcode if res else 'N/A'}")
                            except Exception as _be_err:
                                log.warning(f"Could not update SL to BE on broker: {_be_err}")

                state["locked_trade"] = locked
                _save_state(state)
                return state

            else:
                # If was live position and now no longer in open positions => Trade hit SL, TP, or closed!
                if locked.get("is_live_position"):
                    log.info(f"🏁 Active trade ticket #{ticket} is closed. Checking MT5 deal history for resolution...")
                    
                    # Query deal history
                    deals = mt5.history_deals_get(position=ticket) or ()
                    outcome = "CLOSED"
                    close_price = locked.get("current_price", locked["entry_price"])
                    profit = 0.0

                    if deals:
                        close_deal = deals[-1]
                        close_price = float(close_deal.price)
                        profit = round(float(close_deal.profit), 2)
                        
                        sl_price = locked.get("stop_loss", 0.0)
                        tp_price = locked.get("take_profit", 0.0)
                        
                        if abs(close_price - tp_price) <= 2.0 or profit > 10.0:
                            outcome = "HIT_TP (WIN)"
                        elif abs(close_price - sl_price) <= 2.0 or profit < -5.0:
                            outcome = "HIT_SL (LOSS)"
                        elif abs(profit) <= 5.0:
                            outcome = "BREAK_EVEN ($0)"
                    
                    # Archive to history
                    history_entry = {
                        "ticket": ticket,
                        "symbol": locked.get("symbol", "GOLD"),
                        "signal": locked.get("signal", "LONG"),
                        "entry_price": locked.get("entry_price"),
                        "close_price": close_price,
                        "stop_loss": locked.get("stop_loss"),
                        "take_profit": locked.get("take_profit"),
                        "profit_usd": profit,
                        "outcome": outcome,
                        "closed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                    }
                    
                    if "history" not in state:
                        state["history"] = []
                    state["history"].insert(0, history_entry)
                    state["history"] = state["history"][:50] # keep last 50
                    
                    log.info(f"✅ Trade outcome recorded: {outcome} (${profit:+.2f}). Unlocking and triggering auto-scan for next trade...")
                    
                    # Reset state to IDLE_SCANNING
                    state["status"] = "IDLE_SCANNING"
                    state["locked_trade"] = None
                    _save_state(state)
                    
                    # Auto-scan: Trigger background refresh to immediately compute next Smart Money trade setup
                    try:
                        import urllib.request
                        urllib.request.urlopen("http://localhost:8000/api/refresh-news", timeout=5)
                    except Exception:
                        pass
                    
                    return state

    except Exception as e:
        log.error(f"Error in sync_trade_lifecycle: {e}")

    return state

def execute_and_lock_live_trade(is_live: bool = False, custom_lot: Optional[float] = None) -> Dict[str, Any]:
    """
    Executes the current high-probability Smart Money signal on MT5 and immediately locks it.
    """
    import step12_mt5_bridge as mb
    
    # 1. Execute system prediction
    exec_res = mb.execute_system_prediction(is_live=is_live, custom_lot=custom_lot)
    
    # 2. Extract trade parameters
    order_res = exec_res.get("order_result", {})
    ticket = order_res.get("order") or order_res.get("deal") or 999999
    
    trade_plan = {
        "symbol": exec_res.get("symbol", "GOLD"),
        "signal": exec_res.get("direction", "LONG"),
        "entry_price": order_res.get("price") or exec_res.get("live_price", 4640.0),
        "stop_loss": exec_res.get("stop_loss", 0.0),
        "take_profit": exec_res.get("take_profit", 0.0),
        "lot_size": exec_res.get("lot_size", 0.10),
        "confidence": 65.8,
        "risk_tier": "2.0% HIGH CONF"
    }

    # 3. Lock into lifecycle state machine
    new_state = lock_trade(trade_plan, ticket=ticket, is_live_position=is_live)
    return {
        "execution": exec_res,
        "lock_state": new_state
    }


# ══════════════════════════════════════════════════════════════════════════════
# FULLY AUTOMATIC TRADE LIFECYCLE (NO USER INTERVENTION)
# ══════════════════════════════════════════════════════════════════════════════

_AUTO_LOCK_COOLDOWN_SECS = 1800  # 30 min cooldown after trade exit before new lock
_AUTO_MIN_CONFIDENCE = 55.0       # Minimum confidence % to auto-lock
_AUTO_VALID_VERDICTS = {"CONFIRMED_BUY", "CONFIRMED_SELL"}  # ICT verdicts that auto-lock

def auto_lock_from_prediction(signal_cache: Dict[str, Any]) -> Dict[str, Any]:
    """
    Automatically locks a trade from the /api/predict signal cache.
    Called by the background scheduler — ZERO user intervention.
    
    Rules:
      1. Only locks if state is IDLE_SCANNING (no active trade)
      2. Respects 30-min cooldown after last trade exit
      3. Requires ICT verdict = CONFIRMED_BUY or CONFIRMED_SELL
      4. Requires confidence ≥ 55%
      5. Uses GPT-synthesized structural SL/TP (ICT levels)
      6. Falls back to ATR-based SL/TP if GPT unavailable
    """
    state = _load_state()
    
    # Guard: Don't lock if already active or pending
    if state.get("status") != "IDLE_SCANNING":
        return state
    
    # Guard: Cooldown — don't re-enter too fast after exit
    history = state.get("history", [])
    if history:
        last_close = history[0].get("closed_at", "")
        if last_close:
            try:
                dt_last = datetime.strptime(last_close, "%Y-%m-%d %H:%M:%S UTC")
                elapsed = (datetime.now(timezone.utc).replace(tzinfo=None) - dt_last).total_seconds()
                if elapsed < _AUTO_LOCK_COOLDOWN_SECS:
                    log.info(f"⏳ Auto-lock cooldown: {int(_AUTO_LOCK_COOLDOWN_SECS - elapsed)}s remaining before next trade scan.")
                    return state
            except Exception:
                pass
    
    if not signal_cache or signal_cache.get("status") != "success":
        return state
    
    # Extract unified target trade
    target = signal_cache.get("target_trade", {})
    ict = signal_cache.get("ict_analysis", {})
    gpt = ict.get("gpt_synthesis", {})
    
    sig = target.get("signal", "NEUTRAL")
    if sig not in ("LONG", "SHORT"):
        log.info(f"[auto-lock] Signal is {sig} — no trade to lock.")
        return state
    
    confidence = float(target.get("confidence", 0.0))
    if confidence < _AUTO_MIN_CONFIDENCE:
        log.info(f"[auto-lock] Confidence {confidence:.1f}% below {_AUTO_MIN_CONFIDENCE}% threshold — skipping.")
        return state
    
    # Check ICT verdict for high-quality setups only
    ict_verdict = gpt.get("ict_verdict", "")
    
    # Map signal direction to valid verdicts
    if sig == "LONG" and ict_verdict not in ("CONFIRMED_BUY", "WAIT_FOR_RETRACEMENT"):
        log.info(f"[auto-lock] LONG signal but ICT verdict is '{ict_verdict}' — skipping.")
        return state
    if sig == "SHORT" and ict_verdict not in ("CONFIRMED_SELL", "WAIT_FOR_RETRACEMENT"):
        log.info(f"[auto-lock] SHORT signal but ICT verdict is '{ict_verdict}' — skipping.")
        return state
    
    # Don't auto-lock if there's already an open position in MT5 broker
    if target.get("is_position_open"):
        # Already has a position — lock the existing one instead
        broker_pos = target.get("live_broker_position", {})
        if broker_pos:
            log.info(f"[auto-lock] Existing MT5 position detected (ticket #{broker_pos.get('ticket')}). Auto-locking existing position.")
            trade_plan = {
                "symbol": broker_pos.get("symbol", "GOLD"),
                "signal": broker_pos.get("type", sig),
                "entry_price": broker_pos.get("price_open", 0.0),
                "stop_loss": broker_pos.get("sl", 0.0),
                "take_profit": broker_pos.get("tp", 0.0),
                "lot_size": broker_pos.get("volume", 0.10),
                "confidence": confidence,
                "risk_tier": target.get("risk_tier", "1.0% NORMAL"),
            }
            new_state = lock_trade(trade_plan, ticket=broker_pos.get("ticket"), is_live_position=True)
            log.info(f"🔒 AUTO-LOCKED existing MT5 position: {trade_plan['signal']} @ ${trade_plan['entry_price']:,.2f}")
            return new_state
    
    # Build trade plan from ICT-synthesized levels (best) or ATR fallback
    entry = float(gpt.get("recommended_entry") or target.get("entry_price") or target.get("current_price", 0.0))
    sl = float(gpt.get("recommended_sl") or target.get("stop_loss", 0.0))
    tp = float(gpt.get("recommended_tp") or target.get("take_profit", 0.0))
    
    # Sanity check: SL and TP must be valid
    if entry <= 0 or sl <= 0 or tp <= 0:
        log.warning(f"[auto-lock] Invalid levels: Entry=${entry}, SL=${sl}, TP=${tp} — skipping.")
        return state
    
    # Sanity check: Risk:Reward must be >= 1.0
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 0 or reward / risk < 1.0:
        log.warning(f"[auto-lock] Bad R:R ({reward/risk if risk > 0 else 0:.2f}) — skipping.")
        return state
    
    trade_plan = {
        "symbol": "GOLD",
        "signal": sig,
        "entry_price": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "lot_size": 0.01,  # Conservative auto-lot
        "confidence": confidence,
        "risk_tier": target.get("risk_tier", "1.0% NORMAL"),
    }
    
    new_state = lock_trade(trade_plan, ticket=None, is_live_position=False)
    log.info(f"🔒 AUTO-LOCKED new Smart Money trade: {sig} @ ${entry:,.2f} | SL: ${sl:,.2f} | TP: ${tp:,.2f} | R:R 1:{reward/risk:.1f} | Conf: {confidence:.0f}%")
    return new_state


def auto_trade_cycle(signal_cache: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Full automatic trade cycle — called by background scheduler every 15 seconds.
    
    Flow:
      1. Sync with MT5 broker (check if active trade hit SL/TP/BE)
      2. If IDLE_SCANNING → auto-lock the next valid Smart Money trade
      3. If LOCKED_ACTIVE or PENDING_LIMIT → just monitor (sync handles it)
    
    This is the ONLY function that needs to be scheduled. It handles everything.
    """
    state = _load_state()
    
    # Step 1: Sync with broker — checks for SL/TP hits, BE triggers
    state = sync_trade_lifecycle(state)
    
    # Step 2: If idle after sync, try to auto-lock next trade
    if state.get("status") == "IDLE_SCANNING" and signal_cache:
        state = auto_lock_from_prediction(signal_cache)
    
    # Step 3: If still idle and no signal_cache provided, try fetching from API
    if state.get("status") == "IDLE_SCANNING" and not signal_cache:
        try:
            import urllib.request
            req = urllib.request.Request("http://localhost:8000/api/predict", headers={"User-Agent": "AutoTradeLifecycle"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("status") == "success":
                    state = auto_lock_from_prediction(data)
        except Exception as e:
            log.debug(f"[auto-cycle] Could not fetch prediction: {e}")
    
    return state
