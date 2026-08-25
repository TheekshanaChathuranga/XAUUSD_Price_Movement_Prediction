import datetime
import sqlite3
import json
from sqlalchemy.orm import Session
import pandas as pd
import models
from database import SessionLocal, engine

# Create tables
models.Base.metadata.create_all(bind=engine)

def save_daily_signal(date_str: str, signal: str, confidence: float, price: float):
    """Save today's signal to the database if it doesn't exist."""
    db: Session = SessionLocal()
    try:
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        existing = db.query(models.SignalHistory).filter(models.SignalHistory.date == dt).first()
        if not existing:
            new_signal = models.SignalHistory(
                date=dt,
                signal=signal,
                confidence=confidence,
                price_at_signal=price
            )
            db.add(new_signal)
            db.commit()
    except Exception as e:
        print(f"Error saving signal: {e}")
    finally:
        db.close()

def resolve_outcomes(prices_df: pd.DataFrame):
    """
    Check PENDING signals and resolve them to WIN/LOSS based on next day's close.
    Also automatically resolves agent session outcomes and triggers self-reflection.
    """
    db: Session = SessionLocal()
    try:
        # 1. Resolve pure ML signals
        pending_signals = db.query(models.SignalHistory).filter(models.SignalHistory.outcome == "PENDING").all()
        for sig in pending_signals:
            future_prices = prices_df[prices_df['Date'] >= pd.to_datetime(sig.date)]
            if len(future_prices) > 0:
                next_day = future_prices.iloc[0]
                next_price = next_day['Close']
                sig.price_next_day = next_price
                
                if sig.signal == "LONG":
                    sig.outcome = "WIN" if next_price > sig.price_at_signal else "LOSS"
                elif sig.signal == "SHORT":
                    sig.outcome = "WIN" if next_price < sig.price_at_signal else "LOSS"
                else:
                    sig.outcome = "FLAT"
                    
                db.commit()
                
        # 2. Resolve agent session outcomes
        resolve_agent_outcomes(prices_df)
        
    except Exception as e:
        print(f"Error resolving outcomes: {e}")
    finally:
        db.close()

def resolve_agent_outcomes(prices_df: pd.DataFrame):
    """
    Check agent shadow decisions in agents_audit.db, resolve outcomes,
    and trigger self-reflection on losing trades.
    """
    from agents.config import cfg
    db_path = cfg.audit_db_path
    if not db_path.exists():
        return

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        # Select approved sessions that don't have outcome resolved yet in their full_log_json
        sessions = conn.execute("""
            SELECT session_id, timestamp_utc, final_direction, quant_signal_in, full_log_json
            FROM agent_sessions
            WHERE final_decision IN ('APPROVE', 'RESIZE')
              AND final_direction IN ('LONG', 'SHORT')
        """).fetchall()
        
        for s in sessions:
            session_id = s['session_id']
            ts_str = s['timestamp_utc']
            final_direction = s['final_direction']
            quant_signal_in = s['quant_signal_in']
            full_log_json = s['full_log_json']
            
            # Check if outcome is already resolved in JSON
            log_data = json.loads(full_log_json)
            if log_data.get("actual_next_day_direction") is not None:
                continue # Already resolved
                
            # Parse timestamp to date
            date_str = ts_str.split("T")[0]
            session_date = pd.to_datetime(date_str)
            
            # Find the close price on the session date
            session_day_rows = prices_df[prices_df['Date'] == session_date]
            if len(session_day_rows) == 0:
                session_day_rows = prices_df[prices_df['Date'] >= session_date]
            if len(session_day_rows) == 0:
                continue
                
            entry_row = session_day_rows.iloc[0]
            entry_price = entry_row['Close']
            
            # Find next trading day's price
            future_prices = prices_df[prices_df['Date'] > entry_row['Date']]
            if len(future_prices) > 0:
                next_row = future_prices.iloc[0]
                next_price = next_row['Close']
                
                # Determine actual direction
                actual_dir = "BULLISH" if next_price > entry_price else "BEARISH" if next_price < entry_price else "NEUTRAL"
                
                # Calculate correctness
                quant_correct = (quant_signal_in == "LONG" and next_price > entry_price) or (quant_signal_in == "SHORT" and next_price < entry_price)
                agent_correct = (final_direction == "LONG" and next_price > entry_price) or (final_direction == "SHORT" and next_price < entry_price)
                
                # Update DB via audit logger
                from agents.audit_logger import update_session_outcome
                update_session_outcome(session_id, actual_dir, quant_correct, agent_correct)
                
                # If agent lost the trade, perform LLM self-reflection
                if not agent_correct:
                    loss_bps = abs(entry_price - next_price) / entry_price * 10000
                    actual_pct = (next_price - entry_price) / entry_price * 100
                    actual_move = f"{actual_dir} ({actual_pct:+.2f}%)"
                    
                    print(f"[reflection] Session {session_id} resulted in a LOSS ({loss_bps:.1f}bps). Running diagnosis...", flush=True)
                    from agents.reflection_agent import diagnose_losing_trade
                    try:
                        diagnose_losing_trade(
                            session_id=session_id,
                            direction=final_direction,
                            loss_bps=loss_bps,
                            actual_market_move=actual_move,
                            use_llm=True
                        )
                    except Exception as re:
                        print(f"[reflection] Diagnosis failed: {re}", flush=True)
                        
        conn.close()
    except Exception as e:
        print(f"Error resolving agent outcomes: {e}", flush=True)

def get_signal_history():
    """Return all signals for the API."""
    db: Session = SessionLocal()
    try:
        signals = db.query(models.SignalHistory).order_by(models.SignalHistory.date.desc()).all()
        return [
            {
                "date": s.date.isoformat(),
                "signal": s.signal,
                "confidence": s.confidence,
                "price_at_signal": s.price_at_signal,
                "outcome": s.outcome,
                "price_next_day": s.price_next_day
            }
            for s in signals
        ]
    finally:
        db.close()
