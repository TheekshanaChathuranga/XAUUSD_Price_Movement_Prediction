from sqlalchemy import Column, Integer, String, Float, Date, DateTime
from database import Base
import datetime

class SignalHistory(Base):
    __tablename__ = "signal_history"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, index=True)
    signal = Column(String)           # 'LONG', 'SHORT', 'NEUTRAL'
    confidence = Column(Float)        # e.g. 0.65
    price_at_signal = Column(Float)   # Close price when signal was generated
    
    # Outcome tracking
    outcome = Column(String, default="PENDING")  # 'WIN', 'LOSS', 'PENDING', 'FLAT'
    price_next_day = Column(Float, nullable=True) # Next day's close
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
