import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    from macro_calendar import get_upcoming_events
    events = get_upcoming_events(30)
    print(f"[OK] macro_calendar — {len(events)} events in next 30 days")
    for e in events[:3]:
        label = e["event"]
        days = e["days_until"]
        impact = e["impact"]
        print(f"     {impact:6} | {e['date']} (in {days}d) | {label}")
except Exception as ex:
    print(f"[FAIL] macro_calendar: {ex}")

print()

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    v = SentimentIntensityAnalyzer()
    score = v.polarity_scores("Gold surges as Fed hints at rate cut")
    print(f"[OK] VADER — compound={score['compound']}")
except Exception as ex:
    print(f"[FAIL] VADER: {ex}")

print()

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    print("[OK] APScheduler")
except Exception as ex:
    print(f"[FAIL] APScheduler: {ex}")

print()

try:
    import catboost, xgboost, lightgbm, shap, joblib
    print("[OK] ML libs — catboost, xgboost, lightgbm, shap, joblib")
except Exception as ex:
    print(f"[FAIL] ML libs: {ex}")

print()
print("All checks complete.")
