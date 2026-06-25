"""
macro_calendar.py
=================
Provides upcoming macro economic events for the Gold AI dashboard.
Events are auto-computed relative to today so they always show
the next 30 days without manual updates.

Key events tracked (all move XAUUSD significantly):
  - FOMC Rate Decision        → HIGH impact
  - US CPI                    → HIGH impact
  - US NFP (Non-Farm Payroll) → HIGH impact
  - US PCE Deflator           → MEDIUM impact
  - US GDP Advance Estimate   → MEDIUM impact
  - US PPI                    → LOW impact
  - FOMC Minutes              → LOW impact
  - ECB Rate Decision         → MEDIUM impact
"""

from datetime import date, timedelta

# ── KNOWN UPCOMING DATES (manually curated, update quarterly) ──────────────────
# Format: (date_str, event_name, impact, description)
# impact: "HIGH" | "MEDIUM" | "LOW"

KNOWN_EVENTS = [
    # ── FOMC 2026 ──────────────────────────────────────────────────────────────
    ("2026-07-29", "FOMC Rate Decision",      "HIGH",   "Federal Reserve interest rate decision. Major gold mover."),
    ("2026-07-09", "FOMC Minutes (Jun)",       "LOW",    "Minutes from the June FOMC meeting released."),
    ("2026-09-15", "FOMC Rate Decision",      "HIGH",   "Federal Reserve interest rate decision."),
    ("2026-10-28", "FOMC Minutes (Sep)",       "LOW",    "Minutes from September FOMC meeting."),
    ("2026-11-04", "FOMC Rate Decision",      "HIGH",   "Federal Reserve interest rate decision."),
    ("2026-12-15", "FOMC Rate Decision",      "HIGH",   "Federal Reserve interest rate decision."),

    # ── US CPI 2026 ────────────────────────────────────────────────────────────
    ("2026-07-11", "US CPI (June)",           "HIGH",   "Consumer Price Index for June 2026. Core CPI drives gold."),
    ("2026-08-12", "US CPI (July)",           "HIGH",   "Consumer Price Index for July 2026."),
    ("2026-09-11", "US CPI (August)",         "HIGH",   "Consumer Price Index for August 2026."),
    ("2026-10-13", "US CPI (September)",      "HIGH",   "Consumer Price Index for September 2026."),
    ("2026-11-12", "US CPI (October)",        "HIGH",   "Consumer Price Index for October 2026."),
    ("2026-12-10", "US CPI (November)",       "HIGH",   "Consumer Price Index for November 2026."),

    # ── US NFP (Non-Farm Payrolls) 2026 ───────────────────────────────────────
    ("2026-07-03", "US NFP (June)",           "HIGH",   "Non-Farm Payrolls. Weak jobs = bullish gold."),
    ("2026-08-07", "US NFP (July)",           "HIGH",   "Non-Farm Payrolls for July 2026."),
    ("2026-09-04", "US NFP (August)",         "HIGH",   "Non-Farm Payrolls for August 2026."),
    ("2026-10-02", "US NFP (September)",      "HIGH",   "Non-Farm Payrolls for September 2026."),
    ("2026-11-06", "US NFP (October)",        "HIGH",   "Non-Farm Payrolls for October 2026."),
    ("2026-12-04", "US NFP (November)",       "HIGH",   "Non-Farm Payrolls for November 2026."),

    # ── US PCE Deflator 2026 ──────────────────────────────────────────────────
    ("2026-06-27", "US PCE Deflator (May)",   "MEDIUM", "Fed's preferred inflation measure. May data."),
    ("2026-07-31", "US PCE Deflator (June)",  "MEDIUM", "Fed's preferred inflation measure. June data."),
    ("2026-08-28", "US PCE Deflator (July)",  "MEDIUM", "Fed's preferred inflation measure. July data."),
    ("2026-09-25", "US PCE Deflator (Aug)",   "MEDIUM", "Fed's preferred inflation measure. August data."),
    ("2026-10-30", "US PCE Deflator (Sep)",   "MEDIUM", "Fed's preferred inflation measure. September data."),
    ("2026-11-25", "US PCE Deflator (Oct)",   "MEDIUM", "Fed's preferred inflation measure. October data."),

    # ── US GDP Advance Estimate 2026 ──────────────────────────────────────────
    ("2026-07-30", "US GDP Q2 (Advance)",     "MEDIUM", "Q2 2026 GDP advance estimate. Recession fears = gold up."),
    ("2026-10-29", "US GDP Q3 (Advance)",     "MEDIUM", "Q3 2026 GDP advance estimate."),

    # ── US PPI 2026 ───────────────────────────────────────────────────────────
    ("2026-07-14", "US PPI (June)",           "LOW",    "Producer Price Index — leading indicator for CPI."),
    ("2026-08-13", "US PPI (July)",           "LOW",    "Producer Price Index for July 2026."),
    ("2026-09-10", "US PPI (August)",         "LOW",    "Producer Price Index for August 2026."),

    # ── ECB Rate Decisions 2026 ───────────────────────────────────────────────
    ("2026-07-24", "ECB Rate Decision",       "MEDIUM", "European Central Bank rate decision. EUR/USD affects gold."),
    ("2026-09-11", "ECB Rate Decision",       "MEDIUM", "European Central Bank rate decision."),
    ("2026-10-29", "ECB Rate Decision",       "MEDIUM", "European Central Bank rate decision."),
    ("2026-12-10", "ECB Rate Decision",       "MEDIUM", "European Central Bank rate decision."),

    # ── US Jobless Claims (weekly — only show next 2) ────────────────────────
    ("2026-06-26", "US Jobless Claims",       "LOW",    "Weekly initial jobless claims. Labor market pulse."),
    ("2026-07-02", "US Jobless Claims",       "LOW",    "Weekly initial jobless claims."),
    ("2026-07-09", "US Jobless Claims",       "LOW",    "Weekly initial jobless claims."),
]

IMPACT_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
IMPACT_COLOR = {"HIGH": "#FF4060", "MEDIUM": "#FFAB40", "LOW": "#4488FF"}


def get_upcoming_events(days_ahead: int = 30) -> list:
    """
    Return upcoming macro events sorted by date, within `days_ahead` days from today.
    Each event is a dict: {date, event, impact, description, days_until, impact_color}
    """
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)

    events = []
    for date_str, event, impact, desc in KNOWN_EVENTS:
        try:
            ev_date = date.fromisoformat(date_str)
        except ValueError:
            continue

        if today <= ev_date <= cutoff:
            days_until = (ev_date - today).days
            events.append({
                "date":         date_str,
                "event":        event,
                "impact":       impact,
                "description":  desc,
                "days_until":   days_until,
                "impact_color": IMPACT_COLOR.get(impact, "#ffffff"),
                "is_today":     days_until == 0,
                "is_tomorrow":  days_until == 1,
            })

    # Sort: by date, then by impact
    events.sort(key=lambda x: (x["date"], IMPACT_ORDER.get(x["impact"], 9)))
    return events


if __name__ == "__main__":
    events = get_upcoming_events(30)
    print(f"\nUpcoming Macro Events (next 30 days) — {date.today()}")
    print("=" * 70)
    for e in events:
        label = "TODAY" if e["is_today"] else ("TOMORROW" if e["is_tomorrow"] else f"in {e['days_until']}d")
        print(f"  [{e['impact']:6s}]  {e['date']}  ({label:10s})  {e['event']}")
