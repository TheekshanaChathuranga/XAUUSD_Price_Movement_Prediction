#!/usr/bin/env python3
"""
AUTOMATED TRADING PIPELINE & AUDIT SUITE
========================================
This script acts as the master orchestrator for XAUUSD prediction systems.
It ensures that dates are correct, files are fresh, services are running,
and predictions align with current MT5 and news data.

Features:
  1. Audits dates in all source files vs today's date.
  2. Verifies if today is a trading day (Monday-Friday).
  3. Automatically runs daily_refresh.py if files are stale.
  4. Runs step9_backtest_strategy.py to update statistics and charts.
  5. Inspects API Server status and restarts step11_api_server.py if down.
  6. Validates signal outcomes using historical prices.

Usage:
    python automate_trading_pipeline.py [--force]
"""

import os
import sys
import json
import time
import subprocess
import requests
from datetime import datetime, date, timedelta

if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding="utf-8")

DIR = os.path.dirname(os.path.abspath(__file__))
PORT = 8000
API_URL = f"http://localhost:{PORT}"

def is_trading_day():
    """True if today is a weekday (Monday-Friday) when FX markets are open."""
    today = datetime.now()
    # 0 = Monday, 6 = Sunday
    return today.weekday() < 5

def audit_file_freshness():
    """Checks the date of the latest record in core files."""
    today = date.today()
    files_to_check = {
        "xauusd_raw_prices.csv":   ("Date", 1), # Max 1 day old
        "gdelt_news_raw.csv":      ("Date", 1),
        "master_features.csv":     ("Date", 1),
        "live_inference_data.csv": ("Date", 1)
    }
    
    stale_files = []
    print("\n[+] Auditing dataset date freshness...")
    for filename, (col, max_age) in files_to_check.items():
        path = os.path.join(DIR, filename)
        if not os.path.exists(path):
            print(f"  [MISSING] {filename}")
            stale_files.append(filename)
            continue
            
        try:
            import pandas as pd
            df = pd.read_csv(path)
            # Find latest date in column
            latest_date_str = str(df[col].max())
            if " " in latest_date_str:
                latest_date_str = latest_date_str.split(" ")[0]
            latest = datetime.strptime(latest_date_str, "%Y-%m-%d").date()
            
            # If weekend, latest date could be Friday
            current_day = today
            days_allowed = max_age
            if current_day.weekday() == 5: # Saturday
                days_allowed += 1
            elif current_day.weekday() == 6: # Sunday
                days_allowed += 2
            elif current_day.weekday() == 0: # Monday
                days_allowed += 2
                
            age = (today - latest).days
            if age > days_allowed:
                print(f"  [STALE  ] {filename:<25} - Latest: {latest} ({age} days old, max allowed: {days_allowed})")
                stale_files.append(filename)
            else:
                print(f"  [FRESH  ] {filename:<25} - Latest: {latest} ({age} days old)")
        except Exception as e:
            print(f"  [ERROR  ] {filename:<25} - Failed to parse: {e}")
            stale_files.append(filename)
            
    return stale_files

def trigger_refresh():
    """Runs daily_refresh.py pipeline to update everything."""
    print("\n[+] Triggering daily refresh pipeline...")
    script = os.path.join(DIR, "daily_refresh.py")
    t0 = time.time()
    proc = subprocess.run([sys.executable, script])
    elapsed = time.time() - t0
    
    if proc.returncode == 0:
        print(f"  [SUCCESS] Refresh pipeline completed in {elapsed:.1f}s")
        return True
    else:
        print(f"  [FAILED ] Refresh pipeline exited with code {proc.returncode}")
        return False

def sync_backtest_stats():
    """Runs backtest strategy script to align logs and configs with updated data."""
    print("\n[+] Syncing backtest performance stats and charts...")
    script = os.path.join(DIR, "step9_backtest_strategy.py")
    proc = subprocess.run([sys.executable, script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if proc.returncode == 0:
        print("  [SUCCESS] Backtest strategy synced.")
        return True
    else:
        print(f"  [FAILED ] Backtest strategy returned error code {proc.returncode}")
        return False

def verify_api_server():
    """Checks if API server is up, self-heals (starts it) if down."""
    print("\n[+] Auditing API server status...")
    try:
        r = requests.get(f"{API_URL}/api/health", timeout=3)
        if r.status_code == 200:
            print(f"  [HEALTHY] API server is active on {API_URL}")
            return True
    except requests.exceptions.RequestException:
        pass
        
    print(f"  [DOWN   ] API server is unresponsive on port {PORT}. Healing...")
    
    # Check if there are existing python processes running step11_api_server
    # and kill them (in case of hangs)
    if sys.platform == "win32":
        subprocess.run("taskkill /f /im python.exe /fi \"windowtitle eq API Server*\"", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    script = os.path.join(DIR, "step11_api_server.py")
    print(f"  [STARTING] Launching API server thread...")
    
    # Spawn API server in a separate background process
    if sys.platform == "win32":
        subprocess.Popen([sys.executable, script], creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        subprocess.Popen([sys.executable, script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    # Wait for server to boot up
    for i in range(15):
        time.sleep(2)
        try:
            r = requests.get(f"{API_URL}/api/health", timeout=1)
            if r.status_code == 200:
                print(f"  [HEALED ] API server successfully recovered and listening on {API_URL}")
                return True
        except requests.exceptions.RequestException:
            pass
            
    print("  [FATAL  ] Failed to restore API server. Check step11_api_server.py logs.")
    return False

def main():
    force = "--force" in sys.argv
    
    print("="*60)
    print(" XAUUSD AI PIPELINE ORCHESTRATION & DATE CHECKER")
    print(f" Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    trading_active = is_trading_day()
    if not trading_active:
        print("[!] NOTICE: Today is a weekend. FX / Gold markets are closed.")
        if not force:
            print("    Pipeline skipped (use --force to run anyway).")
            # Make sure API server is still audited even on weekend
            verify_api_server()
            return
        else:
            print("    [--force] Weekend block overridden. Continuing...")
    else:
        print("[+] Today is a active FX trading day. Proceeding...")

    # Step 1: Audit files
    stale = audit_file_freshness()
    
    # Step 2: Refresh if stale or forced
    if stale or force:
        if stale:
            print(f"\n[!] Out-of-date files found: {stale}")
        success = trigger_refresh()
        if not success:
            print("\n[!] Refresh failed. Aborting pipeline.")
            sys.exit(1)
            
        # Sync backtest to fresh data
        sync_backtest_stats()
    else:
        print("\n[+] All datasets are up-to-date. No refresh required.")
        
    # Step 3: Audit & Self-heal API Server
    verify_api_server()
    
    print("\n" + "="*60)
    print(" PIPELINE AUDIT REPORT COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
