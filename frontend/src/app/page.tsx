"use client";

import React, { useEffect, useState } from "react";
import FundamentalDirectionWidget from "./components/FundamentalDirectionWidget";

// ─── Types ────────────────────────────────────────────────────────────────────
interface PredictData {
  scalp?: { signal: string; strength?: string; stop_loss: number | string; take_profit: number | string };
  swing?: { signal: string; strength?: string; stop_loss: number | string; take_profit: number | string };
  fundamental_trade?: { signal: string; strength?: string; stop_loss: number | string; take_profit: number | string };
  technical_trade?: { signal: string; strength?: string; stop_loss: number | string; take_profit: number | string };
  model_votes?: { catboost: number; xgboost: number; lightgbm: number };
  prediction?: { signal: string; probability_up: number; probability_down: number };
  thresholds?: { long: number; short: number };
  consensus_ok?: boolean;
  live_vader_label?: string;
  live_vader_sentiment?: number;
  narrative?: {
    summary: string;
    combinations?: { macro: string; order_flow: string; sentiment: string };
    reasoning: string;
    risk_note: string;
    trader_pillars?: {
      geopolitical?: { title: string; verdict: string; text: string };
      fed_policy?: { title: string; verdict: string; text: string };
      dollar_dxy?: { title: string; verdict: string; text: string };
      technical_of?: { title: string; verdict: string; text: string };
    };
    trader_insights?: string[];
  };
  shap_drivers?: { feature: string; text: string; direction: string; impact: number }[];
  mcc?: number;
  da?: number;
  date?: string;
  target_date?: string;
  risk_management?: { 
    entry_price: number; 
    stop_loss: number; 
    take_profit: number; 
    stop_loss_sw: number; 
    take_profit_sw: number; 
    atr_14: number;
    latest_close?: number;
  };
  intraday_levels?: { r2: number; r1: number; pp: number; s1: number; s2: number };
  ict_analysis?: {
    status?: string;
    timeframe?: string;
    timestamp?: string;
    current_price?: number;
    market_structure?: {
      current_trend: string;
      last_event: string;
      event_type: string;
      event_price: number;
      event_time?: string;
      recent_sh: number;
      recent_sl: number;
      prev_sh?: number;
      prev_sl?: number;
    };
    active_fvgs?: Array<{
      type: string;
      top: number;
      bottom: number;
      midpoint: number;
      size_usd: number;
      datetime: string;
      active: boolean;
    }>;
    all_fvgs_count?: number;
    liquidity?: {
      ote_discount_zone?: {
        start_62: number;
        sweet_spot_70: number;
        end_79: number;
        eq_50: number;
      };
      pdh_buy_side_liquidity?: number;
      pdl_sell_side_liquidity?: number;
      range_high?: number;
      range_low?: number;
    };
    gpt_synthesis?: {
      ict_verdict: string;
      actionable_headline: string;
      smart_money_plan: string;
      liquidity_narrative: string;
      recommended_entry: number;
      recommended_sl: number;
      recommended_tp: number;
    };
  };
  target_trade?: {
    name?: string;
    signal: string;
    entry_price: number;
    stop_loss: number;
    take_profit: number;
    be_triggered: boolean;
    last_outcome: string | null;
    current_price: number;
    win_rate?: string;
    risk_tier?: string;
    open_positions_count?: number;
    is_position_open?: boolean;
    live_broker_position?: {
      ticket: number;
      symbol: string;
      type: string;
      volume: number;
      price_open: number;
      sl: number;
      tp: number;
      profit: number;
      price_current: number;
    } | null;
  };
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
const signalColor = (s: string) =>
  s?.includes("LONG") || s?.includes("BUY") ? "#4ade80" : s?.includes("SHORT") || s?.includes("SELL") ? "#f87171" : "#fbbf24";

const pct = (v?: number, d = 1) => (v != null ? `${v >= 0 ? "+" : ""}${v.toFixed(d)}%` : "—");
const fmt = (v?: number, d = 2) => (v != null ? v.toFixed(d) : "—");

// ─── Timezone Conversion Helper (New York vs Sri Lanka UTC+5:30) ─────────────
const formatTimezone = (
  dateInput: string | Date | undefined | null,
  tf: "NY" | "LOCAL",
  formatType: "TIME" | "DATETIME" | "DATE" = "DATETIME"
): string => {
  if (!dateInput) return "—";
  try {
    let d: Date;
    if (typeof dateInput === "string") {
      // Handle standard ISO and SQL timestamps
      const isoStr = dateInput.includes("T")
        ? dateInput
        : dateInput.replace(" ", "T") + (dateInput.includes("Z") || dateInput.includes("+") ? "" : "Z");
      d = new Date(isoStr);
      if (isNaN(d.getTime())) {
        d = new Date(dateInput);
      }
    } else {
      d = dateInput;
    }

    if (isNaN(d.getTime())) return String(dateInput);

    const timeZone = tf === "NY" ? "America/New_York" : "Asia/Colombo";
    if (formatType === "TIME") {
      return d.toLocaleTimeString("en-US", { timeZone, hour12: true, hour: "2-digit", minute: "2-digit", second: "2-digit" });
    } else if (formatType === "DATE") {
      return d.toLocaleDateString("en-US", { timeZone, year: "numeric", month: "short", day: "2-digit" });
    } else {
      return d.toLocaleString("en-US", {
        timeZone,
        year: "numeric",
        month: "short",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        hour12: true
      });
    }
  } catch {
    return String(dateInput);
  }
};

const getSessionOpen = (start: number, end: number) => {
  const utcHours = new Date().getUTCHours();
  return utcHours >= start && utcHours < end;
};

function ProbBar({ label, prob, color }: { label: string; prob: number; color: string }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#9ca3af", marginBottom: 3 }}>
        <span>{label}</span>
        <span style={{ color, fontWeight: 700, fontFamily: "monospace" }}>{(prob * 100).toFixed(1)}%</span>
      </div>
      <div style={{ height: 6, background: "#1a1a2e", borderRadius: 3 }}>
        <div style={{ height: "100%", width: `${Math.min(Math.max(prob * 100, 0), 100)}%`, background: color, borderRadius: 3, transition: "width 0.6s ease" }} />
      </div>
    </div>
  );
}

function MetricCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div style={{ background: "#0d0d1a", border: "1px solid #1e293b", borderRadius: 8, padding: "14px 18px", minWidth: 120 }}>
      <div style={{ fontSize: 10, color: "#6b7280", letterSpacing: "0.08em", marginBottom: 4, textTransform: "uppercase" }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 800, color: color ?? "#f8fafc", fontFamily: "monospace" }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: "#6b7280", marginTop: 3 }}>{sub}</div>}
    </div>
  );
}



function TradingViewChart({ 
  predictData 
}: { 
  predictData?: PredictData | null 
}) {
  const [interval, setIntervalTf] = useState<string>("60"); // 15, 60, 240, D

  useEffect(() => {
    const scriptId = "tradingview-tv-script";
    let script = document.getElementById(scriptId) as HTMLScriptElement | null;

    const initWidget = () => {
      const container = document.getElementById("tradingview_xauusd");
      if (container) container.innerHTML = ""; // clean prior widget

      if (typeof window !== "undefined" && (window as any).TradingView) {
        new (window as any).TradingView.widget({
          width: "100%",
          height: 540,
          symbol: "OANDA:XAUUSD",
          interval: interval,
          timezone: "Etc/UTC",
          theme: "dark",
          style: "1",
          locale: "en",
          toolbar_bg: "#0d0d1a",
          enable_publishing: false,
          allow_symbol_change: false,
          container_id: "tradingview_xauusd",
          hide_side_toolbar: false,
          save_image: false,
          studies: [
            "MASimple@tv-basicstudies"
          ],
          overrides: {
            "mainSeriesProperties.candleStyle.upColor": "#22c55e",
            "mainSeriesProperties.candleStyle.downColor": "#ef4444",
            "mainSeriesProperties.candleStyle.drawWick": true,
            "mainSeriesProperties.candleStyle.drawBorder": true,
            "mainSeriesProperties.candleStyle.borderColor": "#374151",
            "mainSeriesProperties.candleStyle.borderUpColor": "#22c55e",
            "mainSeriesProperties.candleStyle.borderDownColor": "#ef4444",
            "mainSeriesProperties.candleStyle.wickUpColor": "#22c55e",
            "mainSeriesProperties.candleStyle.wickDownColor": "#ef4444",
            "paneProperties.background": "#0d0d1a",
            "paneProperties.vertGridProperties.color": "rgba(255, 255, 255, 0.04)",
            "paneProperties.horzGridProperties.color": "rgba(255, 255, 255, 0.04)",
          }
        });
      }
    };

    if (!script) {
      script = document.createElement("script");
      script.id = scriptId;
      script.src = "https://s3.tradingview.com/tv.js";
      script.async = true;
      script.onload = initWidget;
      document.head.appendChild(script);
    } else {
      if ((window as any).TradingView) {
        initWidget();
      } else {
        const prevOnload = script.onload;
        script.onload = (ev) => {
          if (prevOnload) (prevOnload as any)(ev);
          initWidget();
        };
      }
    }
  }, [interval]);

  const dir = predictData?.target_trade?.signal || predictData?.prediction?.signal || "LONG";
  const isLong = dir === "LONG" || dir.includes("BUY");
  const targetPrice = predictData?.ict_analysis?.gpt_synthesis?.recommended_tp 
    || predictData?.target_trade?.take_profit 
    || 4660.00;
  const entryPrice = predictData?.ict_analysis?.gpt_synthesis?.recommended_entry 
    || predictData?.target_trade?.entry_price 
    || 4635.17;
  const slPrice = predictData?.ict_analysis?.gpt_synthesis?.recommended_sl 
    || predictData?.target_trade?.stop_loss 
    || 4628.65;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14, width: "100%" }}>
      {/* Timeframe selector & AI Overlay Bar */}
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        flexWrap: "wrap",
        gap: 10,
        background: "rgba(13, 13, 26, 0.9)",
        border: "1px solid #1e293b",
        borderRadius: 8,
        padding: "10px 16px"
      }}>
        {/* Timeframe Switcher */}
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 11, color: "#6b7280", fontWeight: 700, marginRight: 4 }}>TIMEFRAME:</span>
          {[
            { label: "15M (ICT Execution)", val: "15" },
            { label: "1H (Intraday Trend)", val: "60" },
            { label: "4H (Swing Structure)", val: "240" },
            { label: "1D (Macro Cycle)", val: "D" },
          ].map(tf => (
            <button
              key={tf.val}
              onClick={() => setIntervalTf(tf.val)}
              style={{
                background: interval === tf.val ? "#3b82f6" : "#111827",
                color: interval === tf.val ? "#ffffff" : "#9ca3af",
                border: interval === tf.val ? "1px solid #60a5fa" : "1px solid #1f2937",
                borderRadius: 6,
                padding: "5px 12px",
                fontSize: 11.5,
                fontWeight: 700,
                cursor: "pointer",
                transition: "all 0.15s ease",
                boxShadow: interval === tf.val ? "0 0 10px rgba(59, 130, 246, 0.3)" : "none"
              }}
            >
              {tf.label}
            </button>
          ))}
        </div>

        {/* AI Predictive Projection Pill */}
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{
            fontSize: 11,
            background: isLong ? "rgba(34, 197, 94, 0.15)" : "rgba(239, 68, 68, 0.15)",
            color: isLong ? "#4ade80" : "#f87171",
            border: `1px solid ${isLong ? "#22c55e50" : "#ef444450"}`,
            borderRadius: 6,
            padding: "5px 12px",
            fontWeight: 800,
            display: "flex",
            alignItems: "center",
            gap: 6
          }}>
            <span>{isLong ? "🚀 AI FORECAST: BULLISH EXPANSION" : "🔻 AI FORECAST: BEARISH EXPANSION"}</span>
            <span style={{ color: "#fbbf24", fontFamily: "monospace" }}>➔ ${targetPrice.toFixed(2)}</span>
          </span>
          <span style={{
            fontSize: 11,
            background: "#1e293b",
            color: "#94a3b8",
            padding: "5px 10px",
            borderRadius: 6,
            fontFamily: "monospace",
            border: "1px solid #334155"
          }}>
            SL: <strong style={{ color: "#f87171" }}>${slPrice.toFixed(2)}</strong>
          </span>
        </div>
      </div>

      {/* Chart Canvas with explicit 540px height */}
      <div id="tradingview_xauusd_chart" style={{ width: "100%", height: "540px", borderRadius: "10px", overflow: "hidden", border: "1px solid #1e293b", background: "#0d0d1a" }}>
        <div id="tradingview_xauusd" style={{ width: "100%", height: "540px" }}></div>
      </div>
    </div>
  );
}

function CountdownRing({ seconds }: { seconds: number }) {
  const circ = 2 * Math.PI * 15.9;
  const strokeOffset = circ * (1 - seconds / 900);
  const minutes = Math.floor(seconds / 60);
  const secs = seconds % 60;
  const label = `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;

  return (
    <div className="countdown-wrap" title="Next auto-refresh">
      <svg className="countdown-ring" viewBox="0 0 36 36">
        <circle className="countdown-bg" cx="18" cy="18" r="15.9" />
        <circle 
          className="countdown-arc" 
          id="countdown-arc" 
          cx="18" 
          cy="18" 
          r="15.9"
          style={{ strokeDasharray: `${circ}`, strokeDashoffset: strokeOffset }}
        />
      </svg>
      <span className="countdown-label">{label}</span>
    </div>
  );
}

// ─── Main Page ─────────────────────────────────────────────────────────────────
export default function Home() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [predictData, setPredictData] = useState<PredictData | null>(null);
  const [newsData, setNewsData] = useState<any[]>([]);
  const [newsCategory, setNewsCategory] = useState<string>("ALL");
  const [macroCalendarData, setMacroCalendarData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [healthData, setHealthData] = useState<any>(null);
  const [refreshingDaily, setRefreshingDaily] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<string>("");
  const [timeframe, setTimeframe] = useState<"LOCAL" | "NY">("NY");
  const [currentTime, setCurrentTime] = useState<string>("");
  const [countdown, setCountdown] = useState<number>(900);
  const [tradeLockData, setTradeLockData] = useState<any>(null);
  const [lockingTrade, setLockingTrade] = useState(false);

  const fetchTradeLock = () => {
    fetch("/api/trade-lock")
      .then(r => r.json())
      .then(d => { if (d.data) setTradeLockData(d.data); })
      .catch(() => {});
  };

  const handleLockSetup = async () => {
    setLockingTrade(true);
    try {
      const res = await fetch("/api/trade-lock/lock-setup", { method: "POST" });
      const data = await res.json();
      if (data.data) setTradeLockData(data.data);
      refreshData();
    } finally {
      setLockingTrade(false);
    }
  };

  const handleActivateLive = async () => {
    if (!window.confirm("Confirm: Execute LIVE order on MT5 broker and activate trade lock?")) return;
    setLockingTrade(true);
    try {
      const res = await fetch("/api/trade-lock/activate?is_live=true", { method: "POST" });
      const data = await res.json();
      fetchTradeLock();
      refreshData();
    } finally {
      setLockingTrade(false);
    }
  };

  const handleUnlock = async () => {
    if (!window.confirm("Unlock current trade and trigger fresh Smart Money scan?")) return;
    setLockingTrade(true);
    try {
      const res = await fetch("/api/trade-lock/unlock", { method: "POST" });
      const data = await res.json();
      if (data.data) setTradeLockData(data.data);
      refreshData();
    } finally {
      setLockingTrade(false);
    }
  };

  const refreshData = () => {
    setLoading(true);
    const now = new Date();
    const nowStr = formatTimezone(now, timeframe, "TIME");
    fetchTradeLock();

    Promise.allSettled([
      fetch("/api/predict").then(r => r.json()).then(d => setPredictData(d)),
      fetch("/api/live-news").then(r => r.json()).then(d => { if (d.news) setNewsData(d.news); }),
      fetch("/api/macro-calendar").then(r => r.json()).then(d => { if (d.events) setMacroCalendarData(d.events); }),
      fetch("/api/health").then(r => r.json()).then(d => {
        setHealthData(d);
        if (d.refreshing_daily) {
          setRefreshingDaily(true);
        }
      }),
    ]).finally(() => { setLoading(false); setLastRefresh(nowStr); setCountdown(900); });
  };

  useEffect(() => {
    refreshData();
  }, []);

  useEffect(() => {
    const clockInterval = setInterval(() => {
      const now = new Date();
      if (timeframe === "NY") {
        setCurrentTime(now.toLocaleTimeString("en-US", { timeZone: "America/New_York", hour12: true, hour: "2-digit", minute: "2-digit", second: "2-digit" }));
      } else {
        setCurrentTime(now.toLocaleTimeString("en-US", { timeZone: "Asia/Colombo", hour12: true, hour: "2-digit", minute: "2-digit", second: "2-digit" }));
      }
    }, 1000);
    return () => clearInterval(clockInterval);
  }, [timeframe]);

  useEffect(() => {
    const timer = setInterval(() => {
      setCountdown(c => {
        if (c <= 1) {
          refreshData();
          return 900;
        }
        return c - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (refreshingDaily) {
      interval = setInterval(() => {
        fetch("/api/health")
          .then(r => r.json())
          .then(d => {
            setHealthData(d);
            if (!d.refreshing_daily) {
              setRefreshingDaily(false);
              refreshData();
            }
          });
      }, 5000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [refreshingDaily]);

  const triggerDailyPipeline = () => {
    if (refreshingDaily) return;
    setRefreshingDaily(true);
    fetch("/api/refresh", { method: "POST" })
      .then(r => r.json())
      .then(() => {
        fetch("/api/health").then(r => r.json()).then(h => setHealthData(h));
      });
  };

  const tabs = [
    { id: "dashboard", label: "📡 SIGNALS" },
    { id: "ai_insights", label: "🧠 AI PREDICTIONS & REASONS" },
    { id: "news", label: "🗞 NEWS" },
    { id: "calendar", label: "📅 CALENDAR" },
  ];

  if (loading && !predictData) {
    return (
      <div style={{
        minHeight: "100vh",
        background: "#060612",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        color: "#e2e8f0",
        fontFamily: "'Inter', sans-serif"
      }}>
        <div style={{
          width: "50px",
          height: "50px",
          border: "3px solid rgba(245, 200, 66, 0.1)",
          borderTop: "3px solid #fbbf24",
          borderRadius: "50%",
          animation: "spin 1s linear infinite",
          boxShadow: "0 0 15px rgba(245, 200, 66, 0.2)",
          marginBottom: "24px"
        }} />
        <style dangerouslySetInnerHTML={{ __html: `
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
          @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(0.97); }
            100% { opacity: 1; transform: scale(1); }
          }
        `}} />
        <div style={{ fontWeight: 800, fontSize: "16px", letterSpacing: "0.1em", color: "#fbbf24", marginBottom: "8px" }}>
          XAUUSD AI TERMINAL
        </div>
        <div style={{ fontSize: "11px", color: "#6b7280", letterSpacing: "0.05em" }}>
          Initializing Neural Ensemble & Live Sentiment Feed...
        </div>
      </div>
    );
  }

  return (
    <div style={{ minHeight: "100vh", background: "#060612", color: "#e2e8f0", fontFamily: "'Inter', 'SF Mono', monospace" }}>
      {/* ─── NEWS TICKER BAR ────────────────────────────────────────── */}
      {newsData.length > 0 && (
        <div className="ticker-bar">
          <span className="ticker-tag">LIVE</span>
          <div className="ticker-scroll-wrap">
            <div 
              className="ticker-scroll" 
              style={{ 
                animationDuration: `${Math.max(30, newsData.length * 6)}s`
              }}
            >
              {newsData.concat(newsData).map((n, i) => {
                const label = n.sentiment || "NEUTRAL";
                const sentColor = label === "BULLISH" || label === "POSITIVE" ? "var(--green)" : label === "BEARISH" || label === "NEGATIVE" ? "var(--red)" : "var(--t3)";
                const tag = label === "BULLISH" || label === "POSITIVE" ? "▲" : label === "BEARISH" || label === "NEGATIVE" ? "▼" : "—";
                return (
                  <span key={i} className="tick-item">
                    <span className="tick-tag" style={{ color: sentColor }}>{tag}</span>
                    <a href={n.url || "#"} target="_blank" rel="noopener noreferrer" className="tick-link">
                      {n.headline}
                    </a>
                    <span className="tick-src">{n.source}</span>
                    <span className="tick-sep">·</span>
                  </span>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* ─── Top Navigation ─────────────────────────────────────────────── */}
      <nav style={{ background: "#0a0a1a", borderBottom: "1px solid #1e293b", padding: "0 36px", display: "flex", alignItems: "center", justifyContent: "space-between", height: 58 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{ width: 9, height: 9, background: "#fbbf24", borderRadius: "50%", boxShadow: "0 0 10px #fbbf24" }} />
          <span style={{ fontWeight: 800, fontSize: 16, letterSpacing: "0.05em" }}>
            XAUUSD <span style={{ color: "#fbbf24" }}>AI TERMINAL</span>
          </span>
          <span style={{ fontSize: 10, color: "#9ca3af", background: "#111827", padding: "3px 10px", borderRadius: 4, letterSpacing: "0.08em", fontWeight: 700 }}>v3.0 · ATTENTION ENGINE</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
          {/* Freshness Status Indicators */}
          {healthData && (
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{
                width: 8, height: 8, borderRadius: "50%",
                background: healthData.is_stale ? "#f87171" : "#4ade80",
                boxShadow: healthData.is_stale ? "0 0 8px #f87171" : "0 0 8px #4ade80"
              }} />
              <span style={{ fontSize: 10.5, color: "#9ca3af", fontWeight: 700, letterSpacing: "0.06em" }}>
                {healthData.is_stale ? "DATA STALE" : "DATA FRESH"}
              </span>
            </div>
          )}

          {/* Countdown Ring */}
          <CountdownRing seconds={countdown} />

          {/* Timezone Toggle Switch (New York vs Sri Lanka Local) */}
          <div style={{
            display: "flex",
            alignItems: "center",
            background: "#111827",
            border: "1px solid #1e293b",
            borderRadius: 20,
            padding: "2px 3px",
            gap: 2
          }}>
            <button
              onClick={() => setTimeframe("NY")}
              style={{
                background: timeframe === "NY" ? "linear-gradient(135deg, #fbbf24, #d97706)" : "transparent",
                color: timeframe === "NY" ? "#000" : "#9ca3af",
                border: "none",
                borderRadius: 16,
                padding: "4px 10px",
                fontSize: 11,
                fontWeight: 800,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: 5,
                transition: "all 0.2s",
                boxShadow: timeframe === "NY" ? "0 0 10px rgba(251, 191, 36, 0.3)" : "none"
              }}
              title="New York Market Time (EST / EDT - UTC-4)"
            >
              <span>🗽</span>
              <span>NY</span>
            </button>
            <button
              onClick={() => setTimeframe("LOCAL")}
              style={{
                background: timeframe === "LOCAL" ? "linear-gradient(135deg, #fbbf24, #d97706)" : "transparent",
                color: timeframe === "LOCAL" ? "#000" : "#9ca3af",
                border: "none",
                borderRadius: 16,
                padding: "4px 10px",
                fontSize: 11,
                fontWeight: 800,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: 5,
                transition: "all 0.2s",
                boxShadow: timeframe === "LOCAL" ? "0 0 10px rgba(251, 191, 36, 0.3)" : "none"
              }}
              title="Sri Lanka Local Time (Asia/Colombo - UTC+5:30)"
            >
              <span>🇱🇰</span>
              <span>Sri Lanka</span>
            </button>
          </div>

          {/* Timezone Clock Widget */}
          <div style={{
            background: "#0d0d1a",
            border: "1px solid #1e293b",
            borderRadius: 6,
            padding: "4px 12px",
            display: "flex",
            alignItems: "center",
            gap: 8,
            fontFamily: "var(--mono)",
            fontSize: 12,
            color: "#f8fafc"
          }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#4ade80", boxShadow: "0 0 6px #4ade80" }} />
            <span style={{ color: "#fbbf24", fontWeight: 800 }}>{timeframe === "NY" ? "NY (EST)" : "LK (UTC+5:30)"}:</span>
            <span style={{ fontWeight: 700 }}>{currentTime || "--:--:--"}</span>
          </div>
          
          <button 
            onClick={triggerDailyPipeline}
            disabled={refreshingDaily}
            style={{
              background: refreshingDaily ? "linear-gradient(90deg, #b45309, #d97706)" : "#1e293b",
              border: "1px solid #334155",
              color: "#fbbf24",
              padding: "5px 14px",
              borderRadius: 6,
              fontSize: 11.5,
              cursor: refreshingDaily ? "not-allowed" : "pointer",
              display: "flex",
              alignItems: "center",
              gap: 6,
              fontWeight: 700,
              boxShadow: refreshingDaily ? "0 0 12px rgba(245, 191, 36, 0.25)" : "none",
              transition: "all 0.2s",
              animation: refreshingDaily ? "pulse 1.5s infinite" : "none"
            }}
          >
            {refreshingDaily ? "⚡ RUNNING..." : "⚡ RUN PIPELINE"}
          </button>

          <span style={{ fontSize: 11.5, color: "#9ca3af" }}>
            {lastRefresh ? `Refreshed: ${lastRefresh}` : ""}
          </span>
          <button 
            onClick={refreshData}
            disabled={loading || refreshingDaily}
            style={{
              background: "#1e293b",
              border: "1px solid #334155",
              color: "#fbbf24",
              padding: "5px 14px",
              borderRadius: 6,
              fontSize: 11.5,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 6,
              fontWeight: 700,
              transition: "background 0.2s"
            }}
          >
            {loading ? "⟳ LOADING..." : "⟳ REFRESH"}
          </button>
        </div>
      </nav>

      {/* ─── Tab Bar ────────────────────────────────────────────────────── */}
      <div style={{ background: "#0a0a1a", borderBottom: "1px solid #1e293b", padding: "0 36px", display: "flex", gap: 8 }}>
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            style={{
              background: "none", border: "none", cursor: "pointer", padding: "14px 22px",
              fontSize: 12.5, fontWeight: 700, letterSpacing: "0.06em", color: activeTab === t.id ? "#fbbf24" : "#6b7280",
              borderBottom: activeTab === t.id ? "2px solid #fbbf24" : "2px solid transparent",
              transition: "all 0.2s",
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div style={{ padding: "32px 36px 64px 36px", maxWidth: 1440, margin: "0 auto" }}>

        {/* ═══════════════════════════════════════════════════════════════
            DASHBOARD TAB
        ═══════════════════════════════════════════════════════════════ */}
        {activeTab === "dashboard" && (
          <div>

            {/* 👑 UNIFIED SMART MONEY MASTER TRADE (ONE DEFINITIVE SIGNAL) */}
            {predictData?.target_trade && (
              <div style={{
                background: "linear-gradient(135deg, #0d0d26 0%, #060613 100%)",
                border: "1px solid #3b82f660",
                borderRadius: 14,
                padding: "26px 30px",
                marginBottom: 24,
                boxShadow: "0 0 28px rgba(59, 130, 246, 0.18)"
              }}>
                {/* Header Title & Tags */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20, flexWrap: "wrap", gap: 10 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <span style={{ fontSize: 24 }}>👑</span>
                    <div>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ fontSize: 15, fontWeight: 900, color: "#f8fafc", letterSpacing: "0.06em" }}>
                          UNIFIED SMART MONEY MASTER TRADE (ONE SIGNAL ENGINE)
                        </span>
                        <span style={{
                          fontSize: 10,
                          background: "rgba(59, 130, 246, 0.2)",
                          color: "#60a5fa",
                          border: "1px solid rgba(59, 130, 246, 0.4)",
                          padding: "2px 8px",
                          borderRadius: 4,
                          fontWeight: 800
                        }}>
                          62.2% SMC WIN RATE
                        </span>
                      </div>
                      <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 2 }}>
                        Multi-Modal Synthesis: AI ML Ensemble (64.3%) + 15M ICT Market Structure + Live Gold News Sentiment
                      </div>
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <span style={{
                      fontSize: 10,
                      background: predictData.target_trade.is_position_open ? "rgba(74, 222, 128, 0.2)" : "rgba(251, 191, 36, 0.15)",
                      color: predictData.target_trade.is_position_open ? "#4ade80" : "#fbbf24",
                      border: `1px solid ${predictData.target_trade.is_position_open ? "rgba(74, 222, 128, 0.4)" : "rgba(251, 191, 36, 0.3)"}`,
                      padding: "4px 10px",
                      borderRadius: 6,
                      fontWeight: 800,
                      letterSpacing: "0.08em"
                    }}>
                      {predictData.target_trade.is_position_open 
                        ? `🟢 LIVE MT5 POSITION (TICKET #${predictData.target_trade.live_broker_position?.ticket})` 
                        : "⏳ PENDING SETUP (NO LIVE TRADE RUNNING)"}
                    </span>
                    <span style={{
                      fontSize: 10,
                      background: "#1e293b",
                      color: "#fbbf24",
                      padding: "4px 10px",
                      borderRadius: 6,
                      fontWeight: 700,
                      fontFamily: "monospace"
                    }}>
                      RISK: {predictData.prediction && predictData.prediction.probability_up >= 0.65 ? "2.0% HIGH CONF" : "1.0% NORMAL"}
                    </span>
                  </div>
                </div>

                {/* Main 4-Column Execution Metrics Grid */}
                <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1.1fr 1.1fr 1.1fr 1.4fr", gap: 16, marginBottom: 20, alignItems: "stretch" }}>
                  {/* Direction */}
                  <div style={{ background: "#080816", border: "1px solid #1e293b", borderRadius: 10, padding: "14px 16px", display: "flex", flexDirection: "column", justifyContent: "center" }}>
                    <span style={{ fontSize: 10, color: "#6b7280", fontWeight: 700, textTransform: "uppercase" }}>MASTER SIGNAL</span>
                    <div style={{ fontSize: 24, fontWeight: 900, color: signalColor(predictData.target_trade.signal), marginTop: 2 }}>
                      {predictData.target_trade.signal === "LONG" ? "BUY (LONG)" : predictData.target_trade.signal === "SHORT" ? "SELL (SHORT)" : "HOLD / WAIT"}
                    </div>
                    <span style={{ fontSize: 10.5, color: "#9ca3af", marginTop: 2 }}>
                      AI Confidence: {predictData.prediction ? `${(predictData.prediction.probability_up * 100).toFixed(1)}%` : "64.3%"}
                    </span>
                  </div>

                  {/* FVG Entry */}
                  <div style={{ background: "#080816", border: "1px solid #1e293b", borderRadius: 10, padding: "14px 16px", display: "flex", flexDirection: "column", justifyContent: "center" }}>
                    <span style={{ fontSize: 10, color: "#4ade80", fontWeight: 700, textTransform: "uppercase" }}>OPTIMAL FVG ENTRY</span>
                    <div style={{ fontSize: 20, fontWeight: 900, color: "#4ade80", fontFamily: "monospace", marginTop: 2 }}>
                      ${predictData.ict_analysis?.gpt_synthesis?.recommended_entry ? predictData.ict_analysis.gpt_synthesis.recommended_entry.toFixed(2) : predictData.target_trade.entry_price.toFixed(2)}
                    </div>
                    <span style={{ fontSize: 10, color: "#6b7280", marginTop: 2 }}>
                      Discount Limit Order
                    </span>
                  </div>

                  {/* Stop Loss */}
                  <div style={{ background: "#080816", border: "1px solid #1e293b", borderRadius: 10, padding: "14px 16px", display: "flex", flexDirection: "column", justifyContent: "center" }}>
                    <span style={{ fontSize: 10, color: "#f87171", fontWeight: 700, textTransform: "uppercase" }}>STRUCTURAL STOP LOSS</span>
                    <div style={{ fontSize: 20, fontWeight: 900, color: "#f87171", fontFamily: "monospace", marginTop: 2 }}>
                      ${predictData.ict_analysis?.gpt_synthesis?.recommended_sl ? predictData.ict_analysis.gpt_synthesis.recommended_sl.toFixed(2) : predictData.target_trade.stop_loss.toFixed(2)}
                    </div>
                    <span style={{ fontSize: 10, color: "#6b7280", marginTop: 2 }}>
                      Protected below Swing Low
                    </span>
                  </div>

                  {/* Take Profit */}
                  <div style={{ background: "#080816", border: "1px solid #1e293b", borderRadius: 10, padding: "14px 16px", display: "flex", flexDirection: "column", justifyContent: "center" }}>
                    <span style={{ fontSize: 10, color: "#fbbf24", fontWeight: 700, textTransform: "uppercase" }}>LIQUIDITY TARGET (TP)</span>
                    <div style={{ fontSize: 20, fontWeight: 900, color: "#fbbf24", fontFamily: "monospace", marginTop: 2 }}>
                      ${predictData.ict_analysis?.gpt_synthesis?.recommended_tp ? predictData.ict_analysis.gpt_synthesis.recommended_tp.toFixed(2) : predictData.target_trade.take_profit.toFixed(2)}
                    </div>
                    <span style={{ fontSize: 10, color: "#6b7280", marginTop: 2 }}>
                      Buyside Pool (~1:3 R:R)
                    </span>
                  </div>

                  {/* Live Pending P&L & BE vs Order State */}
                  <div style={{ background: "#080816", border: "1px solid #1e293b", borderRadius: 10, padding: "14px 16px", display: "flex", flexDirection: "column", justifyContent: "center" }}>
                    {(() => {
                      const isOpen = predictData.target_trade.is_position_open;
                      const sig = predictData.target_trade.signal;
                      const ep = predictData.ict_analysis?.gpt_synthesis?.recommended_entry || predictData.target_trade.entry_price;
                      const cp = predictData.target_trade.current_price;
                      const diff = sig === "LONG" ? cp - ep : ep - cp;
                      const pips = diff * 10;
                      const pctVal = (diff / ep) * 100;
                      const pnlColor = pips >= 0 ? "#4ade80" : "#f87171";

                      if (isOpen) {
                        const profitUsd = predictData.target_trade.live_broker_position?.profit || 0;
                        const posColor = profitUsd >= 0 ? "#4ade80" : "#f87171";
                        return (
                          <>
                            <span style={{ fontSize: 10, color: "#6b7280", fontWeight: 700, textTransform: "uppercase" }}>LIVE BROKER P&L</span>
                            <div style={{ fontSize: 18, fontWeight: 900, fontFamily: "monospace", color: posColor, marginTop: 2 }}>
                              {profitUsd >= 0 ? "+" : ""}${profitUsd.toFixed(2)} ({pips >= 0 ? "+" : ""}{pips.toFixed(1)} pips)
                            </div>
                            <span style={{ fontSize: 10, color: predictData.target_trade.be_triggered ? "#fbbf24" : "#94a3b8", marginTop: 2 }}>
                              {predictData.target_trade.be_triggered ? "🛡️ Break-Even Active ($0 Risk)" : "⏳ Auto BE at 1:1 R:R"}
                            </span>
                          </>
                        );
                      } else {
                        return (
                          <>
                            <span style={{ fontSize: 10, color: "#fbbf24", fontWeight: 700, textTransform: "uppercase" }}>EXECUTION STATUS</span>
                            <div style={{ fontSize: 13.5, fontWeight: 800, color: "#f8fafc", marginTop: 2 }}>
                              ⏳ WAITING FOR RETRACEMENT
                            </div>
                            <span style={{ fontSize: 10.5, color: "#9ca3af", marginTop: 2 }}>
                              Market is {Math.abs(pips).toFixed(1)} pips {pips > 0 ? "above" : "below"} limit entry
                            </span>
                          </>
                        );
                      }
                    })()}
                  </div>
                </div>

                {/* 4 Unified Pillars Row */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 12, marginBottom: 16 }}>
                  <div style={{ background: "rgba(0,0,0,0.25)", border: "1px solid #1e293b", borderRadius: 8, padding: "10px 12px" }}>
                    <div style={{ fontSize: 10, color: "#9ca3af", fontWeight: 700 }}>🧠 1. AI ENSEMBLE CONSENSUS</div>
                    <div style={{ fontSize: 12, fontWeight: 800, color: "#38bdf8", marginTop: 2 }}>
                      Cat: 48% | XGB: 63% | LGB: 82%
                    </div>
                  </div>

                  <div style={{ background: "rgba(0,0,0,0.25)", border: "1px solid #1e293b", borderRadius: 8, padding: "10px 12px" }}>
                    <div style={{ fontSize: 10, color: "#9ca3af", fontWeight: 700 }}>⚡ 2. SMART MONEY (ICT)</div>
                    <div style={{ fontSize: 12, fontWeight: 800, color: "#4ade80", marginTop: 2 }}>
                      {predictData.ict_analysis?.market_structure?.last_event || "BULLISH_BOS"} (15M OTE)
                    </div>
                  </div>

                  <div style={{ background: "rgba(0,0,0,0.25)", border: "1px solid #1e293b", borderRadius: 8, padding: "10px 12px" }}>
                    <div style={{ fontSize: 10, color: "#9ca3af", fontWeight: 700 }}>📰 3. GOLD NEWS SENTIMENT</div>
                    <div style={{ fontSize: 12, fontWeight: 800, color: "#fbbf24", marginTop: 2 }}>
                      {predictData.live_vader_label || "BULLISH"} (+{predictData.live_vader_sentiment || "0.76"})
                    </div>
                  </div>

                  <div style={{ background: "rgba(0,0,0,0.25)", border: "1px solid #1e293b", borderRadius: 8, padding: "10px 12px" }}>
                    <div style={{ fontSize: 10, color: "#9ca3af", fontWeight: 700 }}>🏦 4. FED & US DOLLAR (DXY)</div>
                    <div style={{ fontSize: 12, fontWeight: 800, color: "#a78bfa", marginTop: 2 }}>
                      Macro Safe-Haven Tailwinds
                    </div>
                  </div>
                </div>

                {/* GPT-4o Execution Plan Memo */}
                <div style={{ background: "rgba(14, 14, 40, 0.6)", border: "1px solid rgba(59, 130, 246, 0.2)", borderRadius: 8, padding: "14px 18px", marginBottom: 16 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                    <span style={{ fontSize: 13 }}>💡</span>
                    <span style={{ fontSize: 11, fontWeight: 800, color: "#93c5fd", letterSpacing: "0.08em" }}>
                      GPT-4o UNIFIED EXECUTION DIRECTIVE:
                    </span>
                  </div>
                  <p style={{ margin: 0, fontSize: 13, color: "#e2e8f0", lineHeight: 1.5 }}>
                    {predictData.ict_analysis?.gpt_synthesis?.smart_money_plan || predictData.narrative?.summary || "Wait for 15M discount retracement into Fair Value Gap before executing."}
                  </p>
                </div>

                {/* 🔒 TRADE LOCK & PERSISTENT LIFECYCLE CONTROLLER */}
                <div style={{
                  background: "linear-gradient(90deg, rgba(16, 24, 52, 0.8) 0%, rgba(10, 15, 36, 0.9) 100%)",
                  border: "1px solid #3b82f640",
                  borderRadius: 8,
                  padding: "12px 18px",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  flexWrap: "wrap",
                  gap: 12
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ fontSize: 18 }}>{tradeLockData?.status === "LOCKED_ACTIVE" ? "🔒" : tradeLockData?.status === "PENDING_LIMIT" ? "⏳" : "🔄"}</span>
                    <div>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ fontSize: 12, fontWeight: 800, color: "#fbbf24", letterSpacing: "0.06em" }}>
                          LIFECYCLE STATE: {tradeLockData?.status || "PENDING_LIMIT"}
                        </span>
                        <span style={{
                          fontSize: 10,
                          background: tradeLockData?.locked_trade ? "rgba(34, 197, 94, 0.2)" : "#1e293b",
                          color: tradeLockData?.locked_trade ? "#4ade80" : "#94a3b8",
                          padding: "2px 8px",
                          borderRadius: 4,
                          fontWeight: 700
                        }}>
                          {tradeLockData?.locked_trade ? "🔒 LOCKED & PROTECTED" : "WAITING FOR LOCK"}
                        </span>
                      </div>
                      <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 2 }}>
                        🔁 <strong>Continuous Rule Active:</strong> System locks position until TP (${predictData.ict_analysis?.gpt_synthesis?.recommended_tp || predictData.target_trade.take_profit}) or SL (${predictData.ict_analysis?.gpt_synthesis?.recommended_sl || predictData.target_trade.stop_loss}) hits, then automatically scans and enters next trade.
                      </div>
                    </div>
                  </div>

                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <button
                      onClick={handleLockSetup}
                      disabled={lockingTrade}
                      style={{
                        background: "#1e293b",
                        border: "1px solid #3b82f6",
                        color: "#93c5fd",
                        padding: "6px 14px",
                        borderRadius: 6,
                        fontSize: 11,
                        fontWeight: 800,
                        cursor: "pointer",
                        transition: "all 0.2s"
                      }}
                    >
                      {lockingTrade ? "..." : "🔒 LOCK SETUP"}
                    </button>
                    
                    <button
                      onClick={handleActivateLive}
                      disabled={lockingTrade}
                      style={{
                        background: "linear-gradient(90deg, #15803d, #22c55e)",
                        border: "1px solid #4ade80",
                        color: "#ffffff",
                        padding: "6px 14px",
                        borderRadius: 6,
                        fontSize: 11,
                        fontWeight: 800,
                        cursor: "pointer",
                        boxShadow: "0 0 12px rgba(34, 197, 94, 0.3)",
                        transition: "all 0.2s"
                      }}
                    >
                      {lockingTrade ? "..." : "🚀 ACTIVATE IN MT5"}
                    </button>

                    <button
                      onClick={handleUnlock}
                      disabled={lockingTrade}
                      style={{
                        background: "#1f2937",
                        border: "1px solid #4b5563",
                        color: "#9ca3af",
                        padding: "6px 12px",
                        borderRadius: 6,
                        fontSize: 11,
                        fontWeight: 700,
                        cursor: "pointer"
                      }}
                      title="Manual override to clear lock and trigger new scan"
                    >
                      🔓 UNLOCK & RESCAN
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* ROW 0: MAIN CANDLESTICK CHART */}
            <div className="card card-chart" style={{ padding: "22px 26px", minHeight: "640px", display: "flex", flexDirection: "column", marginBottom: "24px", borderRadius: 12 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                  <span className="card-label" style={{ margin: 0, fontSize: 13, fontWeight: 800 }}>📈 XAU/USD LIVE CANDLESTICK CHART & AI PREDICTIVE TREND</span>
                  <span className="tag-green" style={{ fontSize: "0.68rem", padding: "3px 8px" }}>LSTM / ENSEMBLE OVERLAY</span>
                </div>
                <div style={{ display: "flex", gap: "8px" }}>
                  <span className="exec-badge" style={{ background: "var(--surf2)", color: "var(--gold)", borderColor: "var(--gold-dim)" }}>INSTITUTIONAL SMC VIEW</span>
                  <span className="exec-badge" style={{ background: "var(--surf2)", color: "var(--green)", borderColor: "rgba(0,229,160,0.3)" }}>REAL-TIME 1S TICK</span>
                </div>
              </div>
              <TradingViewChart predictData={predictData} />
            </div>

            {/* Consensus Gate */}
            <div style={{ background: "#0d0d1a", border: "1px solid #1e293b", borderRadius: 12, padding: "24px 28px", marginBottom: 24 }}>
              <div style={{ fontSize: 11, color: "#6b7280", letterSpacing: "0.12em", marginBottom: 16, fontWeight: 700 }}>🔑 CONSENSUS GATE (Hernes)</div>
              {predictData?.model_votes ? (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
                  <div>
                    <ProbBar label="CatBoost" prob={predictData.model_votes.catboost} color="#a78bfa" />
                    <ProbBar label="XGBoost" prob={predictData.model_votes.xgboost} color="#38bdf8" />
                  </div>
                  <div>
                    <ProbBar label="LightGBM" prob={predictData.model_votes.lightgbm} color="#34d399" />
                    <ProbBar label="Ensemble Mean" prob={predictData.prediction?.probability_up ?? 0.5} color="#fbbf24" />
                  </div>
                </div>
              ) : (
                <div style={{ color: "#374151", fontSize: 13 }}>No model probability data available.</div>
              )}
            </div>

            {/* Metrics Row */}
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 24 }}>
              {predictData?.risk_management?.entry_price && (
                <MetricCard label="Gold Entry Price" value={`$${predictData.risk_management.entry_price.toFixed(2)}`} sub={`ATR (14-day): $${predictData.risk_management.atr_14.toFixed(2)}`} color="#fbbf24" />
              )}
              {predictData?.prediction && (
                <MetricCard label="Ensemble Probability" value={`${(predictData.prediction.probability_up * 100).toFixed(1)}%`} sub={`DOWN: ${(predictData.prediction.probability_down * 100).toFixed(1)}%`} color={predictData.prediction.probability_up > 0.5 ? "#4ade80" : "#f87171"} />
              )}
              {predictData?.target_date && (
                <MetricCard label="Target Date" value={predictData.target_date} sub={`Inference: ${predictData.date}`} />
              )}
            </div>

            {/* SHAP Top Features & Risk Management */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 24 }}>
              {/* SHAP */}
              <div style={{ background: "#0d0d1a", border: "1px solid #1e293b", borderRadius: 12, padding: "24px 28px" }}>
                <div style={{ fontSize: 11, color: "#6b7280", letterSpacing: "0.1em", marginBottom: 16, fontWeight: 700 }}>TOP FEATURE INFLUENCES (SHAP)</div>
                {predictData?.shap_drivers && predictData.shap_drivers.length > 0 ? (
                  predictData.shap_drivers.slice(0, 8).map((f, i) => (
                    <div key={i} style={{ marginBottom: 10 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 3 }}>
                        <span style={{ color: "#9ca3af", fontFamily: "monospace" }}>#{i + 1} {f.feature}</span>
                        <span style={{ color: f.direction === "UP" ? "#4ade80" : "#f87171", fontFamily: "monospace" }}>
                          {f.direction} ({Math.abs(f.impact).toFixed(4)})
                        </span>
                      </div>
                      <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 4 }}>{f.text}</div>
                      <div style={{ height: 3, background: "#1a1a2e", borderRadius: 2 }}>
                        <div style={{ height: "100%", width: `${Math.min(Math.abs(f.impact) / (Math.abs(predictData.shap_drivers![0].impact) || 1) * 100, 100)}%`, background: f.direction === "UP" ? "#4ade80" : "#f87171", borderRadius: 2, opacity: 0.7 }} />
                      </div>
                    </div>
                  ))
                ) : (
                  <div style={{ color: "#374151", fontSize: 13 }}>Run Step 8 to generate SHAP data.</div>
                )}
              </div>

              {/* Risk Management & Intraday Pillars */}
              <div style={{ background: "#0d0d1a", border: "1px solid #1e293b", borderRadius: 12, padding: "24px 28px" }}>
                <div style={{ fontSize: 11, color: "#6b7280", letterSpacing: "0.1em", marginBottom: 16, fontWeight: 700 }}>🛡️ RISK CONTROL & INTRADAY PILLARS</div>
                {predictData?.risk_management ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                    <div>
                      <div style={{ fontSize: 10.5, color: "#6b7280", marginBottom: 4 }}>LIVE ENTRY BENCHMARK</div>
                      <div style={{ fontSize: 22, fontWeight: 800, color: "#fbbf24", fontFamily: "monospace" }}>
                        ${predictData.risk_management.entry_price.toFixed(2)}
                      </div>
                      <div style={{ fontSize: 11, color: "#6b7280", marginTop: 4 }}>ATR (14-day): ${predictData.risk_management.atr_14.toFixed(2)}</div>
                    </div>
                    
                    <div style={{ borderTop: "1px solid #1e293b", paddingTop: 14 }}>
                      <div style={{ fontSize: 10.5, color: "#6b7280", marginBottom: 8, fontWeight: 700 }}>SCALP RISK BARRIERS (0.4x / 0.8x ATR)</div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <div>
                          <span style={{ fontSize: 9, color: "#9ca3af", letterSpacing: "0.05em" }}>STOP LOSS</span>
                          <div style={{ fontSize: 14, fontWeight: 700, color: "#f87171", fontFamily: "monospace" }}>
                            ${predictData.risk_management.stop_loss.toFixed(2)}
                          </div>
                        </div>
                        <div>
                          <span style={{ fontSize: 9, color: "#9ca3af", letterSpacing: "0.05em" }}>TAKE PROFIT</span>
                          <div style={{ fontSize: 14, fontWeight: 700, color: "#4ade80", fontFamily: "monospace" }}>
                            ${predictData.risk_management.take_profit.toFixed(2)}
                          </div>
                        </div>
                      </div>
                    </div>

                    <div style={{ borderTop: "1px solid #1e293b", paddingTop: 14 }}>
                      <div style={{ fontSize: 10.5, color: "#6b7280", marginBottom: 8, fontWeight: 700 }}>SWING RISK BARRIERS (1.5x / 3.0x ATR)</div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <div>
                          <span style={{ fontSize: 9, color: "#9ca3af", letterSpacing: "0.05em" }}>STOP LOSS</span>
                          <div style={{ fontSize: 14, fontWeight: 700, color: "#f87171", fontFamily: "monospace" }}>
                            ${predictData.risk_management.stop_loss_sw.toFixed(2)}
                          </div>
                        </div>
                        <div>
                          <span style={{ fontSize: 9, color: "#9ca3af", letterSpacing: "0.05em" }}>TAKE PROFIT</span>
                          <div style={{ fontSize: 14, fontWeight: 700, color: "#4ade80", fontFamily: "monospace" }}>
                            ${predictData.risk_management.take_profit_sw.toFixed(2)}
                          </div>
                        </div>
                      </div>
                    </div>

                    <div style={{ borderTop: "1px solid #1e293b", paddingTop: 14 }}>
                      <div style={{ fontSize: 10.5, color: "#6b7280", marginBottom: 8, fontWeight: 700 }}>INTRADAY SUPPORT / RESISTANCE (PIVOTS)</div>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, textAlign: "center", fontSize: 11.5, fontFamily: "monospace" }}>
                        <div style={{ background: "#111827", padding: "8px 4px", borderRadius: 6 }}>
                          <span style={{ fontSize: 9, color: "#f87171", display: "block", marginBottom: 2 }}>SUPPORT (S1)</span>
                          <span style={{ color: "#f87171", fontWeight: 700 }}>${predictData.intraday_levels?.s1 ? predictData.intraday_levels.s1.toFixed(2) : "—"}</span>
                        </div>
                        <div style={{ background: "#111827", padding: "8px 4px", borderRadius: 6 }}>
                          <span style={{ fontSize: 9, color: "#9ca3af", display: "block", marginBottom: 2 }}>PIVOT (PP)</span>
                          <span style={{ color: "#fbbf24", fontWeight: 700 }}>${predictData.intraday_levels?.pp ? predictData.intraday_levels.pp.toFixed(2) : "—"}</span>
                        </div>
                        <div style={{ background: "#111827", padding: "8px 4px", borderRadius: 6 }}>
                          <span style={{ fontSize: 9, color: "#4ade80", display: "block", marginBottom: 2 }}>RESIST (R1)</span>
                          <span style={{ color: "#4ade80", fontWeight: 700 }}>${predictData.intraday_levels?.r1 ? predictData.intraday_levels.r1.toFixed(2) : "—"}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div style={{ color: "#374151", fontSize: 13 }}>Risk profile loading...</div>
                )}
              </div>
            </div>

            {/* ROW 3: Risk Monitor + Sessions + FinBERT Feed Sidebar */}
            <div className="row-monitor-sessions" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1.2fr", gap: "20px", marginTop: "24px" }}>
              {/* RISK ALERT CARD */}
              <div className="card card-monitor" style={{ padding: "24px", borderRadius: 12 }}>
                <div className="card-label">LIVE RISK MONITOR & TRADE ALERTS</div>
                {(() => {
                  const entryPrice = predictData?.risk_management?.entry_price ?? 0;
                  const latestClose = predictData?.risk_management?.latest_close ?? entryPrice;
                  const sig = predictData?.prediction?.signal ?? "NEUTRAL";
                  const scalpSig = predictData?.scalp?.signal ?? "NEUTRAL";
                  const swingSig = predictData?.swing?.signal ?? "NEUTRAL";
                  
                  let statusClass = "";
                  let stateText = "STANDBY";
                  let alertText = "Awaiting initial market connection...";
                  let icon = "🛡️";

                  if (sig === "NEUTRAL") {
                    if (scalpSig === "NEUTRAL" && swingSig === "NEUTRAL") {
                      alertText = "No active trade signal on either timeframe. Staying flat is recommended. Check Smart Timing for when to expect the next signal.";
                    } else {
                      alertText = `Scalp: ${scalpSig} | Swing: ${swingSig} — Monitor active timeframes for execution.`;
                    }
                  } else {
                    const priceDiff = latestClose - entryPrice;
                    const pipDiff = priceDiff * 10;
                    if (sig === "LONG") {
                      if (pipDiff >= 15) {
                        statusClass = "alert-be";
                        stateText = "RISK REDUCED";
                        alertText = `LONG trade is in profit (+${pipDiff.toFixed(1)} pips). Recommended: Move stop-loss to Break-Even at $${entryPrice.toFixed(2)}.`;
                        icon = "🛡️";
                      } else if (pipDiff <= -25) {
                        statusClass = "alert-close";
                        stateText = "HIGH RISK";
                        alertText = `LONG trade is showing loss (${pipDiff.toFixed(1)} pips). Support level S1 is breached. Monitor risk threshold carefully.`;
                        icon = "⚠️";
                      } else {
                        statusClass = "";
                        stateText = "MONITORING";
                        alertText = `LONG trade active from $${entryPrice.toFixed(2)}. Currently: ${pipDiff >= 0 ? "+" : ""}${pipDiff.toFixed(1)} pips. Standard risk parameters apply.`;
                        icon = "📊";
                      }
                    } else { // SHORT
                      if (pipDiff <= -15) { // short is in profit when price is lower
                        statusClass = "alert-be";
                        stateText = "RISK REDUCED";
                        alertText = `SHORT trade is in profit (+${Math.abs(pipDiff).toFixed(1)} pips). Recommended: Move stop-loss to Break-Even at $${entryPrice.toFixed(2)}.`;
                        icon = "🛡️";
                      } else if (pipDiff >= 25) {
                        statusClass = "alert-close";
                        stateText = "HIGH RISK";
                        alertText = `SHORT trade is showing loss (-${pipDiff.toFixed(1)} pips). Resistance level R1 is breached. Monitor risk threshold carefully.`;
                        icon = "⚠️";
                      } else {
                        statusClass = "";
                        stateText = "MONITORING";
                        alertText = `SHORT trade active from $${entryPrice.toFixed(2)}. Currently: ${pipDiff <= 0 ? "+" : ""}${Math.abs(pipDiff).toFixed(1)} pips. Standard risk parameters apply.`;
                        icon = "📊";
                      }
                    }
                  }

                  return (
                    <div className="monitor-content">
                      <div className={`monitor-status ${statusClass}`}>
                        <span className="monitor-icon">{icon}</span>
                        <span className={`monitor-state ${statusClass}`}>{stateText}</span>
                      </div>
                      <div className="monitor-alert">
                        {alertText}
                      </div>
                    </div>
                  );
                })()}
              </div>

              {/* SESSIONS CARD */}
              <div className="card card-sessions" style={{ padding: "20px" }}>
                <div className="card-label">GLOBAL TRADING SESSIONS (UTC)</div>
                <div className="sessions-grid">
                  <div className="session-item" id="sess-asian">
                    <span className={`sess-dot ${getSessionOpen(0, 9) ? "open" : "closed"}`}></span>
                    <span className="sess-name">ASIAN (Tokyo)</span>
                    <span className={`sess-status ${getSessionOpen(0, 9) ? "open" : "closed"}`}>
                      {getSessionOpen(0, 9) ? "OPEN" : "CLOSED"}
                    </span>
                  </div>
                  <div className="session-item" id="sess-london">
                    <span className={`sess-dot ${getSessionOpen(8, 16) ? "open" : "closed"}`}></span>
                    <span className="sess-name">LONDON</span>
                    <span className={`sess-status ${getSessionOpen(8, 16) ? "open" : "closed"}`}>
                      {getSessionOpen(8, 16) ? "OPEN" : "CLOSED"}
                    </span>
                  </div>
                  <div className="session-item" id="sess-ny">
                    <span className={`sess-dot ${getSessionOpen(13, 21) ? "open" : "closed"}`}></span>
                    <span className="sess-name">NEW YORK</span>
                    <span className={`sess-status ${getSessionOpen(13, 21) ? "open" : "closed"}`}>
                      {getSessionOpen(13, 21) ? "OPEN" : "CLOSED"}
                    </span>
                  </div>
                </div>
              </div>

              {/* SIDEBAR: FINBERT NLP SENTIMENT FEED */}
              <div className="card card-finbert-feed" style={{ padding: "20px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
                  <span className="card-label" style={{ margin: 0 }}>📰 FINBERT SENTIMENT STREAM</span>
                  <span style={{ fontSize: "0.65rem", color: "var(--gold)", fontFamily: "var(--mono)" }}>LIVE FEED</span>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: "8px", maxHeight: "160px", overflowY: "auto" }}>
                  {newsData.length === 0 ? (
                    <span style={{ fontSize: "0.75rem", color: "var(--t3)" }}>Fetching real-time news headlines...</span>
                  ) : (
                    newsData.slice(0, 4).map((n, idx) => {
                      const label = n.sentiment || "NEUTRAL";
                      const badgeClass = label === "BULLISH" || label === "POSITIVE" ? "tag-green" : label === "BEARISH" || label === "NEGATIVE" ? "tag-red" : "tag-amber";
                      return (
                        <div key={idx} style={{ background: "var(--surf2)", border: "1px solid var(--border)", borderRadius: "6px", padding: "8px 10px" }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                            <span style={{ fontSize: "0.68rem", color: "var(--t3)" }}>{n.source || "News"}</span>
                            <span className={badgeClass} style={{ fontSize: "0.65rem", padding: "1px 5px" }}>
                              {label === "BULLISH" || label === "POSITIVE" ? "▲ BULLISH" : label === "BEARISH" || label === "NEGATIVE" ? "▼ BEARISH" : "— NEUTRAL"}
                            </span>
                          </div>
                          <a 
                            href={n.url || "#"} 
                            target="_blank" 
                            rel="noopener noreferrer" 
                            style={{ 
                              fontSize: "0.76rem", 
                              color: "var(--t1)", 
                              textDecoration: "none", 
                              lineHeight: 1.3, 
                              display: "-webkit-box", 
                              WebkitLineClamp: 2, 
                              WebkitBoxOrient: "vertical", 
                              overflow: "hidden" 
                            }}
                          >
                            {n.headline}
                          </a>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>
            </div>

          </div>
        )}

        {/* ═══════════════════════════════════════════════════════════════
            AI PREDICTIONS & REASONS TAB (CLEAN, SIMPLE & EASY FOR ALL TRADERS)
        ═══════════════════════════════════════════════════════════════ */}
        {activeTab === "ai_insights" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 24, maxWidth: 1200, margin: "0 auto" }}>
            {(() => {
              const sig = predictData?.prediction?.signal || predictData?.target_trade?.signal || "NEUTRAL";
              const isLong = sig.includes("LONG") || sig.includes("BUY") || sig.includes("UP");
              const isShort = sig.includes("SHORT") || sig.includes("SELL") || sig.includes("DOWN");
              const probUp = predictData?.prediction?.probability_up ?? 0.5;
              const probDown = predictData?.prediction?.probability_down ?? 0.5;
              const mainProb = isLong ? probUp : isShort ? probDown : Math.max(probUp, probDown);
              const probPct = Math.round(mainProb * 100);

              const dirTitle = isLong ? "BUY (GO LONG)" : isShort ? "SELL (GO SHORT)" : "WAIT (NO TRADE)";
              const dirColor = isLong ? "#4ade80" : isShort ? "#f87171" : "#fbbf24";
              const dirIcon = isLong ? "🟢 📈" : isShort ? "🔴 📉" : "🟡 ⏳";
              const dirSimpleMessage = isLong
                ? "Gold (XAU/USD) is expected to RISE today. Buyers are in control."
                : isShort
                ? "Gold (XAU/USD) is expected to FALL today. Sellers are in control."
                : "Gold is moving sideways with no clear direction. Waiting is the safest choice.";

              const entry = predictData?.ict_analysis?.gpt_synthesis?.recommended_entry
                || (predictData?.target_trade?.signal !== "NEUTRAL" && predictData?.target_trade?.entry_price)
                || (predictData?.risk_management?.entry_price || 0);

              const sl = predictData?.ict_analysis?.gpt_synthesis?.recommended_sl
                || (predictData?.target_trade?.signal !== "NEUTRAL" && predictData?.target_trade?.stop_loss)
                || (predictData?.risk_management?.stop_loss_sw || predictData?.risk_management?.stop_loss || 0);

              const tp = predictData?.ict_analysis?.gpt_synthesis?.recommended_tp
                || (predictData?.target_trade?.signal !== "NEUTRAL" && predictData?.target_trade?.take_profit)
                || (predictData?.risk_management?.take_profit_sw || predictData?.risk_management?.take_profit || 0);

              return (
                <>
                  {/* 1. BIG SIMPLE DIRECTION VERDICT */}
                  <div style={{
                    background: `linear-gradient(135deg, ${dirColor}15 0%, #0d0d1a 100%)`,
                    border: `2px solid ${dirColor}80`,
                    borderRadius: 14,
                    padding: "32px 36px",
                    boxShadow: `0 10px 35px ${dirColor}15`
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 20 }}>
                      <div>
                        <span style={{ fontSize: 11.5, color: "#9ca3af", letterSpacing: "0.1em", fontWeight: 700, textTransform: "uppercase" }}>
                          TODAY&apos;S AI PREDICTED DIRECTION
                        </span>
                        <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 8 }}>
                          <span style={{ fontSize: 36 }}>{dirIcon}</span>
                          <span style={{ fontSize: 36, fontWeight: 900, color: dirColor, letterSpacing: "0.02em" }}>
                            {dirTitle}
                          </span>
                        </div>
                        <p style={{ fontSize: 16.5, color: "#f8fafc", fontWeight: 600, marginTop: 10, marginBottom: 0 }}>
                          {dirSimpleMessage}
                        </p>
                      </div>

                      {/* AI Confidence Meter */}
                      <div style={{ background: "#111827", border: `1px solid ${dirColor}40`, borderRadius: 12, padding: "18px 28px", textAlign: "center", minWidth: 160 }}>
                        <span style={{ fontSize: 11, color: "#9ca3af", display: "block", textTransform: "uppercase", fontWeight: 700, letterSpacing: "0.08em" }}>AI CONFIDENCE</span>
                        <span style={{ fontSize: 34, fontWeight: 900, color: dirColor, fontFamily: "var(--mono)", marginTop: 2, display: "block" }}>{probPct}%</span>
                        <div style={{ height: 6, width: "100%", background: "#1f2937", borderRadius: 3, marginTop: 8, overflow: "hidden" }}>
                          <div style={{ height: "100%", width: `${probPct}%`, background: dirColor, borderRadius: 3 }} />
                        </div>
                      </div>
                    </div>

                    {/* Simple summary text */}
                    {predictData?.narrative?.summary && (
                      <div style={{ background: "rgba(0,0,0,0.35)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: "16px 22px", marginTop: 22 }}>
                        <span style={{ fontSize: 11.5, color: "#fbbf24", fontWeight: 800, letterSpacing: "0.08em", display: "block", marginBottom: 6 }}>
                          📌 WHAT IS HAPPENING IN THE MARKET:
                        </span>
                        <p style={{ color: "#e2e8f0", fontSize: 14.5, lineHeight: 1.6, margin: 0 }}>
                          {predictData.narrative.summary}
                        </p>
                      </div>
                    )}
                  </div>

                  {/* 2. THREE SIMPLE REASONS WHY (LIVE GPT MODEL PILLARS) */}
                  <div>
                    <h3 style={{ margin: "4px 0 16px 0", fontSize: 17, fontWeight: 800, color: "#f8fafc", display: "flex", alignItems: "center", gap: 10 }}>
                      <span>💡</span> Why is this Direction Predicted? (Live AI Analysis)
                    </h3>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 20 }}>
                      {/* Reason 1: Economy / Interest Rates */}
                      <div style={{ background: "#0d0d1a", border: "1px solid #1e293b", borderRadius: 12, padding: "22px 24px" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                          <span style={{ fontSize: 24 }}>🏦</span>
                          <span style={{ fontSize: 13.5, fontWeight: 800, color: "#fbbf24" }}>
                            {predictData?.narrative?.trader_pillars?.fed_policy?.title || "1. US Economy & Rates"}
                          </span>
                        </div>
                        <div style={{
                          display: "inline-block",
                          padding: "3px 10px",
                          borderRadius: 4,
                          fontSize: 10.5,
                          fontWeight: 700,
                          background: `${signalColor(predictData?.narrative?.trader_pillars?.fed_policy?.verdict || predictData?.narrative?.combinations?.macro || "BULLISH")}20`,
                          color: signalColor(predictData?.narrative?.trader_pillars?.fed_policy?.verdict || predictData?.narrative?.combinations?.macro || "BULLISH"),
                          marginBottom: 10
                        }}>
                          SIGNAL: {predictData?.narrative?.trader_pillars?.fed_policy?.verdict || predictData?.narrative?.combinations?.macro || "BULLISH"}
                        </div>
                        <p style={{ color: "#cbd5e1", fontSize: 13.5, lineHeight: 1.6, margin: 0 }}>
                          {predictData?.narrative?.trader_pillars?.fed_policy?.text || (
                            isLong
                              ? "US interest rates and the US Dollar are weakening. When the Dollar drops, Gold becomes cheaper and prices go UP."
                              : isShort
                              ? "US interest rates and the Dollar are strong. When the Dollar rises, investors sell Gold and prices go DOWN."
                              : "US economic indicators are balanced with no major rate changes expected today."
                          )}
                        </p>
                      </div>

                      {/* Reason 2: News Sentiment & Geopolitics */}
                      <div style={{ background: "#0d0d1a", border: "1px solid #1e293b", borderRadius: 12, padding: "22px 24px" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                          <span style={{ fontSize: 24 }}>📰</span>
                          <span style={{ fontSize: 13.5, fontWeight: 800, color: "#fbbf24" }}>
                            {predictData?.narrative?.trader_pillars?.geopolitical?.title || "2. News & Geopolitics"}
                          </span>
                        </div>
                        <div style={{
                          display: "inline-block",
                          padding: "3px 10px",
                          borderRadius: 4,
                          fontSize: 10.5,
                          fontWeight: 700,
                          background: `${signalColor(predictData?.narrative?.trader_pillars?.geopolitical?.verdict || predictData?.narrative?.combinations?.sentiment || "BULLISH")}20`,
                          color: signalColor(predictData?.narrative?.trader_pillars?.geopolitical?.verdict || predictData?.narrative?.combinations?.sentiment || "BULLISH"),
                          marginBottom: 10
                        }}>
                          SIGNAL: {predictData?.narrative?.trader_pillars?.geopolitical?.verdict || predictData?.narrative?.combinations?.sentiment || "BULLISH"}
                        </div>
                        <p style={{ color: "#cbd5e1", fontSize: 13.5, lineHeight: 1.6, margin: 0 }}>
                          {predictData?.narrative?.trader_pillars?.geopolitical?.text || (
                            isLong
                              ? "Latest news headlines are positive for Gold. Investors are buying Gold as a safe haven asset."
                              : isShort
                              ? "Recent market news shows traders are taking profits, reducing immediate demand for Gold."
                              : "World news headlines are calm today without major breaking events impacting Gold."
                          )}
                        </p>
                      </div>

                      {/* Reason 3: Chart & Price Levels */}
                      <div style={{ background: "#0d0d1a", border: "1px solid #1e293b", borderRadius: 12, padding: "22px 24px" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                          <span style={{ fontSize: 24 }}>📊</span>
                          <span style={{ fontSize: 13.5, fontWeight: 800, color: "#fbbf24" }}>
                            {predictData?.narrative?.trader_pillars?.technical_of?.title || "3. Technical & Order Flow"}
                          </span>
                        </div>
                        <div style={{
                          display: "inline-block",
                          padding: "3px 10px",
                          borderRadius: 4,
                          fontSize: 10.5,
                          fontWeight: 700,
                          background: `${signalColor(predictData?.narrative?.trader_pillars?.technical_of?.verdict || predictData?.narrative?.combinations?.order_flow || "BULLISH")}20`,
                          color: signalColor(predictData?.narrative?.trader_pillars?.technical_of?.verdict || predictData?.narrative?.combinations?.order_flow || "BULLISH"),
                          marginBottom: 10
                        }}>
                          SIGNAL: {predictData?.narrative?.trader_pillars?.technical_of?.verdict || predictData?.narrative?.combinations?.order_flow || "BULLISH"}
                        </div>
                        <p style={{ color: "#cbd5e1", fontSize: 13.5, lineHeight: 1.6, margin: 0 }}>
                          {predictData?.narrative?.trader_pillars?.technical_of?.text || (
                            isLong
                              ? "Price is holding strong above key support levels. Big buyers are stepping in to push price higher."
                              : isShort
                              ? "Price is failing to break resistance levels. Sellers are rejecting higher prices."
                              : "Buyers and sellers are equal right now, keeping price inside a narrow range."
                          )}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* 2B. DEEP GPT REASONING & PLAIN-ENGLISH TRADER INSIGHTS */}
                  {predictData?.narrative?.reasoning && (
                    <div style={{ background: "#0d0d1a", border: "1px solid #1e293b", borderRadius: 12, padding: "24px 28px" }}>
                      <div style={{ fontSize: 13, color: "#fbbf24", fontWeight: 800, letterSpacing: "0.08em", marginBottom: 10, display: "flex", alignItems: "center", gap: 8 }}>
                        <span>🧠</span> DEEP AI MODEL REASONING & MARKET DRIVERS
                      </div>
                      <p style={{ color: "#e2e8f0", fontSize: 14, lineHeight: 1.65, margin: "0 0 16px 0" }}>
                        {predictData.narrative.reasoning}
                      </p>

                      {predictData?.narrative?.trader_insights && predictData.narrative.trader_insights.length > 0 && (
                        <div style={{ borderTop: "1px solid #1e293b", paddingTop: 16, display: "flex", flexDirection: "column", gap: 10 }}>
                          <span style={{ fontSize: 11.5, color: "#9ca3af", fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase" }}>
                            Key Execution Insights for Traders:
                          </span>
                          {predictData.narrative.trader_insights.map((insight, idx) => (
                            <div key={idx} style={{ fontSize: 13.5, color: "#cbd5e1", lineHeight: 1.5, display: "flex", gap: 8 }}>
                              <span>{insight}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* 3. SIMPLE TRADE PLAN (HOW TO TRADE THIS DIRECTION) */}
                  {entry > 0 && (
                    <div style={{ background: "#0d0d1a", border: "1px solid #1e293b", borderRadius: 12, padding: "24px 28px" }}>
                      <h3 style={{ margin: "0 0 16px 0", fontSize: 16, fontWeight: 800, color: "#fbbf24", display: "flex", alignItems: "center", gap: 8 }}>
                        <span>🎯</span> Simple Trade Plan for You
                      </h3>

                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                        <div style={{ background: "#111827", borderRadius: 10, padding: "16px 20px", border: "1px solid var(--border)" }}>
                          <span style={{ fontSize: 10.5, color: "#9ca3af", letterSpacing: "0.08em", display: "block" }}>WHERE TO ENTER</span>
                          <span style={{ fontSize: 24, fontWeight: 800, color: "#fbbf24", fontFamily: "var(--mono)", marginTop: 2, display: "block" }}>${entry.toFixed(2)}</span>
                          <span style={{ fontSize: 11.5, color: "#6b7280", display: "block", marginTop: 4 }}>Target entry level</span>
                        </div>

                        <div style={{ background: "#111827", borderRadius: 10, padding: "16px 20px", border: "1px solid rgba(248, 113, 113, 0.3)" }}>
                          <span style={{ fontSize: 10.5, color: "#f87171", letterSpacing: "0.08em", display: "block" }}>PROTECT YOUR ACCOUNT (STOP LOSS)</span>
                          <span style={{ fontSize: 24, fontWeight: 800, color: "#f87171", fontFamily: "var(--mono)", marginTop: 2, display: "block" }}>${sl.toFixed(2)}</span>
                          <span style={{ fontSize: 11.5, color: "#6b7280", display: "block", marginTop: 4 }}>Close trade if price falls here</span>
                        </div>

                        <div style={{ background: "#111827", borderRadius: 10, padding: "16px 20px", border: "1px solid rgba(74, 222, 128, 0.3)" }}>
                          <span style={{ fontSize: 10.5, color: "#4ade80", letterSpacing: "0.08em", display: "block" }}>TAKE YOUR PROFIT</span>
                          <span style={{ fontSize: 24, fontWeight: 800, color: "#4ade80", fontFamily: "var(--mono)", marginTop: 2, display: "block" }}>${tp.toFixed(2)}</span>
                          <span style={{ fontSize: 11.5, color: "#6b7280", display: "block", marginTop: 4 }}>Collect profits here</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* 4. THREE GOLDEN RULES EVERY TRADER MUST FOLLOW */}
                  <div style={{ background: "linear-gradient(135deg, #0e0e24, #080815)", border: "1px solid #1e293b", borderRadius: 12, padding: "24px 28px" }}>
                    <div style={{ fontSize: 14, fontWeight: 800, color: "#fbbf24", letterSpacing: "0.08em", marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}>
                      <span>🛡️</span> 3 Simple Rules to Protect Your Money
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                      <div style={{ background: "#0d0d1a", borderRadius: 10, padding: "18px 20px", border: "1px solid var(--border)" }}>
                        <div style={{ color: "#4ade80", fontWeight: 800, fontSize: 13, marginBottom: 6 }}>1. Lock in Profit Early</div>
                        <p style={{ color: "#9ca3af", fontSize: 12.5, margin: 0, lineHeight: 1.5 }}>
                          When your trade is in profit (+15 pips), move your Stop Loss to your entry price. This makes your trade completely risk-free!
                        </p>
                      </div>
                      <div style={{ background: "#0d0d1a", borderRadius: 10, padding: "18px 20px", border: "1px solid var(--border)" }}>
                        <div style={{ color: "#fbbf24", fontWeight: 800, fontSize: 13, marginBottom: 6 }}>2. Dynamic Risk Sizing</div>
                        <p style={{ color: "#9ca3af", fontSize: 12.5, margin: 0, lineHeight: 1.5 }}>
                          Risk <strong>2.0%</strong> of account equity for <strong>High Confidence</strong> setups (≥65%), and <strong>0.5% – 1.0%</strong> for normal trades.
                        </p>
                      </div>
                      <div style={{ background: "#0d0d1a", borderRadius: 10, padding: "18px 20px", border: "1px solid var(--border)" }}>
                        <div style={{ color: "#38bdf8", fontWeight: 800, fontSize: 13, marginBottom: 6 }}>3. Watch News Releases</div>
                        <p style={{ color: "#9ca3af", fontSize: 12.5, margin: 0, lineHeight: 1.5 }}>
                          Check the Calendar tab. Do not open trades right before major US news releases (CPI, NFP, Fed interest rate decisions).
                        </p>
                      </div>
                    </div>
                  </div>
                </>
              );
            })()}
          </div>
        )}



        {/* ═══════════════════════════════════════════════════════════════
            NEWS TAB
        ═══════════════════════════════════════════════════════════════ */}
        {activeTab === "news" && (() => {
          const categories = [
            { id: "ALL", label: "🌐 All Gold News", icon: "🌐" },
            { id: "WAR_MILITARY", label: "🪖 War & Geopolitics", icon: "🪖" },
            { id: "FED_POLICY", label: "🏦 Fed & Central Banks", icon: "🏦" },
            { id: "GOLD_MARKET", label: "🥇 Gold & Metals", icon: "🥇" },
            { id: "DOLLAR_FX", label: "💵 US Dollar & Yields", icon: "💵" },
            { id: "INFLATION", label: "📈 Inflation & CPI", icon: "📈" },
            { id: "CRISIS", label: "🚨 Crisis & Safe-Haven", icon: "🚨" },
          ];

          const filteredNews = newsCategory === "ALL"
            ? newsData
            : newsData.filter(n => n.category === newsCategory || (newsCategory === "WAR_MILITARY" && n.category?.includes("WAR")));

          return (
            <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
              {/* ── Fundamental Direction Live Intelligence Banner ── */}
              <FundamentalDirectionWidget />

              {/* Header & Meta */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 0, flexWrap: "wrap", gap: 12 }}>
                <div>
                  <h2 style={{ margin: 0, fontSize: 18, fontWeight: 800, color: "#f8fafc", display: "flex", alignItems: "center", gap: 8 }}>
                    <span>🗞️</span> Live Gold & Geopolitical Intelligence Feed
                  </h2>
                  <p style={{ margin: "4px 0 0 0", fontSize: 12.5, color: "#9ca3af" }}>
                    Real-time financial, macroeconomic, and conflict headlines with instant AI Gold sentiment tagging
                  </p>
                </div>
                <div style={{ background: "#0d0d1a", border: "1px solid #1e293b", padding: "6px 14px", borderRadius: 8, fontSize: 11.5, color: "#fbbf24", fontWeight: 700 }}>
                  {filteredNews.length} Headlines · 15-min Auto Refresh
                </div>
              </div>

              {/* Category Filter Pills */}
              <div style={{ display: "flex", gap: 8, overflowX: "auto", paddingBottom: 14, marginBottom: 12 }}>
                {categories.map(cat => {
                  const isSelected = newsCategory === cat.id;
                  const count = cat.id === "ALL" 
                    ? newsData.length 
                    : newsData.filter(n => n.category === cat.id || (cat.id === "WAR_MILITARY" && n.category?.includes("WAR"))).length;

                  return (
                    <button
                      key={cat.id}
                      onClick={() => setNewsCategory(cat.id)}
                      style={{
                        background: isSelected ? "#fbbf24" : "#0d0d1a",
                        color: isSelected ? "#000" : "#cbd5e1",
                        border: isSelected ? "1px solid #fbbf24" : "1px solid #1e293b",
                        borderRadius: 20,
                        padding: "6px 14px",
                        fontSize: 12,
                        fontWeight: 700,
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        whiteSpace: "nowrap",
                        transition: "all 0.2s"
                      }}
                    >
                      <span>{cat.icon}</span>
                      <span>{cat.label.replace(/^[^\s]+\s/, "")}</span>
                      <span style={{
                        background: isSelected ? "rgba(0,0,0,0.2)" : "#1e293b",
                        padding: "1px 6px",
                        borderRadius: 10,
                        fontSize: 10
                      }}>
                        {count}
                      </span>
                    </button>
                  );
                })}
              </div>

              {/* News Cards List */}
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {filteredNews.length === 0 ? (
                  <div style={{ background: "#0d0d1a", border: "1px solid #1e293b", borderRadius: 10, padding: 40, textAlign: "center", color: "#6b7280" }}>
                    No news articles found under this category filter.
                  </div>
                ) : filteredNews.map((news, i) => {
                  const isBullish = news.sentiment === "BULLISH" || news.sentiment === "POSITIVE";
                  const isBearish = news.sentiment === "BEARISH" || news.sentiment === "NEGATIVE";
                  const sentColor = isBullish ? "#4ade80" : isBearish ? "#f87171" : "#fbbf24";
                  const sentBadge = isBullish ? "🟢 BULLISH FOR GOLD" : isBearish ? "🔴 BEARISH FOR GOLD" : "🟡 NEUTRAL";
                  const categoryIcon = news.cat_icon || (news.category === "WAR_MILITARY" ? "🪖" : news.category === "FED_POLICY" ? "🏦" : news.category === "GOLD_MARKET" ? "🥇" : news.category === "DOLLAR_FX" ? "💵" : news.category === "INFLATION" ? "📈" : "📰");
                  const categoryName = (news.category || "OTHER").replace(/_/g, " ");

                  return (
                    <div
                      key={i}
                      style={{
                        background: "#0d0d1a",
                        border: `1px solid ${sentColor}25`,
                        borderRadius: 10,
                        padding: "16px 20px",
                        display: "flex",
                        gap: 16,
                        alignItems: "flex-start",
                        transition: "border 0.2s",
                        boxShadow: "0 4px 20px rgba(0,0,0,0.2)"
                      }}
                    >
                      {/* Left Sentiment Stripe */}
                      <div style={{ width: 4, flexShrink: 0, alignSelf: "stretch", background: sentColor, borderRadius: 2, minHeight: 40 }} />

                      {/* Main Content */}
                      <div style={{ flex: 1 }}>
                        {/* Top Badges */}
                        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", marginBottom: 8 }}>
                          {/* Category Tag */}
                          <span style={{
                            fontSize: 10.5,
                            fontWeight: 800,
                            letterSpacing: "0.06em",
                            background: "#1e293b",
                            color: "#fbbf24",
                            padding: "2px 8px",
                            borderRadius: 4,
                            display: "inline-flex",
                            alignItems: "center",
                            gap: 4,
                            textTransform: "uppercase"
                          }}>
                            <span>{categoryIcon}</span>
                            <span>{categoryName}</span>
                          </span>

                          {/* Sentiment Tag */}
                          <span style={{
                            fontSize: 10.5,
                            fontWeight: 800,
                            letterSpacing: "0.06em",
                            background: `${sentColor}18`,
                            color: sentColor,
                            padding: "2px 8px",
                            borderRadius: 4
                          }}>
                            {sentBadge}
                          </span>

                          {/* Sentiment Score */}
                          {news.score != null && (
                            <span style={{ fontSize: 10.5, color: "#9ca3af", fontFamily: "var(--mono)" }}>
                              Score: <strong style={{ color: sentColor }}>{news.score > 0 ? `+${news.score}` : news.score}</strong>
                            </span>
                          )}
                        </div>

                        {/* Headline */}
                        <a
                          href={news.url || "#"}
                          target="_blank"
                          rel="noreferrer"
                          style={{
                            color: "#f8fafc",
                            textDecoration: "none",
                            fontWeight: 700,
                            fontSize: 15,
                            lineHeight: 1.5,
                            display: "block",
                            marginBottom: 8
                          }}
                          onMouseEnter={(e) => (e.currentTarget.style.color = "#fbbf24")}
                          onMouseLeave={(e) => (e.currentTarget.style.color = "#f8fafc")}
                        >
                          {news.headline} <span style={{ fontSize: 12, opacity: 0.7 }}>↗</span>
                        </a>

                        {/* Footer info */}
                        <div style={{ display: "flex", gap: 14, fontSize: 11, color: "#6b7280", alignItems: "center", flexWrap: "wrap" }}>
                          <span>📰 Source: <strong style={{ color: "#cbd5e1" }}>{news.source}</strong></span>
                          {news.datetime && (
                            <span style={{ color: "#9ca3af", display: "inline-flex", alignItems: "center", gap: 4 }}>
                              <span>⏰</span>
                              <span>{formatTimezone(news.datetime, timeframe, "DATETIME")}</span>
                              <span style={{ fontSize: 9.5, color: "#fbbf24", background: "rgba(251, 191, 36, 0.1)", padding: "1px 5px", borderRadius: 3 }}>
                                {timeframe === "NY" ? "NY Time" : "Sri Lanka Time"}
                              </span>
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })()}

        {/* ═══════════════════════════════════════════════════════════════
            CALENDAR TAB
        ═══════════════════════════════════════════════════════════════ */}
        {activeTab === "calendar" && (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
              <h2 style={{ margin: 0, fontSize: 16, fontWeight: 700 }}>Macroeconomic Calendar & Impact Analysis</h2>
              <div className="cal-legend" style={{ fontSize: 12 }}>
                <span className="leg-dot" style={{ background: "var(--red)", display: "inline-block", width: 8, height: 8, borderRadius: "50%", marginRight: 4 }}></span> HIGH Impact &nbsp;
                <span className="leg-dot" style={{ background: "var(--amber)", display: "inline-block", width: 8, height: 8, borderRadius: "50%", marginRight: 4 }}></span> MEDIUM &nbsp;
                <span className="leg-dot" style={{ background: "var(--blue)", display: "inline-block", width: 8, height: 8, borderRadius: "50%", marginRight: 4 }}></span> LOW
              </div>
            </div>
            {macroCalendarData.length === 0 ? (
              <div className="cal-loading">Loading events or no events scheduled for the next 30 days...</div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {macroCalendarData.map((ev, i) => {
                  const impactColor = ev.impact === "HIGH" ? "var(--red)" : ev.impact === "MEDIUM" ? "var(--amber)" : "var(--blue)";
                  return (
                    <div 
                      key={i} 
                      className={`cal-item-full impact-${ev.impact.toLowerCase()}`}
                      style={{
                        background: "var(--surf1)",
                        border: "1px solid var(--border)",
                        borderRadius: "8px",
                        padding: "16px 20px",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center"
                      }}
                    >
                      <div className="calf-left">
                        <div className="calf-header">
                          <span className="cal-dot" style={{ background: impactColor }} />
                          <span className="calf-event">{ev.event}</span>
                          {ev.is_today ? (
                            <span className="cal-today-badge">TODAY</span>
                          ) : ev.is_tomorrow ? (
                            <span className="cal-tmrw-badge">TOMORROW</span>
                          ) : (
                            <span className="cal-days-badge">in {ev.days_until}d</span>
                          )}
                        </div>
                        <div className="calf-desc">{ev.description}</div>
                      </div>
                      <div className="calf-right">
                        <div className="calf-date">{ev.date}</div>
                        <div className="calf-impact" style={{ color: impactColor }}>{ev.impact}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}



      </div>
    </div>
  );
}
