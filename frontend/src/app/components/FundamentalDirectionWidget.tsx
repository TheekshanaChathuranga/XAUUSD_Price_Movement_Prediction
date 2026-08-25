"use client";

/**
 * FundamentalDirectionWidget
 * ===========================
 * Persistent live widget showing the current fundamental (news-driven)
 * directional bias for gold: BULLISH / BEARISH / NEUTRAL.
 *
 * Data flow:
 *   1. On mount: GET /api/fundamental-direction (initial page-load state)
 *   2. Connect WebSocket ws://localhost:8000/ws/fundamental-direction
 *   3. WS messages update direction, confidence, headlines in-place (no reload)
 *   4. If WS drops: exponential backoff reconnect (1s -> 2s -> 4s -> ... -> 30s max)
 *   5. Fallback REST poll every 30s while WS is disconnected
 *
 * Visual features:
 *   - Color-coded badge: green (BULLISH), red (BEARISH), gray (NEUTRAL)
 *   - Directional arrow with entrance animation
 *   - Confidence bar
 *   - Top 2-3 contributing headlines with category icon + score
 *   - "Last updated Xs ago" live counter
 *   - Connection status: LIVE / RECONNECTING / OFFLINE
 *   - Flash animation on direction change
 */

import React, { useEffect, useRef, useState, useCallback } from "react";

// ── Types ─────────────────────────────────────────────────────────────────────
interface Headline {
  headline: string;
  category: string;
  score: number;
}

interface FundamentalState {
  direction: "BULLISH" | "BEARISH" | "NEUTRAL";
  confidence: number;
  top_headlines: Headline[];
  computed_at: string | null;
  trigger: string;
  news_count: number;
}

type ConnectionStatus = "connecting" | "live" | "reconnecting" | "offline";

// ── Constants ─────────────────────────────────────────────────────────────────
const API_BASE   = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
const WS_BASE    = API_BASE.replace(/^http/, "ws");
const POLL_MS    = 30_000;   // REST fallback polling interval
const MAX_BACKOFF = 30_000;  // max reconnect wait

const CATEGORY_ICONS: Record<string, string> = {
  WAR_MILITARY: "🪖",
  FED_POLICY:   "🏦",
  INFLATION:    "📈",
  DOLLAR_FX:    "💵",
  CRISIS:       "🚨",
  ENERGY:       "🛢️",
  GOLD_MARKET:  "🥇",
  OTHER:        "📰",
};

// ── Helpers ───────────────────────────────────────────────────────────────────
function secondsAgo(isoStr: string | null): string {
  if (!isoStr) return "—";
  const dt = new Date(isoStr.endsWith("Z") ? isoStr : isoStr + "Z");
  const diff = Math.floor((Date.now() - dt.getTime()) / 1000);
  if (diff < 60)   return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

function directionColor(d: string): string {
  if (d === "BULLISH") return "#4ade80";
  if (d === "BEARISH") return "#f87171";
  return "#9ca3af";
}

function directionGlow(d: string): string {
  if (d === "BULLISH") return "0 0 18px #4ade8055, 0 0 6px #4ade8033";
  if (d === "BEARISH") return "0 0 18px #f8717155, 0 0 6px #f8717133";
  return "none";
}

function directionBg(d: string): string {
  if (d === "BULLISH") return "linear-gradient(135deg,#052e16 0%,#0d1f0d 100%)";
  if (d === "BEARISH") return "linear-gradient(135deg,#2d0a0a 0%,#1a0808 100%)";
  return "linear-gradient(135deg,#111827 0%,#0f1723 100%)";
}

function arrowIcon(d: string): string {
  if (d === "BULLISH") return "↑";
  if (d === "BEARISH") return "↓";
  return "→";
}

// ── Main Component ────────────────────────────────────────────────────────────
export default function FundamentalDirectionWidget() {
  const [state, setState] = useState<FundamentalState>({
    direction:     "NEUTRAL",
    confidence:    0,
    top_headlines: [],
    computed_at:   null,
    trigger:       "startup",
    news_count:    0,
  });
  const [connStatus, setConnStatus]   = useState<ConnectionStatus>("connecting");
  const [flashKey, setFlashKey]       = useState(0);      // incremented on direction change
  const [ageStr,  setAgeStr]          = useState("—");
  const prevDirection                 = useRef("NEUTRAL");

  const wsRef        = useRef<WebSocket | null>(null);
  const backoffRef   = useRef(1000);
  const reconnTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollTimerRef   = useRef<ReturnType<typeof setInterval> | null>(null);
  const ageTimerRef    = useRef<ReturnType<typeof setInterval> | null>(null);
  const mountedRef     = useRef(true);

  // ── Fetch REST state ───────────────────────────────────────────────────────
  const fetchRest = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/fundamental-direction`);
      if (!res.ok) return;
      const data = await res.json();
      if (!mountedRef.current) return;
      updateState(data);
    } catch { /* ignore */ }
  }, []);

  // ── Apply incoming data ────────────────────────────────────────────────────
  function updateState(data: Partial<FundamentalState>) {
    if (!mountedRef.current) return;
    setState(prev => {
      const next: FundamentalState = {
        direction:     (data.direction     ?? prev.direction) as FundamentalState["direction"],
        confidence:    data.confidence     ?? prev.confidence,
        top_headlines: data.top_headlines  ?? prev.top_headlines,
        computed_at:   data.computed_at    ?? prev.computed_at,
        trigger:       data.trigger        ?? prev.trigger,
        news_count:    data.news_count     ?? prev.news_count,
      };
      // Trigger flash if direction changed
      if (next.direction !== prevDirection.current) {
        prevDirection.current = next.direction;
        setFlashKey(k => k + 1);
      }
      return next;
    });
  }

  // ── WebSocket connection ───────────────────────────────────────────────────
  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    setConnStatus("connecting");

    const ws = new WebSocket(`${WS_BASE}/ws/fundamental-direction`);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) return;
      backoffRef.current = 1000;          // reset backoff on success
      setConnStatus("live");
      stopPoll();                          // WS is live, stop REST polling
    };

    ws.onmessage = (evt) => {
      if (!mountedRef.current) return;
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === "direction_update") {
          updateState(msg as FundamentalState);
        }
        // "ping" messages are silently ignored
      } catch { /* ignore malformed messages */ }
    };

    ws.onerror = () => {
      if (!mountedRef.current) return;
      setConnStatus("reconnecting");
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setConnStatus("reconnecting");
      wsRef.current = null;
      startPoll();        // fallback while WS is down
      scheduleReconnect();
    };
  }, [fetchRest]);

  function scheduleReconnect() {
    if (!mountedRef.current) return;
    if (reconnTimerRef.current) clearTimeout(reconnTimerRef.current);
    const delay = Math.min(backoffRef.current, MAX_BACKOFF);
    backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF);
    reconnTimerRef.current = setTimeout(() => {
      if (mountedRef.current) connect();
    }, delay);
  }

  function startPoll() {
    if (pollTimerRef.current) return;
    fetchRest();   // immediate poll
    pollTimerRef.current = setInterval(fetchRest, POLL_MS);
  }

  function stopPoll() {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────────
  useEffect(() => {
    mountedRef.current = true;

    // Initial REST load before WS connects
    fetchRest();

    // Connect WS
    connect();

    // Live "Xs ago" counter
    ageTimerRef.current = setInterval(() => {
      setState(prev => {
        setAgeStr(secondsAgo(prev.computed_at));
        return prev;
      });
    }, 1000);

    return () => {
      mountedRef.current = false;
      wsRef.current?.close();
      if (reconnTimerRef.current) clearTimeout(reconnTimerRef.current);
      stopPoll();
      if (ageTimerRef.current) clearInterval(ageTimerRef.current);
    };
  }, []);

  // ── Derived values ─────────────────────────────────────────────────────────
  const color  = directionColor(state.direction);
  const border = `1px solid ${color}44`;
  const bg     = directionBg(state.direction);
  const glow   = directionGlow(state.direction);

  const statusDot: Record<ConnectionStatus, { color: string; label: string }> = {
    live:         { color: "#4ade80", label: "LIVE" },
    connecting:   { color: "#fbbf24", label: "CONNECTING" },
    reconnecting: { color: "#fbbf24", label: "RECONNECTING" },
    offline:      { color: "#f87171", label: "OFFLINE" },
  };
  const dot = statusDot[connStatus];

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <>
      {/* Flash keyframe injected once */}
      <style>{`
        @keyframes fd-flash {
          0%   { opacity: 1; }
          15%  { opacity: 0.4; }
          30%  { opacity: 1; }
          50%  { opacity: 0.6; }
          100% { opacity: 1; }
        }
        @keyframes fd-arrow-bounce {
          0%, 100% { transform: translateY(0); }
          50%       { transform: translateY(-4px); }
        }
        .fd-flash { animation: fd-flash 0.7s ease; }
        .fd-arrow-live { animation: fd-arrow-bounce 1.5s ease-in-out infinite; }
      `}</style>

      <div
        key={flashKey}
        className={flashKey > 0 ? "fd-flash" : ""}
        style={{
          background: bg,
          border,
          borderRadius: 12,
          padding: "20px 24px",
          marginBottom: 20,
          boxShadow: glow,
          transition: "background 0.6s ease, box-shadow 0.6s ease",
        }}
        id="fundamental-direction-widget"
        aria-label="Fundamental Direction Indicator"
      >
        {/* ── Header row ─────────────────────────────────────────────────── */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 11, color: "#fbbf24", letterSpacing: "0.12em", fontWeight: 800, textTransform: "uppercase" }}>
              🥇 LIVE NEWS-DRIVEN FUNDAMENTAL BIAS
            </span>
            {state.trigger === "high_impact" && (
              <span style={{ fontSize: 9.5, background: "#7c3aed22", color: "#a78bfa",
                             border: "1px solid #7c3aed44", padding: "2px 8px", borderRadius: 4,
                             letterSpacing: "0.08em", fontWeight: 700 }}>
                ⚡ HIGH IMPACT NEWS
              </span>
            )}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            {/* Connection status */}
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <div style={{
                width: 7, height: 7, borderRadius: "50%", background: dot.color,
                boxShadow: `0 0 8px ${dot.color}`,
              }} />
              <span style={{ fontSize: 10, color: dot.color, letterSpacing: "0.08em", fontWeight: 700 }}>
                {dot.label}
              </span>
            </div>
            {/* Last updated */}
            <span style={{ fontSize: 11, color: "#6b7280" }}>
              {ageStr}
            </span>
          </div>
        </div>

        {/* ── Main direction + confidence ────────────────────────────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "180px 1fr", gap: 24, alignItems: "center" }}>
          {/* Direction badge */}
          <div style={{ textAlign: "center", background: "#0a0a16", border: `1px solid ${color}30`, borderRadius: 10, padding: "16px 12px" }}>
            <div
              className={connStatus === "live" ? "fd-arrow-live" : ""}
              style={{
                fontSize: 44,
                fontWeight: 900,
                color,
                lineHeight: 1,
                fontFamily: "monospace",
                textShadow: glow ? `0 0 20px ${color}` : "none",
                transition: "color 0.6s ease",
              }}
            >
              {arrowIcon(state.direction)}
            </div>
            <div style={{
              fontSize: 14, fontWeight: 900, color, marginTop: 6,
              letterSpacing: "0.08em", transition: "color 0.6s ease",
            }}>
              {state.direction === "BULLISH" ? "BULLISH (BUY)" : state.direction === "BEARISH" ? "BEARISH (SELL)" : "NEUTRAL (HOLD)"}
            </div>
            <span style={{ fontSize: 9.5, color: "#9ca3af", display: "block", marginTop: 4, fontWeight: 600 }}>
              FOR GOLD (XAU/USD)
            </span>
          </div>

          {/* Right column: confidence + headlines */}
          <div>
            {/* Confidence bar & description */}
            <div style={{ marginBottom: 14 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                <span style={{ fontSize: 10.5, color: "#9ca3af", letterSpacing: "0.08em", fontWeight: 700 }}>
                  FUNDAMENTAL SENTIMENT CONFIDENCE
                </span>
                <span style={{ fontSize: 12, color, fontFamily: "monospace", fontWeight: 800 }}>
                  {(state.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <div style={{ height: 6, background: "#1a1a2e", borderRadius: 3, overflow: "hidden" }}>
                <div style={{
                  height: "100%",
                  width: `${Math.min(state.confidence * 100, 100)}%`,
                  background: `linear-gradient(90deg, ${color}88 0%, ${color} 100%)`,
                  borderRadius: 3,
                  transition: "width 0.8s ease",
                }} />
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#6b7280", marginTop: 4 }}>
                <span>
                  {state.direction === "BULLISH"
                    ? "Safe haven buying, war risks, or rate cuts are lifting gold demand."
                    : state.direction === "BEARISH"
                    ? "Dollar strength, yield pressure, or profit taking are weighing on gold."
                    : "Macro drivers are balanced with no single dominant bias."}
                </span>
                {state.news_count > 0 && (
                  <span style={{ flexShrink: 0, marginLeft: 8 }}>
                    ({state.news_count} scored headlines)
                  </span>
                )}
              </div>
            </div>

            {/* Top headlines */}
            {state.top_headlines.length > 0 && (
              <div>
                <div style={{ fontSize: 10, color: "#9ca3af", letterSpacing: "0.08em", marginBottom: 6, fontWeight: 700 }}>
                  TOP CONTRIBUTING NEWS DRIVERS:
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {state.top_headlines.map((h, i) => (
                    <div key={i} style={{
                      display: "flex", alignItems: "center", gap: 8,
                      background: "#0d0d1a", borderRadius: 6, padding: "6px 10px",
                      borderLeft: `3px solid ${h.score > 0 ? "#4ade80" : "#f87171"}`,
                    }}>
                      <span style={{ fontSize: 13, flexShrink: 0 }}>
                        {CATEGORY_ICONS[h.category] ?? "📰"}
                      </span>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{
                          fontSize: 11.5, color: "#e2e8f0", lineHeight: 1.3,
                          whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                        }}>
                          {h.headline}
                        </div>
                      </div>
                      <span style={{
                        fontSize: 10, fontFamily: "monospace", flexShrink: 0,
                        color: h.score > 0 ? "#4ade80" : "#f87171",
                        background: h.score > 0 ? "rgba(74,222,128,0.1)" : "rgba(248,113,113,0.1)",
                        padding: "1px 6px", borderRadius: 3,
                        fontWeight: 700,
                      }}>
                        {h.score > 0 ? "▲ +" : "▼ "}{h.score.toFixed(2)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {state.top_headlines.length === 0 && (
              <div style={{ fontSize: 11.5, color: "#4b5563", fontStyle: "italic" }}>
                Scanning live RSS feeds for fundamental headlines...
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
