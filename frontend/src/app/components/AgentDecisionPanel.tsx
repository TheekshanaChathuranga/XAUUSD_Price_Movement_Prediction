"use client";

import React, { useState, useEffect } from "react";

export default function AgentDecisionPanel() {
  const [activeTab, setActiveTab] = useState<"specialists" | "syntheses" | "decision" | "reflection">("specialists");
  const [decisionData, setDecisionData] = useState<any>(null);
  const [reflectionsData, setReflectionsData] = useState<any[]>([]);
  const [runStatusText, setRunStatusText] = useState("System ready");
  const [isRunning, setIsRunning] = useState(false);
  const [lastRunText, setLastRunText] = useState("Last run: N/A");

  const fetchAgentDecisionData = async () => {
    try {
      const res = await fetch("/api/agent-decision");
      const data = await res.json();
      if (data.status === "available") {
        const cachedData = { ...data, full_log: null };
        const session_id = data.runner_status?.latest_decision?.session_id;
        if (session_id) {
          const logRes = await fetch(`/api/agent-session-log/${session_id}`);
          const logData = await logRes.json();
          if (logData.status === "ok") {
            cachedData.full_log = logData.session;
          }
        }
        setDecisionData(cachedData);
        if (cachedData.runner_status?.last_run_at) {
          const ts = cachedData.runner_status.last_run_at.split(".")[0].replace("T", " ");
          setLastRunText(`Last run: ${ts} UTC`);
        }
      }
    } catch (e) {
      console.error("Failed to fetch agent decision details:", e);
    }
  };

  const fetchAgentReflectionsData = async () => {
    try {
      const res = await fetch("/api/agent-reflections?limit=3");
      const data = await res.json();
      if (data.status === "success") {
        setReflectionsData(data.reflections || []);
      }
    } catch (e) {
      console.error("Failed to fetch agent reflections:", e);
    }
  };

  useEffect(() => {
    fetchAgentDecisionData();
    fetchAgentReflectionsData();
    const interval = setInterval(() => {
      fetchAgentDecisionData();
      fetchAgentReflectionsData();
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  const triggerAgentRunLive = async () => {
    setIsRunning(true);
    setRunStatusText("Running LLM team...");
    try {
      const res = await fetch("/api/agent-run?force=true", { method: "POST" });
      const data = await res.json();
      if (data.status === "success" || data.status === "error") {
        setRunStatusText(data.status === "success" ? "Run successful!" : "Run failed.");
        await fetchAgentDecisionData();
        await fetchAgentReflectionsData();
      } else {
        setRunStatusText("Trigger error");
      }
    } catch (e) {
      console.error(e);
      setRunStatusText("Network error");
    } finally {
      setTimeout(() => {
        setIsRunning(false);
        setRunStatusText("System ready");
      }, 3000);
    }
  };

  const getTagClass = (status: string) => {
    if (!status) return "tag-amber";
    const s = status.toUpperCase();
    if (s === "BULLISH" || s === "LONG" || s === "IMMINENT" || s === "APPROACHING") return "tag-green";
    if (s === "BEARISH" || s === "SHORT") return "tag-red";
    return "tag-amber";
  };

  const renderSpecialists = () => {
    if (!decisionData?.full_log) {
      return (
        <div style={{ fontSize: "0.75rem", color: "var(--t3)", textAlign: "center", padding: "20px" }}>
          No agent run logs found. Force run pipeline to generate.
        </div>
      );
    }

    const log = decisionData.full_log;
    const items = [
      { name: "📊 Technical Specialist", status: log.technical_specialist?.trend_direction, details: `RSI: ${log.technical_specialist?.rsi_value?.toFixed(1)} | MACD: ${log.technical_specialist?.momentum_state?.split(" — ")[0] || "N/A"}` },
      { name: "🤖 Quantitative Specialist", status: log.quantitative_specialist?.quant_signal, details: `P(up): ${(log.quantitative_specialist?.quant_prob_up * 100).toFixed(1)}% | Consensus: ${log.quantitative_specialist?.quant_consensus ? "YES" : "NO"}` },
      { name: "🔥 Macro SVAR Specialist", status: log.macro_svar_specialist?.macro_bias, details: `Oil 5d: ${log.macro_svar_specialist?.oil_return_5d?.toFixed(1)}% | VIX: ${log.macro_svar_specialist?.vix_level?.toFixed(1)}` },
      { name: "🏦 FRED Policy Specialist", status: log.policy_fred_specialist?.policy_bias, details: `Fed rate: ${log.policy_fred_specialist?.fed_funds_rate}% | Stance: ${log.policy_fred_specialist?.fed_policy_stance}` },
      { name: "📰 News Sentiment Specialist", status: log.sentiment_specialist?.sentiment_direction, details: `Blend: ${log.sentiment_specialist?.blended_score?.toFixed(2)} | Headlines: ${log.sentiment_specialist?.news_volume_24h}` },
      { name: "🌐 Geopolitical Specialist", status: log.geopolitical_gdelt_specialist?.geopolitical_bias, details: `GDELT War: ${log.geopolitical_gdelt_specialist?.war_geopolitical_active ? "YES" : "NO"} | Surge: ${log.geopolitical_gdelt_specialist?.geo_surge_score?.toFixed(1)}` },
      { name: "📅 Calendar Specialist", status: log.calendar_specialist?.volatility_flag, details: `Caution: ${log.calendar_specialist?.caution_level}/3 | Window: ${log.calendar_specialist?.event_window_active ? "ACTIVE" : "IDLE"}` }
    ];

    return (
      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        {items.map((i, idx) => (
          <div key={idx} style={{ background: "var(--surf2)", border: "1px solid var(--border)", borderRadius: "6px", padding: "6px 10px", display: "flex", flexDirection: "column", gap: "2px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--t1)" }}>{i.name}</span>
              <span className={getTagClass(i.status)} style={{ fontSize: "0.6rem", padding: "1px 5px" }}>{i.status || "NEUTRAL"}</span>
            </div>
            <div style={{ fontSize: "0.65rem", color: "var(--t3)" }}>{i.details}</div>
          </div>
        ))}
      </div>
    );
  };

  const renderSyntheses = () => {
    if (!decisionData?.full_log) {
      return (
        <div style={{ fontSize: "0.75rem", color: "var(--t3)", textAlign: "center", padding: "20px" }}>
          No synthesis data available.
        </div>
      );
    }

    const log = decisionData.full_log;
    const tq = log.technical_quant_synthesis;
    const ms = log.macro_sentiment_synthesis;

    return (
      <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
        <div style={{ background: "var(--surf2)", border: "1px solid var(--border)", borderRadius: "8px", padding: "10px 12px", display: "flex", flexDirection: "column", gap: "4px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--t1)" }}>📊 Tech & Quant Synthesis</span>
            <span className={getTagClass(tq?.combined_bias)} style={{ fontSize: "0.62rem", padding: "2px 6px" }}>{tq?.combined_bias || "NEUTRAL"}</span>
          </div>
          <div style={{ fontSize: "0.68rem", color: "var(--t3)", lineHeight: 1.3, fontStyle: "italic" }}>"{tq?.summary || "No technical synthesis."}"</div>
          <div style={{ fontSize: "0.62rem", color: "var(--t2)", marginTop: "2px" }}>Confluences: {tq?.confluence_indicators?.slice(0, 3).join(", ") || "None"}</div>
        </div>

        <div style={{ background: "var(--surf2)", border: "1px solid var(--border)", borderRadius: "8px", padding: "10px 12px", display: "flex", flexDirection: "column", gap: "4px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--t1)" }}>🏛️ Macro & Sentiment Synthesis</span>
            <span className={getTagClass(ms?.combined_bias)} style={{ fontSize: "0.62rem", padding: "2px 6px" }}>{ms?.combined_bias || "NEUTRAL"}</span>
          </div>
          <div style={{ fontSize: "0.68rem", color: "var(--t3)", lineHeight: 1.3, fontStyle: "italic" }}>"{ms?.summary || "No fundamental synthesis."}"</div>
          <div style={{ fontSize: "0.62rem", color: "var(--t2)", marginTop: "2px" }}>Drivers: {ms?.key_macro_drivers?.slice(0, 2).join(", ") || "None"}</div>
        </div>
      </div>
    );
  };

  const renderDecision = () => {
    if (!decisionData?.full_log) {
      return (
        <div style={{ fontSize: "0.75rem", color: "var(--t3)", textAlign: "center", padding: "20px" }}>
          No decision logs available.
        </div>
      );
    }

    const log = decisionData.full_log;
    const fm = log.fund_manager_decision;
    const fmDec = fm?.final_decision || "REJECT";
    const fmDir = fm?.final_direction || "HOLD";
    const decClass = fmDec === "APPROVE" ? "tag-green" : fmDec === "RESIZE" ? "tag-blue" : "tag-red";

    return (
      <div style={{ background: "var(--surf2)", border: "1px solid var(--border)", borderRadius: "8px", padding: "12px", display: "flex", flexDirection: "column", gap: "6px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--t1)" }}>💼 Portfolio Manager Final Gate</span>
          <span className={decClass} style={{ fontSize: "0.65rem", padding: "2px 6px", fontWeight: 700 }}>{fmDec}</span>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px", marginTop: "4px" }}>
          <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--border)", borderRadius: "4px", padding: "6px", textAlign: "center" }}>
            <div style={{ fontSize: "0.55rem", color: "var(--t3)", textTransform: "uppercase" }}>Execution Direction</div>
            <span className={getTagClass(fmDir)} style={{ fontSize: "0.75rem", display: "inline-block", marginTop: "2px" }}>{fmDir}</span>
          </div>
          <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--border)", borderRadius: "4px", padding: "6px", textAlign: "center" }}>
            <div style={{ fontSize: "0.55rem", color: "var(--t3)", textTransform: "uppercase" }}>Allocated Lot Size</div>
            <span style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--gold)", display: "inline-block", marginTop: "2px" }}>{(fm?.final_lot_size || 0.0).toFixed(2)} Lots</span>
          </div>
        </div>

        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.65rem", color: "var(--t2)", paddingTop: "4px", borderTop: "1px dashed var(--border)" }}>
          <span>SL: {fm?.final_sl_pips || 0} pips</span>
          <span>TP: {fm?.final_tp_pips || 0} pips</span>
          <span>Regime: {fm?.current_regime || "CALM"}</span>
        </div>

        <div style={{ fontSize: "0.68rem", color: "var(--t3)", lineHeight: 1.35, paddingTop: "4px", fontStyle: "italic" }}>
          "{fm?.full_reasoning || "No details provided."}"
        </div>
      </div>
    );
  };

  const renderReflections = () => {
    if (!reflectionsData || !reflectionsData.length) {
      return (
        <div style={{ fontSize: "0.75rem", color: "var(--t3)", textAlign: "center", padding: "20px" }}>
          No trade loss reflections recorded yet. System learning loop is clear.
        </div>
      );
    }

    return (
      <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
        {reflectionsData.map((r, idx) => (
          <div key={idx} style={{ background: "rgba(235,87,87,0.05)", border: "1px solid rgba(235,87,87,0.2)", borderRadius: "8px", padding: "8px 10px", display: "flex", flexDirection: "column", gap: "3px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--red)" }}>⚠️ LOSS REFLECTION</span>
              <span style={{ fontSize: "0.6rem", color: "var(--t3)" }}>{r.date || "UTC"}</span>
            </div>
            <div style={{ fontSize: "0.65rem", color: "var(--t2)", fontWeight: 600 }}>Trade: {r.direction} lost {(r.loss_bps || 0).toFixed(0)}bps (Market: {r.actual_move})</div>
            <div style={{ fontSize: "0.65rem", color: "var(--t3)", lineHeight: 1.3, marginTop: "2px" }}>
              <strong>Diagnosis:</strong> {r.diagnosis}
            </div>
            <div style={{ fontSize: "0.65rem", color: "var(--gold)", fontWeight: 600, marginTop: "2px", borderTop: "1px dashed rgba(255,255,255,0.05)", paddingTop: "3px" }}>
              💡 Lesson: {r.lesson}
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="card card-agents-panel" style={{ padding: "16px", display: "flex", flexDirection: "column", minHeight: "480px", justifyContent: "space-between", overflowY: "auto" }}>
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
          <span className="card-label" style={{ margin: 0, fontSize: "0.85rem", letterSpacing: "0.05em" }}>🤖 AGENT DECISION LAYER v2</span>
          <span className="consensus-badge consensus-ok" style={{ fontSize: "0.65rem", padding: "2px 6px" }}>9 AGENTS ACTIVE</span>
        </div>

        {/* Tab Switcher */}
        <div style={{ display: "flex", gap: "4px", marginBottom: "12px", background: "var(--surf1)", padding: "4px", borderRadius: "6px", border: "1px solid var(--border)" }}>
          {(["specialists", "syntheses", "decision", "reflection"] as const).map((tab) => {
            const isActive = activeTab === tab;
            return (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                style={{
                  flex: 1,
                  background: isActive ? "rgba(255,255,255,0.08)" : "none",
                  border: "none",
                  color: isActive ? "var(--t1)" : "var(--t3)",
                  fontSize: "0.68rem",
                  fontWeight: 600,
                  padding: "6px 0",
                  borderRadius: "4px",
                  cursor: "pointer",
                  textTransform: "capitalize"
                }}
              >
                {tab === "reflection" ? "Reflections" : tab === "decision" ? "PM Decision" : tab}
              </button>
            );
          })}
        </div>

        {/* Tab Contents */}
        {activeTab === "specialists" && renderSpecialists()}
        {activeTab === "syntheses" && renderSyntheses()}
        {activeTab === "decision" && renderDecision()}
        {activeTab === "reflection" && renderReflections()}
      </div>

      {/* Trigger Button Widget */}
      <div style={{ marginTop: "14px", paddingTop: "10px", borderTop: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", flexDirection: "column" }}>
          <span style={{ fontSize: "0.6rem", color: "var(--t3)", textTransform: "uppercase" }}>{lastRunText}</span>
          <span style={{ fontSize: "0.65rem", color: isRunning ? "var(--gold)" : "var(--green)", fontWeight: 600 }}>{runStatusText}</span>
        </div>
        <button
          onClick={triggerAgentRunLive}
          disabled={isRunning}
          style={{
            background: "var(--blue)",
            border: "none",
            color: "white",
            fontSize: "0.7rem",
            fontWeight: 700,
            padding: "8px 12px",
            borderRadius: "6px",
            cursor: isRunning ? "not-allowed" : "pointer",
            opacity: isRunning ? 0.5 : 1,
            display: "flex",
            alignItems: "center",
            gap: "4px",
            boxShadow: "0 0 10px rgba(0,102,255,0.3)",
            whiteSpace: "nowrap"
          }}
        >
          <span>⚡ FORCE RUN PIPELINE</span>
        </button>
      </div>
    </div>
  );
}
