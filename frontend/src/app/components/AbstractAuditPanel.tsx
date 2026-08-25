"use client";

import React, { useEffect, useState } from "react";

interface AuditMetric {
  name: string;
  claimed: number;
  actual: number;
  unit: string;
  status: "achieved" | "close" | "gap";
  description: string;
}

interface AuditData {
  status: string;
  metrics: {
    win_rate: AuditMetric;
    sharpe_ratio: AuditMetric;
    max_drawdown: AuditMetric;
    long_win_rate: AuditMetric;
  };
}

export default function AbstractAuditPanel() {
  const [data, setData] = useState<AuditData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/abstract-audit")
      .then((r) => r.json())
      .then((d) => {
        if (d.status === "success") {
          setData(d);
        }
      })
      .catch((err) => console.error("Error fetching audit data:", err))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div style={{ padding: "40px", textAlign: "center", color: "#9ca3af", fontSize: "14px" }}>
        Loading Abstract Audit parameters...
      </div>
    );
  }

  if (!data) {
    return (
      <div style={{ padding: "40px", textAlign: "center", color: "#f87171", fontSize: "14px" }}>
        Failed to load Abstract Audit data.
      </div>
    );
  }

  const { metrics } = data;

  const renderBadge = (status: "achieved" | "close" | "gap") => {
    switch (status) {
      case "achieved":
        return (
          <span style={{ background: "rgba(0, 229, 160, 0.15)", color: "#00E5A0", padding: "4px 8px", borderRadius: "4px", fontSize: "11px", fontWeight: "700", letterSpacing: "0.05em" }}>
            ✓ ACHIEVED
          </span>
        );
      case "close":
        return (
          <span style={{ background: "rgba(251, 191, 36, 0.15)", color: "#FBBF24", padding: "4px 8px", borderRadius: "4px", fontSize: "11px", fontWeight: "700", letterSpacing: "0.05em" }}>
            ⚡ SIGNIFICANT RECOVERY
          </span>
        );
      case "gap":
        return (
          <span style={{ background: "rgba(255, 64, 96, 0.15)", color: "#FF4060", padding: "4px 8px", borderRadius: "4px", fontSize: "11px", fontWeight: "700", letterSpacing: "0.05em" }}>
            ⚠ GAP
          </span>
        );
    }
  };

  const getPercent = (m: AuditMetric) => {
    if (m.status === "achieved") return 100;
    const ratio = m.actual / m.claimed;
    return Math.min(Math.max(ratio * 100, 0), 100);
  };

  return (
    <div style={{ marginTop: "24px" }}>
      <div style={{ marginBottom: "20px" }}>
        <h2 style={{ fontSize: "16px", fontWeight: "800", color: "#F5C842", letterSpacing: "0.05em" }}>
          📋 ACADEMIC PAPER & TARGET AUDIT
        </h2>
        <p style={{ fontSize: "12px", color: "#5C697A", marginTop: "4px" }}>
          Comparing RISTCON 2027 Extended Abstract claims against optimized actual system outputs.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "20px" }}>
        {Object.values(metrics).map((m: AuditMetric) => {
          const isNegative = m.claimed < 0;
          return (
            <div key={m.name} style={{ background: "#0d0d1a", border: "1px solid #1e293b", borderRadius: "10px", padding: "20px", display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "16px" }}>
                  <span style={{ fontSize: "13px", fontWeight: "600", color: "#A8B3C8" }}>{m.name}</span>
                  {renderBadge(m.status)}
                </div>

                <div style={{ display: "flex", gap: "24px", alignItems: "baseline", marginBottom: "16px" }}>
                  <div>
                    <div style={{ fontSize: "10px", color: "#5C697A", textTransform: "uppercase", letterSpacing: "0.05em" }}>Stated in Abstract</div>
                    <div style={{ fontSize: "24px", fontWeight: "800", color: "#FFFFFF", fontFamily: "monospace", marginTop: "2px" }}>
                      {m.claimed}
                      <span style={{ fontSize: "14px", color: "#A8B3C8" }}>{m.unit}</span>
                    </div>
                  </div>
                  <div style={{ borderLeft: "1px dashed rgba(255, 255, 255, 0.1)", height: "30px" }} />
                  <div>
                    <div style={{ fontSize: "10px", color: "#5C697A", textTransform: "uppercase", letterSpacing: "0.05em" }}>Actual Backtest</div>
                    <div style={{ fontSize: "24px", fontWeight: "800", color: m.status === "achieved" ? "#00E5A0" : m.status === "close" ? "#FBBF24" : "#FF4060", fontFamily: "monospace", marginTop: "2px" }}>
                      {m.actual}
                      <span style={{ fontSize: "14px" }}>{m.unit}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <div style={{ fontSize: "11px", color: "#A8B3C8", display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
                  <span>Goal Proximity</span>
                  <span style={{ fontFamily: "monospace" }}>
                    {isNegative ? "100.0%" : `${getPercent(m).toFixed(1)}%`}
                  </span>
                </div>
                <div style={{ height: "6px", background: "rgba(255,255,255,0.03)", borderRadius: "3px", overflow: "hidden", marginBottom: "16px" }}>
                  <div
                    style={{
                      height: "100%",
                      width: isNegative ? "100%" : `${getPercent(m)}%`,
                      background: m.status === "achieved" ? "#00E5A0" : m.status === "close" ? "#FBBF24" : "#FF4060",
                      borderRadius: "3px",
                      transition: "width 0.8s ease-out",
                    }}
                  />
                </div>

                <div style={{ fontSize: "11px", color: "#5C697A", lineHeight: "1.5", borderTop: "1px solid rgba(255, 255, 255, 0.04)", paddingTop: "10px" }}>
                  💡 {m.description}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
