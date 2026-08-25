"use client";

import React from "react";

interface ModelStats {
  name: string;
  acc: number;
  auc: number;
  f1: number;
  prec: number;
  rec: number;
  type: string;
}

export default function ModelPerformancePanel() {
  const models: ModelStats[] = [
    { name: "CatBoost", acc: 0.494, auc: 0.449, f1: 0.602, prec: 0.546, rec: 0.671, type: "Base Booster" },
    { name: "XGBoost", acc: 0.494, auc: 0.446, f1: 0.602, prec: 0.546, rec: 0.671, type: "Base Booster" },
    { name: "LightGBM", acc: 0.483, auc: 0.433, f1: 0.579, prec: 0.541, rec: 0.624, type: "Base Booster" },
    { name: "Multi-Modal Ensemble", acc: 0.471, auc: 0.440, f1: 0.537, prec: 0.537, rec: 0.537, type: "Stacked Consensus" },
  ];

  return (
    <div style={{ marginTop: "32px" }}>
      <div style={{ marginBottom: "20px" }}>
        <h2 style={{ fontSize: "16px", fontWeight: "800", color: "#fbbf24", letterSpacing: "0.05em" }}>
          🤖 RAW MACHINE LEARNING PERFORMANCE
        </h2>
        <p style={{ fontSize: "12px", color: "#5c697a", marginTop: "4px" }}>
          Out-of-sample raw classification metrics on spot direction prediction (2025-05 to 2026-07).
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "20px" }}>
        {models.map((m) => {
          const isEnsemble = m.name.includes("Ensemble");
          return (
            <div
              key={m.name}
              style={{
                background: isEnsemble ? "rgba(245, 200, 66, 0.03)" : "#0d0d1a",
                border: isEnsemble ? "1px solid rgba(245, 200, 66, 0.25)" : "1px solid #1e293b",
                borderRadius: "10px",
                padding: "20px",
                boxShadow: isEnsemble ? "0 0 15px rgba(245, 200, 66, 0.05)" : "none",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
                <span style={{ fontSize: "14px", fontWeight: "700", color: isEnsemble ? "#F5C842" : "#FFFFFF" }}>
                  {m.name}
                </span>
                <span style={{ fontSize: "9px", background: "rgba(255,255,255,0.04)", color: "#a8b3c8", padding: "2px 6px", borderRadius: "3px", textTransform: "uppercase" }}>
                  {m.type}
                </span>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
                <div>
                  <div style={{ fontSize: "10px", color: "#5c697a" }}>Accuracy (DA)</div>
                  <div style={{ fontSize: "18px", fontWeight: "700", fontFamily: "monospace", color: "#FFFFFF", marginTop: "2px" }}>
                    {(m.acc * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: "10px", color: "#5c697a" }}>ROC-AUC</div>
                  <div style={{ fontSize: "18px", fontWeight: "700", fontFamily: "monospace", color: "#a8b3c8", marginTop: "2px" }}>
                    {m.auc.toFixed(3)}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: "10px", color: "#5c697a" }}>F1-Score</div>
                  <div style={{ fontSize: "18px", fontWeight: "700", fontFamily: "monospace", color: "#a8b3c8", marginTop: "2px" }}>
                    {m.f1.toFixed(3)}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: "10px", color: "#5c697a" }}>Precision</div>
                  <div style={{ fontSize: "18px", fontWeight: "700", fontFamily: "monospace", color: "#a8b3c8", marginTop: "2px" }}>
                    {(m.prec * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              <div style={{ marginTop: "14px", paddingTop: "12px", borderTop: "1px solid rgba(255,255,255,0.03)", display: "flex", justifyContent: "space-between", fontSize: "11px" }}>
                <span style={{ color: "#5c697a" }}>Recall Rate:</span>
                <span style={{ color: "#a8b3c8", fontWeight: "600", fontFamily: "monospace" }}>
                  {(m.rec * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
