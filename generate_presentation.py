#!/usr/bin/env python3
"""
generate_presentation.py
Generuje prezentację reveal.js dla projektu GGSN Temporal GNN.

Uruchomienie:
    uv run python generate_presentation.py

Wyjście:    presentation.html   (samodzielny plik — otwórz w przeglądarce)
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "presentation.html"

# ─── Prawdziwe wyniki z data/snapshots/gnn/ ──────────────────────────────────
# (name, auroc, auprc, group, brier, sens_at_95spec)
EXPERIMENTS = [
    ("signal only", 0.784, 0.381, "ablation", 0.181, 0.301),
    ("e2e fine-tuning", 0.790, 0.303, "ablation", 0.204, 0.193),
    ("+ allstay (fixed)", 0.820, 0.396, "feature", 0.170, 0.327),
    ("+ ICD node", 0.823, 0.377, "feature", 0.202, 0.311),
    ("+ focal γ=2 only", 0.827, 0.408, "loss", 0.182, 0.336),
    ("baseline", 0.829, 0.403, "baseline", 0.180, 0.326),
    ("2 GIN layers", 0.835, 0.438, "arch", 0.190, 0.352),
    ("demo + 4 layers", 0.836, 0.408, "arch", 0.203, 0.330),
    ("+ demographics", 0.841, 0.429, "demo", 0.172, 0.352),
    ("demo + focal γ=2", 0.843, 0.426, "combined", 0.170, 0.353),
    ("demo + attention pool", 0.844, 0.448, "pooling", 0.197, 0.378),
    ("demo + dual pool", 0.846, 0.435, "pooling", 0.170, 0.352),
    ("demo+attn+focal ★", 0.850, 0.465, "best", 0.195, 0.388),
]

GROUP_COLORS = {
    "ablation": "#f87171",
    "feature": "#60a5fa",
    "baseline": "#818cf8",
    "loss": "#a78bfa",
    "arch": "#06b6d4",
    "demo": "#34d399",
    "combined": "#f59e0b",
    "pooling": "#10b981",
    "best": "#fbbf24",
}

# matplotlib dark theme
plt.rcParams.update(
    {
        "figure.facecolor": "#0a0a0a",
        "axes.facecolor": "#111111",
        "axes.edgecolor": "#2d2d2d",
        "axes.labelcolor": "#94a3b8",
        "xtick.color": "#64748b",
        "ytick.color": "#94a3b8",
        "text.color": "#e2e8f0",
        "grid.color": "#1e293b",
        "grid.linewidth": 0.8,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ─── Chart helpers ────────────────────────────────────────────────────────────


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", bbox_inches="tight", facecolor="#0a0a0a", edgecolor="none", dpi=160
    )
    buf.seek(0)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def chart_auroc() -> str:
    names = [e[0] for e in EXPERIMENTS]
    aurocs = [e[1] for e in EXPERIMENTS]
    groups = [e[3] for e in EXPERIMENTS]
    colors = [GROUP_COLORS[g] for g in groups]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(names))
    bars = ax.barh(y, [a - 0.74 for a in aurocs], left=0.74, color=colors, height=0.62, zorder=3)

    ax.axvline(0.88, color="#f87171", lw=1.5, ls="--", alpha=0.85, zorder=4)
    ax.text(0.8815, len(names) - 0.5, "SOTA 0.88", color="#f87171", fontsize=8.5, va="top")
    ax.axvline(0.829, color="#818cf8", lw=1, ls=":", alpha=0.5, zorder=4)

    for i, (bar, v, g) in enumerate(zip(bars, aurocs, groups)):
        ax.text(
            v + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center",
            ha="left",
            color="#fbbf24" if g == "best" else "#e2e8f0",
            fontsize=9.5,
            fontweight="bold" if g == "best" else "normal",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlim(0.74, 0.895)
    ax.set_xlabel("AUROC", fontsize=11)
    ax.set_title(
        "AUROC — wszystkie eksperymenty (n=13)", color="#e2e8f0", fontsize=13, pad=12, loc="left"
    )
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)

    seen = {}
    for g, c in GROUP_COLORS.items():
        if g in groups and g not in seen:
            seen[g] = c
    patches = [mpatches.Patch(color=c, label=g) for g, c in seen.items()]
    ax.legend(
        handles=patches,
        fontsize=8,
        loc="lower right",
        facecolor="#1e1e2e",
        edgecolor="#2d2d2d",
        labelcolor="#e2e8f0",
        framealpha=0.9,
    )

    fig.tight_layout()
    return _b64(fig)


def chart_auprc() -> str:
    names = [e[0] for e in EXPERIMENTS]
    auprcs = [e[2] for e in EXPERIMENTS]
    groups = [e[3] for e in EXPERIMENTS]
    colors = [GROUP_COLORS[g] for g in groups]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(names))
    bars = ax.barh(y, [a - 0.25 for a in auprcs], left=0.25, color=colors, height=0.62, zorder=3)

    ax.axvline(0.127, color="#f59e0b", lw=1.5, ls="--", alpha=0.8, zorder=4)
    ax.text(0.132, len(names) - 0.5, "random 0.127", color="#f59e0b", fontsize=8.5, va="top")

    for bar, v, g in zip(bars, auprcs, groups):
        ax.text(
            v + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center",
            ha="left",
            color="#fbbf24" if g == "best" else "#e2e8f0",
            fontsize=9.5,
            fontweight="bold" if g == "best" else "normal",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlim(0.25, 0.52)
    ax.set_xlabel("AUPRC (precision–recall AUC)", fontsize=11)
    ax.set_title(
        "AUPRC — 3.6× powyżej losowego (0.127)", color="#e2e8f0", fontsize=13, pad=12, loc="left"
    )
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _b64(fig)


def chart_leakage() -> str:
    fig, ax = plt.subplots(figsize=(6, 3.8))

    labels = ["Ostatnie 50 sygnałów\n(temporal leakage)", "Pierwsze 50 sygnałów\n(naprawione)"]
    vals = [0.931, 0.820]
    colors = ["#f87171", "#34d399"]

    bars = ax.bar([0, 1], vals, color=colors, width=0.55, zorder=3)
    ax.set_ylim(0.75, 0.97)
    ax.set_ylabel("AUROC")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(
        "Wykryty błąd — temporal leakage", color="#e2e8f0", fontsize=13, pad=12, loc="left"
    )

    for bar, v, c in zip(bars, vals, colors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.005,
            f"{v:.3f}",
            ha="center",
            fontsize=16,
            fontweight="bold",
            color=c,
        )

    ax.annotate(
        "sygnały bliskie śmierci\n→ trywialnie wysoki AUROC",
        xy=(0, 0.931),
        xytext=(0.55, 0.895),
        arrowprops=dict(arrowstyle="->", color="#f87171", lw=1.5),
        color="#f87171",
        fontsize=9,
    )

    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _b64(fig)


def chart_contributions() -> str:
    baseline = 0.829
    components = [
        ("− notatki kliniczne", 0.784 - baseline, "#f87171"),
        ("+ ICD node", 0.823 - baseline, "#f87171"),
        ("+ allstay first-50", 0.820 - baseline, "#f87171"),
        ("+ focal γ=2 only", 0.827 - baseline, "#a78bfa"),
        ("+ demographics", 0.841 - baseline, "#34d399"),
        ("+ attention pooling", 0.844 - baseline, "#06b6d4"),
        ("+ dual pooling", 0.846 - baseline, "#10b981"),
        ("+ demo + focal", 0.843 - baseline, "#f59e0b"),
        ("demo + attn + focal ★", 0.850 - baseline, "#fbbf24"),
    ]
    labels = [c[0] for c in components]
    deltas = [c[1] for c in components]
    colors = [c[2] for c in components]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.barh(range(len(labels)), deltas, color=colors, height=0.6, zorder=3)
    ax.axvline(0, color="#94a3b8", lw=1.5, zorder=4)

    for bar, d in zip(bars, deltas):
        xoff = 0.0008 if d >= 0 else -0.0008
        ha = "left" if d >= 0 else "right"
        ax.text(
            d + xoff,
            bar.get_y() + bar.get_height() / 2,
            f"{d:+.3f}",
            va="center",
            ha=ha,
            fontsize=9.5,
            color="#e2e8f0",
        )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("ΔAUROC vs baseline (0.829)", fontsize=11)
    ax.set_title("Wkład każdego komponentu", color="#e2e8f0", fontsize=13, pad=12, loc="left")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _b64(fig)


def chart_sota() -> str:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    models = ["Losowy\nbaseline", "Signal\nonly", "Nasz\nbest ★", "SOTA\n(lit.)"]
    vals = [0.50, 0.784, 0.850, 0.880]
    colors = ["#475569", "#f87171", "#fbbf24", "#818cf8"]

    bars = ax.bar(range(4), vals, color=colors, width=0.55, zorder=3)
    ax.set_ylim(0.40, 0.935)
    ax.set_ylabel("AUROC")
    ax.set_xticks(range(4))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_title("Nasz wynik vs SOTA", color="#e2e8f0", fontsize=13, pad=12, loc="left")

    for bar, v, c in zip(bars, vals, colors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.01,
            f"{v:.3f}",
            ha="center",
            fontsize=13,
            fontweight="bold",
            color=c,
        )

    # gap annotation
    ax.annotate(
        "",
        xy=(3, 0.880),
        xytext=(2, 0.850),
        arrowprops=dict(arrowstyle="<->", color="#94a3b8", lw=1.5),
    )
    ax.text(2.55, 0.875, "Δ 0.030", ha="center", color="#94a3b8", fontsize=10)

    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _b64(fig)


# ─── SVG/CSS animations ───────────────────────────────────────────────────────

GRAPH_ANIMATION_SVG = """
<svg viewBox="0 0 520 280" style="width:100%;max-width:560px;margin:0 auto;display:block;">
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#4a5568"/>
    </marker>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <style>
    .node { opacity:0; animation: fadeNode 0.5s ease forwards; }
    .edge { stroke-dasharray:120; stroke-dashoffset:120;
            animation: drawEdge 0.6s ease forwards; opacity:0.55; }
    @keyframes fadeNode { to { opacity:1; } }
    @keyframes drawEdge { to { stroke-dashoffset:0; } }

    .n-icd  { fill:#818cf8; animation-delay:0.2s; }
    .n-sig1 { animation-delay:0.7s; }
    .n-sig2 { animation-delay:1.1s; }
    .n-sig3 { animation-delay:1.6s; }
    .n-note1{ animation-delay:1.3s; }
    .n-note2{ animation-delay:2.0s; }

    .e1 { animation-delay:0.9s; }
    .e2 { animation-delay:1.4s; }
    .e3 { animation-delay:1.8s; }
    .e4 { animation-delay:1.6s; }
    .e5 { animation-delay:2.2s; }
    .e6 { animation-delay:2.4s; }
    .e7 { animation-delay:2.6s; }
  </style>

  <!-- timeline axis -->
  <line x1="50" y1="245" x2="470" y2="245" stroke="#2d3748" stroke-width="1.5"/>
  <text x="50" y="265" fill="#475569" font-size="11" text-anchor="middle">t = -1h</text>
  <text x="160" y="265" fill="#475569" font-size="11" text-anchor="middle">0h</text>
  <text x="260" y="265" fill="#475569" font-size="11" text-anchor="middle">8h</text>
  <text x="350" y="265" fill="#475569" font-size="11" text-anchor="middle">16h</text>
  <text x="450" y="265" fill="#475569" font-size="11" text-anchor="middle">24h</text>

  <!-- edges -->
  <line class="edge e1" x1="66" y1="130" x2="152" y2="105" stroke="#4a5568" stroke-width="1.5" marker-end="url(#arr)"/>
  <line class="edge e2" x1="66" y1="130" x2="152" y2="160" stroke="#4a5568" stroke-width="1.5" marker-end="url(#arr)"/>
  <line class="edge e3" x1="172" y1="100" x2="242" y2="140" stroke="#4a5568" stroke-width="1.5" marker-end="url(#arr)"/>
  <line class="edge e4" x1="172" y1="165" x2="242" y2="145" stroke="#4a5568" stroke-width="1.5" marker-end="url(#arr)"/>
  <line class="edge e5" x1="262" y1="135" x2="332" y2="100" stroke="#4a5568" stroke-width="1.5" marker-end="url(#arr)"/>
  <line class="edge e6" x1="262" y1="135" x2="332" y2="165" stroke="#4a5568" stroke-width="1.5" marker-end="url(#arr)"/>
  <line class="edge e7" x1="352" y1="100" x2="432" y2="130" stroke="#4a5568" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- ICD node (diamond) -->
  <g class="node n-icd" filter="url(#glow)">
    <polygon points="50,118 66,130 50,142 34,130" fill="#818cf8" opacity="0.9"/>
    <text x="50" y="156" fill="#818cf8" font-size="10" text-anchor="middle" font-weight="bold">ICD</text>
    <text x="50" y="168" fill="#475569" font-size="9" text-anchor="middle">Charlson</text>
  </g>

  <!-- Signal nodes (circles, cyan) -->
  <g class="node n-sig1">
    <circle cx="162" cy="100" r="16" fill="#0e7490" stroke="#06b6d4" stroke-width="1.5"/>
    <text x="162" y="104" fill="#e2e8f0" font-size="9" text-anchor="middle" font-weight="bold">HR</text>
    <text x="162" y="128" fill="#475569" font-size="9" text-anchor="middle">signal</text>
  </g>
  <g class="node n-sig2">
    <circle cx="162" cy="165" r="16" fill="#0e7490" stroke="#06b6d4" stroke-width="1.5"/>
    <text x="162" y="169" fill="#e2e8f0" font-size="9" text-anchor="middle" font-weight="bold">SpO₂</text>
    <text x="162" y="193" fill="#475569" font-size="9" text-anchor="middle">signal</text>
  </g>
  <g class="node n-sig3">
    <circle cx="342" cy="100" r="16" fill="#0e7490" stroke="#06b6d4" stroke-width="1.5"/>
    <text x="342" y="104" fill="#e2e8f0" font-size="9" text-anchor="middle" font-weight="bold">MAP</text>
    <text x="342" y="128" fill="#475569" font-size="9" text-anchor="middle">signal</text>
  </g>

  <!-- Note nodes (rounded rect, indigo) -->
  <g class="node n-note1">
    <rect x="236" y="118" width="52" height="30" rx="8"
          fill="#312e81" stroke="#818cf8" stroke-width="1.5"/>
    <text x="262" y="136" fill="#c7d2fe" font-size="9" text-anchor="middle" font-weight="bold">NOTE</text>
    <text x="262" y="162" fill="#475569" font-size="9" text-anchor="middle">128-dim</text>
  </g>
  <g class="node n-note2">
    <rect x="420" y="115" width="52" height="30" rx="8"
          fill="#312e81" stroke="#818cf8" stroke-width="1.5"/>
    <text x="446" y="133" fill="#c7d2fe" font-size="9" text-anchor="middle" font-weight="bold">NOTE</text>
    <text x="446" y="159" fill="#475569" font-size="9" text-anchor="middle">128-dim</text>
  </g>

  <!-- Note node label (static) -->
  <g class="node n-note2">
    <rect x="420" y="115" width="52" height="30" rx="8"
          fill="#312e81" stroke="#818cf8" stroke-width="1.5"/>
    <text x="446" y="133" fill="#c7d2fe" font-size="9" text-anchor="middle" font-weight="bold">NOTE</text>
  </g>

  <!-- legend -->
  <circle cx="30" cy="13" r="7" fill="#0e7490" stroke="#06b6d4" stroke-width="1.5"/>
  <text x="42" y="17" fill="#94a3b8" font-size="10">Sygnał (vital/lab)</text>
  <rect x="148" y="6" width="14" height="14" rx="3" fill="#312e81" stroke="#818cf8" stroke-width="1.5"/>
  <text x="168" y="17" fill="#94a3b8" font-size="10">Notatka kliniczna</text>
  <polygon points="295,13 305,19 295,25 285,19" fill="#818cf8"/>
  <text x="312" y="17" fill="#94a3b8" font-size="10">ICD Charlson</text>
</svg>
"""

TWOTOWER_ANIMATION = """
<style>
  .tower-box {
    background: #1e1e2e;
    border: 1px solid #2d2d4e;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    width: 180px;
    position: relative;
  }
  .tower-title { color: #818cf8; font-size: 13px; font-weight: 700; margin-bottom: 6px; }
  .tower-sub   { color: #64748b; font-size: 11px; margin-bottom: 12px; }
  .emb-vec {
    display: flex; gap: 3px; justify-content: center;
    animation: pulse 2s ease-in-out infinite;
  }
  .emb-cell {
    width: 12px; height: 28px; border-radius: 3px;
    background: linear-gradient(180deg, #818cf8, #06b6d4);
    opacity: 0.7;
  }
  .emb-cell:nth-child(2n)   { opacity: 0.4; transform: scaleY(0.7); }
  .emb-cell:nth-child(3n)   { opacity: 0.9; transform: scaleY(1.15); }
  @keyframes pulse { 0%,100%{opacity:0.7} 50%{opacity:1} }

  .loss-badge {
    background: #0f172a;
    border: 1px solid #34d399;
    border-radius: 20px;
    padding: 8px 18px;
    color: #34d399;
    font-size: 12px;
    font-weight: 700;
    animation: glow 2s ease-in-out infinite;
  }
  @keyframes glow {
    0%,100%{box-shadow:0 0 4px #34d39944}
    50%{box-shadow:0 0 14px #34d39999}
  }
  .arrow-down {
    color: #4a5568; font-size: 20px; line-height: 1;
    animation: arrowPulse 2s ease-in-out infinite;
  }
  @keyframes arrowPulse { 0%,100%{color:#4a5568} 50%{color:#818cf8} }
  .connector-line {
    width: 80px; height: 2px;
    background: linear-gradient(90deg, #818cf800, #34d399, #818cf800);
    animation: connPulse 2s ease-in-out infinite;
  }
  @keyframes connPulse { 0%,100%{opacity:0.3} 50%{opacity:1} }
</style>
<div style="display:flex;align-items:center;justify-content:center;gap:20px;padding:10px 0;">
  <div style="display:flex;flex-direction:column;align-items:center;gap:10px;">
    <div style="background:#1e293b;border:1px solid #2d2d4e;border-radius:8px;padding:8px 12px;color:#94a3b8;font-size:11px;text-align:center;">
      📄 Notatka kliniczna<br><span style="color:#4a5568;font-size:10px;">„Patient appears..."</span>
    </div>
    <div class="arrow-down">↓</div>
    <div class="tower-box">
      <div class="tower-title">TextTower</div>
      <div class="tower-sub">Bio_ClinicalBERT<br>768→384→128</div>
      <div class="emb-vec">
        <div class="emb-cell"></div><div class="emb-cell"></div><div class="emb-cell"></div>
        <div class="emb-cell"></div><div class="emb-cell"></div><div class="emb-cell"></div>
        <div class="emb-cell"></div><div class="emb-cell"></div>
      </div>
    </div>
    <div class="arrow-down">↓</div>
    <div style="color:#818cf8;font-size:11px;font-weight:600;">z_text ∈ ℝ¹²⁸ (L2-norm)</div>
  </div>

  <div style="display:flex;flex-direction:column;align-items:center;gap:8px;">
    <div style="height:60px;"></div>
    <div class="connector-line"></div>
    <div class="loss-badge">InfoNCE</div>
    <div class="connector-line"></div>
  </div>

  <div style="display:flex;flex-direction:column;align-items:center;gap:10px;">
    <div style="background:#1e293b;border:1px solid #2d2d4e;border-radius:8px;padding:8px 12px;color:#94a3b8;font-size:11px;text-align:center;">
      📈 Sygnały fizjologiczne<br><span style="color:#4a5568;font-size:10px;">HR=82, SpO₂=97, ...</span>
    </div>
    <div class="arrow-down">↓</div>
    <div class="tower-box">
      <div class="tower-title">SignalTower</div>
      <div class="tower-sub">Embedding+1DCNN<br>GAP→128</div>
      <div class="emb-vec">
        <div class="emb-cell"></div><div class="emb-cell"></div><div class="emb-cell"></div>
        <div class="emb-cell"></div><div class="emb-cell"></div><div class="emb-cell"></div>
        <div class="emb-cell"></div><div class="emb-cell"></div>
      </div>
    </div>
    <div class="arrow-down">↓</div>
    <div style="color:#818cf8;font-size:11px;font-weight:600;">z_signal ∈ ℝ¹²⁸ (L2-norm)</div>
  </div>
</div>
"""

DATAFLOW_ANIMATION = """
<style>
  .flow-stage {
    background:#1e1e2e;border:1px solid #2d2d4e;
    border-radius:10px;padding:12px 16px;
    text-align:center;min-width:120px;
  }
  .flow-title { color:#e2e8f0;font-size:12px;font-weight:700;margin-bottom:4px; }
  .flow-sub   { color:#64748b;font-size:10px; }
  .flow-arrow { color:#4a5568;font-size:22px;padding:0 6px;align-self:center; }
  .flow-badge {
    display:inline-block;border-radius:20px;padding:3px 10px;
    font-size:10px;font-weight:700;margin-top:6px;
  }
  .badge-phase1 { background:#312e8133;border:1px solid #818cf8;color:#818cf8; }
  .badge-phase2 { background:#0c4a6e33;border:1px solid #06b6d4;color:#06b6d4; }
  .badge-out    { background:#14532d33;border:1px solid #34d399;color:#34d399; }
</style>
<div style="display:flex;align-items:stretch;justify-content:center;flex-wrap:wrap;gap:6px;padding:8px 0;">
  <div class="flow-stage">
    <div class="flow-title">MIMIC-IV</div>
    <div class="flow-sub">52K pobytów<br>notatki + sygnały</div>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-stage">
    <div class="flow-title">Phase 1</div>
    <div class="flow-sub">Two-Tower<br>InfoNCE</div>
    <div class="flow-badge badge-phase1">pre-training</div>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-stage">
    <div class="flow-title">Embeddingi</div>
    <div class="flow-sub">128-dim<br>zamrożone</div>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-stage">
    <div class="flow-title">Phase 2</div>
    <div class="flow-sub">GINEConv×3<br>+ klasyfikator</div>
    <div class="flow-badge badge-phase2">fine-tuning</div>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-stage">
    <div class="flow-title">Predykcja</div>
    <div class="flow-sub">p(śmierć)<br>logit</div>
    <div class="flow-badge badge-out">AUROC 0.850</div>
  </div>
</div>
"""


# ─── Slide builders ───────────────────────────────────────────────────────────


def slide(content: str, classes: str = "") -> str:
    return f'<section class="{classes}">{content}</section>'


def build_slides(charts: dict[str, str]) -> list[str]:
    slides = []

    # 1 — Title
    slides.append(
        slide(
            """
      <div class="slide-center" style="text-align:center">
        <div class="tag">GGSN · 2024/25</div>
        <h1 style="font-size:2.4em;margin:16px 0 8px">Temporal GNN<br>
          <span class="accent">dla predykcji śmiertelności</span></h1>
        <p class="subtitle">In-hospital mortality · MIMIC-IV 3.1 · Bio_ClinicalBERT + GINEConv</p>
        <div class="stats-row" style="margin-top:36px">
          <div class="stat"><div class="stat-num">52 727</div><div class="stat-lbl">pobytów OIOM</div></div>
          <div class="stat"><div class="stat-num">12.7%</div><div class="stat-lbl">śmiertelność</div></div>
          <div class="stat"><div class="stat-num accent">0.850</div><div class="stat-lbl">AUROC (best)</div></div>
          <div class="stat"><div class="stat-num">15</div><div class="stat-lbl">eksperymentów</div></div>
        </div>
      </div>
    """,
            "title-slide",
        )
    )

    # 2 — Problem
    slides.append(
        slide("""
      <div class="tag">Problem</div>
      <h2>Prognozowanie wczesnego zgonu — dlaczego trudne?</h2>
      <div class="two-col" style="margin-top:24px;gap:32px">
        <div>
          <ul class="spaced">
            <li><strong>Dane heterogeniczne</strong> — notatki lekarskie + sygnały fizjologiczne + kody diagnoz</li>
            <li><strong>Nierównoważne klasy</strong> — 1:7 (5 343 zgonów / 36 853 przeżyć)</li>
            <li><strong>Czas</strong> — zdarzenia w różnych momentach pobytu; kolejność ma znaczenie</li>
            <li><strong>Data leakage</strong> — łatwo nieumyślnie "widzieć przyszłość"</li>
          </ul>
        </div>
        <div style="display:flex;flex-direction:column;gap:12px">
          <div class="card" style="text-align:center">
            <div style="font-size:2.4em;font-weight:800;color:#f87171">1 : 6.9</div>
            <div style="color:#64748b;font-size:12px;margin-top:4px">imbalans klas<br>pos_weight = 6.897</div>
          </div>
          <div class="card" style="text-align:center">
            <div style="font-size:1.3em;font-weight:700;color:#818cf8">AUROC random ≈ 0.50<br>AUPRC random ≈ 0.127</div>
            <div style="color:#64748b;font-size:12px;margin-top:4px">baseline losowy</div>
          </div>
        </div>
      </div>
    """)
    )

    # 3 — Data
    slides.append(
        slide("""
      <div class="tag">Dane</div>
      <h2>MIMIC-IV 3.1 — trzy modalności</h2>
      <div class="three-col" style="margin-top:28px">
        <div class="card">
          <div class="card-icon">📄</div>
          <div class="card-title">Notatki kliniczne</div>
          <div class="card-stat">102 221</div>
          <div class="card-sub">unikalnych notatek<br>Bio_ClinicalBERT tokenizacja</div>
        </div>
        <div class="card">
          <div class="card-icon">📊</div>
          <div class="card-title">Sygnały (vitals + labs)</div>
          <div class="card-stat">55.7M</div>
          <div class="card-sub">wierszy sygnałów<br>14 typów: HR, SpO₂, MAP, ...</div>
        </div>
        <div class="card">
          <div class="card-icon">🏥</div>
          <div class="card-title">Diagnozy ICD-10</div>
          <div class="card-stat">19</div>
          <div class="card-sub">kategorii Charlson<br>wektor binarny per pobyt</div>
        </div>
      </div>
      <div class="note" style="margin-top:20px">
        Podział <strong>subject-disjoint</strong> — pacjent nigdy jednocześnie w train i val
      </div>
    """)
    )

    # 4 — Architecture Overview (animated)
    slides.append(
        slide(f"""
      <div class="tag">Architektura</div>
      <h2>Dwufazowe podejście</h2>
      {DATAFLOW_ANIMATION}
      <div class="two-col" style="margin-top:20px;gap:20px">
        <div class="card" style="border-color:#818cf844">
          <div style="color:#818cf8;font-weight:700;font-size:13px;margin-bottom:8px">PHASE 1 — Pre-training</div>
          <div style="color:#94a3b8;font-size:12px">Two-Tower contrastive learning.<br>
          TextTower (BERT) + SignalTower (1D-CNN)<br>
          wspólna przestrzeń 128-dim via InfoNCE</div>
        </div>
        <div class="card" style="border-color:#06b6d444">
          <div style="color:#06b6d4;font-weight:700;font-size:13px;margin-bottom:8px">PHASE 2 — Fine-tuning</div>
          <div style="color:#94a3b8;font-size:12px">Graf pacjenta + GINEConv×3.<br>
          Zamrożone embeddingi z Phase 1 jako cechy nodów.<br>
          Focal Loss + pooling wariants</div>
        </div>
      </div>
    """)
    )

    # 5 — Phase 1 (animated)
    slides.append(
        slide(f"""
      <div class="tag">Phase 1</div>
      <h2>Contrastive Pre-training — wyrównanie modalności</h2>
      {TWOTOWER_ANIMATION}
      <div class="note" style="margin-top:14px">
        Temperatura τ = 0.07 · Hard negative mining · Gradient accumulation
        · Wynik: <strong style="color:#818cf8">102K zamrożonych embeddingów</strong> dla Phase 2
      </div>
    """)
    )

    # 6 — Phase 2 graph (animated SVG)
    slides.append(
        slide(f"""
      <div class="tag">Phase 2 — Graf</div>
      <h2>Budowanie grafu pacjenta</h2>
      {GRAPH_ANIMATION_SVG}
      <div class="two-col" style="margin-top:8px;gap:16px;font-size:12px">
        <div>
          <span style="color:#06b6d4">●</span> <strong>Signal nodes</strong> — [one-hot(14) | val | t/24] = 16-dim → Linear(16→64)<br>
          <span style="color:#818cf8">◆</span> <strong>ICD node</strong> — wektor 19-dim Charlson → Linear(19→64), t=-1h
        </div>
        <div>
          <span style="color:#818cf8">■</span> <strong>Note nodes</strong> — embedding 128-dim z Phase 1 → Linear(128→64)<br>
          <span style="color:#4a5568">→</span> <strong>Krawędzie</strong> — skierowane, attr = Δt/24, tylko do przodu w czasie
        </div>
      </div>
    """)
    )

    # 7 — GNN Architecture
    slides.append(
        slide("""
      <div class="tag">Phase 2 — GNN</div>
      <h2>Architektura sieci grafowej</h2>
      <div class="two-col" style="gap:24px;margin-top:16px">
        <div style="font-family:monospace;font-size:11.5px;color:#94a3b8;line-height:2">
          <span style="color:#818cf8">h</span> ∈ ℝ<sup>N×64</sup>  (projekcja per typ nodu)<br>
          ├── <span style="color:#06b6d4">GINEConv</span>(64→128) + ReLU + Dropout<br>
          ├── <span style="color:#06b6d4">GINEConv</span>(128→128) + ReLU + Dropout<br>
          └── <span style="color:#06b6d4">GINEConv</span>(128→128) + ReLU + Dropout<br>
          <span style="color:#818cf8">h</span> ∈ ℝ<sup>N×128</sup><br>
          ↓ pooling (mean/attention/<span style="color:#34d399">dual</span>)<br>
          <span style="color:#818cf8">g</span> ∈ ℝ<sup>B×(128 lub 256)</sup><br>
          <span style="color:#f59e0b">⊕ demo</span> ∈ ℝ<sup>B×4</sup>  [wiek, płeć, tryb]<br>
          ↓ Linear(132→32) → ReLU → Linear(32→1)<br>
          <span style="color:#34d399">logit</span> → sigmoid → p(śmierć)
        </div>
        <div style="display:flex;flex-direction:column;gap:10px">
          <div class="card" style="border-color:#818cf844">
            <div style="color:#818cf8;font-size:12px;font-weight:700">GINEConv</div>
            <div style="color:#64748b;font-size:11px">h_v' = MLP(h_v + Σ(h_u + edge_attr))<br>uwzględnia cechy krawędzi (Δt)</div>
          </div>
          <div class="card" style="border-color:#f59e0b44">
            <div style="color:#f59e0b;font-size:12px;font-weight:700">Demografika po poolingu</div>
            <div style="color:#64748b;font-size:11px">g(128) ⊕ demo(4) → concat → 132<br>statyczna cecha pobytu, nie node</div>
          </div>
          <div class="card" style="border-color:#34d39944">
            <div style="color:#34d399;font-size:12px;font-weight:700">~57K parametrów</div>
            <div style="color:#64748b;font-size:11px">lekki model vs SOTA (110M BERT)</div>
          </div>
        </div>
      </div>
    """)
    )

    # 8 — The Bug
    slides.append(
        slide(f"""
      <div class="tag">Wykryty błąd</div>
      <h2>🐛 Temporal Leakage — AUROC 0.931 = alarm</h2>
      <div class="two-col" style="gap:24px;align-items:center">
        <img src="{charts["leakage"]}" style="width:100%;max-width:440px;border-radius:12px"/>
        <div style="display:flex;flex-direction:column;gap:12px">
          <div class="card" style="border-color:#f8717144">
            <div style="color:#f87171;font-weight:700;margin-bottom:6px">❌ Stary kod</div>
            <div style="color:#94a3b8;font-size:12px"><code>sig_rows[-max_signals:]</code><br>
            = ostatnie N sygnałów<br>= sygnały tuż przed śmiercią<br>
            <strong style="color:#f87171">AUROC 0.931 — leakage!</strong></div>
          </div>
          <div class="card" style="border-color:#34d39944">
            <div style="color:#34d399;font-weight:700;margin-bottom:6px">✅ Poprawka</div>
            <div style="color:#94a3b8;font-size:12px"><code>sig_rows[:max_signals]</code><br>
            = pierwsze N sygnałów<br>= wczesna faza pobytu<br>
            <strong style="color:#34d399">AUROC 0.820 — realny wynik</strong></div>
          </div>
        </div>
      </div>
    """)
    )

    # 9 — AUROC chart
    slides.append(
        slide(f"""
      <div class="tag">Wyniki</div>
      <h2>AUROC — 13 eksperymentów</h2>
      <img src="{charts["auroc"]}" style="width:100%;max-width:820px;border-radius:12px;margin-top:8px"/>
    """)
    )

    # 10 — AUPRC chart
    slides.append(
        slide(f"""
      <div class="tag">Wyniki</div>
      <h2>AUPRC — 3.6× powyżej losowego</h2>
      <img src="{charts["auprc"]}" style="width:100%;max-width:820px;border-radius:12px;margin-top:8px"/>
      <div class="note">AUPRC ważniejsza niż AUROC przy silnym imbalansie klas (1:7)</div>
    """)
    )

    # 11 — Contributions
    slides.append(
        slide(f"""
      <div class="tag">Analiza</div>
      <h2>Wkład każdego komponentu</h2>
      <div class="two-col" style="gap:24px;align-items:center">
        <img src="{charts["contributions"]}" style="width:100%;max-width:520px;border-radius:12px"/>
        <div style="display:flex;flex-direction:column;gap:10px">
          <div class="card" style="border-color:#34d39944">
            <div style="color:#34d399;font-weight:700;font-size:13px">✅ Co działa</div>
            <ul style="font-size:12px;color:#94a3b8;margin:8px 0 0;padding-left:16px;line-height:1.9">
              <li>Notatki kliniczne: <strong>+0.045</strong> AUROC</li>
              <li>Demografika: <strong>+0.012</strong></li>
              <li>Attention pooling: <strong>+0.015</strong></li>
              <li>Focal loss γ=2: lepsza sens@95spec</li>
            </ul>
          </div>
          <div class="card" style="border-color:#f8717144">
            <div style="color:#f87171;font-weight:700;font-size:13px">❌ Co nie działa</div>
            <ul style="font-size:12px;color:#94a3b8;margin:8px 0 0;padding-left:16px;line-height:1.9">
              <li>ICD node: 56% stays = zero Charlson</li>
              <li>All-stay signals: nie bije ±2h okna</li>
              <li>E2E BERT: catastrophic forgetting</li>
            </ul>
          </div>
        </div>
      </div>
    """)
    )

    # 12 — Best model
    slides.append(
        slide("""
      <div class="tag">Najlepszy model</div>
      <h2><span class="accent">demo_attention_focal2</span> — AUROC 0.850</h2>
      <div class="three-col" style="margin-top:24px">
        <div class="card" style="text-align:center;border-color:#fbbf2444">
          <div style="font-size:2.2em;font-weight:800;color:#fbbf24">0.850</div>
          <div style="color:#64748b;font-size:12px">AUROC</div>
        </div>
        <div class="card" style="text-align:center;border-color:#34d39944">
          <div style="font-size:2.2em;font-weight:800;color:#34d399">0.465</div>
          <div style="color:#64748b;font-size:12px">AUPRC</div>
        </div>
        <div class="card" style="text-align:center;border-color:#06b6d444">
          <div style="font-size:2.2em;font-weight:800;color:#06b6d4">0.388</div>
          <div style="color:#64748b;font-size:12px">sens@95spec</div>
        </div>
      </div>
      <div class="card" style="margin-top:16px;font-family:monospace;font-size:12px;color:#94a3b8">
        <span style="color:#34d399">uv run python -m src.training.train_gnn</span>
        --pooling <span style="color:#818cf8">attention</span>
        --focal-gamma <span style="color:#818cf8">2.0</span>
        --demo-path <span style="color:#06b6d4">data/processed/demographics.csv</span><br>
        <span style="color:#64748b"># hidden_dim=128, n_layers=3, dropout=0.3, lr=1e-3</span>
      </div>
      <div class="note" style="margin-top:12px">~57 000 parametrów · 58 epok · waga pos_weight=6.897</div>
    """)
    )

    # 13 — SOTA comparison
    slides.append(
        slide(f"""
      <div class="tag">Kontekst</div>
      <h2>Nasz wynik vs SOTA</h2>
      <div class="two-col" style="gap:28px;align-items:center">
        <img src="{charts["sota"]}" style="width:100%;max-width:440px;border-radius:12px"/>
        <div style="display:flex;flex-direction:column;gap:12px">
          <div class="card">
            <div style="color:#e2e8f0;font-weight:700;margin-bottom:8px">Dlaczego gap 0.030?</div>
            <ul style="font-size:12px;color:#94a3b8;padding-left:16px;line-height:2">
              <li>SOTA: end-to-end BERT (110M params)</li>
              <li>My: zamrożone embeddingi (Phase 1 bottleneck)</li>
              <li>SOTA: multi-GPU + gradient checkpointing</li>
              <li>My: RTX 4060 8GB VRAM</li>
            </ul>
          </div>
          <div class="card" style="border-color:#34d39944">
            <div style="color:#34d399;font-weight:700;margin-bottom:8px">Dlaczego to dobry wynik</div>
            <ul style="font-size:12px;color:#94a3b8;padding-left:16px;line-height:2">
              <li>57K params vs 110M — 2000× mniejszy model</li>
              <li>AUPRC 0.465 = 3.6× powyżej losowego</li>
              <li>Systematyczne ablacje — dojrzałość badawcza</li>
            </ul>
          </div>
        </div>
      </div>
    """)
    )

    # 14 — Takeaways
    slides.append(
        slide("""
      <div class="tag">Wnioski</div>
      <h2>5 kluczowych lekcji</h2>
      <div style="display:flex;flex-direction:column;gap:12px;margin-top:20px">
        <div class="takeaway">
          <span class="tk-num accent">01</span>
          <div><strong>Multimodalne GNN działają</strong> — ablacja signal_only potwierdza: notatki kliniczne dają +0.045 AUROC</div>
        </div>
        <div class="takeaway">
          <span class="tk-num" style="color:#34d399">02</span>
          <div><strong>Phase 1 contrastive pre-training → dobre zamrożone reprezentacje</strong> — E2E fine-tuning niszczy je (catastrophic forgetting, AUROC 0.790)</div>
        </div>
        <div class="takeaway">
          <span class="tk-num" style="color:#f59e0b">03</span>
          <div><strong>Temporal leakage jest subtelny</strong> — AUROC 0.931 = alarm. Ostatnie N sygnałów = sygnały przed śmiercią → pierwsze N = realny wynik 0.820</div>
        </div>
        <div class="takeaway">
          <span class="tk-num" style="color:#06b6d4">04</span>
          <div><strong>Demografika to prosty bonus</strong> — 4 cechy (wiek, płeć, tryb przyjęcia) dają +0.012 AUROC jako graph-level feature</div>
        </div>
        <div class="takeaway">
          <span class="tk-num" style="color:#f87171">05</span>
          <div><strong>Negatywne wyniki to też wyniki</strong> — ICD node (56% pustych), all-stay signals, e2e fine-tuning → udokumentowane jako wartość badawcza</div>
        </div>
      </div>
    """)
    )

    return slides


# ─── HTML template ────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg:      #0a0a0a;
  --surface: #111111;
  --card:    #1a1a2e;
  --accent:  #818cf8;
  --cyan:    #06b6d4;
  --green:   #34d399;
  --red:     #f87171;
  --yellow:  #fbbf24;
  --text:    #e2e8f0;
  --muted:   #64748b;
  --border:  #2d2d4e;
}

.reveal-viewport { background: var(--bg); }
.reveal .slides   { text-align: left; }
.reveal .slides section {
  font-family: 'Inter', sans-serif;
  color: var(--text);
  padding: 28px 40px;
  height: 100%;
  box-sizing: border-box;
}

/* ---- Typography ---- */
.reveal h1, .reveal h2, .reveal h3 {
  font-family: 'Inter', sans-serif;
  font-weight: 800;
  letter-spacing: -0.025em;
  color: var(--text);
  text-transform: none;
  margin: 0 0 12px;
}
.reveal h1 { font-size: 2.4em; line-height: 1.1; }
.reveal h2 { font-size: 1.55em; line-height: 1.2; }
.reveal p  { font-size: 0.9em; color: var(--muted); }
.accent    { color: var(--accent); }

/* ---- Tag chip ---- */
.tag {
  display: inline-block;
  background: #1e1e3a;
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 3px 12px;
  font-size: 11px;
  font-weight: 600;
  color: var(--accent);
  letter-spacing: 0.05em;
  text-transform: uppercase;
  margin-bottom: 10px;
}

/* ---- Cards ---- */
.card {
  background: #13131f;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px;
}
.card-icon  { font-size: 1.6em; margin-bottom: 6px; }
.card-title { font-size: 13px; font-weight: 700; color: var(--text); margin-bottom: 4px; }
.card-stat  { font-size: 1.8em; font-weight: 800; color: var(--accent); line-height: 1.1; }
.card-sub   { font-size: 11px; color: var(--muted); margin-top: 4px; }

/* ---- Layout helpers ---- */
.two-col   { display: grid; grid-template-columns: 1fr 1fr; align-items: start; gap: 20px; }
.three-col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
.slide-center { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; }

/* ---- Stats row (title slide) ---- */
.stats-row { display: flex; gap: 28px; }
.stat { text-align: center; }
.stat-num { font-size: 2em; font-weight: 800; color: var(--text); line-height: 1; }
.stat-lbl { font-size: 11px; color: var(--muted); margin-top: 4px; }

/* ---- Lists ---- */
.reveal ul.spaced { list-style: none; padding: 0; }
.reveal ul.spaced li { padding: 6px 0; font-size: 0.88em; color: var(--muted); }
.reveal ul.spaced li::before { content: "→ "; color: var(--accent); }
.reveal ul.spaced li strong { color: var(--text); }

/* ---- Note bar ---- */
.note {
  background: #0f172a;
  border-left: 3px solid var(--accent);
  border-radius: 0 6px 6px 0;
  padding: 8px 14px;
  font-size: 12px;
  color: var(--muted);
}
.note strong { color: var(--text); }

/* ---- Takeaway rows ---- */
.takeaway {
  display: flex;
  align-items: center;
  gap: 16px;
  background: #13131f;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 16px;
  font-size: 13px;
  color: var(--muted);
}
.takeaway strong { color: var(--text); }
.tk-num { font-size: 1.4em; font-weight: 800; min-width: 36px; }

/* ---- Title slide specifics ---- */
.title-slide .subtitle { font-size: 14px; color: var(--muted); margin: 0; }

/* ---- Code ---- */
code {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 0.92em;
  background: #1e1e2e;
  padding: 1px 5px;
  border-radius: 4px;
  color: #06b6d4;
}

/* ---- Progress / controls ---- */
.reveal .progress       { color: var(--accent); }
.reveal .slide-number   { background: transparent; color: var(--muted); font-size: 11px; }

/* ---- Fragment animation ---- */
.reveal .fragment { opacity: 0.15; }
.reveal .fragment.visible { opacity: 1; }
"""


def build_html(charts: dict[str, str]) -> str:
    slides_html = "\n".join(build_slides(charts))

    return f"""<!DOCTYPE html>
<html lang="pl">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Temporal GNN — GGSN Presentation</title>
  <link rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.css"/>
  <link rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/theme/black.css"/>
  <style>{CSS}</style>
</head>
<body>
<div class="reveal">
  <div class="slides">
    {slides_html}
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.js"></script>
<script>
  Reveal.initialize({{
    hash: true,
    slideNumber: 'c/t',
    transition: 'fade',
    transitionSpeed: 'fast',
    backgroundTransition: 'none',
    controls: true,
    progress: true,
    center: false,
    width: 1280,
    height: 720,
    margin: 0.04,
    plugins: [],
  }});
</script>
</body>
</html>
"""


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print("Generowanie wykresów…")
    charts = {
        "auroc": chart_auroc(),
        "auprc": chart_auprc(),
        "leakage": chart_leakage(),
        "contributions": chart_contributions(),
        "sota": chart_sota(),
    }
    print(f"  {len(charts)} wykresów gotowych.")

    print("Budowanie HTML…")
    html = build_html(charts)

    OUT.write_text(html, encoding="utf-8")
    size_kb = OUT.stat().st_size // 1024
    print(f"  Zapisano: {OUT}  ({size_kb} KB)")
    print("\n  Otwórz w przeglądarce:")
    print(f"    file://{OUT.resolve()}")


if __name__ == "__main__":
    main()
