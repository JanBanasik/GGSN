"""
Animacja ewolucji macierzy podobieństw cosine(text, signal) w trakcie treningu.

Wejście:
    data/snapshots/run_<ts>/epoch_NNN/similarity_matrix.npy  (N×N)
    data/snapshots/run_<ts>/config.json

Wyjście:
    media/videos/animate_similarity/1080p60/SimilarityEvolution.mp4

Idea: każda klatka = macierz podobieństw z jednej epoki, kolor = colormap(value).
W epoce 0 macierz jest hałaśliwa; w trakcie treningu przekątna stopniowo
świeci (text_i ↔ signal_i, positive pairs).

Uruchomienie:
    cd GGSN_Projektowe
    uv run manim -pqh src/visualization/animate_similarity.py SimilarityEvolution
        -pqh = preview, quality high (1080p60)
        -pql = preview, quality low (480p15) — szybciej
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from manim import (
    BLUE_E,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Create,
    FadeIn,
    Rectangle,
    Scene,
    Text,
    VGroup,
    interpolate_color,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "snapshots"


def latest_run() -> Path:
    """Return $GGSN_RUN_DIR if set, else newest run_* dir under data/snapshots/."""
    override = os.environ.get("GGSN_RUN_DIR")
    if override:
        return Path(override)
    runs = sorted(SNAPSHOTS_DIR.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No runs in {SNAPSHOTS_DIR}")
    return runs[-1]


def load_matrices(run_dir: Path) -> tuple[list[np.ndarray], list[int]]:
    """Returns (matrices_in_order, epoch_indices)."""
    epoch_dirs = sorted(run_dir.glob("epoch_*"))
    matrices, epochs = [], []
    for d in epoch_dirs:
        path = d / "similarity_matrix.npy"
        if path.exists():
            matrices.append(np.load(path))
            epochs.append(int(d.name.split("_")[1]))
    if not matrices:
        raise FileNotFoundError(f"No similarity_matrix.npy snapshots in {run_dir}")
    return matrices, epochs


def value_to_color(v: float, vmin: float, vmax: float):
    """Map similarity value to a color: BLUE_E (low) → WHITE → YELLOW (high)."""
    t = (v - vmin) / max(1e-9, vmax - vmin)
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.5:
        return interpolate_color(BLUE_E, WHITE, t * 2.0)
    return interpolate_color(WHITE, YELLOW, (t - 0.5) * 2.0)


class SimilarityEvolution(Scene):
    """Animuje N×N macierz cosine sim epoka-po-epoce."""

    # Config
    CELL_SIZE = 0.20  # większe komórki = wyraźniejszy kontrast
    GRID_MAX_N = 16  # mniej komórek = każda lepiej widoczna
    HOLD_PER_EPOCH = 1.0  # wolniej = łatwiej zobaczyć zmiany
    PER_EPOCH_NORMALIZE = True  # normalize colors per-frame (relative diagonal pop)

    def construct(self):
        run_dir = latest_run()
        config = json.loads((run_dir / "config.json").read_text())
        matrices, epochs = load_matrices(run_dir)

        # Przytnij N dla czytelności
        n_full = matrices[0].shape[0]
        n = min(n_full, self.GRID_MAX_N)
        matrices = [m[:n, :n] for m in matrices]

        # Skala kolorów: per-epoch (każda klatka pokazuje swój relatywny kontrast)
        # albo global (porównanie absolutnych wartości między epokami)
        if self.PER_EPOCH_NORMALIZE:
            scales = [(float(m.min()), float(m.max())) for m in matrices]
        else:
            all_vals = np.concatenate([m.flatten() for m in matrices])
            vmin_g, vmax_g = float(all_vals.min()), float(all_vals.max())
            scales = [(vmin_g, vmax_g)] * len(matrices)
        vmin, vmax = scales[0]
        legend_vmin = min(s[0] for s in scales)
        legend_vmax = max(s[1] for s in scales)

        # === Tytuł i podtytuł ===
        # Użyj effective_batch jeśli config go ma (po refactor val_loss_mode=macro),
        # fallback do batch_size dla starych runów.
        eff_b = int(config.get("effective_batch", config["batch_size"]))
        baseline_eff = float(np.log(eff_b))
        accum = int(config.get("grad_accum_steps", 1))
        batch_str = (
            f"batch={config['batch_size']}×accum={accum}=eff {eff_b}"
            if accum > 1
            else f"batch={eff_b}"
        )
        title = Text("Macierz podobieństw cos(text, signal)", font_size=32, weight="BOLD").to_edge(
            UP, buff=0.4
        )
        subtitle = Text(
            f"τ={config['temperature']} · {batch_str} · baseline ln({eff_b})={baseline_eff:.2f}",
            font_size=20,
        ).next_to(title, DOWN, buff=0.15)

        # === Etykieta epoki (zmieniana między klatkami) ===
        epoch_label = Text(f"Epoka {epochs[0]:>3d} / {epochs[-1]}", font_size=28).to_edge(
            DOWN, buff=0.6
        )

        # === Diagonalna podpowiedź ===
        diag_hint = Text(
            "← przekątna = positive pairs (text_i ↔ signal_i)",
            font_size=18,
            color=YELLOW,
        ).next_to(epoch_label, UP, buff=0.2)

        # === Grid komórek ===
        grid_w = n * self.CELL_SIZE
        origin = np.array([-grid_w / 2 + self.CELL_SIZE / 2, grid_w / 2 - self.CELL_SIZE / 2, 0.0])

        cells: list[list[Rectangle]] = []
        cell_group = VGroup()
        for i in range(n):
            row: list[Rectangle] = []
            for j in range(n):
                pos = origin + np.array([j * self.CELL_SIZE, -i * self.CELL_SIZE, 0.0])
                rect = Rectangle(
                    width=self.CELL_SIZE * 0.97,
                    height=self.CELL_SIZE * 0.97,
                    stroke_width=0,
                ).move_to(pos)
                color = value_to_color(matrices[0][i, j], vmin, vmax)
                rect.set_fill(color, opacity=1.0)
                row.append(rect)
                cell_group.add(rect)
            cells.append(row)

        cell_group.shift(0.2 * DOWN)

        # === Pasek skali (legend) po prawej ===
        legend = self._build_legend(legend_vmin, legend_vmax).next_to(cell_group, RIGHT, buff=0.4)

        # === Etykieta osi ===
        x_axis_label = Text("signal_j →", font_size=18).next_to(cell_group, DOWN, buff=0.1)
        y_axis_label = Text("text_i ↓", font_size=18).rotate(0).next_to(cell_group, LEFT, buff=0.15)

        # === Render ===
        self.play(FadeIn(title), FadeIn(subtitle))
        self.play(
            Create(cell_group, run_time=1.2),
            FadeIn(legend),
            FadeIn(x_axis_label),
            FadeIn(y_axis_label),
            FadeIn(epoch_label),
            FadeIn(diag_hint),
        )
        self.wait(0.6)

        # Animacja po epokach
        for ep_pos, (epoch_idx, mat) in enumerate(
            zip(epochs[1:], matrices[1:], strict=False), start=1
        ):
            new_label = Text(f"Epoka {epoch_idx:>3d} / {epochs[-1]}", font_size=28).move_to(
                epoch_label
            )

            v_lo, v_hi = scales[ep_pos]
            anims = [epoch_label.animate.become(new_label)]
            for i in range(n):
                for j in range(n):
                    new_color = value_to_color(mat[i, j], v_lo, v_hi)
                    anims.append(cells[i][j].animate.set_fill(new_color, opacity=1.0))
            self.play(*anims, run_time=self.HOLD_PER_EPOCH)

        self.wait(1.2)

    def _build_legend(self, vmin: float, vmax: float) -> VGroup:
        """Pionowy pasek od vmin (BLUE_E) do vmax (YELLOW) z etykietami."""
        n_steps = 40
        bar_h = 3.0
        bar_w = 0.18
        steps = VGroup()
        for k in range(n_steps):
            v = vmax - (vmax - vmin) * (k / (n_steps - 1))
            color = value_to_color(v, vmin, vmax)
            seg = Rectangle(width=bar_w, height=bar_h / n_steps, stroke_width=0).set_fill(
                color, opacity=1.0
            )
            seg.shift(np.array([0.0, bar_h / 2 - k * bar_h / n_steps - bar_h / (2 * n_steps), 0.0]))
            steps.add(seg)

        max_label = Text(f"{vmax:+.2f}", font_size=14).next_to(steps, UP, buff=0.05)
        min_label = Text(f"{vmin:+.2f}", font_size=14).next_to(steps, DOWN, buff=0.05)
        title_label = Text("cos sim", font_size=14).next_to(steps, RIGHT, buff=0.1)

        return VGroup(steps, max_label, min_label, title_label)
