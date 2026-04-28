"""
Animacja ewolucji embeddingów notatek w przestrzeni UMAP-2D, kolor = mortality.

Wejście:
    data/snapshots/run_<ts>/epoch_NNN/val_embeddings.pt   ({note_id: tensor(D,)})
    data/snapshots/run_<ts>/epoch_NNN/val_labels.json     ({note_id: 0|1})

Strategia stabilnego UMAP:
    1. Fit UMAP(random_state=42) na embeddingach z OSTATNIEJ epoki.
    2. transform() na embeddingach z każdej wcześniejszej epoki.
    Dzięki temu przejście między klatkami pokazuje, jak punkty migrują w
    docelowej przestrzeni 2D (a nie w losowo nowej co klatkę).

Wyjście:
    media/videos/animate_umap/<quality>/UmapEvolution.mp4

Uruchomienie:
    cd GGSN_Projektowe
    uv run manim -ql src/visualization/animate_umap.py UmapEvolution    # szybko
    uv run manim -qh src/visualization/animate_umap.py UmapEvolution    # 1080p
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import umap
from manim import (
    BLUE,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Dot,
    FadeIn,
    RED,
    Scene,
    Text,
    Transform,
    VGroup,
    WHITE,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "snapshots"

MAX_POINTS = 250          # ile punktów rysować (dla czytelności)
PLOT_HALF_WIDTH = 4.5     # zasięg manim x ∈ [-W, +W]
PLOT_HALF_HEIGHT = 3.0    # y ∈ [-H, +H]
DOT_RADIUS = 0.05
HOLD_PER_EPOCH = 0.65


def latest_run() -> Path:
    override = os.environ.get("GGSN_RUN_DIR")
    if override:
        return Path(override)
    runs = sorted(SNAPSHOTS_DIR.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No runs in {SNAPSHOTS_DIR}")
    return runs[-1]


def load_snapshots(run_dir: Path) -> tuple[list[int], list[np.ndarray], np.ndarray, list[str]]:
    """
    Returns:
        epochs            list[int]
        embeds_per_epoch  list of (N, D) np.ndarray
        labels            (N,) int  (0/1) — ten sam ordering co rzędy w embeds
        note_ids          list[str] (długość N)
    """
    epoch_dirs = sorted(run_dir.glob("epoch_*"))
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch_NNN dirs in {run_dir}")

    # Wspólny ordering: użyj note_id z epoki ostatniej, sortowane
    last = epoch_dirs[-1]
    last_embeds: dict[str, torch.Tensor] = torch.load(last / "val_embeddings.pt", weights_only=False)
    note_ids = sorted(last_embeds.keys())
    if len(note_ids) > MAX_POINTS:
        rng = np.random.default_rng(42)
        note_ids = list(rng.choice(note_ids, MAX_POINTS, replace=False))
        note_ids.sort()

    labels_dict = json.loads((last / "val_labels.json").read_text())
    labels = np.array([int(labels_dict[nid]) for nid in note_ids])

    epochs: list[int] = []
    embeds_per_epoch: list[np.ndarray] = []
    for d in epoch_dirs:
        emb_path = d / "val_embeddings.pt"
        if not emb_path.exists():
            continue
        embeds: dict[str, torch.Tensor] = torch.load(emb_path, weights_only=False)
        # Niektóre note_id mogą się nie znaleźć (różny val ze względu na splity)
        # — w naszym setupie val_idx jest deterministyczny per run, więc OK
        try:
            arr = np.stack([embeds[nid].numpy() for nid in note_ids])
        except KeyError:
            continue
        epochs.append(int(d.name.split("_")[1]))
        embeds_per_epoch.append(arr)

    return epochs, embeds_per_epoch, labels, note_ids


def project_with_stable_umap(embeds_per_epoch: list[np.ndarray]) -> list[np.ndarray]:
    """Fit UMAP on the last epoch's embeddings, transform all earlier epochs."""
    final = embeds_per_epoch[-1]
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.15,
        random_state=42,
        metric="cosine",
    )
    reducer.fit(final)
    coords = [reducer.transform(e) for e in embeds_per_epoch]
    return coords


def normalize_coords(coords: list[np.ndarray]) -> list[np.ndarray]:
    """Skaluje wszystkie klatki do prostokąta [-W,+W] × [-H,+H] zachowując proporcje."""
    all_xy = np.concatenate(coords, axis=0)
    xmin, xmax = all_xy[:, 0].min(), all_xy[:, 0].max()
    ymin, ymax = all_xy[:, 1].min(), all_xy[:, 1].max()
    span_x = max(1e-9, xmax - xmin)
    span_y = max(1e-9, ymax - ymin)

    out = []
    for c in coords:
        x = (c[:, 0] - xmin) / span_x * (2 * PLOT_HALF_WIDTH) - PLOT_HALF_WIDTH
        y = (c[:, 1] - ymin) / span_y * (2 * PLOT_HALF_HEIGHT) - PLOT_HALF_HEIGHT
        out.append(np.stack([x, y, np.zeros_like(x)], axis=1))
    return out


class UmapEvolution(Scene):
    """Animuje migrację embeddingów w docelowej przestrzeni UMAP-2D."""

    def construct(self):
        run_dir = latest_run()
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}

        epochs, embeds_per_epoch, labels, _ = load_snapshots(run_dir)
        if len(epochs) < 2:
            self.add(Text("Nie ma wystarczająco snapshotów (≥2 epok).", font_size=28))
            self.wait(2)
            return

        print(f"[UmapEvolution] {len(epochs)} epok, {len(labels)} punktów")
        coords_2d = project_with_stable_umap(embeds_per_epoch)
        coords_2d = normalize_coords(coords_2d)

        # === Tytuł ===
        title = Text("Embeddingi notatek w przestrzeni UMAP-2D", font_size=30, weight="BOLD").to_edge(UP, buff=0.3)
        sub = Text(
            f"Bio_ClinicalBERT + InfoNCE | val={len(labels)} notatek | "
            f"kolor: mortality (czerwony=zgon)",
            font_size=18,
        ).next_to(title, DOWN, buff=0.15)

        epoch_label = Text(f"Epoka {epochs[0]:>3d} / {epochs[-1]}", font_size=24).to_edge(DOWN, buff=0.4)

        # === Punkty ===
        dots = VGroup()
        first = coords_2d[0]
        for k in range(len(labels)):
            pos = first[k]
            color = RED if labels[k] == 1 else BLUE
            d = Dot(point=pos, radius=DOT_RADIUS, color=color, fill_opacity=0.85)
            dots.add(d)

        legend = VGroup(
            Dot(radius=0.08, color=RED).shift(LEFT * 3 + UP * 0.05),
            Text("zgon (1)", font_size=18).next_to(LEFT * 2.7, RIGHT, buff=0.1),
            Dot(radius=0.08, color=BLUE).shift(LEFT * 1.2 + UP * 0.05),
            Text("przeżył (0)", font_size=18).next_to(LEFT * 0.9, RIGHT, buff=0.1),
        ).to_edge(DOWN, buff=1.1)

        self.play(FadeIn(title), FadeIn(sub))
        self.play(FadeIn(dots), FadeIn(epoch_label), FadeIn(legend))
        self.wait(0.6)

        # Animacja epoka → epoka
        for ep_idx in range(1, len(epochs)):
            new_label = Text(f"Epoka {epochs[ep_idx]:>3d} / {epochs[-1]}", font_size=24).move_to(epoch_label)
            anims = [Transform(epoch_label, new_label)]
            target_xy = coords_2d[ep_idx]
            for k, d in enumerate(dots):
                anims.append(d.animate.move_to(target_xy[k]))
            self.play(*anims, run_time=HOLD_PER_EPOCH)

        self.wait(1.5)
