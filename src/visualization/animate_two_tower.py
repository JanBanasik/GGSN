"""
Dydaktyczna animacja architektury Two-Tower + InfoNCE.

Cele:
1. Pokazać przepływ batcha B=4 par (notatka, sygnał) przez wieże.
2. Pokazać projekcje do wspólnej przestrzeni 128-D jako punkty na płaszczyźnie.
3. Pokazać InfoNCE: positive pairs (text_i ↔ signal_i) przyciągane,
   negatives odpychane.
4. Pokazać macierz podobieństw NxN i jej "świecenie" przekątnej po treningu.

Skrypt jest CZYSTO DYDAKTYCZNY — używa danych syntetycznych, nie ładuje snapshotów.
Cel: dać widzom intuicję "co to InfoNCE i jak działa Two-Tower" przed pokazaniem
realnych animacji similarity / UMAP.

Uruchomienie:
    cd GGSN_Projektowe
    uv run manim -ql src/visualization/animate_two_tower.py TwoTowerInfoNCE
    uv run manim -qh src/visualization/animate_two_tower.py TwoTowerInfoNCE
"""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    BLUE_E,
    DOWN,
    GREEN,
    GREY,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    Rectangle,
    Scene,
    Text,
    Transform,
    VGroup,
    Write,
    interpolate_color,
)

BATCH = 4
PAIR_COLORS = [BLUE, GREEN, YELLOW, "#FF7F50"]  # 4 distinct colors per pair


def make_tower_box(label: str, sublabel: str, color: str) -> VGroup:
    box = Rectangle(width=2.4, height=1.4, color=color, stroke_width=3).set_fill(
        color, opacity=0.12
    )
    title = Text(label, font_size=22, weight="BOLD", color=color)
    sub = Text(sublabel, font_size=14, color=GREY)
    txt = VGroup(title, sub).arrange(DOWN, buff=0.08)
    txt.move_to(box.get_center())
    return VGroup(box, txt)


def value_to_diag_color(v: float):
    """Kolormap dla similarity matrix po treningu: blue (low) → yellow (high)."""
    t = float(np.clip(v, 0.0, 1.0))
    if t < 0.5:
        return interpolate_color(BLUE_E, WHITE, t * 2.0)
    return interpolate_color(WHITE, YELLOW, (t - 0.5) * 2.0)


class TwoTowerInfoNCE(Scene):
    def construct(self):
        self._scene_intro()
        self._scene_towers()
        self._scene_shared_space()
        self._scene_similarity_matrix()

    # -----------------------------------------------------------------------
    def _scene_intro(self) -> None:
        title = Text("Two-Tower Contrastive Pre-training", font_size=40, weight="BOLD")
        sub = Text(
            "Tekst (notatki klinicznie) ⟷ Sygnały (vital signs + laby)", font_size=22, color=GREY
        )
        sub.next_to(title, DOWN, buff=0.3)
        formula = Text(
            "L = ½ · [ CE(S, diag) + CE(Sᵀ, diag) ]    gdzie  S[i,j] = z_text[i] · z_sig[j] / τ",
            font_size=20,
            color=YELLOW,
        )
        formula.next_to(sub, DOWN, buff=0.5)

        self.play(Write(title))
        self.play(FadeIn(sub))
        self.play(FadeIn(formula))
        self.wait(2.0)
        self.play(FadeOut(VGroup(title, sub, formula)))

    # -----------------------------------------------------------------------
    def _scene_towers(self) -> None:
        section = Text("Krok 1: Batch B=4 par przepływa przez wieże", font_size=24).to_edge(
            UP, buff=0.3
        )
        self.play(FadeIn(section))

        text_tower = make_tower_box("Text Tower", "Bio_ClinicalBERT + proj", BLUE).shift(
            LEFT * 4 + UP * 0.2
        )
        sig_tower = make_tower_box("Signal Tower", "Embed(type) + 1D-CNN", GREEN).shift(
            RIGHT * 4 + UP * 0.2
        )

        # 4 input "tokens" po lewej / 4 input "spike traces" po prawej
        text_inputs = VGroup()
        sig_inputs = VGroup()
        for i in range(BATCH):
            y = 1.5 - i * 1.0
            t_dot = Dot(point=LEFT * 6.4 + UP * y, radius=0.10, color=PAIR_COLORS[i])
            t_lab = Text(f"note_{i}", font_size=14, color=PAIR_COLORS[i]).next_to(
                t_dot, LEFT, buff=0.1
            )
            text_inputs.add(VGroup(t_dot, t_lab))

            s_dot = Dot(point=RIGHT * 6.4 + UP * y, radius=0.10, color=PAIR_COLORS[i])
            s_lab = Text(f"sig_{i}", font_size=14, color=PAIR_COLORS[i]).next_to(
                s_dot, RIGHT, buff=0.1
            )
            sig_inputs.add(VGroup(s_dot, s_lab))

        self.play(Create(text_tower), Create(sig_tower))
        self.play(FadeIn(text_inputs), FadeIn(sig_inputs))

        # Strzałki wejść do wież
        in_arrows_l = VGroup(
            *[
                Arrow(
                    text_inputs[i][0].get_right(),
                    text_tower[0].get_left() + UP * (1.5 - i * 1.0) * 0.0,
                    buff=0.15,
                    stroke_width=2.5,
                    color=PAIR_COLORS[i],
                    max_tip_length_to_length_ratio=0.10,
                )
                for i in range(BATCH)
            ]
        )
        in_arrows_r = VGroup(
            *[
                Arrow(
                    sig_inputs[i][0].get_left(),
                    sig_tower[0].get_right() + UP * (1.5 - i * 1.0) * 0.0,
                    buff=0.15,
                    stroke_width=2.5,
                    color=PAIR_COLORS[i],
                    max_tip_length_to_length_ratio=0.10,
                )
                for i in range(BATCH)
            ]
        )
        self.play(Create(in_arrows_l), Create(in_arrows_r))

        # Wyjścia: małe pudełka z napisem (B, 128) na środku
        out_text = Text("z_text\n(B, 128)\nL2-norm", font_size=14, color=BLUE).next_to(
            text_tower, DOWN, buff=0.5
        )
        out_sig = Text("z_signal\n(B, 128)\nL2-norm", font_size=14, color=GREEN).next_to(
            sig_tower, DOWN, buff=0.5
        )
        self.play(FadeIn(out_text), FadeIn(out_sig))
        self.wait(1.0)

        self.play(
            FadeOut(
                VGroup(
                    section,
                    text_inputs,
                    sig_inputs,
                    in_arrows_l,
                    in_arrows_r,
                    text_tower,
                    sig_tower,
                    out_text,
                    out_sig,
                )
            )
        )

    # -----------------------------------------------------------------------
    def _scene_shared_space(self) -> None:
        section = Text(
            "Krok 2: Wspólna przestrzeń 128-D — InfoNCE attraction/repulsion", font_size=22
        ).to_edge(UP, buff=0.3)
        self.play(FadeIn(section))

        # Okrąg jednostkowy (L2-norm) — wszystkie punkty na nim
        sphere = Circle(radius=2.5, color=GREY, stroke_width=2).shift(DOWN * 0.3)
        sphere_lab = Text("|z| = 1  (L2-normalised)", font_size=16, color=GREY).next_to(
            sphere, UP, buff=0.1
        )
        self.play(Create(sphere), FadeIn(sphere_lab))

        # Początkowe (losowe) pozycje text i signal — przed treningiem
        rng = np.random.default_rng(0)
        radius = 2.5
        text_init_angles = rng.uniform(0, 2 * np.pi, BATCH)
        sig_init_angles = rng.uniform(0, 2 * np.pi, BATCH)

        # Po treningu: text_i i sig_i powinny być blisko siebie (positive pairs)
        target_angles = np.array([0.4, 1.9, 3.4, 5.1])  # arbitrary placement

        def angle_to_point(theta: float, r: float, center: np.ndarray) -> np.ndarray:
            return center + np.array([r * np.cos(theta), r * np.sin(theta), 0.0])

        center = sphere.get_center()
        text_dots = VGroup(
            *[
                Dot(
                    angle_to_point(text_init_angles[i], radius, center),
                    radius=0.10,
                    color=PAIR_COLORS[i],
                )
                for i in range(BATCH)
            ]
        )
        sig_dots = VGroup(
            *[
                Dot(
                    angle_to_point(sig_init_angles[i], radius, center),
                    radius=0.10,
                    color=PAIR_COLORS[i],
                    fill_opacity=0.6,
                ).set_stroke(color=PAIR_COLORS[i], width=2.5)
                for i in range(BATCH)
            ]
        )
        text_labels = VGroup(
            *[
                Text(f"t_{i}", font_size=12, color=PAIR_COLORS[i]).move_to(
                    text_dots[i].get_center() + 0.3 * (text_dots[i].get_center() - center) / radius
                )
                for i in range(BATCH)
            ]
        )
        sig_labels = VGroup(
            *[
                Text(f"s_{i}", font_size=12, color=PAIR_COLORS[i]).move_to(
                    sig_dots[i].get_center() + 0.3 * (sig_dots[i].get_center() - center) / radius
                )
                for i in range(BATCH)
            ]
        )

        self.play(FadeIn(text_dots), FadeIn(sig_dots), FadeIn(text_labels), FadeIn(sig_labels))

        before_lab = Text(
            "Przed treningiem: pary t_i, s_i są daleko", font_size=18, color=GREY
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(before_lab))
        self.wait(1.0)

        # Animacja: text_i i sig_i zbiegają się do tego samego kąta (positive pair attraction)
        new_text_targets = [
            angle_to_point(target_angles[i] - 0.10, radius, center) for i in range(BATCH)
        ]
        new_sig_targets = [
            angle_to_point(target_angles[i] + 0.10, radius, center) for i in range(BATCH)
        ]

        anims = []
        for i in range(BATCH):
            anims.append(text_dots[i].animate.move_to(new_text_targets[i]))
            anims.append(sig_dots[i].animate.move_to(new_sig_targets[i]))
            anims.append(
                text_labels[i].animate.move_to(
                    new_text_targets[i] + 0.3 * (new_text_targets[i] - center) / radius
                )
            )
            anims.append(
                sig_labels[i].animate.move_to(
                    new_sig_targets[i] + 0.3 * (new_sig_targets[i] - center) / radius
                )
            )

        after_lab = Text(
            "Po treningu: pary t_i, s_i blisko, różne pary daleko", font_size=18, color=YELLOW
        ).to_edge(DOWN, buff=0.5)
        self.play(*anims, Transform(before_lab, after_lab), run_time=2.5)
        self.wait(0.7)

        # Zaznaczenie positive pair connection (zielone linie pomiędzy t_i — s_i)
        pos_lines = VGroup(
            *[
                Line(
                    text_dots[i].get_center(), sig_dots[i].get_center(), color=GREEN, stroke_width=3
                )
                for i in range(BATCH)
            ]
        )
        pos_caption = Text(
            "zielone = positive pairs (przyciągamy)", font_size=16, color=GREEN
        ).to_edge(DOWN, buff=1.1)
        self.play(Create(pos_lines), FadeIn(pos_caption))
        self.wait(1.0)

        # Negatywne (czerwone) — wszystkie t_i ↔ s_j gdzie i ≠ j
        neg_lines = VGroup()
        for i in range(BATCH):
            for j in range(BATCH):
                if i == j:
                    continue
                neg_lines.add(
                    Line(
                        text_dots[i].get_center(),
                        sig_dots[j].get_center(),
                        color="#CC2222",
                        stroke_width=1.0,
                        stroke_opacity=0.4,
                    )
                )
        neg_caption = Text(
            "czerwone = negatives (odpychamy)", font_size=16, color="#CC2222"
        ).to_edge(DOWN, buff=1.5)
        self.play(Create(neg_lines), FadeIn(neg_caption))
        self.wait(1.5)

        self.play(
            FadeOut(
                VGroup(
                    section,
                    sphere,
                    sphere_lab,
                    text_dots,
                    sig_dots,
                    text_labels,
                    sig_labels,
                    before_lab,
                    pos_lines,
                    pos_caption,
                    neg_lines,
                    neg_caption,
                )
            )
        )

    # -----------------------------------------------------------------------
    def _scene_similarity_matrix(self) -> None:
        section = Text("Krok 3: Macierz podobieństw NxN — przekątna świeci", font_size=22).to_edge(
            UP, buff=0.3
        )
        self.play(FadeIn(section))

        N = BATCH
        cell_size = 0.9
        grid_w = N * cell_size
        origin = np.array([-grid_w / 2 + cell_size / 2, grid_w / 2 - cell_size / 2, 0.0])

        # Pre-training: macierz hałaśliwa (jednorodna ~0.5)
        before_vals = np.array(
            [
                [0.55, 0.45, 0.50, 0.48],
                [0.49, 0.52, 0.51, 0.47],
                [0.46, 0.50, 0.54, 0.49],
                [0.51, 0.48, 0.47, 0.53],
            ]
        )
        # After training: przekątna ~0.95, off-diag ~0.05
        after_vals = np.array(
            [
                [0.95, 0.08, 0.10, 0.05],
                [0.06, 0.92, 0.07, 0.09],
                [0.11, 0.05, 0.94, 0.04],
                [0.07, 0.10, 0.06, 0.93],
            ]
        )

        cells = VGroup()
        cell_objs: list[list[Rectangle]] = []
        for i in range(N):
            row = []
            for j in range(N):
                pos = origin + np.array([j * cell_size, -i * cell_size, 0.0])
                rect = Rectangle(
                    width=cell_size * 0.95, height=cell_size * 0.95, stroke_width=1, color=GREY
                )
                rect.set_fill(value_to_diag_color(before_vals[i, j]), opacity=1.0)
                rect.move_to(pos)
                row.append(rect)
                cells.add(rect)
            cell_objs.append(row)

        # Etykiety osi
        col_labels = VGroup(
            *[
                Text(f"s_{j}", font_size=18).move_to(
                    origin + np.array([j * cell_size, cell_size * 0.7, 0.0])
                )
                for j in range(N)
            ]
        )
        row_labels = VGroup(
            *[
                Text(f"t_{i}", font_size=18).move_to(
                    origin + np.array([-cell_size * 0.7, -i * cell_size, 0.0])
                )
                for i in range(N)
            ]
        )

        # Wartości w komórkach
        before_texts = VGroup(
            *[
                Text(f"{before_vals[i, j]:.2f}", font_size=14, color=WHITE).move_to(
                    cell_objs[i][j].get_center()
                )
                for i in range(N)
                for j in range(N)
            ]
        )

        before_lab = Text(
            "Przed treningiem: cała macierz ~0.5 (losowo)", font_size=18, color=GREY
        ).to_edge(DOWN, buff=0.6)

        self.play(Create(cells), FadeIn(col_labels), FadeIn(row_labels), FadeIn(before_texts))
        self.play(FadeIn(before_lab))
        self.wait(1.5)

        # Animacja przejścia do "po treningu"
        anims = []
        for i in range(N):
            for j in range(N):
                anims.append(
                    cell_objs[i][j].animate.set_fill(
                        value_to_diag_color(after_vals[i, j]), opacity=1.0
                    )
                )
        after_lab = Text(
            "Po treningu: przekątna ~0.95, off-diag ~0.05  ← positive pairs wygrały",
            font_size=18,
            color=YELLOW,
        ).to_edge(DOWN, buff=0.6)

        # Tekst wartości też update
        after_texts = VGroup(
            *[
                Text(
                    f"{after_vals[i, j]:.2f}",
                    font_size=14,
                    color=("#000000" if after_vals[i, j] > 0.5 else WHITE),
                ).move_to(cell_objs[i][j].get_center())
                for i in range(N)
                for j in range(N)
            ]
        )
        anims.append(Transform(before_texts, after_texts))
        anims.append(Transform(before_lab, after_lab))

        self.play(*anims, run_time=2.0)
        self.wait(2.5)
