"""
Dydaktyczna animacja Phase 2 — konstrukcja heterogenicznego grafu pacjenta
i predykcja śmiertelności przez GINEConv.

Cele:
1. Pokazać że pobyt na OIOM = ciąg eventów w czasie (sygnały + notatki + ICD).
2. Pokazać konstrukcję grafu: węzły 3 typów + krawędzie temporalne (Δt).
3. Pokazać message passing GINEConv — embedding nodu się aktualizuje
   na podstawie sąsiadów + edge attribute.
4. Pokazać pooling + demografika + klasyfikator → prawdopodobieństwo śmierci.

Skrypt CZYSTO DYDAKTYCZNY — dane syntetyczne, mały graf (7 eventów).

Uruchomienie:
    cd GGSN_Projektowe
    uvx --from manim manim -ql src/visualization/animate_patient_graph.py PatientGraphPipeline
    uvx --from manim manim -qh src/visualization/animate_patient_graph.py PatientGraphPipeline
"""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY,
    LEFT,
    RED,
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
    Indicate,
    Line,
    Rectangle,
    Scene,
    Text,
    VGroup,
    Write,
    interpolate_color,
)

# Kolory per typ węzła (zgodne z RAPORT.md sekcja 1.2)
SIGNAL_COLOR = BLUE
NOTE_COLOR = GREEN
ICD_COLOR = "#FF7F50"  # coral
DEMO_COLOR = YELLOW

# Syntetyczny pobyt: (typ, czas_h, etykieta)
# typ: 0=signal, 1=note, 2=icd
EVENTS = [
    (2, -1.0, "ICD"),  # ICD node — wiedza a priori, t=-1
    (0, 0.5, "HR"),  # tętno
    (0, 1.8, "SpO2"),  # saturacja
    (1, 2.5, "note₁"),  # notatka radiologiczna
    (0, 3.2, "MAP"),  # ciśnienie
    (1, 5.0, "note₂"),
    (0, 6.5, "Lac"),  # mleczan (lab)
]

# Pozycje X dla osi czasu (po normalizacji do okna -1..7h)
X_LEFT, X_RIGHT = -5.5, 5.5


def time_to_x(t: float, t_min: float = -1.0, t_max: float = 7.0) -> float:
    """Mapuj godziny od intime na pozycję X na ekranie."""
    return X_LEFT + (t - t_min) / (t_max - t_min) * (X_RIGHT - X_LEFT)


def node_color(node_type: int) -> str:
    return [SIGNAL_COLOR, NOTE_COLOR, ICD_COLOR][node_type]


def node_y(node_type: int) -> float:
    """Lekkie pionowe przesunięcie per typ żeby krawędzie się nie nakładały."""
    return {0: -0.4, 1: 0.6, 2: 0.0}[node_type]


def make_node(node_type: int, label: str, t: float) -> VGroup:
    color = node_color(node_type)
    x, y = time_to_x(t), node_y(node_type)
    circle = Circle(radius=0.22, color=color, stroke_width=3).set_fill(color, opacity=0.4)
    circle.move_to([x, y, 0])
    txt = Text(label, font_size=14, color=WHITE).move_to(circle.get_center())
    return VGroup(circle, txt)


def edge_color(dt: float, dt_max: float = 8.0) -> str:
    """Krótkie krawędzie = jasne (silne), długie = ciemne (słabe)."""
    t = float(np.clip(dt / dt_max, 0.0, 1.0))
    return interpolate_color(WHITE, GREY, t)


class PatientGraphPipeline(Scene):
    def construct(self):
        self._scene_intro()
        self._scene_timeline()
        self._scene_build_graph()
        self._scene_message_passing()
        self._scene_classify()
        self._scene_outro()

    # -----------------------------------------------------------------------
    def _scene_intro(self) -> None:
        title = Text("Phase 2 — Temporal Heterogeneous GNN", font_size=38, weight="BOLD")
        sub = Text(
            "Graf pacjenta na OIOM → predykcja śmiertelności wewnątrzszpitalnej",
            font_size=22,
            color=GREY,
        )
        sub.next_to(title, DOWN, buff=0.3)
        eq = Text(
            "logit = Classifier( Pool( GINEConv³( signal ∪ note ∪ icd ) ) ⊕ demo )",
            font_size=20,
            color=YELLOW,
        )
        eq.next_to(sub, DOWN, buff=0.5)

        self.play(Write(title))
        self.play(FadeIn(sub))
        self.play(FadeIn(eq))
        self.wait(2.0)
        self.play(FadeOut(VGroup(title, sub, eq)))

    # -----------------------------------------------------------------------
    def _scene_timeline(self) -> None:
        section = Text("Krok 1: pobyt na OIOM = ciąg eventów w czasie", font_size=24).to_edge(
            UP, buff=0.3
        )
        self.play(FadeIn(section))

        # Oś czasu
        axis = Line([X_LEFT, -1.6, 0], [X_RIGHT, -1.6, 0], color=GREY, stroke_width=2)
        t_min_lab = Text("t = -1h (ICD)", font_size=14, color=GREY).next_to(
            axis.get_start(), DOWN, buff=0.2
        )
        t_max_lab = Text("t = +7h", font_size=14, color=GREY).next_to(
            axis.get_end(), DOWN, buff=0.2
        )
        intime_mark = Line(
            [time_to_x(0), -1.7, 0], [time_to_x(0), -1.5, 0], color=WHITE, stroke_width=3
        )
        intime_lab = Text("intime", font_size=14, color=WHITE).next_to(intime_mark, DOWN, buff=0.1)

        self.play(Create(axis), FadeIn(t_min_lab), FadeIn(t_max_lab))
        self.play(FadeIn(intime_mark), FadeIn(intime_lab))

        # Legenda 3 typów węzłów
        legend_items = [
            (SIGNAL_COLOR, "signal node — [one_hot(14) | val | t/24] → R¹⁶"),
            (NOTE_COLOR, "note node — embedding z Phase 1 → R¹²⁸"),
            (ICD_COLOR, "ICD node — Charlson comorbidity → R¹⁹ (1× per stay, t=-1)"),
        ]
        legend = VGroup()
        for i, (col, txt) in enumerate(legend_items):
            dot = Dot(color=col, radius=0.12)
            label = Text(txt, font_size=15, color=WHITE).next_to(dot, RIGHT, buff=0.2)
            row = VGroup(dot, label).move_to(LEFT * 2.0 + UP * (2.2 - i * 0.5))
            legend.add(row)
        legend.to_edge(LEFT, buff=0.4)
        self.play(FadeIn(legend))
        self.wait(1.5)

        self.play(FadeOut(VGroup(legend, section)))
        # Zostaw oś — będzie używana w następnej scenie
        self._axis_group = VGroup(axis, t_min_lab, t_max_lab, intime_mark, intime_lab)

    # -----------------------------------------------------------------------
    def _scene_build_graph(self) -> None:
        section = Text("Krok 2: budowa heterogenicznego grafu temporalnego", font_size=24).to_edge(
            UP, buff=0.3
        )
        self.play(FadeIn(section))

        # Pojawiają się węzły jeden po drugim (w kolejności czasowej)
        self._nodes: list[VGroup] = []
        sorted_events = sorted(EVENTS, key=lambda e: e[1])
        for ev in sorted_events:
            node_type, t, label = ev
            n = make_node(node_type, label, t)
            # Pionowa kreseczka w dół do osi czasu (timestamp drop)
            drop = Line(
                [n[0].get_center()[0], n[0].get_center()[1] - 0.22, 0],
                [n[0].get_center()[0], -1.6, 0],
                color=node_color(node_type),
                stroke_width=1.5,
                stroke_opacity=0.4,
            )
            self.play(FadeIn(n), Create(drop), run_time=0.35)
            self._nodes.append(VGroup(n, drop))

        self.wait(0.5)

        # Teraz krawędzie temporalne — wszystkie pary (i,j) z t_i < t_j
        sub = Text(
            "Krawędzie: skierowane (wcześniejszy → późniejszy),  edge_attr = Δt/24h",
            font_size=18,
            color=YELLOW,
        ).next_to(section, DOWN, buff=0.2)
        self.play(FadeIn(sub))

        self._edges = VGroup()
        n_events = len(sorted_events)
        for i in range(n_events):
            new_edges = VGroup()
            for j in range(i + 1, n_events):
                # Oblicz dt
                _, t_i, _ = sorted_events[i]
                _, t_j, _ = sorted_events[j]
                dt = t_j - t_i

                src = self._nodes[i][0][0]  # Circle of source node
                dst = self._nodes[j][0][0]
                edge = Arrow(
                    src.get_center(),
                    dst.get_center(),
                    buff=0.25,
                    stroke_width=1.5,
                    color=edge_color(dt),
                    max_tip_length_to_length_ratio=0.04,
                )
                new_edges.add(edge)
            if len(new_edges) > 0:
                self.play(Create(new_edges), run_time=0.4)
                self._edges.add(*new_edges)

        # Stats
        n_nodes = len(self._nodes)
        n_edges = n_nodes * (n_nodes - 1) // 2
        stats = Text(
            f"{n_nodes} węzłów,  {n_edges} krawędzi  (O(n²) — dense temporal graph)",
            font_size=16,
            color=GREY,
        ).to_edge(DOWN, buff=0.2)
        self.play(FadeIn(stats))
        self.wait(1.5)

        self.play(FadeOut(VGroup(section, sub, stats, self._axis_group)))

    # -----------------------------------------------------------------------
    def _scene_message_passing(self) -> None:
        section = Text("Krok 3: GINEConv × 3 — message passing z edge attributes", font_size=24)
        section.to_edge(UP, buff=0.3)
        self.play(FadeIn(section))

        formula = Text(
            "h_v' = MLP( h_v + Σ_{u∈N(v)} (h_u + edge_attr_uv) )",
            font_size=20,
            color=YELLOW,
        ).next_to(section, DOWN, buff=0.2)
        self.play(Write(formula))
        self.wait(0.8)

        # Wybierz "fokus" node = note₁ (środek grafu czasowo)
        focus_idx = next(
            i for i, e in enumerate(sorted(EVENTS, key=lambda e: e[1])) if e[2] == "note₁"
        )
        focus = self._nodes[focus_idx][0][0]  # circle

        # Podświetl fokus
        focus_ring = Circle(radius=0.32, color=YELLOW, stroke_width=4).move_to(focus.get_center())
        self.play(Create(focus_ring))

        # Pokaż 3 iteracje message passing
        for layer in range(1, 4):
            layer_text = Text(
                f"Warstwa {layer}/3 — agregacja od sąsiadów",
                font_size=18,
                color=WHITE,
            ).to_edge(DOWN, buff=0.4)
            self.play(FadeIn(layer_text))

            # Animuj "wiadomości" płynące do fokusu od wszystkich pozostałych
            messages = VGroup()
            for i, node_grp in enumerate(self._nodes):
                if i == focus_idx:
                    continue
                src_circle = node_grp[0][0]
                # Mała kropka która "leci" od src do focus
                msg = Dot(point=src_circle.get_center(), radius=0.07, color=src_circle.color)
                messages.add(msg)
            self.add(messages)

            anims = []
            for i, msg in enumerate(messages):
                idx = i if i < focus_idx else i + 1
                src_pos = self._nodes[idx][0][0].get_center()
                # Refresh src pos w razie ruchu
                msg.move_to(src_pos)
                anims.append(msg.animate.move_to(focus.get_center()))
            self.play(*anims, run_time=1.0)

            # Po dotarciu: pulse fokusu (embedding się aktualizuje)
            self.play(Indicate(focus, color=YELLOW, scale_factor=1.3), run_time=0.5)
            self.play(FadeOut(messages), FadeOut(layer_text), run_time=0.3)

        self.wait(0.5)
        self.play(FadeOut(VGroup(section, formula, focus_ring)))

    # -----------------------------------------------------------------------
    def _scene_classify(self) -> None:
        section = Text("Krok 4: pooling + demografika → klasyfikator", font_size=24).to_edge(
            UP, buff=0.3
        )
        self.play(FadeIn(section))

        # Wszystkie węzły grafu pulsują, potem zwijają się do jednego wektora "g"
        graph_group = VGroup(*[n[0] for n in self._nodes])
        for n in self._nodes:
            n[1].set_opacity(0.0)  # ukryj kreseczki

        # Wszystkie krawędzie znikają
        self.play(FadeOut(self._edges))

        # Pooling: węzły lecą do jednego punktu (attention pooling)
        pool_target = np.array([-2.0, -0.5, 0.0])
        anims = [n[0].animate.move_to(pool_target).scale(0.4) for n in self._nodes]
        pool_label = Text("AttentionalAggregation", font_size=18, color=YELLOW).move_to(
            pool_target + np.array([0, 0.8, 0])
        )
        self.play(*anims, FadeIn(pool_label), run_time=1.2)

        # Wektor "g" w R^128
        g_box = Rectangle(width=1.6, height=0.5, color=YELLOW, stroke_width=3).set_fill(
            YELLOW, opacity=0.15
        )
        g_box.move_to(pool_target + np.array([0, -0.8, 0]))
        g_label = Text("g ∈ R¹²⁸", font_size=18, color=YELLOW).move_to(g_box.get_center())
        self.play(FadeOut(VGroup(*[n[0] for n in self._nodes])), Create(g_box), FadeIn(g_label))

        # Demografika obok
        demo_box = Rectangle(width=1.2, height=0.5, color=DEMO_COLOR, stroke_width=3).set_fill(
            DEMO_COLOR, opacity=0.15
        )
        demo_box.move_to(pool_target + np.array([2.6, -0.8, 0]))
        demo_label = Text("demo ∈ R⁴", font_size=16, color=DEMO_COLOR).move_to(
            demo_box.get_center()
        )
        demo_sub = Text("(age, gender,\nis_emer, is_elec)", font_size=11, color=GREY).next_to(
            demo_box, DOWN, buff=0.1
        )
        self.play(Create(demo_box), FadeIn(demo_label), FadeIn(demo_sub))

        # Concat → klasyfikator
        concat_arrow_g = Arrow(
            g_box.get_right(),
            [g_box.get_right()[0] + 0.7, g_box.get_right()[1], 0],
            buff=0.05,
            stroke_width=2,
            color=WHITE,
        )
        concat_arrow_d = Arrow(
            demo_box.get_left(),
            [demo_box.get_left()[0] - 0.7, demo_box.get_left()[1], 0],
            buff=0.05,
            stroke_width=2,
            color=WHITE,
        )
        concat_box = Rectangle(width=1.4, height=0.5, color=WHITE, stroke_width=2).set_fill(
            WHITE, opacity=0.1
        )
        concat_box.move_to([1.4, -1.3, 0])
        concat_label = Text("R¹³²", font_size=18, color=WHITE).move_to(concat_box.get_center())
        self.play(
            Create(concat_arrow_g), Create(concat_arrow_d), Create(concat_box), FadeIn(concat_label)
        )

        # Klasyfikator
        clf_arrow = Arrow(
            concat_box.get_right(),
            [concat_box.get_right()[0] + 0.8, concat_box.get_right()[1], 0],
            buff=0.05,
            stroke_width=2.5,
            color=WHITE,
        )
        clf_box = Rectangle(width=2.0, height=0.7, color=RED, stroke_width=3).set_fill(
            RED, opacity=0.2
        )
        clf_box.move_to([4.0, -1.3, 0])
        clf_lab = Text("Linear(132→32→1)", font_size=15, color=WHITE).move_to(clf_box.get_center())
        self.play(Create(clf_arrow), Create(clf_box), FadeIn(clf_lab))

        # Output: prawdopodobieństwo śmierci
        prob_arrow = Arrow(
            clf_box.get_right(),
            [clf_box.get_right()[0] + 0.7, clf_box.get_right()[1], 0],
            buff=0.05,
            stroke_width=2.5,
            color=RED,
        )
        prob_box = Rectangle(width=1.3, height=0.7, color=RED, stroke_width=3).set_fill(
            RED, opacity=0.4
        )
        prob_box.move_to([6.0, -1.3, 0])
        prob_lab = Text("p = 0.72", font_size=20, color=WHITE, weight="BOLD").move_to(
            prob_box.get_center()
        )
        prob_sub = Text("mortality", font_size=12, color=GREY).next_to(prob_box, DOWN, buff=0.1)
        self.play(Create(prob_arrow), Create(prob_box), FadeIn(prob_lab), FadeIn(prob_sub))

        self.wait(2.0)
        # Sprzątanie
        self.play(
            FadeOut(
                VGroup(
                    section,
                    g_box,
                    g_label,
                    demo_box,
                    demo_label,
                    demo_sub,
                    concat_arrow_g,
                    concat_arrow_d,
                    concat_box,
                    concat_label,
                    clf_arrow,
                    clf_box,
                    clf_lab,
                    prob_arrow,
                    prob_box,
                    prob_lab,
                    prob_sub,
                    pool_label,
                )
            )
        )

    # -----------------------------------------------------------------------
    def _scene_outro(self) -> None:
        title = Text("Wynik na MIMIC-IV (52 727 pobytów)", font_size=32, weight="BOLD")
        title.to_edge(UP, buff=1.0)

        # 3 metryki w boxach
        metrics = [
            ("AUROC", "0.850", GREEN),
            ("AUPRC", "0.465", BLUE),
            ("Brier", "0.195", YELLOW),
        ]
        boxes = VGroup()
        for i, (name, value, color) in enumerate(metrics):
            box = Rectangle(width=2.5, height=1.5, color=color, stroke_width=3).set_fill(
                color, opacity=0.15
            )
            n = Text(name, font_size=22, color=color)
            v = Text(value, font_size=42, color=WHITE, weight="BOLD")
            grp = VGroup(box, VGroup(n, v).arrange(DOWN, buff=0.15).move_to(box.get_center()))
            grp.move_to(LEFT * 4.0 + RIGHT * 4.0 * i / 2.0)
            boxes.add(grp)

        sub = Text(
            "demo_attention_focal2  —  najlepszy z 14 wariantów ablacyjnych",
            font_size=18,
            color=GREY,
        )
        sub.next_to(boxes, DOWN, buff=0.6)

        self.play(Write(title))
        for grp in boxes:
            self.play(FadeIn(grp), run_time=0.4)
        self.play(FadeIn(sub))
        self.wait(3.0)
        self.play(FadeOut(VGroup(title, boxes, sub)))
