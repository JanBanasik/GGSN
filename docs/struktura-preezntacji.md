# Struktura prezentacji (dla prowadzącego) — v1

## Założenia
- Czas: ok. `5–6 min`.
- Wariant mówienia: poziom `mid` (intuicja + najważniejsze detale, bez wchodzenia w pełną implementację).
- Punkt startu: obecnie gotowa jest `Faza 1` (contrastive Two-Tower + InfoNCE), a `Faza 2` (Temporal GNN) jest kolejnym krokiem.
- Animacje: wstawki z `manim` jako „dowód wizualny” (intuicja -> wyniki -> next steps).

## Slajdy (kolejność + czasy)

### Slajd 1 — Cel i motywacja (ok. `0:35`)
**Co na slajdzie**
- Problem: predykcja `in-hospital mortality` na podstawie pierwszych `24h` pobytu.
- Dane: miks `tekstu (notatki)` + `sygnałów (vitals/laby)`.
- Jedno zdanie: „chcemy nauczyć reprezentacje, które rozumieją wspólną treść kliniczną obu modalności”.

**Speaker notes (1–2 zdania)**
„Patrzymy na pacjenta w czasie: z notatek i z monitora chcemy oszacować ryzyko zgonu. Problem jest trudny, bo nie mamy wygodnych etykiet na wejściu, więc używamy `contrastive learning` do nauczenia ‘wspólnego języka’ dla tekstu i sygnałów.”

---

### Slajd 2 — Pipeline w 2 fazach (ok. `0:45`)
**Co na slajdzie**
- `Faza 1`: multimodal contrastive pre-training -> `embeddings` (zamrożone do Fazy 2).
- `Faza 2`: Temporal GNN -> predykcja `P(mortality)` z grafu pacjenta.
- Podkreślenie roli: `edge_attr = Δt` (odstęp czasowy między zdarzeniami).

**Speaker notes (1–2 zdania)**
„Najpierw budujemy reprezentacje dla tekstu i sygnałów bez etykiet śmiertelności (Faza 1). Potem dokładamy dynamikę: w Fazie 2 poruszamy się po grafie zdarzeń, a czas między nimi jest jawnie kodowany jako `Δt`.”

---

### Slajd 3 — Faza 1: Two-Tower + InfoNCE (intuicja) (ok. `0:55`)
**Co na slajdzie**
- Wprowadzenie pojęć z animacji:
  - `positive pair`: `note_i ↔ signal_i` (ta sama para w batchu),
  - `negatives`: pozostałe pary w batchu (`i != j`),
  - wektory: `z_text` i `z_signal` w tej samej przestrzeni `128-D` (po L2-normalizacji).

**Animacja (wstawka)**
- `src/visualization/animate_two_tower.py` -> scena `TwoTowerInfoNCE`

**Speaker notes (1–2 zdania)**
„W Two-Tower InfoNCE pozytywne pary starają się trafić do siebie w przestrzeni embeddingów, a negatywy są odpychane. W efekcie model uczy się reprezentacji, w których ta sama treść kliniczna w tekście i w sygnale staje się podobna.”

---

### Slajd 4 — Dane v1 i co już mamy (ok. `0:40`)
**Co na slajdzie**
- Dane v1 (krótko liczby):
  - ~`10 775` stays,
  - ~`9 749` unikalnych notatek,
  - ~`314k` par (note × signal).
- Teza: „PoC + rozwinięcie do v1, gdzie poprawiliśmy etykietowanie i pipeline”.

**Speaker notes (1–2 zdania)**
„Dla v1 mamy już większą kohortę CCU/CVICU i poprawne `in-hospital mortality`. Dzięki temu trening kontrastywny ma wystarczający materiał, żeby embeddingi zaczęły łapać sens kliniczny.”

---

### Slajd 5 — Co widać po treningu: similarity matrix (ok. `0:55`)
**Co na slajdzie**
- Wniosek: rośnie kontrast `diagonal` vs `off-diagonal`.
- Interpretacja: `diagonal` odpowiada `positive pairs (i=i)`, a poza przekątną to `negatives`.

**Animacja (wstawka)**
- `src/visualization/animate_similarity.py` -> scena `SimilarityEvolution`

**Speaker notes (1–2 zdania)**
„Na similarity matrix widać bezpośrednio, czy model rozróżnia pary. Po treningu przekątna zaczyna ‘świecić’ — to znak, że pozytywne pary tekst-sygnał mają większą zgodność niż reszta.”

---

### Slajd 6 — Co widać po treningu: UMAP embeddingów (ok. `0:55`)
**Co na slajdzie**
- Punkty w `UMAP-2D` (kolor wg `mortality`).
- Dodatkowo: płynna migracja punktów między epokami (UMAP fit na ostatniej epoce, transform na wcześniejszych).

**Animacja (wstawka)**
- `src/visualization/animate_umap.py` -> scena `UmapEvolution`

**Speaker notes (1–2 zdania)**
„UMAP pokazuje, że embeddingi nie tylko ‘dopasowują się’ kontrastywnie, ale też zaczynają układać się w przestrzeni, która ma związek z mortality. To ważny sanity-check przed przejściem do GNN, które będzie dodatkowo wykorzystywać strukturę czasową.”

---

### Slajd 7 — Następne kroki (Faza 2 + poprawki Faz1) (ok. `0:40`)
**Co na slajdzie**
- Faza 2 (must-have): `graph_builder.py` + `tgnn_model.py` + `train_gnn.py`.
- Kluczowe elementy Fazy 2:
  - węzły = zdarzenia/notatki z embeddingami,
  - krawędzie skierowane chronologicznie,
  - `edge_attr = Δt`.
- Najważniejsze ulepszenia Faz1 (z roadmapy): lepszy pairing + hard negatives + skalowanie danych.

**Speaker notes (1–2 zdania)**
„Następny krok to przejście z embeddingów do dynamiki: budujemy graf pacjenta i uczymy Temporal GNN, gdzie czas między zdarzeniami jest wprost w wejściu. Równolegle poprawiamy Faza 1: chcemy więcej i ‘mądrzejszych’ negatywów oraz sensowniejszy pairing, żeby kontrast był silniejszy.”

---

## Jak wpleść animacje (praktycznie)
- Animacja `TwoTowerInfoNCE` (intuicja) w slajdzie 3: zatrzymaj się na momencie „positive pairs” i krótkim podsumowaniu.
- Animacja `SimilarityEvolution` w slajdzie 5: użyj hasła „diagonal = positive pairs”.
- Animacja `UmapEvolution` w slajdzie 6: powiedz, że kolor odpowiada mortality, a punkty migrują między epokami.

## Komendy do odtworzenia (opcjonalnie, na końcu dla prowadzącego)
> Najwygodniej odpalać tryb preview (`-pqh`) podczas przygotowania i potem podmienić MP4 w prezentacji.
```bash
# Intuicja Two-Tower + InfoNCE (syntetyczne dane)
uv run manim -pqh src/visualization/animate_two_tower.py TwoTowerInfoNCE

# Similarity matrix: bierze ostatni run z data/snapshots/ (lub GGSN_RUN_DIR)
uv run manim -pqh src/visualization/animate_similarity.py SimilarityEvolution

# UMAP embeddingów: bierze ostatni run z data/snapshots/ (lub GGSN_RUN_DIR)
uv run manim -pqh src/visualization/animate_umap.py UmapEvolution
```

