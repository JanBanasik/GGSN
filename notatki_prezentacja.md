# Notatki do prezentacji — Temporal GNN + Contrastive Learning (MIMIC-IV)

---

## Slajd 1 — Tytuł

**CO TO JEST MIMIC-IV?**
Publiczna baza danych Beth Israel Deaconess Medical Center (Boston). Zawiera dane EHR 94 429 unikalnych pacjentów. Dostęp wymaga szkolenia z etyki przez PhysioNet. Wersja 3.1 — najnowsza, użyta w projekcie.

**DLACZEGO PREDYKCJA ŚMIERTELNOŚCI?**
Kliniczne uzasadnienie: wczesne rozpoznanie pacjentów wysokiego ryzyka pozwala na szybsze leczenie, priorytetyzację zasobów OIOM, właściwy poziom opieki. AUROC 0.88 SOTA (literatura) pochodzi z modeli z pełnymi zasobami obliczeniowymi; nasz wynik 0.850 to realistyczny cel przy ograniczeniach jednego GPU laptopowego.

**CZYM SIĘ RÓŻNI OD KLASYCZNEJ KLASYFIKACJI?**
Dane są sekwencyjne w czasie (zdarzenia), heterogeniczne (tekst + liczby + kody), i mają silny imbalans klas. To wymaga specjalizowanej architektury.

---

## Slajd 2 — Problem

**DLACZEGO 1:7 TO PROBLEM? NIE WYSTARCZY BCE?**
BCE (binary cross-entropy) traktuje wszystkie przykłady równo. Przy 1:7 model który zawsze mówi "przeżyje" ma ~87% accuracy. Musimy użyć:
1. `pos_weight` w losie — skaluje gradient dla klasy pozytywnej (pos_weight=6.897)
2. Focal Loss — dodaje czynnik `(1-p)^gamma` skupiający gradient na trudnych przykładach, niezależnie od klasy

**DLACZEGO AUROC A NIE ACCURACY?**
AUROC mierzy zdolność rankingowania: czy model wyżej rankuje przypadki śmiertelne powyżej ocalonych, niezależnie od progu decyzyjnego. Jest odporna na imbalans. AUPRC (precision-recall) jest jeszcze lepsza przy silnym imbalansie — AUROC może być mylący jeśli wiele True Negatives zawyża metryki.

**CO TO TEMPORAL LEAKAGE?**
Sytuacja gdy model używa informacji która nie byłaby dostępna w czasie predykcji. W naszym przypadku: wzięliśmy OSTATNIE N sygnałów z całego pobytu. Pacjent który umiera ma krytyczne vitals bezpośrednio przed śmiercią — model widział "zapis umierania" zamiast prognozować na podstawie wczesnych danych.

---

## Slajd 3 — Dane

**DLACZEGO BIO_CLINICALBERT A NIE ZWYKŁY BERT?**
Bio_ClinicalBERT (emilyalsentzer) był dotrenowany na notatkach z MIMIC-III i literaturze biomedycznej PubMed. Kluczowy: zna skróty kliniczne (PRN, QID, SBP, MAP), terminologię medyczną, styl notatek pielęgniarskich. Ogólny BERT traktuje "WBC 12.5" jak zwykły ciąg znaków; Bio_ClinicalBERT rozumie że to jest wynik morfologii.

**DLACZEGO AKURAT 14 TYPÓW SYGNAŁÓW?**
Wybrano 14 najczęściej występujących i klinicznie istotnych sygnałów w MIMIC-IV: HR (tętno), SpO2 (saturacja), MAP (ciśnienie), RR (oddech), GCS (świadomość), temperatura, i wybrane wyniki laboratoryjne. Można by wziąć więcej, ale 14 zapewnia pokrycie >99% pobytów przy zachowaniu rozsądnym rozmiarze modelu.

**DLACZEGO SUBJECT-DISJOINT SPLIT A NIE LOSOWY?**
Pacjent może mieć wiele pobytów na OIOM-ie. Losowy podział może dać ten sam podmiot w train i val — model uczy się cech specyficznych dla konkretnego pacjenta (wiek, historia choroby), co sztucznie zawyża wyniki. Subject-disjoint gwarantuje że ewaluujemy generalizację do nowych pacjentów, nie nowych pobytów.

**CO TO CHARLSON COMORBIDITY INDEX?**
Skala z 1987 roku (Charlson et al.), zmodyfikowana przez Quan et al. 2005 — 17 schorzeń przewlekłych ważona według ryzyka śmiertelności (zawał, udar, nowotwory przerzutowe...). Używamy wersji 19-kategoryjnej jako 0/1 wektora binarnego. Ważne: kody ICD są przypisywane PRZY WYPISIE, ale kategorie Charlson to schorzenia istniejące przed hospitalizacją — brak leakage.

---

## Slajd 4 — Architektura

**DLACZEGO DWIE FAZY A NIE END-TO-END OD RAZU?**
E2E fine-tuning przetestowaliśmy (slajd o wynikach). Wynik: AUROC 0.790 < 0.829 baseline — catastrophic forgetting. BERT ma 110M parametrów, GNN ~65K. Przy wspólnym treningu gradient z klasyfikatora "niszczy" reprezentacje BERT wyuczone na 102K notatkach. Phase 1 daje stabilne, bogate reprezentacje które Phase 2 może efektywnie wykorzystać bez ich niszczenia.

**CZYM RÓŻNI SIĘ OD FINE-TUNINGU BERT BEZPOŚREDNIO?**
Tradycyjny fine-tuning BERT na klasyfikacji: pobieramy `[CLS]` token i uczymy klasyfikatora. Nie korzysta z sygnałów fizjologicznych ani kolejności zdarzeń. Nasze podejście: Phase 1 tworzy WSPÓLNĄ przestrzeń tekst+sygnał, potem GNN modeluje RELACJE TEMPORALNE między zdarzeniami. To architektonicznie bogatsze.

**JAKI JEST WKŁAD KAŻDEJ FAZY?**
- Phase 1: dostarcza reprezentacje notatek (bez niej: AUROC 0.784, delta −0.045)
- Phase 2: modeluje kontekst temporalny i relacje między zdarzeniami w czasie
- Razem: AUROC 0.850

---

## Slajd 5 — Phase 1: Two-Tower

**CO TO JEST INFONCE / NT-XENT?**
InfoNCE (Noise Contrastive Estimation) minimalizuje temperatura-skalowane cross-entropy na macierzy podobieństwa: `L = CE(sim/tau, labels_diagonal)`. Każda para (notatka_i, sygnał_i) z tego samego pobytu jest parą pozytywną, wszystkie inne pary w batchu są negatywne. Pozwala na O(B²) negatywnych par z B przykładów — efektywniejsze niż triplet loss.

**DLACZEGO TEMPERATURA τ=0.07?**
Temperatura kontroluje "ostrość" dystrybucji podobieństwa:
- Małe τ (0.07): ostre piki, model musi być bardzo pewny par pozytywnych
- Duże τ (>0.5): rozmyta dystrybucja, słabszy sygnał treningowy

τ=0.07 pochodzi z SimCLR (Chen et al. 2020) i CLIP (Radford et al. 2021) — empirycznie sprawdzone na multimodalnym uczeniu kontrastywnym.

**DLACZEGO L2-NORMALIZACJA WEKTORÓW?**
L2-normalizacja projektuje wektory na sferze jednostkowej. Wtedy: `dot(z1, z2) = cosine_similarity(z1, z2) ∈ [-1, 1]`. Zapobiega "collapse" (wszystkie wektory blisko zera), stabilizuje trening, ułatwia interpretację podobieństwa jako kosinusowe.

**CO TO HARD NEGATIVE MINING?**
W batchu szukamy par które są trudne — tj. notatki i sygnały z RÓŻNYCH pobytów ale o wysokim podobieństwie. Takie "trudne negatywne" są bardziej informacyjne niż losowe (np. notatka z kardiologii vs sygnały z neurologii — łatwo rozróżnić; dwie notatki z podobnymi vitals z różnych pobytów — trudno). Hard negatywne przyspieszają uczenie się wyraźnych granic w przestrzeni embeddingów.

---

## Slajd 6 — Phase 2: Graf

**DLACZEGO GRAF A NIE TRANSFORMER / LSTM?**
Transformer (attention) zakłada stałą długość sekwencji i równo-rozmieszczone punkty czasowe. Nasze zdarzenia są nieregularnie rozmieszczone w czasie (notatki co kilka godzin, sygnały co minuty lub godziny). Graf pozwala naturalnie reprezentować nieregularne struktury czasowe. LSTM przetwarza sekwencje liniowo — nie uwzględnia że zdarzenia mogą być z różnych typów (notatka vs sygnał) i w różnych momentach.

**DLACZEGO KRAWĘDZIE TYLKO DO PRZODU W CZASIE?**
Krawędzie skierowane zapewniają kauzalność: wcześniejsze zdarzenia mogą wpływać na interpretację późniejszych, ale nie odwrotnie. Cofanie krawędzi byłoby równoważne z data leakage — model "widziałby przyszłość" przez propagację wiadomości wstecz.

**DLACZEGO ICD NODE W t=−1H?**
ICD Charlson reprezentuje PRZEWLEKŁE schorzenia istniejące PRZED hospitalizacją. Umieszczamy go przed wszystkimi zdarzeniami (t=−1h) jako "prior knowledge" — wiedza a priori o stanie zdrowia pacjenta. Krawędź ICD→każdy_node oznacza że historia chorobowa warunkuje interpretację każdego zdarzenia.

**DLACZEGO DEMOGRAFIKA JAKO GRAPH-LEVEL A NIE NODE?**
Wiek, płeć, tryb przyjęcia to cechy STATYCZNE dla całego pobytu — nie zdarzenia w czasie. Dodanie ich jako osobnego nodu w grafie temporalnym byłoby niespójne architektonicznie. Jako graph-level feature (konkatenacja po poolingu) są dostępne klasyfikatorowi bez wpływania na propagację wiadomości w grafie.

---

## Slajd 7 — Architektura GNN

**DLACZEGO GINECONV A NIE GAT (Graph Attention Network)?**
GAT (Veličković 2018) uczy się wag uwagi dla krawędzi na podstawie cech NODÓW. Problem: nie uwzględnia natywnie cech krawędzi (u nas: Δt). GINEConv (Hu et al. 2019) rozszerza GIN o cechy krawędzi: dodaje `edge_attr` do cechy nodu przed agregacją. Ponieważ Δt (różnica czasu) jest kluczowa semantycznie — odległość 30 minut vs 12 godzin między notatką a sygnałami ma inne znaczenie — potrzebujemy modelu który te różnice może explicite przetworzyć.

**DLACZEGO GIN A NIE GCN?**
GCN (Kipf & Welling 2017) używa normalizacji stopni węzłów do agregacji. Jest mniej ekspresywny — nie rozróżnia grafów które GIN rozróżnia. GIN (Xu et al. 2019) jest tak ekspresywny jak test Weisfeiler-Leman (WL-test) — teoretyczne maksimum dla message-passing GNN. MLP wewnątrz konwolucji zamiast prostej sumy daje dodatkową ekspresywność.

**DLACZEGO 3 WARSTWY A NIE WIĘCEJ?**
Ablacja pokazała: 2 warstwy → AUROC 0.835, 3 warstwy → 0.829 (baseline), 4 warstwy → 0.822. Z demografiką: demo+4layers → 0.836 (gorsze niż demo+3layers → 0.844). Więcej warstw = większy receptive field = bardziej globalna agregacja = over-smoothing. W grafach temporalnych zbyt globalna perspektywa zaciera lokalne wzorce kliniczne (co działo się w oknie ±2h jest ważniejsze niż całościowa średnia pobytu).

**CO TO ATTENTION POOLING I DLACZEGO LEPSZY OD MEAN?**
Mean pooling bierze prostą średnią cech wszystkich nodów — traktuje pierwszą godzinę pobytu tak samo jak godzinę przed kryzysem. Attention pooling (AttentionalAggregation) uczy się współczynników bramkujących: `alpha_v = softmax(MLP_gate(h_v)) → g = Σ alpha_v * h_v`. Model może nauczyć się skupiać na klinicznie ważnych momentach (np. deterioracja stanu). Wynik: +0.015 AUROC vs mean pooling (0.844 vs 0.829).

**CO TO FOCAL LOSS I KIEDY LEPSZY OD WAŻONEGO BCE?**
BCE z `pos_weight` skaluje loss dla klasy pozytywnej stale (współczynnik 6.9). Focal Loss: `FL(p_t) = −alpha_t * (1−p_t)^gamma * log(p_t)`. Czynnik `(1−p_t)^gamma` dynamicznie redukuje loss dla łatwo sklasyfikowanych przykładów (p_t duże) i skupia gradient na trudnych (p_t małe). Gamma=2.0 jest standardowym wyborem z oryginalnego paperu RetinaNet (Lin 2017). Efekt: +0.002 AUROC, ale znacząca poprawa sens@95spec (0.388 vs 0.326 baseline). Klinicznie ważne: wyższy recall przy danej specyficzności.

---

## Slajd 8 — Bug / Temporal Leakage

**JAK WYKRYTO LEAKAGE?**
AUROC 0.931 jest podejrzanie wysoki — SOTA na tym samym zadaniu i zbiorze danych to 0.88, a nasz poprzedni najlepszy wynik to 0.850. Kiedy wynik modelu bije SOTA o 4%, należy sprawdzić preprocessing, nie świętować. Analizując kod `gnn_dataset.py` znaleźliśmy: `sig_rows[-max_signals:]` zamiast `sig_rows[:max_signals]`.

**DLACZEGO TO LEAKAGE?**
Pacjent który umiera ma terminalne vitals (ekstremalne HR, BP, SpO2) bezpośrednio przed śmiercią. Używając OSTATNICH N sygnałów z całego pobytu model "widział" te sygnały terminalne — de facto wiedział że pacjent zaraz umrze, zanim został "zapytany" o predykcję. To nie jest predykcja, to opis.

**DLACZEGO PIERWSZE N SYGNAŁÓW?**
Klinicznie sensowna predykcja: czy pacjent przyjęty na OIOM umrze, na podstawie danych z WCZESNEJ FAZY pobytu (pierwsze godziny / pierwsza doba)? To odpowiada prawdziwemu scenariuszowi klinicznemu: lekarz chce wiedzieć o ryzyku wcześnie, nie wtedy gdy pacjent już umiera.

**CZY INNE DANE TEŻ MOGŁY MIEĆ LEAKAGE?**
Notatki kliniczne: używamy notatek z całego pobytu, ale są one PRZYCZYNOWO uzasadnione — notatka z godziny 6 pobytu NIE opisuje tego co wydarzy się w godzinie 48. Sygnały all-stay po naprawieniu: pierwsze 50 = brak leakage. ICD Charlson: kody ICD są przypisywane przy wypisie, ale dotyczą PRZEWLEKŁYCH schorzeń istniejących przed hospitalizacją — Charlson index jest właśnie do tego zaprojektowany.

---

## Slajd 9 — AUROC

**JAK CZYTAĆ TEN WYKRES?**
Przerywana fioletowa linia = baseline (AUROC 0.829, mean pooling bez demo i focal). Przerywana czerwona linia = SOTA 0.880 z literatury. Kolory pasków: czerwony = ablacje (gorsze od baseline), indigo = baseline, zielony/cyan = warianty lepsze od baseline.

**CZEMU E2E (0.790) JEST GORSZY OD BASELINE (0.829)?**
Catastrophic forgetting: BERT (110M param) jest trenowany z LR ~5e-7, GNN z ~1e-3. Różnica 200× w LR powinna być wystarczająca. Ale gradient z małego klasyfikatora (65K param) przez duży BERT (110M param) zaburza reprezentacje wyuczone w Phase 1 nawet przy tak niskim LR. Zamrożenie 8 z 12 warstw częściowo pomaga, ale nie eliminuje problemu. Na 8GB VRAM bez gradient checkpointing pełne E2E jest również problematyczne pamięcią.

**CZEMU SIGNAL_ONLY (0.784) TAK WIELE TRACI?**
Notatki kliniczne zawierają bogate, nieskwantyfikowane informacje kliniczne — oceny stanu ogólnego, plany leczenia, zmiany w zachowaniu pacjenta. Sygnały fizjologiczne są dokładne liczbowo, ale ograniczone do mierzalnych parametrów. Połączenie obu: notatki dają kontekst interpretacyjny dla sygnałów.

---

## Slajd 10 — AUPRC

**DLACZEGO AUPRC WAŻNIEJSZA OD AUROC PRZY IMBALANSIE?**
AUROC mierzy: P(score(positive) > score(negative)) dla losowej pary. Przy 1:7 imbalansie: model może mieć AUROC 0.85 z wieloma False Negatives bo jest bardzo wiele True Negatives które zawyżają metryki ROC. AUPRC skupia się na wydajności dla klasy pozytywnej: precision@recall_k. Random baseline AUPRC = prevalencja klasy = 0.127. Random baseline AUROC = 0.5 zawsze — tak samo dla zbilansowanych i niezbilansowanych.

**CZY NASZ AUROC 0.850 JEST WIARYGODNY?**
Tak — mamy również obu metryk z podobną hierarchią eksperymentów. Best model: AUROC 0.850, AUPRC 0.465. Relacja 0.850/0.465 vs random 0.5/0.127 jest spójna: model 3.6× powyżej losowego w precision-recall. Gdyby AUROC był zawyżony przez TN (False Negative problem), widzielibyśmy niską AUPRC — a mamy 0.465 czyli wyraźnie powyżej 0.127.

---

## Slajd 11 — Krzywa uczenia

**CO POKAZUJE KRZYWA UCZENIA?**
Niebieski: val AUROC na zbiorze walidacyjnym po każdej epoce. Szary (cienki): train loss (Focal Loss na zbiorze treningowym). Najlepszy punkt: zaznaczony granatowym kółkiem.

**CZY MODEL SIĘ PRZETRENOWUJE?**
Val AUROC stabilizuje się około 0.845 od ~40 epoki i dalej nieznacznie rośnie. Train loss maleje monotonicznie — model dalej się uczy, ale val AUROC nie spada — brak overfittingu. Dropout 0.3, early stopping patience=15 zapewniają regularyzację.

**DLACZEGO WYNIKI VERSION_10 I VERSION_14 SĄ RÓŻNE MIMO IDENTYCZNYCH HPARAMS?**
version_10: AUROC 0.850, version_14: AUROC 0.848. Różnica 0.002 AUROC. To jest wariancja losowa — seed deterministyczny ale PyG/CUDA operacje mogą mieć niezbyt reprodukowalny wynik. Różnica mniejsza niż 0.003 jest nieistotna statystycznie. Oznacza to że nasz wynik jest stabilny.

---

## Slajd 12 — Wkład komponentów

**DLACZEGO ICD NODE NIE POMAGA?**
56.4% pobytów ma zerowy wektor Charlson (brak zapisanych chorób przewlekłych). Dla tych pobytów ICD node to wektor samych zer — nieinformatywny. Pozostałe 43.6% z niezerowym Charlsonem: 19-wymiarowy rzadki wektor binarny jest za mało informatywny dla 65K-parametrowego modelu przy imbalansie 1:7. Jeśli mielibyśmy więcej parametrów lub gęstszy wektor kodowania ICD, być może by pomogło. Alternatywnie: użyć ICD jako graph-level feature (jak demografikę) zamiast osobnego nodu — to jest propozycja w future work.

**DLACZEGO ALL-STAY SIGNALS (0.820) GORSZE OD BASELINE (0.829)?**
Baseline używa sygnałów w oknie ±2h od każdej notatki — to okno jest już wybrane przez Phase 1 jako klinicznie relewantne. Wzięcie pierwszych 50 sygnałów z całego pobytu (medianycznie 315 sygnałów, max 30K) przy cap=50 daje fragmentaryczne pokrycie całego pobytu. Okno ±2h jest bogato pokryte przez Phase 1; random sample całego pobytu jest zbyt rzadki. Potencjalne rozwiązanie: więcej sygnałów (cap=200+), ale wymaga więcej RAM.

**DLACZEGO FOCAL LOSS BARDZIEJ POMAGA NA SENS@95SPEC NIŻ AUROC?**
Focal Loss skupia gradient na trudnych przykładach — czyli pacjentach trudno klasyfikowalnych (ci co są blisko granicy decyzyjnej). Przy wysokiej specyficzności (95%) chcemy wyłapać tak dużo True Positives jak możliwe. Focal Loss przesuwa gradient w kierunku właściwej klasyfikacji trudnych przypadków pozytywnych — stąd poprawa recallu przy danym progu specyficzności. Nie zmienia dramatycznie AUROC (globalny ranking), ale poprawia kliniczny punkt decyzyjny.

---

## Slajd 13 — Najlepszy model

**CZEMU AKURAT ATTENTION A NIE DUAL POOLING?**
Dual pooling (osobne pule dla nodów sygnałów i notatek, konkatenowane) daje AUROC 0.846 — prawie tak dobry jak attention (0.844 bez focal, 0.850 z focal). Dual pooling ma lepszą kalibrację (Brier score 0.170 vs 0.195 dla attention). Attention pooling + focal loss daje najlepsze AUROC i sens@95spec. W zastosowaniach klinicznych gdzie zależy nam na kalibracji (precyzja prawdopod.) dual pooling może być lepszym wyborem. Dla maksymalizacji AUROC — attention+focal.

**CO OZNACZA SENS@95SPEC = 0.388?**
Przy ustawieniu progu klasyfikacji tak że 95% ocalonych jest poprawnie identyfikowanych jako "niskie ryzyko" (specificity=0.95) — model prawidłowo identyfikuje 38.8% pacjentów wysokiego ryzyka (sensitivity=0.388). W praktyce: ze 100 pacjentów śmiertelnych na OIOM model oznaczy alarmem ~39 z nich przy tylko 5% fałszywych alarmów wśród ocalonych. To jest klinicznie użyteczne — szczególnie w środowisku z ograniczonymi zasobami.

**DLACZEGO REDUCELRONPLATEAU A NIE COSINE DECAY?**
ReduceLROnPlateau monitoruje val_auroc i zmniejsza LR o połowę gdy brak poprawy przez 5 epok (patience=5, factor=0.5). Jest adaptacyjny — nie wymaga ustalonego harmonogramu. Cosine decay byłby odpowiedni gdybyśmy znali dokładną liczbę epok z góry; tutaj early stopping z patience=15 przerywa trening w różnych momentach dla różnych eksperymentów.

---

## Slajd 14 — vs SOTA

**CO DOKŁADNIE ROBI SOTA CZEGO MY NIE ROBIMY?**
Modele SOTA na predykcji śmiertelności (np. ClinicalBERT + temporal attention, MIMIC-Extract + transformer) używają:
1. End-to-end fine-tuning BERT z gradient checkpointing (eliminuje OOM)
2. Multi-GPU distributed training (można trenować batch_size=64+)
3. Temporal attention zamiast GINEConv (lepsze modelowanie nieregularnych sygnałów)
4. Często więcej danych pomocniczych (leki, procedury) których my nie uwzględniamy

**DLACZEGO NIE MOŻNA BYŁO TEGO ZROBIĆ NA NASZYM SPRZĘCIE?**
RTX 4060 Laptop GPU: 8GB VRAM. Pełny E2E BERT (110M param) + batch=32 = ~12GB VRAM (OOM). Gradient checkpointing zmniejszyłoby zużycie 4× kosztem 30% spowolnienia — nie zaimplementowane w obecnej wersji. Na dedykowanym serwerze (A100 80GB) E2E byłoby możliwe w ~5-8 min/epokę zamiast crash.

**JAK MIERZYĆ WKŁAD NASZEJ PRACY SKORO NIE BIJEMY SOTA?**
1. Systematyczne ablacje: pokazujemy co konkretnie wnosi każda składowa
2. Wykrycie i naprawa temporal leakage: metodologiczny wkład
3. Dokumentacja negatywnych wyników: wartość dla powtarzalności nauki
4. Efektywność parametrów: 65K param osiąga 96.6% wydajności SOTA (110M param)
5. Otwarta reprodukowalna implementacja na MIMIC-IV 3.1

---

## Slajd 15 — Wnioski

**PODSUMOWANIE KLUCZOWYCH DECYZJI PROJEKTOWYCH:**

1. **GINEConv** — jedyny standardowy GNN oprócz PNA który natywnie obsługuje cechy krawędzi. Δt jest kluczowe dla temporalnego kontekstu klinicznego.

2. **Phase 1 zamiast E2E** — catastrophic forgetting potwierdza że rząd wielkości różnica LR (1e-3 vs 5e-7) nie wystarczy. Zamrożenie reprezentacji zachowuje wiedzę zdobytą na 102K notatkach.

3. **InfoNCE** — efektywny batch-level contrastive loss. O(B²) negatywnych par vs O(B) w triplet loss przy tym samym koszcie obliczeniowym.

4. **Temporal Leakage** — kluczowa lekcja metodologiczna. W danych medycznych sekwencyjnych zawsze sprawdzaj kierunek czasu w preprocessingu.

5. **Focal Loss** — poprawia klinicznie istotne metryki (sens@95spec) nawet jeśli AUROC rośnie marginalnie. Ważne jeśli model ma być użyteczny w praktyce.

**CZY PROJEKT JEST POWTARZALNY?**
Tak. Kod jest publiczny, dane dostępne przez PhysioNet po rejestracji. Seed deterministyczny (42). Szczegółowe logi w `data/snapshots/gnn/`. Jedyna potencjalna niereproduko­walność: niedeterministyczne operacje CUDA. Odchylenie standardowe między uruchomieniami ~0.002 AUROC.
