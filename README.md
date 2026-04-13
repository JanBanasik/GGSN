# **Temporal GNN \+ Contrastive Learning na zbiorze MIMIC**

## **📌 Cel i Definicja Problemu**

Predykcja śmiertelności szpitalnej (In-hospital Mortality) na podstawie danych z pierwszych 24h pobytu na OIOM (zbiór MIMIC-III/IV).

**Wyzwania:**

1. **Niestrukturalność:** Dane to mieszanka tekstów (notatki) i sygnałów (monitory).  
2. **Brak etykiet:** Medyczna adnotacja jest droga (rozwiązanie: Contrastive Learning).  
3. **Dynamika czasowa:** Stan pacjenta zależy od kolejności i odstępów czasowych między zdarzeniami (rozwiązanie: Temporal GNN).

## **🏗️ Architektura Systemu (Two-Phase Pipeline)**

### **Faza 1: Multimodal Contrastive Pre-training (Self-Supervised)**

Celem jest nauczenie "wspólnego języka" dla tekstu i sygnałów bez użycia etykiet zgonu.

* **Modele:** Two-Tower Architecture. Text Tower (BioBERT) \+ Signal Tower (1D CNN/LSTM).  
* **Loss:** InfoNCE (zbliżanie reprezentacji notatki i sygnału tego samego pacjenta z tego samego okna czasowego).  
* **Output:** Zamrożone enkodery generujące 768-wymiarowe embeddingi zdarzeń.

### **Faza 2: Temporal Graph Neural Network (Supervised)**

Budowa dynamicznego grafu reprezentującego historię pacjenta.

* **Węzły (Nodes):** Każde zdarzenie kliniczne (notatka, wynik badania) to węzeł z embeddingiem z Fazy 1\.  
* **Krawędzie (Edges):** Połączenia skierowane chronologicznie.  
* **Edge Features (![][image1]):** Kluczowy element Temporal GNN. Każda krawędź posiada atrybut określający czas, jaki upłynął między zdarzeniami.  
* **Readout:** Global Mean/Max Pooling zamieniający graf pacjenta w jeden wektor predykcji.  
* **Output:** Prawdopodobieństwo śmiertelności (Sigmoid).

## **📂 Struktura Projektu**

.  
├── data/                      \# Dane (Ignorowane przez git)  
│   ├── raw/                   \# Surowe CSV z MIMIC  
│   ├── processed/             \# Sparowane pary (tekst-sygnał) do Fazy 1  
│   └── embeddings/            \# Zapisane wektory (.pt) po Fazie 1  
├── src/  
│   ├── data\_prep/  
│   │   ├── cleaner.py         \# Filtrowanie Data Leakage (usuwanie fraz o zgonie)  
│   │   ├── extractor.py       \# SQL/Pandas extraction z MIMIC  
│   │   └── preprocessor.py    \# Imputacja sygnałów (Forward Fill) i pairing  
│   ├── models/  
│   │   ├── towers.py          \# BioBERT \+ CNN  
│   │   └── tgnn\_model.py      \# PyTorch Geometric (SAGEConv z Edge Features)  
│   ├── training/  
│   │   ├── train\_contrastive.py  
│   │   └── train\_gnn.py  
│   └── utils/  
│       ├── graph\_builder.py   \# Tworzenie obiektów Data(x, edge\_index, edge\_attr)  
│       └── metrics.py         \# AUROC, AUPRC  
└── README.md

## **🧠 Instrukcje dla AI (Claude/Cursor)**

1. **Data Leakage First:** Przy implementacji cleaner.py upewnij się, że usuwamy notatki "Discharge Summary" oraz frazy takie jak "expired", "deceased", "autopsy", aby model nie uczył się trywialnych wzorców.  
2. ![][image1] **Implementation:** W modelu T-GNN użyj warstw wspierających edge\_attr (np. GINEConv lub modyfikowany SAGEConv), aby sieć brała pod uwagę odstępy czasowe między węzłami.  
3. **Phase Separation:** Nigdy nie trenuj BioBERTa i GNN jednocześnie. Najpierw wygeneruj embeddingi do data/embeddings/, a potem ładuj je jako statyczne cechy węzłów.  
4. **Sampling:** Ze względu na rozmiar MIMIC, zacznij od podzbioru 1000 pacjentów (Toy Dataset).

## **📈 Metryki Sukcesu**

* **AUROC:** Cel \> 0.88  
* **AUPRC:** Kluczowa metryka przy niezbalansowanych klasach (śmiertelność to \~10-15%).

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAkCAYAAABixKGjAAACZUlEQVR4Xq2Vz0sVURTH31/iPG0nj/z9QnoEarhTokULFwYW9NpIj8B0I4atnrjwB9KLBlLTXrR4hAX9ICKoaWH/QRD+gMKFmxaCvsVx7pm5b+6ce8/MOLr4cLnn3vu958c9M5nqzxMQvPZHM3WoOtQWfyajG30MYka7w1+S4RZikZdEiVPDRXKh4jSCeHEMv67bE5A+5xFIzXjPzwHruf3pAAr9Q1AYGIYXXw6DNfoUI/A913P6YHYVLMtCpuZr2rqJV84xbP44ouI6rblOFJUX0HUT+asDcO/RQoS4G7b98R+MFKfRCylufz7QxCgtLZdg9tlXf14Pci7H5x/+QndvH4Yo5mOlMoqL/Is6qGJqvTa+e46IUdq0go4/tiHX3t2YL9d+41wcnCi/Ce0VPN3ageLUMtwcLUGTu+fW2ARi/HDl2ntQULWtvP3D595/PcXJJcy5uhYSH5+xNWHJ5Y48iotXFIrWFa+834e2rl4oPVnjxdVcU+4+nEfxfOG6tiaKmM1mYe7lL5xrHVrZ2kWBxiFDsxQnF72X474m1T544zbaqWMN8ftuzkRuw4JKczmieLuN1Kj2XJtXcGkrr28H4qIYzW5YstJG7nijCJ9egOnyixm8Fl+868o13GBZTf4YjXhyau5x7oqvf/uP/VB5txeIJ8JQAxXRYLLJtIKGm4n+7ek8iqBO7Ffx/Bg6NImH9JPBwXue4JI4SM4Nl6TGlBaOFJHo4ilEOHRx5Gzp4QrMiJ/wEXB2A7w4YojA4T2ldvKbM4tRGxXhiPE8DVr7J/UmHBmeMUQm9yb3HEWSpc2jDqf1HGuMHRE6BgAAAABJRU5ErkJggg==>