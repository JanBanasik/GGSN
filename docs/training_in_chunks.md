# Trening partiami — plan implementacji resume z checkpointu

**Cel**: móc trenować Fazę 1 w blokach po 2-4h (np. nocą, między zajęciami) bez utraty
postępu. Restart z checkpointu, kontynuacja od następnej epoki.

**Status**: nie zaimplementowane (2026-04-28). Obecnie `train()` zaczyna zawsze od epoki 1
i zapisuje tylko końcowe / best wagi modelu, bez stanu optimizera/schedulera/AMP/RNG.
Restart = stracone wszystko.

---

## Czy się da?

**TAK**. To jest standardowa funkcjonalność, ~80 linii kodu w `train_contrastive.py`.
Dwie zalety dla nas:

1. Można trenować **w sesjach po 2-4h** zamiast jednego 15h batcha — laptop nie pali się
   przez noc, można odzyskać sprzęt na inne rzeczy między sesjami.
2. Można **przedłużać dobry run** — jeśli po 25 epokach val_loss dalej spada, łatwo dolejesz
   kolejne 25 bez restartu treningu od zera.

**Pułapki** (nie blokujące, ale do uwagi):
- Pełny resume wymaga zapisu **całego stanu**, nie tylko wag modelu. Sam `state_dict()` modelu
  nie wystarczy — Adam ma momentum buffers per-param, scheduler pamięta krok, GradScaler
  pamięta scale, RNG pamięta gdzie był.
- Hyperparams zamrożone w checkpoint — jeśli chcesz zmienić `freeze_bottom_layers` w trakcie,
  optimizer state nie pasuje (różna liczba trainable params). Wtedy trzeba resume wag
  modelu i resetować optimizer.
- DataLoader workers + shuffle: trzeba zapisać RNG generatora samplera, inaczej kolejność
  batchy się nie zgodzi (mała sprawa, ale wpływa na powtarzalność).

---

## Co dokładnie trzeba zapisać w checkpoincie

W jednym pliku `data/snapshots/run_<ts>/checkpoint.pt`:

```python
{
    # Wagi modeli
    "text_tower":   text_tower.state_dict(),
    "signal_tower": signal_tower.state_dict(),
    # Stan optimizera (Adam momentum buffers — krytyczne)
    "optimizer":    optimizer.state_dict(),
    # Pozycja schedulera (cosine annealing wie ile kroków zostało)
    "scheduler":    scheduler.state_dict(),
    # Stan AMP scaler (skala dynamicznie się dostosowuje, nie chcesz resetu)
    "scaler":       scaler.state_dict(),
    # RNG state — żeby kontynuować deterministycznie
    "torch_rng":    torch.get_rng_state(),
    "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    "numpy_rng":    np.random.get_state(),
    "python_rng":   random.getstate(),
    # Stan treningu
    "epoch":        epoch,             # ostatnia ZAKOŃCZONA epoka
    "best_val":     best_val,
    "bad_epochs":   bad_epochs,
    # Hyperparams (do walidacji że nie zmieniono ich między sesjami)
    "config":       config,
}
```

Zapisuj **co epokę** (nadpisuj `checkpoint.pt`) + opcjonalnie kopię `checkpoint_epoch_NNN.pt`
co 5 epok jako rollback.

---

## CLI flagi do dodania

```bash
# Nowy run, jak teraz
uv run python -m src.training.train_contrastive --epochs 25

# Resume z istniejącego run dir — kontynuuje od checkpoint.pt
uv run python -m src.training.train_contrastive \
    --resume data/snapshots/run_20260501_220000 \
    --epochs 50    # epochs to TARGET liczba, nie dodatkowa

# Zatrzymaj się grzecznie po N godzinach (zapisz checkpoint, wyjdź)
uv run python -m src.training.train_contrastive --epochs 50 --max-time-hours 4
```

**Sugerowane zachowanie**:
- `--resume <dir>` ładuje `<dir>/checkpoint.pt`, kontynuuje **w tym samym `run_dir`**
  (snapshoty per-epoch lecą do tego samego katalogu, log.csv jest dopisywany)
- `--epochs N` to **target total**, nie inkrementalny — jeśli checkpoint ma epoch=10
  i podasz `--epochs 25`, robi epoki 11-25
- `--max-time-hours T` po przekroczeniu T godzin czeka na koniec bieżącej epoki,
  zapisuje checkpoint, wychodzi z exit code 0
- Walidacja przy resume: jeśli config w checkpoint != obecny config (np. zmieniłeś
  `lr_bert`), wypisz warning i zapytaj `--force` żeby kontynuować

---

## Implementacja — gdzie co dotknąć

### 1. Funkcje pomocnicze w [train_contrastive.py](../src/training/train_contrastive.py)

```python
def save_checkpoint(path: Path, **state) -> None:
    torch.save(state, path)

def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)

def restore_rng(ckpt: dict) -> None:
    torch.set_rng_state(ckpt["torch_rng"])
    if ckpt.get("torch_cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng"])
    np.random.set_state(ckpt["numpy_rng"])
    random.setstate(ckpt["python_rng"])
```

### 2. Modyfikacja `train()`

Na początku, po stworzeniu modeli/optimizera:

```python
start_epoch = 1
best_val = float("inf")
bad_epochs = 0

if resume_from is not None:
    ckpt = load_checkpoint(Path(resume_from) / "checkpoint.pt", device)
    text_tower.load_state_dict(ckpt["text_tower"])
    signal_tower.load_state_dict(ckpt["signal_tower"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    restore_rng(ckpt)
    start_epoch = ckpt["epoch"] + 1
    best_val = ckpt["best_val"]
    bad_epochs = ckpt["bad_epochs"]
    run_dir = Path(resume_from)  # KONTYNUUJ w tym samym dir, nie nowy
    print(f"Resumed from epoch {ckpt['epoch']}, best_val={best_val:.4f}")
```

Pętla zmienia się na `for epoch in range(start_epoch, epochs + 1):` — i na końcu każdej
epoki dodajesz `save_checkpoint(run_dir / "checkpoint.pt", ...)`.

### 3. Max-time guard

Tuż przed pętlą:
```python
import time
start_time = time.time()
max_seconds = max_time_hours * 3600 if max_time_hours else float("inf")
```

Po każdej epoce:
```python
if time.time() - start_time >= max_seconds:
    print(f"Max time {max_time_hours}h reached, stopping cleanly. Resume with:")
    print(f"  --resume {run_dir}")
    break
```

### 4. CLI args (3 nowe)

```python
p.add_argument("--resume", type=str, default=None,
               help="Resume from checkpoint in this run dir")
p.add_argument("--max-time-hours", type=float, default=None,
               help="Stop training cleanly after N hours")
p.add_argument("--force", action="store_true",
               help="Allow resume despite config mismatch")
```

---

## Praktyczne strategie sesji treningowych

Dla pełnej Fazy 1 na full MIMIC (estimated 10-15h):

### Strategia A — sesje 2-3h, 5-7 dni
```
Dzień 1 wieczór:   uv run ... --epochs 50 --max-time-hours 3   # epoki 1-12
Dzień 2 południe:  uv run ... --resume <dir> --epochs 50 --max-time-hours 3   # 13-24
Dzień 3 wieczór:   uv run ... --resume <dir> --epochs 50 --max-time-hours 3   # 25-36
Dzień 4 weekend:   uv run ... --resume <dir> --epochs 50      # 37-50 (final)
```
Zalety: laptop max 3h pod obciążeniem, można reagować na problemy między sesjami.

### Strategia B — jedna nocka 8h
```
Wieczór:  uv run ... --epochs 25 --max-time-hours 8   # robi co się da
Rano:     sprawdzasz log.csv, decydujesz czy resume jeszcze 8h
```
Prostsze ale ryzyko: laptop pali się 8h, throttling termiczny.

### Strategia C — kolega trenuje na M3 Pro
M3 Pro ma 18GB unified memory ale ~2-3× wolniejszy od 4060. Realnie nie warto trenować
na nim głównego BERTa, ale **sweep hyperparam na małej kohorcie** (toy=5000) tam jest sens —
np. jego Mac szuka najlepszej `temperature`, Twój 4060 robi finalny long run.

---

## Kolejność robót (do zrobienia za 1 sesję)

1. Dodać `save_checkpoint` + `load_checkpoint` + `restore_rng` (5 min)
2. Modyfikacja `train()`: `resume_from`, `max_time_hours` params + logika (15 min)
3. CLI args + propagacja (5 min)
4. Sanity test: trenuj 2 epoki, zatrzymaj, resume z checkpoint, sprawdź czy log.csv
   ma 4 epoki bez przerw i czy val_loss się zgadza ze "spodziewaną" trajektorią (15 min)
5. Walidacja config-mismatch + `--force` flag (10 min)

**Razem: ~50 min implementacji + test**. Dobry kandydat na wieczorną sesję, najlepiej
**przed** rozpoczęciem długiego treningu pełnego MIMIC.

---

## Edge cases / pułapki

- **OOM przy resume**: czasami `torch.load` na CUDA powoduje pik VRAM (kopia w fp32 ładowana
  zanim cast). Fix: ładuj na CPU, potem `.to(device)` per-tensor.
- **Scheduler step count**: cosine annealing zna `T_max = epochs * len(train_loader)`. Jeśli
  zmienisz batch size między sesjami, `len(train_loader)` się zmienia → scheduler się gubi.
  Trzymaj batch_size stały lub przekalibruj scheduler przy resume.
- **Best val tracking**: jeśli zmieniasz `early_stop_patience` przy resume, rozważ reset
  `bad_epochs = 0` (dać ile pacjent dla zmienionego trybu).
- **Snapshot per-epoch nadpisany**: obecnie `_take_snapshot()` zapisuje do
  `epoch_NNN/`. Przy resume kontynuuje numerację — OK, nadpisze pierwsze epoki nie tknie.
  Tylko jeśli chcesz **resetu treningu od checkpoint** (np. inne hyperparams), trzeba
  ręcznie usunąć `epoch_NNN/` dla N > start_epoch.
- **AMP scaler**: jak za pierwszym razem skaler się ustabilizuje na np. 65536, restart
  z resetowanym skalerem (np. 2^16 default) na pierwszej iteracji może mieć overflow.
  Dlatego MUSISZ ładować `scaler.state_dict()` z checkpointu.

---

## Bonus: wandb / mlflow integration

Resume jest dużo łatwiejszy z trackerem eksperymentów:
- wandb ma `wandb.init(resume="must", id=...)` — wątek metryk i artifacts continues
- log.csv to "ubogi tracker" — działa, ale gdy masz 10+ runów, wandb daje filtrowanie/wykresy

Opcjonalnie po implementacji resume: P1.6 z `improvement_roadmap.md` (wandb integration).
