from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.data_prep.cleaner import filter_leakage_phrases

# ---------------------------------------------------------------------------
# Config – hard-coded paths to raw MIMIC-IV data
# ---------------------------------------------------------------------------
MIMIC_BASE = Path(
    "/home/jan_b/Semestr6/Glebokie_i_grafowe_sieci_neuronowe"
    "/datasets/mimic/files/mimiciv/3.1"
)
NOTES_BASE = Path(
    "/home/jan_b/Semestr6/Glebokie_i_grafowe_sieci_neuronowe"
    "/datasets/mimic-notes/files/mimic-iv-note/2.2"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CARDIO_UNITS = [
    "Coronary Care Unit (CCU)",
    "Cardiovascular Intensive Care Unit (CVICU)",
]

# ---------------------------------------------------------------------------
# Signal catalog – 7 vitals + 7 labs, mapped to a contiguous item_type_id
# ---------------------------------------------------------------------------
# Vitals (icu/chartevents). Mean values where multiple variants exist.
VITALS_CATALOG: dict[str, list[int]] = {
    "HR":         [220045],            # Heart Rate
    "BP_mean":    [220181, 220052],    # NIBP mean (preferred), Art mean (fallback)
    "BP_sys":     [220179, 220050],
    "BP_dia":     [220180, 220051],
    "SpO2":       [220277],
    "RR":         [220210],
    "Temp_F":     [223761],
}

# Labs (hosp/labevents). Use most common itemid for each label.
LABS_CATALOG: dict[str, list[int]] = {
    "Troponin_I": [51002],
    "NTproBNP":   [50963],
    "Creatinine": [50912],
    "Lactate":    [50813],
    "Potassium":  [50971],
    "Hemoglobin": [51222],
    "WBC":        [51301],
}

# Build signal_type → item_type_id mapping (vitals first, then labs)
ALL_SIGNAL_NAMES = list(VITALS_CATALOG.keys()) + list(LABS_CATALOG.keys())
SIGNAL_NAME_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(ALL_SIGNAL_NAMES)}

# Reverse map: itemid → signal_name (only first match wins; preferred itemid first)
ITEMID_TO_SIGNAL: dict[int, str] = {}
for _name, _ids in {**VITALS_CATALOG, **LABS_CATALOG}.items():
    for _id in _ids:
        ITEMID_TO_SIGNAL.setdefault(_id, _name)

ALL_VITAL_ITEMIDS: list[int] = sorted({i for ids in VITALS_CATALOG.values() for i in ids})
ALL_LAB_ITEMIDS: list[int] = sorted({i for ids in LABS_CATALOG.values() for i in ids})

# Plausible physiological ranges for clipping (lo, hi) per signal name.
# Used to compute normalized value ∈ [0, 1].
NORM_RANGE: dict[str, tuple[float, float]] = {
    "HR":         (30.0, 250.0),
    "BP_mean":    (20.0, 200.0),
    "BP_sys":     (40.0, 250.0),
    "BP_dia":     (20.0, 150.0),
    "SpO2":       (50.0, 100.0),
    "RR":         (4.0, 60.0),
    "Temp_F":     (90.0, 108.0),
    "Troponin_I": (0.0, 50.0),
    "NTproBNP":   (0.0, 35000.0),
    "Creatinine": (0.1, 15.0),
    "Lactate":    (0.2, 20.0),
    "Potassium":  (1.5, 8.0),
    "Hemoglobin": (4.0, 20.0),
    "WBC":        (0.5, 100.0),
}

_DT_FMT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Step 1 – Cardio cohort + in-hospital mortality label
# ---------------------------------------------------------------------------
def load_cohort() -> pl.DataFrame:
    """
    Loads ICU stays for CCU/CVICU and joins in-hospital mortality from
    `admissions.hospital_expire_flag` (true in-hospital death indicator).

    Returns columns:
        subject_id, hadm_id, stay_id, first_careunit, intime, outtime, mortality
    """
    icustays_lf = (
        pl.scan_csv(MIMIC_BASE / "icu" / "icustays.csv.gz")
        .filter(pl.col("first_careunit").is_in(CARDIO_UNITS))
        .with_columns(
            pl.col("intime").str.to_datetime(_DT_FMT, strict=False),
            pl.col("outtime").str.to_datetime(_DT_FMT, strict=False),
        )
        .select(["subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime"])
    )

    admissions_lf = (
        pl.scan_csv(MIMIC_BASE / "hosp" / "admissions.csv.gz")
        .select(["hadm_id", "deathtime", "dischtime", "hospital_expire_flag"])
        .with_columns(
            pl.col("deathtime").str.to_datetime(_DT_FMT, strict=False),
            pl.col("dischtime").str.to_datetime(_DT_FMT, strict=False),
            pl.col("hospital_expire_flag").cast(pl.Int8),
        )
    )

    cohort = (
        icustays_lf
        .join(admissions_lf, on="hadm_id", how="left")
        .with_columns(
            pl.col("hospital_expire_flag").fill_null(0).alias("mortality")
        )
        .select(
            ["subject_id", "hadm_id", "stay_id", "first_careunit",
             "intime", "outtime", "deathtime", "dischtime", "mortality"]
        )
        .collect()
    )

    print(
        f"  Cohort: {len(cohort)} stays | "
        f"mortality rate (in-hospital): {cohort['mortality'].mean():.1%} | "
        f"unique subjects: {cohort['subject_id'].n_unique()}"
    )
    return cohort


# ---------------------------------------------------------------------------
# Step 2 – Radiology notes filtered to [intime, intime + 24 h]
# ---------------------------------------------------------------------------
def load_notes(cohort: pl.DataFrame) -> pl.DataFrame:
    """
    Reads radiology.csv.gz, keeps notes within [intime, intime + 24h] for the
    matched ICU stay, and filters out leakage phrases (see cleaner.py).

    Returns columns: note_id, stay_id, note_time, text
    """
    stay_times = cohort.select(["hadm_id", "stay_id", "intime"]).lazy()

    notes = (
        pl.scan_csv(NOTES_BASE / "note" / "radiology.csv.gz", quote_char='"')
        .select(["note_id", "subject_id", "hadm_id", "charttime", "text"])
        .with_columns(pl.col("charttime").str.to_datetime(_DT_FMT, strict=False))
        .join(stay_times, on="hadm_id", how="inner")
        .filter(
            (pl.col("charttime") >= pl.col("intime"))
            & (pl.col("charttime") <= pl.col("intime") + pl.duration(hours=24))
        )
        .select(["note_id", "stay_id", "charttime", "text"])
        .rename({"charttime": "note_time"})
        .collect()
    )

    print(f"  Notes (raw): {len(notes)} rows across {notes['stay_id'].n_unique()} stays")
    notes = filter_leakage_phrases(notes, text_col="text")
    print(f"  Notes (post-cleaner): {len(notes)} rows across {notes['stay_id'].n_unique()} stays")
    return notes


# ---------------------------------------------------------------------------
# Step 3a – Vitals (chartevents)
# ---------------------------------------------------------------------------
def load_vitals(cohort: pl.DataFrame) -> pl.DataFrame:
    """
    Lazily scans chartevents.csv.gz for the configured vital itemids,
    restricted to cohort stay_ids and the [intime, intime+24h] window.

    Returns columns: stay_id, event_time, signal_name, valuenum
    """
    stay_window = cohort.select(["stay_id", "intime"]).lazy()
    itemid_lookup = pl.LazyFrame({
        "itemid": list(ITEMID_TO_SIGNAL.keys()),
        "signal_name": list(ITEMID_TO_SIGNAL.values()),
    })

    vitals = (
        pl.scan_csv(MIMIC_BASE / "icu" / "chartevents.csv.gz")
        .filter(pl.col("itemid").is_in(ALL_VITAL_ITEMIDS))
        .select(["stay_id", "charttime", "itemid", "valuenum"])
        .with_columns(
            pl.col("charttime").str.to_datetime(_DT_FMT, strict=False),
            pl.col("valuenum").cast(pl.Float32, strict=False),
        )
        .join(stay_window, on="stay_id", how="inner")
        .filter(
            (pl.col("charttime") >= pl.col("intime"))
            & (pl.col("charttime") <= pl.col("intime") + pl.duration(hours=24))
        )
        .drop_nulls("valuenum")
        .join(itemid_lookup, on="itemid", how="inner")
        .rename({"charttime": "event_time"})
        .select(["stay_id", "event_time", "signal_name", "valuenum"])
        .collect()
    )

    print(
        f"  Vitals: {len(vitals)} rows | "
        f"signals: {sorted(vitals['signal_name'].unique().to_list())}"
    )
    return vitals


# ---------------------------------------------------------------------------
# Step 3b – Labs (labevents)
# ---------------------------------------------------------------------------
def load_labs(cohort: pl.DataFrame) -> pl.DataFrame:
    """
    Lazily scans labevents.csv.gz for the configured lab itemids, restricted
    to the cohort's hadm_id list and the [intime, intime+24h] window.

    Lab events are joined on hadm_id (not stay_id, since labs aren't ICU-scoped).
    The intime window filter is applied via stay_window join.

    Returns columns: stay_id, event_time, signal_name, valuenum
    """
    stay_window = cohort.select(["hadm_id", "stay_id", "intime"]).lazy()
    itemid_lookup = pl.LazyFrame({
        "itemid": list(ITEMID_TO_SIGNAL.keys()),
        "signal_name": list(ITEMID_TO_SIGNAL.values()),
    })

    labs = (
        pl.scan_csv(MIMIC_BASE / "hosp" / "labevents.csv.gz")
        .filter(pl.col("itemid").is_in(ALL_LAB_ITEMIDS))
        .select(["hadm_id", "charttime", "itemid", "valuenum"])
        .with_columns(
            pl.col("hadm_id").cast(pl.Int64, strict=False),
            pl.col("charttime").str.to_datetime(_DT_FMT, strict=False),
            pl.col("valuenum").cast(pl.Float32, strict=False),
        )
        .drop_nulls(["hadm_id", "valuenum"])
        .join(stay_window, on="hadm_id", how="inner")
        .filter(
            (pl.col("charttime") >= pl.col("intime"))
            & (pl.col("charttime") <= pl.col("intime") + pl.duration(hours=24))
        )
        .join(itemid_lookup, on="itemid", how="inner")
        .rename({"charttime": "event_time"})
        .select(["stay_id", "event_time", "signal_name", "valuenum"])
        .collect()
    )

    print(
        f"  Labs: {len(labs)} rows | "
        f"signals: {sorted(labs['signal_name'].unique().to_list())}"
    )
    return labs


# ---------------------------------------------------------------------------
# Step 4 – Pair each note with all signals within ±2 h
# ---------------------------------------------------------------------------
def pair_notes_signals(notes: pl.DataFrame, signals: pl.DataFrame) -> pl.DataFrame:
    """
    For every note, finds all signals from the same stay_id whose event_time
    lies within ±2 hours of the note_time.

    Returns one row per (note, signal) pair with normalized value and
    item_type_id (0..N_signals-1).
    """
    paired = (
        notes.join(signals, on="stay_id", how="inner")
        .filter(
            (pl.col("event_time") - pl.col("note_time"))
            .dt.total_seconds()
            .abs()
            <= 2 * 3600
        )
    )

    # Per-signal normalization → value clipped to plausible range, scaled to [0,1]
    norm_lo = pl.lit(0.0, dtype=pl.Float32)
    norm_hi = pl.lit(1.0, dtype=pl.Float32)
    lo_expr = pl.col("signal_name").replace_strict(
        {k: v[0] for k, v in NORM_RANGE.items()}, return_dtype=pl.Float32
    )
    hi_expr = pl.col("signal_name").replace_strict(
        {k: v[1] for k, v in NORM_RANGE.items()}, return_dtype=pl.Float32
    )
    type_id_expr = pl.col("signal_name").replace_strict(
        SIGNAL_NAME_TO_ID, return_dtype=pl.Int32
    )

    paired = paired.with_columns(
        ((pl.col("valuenum") - lo_expr) / (hi_expr - lo_expr))
            .clip(norm_lo, norm_hi)
            .alias("norm_value"),
        type_id_expr.alias("item_type_id"),
    )

    print(f"  Pairs: {len(paired)} (note × signal) rows")
    return paired


# ---------------------------------------------------------------------------
# Step 5 – Full pipeline
# ---------------------------------------------------------------------------
def write_signal_metadata(output_dir: Path) -> None:
    """Writes signal_metadata.json so downstream code knows item_type_id mapping."""
    meta = {
        "n_signal_types": len(ALL_SIGNAL_NAMES),
        "signal_names": ALL_SIGNAL_NAMES,
        "signal_name_to_id": SIGNAL_NAME_TO_ID,
        "norm_range": {k: list(v) for k, v in NORM_RANGE.items()},
        "vitals_catalog": VITALS_CATALOG,
        "labs_catalog": LABS_CATALOG,
    }
    out = output_dir / "signal_metadata.json"
    out.write_text(json.dumps(meta, indent=2))
    print(f"  Wrote signal metadata → {out}")


def run_extraction(toy: bool = False) -> pl.DataFrame:
    """
    Runs the full extraction pipeline and writes:
      - data/processed/cardio_pairs.csv  (note × signal pairs)
      - data/processed/signal_metadata.json
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "cardio_pairs.csv"

    print("[1/5] Loading cardio cohort...")
    cohort = load_cohort()
    if toy:
        cohort = cohort.head(1000)
        print(f"  [toy mode] trimmed to {len(cohort)} stays")

    print("[2/5] Loading radiology notes (24 h window, with leakage filter)...")
    notes = load_notes(cohort)

    print("[3/5] Scanning chartevents for vitals (7 itemids)...")
    vitals = load_vitals(cohort)

    print("[4/5] Scanning labevents for labs (7 itemids)...")
    labs = load_labs(cohort)

    print("[5/5] Pairing notes ↔ signals (±2 h window)...")
    signals = pl.concat([vitals, labs], how="vertical")
    pairs = pair_notes_signals(notes, signals)

    pairs = pairs.join(
        cohort.select(["stay_id", "subject_id", "first_careunit", "mortality"]),
        on="stay_id",
        how="left",
    )

    pairs.write_csv(output_path)
    write_signal_metadata(PROCESSED_DIR)

    print(
        f"\nSaved {len(pairs):,} pairs → {output_path}\n"
        f"  unique notes:  {pairs['note_id'].n_unique()}\n"
        f"  unique stays:  {pairs['stay_id'].n_unique()}\n"
        f"  unique subj:   {pairs['subject_id'].n_unique()}\n"
        f"  signals seen:  {sorted(pairs['signal_name'].unique().to_list())}\n"
        f"  mortality:     {pairs.unique('stay_id')['mortality'].mean():.1%}"
    )
    return pairs


if __name__ == "__main__":
    import sys
    toy_mode = "--full" not in sys.argv
    run_extraction(toy=toy_mode)
