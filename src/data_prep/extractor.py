from __future__ import annotations

from pathlib import Path

import polars as pl

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
# Heart Rate (220045), Non-Invasive Blood Pressure mean (220181)
VITAL_ITEM_IDS = [220045, 220181]

_DT_FMT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Step 1 – Cardio cohort + mortality label
# ---------------------------------------------------------------------------
def load_cohort() -> pl.DataFrame:
    """
    Loads ICU stays for CCU/CVICU units and joins a binary mortality label
    derived from the patients.dod column.

    Returns a small collected DataFrame (only cardio stays).
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

    patients_lf = (
        pl.scan_csv(MIMIC_BASE / "hosp" / "patients.csv.gz")
        .select(["subject_id", "dod"])
        .with_columns(
            pl.when(pl.col("dod").is_not_null())
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias("mortality")
        )
        .drop("dod")
    )

    cohort = (
        icustays_lf
        .join(patients_lf, on="subject_id", how="left")
        .with_columns(pl.col("mortality").fill_null(pl.lit(0, dtype=pl.Int8)))
        .collect()
    )

    print(
        f"  Cohort: {len(cohort)} stays | "
        f"mortality rate: {cohort['mortality'].mean():.1%}"
    )
    return cohort


# ---------------------------------------------------------------------------
# Step 2 – Radiology notes filtered to [intime, intime + 24 h]
# ---------------------------------------------------------------------------
def load_notes(cohort: pl.DataFrame) -> pl.DataFrame:
    """
    Reads radiology.csv.gz and keeps only notes whose charttime falls within
    [intime, intime + 24h] for the matched ICU stay.  Join key: hadm_id.

    Returns a collected DataFrame with columns:
        note_id, stay_id, note_time, text
    """
    stay_times = cohort.select(["hadm_id", "stay_id", "intime"]).lazy()

    notes = (
        pl.scan_csv(NOTES_BASE / "note" / "radiology.csv.gz", quote_char='"')
        .select(["note_id", "subject_id", "hadm_id", "charttime", "text"])
        .with_columns(
            pl.col("charttime").str.to_datetime(_DT_FMT, strict=False)
        )
        .join(stay_times, on="hadm_id", how="inner")
        .filter(
            (pl.col("charttime") >= pl.col("intime"))
            & (pl.col("charttime") <= pl.col("intime") + pl.duration(hours=24))
        )
        .select(["note_id", "stay_id", "charttime", "text"])
        .rename({"charttime": "note_time"})
        .collect()
    )

    print(f"  Notes: {len(notes)} rows across {notes['stay_id'].n_unique()} stays")
    return notes


# ---------------------------------------------------------------------------
# Step 3 – Vitals from chartevents  (huge file – always use scan_csv!)
# ---------------------------------------------------------------------------
def load_vitals(cohort: pl.DataFrame) -> pl.DataFrame:
    """
    Lazily scans chartevents.csv.gz, keeps Heart Rate (220045) and
    Non-Invasive Blood Pressure mean (220181), restricted to cohort stay_ids.

    Returns a collected DataFrame with columns:
        stay_id, vital_time, itemid, valuenum
    """
    valid_stay_ids = cohort.select("stay_id")

    vitals = (
        pl.scan_csv(MIMIC_BASE / "icu" / "chartevents.csv.gz")
        .filter(pl.col("itemid").is_in(VITAL_ITEM_IDS))
        .select(["stay_id", "charttime", "itemid", "valuenum"])
        .with_columns(
            pl.col("charttime").str.to_datetime(_DT_FMT, strict=False),
            pl.col("valuenum").cast(pl.Float32, strict=False),
        )
        .join(valid_stay_ids.lazy(), on="stay_id", how="inner")
        .drop_nulls("valuenum")
        .rename({"charttime": "vital_time"})
        .collect()
    )

    print(
        f"  Vitals: {len(vitals)} rows | "
        f"items found: {sorted(vitals['itemid'].unique().to_list())}"
    )
    return vitals


# ---------------------------------------------------------------------------
# Step 4 – Pair each note with vitals within ±2 h
# ---------------------------------------------------------------------------
def pair_notes_vitals(notes: pl.DataFrame, vitals: pl.DataFrame) -> pl.DataFrame:
    """
    For every note, finds all vitals from the same stay_id whose vital_time
    lies within ±2 hours of the note_time.

    Strategy: join on stay_id (one-to-many), then filter by time delta.
    Both inputs are already small (post-cohort filtering), so the cross-join
    per stay is memory-safe.

    Returns a DataFrame with one row per (note, vital) pair.
    """
    paired = (
        notes.join(vitals, on="stay_id", how="inner")
        .filter(
            (pl.col("vital_time") - pl.col("note_time"))
            .dt.total_seconds()
            .abs()
            <= 2 * 3600
        )
    )

    print(f"  Pairs: {len(paired)} (note × vital) rows")
    return paired


# ---------------------------------------------------------------------------
# Step 5 – Full pipeline
# ---------------------------------------------------------------------------
def run_extraction(toy: bool = False) -> pl.DataFrame:
    """
    Runs the complete extraction pipeline and writes the result to
    data/processed/cardio_pairs.csv.

    Args:
        toy: Limit to the first 1 000 stays for fast local iteration.
             Per README instructions: start with a Toy Dataset.

    Returns:
        The final paired DataFrame (already collected).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "cardio_pairs.csv"

    print("[1/4] Loading cardio cohort...")
    cohort = load_cohort()
    if toy:
        cohort = cohort.head(1000)
        print(f"  [toy mode] trimmed to {len(cohort)} stays")

    print("[2/4] Loading radiology notes (24 h window)...")
    notes = load_notes(cohort)

    print("[3/4] Scanning chartevents for vitals (HR + BP)...")
    vitals = load_vitals(cohort)

    print("[4/4] Pairing notes ↔ vitals (±2 h window)...")
    pairs = pair_notes_vitals(notes, vitals)

    # Attach care-unit label and mortality for downstream use
    pairs = pairs.join(
        cohort.select(["stay_id", "first_careunit", "mortality"]),
        on="stay_id",
        how="left",
    )

    pairs.write_csv(output_path)
    print(f"\nSaved {len(pairs):,} rows → {output_path}")
    return pairs


if __name__ == "__main__":
    run_extraction(toy=True)
