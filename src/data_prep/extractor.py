from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl

from src.data_prep.cleaner import filter_leakage_phrases

# ---------------------------------------------------------------------------
# Config – MIMIC-IV data roots (override with env vars — pendrive / another disk)
# ---------------------------------------------------------------------------
# Default: <repo>/data/raw/mimiciv/3.1 and …/mimic-iv-note/2.2
# Override if needed:
#   export MIMIC_IV_ROOT="/Volumes/MyDrive/mimiciv/3.1"
#   export MIMIC_IV_NOTE_ROOT="/Volumes/MyDrive/mimic-iv-note/2.2"
#
# Expected layout under MIMIC_IV_ROOT:
#   icu/icustays.csv.gz, icu/chartevents.csv.gz, hosp/admissions.csv.gz, hosp/labevents.csv.gz
# Under MIMIC_IV_NOTE_ROOT:
#   note/radiology.csv.gz

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _root_from_env(var: str, fallback: Path) -> Path:
    raw = os.getenv(var, "").strip()
    return Path(raw).expanduser() if raw else fallback


MIMIC_BASE = _root_from_env(
    "MIMIC_IV_ROOT",
    PROJECT_ROOT / "data" / "raw" / "mimiciv" / "3.1",
)
NOTES_BASE = _root_from_env(
    "MIMIC_IV_NOTE_ROOT",
    PROJECT_ROOT / "data" / "raw" / "mimic-iv-note" / "2.2",
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CARDIO_UNITS = [
    "Coronary Care Unit (CCU)",
    "Cardiovascular Intensive Care Unit (CVICU)",
]

# Cohort presets: None means "no filter — use all ICU types".
COHORT_PRESETS: dict[str, list[str] | None] = {
    "cardio":   CARDIO_UNITS,
    "all-icus": None,
}
DEFAULT_COHORT = "cardio"

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
def load_cohort(cohort: str = DEFAULT_COHORT) -> pl.DataFrame:
    """
    Loads ICU stays for the given cohort preset and joins in-hospital mortality
    from ``admissions.hospital_expire_flag``.

    Args:
        cohort: key into ``COHORT_PRESETS`` — ``"cardio"`` for CCU/CVICU only,
                ``"all-icus"`` to skip the unit filter and take every ICU stay.

    Returns columns:
        subject_id, hadm_id, stay_id, first_careunit, intime, outtime, mortality
    """
    if cohort not in COHORT_PRESETS:
        raise ValueError(f"Unknown cohort {cohort!r}; expected one of {list(COHORT_PRESETS)}")
    units = COHORT_PRESETS[cohort]
    icustays_lf = pl.scan_csv(MIMIC_BASE / "icu" / "icustays.csv.gz")
    if units is not None:
        icustays_lf = icustays_lf.filter(pl.col("first_careunit").is_in(units))
    icustays_lf = (
        icustays_lf
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

    cohort_df = (
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
        f"  Cohort [{cohort}]: {len(cohort_df)} stays | "
        f"mortality rate (in-hospital): {cohort_df['mortality'].mean():.1%} | "
        f"unique subjects: {cohort_df['subject_id'].n_unique()}"
    )
    return cohort_df


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
        .select(["stay_id", "event_time", "signal_name", "valuenum", "intime"])
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
        .select(["stay_id", "event_time", "signal_name", "valuenum", "intime"])
        .collect()
    )

    print(
        f"  Labs: {len(labs)} rows | "
        f"signals: {sorted(labs['signal_name'].unique().to_list())}"
    )
    return labs


# ---------------------------------------------------------------------------
# Step 4 – Pair notes ↔ signals (note-level window or stay-level)
# ---------------------------------------------------------------------------
PAIR_STRATEGY_NOTE = "note_level"
PAIR_STRATEGY_STAY = "stay_level"


def aggregate_notes_per_stay(notes: pl.DataFrame) -> pl.DataFrame:
    """
    Collapse radiology notes to one row per stay: texts concatenated in time order.
    Synthetic ``note_id``: ``stay_<stay_id>`` for downstream CardiacPairsDataset.
    """
    out = (
        notes.sort("note_time")
        .group_by("stay_id", maintain_order=True)
        .agg([
            pl.col("text").implode().list.join("\n\n").alias("text"),
            pl.col("note_time").min().alias("note_time"),
        ])
        .with_columns(
            (pl.lit("stay_") + pl.col("stay_id").cast(pl.Utf8)).alias("note_id")
        )
    )
    return out.select(["note_id", "stay_id", "note_time", "text"])


def pair_notes_signals(
    notes: pl.DataFrame,
    signals: pl.DataFrame,
    *,
    pair_strategy: str = PAIR_STRATEGY_NOTE,
    pair_window_hours: float = 2.0,
) -> pl.DataFrame:
    """
    ``note_level``: for each note, keep signals with ``event_time`` within
    ±pair_window_hours of ``note_time``.

    ``stay_level``: one synthetic note per stay (concatenated texts); pair with
    **all** signals in the stay's 24h window (already enforced in ``load_*``).

    Returns one row per (note, signal) pair with normalized value and
    item_type_id (0..N_signals-1).
    """
    if pair_strategy == PAIR_STRATEGY_STAY:
        note_rows = aggregate_notes_per_stay(notes)
        paired = note_rows.join(signals, on="stay_id", how="inner")
        print("  Pairing strategy: stay_level (full 24h vitals/labs vs concatenated notes)")
    elif pair_strategy == PAIR_STRATEGY_NOTE:
        window_sec = float(pair_window_hours) * 3600.0
        paired = (
            notes.join(signals, on="stay_id", how="inner")
            .filter(
                (pl.col("event_time") - pl.col("note_time"))
                .dt.total_seconds()
                .abs()
                <= window_sec
            )
        )
        print(
            f"  Pairing strategy: note_level (±{pair_window_hours:g} h window)"
        )
    else:
        raise ValueError(
            f"pair_strategy must be '{PAIR_STRATEGY_NOTE}' or '{PAIR_STRATEGY_STAY}', "
            f"got {pair_strategy!r}"
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
        # Hours from ICU admission (intime). Clipped to [0, 24] — the 24h
        # extraction window guarantees this range, but clip keeps it explicit
        # for downstream normalization.
        ((pl.col("event_time") - pl.col("intime"))
            .dt.total_seconds() / 3600.0)
            .clip(pl.lit(0.0, dtype=pl.Float32), pl.lit(24.0, dtype=pl.Float32))
            .cast(pl.Float32)
            .alias("event_hours_from_intime"),
    ).drop("intime")  # not needed in final CSV

    print(f"  Pairs: {len(paired)} (note × signal) rows")
    return paired


# ---------------------------------------------------------------------------
# Step 5 – Full pipeline
# ---------------------------------------------------------------------------
def pairs_filename(cohort: str, pair_strategy: str) -> str:
    """Canonical CSV name per (cohort, strategy) so different runs don't overwrite."""
    return f"pairs_{cohort}_{pair_strategy}.csv"


def metadata_filename(cohort: str, pair_strategy: str) -> str:
    return f"signal_metadata_{cohort}_{pair_strategy}.json"


def write_signal_metadata(
    output_dir: Path,
    *,
    cohort: str | None = None,
    pair_strategy: str | None = None,
    pair_window_hours: float | None = None,
    out_name: str = "signal_metadata.json",
) -> None:
    """Writes signal metadata so downstream code knows item_type_id mapping."""
    meta: dict = {
        "n_signal_types": len(ALL_SIGNAL_NAMES),
        "signal_names": ALL_SIGNAL_NAMES,
        "signal_name_to_id": SIGNAL_NAME_TO_ID,
        "norm_range": {k: list(v) for k, v in NORM_RANGE.items()},
        "vitals_catalog": VITALS_CATALOG,
        "labs_catalog": LABS_CATALOG,
    }
    if cohort is not None:
        meta["cohort"] = cohort
    if pair_strategy is not None:
        meta["pair_strategy"] = pair_strategy
    if pair_window_hours is not None:
        meta["pair_window_hours"] = pair_window_hours
    out = output_dir / out_name
    out.write_text(json.dumps(meta, indent=2))
    print(f"  Wrote signal metadata → {out}")


def run_extraction(
    toy: bool = False,
    *,
    cohort: str = DEFAULT_COHORT,
    pair_strategy: str = PAIR_STRATEGY_NOTE,
    pair_window_hours: float = 2.0,
    output_name: str | None = None,
    metadata_name: str | None = None,
) -> pl.DataFrame:
    """
    Runs the full extraction pipeline.

    Output files (default names depend on cohort + strategy so different
    extractions don't overwrite each other):
      - data/processed/pairs_<cohort>_<strategy>.csv
      - data/processed/signal_metadata_<cohort>_<strategy>.json
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csv_name = output_name or pairs_filename(cohort, pair_strategy)
    meta_name = metadata_name or metadata_filename(cohort, pair_strategy)
    output_path = PROCESSED_DIR / csv_name

    print(f"[1/5] Loading cohort [{cohort}]...")
    cohort_df = load_cohort(cohort)
    if toy:
        cohort_df = cohort_df.head(1000)
        print(f"  [toy mode] trimmed to {len(cohort_df)} stays")

    print("[2/5] Loading radiology notes (24 h window, with leakage filter)...")
    notes = load_notes(cohort_df)

    print("[3/5] Scanning chartevents for vitals (7 itemids)...")
    vitals = load_vitals(cohort_df)

    print("[4/5] Scanning labevents for labs (7 itemids)...")
    labs = load_labs(cohort_df)

    print("[5/5] Pairing notes ↔ signals...")
    signals = pl.concat([vitals, labs], how="vertical")
    pairs = pair_notes_signals(
        notes,
        signals,
        pair_strategy=pair_strategy,
        pair_window_hours=pair_window_hours,
    )

    pairs = pairs.join(
        cohort_df.select(["stay_id", "subject_id", "first_careunit", "mortality"]),
        on="stay_id",
        how="left",
    )

    pairs.write_csv(output_path)
    write_signal_metadata(
        PROCESSED_DIR,
        cohort=cohort,
        pair_strategy=pair_strategy,
        pair_window_hours=pair_window_hours,
        out_name=meta_name,
    )

    print(
        f"\nSaved {len(pairs):,} pairs → {output_path}\n"
        f"  cohort:        {cohort}\n"
        f"  pair_strategy: {pair_strategy}\n"
        f"  unique notes:  {pairs['note_id'].n_unique()}\n"
        f"  unique stays:  {pairs['stay_id'].n_unique()}\n"
        f"  unique subj:   {pairs['subject_id'].n_unique()}\n"
        f"  signals seen:  {sorted(pairs['signal_name'].unique().to_list())}\n"
        f"  mortality:     {pairs.unique('stay_id')['mortality'].mean():.1%}"
    )
    return pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MIMIC → pairs CSV (cohort + strategy aware)")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full cohort (default: toy subset of 1000 stays)",
    )
    parser.add_argument(
        "--cohort",
        choices=tuple(COHORT_PRESETS.keys()),
        default=DEFAULT_COHORT,
        help="cardio = CCU+CVICU only; all-icus = no unit filter (full MIMIC ICU)",
    )
    parser.add_argument(
        "--pair-strategy",
        choices=(PAIR_STRATEGY_NOTE, PAIR_STRATEGY_STAY),
        default=PAIR_STRATEGY_NOTE,
        help="note_level: ±window around each note; stay_level: one sample per stay (concat notes + 24h signals)",
    )
    parser.add_argument(
        "--pair-window-hours",
        type=float,
        default=2.0,
        help="Half-width in hours for note_level pairing (ignored for stay_level)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Override output CSV name (default: pairs_<cohort>_<strategy>.csv)",
    )
    args = parser.parse_args()
    run_extraction(
        toy=not args.full,
        cohort=args.cohort,
        pair_strategy=args.pair_strategy,
        pair_window_hours=args.pair_window_hours,
        output_name=args.output_name,
    )
