"""
Extract full ICU-stay signals (vitals + labs) for Phase 2 GNN training.

Current pipeline (extractor.py) only loads signals within [intime, intime+24h]
because Phase 1 pairing pairs notes with nearby signals. The GNN (Phase 2) can
benefit from the FULL stay trajectory — chartevents every 15-30 min for the
entire admission, not just the first 24 hours.

DATA LEAKAGE NOTE:
  Signals are filtered to [intime, outtime] — i.e. within the ICU stay.
  The label is in-hospital mortality, and these signals are from within the
  same stay. This is NOT leakage: we're predicting whether the patient dies
  during this stay using data from this stay. Deteriorating vitals near death
  are part of the mortality signal we want the model to learn.

  We do NOT include events after outtime (discharge/death), so post-outcome
  chart entries (if any) are excluded.

Usage:
    uv run python -m src.data_prep.extract_all_stay_signals [--cohort all-icus]

Output:
    data/processed/all_stay_signals_<cohort>.csv
    Columns: stay_id, event_hours_from_intime, item_type_id, norm_value
"""

from __future__ import annotations

import argparse

import polars as pl

from src.data_prep.extractor import (
    ALL_LAB_ITEMIDS,
    ALL_VITAL_ITEMIDS,
    ITEMID_TO_SIGNAL,
    MIMIC_BASE,
    NORM_RANGE,
    PROCESSED_DIR,
    SIGNAL_NAME_TO_ID,
    load_cohort,
)

_DT_FMT = "%Y-%m-%d %H:%M:%S"


def load_all_stay_vitals(cohort_df: pl.DataFrame) -> pl.DataFrame:
    """
    Load chartevents for the full ICU stay duration [intime, outtime].

    Unlike extractor.load_vitals(), this does NOT cap at intime+24h.
    event_hours_from_intime can exceed 24 (typical ICU stays are 1-10 days).
    """
    stay_window = cohort_df.select(["stay_id", "intime", "outtime"]).lazy()
    itemid_lookup = pl.LazyFrame(
        {
            "itemid": list(ITEMID_TO_SIGNAL.keys()),
            "signal_name": list(ITEMID_TO_SIGNAL.values()),
        }
    )

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
            & (pl.col("charttime") <= pl.col("outtime"))  # full stay, not 24h cap
        )
        .drop_nulls("valuenum")
        .join(itemid_lookup, on="itemid", how="inner")
        .rename({"charttime": "event_time"})
        .select(["stay_id", "event_time", "signal_name", "valuenum", "intime"])
        .collect()
    )

    print(
        f"  Vitals (full stay): {len(vitals):,} rows | "
        f"signals: {sorted(vitals['signal_name'].unique().to_list())}"
    )
    return vitals


def load_all_stay_labs(cohort_df: pl.DataFrame) -> pl.DataFrame:
    """
    Load labevents for the full ICU stay duration [intime, outtime].
    """
    stay_window = cohort_df.select(["hadm_id", "stay_id", "intime", "outtime"]).lazy()
    itemid_lookup = pl.LazyFrame(
        {
            "itemid": list(ITEMID_TO_SIGNAL.keys()),
            "signal_name": list(ITEMID_TO_SIGNAL.values()),
        }
    )

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
            (pl.col("charttime") >= pl.col("intime")) & (pl.col("charttime") <= pl.col("outtime"))
        )
        .join(itemid_lookup, on="itemid", how="inner")
        .rename({"charttime": "event_time"})
        .select(["stay_id", "event_time", "signal_name", "valuenum", "intime"])
        .collect()
    )

    print(
        f"  Labs (full stay): {len(labs):,} rows | "
        f"signals: {sorted(labs['signal_name'].unique().to_list())}"
    )
    return labs


def normalize_signals(signals: pl.DataFrame) -> pl.DataFrame:
    """Apply per-signal normalization and compute event_hours_from_intime."""
    lo_expr = pl.col("signal_name").replace_strict(
        {k: v[0] for k, v in NORM_RANGE.items()}, return_dtype=pl.Float32
    )
    hi_expr = pl.col("signal_name").replace_strict(
        {k: v[1] for k, v in NORM_RANGE.items()}, return_dtype=pl.Float32
    )
    type_id_expr = pl.col("signal_name").replace_strict(SIGNAL_NAME_TO_ID, return_dtype=pl.Int32)

    return signals.with_columns(
        ((pl.col("valuenum") - lo_expr) / (hi_expr - lo_expr)).clip(0.0, 1.0).alias("norm_value"),
        type_id_expr.alias("item_type_id"),
        # Hours from ICU admission — NOT capped (full stay can be >24h)
        ((pl.col("event_time") - pl.col("intime")).dt.total_seconds() / 3600.0)
        .cast(pl.Float32)
        .alias("event_hours_from_intime"),
    ).select(["stay_id", "event_hours_from_intime", "item_type_id", "norm_value"])


def run_extraction(cohort: str = "all-icus") -> pl.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"all_stay_signals_{cohort}.csv"

    print(f"[1/3] Loading cohort [{cohort}]…")
    cohort_df = load_cohort(cohort)

    print("[2/3] Loading full-stay vitals…")
    vitals = load_all_stay_vitals(cohort_df)

    print("[3/3] Loading full-stay labs…")
    labs = load_all_stay_labs(cohort_df)

    print("Normalizing and saving…")
    signals = normalize_signals(pl.concat([vitals, labs], how="vertical"))
    signals = signals.sort(["stay_id", "event_hours_from_intime"])

    signals.write_csv(out_path)
    print(
        f"Saved → {out_path}\n"
        f"  {len(signals):,} signal events | "
        f"{signals['stay_id'].n_unique():,} stays | "
        f"median {len(signals) // max(signals['stay_id'].n_unique(), 1)} events/stay"
    )
    return signals


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract full-stay signals (vitals + labs)")
    p.add_argument("--cohort", choices=("cardio", "all-icus"), default="all-icus")
    args = p.parse_args()
    run_extraction(cohort=args.cohort)
