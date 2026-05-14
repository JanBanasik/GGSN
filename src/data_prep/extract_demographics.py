"""
Extract per-stay demographic features from MIMIC-IV and save to
data/processed/demographics.csv.

Features (DEMO_DIM = 4):
    age_norm        float [0, 1]  — (anchor_age - 18) / 72, clipped
    gender_f        int   {0, 1}  — 1 = female
    is_emergency    int   {0, 1}  — EW EMER. or DIRECT EMER.
    is_elective     int   {0, 1}  — ELECTIVE or SURGICAL SAME DAY ADMISSION

Usage (from GGSN_Projektowe/):
    MIMIC_IV_ROOT=/path/to/mimiciv/3.1 uv run python -m src.data_prep.extract_demographics
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from src.data_prep.extractor import MIMIC_BASE, PROCESSED_DIR

# admission_type values that map to each binary flag
EMERGENCY_TYPES = {"EW EMER.", "DIRECT EMER."}
ELECTIVE_TYPES = {"ELECTIVE", "SURGICAL SAME DAY ADMISSION"}

AGE_MIN, AGE_MAX = 18.0, 90.0


def extract_demographics(out_path: Path | None = None) -> pl.DataFrame:
    """
    Join icustays × admissions × patients to produce one row per stay_id.

    Returns a DataFrame with columns:
        stay_id, age_norm, gender_f, is_emergency, is_elective
    """
    out_path = out_path or (PROCESSED_DIR / "demographics.csv")

    print("Loading MIMIC tables…")
    icustays = (
        pl.scan_csv(MIMIC_BASE / "icu" / "icustays.csv.gz")
        .select(["stay_id", "hadm_id", "subject_id"])
    )
    admissions = (
        pl.scan_csv(MIMIC_BASE / "hosp" / "admissions.csv.gz")
        .select(["hadm_id", "admission_type"])
    )
    patients = (
        pl.scan_csv(MIMIC_BASE / "hosp" / "patients.csv.gz")
        .select(["subject_id", "anchor_age", "gender"])
    )

    df = (
        icustays
        .join(admissions, on="hadm_id", how="left")
        .join(patients, on="subject_id", how="left")
        .with_columns([
            # age normalised to [0, 1]; fill missing with 0.5
            ((pl.col("anchor_age").cast(pl.Float32).clip(AGE_MIN, AGE_MAX) - AGE_MIN)
             / (AGE_MAX - AGE_MIN))
            .fill_null(0.5)
            .alias("age_norm"),

            # gender: F=1, M=0, unknown=0
            (pl.col("gender") == "F").cast(pl.Int8).fill_null(0).alias("gender_f"),

            # admission urgency flags
            pl.col("admission_type")
            .is_in(list(EMERGENCY_TYPES)).cast(pl.Int8).fill_null(0)
            .alias("is_emergency"),

            pl.col("admission_type")
            .is_in(list(ELECTIVE_TYPES)).cast(pl.Int8).fill_null(0)
            .alias("is_elective"),
        ])
        .select(["stay_id", "age_norm", "gender_f", "is_emergency", "is_elective"])
        .collect()
    )

    print(f"  {len(df):,} stays extracted")
    print(f"  age_norm: mean={df['age_norm'].mean():.3f}")
    print(f"  gender_f: {df['gender_f'].mean():.1%} female")
    print(f"  is_emergency: {df['is_emergency'].mean():.1%}")
    print(f"  is_elective:  {df['is_elective'].mean():.1%}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out_path)
    print(f"Saved → {out_path}")
    return df


if __name__ == "__main__":
    extract_demographics()
