"""
Extract Charlson comorbidity ICD-10 codes per ICU stay.

DATA LEAKAGE NOTE:
  MIMIC-IV diagnoses_icd codes are assigned at DISCHARGE, not at ICU admission.
  Using ALL ICD codes risks leakage: codes assigned post-outcome for deceased
  patients may correlate with mortality for documentation reasons, not clinical ones.

  Mitigation: we use ONLY Charlson comorbidity categories — pre-existing chronic
  conditions (diabetes, CHF, COPD, cancer, etc.) that are coded consistently
  regardless of whether the patient survived. These represent what the patient
  HAD going in, not what happened during or after the stay.

  ICD-10 mapping based on: Quan et al. (2005) "Updating and Validating the
  Charlson Comorbidity Index and Score for Risk Adjustment in Hospital Discharge
  Abstracts" (Med Care 43:1130-1139).

Usage:
    uv run python -m src.data_prep.extract_icd [--cohort all-icus]

Output:
    data/processed/icd_charlson_<cohort>.csv
    Columns: stay_id, c_mi, c_chf, c_pvd, ..., c_aids  (N_CHARLSON binary cols)
"""

from __future__ import annotations

import argparse

import polars as pl

from src.data_prep.extractor import MIMIC_BASE, PROCESSED_DIR, load_cohort

# ---------------------------------------------------------------------------
# Charlson comorbidity ICD-10 mapping (prefix-based, dots stripped)
# ---------------------------------------------------------------------------
# Each tuple: (column_name, [ICD-10 prefixes without dots, uppercase])
# A code matches a category if it starts with ANY of the listed prefixes.
CHARLSON_CATEGORIES: list[tuple[str, list[str]]] = [
    ("c_mi", ["I21", "I22"]),
    ("c_chf", ["I50"]),
    (
        "c_pvd",
        [
            "I70",
            "I71",
            "I731",
            "I738",
            "I739",
            "I771",
            "I790",
            "I792",
            "K551",
            "K558",
            "K559",
            "Z958",
            "Z959",
        ],
    ),
    (
        "c_cvd",
        [
            "I60",
            "I61",
            "I62",
            "I63",
            "I64",
            "I65",
            "I66",
            "I67",
            "I68",
            "I69",
            "G45",
            "G46",
            "H340",
        ],
    ),
    ("c_dementia", ["F00", "F01", "F02", "F03", "G30", "G311"]),
    (
        "c_copd",
        [
            "J40",
            "J41",
            "J42",
            "J43",
            "J44",
            "J45",
            "J46",
            "J47",
            "J60",
            "J61",
            "J62",
            "J63",
            "J64",
            "J65",
            "J66",
            "J67",
            "J684",
            "J701",
            "J703",
        ],
    ),
    ("c_ctd", ["M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360"]),
    ("c_pud", ["K25", "K26", "K27", "K28"]),
    (
        "c_liver_mild",
        [
            "B18",
            "K700",
            "K701",
            "K702",
            "K703",
            "K709",
            "K713",
            "K714",
            "K715",
            "K717",
            "K73",
            "K74",
            "K760",
            "K762",
            "K763",
            "K764",
            "K768",
            "K769",
            "Z944",
        ],
    ),
    (
        "c_dm_mild",
        [
            "E100",
            "E101",
            "E106",
            "E108",
            "E109",
            "E110",
            "E111",
            "E116",
            "E118",
            "E119",
            "E120",
            "E121",
            "E126",
            "E128",
            "E129",
            "E130",
            "E131",
            "E136",
            "E138",
            "E139",
            "E140",
            "E141",
            "E146",
            "E148",
            "E149",
        ],
    ),
    (
        "c_dm_comp",
        [
            "E102",
            "E103",
            "E104",
            "E105",
            "E107",
            "E112",
            "E113",
            "E114",
            "E115",
            "E117",
            "E122",
            "E123",
            "E124",
            "E125",
            "E127",
            "E132",
            "E133",
            "E134",
            "E135",
            "E137",
            "E142",
            "E143",
            "E144",
            "E145",
            "E147",
        ],
    ),
    (
        "c_hemi",
        [
            "G041",
            "G114",
            "G801",
            "G802",
            "G81",
            "G82",
            "G830",
            "G831",
            "G832",
            "G833",
            "G834",
            "G839",
        ],
    ),
    (
        "c_ckd",
        [
            "I120",
            "I131",
            "N032",
            "N033",
            "N034",
            "N035",
            "N036",
            "N037",
            "N052",
            "N053",
            "N054",
            "N055",
            "N056",
            "N057",
            "N183",
            "N184",
            "N185",
            "N19",
            "N250",
            "Z490",
            "Z491",
            "Z492",
            "Z940",
            "Z992",
        ],
    ),
    (
        "c_solid_tumor",
        [
            "C00",
            "C01",
            "C02",
            "C03",
            "C04",
            "C05",
            "C06",
            "C07",
            "C08",
            "C09",
            "C10",
            "C11",
            "C12",
            "C13",
            "C14",
            "C15",
            "C16",
            "C17",
            "C18",
            "C19",
            "C20",
            "C21",
            "C22",
            "C23",
            "C24",
            "C25",
            "C26",
            "C30",
            "C31",
            "C32",
            "C33",
            "C34",
            "C37",
            "C38",
            "C39",
            "C40",
            "C41",
            "C43",
            "C45",
            "C46",
            "C47",
            "C48",
            "C49",
            "C50",
            "C51",
            "C52",
            "C53",
            "C54",
            "C55",
            "C56",
            "C57",
            "C58",
            "C60",
            "C61",
            "C62",
            "C63",
            "C64",
            "C65",
            "C66",
            "C67",
            "C68",
            "C69",
            "C70",
            "C71",
            "C72",
            "C73",
            "C74",
            "C75",
            "C76",
        ],
    ),
    ("c_leukemia", ["C91", "C92", "C93", "C94", "C95"]),
    ("c_lymphoma", ["C81", "C82", "C83", "C84", "C85", "C88", "C96"]),
    (
        "c_liver_sev",
        ["I850", "I859", "I864", "I982", "K704", "K711", "K721", "K729", "K765", "K766", "K767"],
    ),
    ("c_mets", ["C77", "C78", "C79", "C80"]),
    ("c_aids", ["B20", "B21", "B22", "B24"]),
]

CHARLSON_COLS = [name for name, _ in CHARLSON_CATEGORIES]
N_CHARLSON = len(CHARLSON_CATEGORIES)  # 19


def _code_matches(code_norm: str) -> list[int]:
    """Return a binary vector (length N_CHARLSON) for one normalized ICD code."""
    return [
        1 if any(code_norm.startswith(pfx) for pfx in prefixes) else 0
        for _, prefixes in CHARLSON_CATEGORIES
    ]


def extract_charlson(cohort_df: pl.DataFrame) -> pl.DataFrame:
    """
    Join ICU stays with diagnoses_icd, map to Charlson comorbidity binary vector.

    Args:
        cohort_df: DataFrame with columns (subject_id, hadm_id, stay_id, ...).

    Returns:
        DataFrame with columns [stay_id, c_mi, c_chf, ..., c_aids].
        One row per stay; stays without any diagnoses get all-zero vectors.
    """
    diag = (
        pl.scan_csv(MIMIC_BASE / "hosp" / "diagnoses_icd.csv.gz")
        .filter(pl.col("icd_version") == 10)
        .select(["hadm_id", "icd_code"])
        .collect()
    )

    # Normalize: remove dots, uppercase
    diag = diag.with_columns(
        pl.col("icd_code").str.replace_all(r"\.", "").str.to_uppercase().alias("icd_norm")
    )

    # Map each unique normalized code to a Charlson binary vector
    unique_codes = diag["icd_norm"].unique().to_list()
    code_to_vec: dict[str, list[int]] = {c: _code_matches(c) for c in unique_codes}

    # Expand: one row per (hadm_id, category)
    records = []
    for row in diag.iter_rows(named=True):
        vec = code_to_vec[row["icd_norm"]]
        records.append({"hadm_id": row["hadm_id"], **dict(zip(CHARLSON_COLS, vec, strict=False))})

    diag_expanded = pl.DataFrame(records)

    # Aggregate to hadm_id: OR across all codes (1 if any code matches the category)
    agg_exprs = [pl.col(c).max().alias(c) for c in CHARLSON_COLS]
    diag_per_hadm = diag_expanded.group_by("hadm_id").agg(agg_exprs)

    # Join to stays (left join: stays without diagnoses → all zeros)
    stay_hadm = cohort_df.select(["stay_id", "hadm_id"])
    result = (
        stay_hadm.join(diag_per_hadm, on="hadm_id", how="left")
        .with_columns([pl.col(c).fill_null(0).cast(pl.Int8) for c in CHARLSON_COLS])
        .drop("hadm_id")
    )

    n_any = result.select((pl.sum_horizontal([pl.col(c) for c in CHARLSON_COLS]) > 0).sum()).item()
    print(
        f"  {len(result):,} stays | {n_any:,} ({n_any / len(result):.1%}) have ≥1 Charlson comorbidity"
    )
    prevalence = {c: result[c].mean() for c in CHARLSON_COLS}
    top5 = sorted(prevalence.items(), key=lambda x: -x[1])[:5]
    print(f"  Top-5 comorbidities: {', '.join(f'{c}={v:.1%}' for c, v in top5)}")
    return result


def run_extraction(cohort: str = "all-icus") -> pl.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"icd_charlson_{cohort}.csv"

    print(f"[1/2] Loading cohort [{cohort}]…")
    cohort_df = load_cohort(cohort)

    print("[2/2] Extracting Charlson comorbidities from diagnoses_icd…")
    result = extract_charlson(cohort_df)

    result.write_csv(out_path)
    print(f"Saved → {out_path}  ({len(result):,} rows, {N_CHARLSON} comorbidity columns)")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract Charlson ICD-10 comorbidities per stay")
    p.add_argument("--cohort", choices=("cardio", "all-icus"), default="all-icus")
    args = p.parse_args()
    run_extraction(cohort=args.cohort)
