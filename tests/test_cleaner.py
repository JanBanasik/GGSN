import polars as pl

from src.data_prep.cleaner import filter_leakage_phrases


def test_filter_leakage_phrases_is_case_insensitive() -> None:
    notes = pl.DataFrame(
        {
            "note_id": ["safe", "expired", "dnr"],
            "text": ["Routine chest radiograph", "Patient EXPIRED", "DNR documented"],
        }
    )

    filtered = filter_leakage_phrases(notes)

    assert filtered["note_id"].to_list() == ["safe"]


def test_filter_leakage_phrases_preserves_empty_frame() -> None:
    notes = pl.DataFrame({"text": []}, schema={"text": pl.String})

    assert filter_leakage_phrases(notes).equals(notes)
