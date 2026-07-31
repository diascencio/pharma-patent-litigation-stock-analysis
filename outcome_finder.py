"""Attach FJC judgment outcomes to the pharmaceutical litigation dataset.

The two FJC source files are expected to sit beside this module by default:

- ``Civil 1970 to 1987.txt``
- ``civil_cases_1988_present.txt``

Call :func:`extract_outcome` with a prepared FJC dataframe to use another data
source without changing the matching logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd


MODULE_DIR: Final = Path(__file__).resolve().parent
FJC_1970_1987_FILE: Final = "Civil 1970 to 1987.txt"
FJC_1988_PRESENT_FILE: Final = "civil_cases_1988_present.txt"

DISTRICT_MAP: Final[dict[str, str]] = {
    "00": "med",
    "01": "mad",
    "04": "prd",
    "05": "ctd",
    "06": "nynd",
    "07": "nyed",
    "08": "nysd",
    "09": "nywd",
    "10": "vtd",
    "11": "ded",
    "12": "njd",
    "13": "paed",
    "14": "pamd",
    "15": "pawd",
    "16": "mdd",
    "17": "nced",
    "18": "ncmd",
    "20": "scd",
    "22": "vaed",
    "23": "vawd",
    "24": "wvnd",
    "26": "alnd",
    "27": "almd",
    "37": "msnd",
    "39": "txnd",
    "40": "txed",
    "41": "txsd",
    "42": "txwd",
    "45": "mied",
    "46": "miwd",
    "47": "ohnd",
    "48": "ohsd",
    "51": "tnwd",
    "52": "ilnd",
    "53": "ilcd",
    "54": "ilsd",
    "56": "insd",
    "57": "wied",
    "58": "wiwd",
    "60": "ared",
    "63": "iasd",
    "64": "mnd",
    "65": "moed",
    "66": "mowd",
    "70": "azd",
    "71": "cand",
    "73": "cacd",
    "74": "casd",
    "78": "nvd",
    "81": "wawd",
    "82": "cod",
    "83": "ksd",
    "87": "okwd",
    "88": "utd",
    "90": "dcd",
    "3A": "flmd",
    "3C": "flsd",
    "3E": "gand",
    "3N": "lamd",
}


def map_district(code: object) -> str | None:
    """Map an FJC district code to the abbreviated ``district_id`` value."""
    if pd.isna(code):
        return None

    code_text = str(code).strip().upper()
    if code_text.endswith(".0"):
        code_text = code_text[:-2]
    return DISTRICT_MAP.get(code_text, DISTRICT_MAP.get(code_text.zfill(2)))


def build_case_number(row: pd.Series) -> str:
    """Build the ``office:year-cv-sequence`` case-number format used downstream."""
    office = str(row["OFFICE"]).strip()
    if office.endswith(".0"):
        office = office[:-2]

    # Preserve leading zeros that may have been removed by spreadsheet software.
    docket = str(row["DOCKET"]).strip()
    if docket.endswith(".0"):
        docket = docket[:-2]
    docket = docket.zfill(7)

    return f"{office}:{docket[:2]}-cv-{docket[2:]}"


def load_fjc_data(data_dir: str | Path = MODULE_DIR) -> pd.DataFrame:
    """Load and combine the two FJC civil-case files from ``data_dir``."""
    data_dir = Path(data_dir)
    source_files = (FJC_1970_1987_FILE, FJC_1988_PRESENT_FILE)

    frames = [
        pd.read_csv(
            data_dir / filename,
            sep="\t",
            encoding="latin-1",
            dtype=str,
        )
        for filename in source_files
    ]
    fjc_data = pd.concat(frames, ignore_index=True)
    fjc_data.columns = fjc_data.columns.str.strip()
    return fjc_data


def _prepare_fjc_data(fjc_data: pd.DataFrame) -> pd.DataFrame:
    """Create matching fields without modifying the caller's dataframe."""
    required_columns = {"DISTRICT", "OFFICE", "DOCKET", "JUDGMENT"}
    missing = sorted(required_columns - set(fjc_data.columns))
    if missing:
        raise ValueError(f"FJC data is missing required columns: {missing}")

    prepared = fjc_data.copy()
    prepared.columns = prepared.columns.str.strip()
    prepared["district_id"] = prepared["DISTRICT"].map(map_district)
    prepared["case_number"] = prepared.apply(build_case_number, axis=1)
    prepared["match_key"] = (
        prepared["district_id"].astype("string").str.strip().str.lower()
        + "_"
        + prepared["case_number"].astype("string").str.strip()
    )
    return prepared


def extract_outcome(
    original_df: pd.DataFrame,
    fjc_data: pd.DataFrame | None = None,
    *,
    data_dir: str | Path = MODULE_DIR,
    show_diagnostics: bool = True,
) -> pd.DataFrame:
    """Merge FJC ``JUDGMENT`` values into the original case dataframe.

    Matching uses both ``district_id`` and ``case_number`` because case numbers
    can repeat across federal districts. The input dataframe is never modified.
    """
    required_columns = {"district_id", "case_number"}
    missing = sorted(required_columns - set(original_df.columns))
    if missing:
        raise ValueError(f"original_df is missing required columns: {missing}")

    source = load_fjc_data(data_dir) if fjc_data is None else fjc_data
    prepared_fjc = _prepare_fjc_data(source)
    prepared_fjc = prepared_fjc.loc[prepared_fjc["match_key"].notna()].copy()

    original = original_df.copy()
    original["match_key"] = (
        original["district_id"].astype("string").str.strip().str.lower()
        + "_"
        + original["case_number"].astype("string").str.strip()
    )

    duplicate_count = int(prepared_fjc["match_key"].duplicated().sum())
    if duplicate_count:
        if show_diagnostics:
            print(
                f"Warning: {duplicate_count:,} duplicate FJC match key(s) found; "
                "keeping the first occurrence."
            )
        prepared_fjc = prepared_fjc.drop_duplicates("match_key", keep="first")

    merged = original.merge(
        prepared_fjc[["match_key", "JUDGMENT"]],
        on="match_key",
        how="left",
        validate="m:1",
    ).drop(columns="match_key")

    if show_diagnostics:
        matched = int(merged["JUDGMENT"].notna().sum())
        total = len(merged)
        match_rate = matched / total if total else 0.0
        print(f"Matched {matched:,} of {total:,} cases ({match_rate:.1%}).")

    return merged


__all__ = [
    "DISTRICT_MAP",
    "build_case_number",
    "extract_outcome",
    "load_fjc_data",
    "map_district",
]
