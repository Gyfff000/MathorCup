"""AI-assisted implementation: OpenAI Codex / GPT-6, 2026-10-24.
All source files are read-only. Never join on precision-damaged beer_id.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

META = ["beer", "beer_id", "tasting_category_fine"]
AGGREGATES = {"kcalperc", "esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum"}
NON_MOLECULES = {"color.EBC.", "betaglucan.mg.L.", "pH", "protein.g.L.", "bitterness", "GFS", "CO2.PSI.", "unfermentables"} | AGGREGATES
CHEMICAL_DEFECTS = ["DMS", "X2.3.butanedione", "X4VG", "X.E..2.nonenal"]


@dataclass
class BeerData:
    root: Path
    meta: pd.DataFrame
    X: pd.DataFrame
    Y: pd.DataFrame
    given_corr: pd.DataFrame
    thresholds: pd.DataFrame
    sensory_names: pd.DataFrame
    calibrated: pd.DataFrame
    chemical_meta: pd.DataFrame


def find_data(argument=None):
    if argument:
        root = Path(argument).resolve()
        if not (root / "1.csv").exists():
            raise FileNotFoundError(f"Expected 1.csv in {root}")
        return root
    repo = Path(__file__).resolve().parents[1]
    for root in [repo / "data/raw", repo.parent / "materials/A/附件"]:
        if (root / "1.csv").exists():
            return root
    raise FileNotFoundError("Supply --data-dir pointing to the original A attachments")


def align_by_beer(chemical, sensory):
    for label, frame in [("chemical", chemical), ("sensory", sensory)]:
        if frame["beer"].isna().any() or frame["beer"].duplicated().any():
            raise ValueError(f"{label}: beer must be non-null and unique")
    if set(chemical.beer) != set(sensory.beer):
        raise ValueError("Chemical and sensory sample sets differ")
    sensory = sensory.set_index("beer").loc[chemical.beer].reset_index()
    if not chemical.tasting_category_fine.reset_index(drop=True).equals(sensory.tasting_category_fine):
        raise ValueError("Style labels do not agree after beer alignment")
    return sensory


def load_data(root):
    root = Path(root)
    chemical = pd.read_csv(root / "1.csv")
    sensory = align_by_beer(chemical, pd.read_csv(root / "4.csv"))
    X = chemical.drop(columns=META).astype(float)
    Y = sensory.drop(columns=META).astype(float)
    if Y.isna().any().any():
        raise ValueError("Targets contain missing values; target handling needs explicit review")
    corr = pd.read_csv(root / "2.csv", index_col=0)
    if corr.shape != (X.shape[1], X.shape[1]):
        raise ValueError("Correlation matrix dimensions do not match chemical fields")
    if list(corr.index) != list(corr.columns):
        raise ValueError("Correlation row/column names differ")
    # The positional alias map is empirically checked over the full matrix.
    # Thus it is not inferred from just matching the number of columns.
    mismatch = np.nanmean(np.abs(X.corr(method="spearman").to_numpy() - corr.to_numpy()))
    if mismatch > 0.002:
        raise ValueError("Cannot verify ordered chemical aliases against sample correlations")
    book = root / "8.xlsx"
    thresholds = pd.read_excel(book, sheet_name="Sensory thresholds", na_values="NA")
    names = pd.read_excel(book, sheet_name="Sensory names", na_values="NA")
    calibrated = pd.read_excel(book, sheet_name="Calibrated chemical dataset", na_values="NA")
    if calibrated.beer.duplicated().any():
        raise ValueError("Calibrated reference has duplicate beer keys")
    calibrated = calibrated.set_index("beer").loc[chemical.beer].reset_index()
    cm = thresholds.dropna(subset=["field_name_R"]).drop_duplicates("field_name_R").set_index("field_name_R")
    rows = []
    for col, alias in zip(X.columns, corr.index):
        row = {"field": col, "matrix_alias": alias, "kind": "aggregate" if col in AGGREGATES else "physical_or_mixture" if col in NON_MOLECULES else "compound_or_element"}
        for name in ["chemical_name", "origin_paper", "unit", "off_flavors", "threshold_median", "threshold_mean"]:
            row[name] = cm.at[col, name] if col in cm.index else np.nan
        rows.append(row)
    return BeerData(root, chemical[META], X, Y, corr, thresholds, names, calibrated, pd.DataFrame(rows))
