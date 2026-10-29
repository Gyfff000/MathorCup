"""AI-assisted implementation: OpenAI Codex / GPT-6, 2026-10-24.
Team verification and modification are required; see docs/AI_USAGE.md.
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(frame, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.9g")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    return {
        "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "networkx", "matplotlib", "openpyxl"]},
    }


class TrainQuantileClipper(TransformerMixin, BaseEstimator):
    """Learn bounds on training samples only. Missing values remain missing."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        a = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(a, self.lower, axis=0)
        self.upper_ = np.nanquantile(a, self.upper, axis=0)
        self.n_features_in_ = a.shape[1]
        return self

    def transform(self, X):
        a = np.asarray(X, dtype=float)
        if a.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from training")
        return np.clip(a, self.lower_, self.upper_)


def preprocessing():
    return Pipeline([
        ("clip", TrainQuantileClipper()),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])


def ecdf_grades(reference, values):
    """ASSUMPTION: seven empirical bins, NOT calibrated sensory intensity."""
    bounds = np.quantile(np.asarray(reference), np.arange(1, 7) / 7, axis=0)
    values = np.asarray(values)
    return (values[:, None, :] > bounds[None, :, :]).sum(axis=1).astype(int)


def progress(text):
    print(text, flush=True)
