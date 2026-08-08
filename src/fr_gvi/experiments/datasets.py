"""Real binary-classification datasets for the Bayesian logistic benchmark.

The synthetic logistic cells control conditioning exactly, which is what makes
them useful for a theorem diagnostic and useless as evidence that the methods
work on data anyone actually has.  These five are the other half: real covariate
correlation, real conditioning, and a spread of ``(n, d)`` wide enough that no
single regime carries the conclusion.

============  ======  =====  ==============================================
Dataset       ``n``   ``d``  Why it is here
============  ======  =====  ==============================================
pima             768      9  small ``d``, mildly correlated
wdbc             569     31  strongly correlated, ``d`` in the thirties
ionosphere       351     34  ill-conditioned design, one degenerate column
sonar            208     61  ``d`` comparable to ``n``
spambase        4601     58  a few thousand observations, heavy-tailed
============  ======  =====  ==============================================

``d`` counts the intercept.  Every problem stays inside the manuscript's
hypotheses: a Bayesian logistic likelihood with a proper Gaussian prior
``N(0, lambda^{-1} I)`` is strongly log-concave with
``alpha = lambda`` and ``beta = lambda + lambda_max(X^T X) / 4``, both available
in closed form, so nothing here needs a hypothesis the theory does not have.

Provenance is pinned rather than trusted.  The raw ARFF bytes are cached under
``data/openml`` and checked against the SHA-256 digests committed in
``configs/datasets/manifest.json``; a silent upstream edit is an error, not a new
result.  Once the cache exists the campaign runs with no network at all.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "openml"
MANIFEST = ROOT / "configs" / "datasets" / "manifest.json"

# The split and standardization are part of the problem definition, so they are
# frozen here rather than left to a config: two runs of the same dataset must be
# the same posterior.
TEST_FRACTION = 0.2
SPLIT_SEED = 20260801


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    openml_id: int
    file_id: int
    name: str
    positive_label: str
    note: str

    @property
    def url(self) -> str:
        return f"https://openml.org/data/v1/download/{self.file_id}/{self.name}.arff"

    @property
    def path(self) -> Path:
        return CACHE / f"{self.key}.arff"


SPECS: dict[str, DatasetSpec] = {
    spec.key: spec
    for spec in (
        DatasetSpec("pima", 37, 37, "diabetes", "tested_positive", "Pima Indians diabetes"),
        DatasetSpec("wdbc", 1510, 1592318, "wdbc", "2", "Wisconsin diagnostic breast cancer"),
        DatasetSpec("ionosphere", 59, 59, "ionosphere", "g", "Johns Hopkins ionosphere radar"),
        DatasetSpec("sonar", 40, 40, "sonar", "Mine", "Sonar mines versus rocks"),
        DatasetSpec("spambase", 44, 44, "spambase", "1", "Spambase email classification"),
    )
}

DATASET_KEYS = tuple(SPECS)


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST.exists():
        return {}
    return json.loads(MANIFEST.read_text(encoding="utf-8")).get("datasets", {})


def raw_bytes(spec: DatasetSpec, *, allow_download: bool = True) -> bytes:
    """Cached ARFF bytes, verified against the committed digest."""

    manifest = _load_manifest()
    expected = manifest.get(spec.key, {}).get("sha256")
    if spec.path.exists():
        content = spec.path.read_bytes()
    else:
        if not allow_download:
            raise FileNotFoundError(
                f"{spec.path} is missing and downloading is disabled; run "
                "`python -m fr_gvi.experiments.datasets --fetch` once with network access"
            )
        with urllib.request.urlopen(spec.url, timeout=120) as handle:
            content = handle.read()
        spec.path.parent.mkdir(parents=True, exist_ok=True)
        spec.path.write_bytes(content)
    if expected is not None and _digest(content) != expected:
        raise ValueError(
            f"{spec.key}: cached bytes hash {_digest(content)} but the committed "
            f"manifest records {expected}; the upstream file has changed and the "
            "recorded results no longer describe it"
        )
    return content


_ATTRIBUTE = re.compile(r"^@attribute\s+(?:'([^']*)'|\"([^\"]*)\"|(\S+))\s+(.*)$", re.IGNORECASE)


def parse_arff(content: bytes) -> tuple[list[str], list[str], list[list[str]]]:
    """Attribute names, the last attribute's declaration, and the raw rows.

    A dependency-free reader is enough here: all five files are dense, comma
    separated, numeric except for a nominal class in the final column, and carry
    no missing values or quoted commas.  Anything outside that is rejected rather
    than guessed at.
    """

    names: list[str] = []
    declarations: list[str] = []
    rows: list[list[str]] = []
    in_data = False
    for raw_line in content.decode("utf-8", errors="strict").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        lowered = line.lower()
        if lowered.startswith("@data"):
            in_data = True
            continue
        if in_data:
            fields = [field.strip() for field in line.split(",")]
            if any(field == "?" for field in fields):
                raise ValueError("missing values are not supported by this reader")
            rows.append(fields)
            continue
        if lowered.startswith("@attribute"):
            match = _ATTRIBUTE.match(line)
            if match is None:
                raise ValueError(f"unparsed attribute declaration: {line}")
            names.append(match.group(1) or match.group(2) or match.group(3))
            declarations.append(match.group(4).strip())
    if not rows:
        raise ValueError("no data rows found")
    return names, declarations, rows


@dataclass(frozen=True)
class LoadedDataset:
    key: str
    features: FloatArray
    labels: FloatArray
    feature_names: list[str]
    dropped: list[str]
    sha256: str

    @property
    def observations(self) -> int:
        return int(self.features.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.features.shape[1])


def load_dataset(key: str, *, allow_download: bool = True) -> LoadedDataset:
    """Numeric design matrix and 0/1 labels, before standardization."""

    spec = SPECS[key]
    content = raw_bytes(spec, allow_download=allow_download)
    names, declarations, rows = parse_arff(content)
    table = np.asarray(rows, dtype=object)
    raw_labels = table[:, -1]
    positive = np.asarray(
        [str(value).strip().strip("'\"") == spec.positive_label for value in raw_labels],
        dtype=np.float64,
    )
    if not (0.0 < positive.mean() < 1.0):
        raise ValueError(
            f"{key}: the positive label {spec.positive_label!r} matched "
            f"{positive.sum():.0f} of {positive.size} rows; the declaration is "
            f"{declarations[-1]!r}"
        )
    features = np.asarray(table[:, :-1], dtype=np.float64)
    # A column with no variation contributes nothing to the likelihood, so the
    # posterior equals the prior along it.  Keeping it would leave the design
    # rank deficient and would put a coordinate in the comparison that no method
    # can move; ionosphere has exactly one such column.
    spread = features.std(axis=0)
    keep = spread > 0.0
    dropped = [name for name, retained in zip(names[:-1], keep) if not retained]
    return LoadedDataset(
        key,
        np.ascontiguousarray(features[:, keep]),
        positive,
        [name for name, retained in zip(names[:-1], keep) if retained],
        dropped,
        _digest(content),
    )


@dataclass(frozen=True)
class DatasetSplit:
    train_features: FloatArray
    train_labels: FloatArray
    test_features: FloatArray
    test_labels: FloatArray
    metadata: dict[str, Any]


def split_and_standardize(dataset: LoadedDataset) -> DatasetSplit:
    """Stratified split, then standardize with training statistics only.

    Standardizing before splitting would let the test rows influence the design
    the posterior is fitted on; the intercept is appended afterwards and left
    alone, since standardizing a constant column is not defined.
    """

    rng = np.random.default_rng(SPLIT_SEED)
    indices = np.arange(dataset.observations)
    test_mask = np.zeros(dataset.observations, dtype=bool)
    for value in (0.0, 1.0):
        members = indices[dataset.labels == value]
        permuted = rng.permutation(members)
        count = max(1, int(round(TEST_FRACTION * members.size)))
        test_mask[permuted[:count]] = True

    train = dataset.features[~test_mask]
    test = dataset.features[test_mask]
    location = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale <= 0.0] = 1.0

    def design(block: FloatArray) -> FloatArray:
        standardized = (block - location) / scale
        return np.ascontiguousarray(
            np.hstack([standardized, np.ones((block.shape[0], 1), dtype=np.float64)])
        )

    metadata = {
        "dataset": dataset.key,
        "sha256": dataset.sha256,
        "openml_id": SPECS[dataset.key].openml_id,
        "observations": dataset.observations,
        "train_observations": int((~test_mask).sum()),
        "test_observations": int(test_mask.sum()),
        # d counts the intercept, which is what every algorithm actually sees.
        "dimension": int(train.shape[1] + 1),
        "raw_features": int(dataset.features.shape[1]),
        "dropped_constant_columns": dataset.dropped,
        "positive_fraction": float(dataset.labels.mean()),
        "split_seed": SPLIT_SEED,
        "test_fraction": TEST_FRACTION,
    }
    return DatasetSplit(
        design(train), dataset.labels[~test_mask], design(test), dataset.labels[test_mask], metadata
    )


def prepared(key: str, *, allow_download: bool = True) -> DatasetSplit:
    return split_and_standardize(load_dataset(key, allow_download=allow_download))


def write_manifest() -> dict[str, Any]:
    """Refresh the committed digests and recorded shapes."""

    payload: dict[str, Any] = {}
    for key, spec in SPECS.items():
        dataset = load_dataset(key)
        split = split_and_standardize(dataset)
        payload[key] = {
            "openml_id": spec.openml_id,
            "file_id": spec.file_id,
            "url": spec.url,
            "note": spec.note,
            "positive_label": spec.positive_label,
            "sha256": dataset.sha256,
            **{
                field: split.metadata[field]
                for field in (
                    "observations",
                    "train_observations",
                    "test_observations",
                    "dimension",
                    "raw_features",
                    "dropped_constant_columns",
                    "positive_fraction",
                )
            },
        }
    document = {
        "schema_version": 1,
        "source": "OpenML",
        "split_seed": SPLIT_SEED,
        "test_fraction": TEST_FRACTION,
        "preprocessing": (
            "constant columns dropped; features standardized with training-split "
            "statistics; a constant intercept column appended after standardization"
        ),
        "datasets": payload,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def main(arguments: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch", action="store_true", help="download any missing dataset into the cache"
    )
    parser.add_argument(
        "--write-manifest", action="store_true", help="refresh the committed digests"
    )
    args = parser.parse_args(arguments)
    if args.write_manifest:
        document = write_manifest()
        for key, entry in sorted(document["datasets"].items()):
            print(
                f"{key:<11} n={entry['observations']:>5}  d={entry['dimension']:>3}  "
                f"train={entry['train_observations']:>5}  positive="
                f"{entry['positive_fraction']:.3f}  sha256={entry['sha256'][:12]}"
            )
        return 0
    for key in DATASET_KEYS:
        dataset = load_dataset(key, allow_download=args.fetch)
        split = split_and_standardize(dataset)
        print(f"{key:<11} {split.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
