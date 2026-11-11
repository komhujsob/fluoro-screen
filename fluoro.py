"""A reproducible nearest-neighbor baseline for fluorescence emission."""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

FEATURE_BITS = 2048
MIN_SIMILARITY = 0.35
NEIGHBORS = 5
MISSING = {"", "-", "na", "n/a", "nan", "none", "null"}
SMILES_TOKEN = re.compile(r"\[[^\]]+\]|Br|Cl|Si|Se|Na|Li|Al|%\d{2}|[A-Za-z0-9()=#+\-@/\\.*:$]")


@dataclass(frozen=True)
class Measurement:
    smiles: str
    solvent: str
    emission_nm: float
    source: str = ""
    replicates: int = 1


@dataclass(frozen=True)
class Neighbor:
    measurement: Measurement
    similarity: float


def tokens(smiles: str) -> tuple[str, ...]:
    parts = tuple(SMILES_TOKEN.findall(smiles))
    if not parts or "".join(parts) != smiles:
        raise ValueError(f"Invalid SMILES text: {smiles!r}")
    return parts


@lru_cache(maxsize=16384)
def fingerprint(smiles: str) -> int:
    parts = ("^", *tokens(smiles), "$")
    bits = 0
    for width in (1, 2, 3, 4):
        for start in range(len(parts) - width + 1):
            fragment = "|".join(parts[start:start + width]).encode()
            position = int.from_bytes(hashlib.blake2b(fragment, digest_size=2).digest())
            bits |= 1 << (position % FEATURE_BITS)
    return bits


def solvent_key(solvent: str) -> str:
    return " ".join(solvent.casefold().split())


def header_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def load_measurements(path: Path) -> tuple[list[Measurement], dict[str, int]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(4096)
        stream.seek(0)
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        delimiter = "\t" if "\t" in first_line else ";" if first_line.count(";") > first_line.count(",") else ","
        reader = csv.DictReader(stream, delimiter=delimiter)
        columns = {header_key(name): name for name in reader.fieldnames or ()}

        def column(*aliases: str) -> str | None:
            return next((columns[alias] for alias in aliases if alias in columns), None)

        smiles_col = column("smiles", "chromophoresmiles", "canonicalsmiles", "chromophore")
        solvent_col = column("solvent", "solventname", "medium", "solventsmiles")
        emission_col = column(
            "emissionnm", "emissionmaxnm", "emissionmaximum", "emissionmaximumnm",
            "emissionmax", "lambdamaxemnm", "fluorescencemaxnm", "emissionwavelengthnm",
        )
        source_col = column("source", "doi", "reference", "ref")
        if not (smiles_col and solvent_col and emission_col):
            raise ValueError(
                "CSV must contain smiles, solvent and emission_nm "
                f"(or recognized aliases). Found: {', '.join(reader.fieldnames or ())}"
            )

        groups: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(list)
        counts: Counter[str] = Counter()
        for row in reader:
            counts["input_rows"] += 1
            smiles = (row.get(smiles_col) or "").strip()
            solvent = solvent_key(row.get(solvent_col) or "")
            emission = (row.get(emission_col) or "").strip()
            source = (row.get(source_col) or "").strip() if source_col else ""
            if {smiles.casefold(), solvent, emission.casefold()} & MISSING:
                counts["missing_rows"] += 1
                continue
            try:
                tokens(smiles)
                value = float(emission)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("Expected a finite positive wavelength")
            except ValueError:
                counts["invalid_rows"] += 1
                continue
            groups[smiles, solvent].append((value, source))

    measurements = [
        Measurement(smiles, solvent, statistics.median(value for value, _ in entries),
                    "; ".join(sorted({source for _, source in entries if source})), len(entries))
        for (smiles, solvent), entries in sorted(groups.items())
    ]
    if not measurements:
        raise ValueError("CSV contains no usable measurements")
    counts["usable_pairs"] = len(measurements)
    counts["collapsed_rows"] = sum(item.replicates - 1 for item in measurements)
    return measurements, dict(counts)


class AnalogModel:
    def __init__(self, measurements: list[Measurement]) -> None:
        if not measurements:
            raise ValueError("Training set is empty")
        self.median = statistics.median(item.emission_nm for item in measurements)
        by_structure: dict[str, list[Measurement]] = defaultdict(list)
        for item in measurements:
            by_structure[item.smiles].append(item)
        self.index = [
            (fingerprint(smiles), items, {item.solvent: item for item in items})
            for smiles, items in by_structure.items()
        ]

    def neighbors(self, smiles: str, solvent: str) -> list[Neighbor]:
        query = fingerprint(smiles)
        query_size = query.bit_count()
        solvent = solvent_key(solvent)
        candidates = []
        for position, (pattern, measurements, media) in enumerate(self.index):
            shared = (query & pattern).bit_count()
            if not shared:
                continue
            similarity = shared / (query_size + pattern.bit_count() - shared)
            same_solvent = media.get(solvent)
            candidates.append((similarity + 0.08 * bool(same_solvent), similarity, -position,
                               same_solvent, measurements))

        neighbors = []
        for _, similarity, _, exact, measurements in heapq.nlargest(NEIGHBORS, candidates):
            if exact is None:
                center = statistics.median(item.emission_nm for item in measurements)
                exact = min(measurements, key=lambda item: abs(item.emission_nm - center))
            neighbors.append(Neighbor(exact, similarity))
        return neighbors

    def predict(self, smiles: str, solvent: str) -> dict:
        neighbors = self.neighbors(smiles, solvent)
        if not neighbors:
            return {"prediction_nm": self.median, "low_similarity": True, "neighbors": []}
        weights = [item.similarity ** 2 for item in neighbors]
        value = sum(item.measurement.emission_nm * weight
                    for item, weight in zip(neighbors, weights, strict=True)) / sum(weights)
        return {
            "prediction_nm": value,
            "low_similarity": max(item.similarity for item in neighbors) < MIN_SIMILARITY,
            "neighbors": [asdict(item) for item in neighbors],
        }


def split_by_structure(measurements: list[Measurement], seed: int = 42
                       ) -> tuple[list[Measurement], list[Measurement]]:
    structures = {item.smiles for item in measurements}
    if len(structures) < 5:
        raise ValueError("Benchmark requires at least 5 distinct canonical SMILES")
    ordered = sorted(structures, key=lambda value: (
        hashlib.sha256(f"{seed}:{value}".encode()).digest(), value))
    test_structures = set(ordered[: max(1, round(len(ordered) * 0.20))])
    train = [item for item in measurements if item.smiles not in test_structures]
    test = [item for item in measurements if item.smiles in test_structures]
    return train, test


def error_metrics(errors: list[float]) -> dict[str, float] | None:
    if not errors:
        return None
    absolute = [abs(error) for error in errors]
    return {
        "mae_nm": statistics.mean(absolute),
        "median_absolute_error_nm": statistics.median(absolute),
        "rmse_nm": math.sqrt(statistics.mean(error ** 2 for error in errors)),
    }


def benchmark(measurements: list[Measurement], seed: int, leave_one_out: bool = False
              ) -> tuple[dict, list[dict]]:
    by_structure: dict[str, list[Measurement]] = defaultdict(list)
    for item in measurements:
        by_structure[item.smiles].append(item)
    if leave_one_out:
        if len(by_structure) < 2:
            raise ValueError("Leave-one-out requires at least 2 distinct canonical SMILES")
        folds = (([item for item in measurements if item.smiles != smiles], items)
                 for smiles, items in by_structure.items())
    else:
        folds = iter([split_by_structure(measurements, seed)])

    rows = []
    for train, test in folds:
        model = AnalogModel(train)
        for item in test:
            result = model.predict(item.smiles, item.solvent)
            rows.append({
                **asdict(item),
                "prediction_nm": result["prediction_nm"],
                "error_nm": result["prediction_nm"] - item.emission_nm,
                "baseline_nm": model.median,
                "low_similarity": result["low_similarity"],
            })
    partition = {
        "split": "leave-one-canonical-SMILES-out",
        "folds": len(by_structure),
    } if leave_one_out else {
        "split": "20% held-out canonical SMILES",
        "seed": seed,
        "train_pairs": len(train),
        "train_structures": len({item.smiles for item in train}),
    }
    report = {
        **partition,
        "test_pairs": len(rows),
        "test_structures": len({row["smiles"] for row in rows}),
        "training_median": error_metrics([row["baseline_nm"] - row["emission_nm"] for row in rows]),
        "solvent_aware_knn": error_metrics([row["error_nm"] for row in rows]),
        "low_similarity_pairs": sum(row["low_similarity"] for row in rows),
        "higher_similarity_subset": error_metrics(
            [row["error_nm"] for row in rows if not row["low_similarity"]]),
    }
    return report, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    inspect = subcommands.add_parser("inspect", help="Summarize CSV cleaning")
    evaluate = subcommands.add_parser("benchmark", help="Evaluate on held-out structures")
    predict = subcommands.add_parser("predict", help="Predict one compound and list analogs")
    for command in (inspect, evaluate, predict):
        command.add_argument("csv", type=Path)
    evaluate.add_argument("--seed", default=42, type=int)
    evaluate.add_argument("--leave-one-out", action="store_true",
                          help="Hold out each structure once (for small datasets)")
    evaluate.add_argument("--predictions", type=Path, help="Optional per-row output CSV")
    predict.add_argument("--smiles", required=True)
    predict.add_argument("--solvent", required=True)
    args = parser.parse_args(argv)
    try:
        measurements, cleaning = load_measurements(args.csv)
        if args.command == "inspect":
            result = {"cleaning": cleaning, "structures": len({item.smiles for item in measurements}),
                      "solvents": len({item.solvent for item in measurements})}
        elif args.command == "benchmark":
            if args.predictions and args.predictions.resolve() == args.csv.resolve():
                raise ValueError("Prediction output must differ from the input file")
            report, rows = benchmark(measurements, args.seed, args.leave_one_out)
            result = {"cleaning": cleaning, **report}
            if args.predictions:
                with args.predictions.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        else:
            if not solvent_key(args.solvent):
                raise ValueError("Solvent cannot be empty")
            result = AnalogModel(measurements).predict(args.smiles.strip(), args.solvent)
    except (OSError, ValueError, UnicodeError, csv.Error) as error:
        print(f"fluoro: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
