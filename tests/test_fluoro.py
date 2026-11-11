import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fluoro import AnalogModel, Measurement, benchmark, load_measurements, split_by_structure

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "synthetic.csv"


class FluoroTests(unittest.TestCase):
    def test_duplicate_pairs_use_median_and_keep_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measurements.csv"
            path.write_text(
                "Chromophore;Solvent;Emission max (nm);DOI\n"
                "CCO;Ethanol;500;a\nCCO; ethanol ;540;b\nCCO;water;;c\n"
                "CCO;water;inf;d\n",
                encoding="utf-8",
            )
            rows, counts = load_measurements(path)
        self.assertEqual(rows, [Measurement("CCO", "ethanol", 520, "a; b", 2)])
        self.assertEqual(counts["missing_rows"], 1)
        self.assertEqual(counts["invalid_rows"], 1)

    def test_split_keeps_all_solvents_of_a_structure_together(self):
        rows, _ = load_measurements(EXAMPLE)
        train, test = split_by_structure(rows)
        self.assertEqual((len(train), len(test)), (16, 4))
        self.assertFalse({item.smiles for item in train} & {item.smiles for item in test})
        _, reversed_test = split_by_structure(list(reversed(rows)))
        self.assertEqual({item.smiles for item in test}, {item.smiles for item in reversed_test})

    def test_same_solvent_is_selected_and_analogs_are_distinct(self):
        rows = [Measurement("Cc1ccccc1", "water", 500),
                Measurement("Cc1ccccc1", "ethanol", 520),
                Measurement("c1ccccc1", "water", 450)]
        result = AnalogModel(rows).predict("Cc1ccccc1", " Ethanol ")
        self.assertEqual(result["neighbors"][0]["measurement"]["emission_nm"], 520)
        self.assertEqual(result["neighbors"][0]["similarity"], 1)
        self.assertEqual(len(result["neighbors"]), 2)

    def test_dissimilar_smiles_set_warning(self):
        result = AnalogModel([Measurement("CCCCCC", "water", 500)]).predict("[Xe]", "water")
        self.assertTrue(result["low_similarity"])

    def test_benchmark_errors_match_per_row_results(self):
        rows, _ = load_measurements(EXAMPLE)
        report, predictions = benchmark(rows, 42)
        mae = sum(abs(row["error_nm"]) for row in predictions) / len(predictions)
        self.assertAlmostEqual(report["solvent_aware_knn"]["mae_nm"], mae)

    def test_leave_one_out_excludes_all_measurements_of_the_held_out_structure(self):
        rows = [
            Measurement("CCO", "water", 100),
            Measurement("CCO", "ethanol", 200),
            Measurement("CCC", "water", 500),
        ]
        report, predictions = benchmark(rows, 42, leave_one_out=True)
        self.assertEqual(report["folds"], 2)
        self.assertEqual(report["test_pairs"], 3)
        for row in predictions:
            self.assertEqual(row["baseline_nm"], 500 if row["smiles"] == "CCO" else 150)

    def test_benchmark_cli_writes_csv_and_protects_input(self):
        command = [sys.executable, str(ROOT / "fluoro.py"), "benchmark", str(EXAMPLE)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.csv"
            result = subprocess.run([*command, "--predictions", str(output)],
                                    capture_output=True, text=True, check=True)
            with output.open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)
        self.assertEqual(json.loads(result.stdout)["test_pairs"], 4)
        result = subprocess.run([*command, "--predictions", str(EXAMPLE)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("must differ", result.stderr)


if __name__ == "__main__":
    unittest.main()
