import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from albench.al.report_io import csv_columns, load_results


class ReportSchemaTests(unittest.TestCase):
    def test_hash_columns_name_current_query_and_next_labeled_sets(self):
        columns = csv_columns(["A"])
        self.assertEqual(
            columns[-4:],
            ["labeled_sha", "query_sha", "next_labeled_sha", "train_s"])

    def test_reads_rows_without_hash_and_timing_columns(self):
        header = [
            "strategy", "seed", "round", "n_labeled", "frac",
            "val_mAP50", "test_mAP50", "test_mAP50_95", "AP_A", "nlab_A",
        ]
        row = ["random", 13, 0, 5, .05, .1, .2, .15, .3, 7]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerow(row)
            record = load_results(str(path)).by[("random", 13)][0]
        self.assertEqual(record["nlab"], [7])
        self.assertEqual(record["labeled_sha"], "")


if __name__ == "__main__":
    unittest.main()
