import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ontogenia.human_alignment import run_human_alignment


class TestHumanAlignment(unittest.TestCase):
    def test_run_human_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = pd.DataFrame(
                [
                    {"model_size": "160m", "task": "t1", "training_step": 0, "acc,none": 0.2},
                    {"model_size": "160m", "task": "t1", "training_step": 1, "acc,none": 0.95},
                    {"model_size": "160m", "task": "t1", "training_step": 2, "acc,none": 0.96},
                    {"model_size": "160m", "task": "t1", "training_step": 3, "acc,none": 0.97},
                    {"model_size": "160m", "task": "t2", "training_step": 0, "acc,none": 0.1},
                    {"model_size": "160m", "task": "t2", "training_step": 1, "acc,none": 0.2},
                    {"model_size": "160m", "task": "t2", "training_step": 2, "acc,none": 0.9},
                    {"model_size": "160m", "task": "t2", "training_step": 3, "acc,none": 0.91},
                    {"model_size": "160m", "task": "t2", "training_step": 4, "acc,none": 0.92},
                ]
            )
            parquet_path = root / "metrics.parquet"
            metrics.to_parquet(parquet_path, index=False)
            aoa = pd.DataFrame([{"task": "t1", "aoa_months": 20}, {"task": "t2", "aoa_months": 30}])
            aoa_path = root / "aoa.csv"
            aoa.to_csv(aoa_path, index=False)

            result = run_human_alignment(
                metrics_parquet=parquet_path,
                aoa_csv=aoa_path,
                metric_col="acc,none",
                target_model_size="160m",
                n_boot=100,
            )
            self.assertEqual(len(result.overlap), 2)
            self.assertIn("rho", result.stats.columns)


if __name__ == "__main__":
    unittest.main()

