import json
import tempfile
import unittest
from pathlib import Path

from ontogenia.aggregate import aggregate_results_dir, collect_metrics_rows


class TestAggregate(unittest.TestCase):
    def test_collect_metrics_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sweep = root / "sweep"
            sweep.mkdir()
            payload = {
                "meta": {
                    "model_size": "14m",
                    "pretrained": "EleutherAI/pythia-14m-deduped",
                    "revision": "step0",
                    "tasks": ["blimp_foo"],
                },
                "lm_eval": {
                    "results": {
                        "blimp_foo": {"acc,none": 0.5, "acc_stderr,none": 0.1, "alias": "x"},
                    }
                },
            }
            (sweep / "14m_step0.json").write_text(json.dumps(payload), encoding="utf-8")
            rows = collect_metrics_rows([sweep / "14m_step0.json"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["training_step"], 0)
            self.assertEqual(rows[0]["task"], "blimp_foo")
            self.assertEqual(rows[0]["acc,none"], 0.5)

    def test_aggregate_results_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sweep").mkdir()
            meta = {"model_size": "14m", "revision": "step1", "pretrained": "x", "tasks": ["t"]}
            data = {
                "meta": meta,
                "lm_eval": {"results": {"t1": {"acc,none": 1.0}}},
            }
            (root / "sweep" / "14m_step1.json").write_text(json.dumps(data), encoding="utf-8")
            out = root / "out.parquet"
            aggregate_results_dir(root, out_parquet=out, include_smoke=False, include_sweep=True)
            self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()
