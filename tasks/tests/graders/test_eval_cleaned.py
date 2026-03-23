"""Tests for the check_eval_cleaned grader."""

from ..conftest import make_training_data, make_workspace, run_check


class TestEvalCleaned:
    @staticmethod
    def _write_distinct_eval_data(path: str, n: int = 20) -> None:
        import json

        with open(path, "w") as f:
            for i in range(n):
                pair = {
                    "instruction": "eval",
                    "query": f"eval query {i}",
                    "positive": f"eval positive {i}",
                    "negatives": [f"eval negative {i}-{j}" for j in range(3)],
                }
                f.write(json.dumps(pair) + "\n")

    def test_fails_when_eval_unchanged(self, tmp_path):
        from tasks.mutations.eval import contaminate_eval_signal

        ws = make_workspace(str(tmp_path))
        make_training_data(ws, "scifact.jsonl", n=50)
        val_path = make_training_data(ws, "val.jsonl", n=20)
        self._write_distinct_eval_data(val_path, n=20)
        contaminate_eval_signal(ws, "eval_leakage", leak_rate=0.5)

        r = run_check("check_eval_cleaned", [ws])
        assert r.returncode == 1, f"Should fail when eval data unchanged: {r.stdout}"

    def test_passes_when_eval_restored(self, tmp_path):
        import shutil

        from tasks.mutations.eval import contaminate_eval_signal

        ws = make_workspace(str(tmp_path))
        make_training_data(ws, "scifact.jsonl", n=50)
        val_path = make_training_data(ws, "val.jsonl", n=20)
        self._write_distinct_eval_data(val_path, n=20)
        clean_copy = str(tmp_path / "clean_val.jsonl")
        shutil.copy2(val_path, clean_copy)

        contaminate_eval_signal(ws, "eval_leakage", leak_rate=0.5)
        shutil.copy2(clean_copy, val_path)

        r = run_check("check_eval_cleaned", [ws])
        assert r.returncode == 0, f"Should pass when eval data restored: {r.stdout}"
