"""Tests for the generic check_threshold grader."""

import json
import os

from ..conftest import make_workspace, run_check


class TestThresholdChecks:
    def test_ndcg_above_threshold(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        with open(os.path.join(ws, ".emb_eval.json"), "w") as f:
            json.dump({"ndcg@10": 0.45, "best_dir": "/fake"}, f)
        assert run_check("check_threshold", [".emb_eval.json", "ndcg@10", "0.30", ws]).returncode == 0

    def test_ndcg_below_threshold(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        with open(os.path.join(ws, ".emb_eval.json"), "w") as f:
            json.dump({"ndcg@10": 0.05, "best_dir": "/fake"}, f)
        assert run_check("check_threshold", [".emb_eval.json", "ndcg@10", "0.30", ws]).returncode == 1

    def test_pass_rate_above_threshold(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        with open(os.path.join(ws, ".moe_eval.json"), "w") as f:
            json.dump({"pass_rate": 0.8}, f)
        assert run_check("check_threshold", [".moe_eval.json", "pass_rate", "0.67", ws]).returncode == 0

    def test_pass_rate_below_threshold(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        with open(os.path.join(ws, ".moe_eval.json"), "w") as f:
            json.dump({"pass_rate": 0.2}, f)
        assert run_check("check_threshold", [".moe_eval.json", "pass_rate", "0.67", ws]).returncode == 1

    def test_lower_is_better_pass(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        with open(os.path.join(ws, ".vlm_eval.json"), "w") as f:
            json.dump({"val_loss": 1.5, "output_dir": "/fake"}, f)
        assert run_check("check_threshold", [".vlm_eval.json", "val_loss", "2.0", ws, "lower"]).returncode == 0

    def test_lower_is_better_fail(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        with open(os.path.join(ws, ".vlm_eval.json"), "w") as f:
            json.dump({"val_loss": 3.0, "output_dir": "/fake"}, f)
        assert run_check("check_threshold", [".vlm_eval.json", "val_loss", "2.0", ws, "lower"]).returncode == 1
