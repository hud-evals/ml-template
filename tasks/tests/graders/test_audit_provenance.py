"""Tests for the check_audit_provenance grader."""

import hashlib
import json

from ..conftest import make_training_data, make_workspace, run_check


class TestAuditProvenance:
    def test_passes_for_clean_data(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        data_path = make_training_data(ws, n=20)
        original_hash = hashlib.md5(open(data_path, "rb").read()).hexdigest()
        # Modify the training data (agent cleaned it)
        with open(data_path, "a") as f:
            f.write(json.dumps({
                "instruction": "clean",
                "query": "extra query",
                "positive": "extra positive",
                "negatives": ["extra negative"],
            }) + "\n")
        info = {
            "type": "label_noise",
            "hash": original_hash,
            "train_file": "data/scifact.jsonl",
        }
        with open("/tmp/.grader_contamination_info.json", "w") as f:
            json.dump(info, f)

        assert run_check("check_audit_provenance", [ws]).returncode == 0

    def test_fails_for_contaminated_data(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        data_path = make_training_data(ws, n=20)
        info = {
            "type": "label_noise",
            "hash": hashlib.md5(open(data_path, "rb").read()).hexdigest(),
            "train_file": "data/scifact.jsonl",
        }
        with open("/tmp/.grader_contamination_info.json", "w") as f:
            json.dump(info, f)

        assert run_check("check_audit_provenance", [ws]).returncode == 1
