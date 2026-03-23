"""Tests for the check_data_cleaned grader."""

import hashlib
import json
import os
import shutil

from ..conftest import make_training_data, make_workspace, run_check


class TestDataCleaned:
    def test_fails_when_data_unchanged(self, tmp_path):
        from tasks.mutations.data import inject_label_noise

        ws = make_workspace(str(tmp_path))
        data_path = make_training_data(ws, n=50)
        info = inject_label_noise(data_path, data_path, noise_rate=0.2)
        info.update({
            "type": "label_noise",
            "hash": hashlib.md5(open(data_path, "rb").read()).hexdigest(),
            "line_count": 50,
        })
        with open("/tmp/.grader_contamination_info.json", "w") as f:
            json.dump(info, f)

        r = run_check("check_data_cleaned", [ws])
        assert r.returncode == 1, f"Should fail when data unchanged: {r.stdout}"

    def test_passes_when_data_modified(self, tmp_path):
        from tasks.mutations.data import inject_label_noise

        ws = make_workspace(str(tmp_path))
        data_path = make_training_data(ws, n=50)
        clean_copy = os.path.join(ws, "data", "clean.jsonl")
        shutil.copy2(data_path, clean_copy)
        info = inject_label_noise(data_path, data_path, noise_rate=0.2)
        shutil.copy2(clean_copy, data_path)
        info.update({
            "type": "label_noise",
            "hash": hashlib.md5(open(clean_copy, "rb").read()).hexdigest(),
            "line_count": 50,
        })
        with open("/tmp/.grader_contamination_info.json", "w") as f:
            json.dump(info, f)

        r = run_check("check_data_cleaned", [ws])
        assert r.returncode == 0, f"Should pass when data modified: {r.stdout}"
