"""Unit tests for mutation helpers in tasks.mutations."""

import json
import os
import subprocess
import sys

import pytest

from .conftest import REPO_ROOT, make_training_data, make_workspace


class TestLabelNoise:
    def test_modifies_data(self, tmp_path):
        from tasks.mutations.data import inject_label_noise

        ws = make_workspace(str(tmp_path))
        data_path = make_training_data(ws, n=100)
        original = open(data_path).read()

        inject_label_noise(data_path, data_path, noise_rate=0.3)
        modified = open(data_path).read()
        assert original != modified, "Label noise should modify the data"


class TestDataLeakage:
    def test_adds_samples(self, tmp_path):
        from tasks.mutations.data import inject_data_leakage

        ws = make_workspace(str(tmp_path))
        train_path = make_training_data(ws, "train.jsonl", n=50)
        val_path = make_training_data(ws, "val.jsonl", n=20)

        original_count = sum(1 for _ in open(train_path))
        output_path = os.path.join(ws, "data", "combined.jsonl")
        inject_data_leakage(train_path, val_path, output_path, leak_rate=0.5)

        new_count = sum(1 for _ in open(output_path))
        assert new_count > original_count, "Leakage should add samples"


class TestDuplicates:
    def test_adds_copies(self, tmp_path):
        from tasks.mutations.data import inject_duplicates

        ws = make_workspace(str(tmp_path))
        data_path = make_training_data(ws, n=50)
        original_count = sum(1 for _ in open(data_path))

        inject_duplicates(data_path, data_path, dup_rate=0.3)
        new_count = sum(1 for _ in open(data_path))
        assert new_count > original_count, "Duplicates should increase sample count"


class TestContaminate:
    def test_label_noise(self, tmp_path):
        from tasks.mutations.data import contaminate

        ws = make_workspace(str(tmp_path))
        make_training_data(ws, n=50)
        info = contaminate(ws, "label_noise", noise_rate=0.3)
        assert info["type"] == "label_noise"
        assert info["corrupted_pairs"] > 0
        assert os.path.exists("/tmp/.grader_contamination_info.json")
        with open("/tmp/.grader_contamination_info.json") as f:
            saved = json.load(f)
        assert saved["type"] == "label_noise"

    def test_data_leakage(self, tmp_path):
        from tasks.mutations.data import contaminate

        ws = make_workspace(str(tmp_path))
        make_training_data(ws, n=50)
        make_training_data(ws, "val.jsonl", n=20)
        info = contaminate(ws, "data_leakage", leak_rate=0.5)
        assert info["type"] == "data_leakage"
        assert info["leaked_test"] > 0

    def test_duplicates(self, tmp_path):
        from tasks.mutations.data import contaminate

        ws = make_workspace(str(tmp_path))
        make_training_data(ws, n=50)
        info = contaminate(ws, "duplicates", noise_rate=0.3)
        assert info["type"] == "duplicates"
        assert info["total_output"] > info["original_count"]

    def test_unknown_mutation_raises(self, tmp_path):
        from tasks.mutations.data import contaminate

        ws = make_workspace(str(tmp_path))
        make_training_data(ws, n=10)
        with pytest.raises(ValueError, match="Unknown mutation"):
            contaminate(ws, "bogus")


class TestCLI:
    def test_label_noise_via_cli(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        make_training_data(ws, n=50)
        result = subprocess.run(
            [sys.executable, "-m", "tasks.mutations", "data", "label_noise", ws,
             "--noise-rate", "0.3"],
            capture_output=True, text=True, timeout=10,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists("/tmp/.grader_contamination_info.json")

    def test_data_leakage_via_cli(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        make_training_data(ws, n=50)
        make_training_data(ws, "val.jsonl", n=20)
        result = subprocess.run(
            [sys.executable, "-m", "tasks.mutations", "data", "data_leakage", ws,
             "--leak-rate", "0.5"],
            capture_output=True, text=True, timeout=10,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"


class TestEvalLeakage:
    def test_adds_train_rows_to_eval(self, tmp_path):
        from tasks.mutations.eval import inject_eval_leakage

        ws = make_workspace(str(tmp_path))
        train_path = make_training_data(ws, "scifact.jsonl", n=50)
        eval_path = make_training_data(ws, "val.jsonl", n=20)

        original_count = sum(1 for _ in open(eval_path))
        inject_eval_leakage(train_path, eval_path, eval_path, leak_rate=0.5)
        new_count = sum(1 for _ in open(eval_path))
        assert new_count > original_count, "Eval leakage should add rows to the visible eval set"


class TestContaminateEvalSignal:
    def test_eval_leakage(self, tmp_path):
        from tasks.mutations.eval import contaminate_eval_signal

        ws = make_workspace(str(tmp_path))
        make_training_data(ws, "scifact.jsonl", n=50)
        make_training_data(ws, "val.jsonl", n=20)

        info = contaminate_eval_signal(ws, "eval_leakage", leak_rate=0.5)
        assert info["type"] == "eval_leakage"
        assert info["leaked_train"] > 0
        assert os.path.exists("/tmp/.grader_eval_signal_info.json")

    def test_unknown_eval_mutation_raises(self, tmp_path):
        from tasks.mutations.eval import contaminate_eval_signal

        ws = make_workspace(str(tmp_path))
        make_training_data(ws, "scifact.jsonl", n=10)
        make_training_data(ws, "val.jsonl", n=10)
        with pytest.raises(ValueError, match="Unknown eval mutation"):
            contaminate_eval_signal(ws, "bogus")


class TestEvalMutationCLI:
    def test_eval_leakage_via_cli(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        make_training_data(ws, "scifact.jsonl", n=50)
        make_training_data(ws, "val.jsonl", n=20)
        result = subprocess.run(
            [sys.executable, "-m", "tasks.mutations", "eval", "eval_leakage", ws,
             "--leak-rate", "0.5"],
            capture_output=True, text=True, timeout=10,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists("/tmp/.grader_eval_signal_info.json")
