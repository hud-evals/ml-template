"""Tests for setup_fixtures."""

import json

import pytest


def test_build_synthetic_dataset_passes_num_negatives(tmp_path, monkeypatch):
    from tasks.utils import build_datasets, setup_fixtures

    call: dict = {}

    def fake_generate_synthetic(**kwargs):
        call.update(kwargs)
        return [{
            "instruction": "test",
            "query": "query",
            "positive": "positive",
            "negatives": ["negative"],
        }]

    monkeypatch.setattr(build_datasets, "generate_synthetic", fake_generate_synthetic)

    setup_fixtures.build_dataset(str(tmp_path), "synthetic")

    assert call == {"corpus_dataset": "scifact", "max_samples": 500, "num_negatives": 7}
    rows = [
        json.loads(line)
        for line in (tmp_path / "data" / "synthetic.jsonl").read_text().splitlines()
    ]
    assert rows == [{
        "instruction": "test",
        "query": "query",
        "positive": "positive",
        "negatives": ["negative"],
    }]


def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


gpu = pytest.mark.skipif(not _has_gpu(), reason="No GPU available")


@gpu
class TestSetupFixtures:
    def test_pipeline_ablation(self):
        import env
        from tasks.emb_pipe_ablation.setup import main as setup_pipeline_ablation

        env._setup_workspace()
        setup_pipeline_ablation(env.WORKSPACE)

    def test_data_influence(self):
        import env
        from tasks.emb_data_influence.setup import main as setup_data_influence

        env._setup_workspace()
        setup_data_influence(env.WORKSPACE)
