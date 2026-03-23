"""Test setup_fixtures actually run on GPU."""

import pytest


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
        env._setup_workspace()
        from tasks.utils.setup_fixtures import setup_pipeline_ablation
        setup_pipeline_ablation(env._workspace)

    def test_data_influence(self):
        import env
        env._setup_workspace()
        from tasks.utils.setup_fixtures import setup_data_influence
        setup_data_influence(env._workspace)
