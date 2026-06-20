"""Tests for Modal runner platform-dispatch helpers."""

from modal_runner import _task_from_runner_config


def test_task_from_runner_config_uses_task_row():
    task = _task_from_runner_config(
        {
            "task": {
                "env": "ml-template-1",
                "id": "repair_degraded_recipe",
                "args": {"prompt": "fix it", "graders": []},
                "slug": "emb_debug_multi",
            },
            # Fallback fields should not override the task row.
            "env_name": "wrong-env",
            "scenario": {"args": {"prompt": "wrong"}},
        }
    )

    assert task.env == "ml-template-1"
    assert task.id == "repair_degraded_recipe"
    assert task.args == {"prompt": "fix it", "graders": []}
    assert task.slug == "emb_debug_multi"
