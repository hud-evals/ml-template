from types import SimpleNamespace

from platform_runner import (
    build_modal_deploy_payload,
    build_submit_payload,
    modal_runtime_config,
    _print_run_rewards,
    _resolve_deploy_name,
)


def test_modal_runtime_config_requests_h100_gpu():
    config = modal_runtime_config()

    assert config["resources"]["gpu"] == {"type": "H100", "count": 1}
    assert config["limits"]["startup_timeout_s"] >= 600
    assert config["limits"]["run_timeout_s"] >= config["limits"]["startup_timeout_s"]


def test_deploy_payload_selects_modal_runtime_provider():
    payload = build_modal_deploy_payload(
        build_id="build-123",
        name="ml-template-1",
        registry_id="registry-123",
        no_cache=True,
    )

    assert payload["runtime_provider"] == "modal"
    assert payload["source"] == "direct"
    assert payload["registry_id"] == "registry-123"
    assert payload["no_cache"] is True


def test_submit_payload_includes_gpu_runtime_config():
    task = SimpleNamespace(env="ml-template-1", id="emb_debug_multi", args={"x": 1})
    runtime_config = modal_runtime_config(gpu_type="L4", gpu_count=2)

    payload = build_submit_payload(
        task=task,
        agent_spec={"provider": "claude", "model": "claude-opus-4-6"},
        job_id="11111111111111111111111111111111",
        trace_id="22222222222222222222222222222222",
        group_id="group-1",
        runtime_config=runtime_config,
    )

    assert payload["runtime_config"]["resources"]["gpu"] == {"type": "L4", "count": 2}
    assert payload["agent"]["provider"] == "claude"
    assert payload["trace_id"] == "22222222-2222-2222-2222-222222222222"
    assert payload["job_id"] == "11111111-1111-1111-1111-111111111111"
    assert payload["group_id"] == "group-1"


def test_resolve_deploy_name_uses_declared_environment_name():
    class Source:
        def served_environment_name(self):
            return "ml-template-1"

        def environment_name(self):
            return "directory-name"

    name = _resolve_deploy_name(Source(), {"registryName": ""}, explicit_name=None)

    assert name == "ml-template-1"


def test_print_run_rewards_repeats_names_for_group(capsys):
    selected = [("task_a", object()), ("task_b", object())]
    runs = [
        SimpleNamespace(reward=0.1),
        SimpleNamespace(reward=0.2),
        SimpleNamespace(reward=0.3),
        SimpleNamespace(reward=0.4),
    ]

    _print_run_rewards(selected, runs, group=2)

    assert capsys.readouterr().out.splitlines() == [
        "  task_a: 0.1",
        "  task_a: 0.2",
        "  task_b: 0.3",
        "  task_b: 0.4",
    ]
