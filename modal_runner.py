"""Modal runner for agent tasks and GPU tests (HUD v6).

Deploy once, then spawn isolated agent runs:

    modal deploy modal_runner.py

    # Single task
    modal run modal_runner.py --task moe_debug_balance

    # Multiple tasks in parallel (each gets its own container)
    modal run modal_runner.py --tasks moe_debug_balance,emb_debug_multi

    # All tasks, 3 repeats each
    modal run modal_runner.py --all --repeats 3 --model claude-sonnet-4-6 --max-steps 100

    # Run GPU tests
    modal run modal_runner.py --test
    modal run modal_runner.py --test --test-filter emb

Setup:
    pip install modal
    modal setup
    modal secret create hud-keys HUD_API_KEY=<key>

Each container serves the v6 Environment in-process with ``LocalRuntime`` (the
same ``python -m hud.environment.server`` entry point the image CMD runs) and
drives it with a gateway-routed agent. The env's ``env.workspace(...)``
capability gives the agent a bwrap-isolated SSH shell with the H100 exposed.
"""

import contextlib
import os
import pathlib

import modal

APP_NAME = "ml-training-dev"
app = modal.App(APP_NAME)

image = (
    modal.Image.from_dockerfile(
        "Dockerfile.hud",
        build_args={"HUD_RUNTIME": "0"},
    )
    # anthropic ships as a core dependency of hud-python v6, so the image's
    # `uv sync` installs it; this is a defensive no-op if resolution skipped it.
    .run_commands("uv pip install --python /opt/venv/bin/python 'anthropic>=0.40'")
    .add_local_dir("tasks", remote_path="/mcp_server/tasks")
)

SRC = "/mcp_server"
WS = "/home/ubuntu/workspace"
TASK_DIR = pathlib.Path(__file__).parent / "tasks"


def _provider_for(model: str) -> str | None:
    """Map a concrete model id to its HUD gateway provider (an ``AgentType``).

    Returns ``None`` if the provider can't be inferred, in which case the caller
    falls back to ``create_agent``'s gateway ``/models`` lookup.
    """
    m = model.lower()
    if m.startswith(("claude", "anthropic")):
        return "claude"
    if m.startswith(("gpt", "o1", "o3", "o4", "openai")):
        return "openai"
    if m.startswith(("gemini", "google")):
        return "gemini"
    return None


def _make_agent(model: str, system_prompt: str, max_steps: int):
    """Build a gateway-routed agent for ``model``.

    Constructs the agent directly from its ``AgentType`` when the provider is
    known, which routes inference for the exact ``model`` id while skipping the
    platform-scoped gateway ``/models`` lookup. Falls back to ``create_agent``
    (registry resolution) for providers we can't infer.
    """
    from hud.agents import create_agent

    provider = _provider_for(model)
    if provider is None:
        return create_agent(model, system_prompt=system_prompt, max_steps=max_steps)

    from hud.types import AgentType
    from hud.utils.gateway import build_gateway_client

    agent_type = AgentType(provider)
    config = agent_type.config_cls(
        model=model,
        system_prompt=system_prompt,
        max_steps=max_steps,
        model_client=build_gateway_client(agent_type.gateway_provider),
    )
    return agent_type.cls(config)


def _available_tasks() -> list[str]:
    return sorted(
        d.name for d in TASK_DIR.iterdir()
        if d.is_dir() and (d / "task.py").exists()
    )


def _deployed_function(name: str):
    return modal.Function.from_name(APP_NAME, name)


def _load_task(task_name: str):
    """Import tasks/<task_name>/task.py and return its built v6 Task."""
    import importlib
    import sys

    sys.path.insert(0, str(pathlib.Path(SRC) / "tasks" / task_name))
    sys.modules.pop("task", None)
    try:
        return importlib.import_module("task").task
    finally:
        sys.path.pop(0)


# This HUD account lives on the production control plane, but the v6 SDK
# defaults to the beta endpoints (api/inference/telemetry.beta.hud.ai), which
# reject a production key with 401. Point every HUD URL at production; these are
# inherited by the LocalRuntime env-server child. ``setdefault`` lets the Modal
# secret / environment override them if ever needed.
HUD_PROD_URLS = {
    "HUD_API_URL": "https://api.hud.ai",
    "HUD_GATEWAY_URL": "https://inference.hud.ai",
    "HUD_TELEMETRY_URL": "https://telemetry.hud.ai/v3/api",
    "HUD_WEB_URL": "https://hud.ai",
}


@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    secrets=[modal.Secret.from_name(os.environ.get("HUD_KEYS_SECRET", "hud-keys"), required_keys=["HUD_API_KEY"])],
)
async def run_agent(
    task_name: str,
    model: str = "claude-opus-4-6",
    max_steps: int = 500,
) -> float:
    """Run an agent against a task in an isolated container; return the reward."""
    os.environ.setdefault("MCP_TESTING_MODE", "1")
    # Must run before the first ``hud`` import so the settings singleton reads
    # the production endpoints.
    for key, value in HUD_PROD_URLS.items():
        os.environ.setdefault(key, value)

    from hud.eval.runtime import LocalRuntime

    from env import AGENT_CONFIG

    task = _load_task(task_name)
    task.slug = task.slug or task_name
    # Surfaced on the trace as metadata (does not configure the agent).
    task.agent_config = AGENT_CONFIG

    agent = _make_agent(
        model,
        system_prompt=AGENT_CONFIG["system_prompt"],
        max_steps=max_steps,
    )

    print(f"=== {task_name} ({model}) ===")
    job = await task.run(agent, runtime=LocalRuntime(f"{SRC}/env.py"))
    reward = job.reward
    print(f"  Reward: {reward:.3f}" if reward is not None else "  Reward: n/a")
    if job.id:
        print(f"  Job: https://hud.ai/jobs/{job.id}")
    return reward


@contextlib.contextmanager
def _scoped_v6_settings(*, api_url: str, api_key: str, gateway_url: str, telemetry_url: str):
    """Bind hud-python's settings singleton to a platform for one rollout.

    The Modal analog of hud-daemon's ``runner_v6._scoped_v6_settings``: the
    rollout atom's reporting posts to ``hud_api_url``, spans export to
    ``hud_telemetry_url``, and gateway agents build clients against
    ``hud_gateway_url`` with ``api_key``. All must point at the dispatching
    control plane rather than the SDK / prod defaults.
    """
    from hud.settings import settings as sdk_settings

    saved = (
        sdk_settings.api_key,
        sdk_settings.hud_api_url,
        sdk_settings.hud_gateway_url,
        sdk_settings.hud_telemetry_url,
        sdk_settings.telemetry_enabled,
    )
    sdk_settings.api_key = api_key
    # PlatformClient prepends its own /v2, so this is the bare origin.
    sdk_settings.hud_api_url = api_url.rstrip("/")
    sdk_settings.hud_gateway_url = gateway_url.rstrip("/")
    sdk_settings.hud_telemetry_url = telemetry_url.rstrip("/")
    sdk_settings.telemetry_enabled = True
    try:
        yield
    finally:
        (
            sdk_settings.api_key,
            sdk_settings.hud_api_url,
            sdk_settings.hud_gateway_url,
            sdk_settings.hud_telemetry_url,
            sdk_settings.telemetry_enabled,
        ) = saved


def _build_agent_from_spec(agent_spec: dict):
    """Rebuild the gateway agent the platform submission serialized.

    Mirrors ``runner_v6._build_agent``: ``agent_spec`` is the SDK's
    ``ToolAgent.hosted_spec`` (a ``type`` from the v6 ``AgentType`` vocabulary
    plus a serialized ``AgentConfig``). The provider client resolves through the
    HUD gateway via the settings bound by :func:`_scoped_v6_settings`.
    """
    from hud.types import AgentType

    raw_type = (agent_spec.get("type") or "").strip().lower()
    try:
        agent_type = AgentType(raw_type)
    except ValueError:
        raise RuntimeError(
            f"Unsupported v6 agent type: {raw_type!r}. "
            f"Expected one of: {', '.join(at.value for at in AgentType)}."
        ) from None
    config = agent_type.config_cls.model_validate(agent_spec.get("config") or {})
    return agent_type.cls(config)


async def _run_dispatched_rollout(
    *,
    runner_config: dict,
    api_url: str,
    api_key: str,
    gateway_url: str,
    telemetry_url: str,
) -> dict:
    """Drive one platform-dispatched v6 rollout, serving the env locally.

    The Modal-side twin of ``hud-daemon.runner_v6.run_rollout_v6``: same
    runner_config contract and same trace-exit result shape, but the env is
    served in-process with ``LocalRuntime`` (this container *is* the env image)
    instead of attaching to a separately-booted container channel. Factored out
    of the Modal entrypoint so it can be unit-tested without Modal. Like the
    atom, it does not raise for in-run failures — the error travels on the
    result.
    """
    from hud.eval.run import rollout
    from hud.eval.runtime import LocalRuntime
    from hud.eval.task import Task
    from hud.telemetry.exporter import flush as flush_telemetry

    task_block = runner_config.get("task") or {}
    task_id = task_block.get("id")
    env_name = runner_config.get("env_name")
    if not task_id or not env_name:
        raise RuntimeError(
            "Runner config has no v6 task reference (task.id + env_name) — "
            "this trace was not created by a v6 submission."
        )
    job_id = runner_config.get("job_id")
    if not job_id:
        raise RuntimeError("Runner config has no job_id")
    group_id = runner_config.get("group_id")

    scenario_info = runner_config.get("scenario") or {}
    args = scenario_info.get("args") or {}
    agent_spec = runner_config.get("agent_config")
    if not agent_spec:
        raise RuntimeError(
            "Runner config has no agent_config — this trace was not created by a "
            "v6 submission (hosted agent spec missing)."
        )

    task = Task(env=env_name, id=task_id, args=args)

    with _scoped_v6_settings(
        api_url=api_url,
        api_key=api_key,
        gateway_url=gateway_url,
        telemetry_url=telemetry_url,
    ):
        try:
            agent = _build_agent_from_spec(agent_spec)
            run = await rollout(
                task,
                agent,
                runtime=LocalRuntime(f"{SRC}/env.py"),
                trace_id=runner_config["trace_id"],
                job_id=job_id,
                group_id=group_id,
            )
        finally:
            if not flush_telemetry():
                print(f"telemetry flush incomplete for trace {runner_config.get('trace_id')}")

    status = run.trace.status or ("error" if run.trace.is_error else "completed")
    result: dict = {
        "status": status,
        "reward": run.reward,
        "evaluation_result": run.evaluation or None,
    }
    if run.trace.is_error:
        result["error"] = run.trace.error
    return result


@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    secrets=[modal.Secret.from_name(os.environ.get("HUD_KEYS_SECRET", "hud-keys"), required_keys=["HUD_API_KEY"])],
)
async def run_rollout_dispatched(
    runner_config: dict,
    api_url: str,
    api_key: str,
    gateway_url: str,
    telemetry_url: str,
) -> dict:
    """Platform-dispatched v6 rollout — the Modal analog of an EC2 hud-daemon run.

    The control plane's provision worker spawns this in place of leasing an EC2
    instance when a registry sets ``sandbox_provider="modal"``. It receives the
    same ``runner_config`` ``get_config`` serves the daemon, plus the platform
    identities to report back to (``api_url`` / ``api_key`` / ``gateway_url`` /
    ``telemetry_url`` — the Modal equivalents of the values baked into EC2
    userdata). The trace it reports under is the platform-assigned id the engine
    is polling, so the run surfaces on the platform exactly as an EC2 run would.
    """
    os.environ.setdefault("MCP_TESTING_MODE", "1")
    # The LocalRuntime env-server child inherits these; the parent additionally
    # binds the settings singleton in _scoped_v6_settings below.
    os.environ["HUD_API_URL"] = api_url.rstrip("/")
    os.environ["HUD_GATEWAY_URL"] = gateway_url.rstrip("/")
    os.environ["HUD_TELEMETRY_URL"] = telemetry_url.rstrip("/")
    os.environ["HUD_API_KEY"] = api_key

    return await _run_dispatched_rollout(
        runner_config=runner_config,
        api_url=api_url,
        api_key=api_key,
        gateway_url=gateway_url,
        telemetry_url=telemetry_url,
    )


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
)
def run_gpu_tests(filter_expr: str = "") -> int:
    """Run GPU-requiring pytest tests inside a Modal container.

    Returns the pytest exit code.
    """
    import subprocess

    cmd = [
        "python",
        "-m",
        "pytest",
        "tasks/tests/tasks/",
        "-v",
        "--tb=short",
    ]
    if filter_expr:
        cmd.extend(["-k", filter_expr])

    print(f"=== Running GPU tests: {' '.join(cmd)} ===")
    result = subprocess.run(cmd, cwd=SRC, env={**os.environ, "PYTHONPATH": SRC})
    return result.returncode


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
)
def run_eval(checkpoint: str, benchmarks: str = "SciFact,NQ", local_evals: str = "") -> dict:
    """Evaluate a checkpoint on MTEB benchmarks and optional local eval files.

    checkpoint is relative to /mcp_server (e.g. 'assets/checkpoints/scifact_base').
    """
    import json
    import sys

    sys.path.insert(0, SRC)

    from torchtitan.experiments.embedding.evaluate import evaluate_local, evaluate_mteb

    ckpt_path = f"{SRC}/{checkpoint}"
    results = {}

    if benchmarks:
        tasks = [t.strip() for t in benchmarks.split(",")]
        print(f"Running MTEB eval on {ckpt_path} for {tasks}...")
        metrics = evaluate_mteb(ckpt_path, tasks)
        results["mteb"] = metrics
        print(f"MTEB results: {json.dumps(metrics, indent=2)}")

    if local_evals:
        for eval_file in local_evals.split(","):
            eval_path = f"{SRC}/{eval_file.strip()}"
            print(f"Running local eval on {eval_file}...")
            metrics = evaluate_local(ckpt_path, eval_path)
            results[eval_file.strip()] = metrics
            print(f"Local {eval_file}: {json.dumps(metrics, indent=2)}")

    return results


def main(
    task: str = "",
    tasks: str = "",
    all: bool = False,
    repeats: int = 1,
    model: str = "claude-opus-4-6",
    max_steps: int = 500,
    test: bool = False,
    test_filter: str = "",
    eval_checkpoint: str = "",
    eval_benchmarks: str = "SciFact,NQ",
    eval_local: str = "",
):
    """Spawn agent runs, GPU tests, or a checkpoint eval.

    Usage:
        modal run modal_runner.py --task moe_debug_balance
        modal run modal_runner.py --tasks moe_debug_balance,emb_debug_multi
        modal run modal_runner.py --all --repeats 3 --model claude-sonnet-4-6 --max-steps 100

        modal run modal_runner.py --test
        modal run modal_runner.py --test --test-filter emb
    """
    if eval_checkpoint:
        print(f"Evaluating {eval_checkpoint} on MTEB: {eval_benchmarks}, local: {eval_local}")
        results = _deployed_function("run_eval").remote(
            checkpoint=eval_checkpoint, benchmarks=eval_benchmarks, local_evals=eval_local,
        )
        import json
        print(json.dumps(results, indent=2))
        return

    if test:
        print(
            f"Running GPU tests on Modal{f' (filter: {test_filter})' if test_filter else ''}..."
        )
        exit_code = _deployed_function("run_gpu_tests").remote(filter_expr=test_filter)
        if exit_code != 0:
            raise SystemExit(exit_code)
        return

    if all:
        task_list = _available_tasks()
    else:
        task_list = [t.strip() for t in tasks.split(",") if t.strip()] if tasks else []
        if task:
            task_list.append(task)

    if not task_list:
        raise SystemExit(
            "Provide --task <name>, --tasks <comma-separated>, --all, or --test"
        )

    unique_tasks = list(dict.fromkeys(task_list))
    expanded = [t for t in task_list for _ in range(repeats)]

    repeat_str = f" x {repeats}" if repeats > 1 else ""
    print(f"Spawning {len(expanded)} agent(s) ({len(unique_tasks)} tasks{repeat_str}): {unique_tasks}")

    run_agent_fn = _deployed_function("run_agent")
    handles = [
        run_agent_fn.spawn(task_name=t, model=model, max_steps=max_steps)
        for t in expanded
    ]
    rewards: dict[str, list[float]] = {}
    for t, handle in zip(expanded, handles):
        try:
            reward = handle.get()
        except Exception as e:  # noqa: BLE001 - report per-task failures, keep going
            print(f"  {t}: FAILED ({e})")
            continue
        if reward is None:
            print(f"  {t}: no reward returned")
            continue
        rewards.setdefault(t, []).append(reward)

    print("\n=== Rewards ===")
    for t in unique_tasks:
        vals = rewards.get(t, [])
        if not vals:
            print(f"  {t:24s} (no result)")
        else:
            mean = sum(vals) / len(vals)
            detail = f" {[round(v, 3) for v in vals]}" if len(vals) > 1 else ""
            print(f"  {t:24s} {mean:.3f}{detail}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spawn jobs against the deployed Modal app.")
    parser.add_argument("--task", default="")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test-filter", default="")
    parser.add_argument("--eval-checkpoint", default="", help="Evaluate a checkpoint (e.g. checkpoints/scifact_base)")
    parser.add_argument("--eval-benchmarks", default="SciFact,NQ", help="Comma-separated MTEB tasks")
    parser.add_argument("--eval-local", default="", help="Comma-separated local eval files (e.g. data/val.jsonl,data/nq_val.jsonl)")
    main(**vars(parser.parse_args()))
