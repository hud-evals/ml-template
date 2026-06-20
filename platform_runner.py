from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any, cast

from hud.eval.runtime import HostedRuntime, RuntimeConfig

MODAL_GPU_TYPE = os.environ.get("HUD_MODAL_GPU_TYPE", "H100")
MODAL_GPU_COUNT = int(os.environ.get("HUD_MODAL_GPU_COUNT", "1"))
MODAL_STARTUP_TIMEOUT_S = int(os.environ.get("HUD_MODAL_STARTUP_TIMEOUT_S", "900"))
MODAL_RUN_TIMEOUT_S = int(os.environ.get("HUD_MODAL_RUN_TIMEOUT_S", "7200"))


def modal_runtime_config(
    *,
    gpu_type: str = MODAL_GPU_TYPE,
    gpu_count: int = MODAL_GPU_COUNT,
    startup_timeout_s: int = MODAL_STARTUP_TIMEOUT_S,
    run_timeout_s: int = MODAL_RUN_TIMEOUT_S,
) -> dict[str, Any]:
    """GPU settings for HUD hosted Modal."""
    return {
        "resources": {"gpu": {"type": gpu_type, "count": gpu_count}},
        "limits": {
            "startup_timeout_s": startup_timeout_s,
            "run_timeout_s": run_timeout_s,
        },
    }


def build_modal_deploy_payload(
    *,
    build_id: str,
    name: str,
    registry_id: str | None = None,
    no_cache: bool = False,
) -> dict[str, Any]:
    """Build trigger payload with Modal runtime metadata."""
    payload: dict[str, Any] = {
        "source": "direct",
        "build_id": build_id,
        "name": name,
        "no_cache": no_cache,
        "runtime_provider": "modal",
    }
    if registry_id:
        payload["registry_id"] = registry_id
    return payload


def build_submit_payload(
    *,
    task: Any,
    agent_spec: dict[str, Any],
    job_id: str,
    trace_id: str,
    runtime_config: dict[str, Any],
    group_id: str | None = None,
) -> dict[str, Any]:
    """Build the hosted run submission payload."""
    payload: dict[str, Any] = {
        "trace_id": str(uuid.UUID(trace_id)),
        "job_id": str(uuid.UUID(job_id)),
        "env": task.env,
        "task": task.id,
        "args": task.args,
        "agent": agent_spec,
        "runtime_config": runtime_config,
    }
    if group_id is not None:
        payload["group_id"] = group_id
    return payload


class ModalHUDRuntime(HostedRuntime):
    """HUD-hosted runtime that submits Modal GPU settings."""


    def __init__(
        self,
        *,
        runtime_config: dict[str, Any] | None = None,
        poll_interval: float = 5.0,
        run_timeout: float = MODAL_RUN_TIMEOUT_S,
    ) -> None:
        super().__init__(poll_interval=poll_interval, run_timeout=run_timeout)
        self.runtime_config = runtime_config or modal_runtime_config()

    async def _submit_and_await(
        self,
        task: Any,
        agent: Any,
        *,
        job_id: str,
        group_id: str | None,
        trace_id: str,
    ) -> dict[str, Any]:
        runtime_task = task.model_copy(
            update={"runtime_config": RuntimeConfig.model_validate(self.runtime_config)}
        )
        return await super()._submit_and_await(
            runtime_task,
            agent,
            job_id=job_id,
            group_id=group_id,
            trace_id=trace_id,
        )


def _provider_for(model: str) -> str | None:
    m = model.lower()
    if m.startswith(("claude", "anthropic")):
        return "claude"
    if m.startswith(("gpt", "o1", "o3", "o4", "openai")):
        return "openai"
    if m.startswith(("gemini", "google")):
        return "gemini"
    return None


def _make_agent(model: str, system_prompt: str, max_steps: int):
    from hud.agents import create_agent

    provider = _provider_for(model)
    if provider is None:
        return create_agent(model, system_prompt=system_prompt, max_steps=max_steps)

    from hud.types import AgentType

    agent_type = AgentType(provider)
    config = agent_type.config_cls(
        model=model,
        system_prompt=system_prompt,
        max_steps=max_steps,
    )
    return agent_type.cls(config)


def _selected_tasks(task: str, tasks_csv: str, all_tasks: bool) -> list[tuple[str, Any]]:
    from tasks import tasks

    if all_tasks:
        names = sorted(tasks)
    else:
        names = [name.strip() for name in tasks_csv.split(",") if name.strip()]
        if task:
            names.append(task)
    if not names:
        raise SystemExit("Provide --task <name>, --tasks <csv>, or --all")

    selected: list[tuple[str, Any]] = []
    for name in dict.fromkeys(names):
        if name not in tasks:
            raise SystemExit(f"Unknown task {name!r}. Available: {', '.join(sorted(tasks))}")
        selected.append((name, tasks[name]))
    return selected


def _resolve_deploy_name(env_source: Any, config: dict[str, Any], explicit_name: str | None) -> str:
    """Resolve the deploy name without requiring saved local config.

    Fresh public checkouts do not have saved config. Prefer an explicit CLI
    name, then existing config, then the literal Environment(...) name from
    source, and finally the SDK's directory-derived fallback.
    """
    if explicit_name:
        return explicit_name

    configured = config.get("registryName")
    if isinstance(configured, str) and configured:
        return configured

    try:
        from hud.cli.deploy import _resolve_declared_name
        from hud.utils.hud_console import HUDConsole

        declared = _resolve_declared_name(env_source, HUDConsole())
    except Exception:
        declared = None
    if isinstance(declared, str) and declared:
        return declared

    return env_source.environment_name()


def _print_run_rewards(selected: list[tuple[str, Any]], runs: list[Any], group: int) -> None:
    expected_names = [
        name
        for name, _ in selected
        for _run_idx in range(group)
    ]
    for name, run in zip(expected_names, runs, strict=False):
        reward = run.reward if hasattr(run, "reward") else getattr(run.grade, "reward", None)
        print(f"  {name}: {reward}")


async def run_modal_platform(
    *,
    task: str = "",
    tasks_csv: str = "",
    all_tasks: bool = False,
    model: str = "claude-opus-4-6",
    max_steps: int = 500,
    group: int = 1,
    max_concurrent: int | None = None,
    gpu_type: str = MODAL_GPU_TYPE,
) -> None:
    """Run tasks through HUD hosted Modal with GPU settings."""
    from hud import Taskset
    from env import AGENT_CONFIG

    selected = _selected_tasks(task, tasks_csv, all_tasks)
    taskset = Taskset("ml-template-v6", [item for _, item in selected])
    agent = _make_agent(model, AGENT_CONFIG["system_prompt"], max_steps)
    runtime = ModalHUDRuntime(runtime_config=modal_runtime_config(gpu_type=gpu_type))

    print(
        f"Submitting {len(selected)} task(s) to HUD hosted Modal "
        f"({gpu_type} x {MODAL_GPU_COUNT})"
    )
    job = await taskset.run(agent, runtime=runtime, group=group, max_concurrent=max_concurrent)
    print(f"Job: https://hud.ai/jobs/{job.id}" if job.id else "Job submitted")
    _print_run_rewards(selected, job.runs, group)


async def deploy_modal_platform(
    *,
    directory: str = ".",
    name: str | None = None,
    registry_id: str | None = None,
    no_cache: bool = False,
    verbose: bool = False,
) -> None:
    """Deploy this environment for HUD hosted Modal."""
    import typer
    from hud.cli.deploy import _create_build_upload, _create_tarball, _upload_build_context
    from hud.cli.utils.build_logs import poll_build_status, stream_build_logs
    from hud.cli.utils.source import EnvironmentSource
    from hud.utils.exceptions import HudRequestError
    from hud.utils.hud_console import HUDConsole
    from hud.utils.platform import PlatformClient

    console = HUDConsole()
    env_dir = Path(directory).resolve()
    env_source = EnvironmentSource.open(env_dir)
    config = env_source.load_config()
    deploy_name = _resolve_deploy_name(env_source, config, name)
    deploy_registry_id = registry_id or cast("str | None", config.get("registryId"))

    platform = PlatformClient.from_settings()
    tarball_path = _create_tarball(env_dir, verbose=verbose, console=console)
    try:
        upload = await _create_build_upload(platform)
        await _upload_build_context(upload.upload_url, tarball_path)
        payload = build_modal_deploy_payload(
            build_id=upload.build_id,
            name=deploy_name,
            registry_id=deploy_registry_id,
            no_cache=no_cache,
        )
        trigger_data = await platform.apost("/builds/trigger", json=payload)
        build_id = trigger_data["id"]
        print(f"Build triggered: {build_id}")
        try:
            status = await stream_build_logs(platform, build_id, console=console)
        except Exception:
            status_data = await poll_build_status(platform, build_id, console=console)
            status = status_data.get("status", "UNKNOWN")
        if str(status).upper() != "SUCCEEDED":
            raise SystemExit(f"Modal build ended with status={status}")
        if trigger_data.get("registry_id"):
            env_source.save_config(
                {"registryId": trigger_data["registry_id"], "registryName": deploy_name}
            )
    except HudRequestError as exc:
        detail = (exc.response_json or {}).get("detail")
        raise SystemExit(f"HUD API error: {detail or exc}") from exc
    except typer.Exit:
        raise
    finally:
        tarball_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ml-template on HUD hosted Modal")
    sub = parser.add_subparsers(dest="command", required=True)

    deploy = sub.add_parser("deploy", help="Deploy for HUD hosted Modal")
    deploy.add_argument("--directory", default=".")
    deploy.add_argument("--name", default=None)
    deploy.add_argument("--registry-id", default=None)
    deploy.add_argument("--no-cache", action="store_true")
    deploy.add_argument("--verbose", action="store_true")

    run = sub.add_parser("run", help="Submit tasks to HUD hosted Modal")
    run.add_argument("--task", default="")
    run.add_argument("--tasks", default="")
    run.add_argument("--all", action="store_true")
    run.add_argument("--model", default="claude-opus-4-6")
    run.add_argument("--max-steps", type=int, default=500)
    run.add_argument("--group", type=int, default=1)
    run.add_argument("--max-concurrent", type=int, default=None)
    run.add_argument("--gpu-type", default=MODAL_GPU_TYPE)

    args = parser.parse_args()
    if args.command == "deploy":
        asyncio.run(
            deploy_modal_platform(
                directory=args.directory,
                name=args.name,
                registry_id=args.registry_id,
                no_cache=args.no_cache,
                verbose=args.verbose,
            )
        )
    else:
        asyncio.run(
            run_modal_platform(
                task=args.task,
                tasks_csv=args.tasks,
                all_tasks=args.all,
                model=args.model,
                max_steps=args.max_steps,
                group=args.group,
                max_concurrent=args.max_concurrent,
                gpu_type=args.gpu_type,
            )
        )


if __name__ == "__main__":
    main()
