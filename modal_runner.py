"""Modal runner for agent tasks and GPU tests.

Deploy once, then spawn isolated agent runs:

    modal deploy modal_runner.py

    # Single task
    modal run modal_runner.py --task emb_finetune

    # Multiple tasks in parallel (each gets its own container)
    modal run modal_runner.py --tasks emb_finetune,emb_debug_loss,vlm_finetune

    # All tasks, 3 repeats each
    modal run modal_runner.py --all --repeats 3 --model claude-sonnet-4-6 --max-steps 100

    # Run GPU tests
    modal run modal_runner.py --test
    modal run modal_runner.py --test --test-filter emb

Setup:
    pip install modal
    modal setup
    modal secret create hud-keys HUD_API_KEY=<key>
"""

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
    .add_local_dir("tasks", remote_path="/mcp_server/tasks")
)

SRC = "/mcp_server"
WS = "/home/ubuntu/workspace"
TASK_DIR = pathlib.Path(__file__).parent / "tasks"


def _available_tasks() -> list[str]:
    return sorted(
        d.name for d in TASK_DIR.iterdir()
        if d.is_dir() and (d / "task.py").exists()
    )


def _deployed_function(name: str):
    return modal.Function.from_name(APP_NAME, name)


def _fetch_taskset_info(taskset_name: str) -> tuple[str, dict[str, str]]:
    """Fetch taskset by name. Returns (taskset_uuid, slug -> task_version_id mapping)."""
    import httpx
    from hud.settings import settings

    headers = {}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    with httpx.Client(timeout=15) as client:
        resp = client.get(
            f"{settings.hud_api_url}/tasks/evalset/{taskset_name}",
            headers=headers,
            params={"all": "true"},
        )
        resp.raise_for_status()
        data = resp.json()

    taskset_id = data.get("evalset_id") or ""
    slug_to_id: dict[str, str] = {}
    for task_data in (data.get("tasks") or {}).values():
        if not isinstance(task_data, dict):
            continue
        slug = task_data.get("slug") or task_data.get("external_id") or ""
        task_version_id = task_data.get("id") or ""
        if slug and task_version_id:
            slug_to_id[slug] = task_version_id
    return taskset_id, slug_to_id


@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    secrets=[modal.Secret.from_name("hud-keys", required_keys=["HUD_API_KEY"])],
)
async def run_agent(
    task_name: str,
    model: str = "claude-opus-4-6",
    max_steps: int = 500,
    job_id: str = "",
    taskset: str = "",
):
    """Run an agent against a specific task. Each invocation gets an isolated container."""
    import importlib
    import sys

    os.environ.setdefault("MCP_TESTING_MODE", "1")

    import env
    import hud
    from env import AGENT_CONFIG
    from hud.agents import create_agent

    env.init_tools()

    sys.path.insert(0, str(pathlib.Path(SRC) / "tasks" / task_name))
    if "task" in sys.modules:
        del sys.modules["task"]
    task = importlib.import_module("task").task
    sys.path.pop(0)

    task.metadata["trace_name"] = task_name
    task.agent_config = AGENT_CONFIG

    # Resolve taskset
    taskset_id: str | None = None
    if taskset:
        try:
            taskset_id, slug_map = _fetch_taskset_info(taskset)
            if task.slug:
                task_version_id = slug_map.get(task.slug)
                if task_version_id:
                    task.id = task_version_id
        except Exception as e:
            print(f"Warning: could not fetch taskset: {e}")

    print(f"=== {task_name} ({model}) ===")
    eval_kwargs: dict = {"name": task_name}
    if job_id:
        eval_kwargs["job_id"] = job_id
    if taskset_id:
        eval_kwargs["taskset_id"] = taskset_id
    async with hud.eval(task, **eval_kwargs) as ctx:
        agent = create_agent(model)
        agent.system_prompt = AGENT_CONFIG["system_prompt"]
        await agent.run(ctx, max_steps=max_steps)
        print(f"Reward: {ctx.reward}")


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
    job_id: str = "",
    taskset: str = "",
    eval_checkpoint: str = "",
    eval_benchmarks: str = "SciFact,NQ",
    eval_local: str = "",
):
    """Spawn agent runs, GPU tests, or calibration server.

    Usage:
        modal run modal_runner.py --task emb_finetune
        modal run modal_runner.py --tasks emb_finetune,emb_debug_loss,vlm_finetune
        modal run modal_runner.py --all --repeats 3 --model claude-sonnet-4-6 --max-steps 100
        modal run modal_runner.py --task emb_finetune --taskset claire-tasks

        modal run modal_runner.py --test
        modal run modal_runner.py --test --test-filter emb
    """
    import asyncio
    import uuid

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

    # Resolve taskset name -> UUID once locally
    taskset_id = ""
    if taskset:
        from hud.datasets.loader import resolve_taskset_id
        taskset_id = resolve_taskset_id(taskset)
        print(f"Taskset: {taskset} ({taskset_id})")

    if not job_id and (len(expanded) > 1 or taskset_id):
        job_id = str(uuid.uuid4())
        asyncio.run(_register_job(job_id, unique_tasks, model, repeats, taskset_id=taskset_id))

    repeat_str = f" x {repeats}" if repeats > 1 else ""
    print(f"Spawning {len(expanded)} agent(s) ({len(unique_tasks)} tasks{repeat_str}): {unique_tasks}")
    if job_id:
        print(f"HUD job: https://hud.ai/jobs/{job_id}")
    run_agent_fn = _deployed_function("run_agent")
    handles = [
        run_agent_fn.spawn(task_name=t, model=model, max_steps=max_steps, job_id=job_id, taskset=taskset)
        for t in expanded
    ]
    for handle in handles:
        handle.get()


async def _register_job(
    job_id: str, task_list: list[str], model: str, repeats: int = 1, taskset_id: str = ""
) -> None:
    """Register a HUD job so all parallel agent runs are grouped together."""
    from hud.eval.manager import _send_job_enter

    repeat_str = f" x {repeats}" if repeats > 1 else ""
    name = f"modal: {model} ({len(task_list)} tasks{repeat_str})"
    await _send_job_enter(
        job_id=job_id,
        name=name,
        variants={"model": [model]},
        group=len(task_list) * repeats,
        api_key=None,
        taskset_id=taskset_id or None,
    )


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
    parser.add_argument("--job-id", default="")
    parser.add_argument("--taskset", default="", help="HUD taskset name to associate traces with")
    parser.add_argument("--eval-checkpoint", default="", help="Evaluate a checkpoint (e.g. checkpoints/scifact_base)")
    parser.add_argument("--eval-benchmarks", default="SciFact,NQ", help="Comma-separated MTEB tasks")
    parser.add_argument("--eval-local", default="", help="Comma-separated local eval files (e.g. data/val.jsonl,data/nq_val.jsonl)")
    main(**vars(parser.parse_args()))
