"""Local test script for the ML training environment.

Run on any machine with a GPU:
    PYTHONPATH=. uv run python local_test.py --task emb_finetune
    PYTHONPATH=. uv run python local_test.py --task emb_debug_loss --model grok-4.20-beta
    PYTHONPATH=. uv run python local_test.py --list

Concurrent runs on separate GPUs (isolated workspaces):
    PYTHONPATH=. uv run python local_test.py --task emb_finetune --gpu 0 &
    PYTHONPATH=. uv run python local_test.py --task vlm_finetune --gpu 1 &
"""

import argparse
import asyncio
import importlib
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MCP_TESTING_MODE", "1")

import hud
from hud.agents import create_agent

TASK_DIR = Path(__file__).parent / "tasks"


def _available_tasks() -> list[str]:
    return sorted(
        d.name for d in TASK_DIR.iterdir()
        if d.is_dir() and (d / "task.py").exists()
    )


def _load_task(name: str):
    sys.path.insert(0, str(TASK_DIR / name))
    # Force reimport in case env._workspace changed between loads
    if "task" in sys.modules:
        del sys.modules["task"]
    mod = importlib.import_module("task")
    sys.path.pop(0)
    return mod.task


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
        data: dict[str, Any] = resp.json()

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


async def main():
    available = _available_tasks()

    parser = argparse.ArgumentParser(description="Run agent against an ML training task")
    parser.add_argument("--task", default="emb_finetune", choices=available)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--gpu", type=int, default=None, help="Pin to a specific GPU and use an isolated workspace")
    parser.add_argument("--job-id", default=None, help="HUD job ID to group this run under")
    parser.add_argument("--taskset", default=None, help="HUD taskset name to associate traces with")
    parser.add_argument("--list", action="store_true", help="List available tasks and exit")
    args = parser.parse_args()

    if args.list:
        for t in available:
            print(t)
        return

    # Pin GPU and isolate workspace
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["MASTER_PORT"] = str(29500 + args.gpu)

        import env
        isolated_ws = str(Path(env._workspace).parent / f"workspace_gpu{args.gpu}")
        env._workspace = isolated_ws
        print(f"[gpu {args.gpu}] workspace: {isolated_ws}, MASTER_PORT: {29500 + args.gpu}")

    import env
    from env import AGENT_CONFIG

    env.init_tools(env._workspace)

    task = _load_task(args.task)
    task.metadata["trace_name"] = args.task

    system_prompt = AGENT_CONFIG["system_prompt"].replace("/workspace", env._workspace)
    task.agent_config = {**AGENT_CONFIG, "system_prompt": system_prompt}

    # Resolve taskset name -> UUID and task slugs -> task_version_ids
    taskset_id: str | None = None
    if args.taskset:
        try:
            taskset_id, slug_map = _fetch_taskset_info(args.taskset)
            if task.slug:
                task_version_id = slug_map.get(task.slug)
                if task_version_id:
                    task.id = task_version_id
                    print(f"Linked task '{task.slug}' -> {task_version_id}")
                else:
                    print(f"Warning: task slug '{task.slug}' not found in taskset '{args.taskset}'")
            print(f"Taskset: {args.taskset} ({taskset_id})")
        except Exception as e:
            print(f"Warning: could not fetch taskset: {e}")

    print(f"=== {args.task} ({args.model}) ===")
    eval_kwargs: dict = {"name": args.task}
    if args.job_id:
        eval_kwargs["job_id"] = args.job_id
    if taskset_id:
        eval_kwargs["taskset_id"] = taskset_id
    async with hud.eval(task, **eval_kwargs) as ctx:
        agent = create_agent(args.model)
        agent.system_prompt = system_prompt
        await agent.run(ctx, max_steps=args.max_steps)
        print(f"Reward: {ctx.reward}")


if __name__ == "__main__":
    asyncio.run(main())
