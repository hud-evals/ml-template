"""Local test script for the ML training environment.

Run on any machine with a GPU:
    PYTHONPATH=. uv run python local_test.py --task finetune_embedding
    PYTHONPATH=. uv run python local_test.py --task debug_embedding_loss --model grok-4-1-fast
    PYTHONPATH=. uv run python local_test.py --list

Concurrent runs on separate GPUs (isolated workspaces):
    PYTHONPATH=. uv run python local_test.py --task finetune_embedding --gpu 0 &
    PYTHONPATH=. uv run python local_test.py --task vlm_finetune --gpu 1 &
"""

import argparse
import asyncio
import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("MCP_TESTING_MODE", "1")

import hud

AGENTS = {
    "claude-opus-4-6": lambda model: __import__("hud.agents.claude", fromlist=["ClaudeAgent"]).ClaudeAgent.create(model=model),
    "claude-sonnet-4-6": lambda model: __import__("hud.agents.claude", fromlist=["ClaudeAgent"]).ClaudeAgent.create(model=model),
    "grok-4-1-fast": lambda model: __import__("hud.agents.openai", fromlist=["OpenAIAgent"]).OpenAIAgent.create(model=model),
}

TASK_DIR = Path(__file__).parent / "tasks"


def _available_tasks() -> list[str]:
    return sorted(
        d.name for d in TASK_DIR.iterdir()
        if d.is_dir() and (d / "task.py").exists()
    )


def _load_task(name: str):
    sys.path.insert(0, str(TASK_DIR / name))
    # Force reimport in case env.WORKSPACE changed between loads
    if "task" in sys.modules:
        del sys.modules["task"]
    mod = importlib.import_module("task")
    sys.path.pop(0)
    return mod.task


async def main():
    available = _available_tasks()

    parser = argparse.ArgumentParser(description="Run agent against an ML training task")
    parser.add_argument("--task", default="finetune_embedding", choices=available)
    parser.add_argument("--model", default="claude-opus-4-6", choices=list(AGENTS.keys()))
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--gpu", type=int, default=None, help="Pin to a specific GPU and use an isolated workspace")
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
        isolated_ws = str(Path(env.WORKSPACE).parent / f"workspace_gpu{args.gpu}")
        env.WORKSPACE = isolated_ws
        print(f"[gpu {args.gpu}] workspace: {isolated_ws}, MASTER_PORT: {29500 + args.gpu}")

    import env
    from env import AGENT_CONFIG, VLM_AGENT_CONFIG

    env.init_tools(env.WORKSPACE)

    task = _load_task(args.task)
    is_vlm = args.task.startswith("vlm") or args.task.startswith("debug_vlm")
    agent_config = VLM_AGENT_CONFIG if is_vlm else AGENT_CONFIG

    # Patch system prompt with actual workspace path (differs from /workspace on Modal)
    system_prompt = agent_config["system_prompt"].replace("/workspace", env.WORKSPACE)
    task.agent_config = {**agent_config, "system_prompt": system_prompt}

    print(f"=== {args.task} ({args.model}) ===")
    async with hud.eval(task) as ctx:
        agent = AGENTS[args.model](args.model)
        agent.system_prompt = system_prompt
        await agent.run(ctx, max_steps=args.max_steps)
        print(f"Reward: {ctx.reward}")


if __name__ == "__main__":
    asyncio.run(main())
