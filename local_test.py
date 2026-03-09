"""Local test script for the ML training environment.

Run on any machine with a GPU:
    PYTHONPATH=. uv run python local_test.py --task finetune_embedding
    PYTHONPATH=. uv run python local_test.py --task debug_embedding_loss --model grok-4-1-fast
    PYTHONPATH=. uv run python local_test.py --list
"""

import argparse
import asyncio
import importlib
import sys
from pathlib import Path

import hud

from env import AGENT_CONFIG, VLM_AGENT_CONFIG

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
    mod = importlib.import_module("task")
    sys.path.pop(0)
    return mod.task


async def main():
    available = _available_tasks()

    parser = argparse.ArgumentParser(description="Run agent against an ML training task")
    parser.add_argument("--task", default="finetune_embedding", choices=available)
    parser.add_argument("--model", default="claude-opus-4-6", choices=list(AGENTS.keys()))
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--list", action="store_true", help="List available tasks and exit")
    args = parser.parse_args()

    if args.list:
        for t in available:
            print(t)
        return

    task = _load_task(args.task)
    task.agent_config = VLM_AGENT_CONFIG if args.task.startswith("vlm") or args.task.startswith("debug_vlm") else AGENT_CONFIG

    print(f"=== {args.task} ({args.model}) ===")
    async with hud.eval(task) as ctx:
        agent = AGENTS[args.model](args.model)
        await agent.run(ctx, max_steps=args.max_steps)
        print(f"Reward: {ctx.reward}")


if __name__ == "__main__":
    asyncio.run(main())
