"""Register calibration tools on the env.

Subclasses AgentTool to return the full Trace (with reward and trajectory)
instead of just the agent's text response. Tools are prefixed with underscore
to hide them from the inner eval agent's tool listings.

Usage:
    from tasks.calibration.calibrate import register_calibration_tools
    register_calibration_tools(model="claude-sonnet-4-6")
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from fastmcp.tools import ToolResult
from mcp.types import TextContent

from hud.tools.agent import AgentTool

from env import AGENT_CONFIG, WORKSPACE, env

TASK_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL = "claude-sonnet-4-6"


class CalibrationTool(AgentTool):
    """AgentTool that returns the full Trace JSON instead of just content.

    The Trace includes reward, trajectory (trace steps), content, error status --
    everything the calibrating agent needs to evaluate a trial.
    """

    async def __call__(self, **kwargs: Any) -> ToolResult:
        from hud.eval.context import get_current_trace_id
        from hud.eval.manager import run_eval
        from hud.telemetry.instrument import instrument

        filtered = {k: v for k, v in kwargs.items() if k in self._visible_params}
        base_args = self._task.args or {}
        task = self._task.model_copy(update={"args": {**base_args, **filtered}})

        parent_trace_id = get_current_trace_id()
        is_nested = parent_trace_id is not None
        should_trace = self._trace and not is_nested

        @instrument(category="subagent", name=self.name)
        async def _run_subagent() -> ToolResult:
            async with run_eval(
                task,
                trace=should_trace,
                trace_id=parent_trace_id,
                quiet=True,
            ) as ctx:
                if self._model:
                    from hud.agents import create_agent

                    agent = create_agent(self._model, **self._agent_params)
                else:
                    agent = self._agent_cls.create(**self._agent_params)  # type: ignore[union-attr]

                result = await agent.run(ctx)
                result.reward = ctx.reward if ctx.reward is not None else result.reward
                return ToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=result.model_dump_json(exclude={"messages"}),
                        )
                    ]
                )

        return await _run_subagent()


def _load_task(slug: str):
    """Load a Task from the tasks/ registry, forcing reimport for fresh thresholds."""
    mod_name = f"tasks.{slug}.task"
    sys.modules.pop(mod_name, None)
    mod = importlib.import_module(mod_name)
    task = mod.task
    system_prompt = AGENT_CONFIG["system_prompt"].replace("/workspace", WORKSPACE)
    task.agent_config = {**AGENT_CONFIG, "system_prompt": system_prompt}
    return task


def register_calibration_tools(model: str = _DEFAULT_MODEL) -> list[str]:
    """Register a CalibrationTool for each task in tasks/.

    Returns the list of registered tool names.
    """
    registered: list[str] = []
    for task_dir in sorted(TASK_DIR.iterdir()):
        if not task_dir.is_dir() or not (task_dir / "task.py").exists():
            continue
        slug = task_dir.name
        task = _load_task(slug)
        tool = CalibrationTool(
            task,
            model=model,
            name=f"_run_{slug}",
            description=f"Run eval agent against the {slug} task. Returns full Trace with reward and trajectory.",
            trace=True,
        )
        tool.register(env)
        registered.append(f"_run_{slug}")
    return registered
