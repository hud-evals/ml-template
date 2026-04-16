"""ML training environment -- torchtitan experiments."""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from hud import Environment
from hud.native.graders import BashGrader, Grade


logger = logging.getLogger(__name__)
MCP_TESTING_MODE = os.environ.get("MCP_TESTING_MODE") in ["1", "true"]


def bash(cmd: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    logger.info("bash: %s", cmd)
    result = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.warning(
            "bash exited %d\nstdout: %s\nstderr: %s",
            result.returncode,
            result.stdout[-3000:] if result.stdout else "(empty)",
            result.stderr[-3000:] if result.stderr else "(empty)",
        )
        if check:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result

_REPO_ROOT = Path(__file__).parent

SRC_DIR = "/mcp_server"
WORKSPACE = "/home/ubuntu/workspace"

if Path("/mcp_server/torchtitan").exists():
    _src_dir = "/mcp_server"
    _workspace = "/home/ubuntu/workspace"
elif Path("/code/torchtitan").exists():
    _src_dir = "/code"
    _workspace = "/workspace"
else:
    _src_dir = str(_REPO_ROOT)
    _workspace = str(_REPO_ROOT / "workspace")

import sys
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

env = Environment("ml-template-1")

AGENT_CONFIG = {
    "system_prompt": (
        "You are an expert ML engineer working with torchtitan.\n\n"
        "Environment:\n"
        "  Workspace: /home/ubuntu/workspace (your shell starts here)\n"
        "  Framework source: /home/ubuntu/workspace/torchtitan  (editable)\n"
        "  No network access.\n"
        "  Pre-staged data and assets: /home/ubuntu/workspace/data/, /home/ubuntu/workspace/assets/\n\n"
        "Running experiments:\n"
        "  Standard models train via run_train.sh (wraps torchrun -m torchtitan.train):\n"
        "    NGPU=1 MODULE=<module> CONFIG=<config> ./run_train.sh [extra CLI flags]\n"
        "  Experiments under torchtitan/experiments/ may define their own entrypoints.\n"
        "  Check the experiment code for usage.\n\n"
        "Constraints:\n"
        "  - Bash sessions time out after 120s with no output. For training runs:\n"
        "      nohup cmd > log.log 2>&1 & echo PID:$! -- then poll with: tail -20 log.log\n"
        "  - Delete intermediate checkpoints to save disk, but always keep your final checkpoint.\n"
    ),
}

_tools_initialized = False


def init_tools(workspace: str | None = None):
    """Initialize coding tools with workspace sandboxing. Call after WORKSPACE is set."""
    global _tools_initialized
    if _tools_initialized:
        return

    # Lock down /mcp_server so the agent can't read source, graders, or patches.
    # Done at runtime (not just Dockerfile) because Modal adds files after build.
    if os.getuid() == 0:
        if os.path.isdir("/mcp_server"):
            os.system(
                "find /mcp_server -maxdepth 0 -exec chmod 700 {} + && "
                "find /mcp_server -mindepth 1 -maxdepth 1 "
                "! -name assets ! -name data "
                "-exec chmod -R 700 {} +"
            )
        # Lock grader state files (contamination info etc.) written during setup.
        os.system("chmod -R 700 /tmp/.grader_* 2>/dev/null || true")
        # Hide other users' processes so the agent can't see task name from ps.
        os.system("mount -o remount,hidepid=2 /proc 2>/dev/null || true")
    _tools_initialized = True

    import asyncio as _aio

    from hud.tools.coding import (
        ApplyPatchTool,
        BashTool,
        ClaudeBashSession,
        EditTool,
        GeminiEditTool,
        GeminiShellTool,
        GeminiWriteTool,
        ShellTool,
    )
    from hud.tools.coding.utils import get_demote_preexec_fn
    from hud.tools.filesystem import (
        GeminiGlobTool,
        GeminiListTool,
        GeminiReadManyTool,
        GeminiReadTool,
        GeminiSearchTool,
    )

    ws = workspace or _workspace

    class _SandboxedSession(ClaudeBashSession):
        """Bash session locked to the workspace directory."""

        async def start(self):
            if self._started:
                await _aio.sleep(0)
                return
            self._process = await _aio.create_subprocess_shell(
                self.command,
                stdin=_aio.subprocess.PIPE,
                stdout=_aio.subprocess.PIPE,
                stderr=_aio.subprocess.PIPE,
                cwd=ws,
                preexec_fn=get_demote_preexec_fn(),
            )
            self._started = True
            self._timed_out = False
            await self.run(
                f'export HOME="{ws}" && '
                f'export PATH="{ws}/.venv/bin:/usr/bin:/bin" && '
                f'export PYTHONPATH="{ws}" && '
                f'_ws="{ws}" && '
                # Restrict cd to workspace
                f'cd() {{ local t="${{1:-.}}"; local r=$(realpath -m "$t" 2>/dev/null || echo "$t"); '
                f'case "$r" in "$_ws"*) builtin cd "$t" ;; '
                f'*) echo "Error: cannot navigate outside workspace" >&2; return 1 ;; esac; }} && '
                # Restrict ls to workspace paths
                f'ls() {{ for a in "$@"; do case "$a" in -*) ;; /*) '
                f'case "$a" in "$_ws"*) ;; *) echo "Error: cannot list outside workspace" >&2; return 1 ;; esac ;; esac; done; '
                f'command ls "$@"; }} && '
                # Restrict find to workspace
                f'find() {{ case "$1" in "$_ws"*|.*) command find "$@" ;; '
                f'*) echo "Error: cannot search outside workspace" >&2; return 1 ;; esac; }} && '
                # Restrict cat/head/tail to workspace paths
                f'_check_path() {{ for a in "$@"; do case "$a" in -*|"") ;; /*) '
                f'case "$a" in "$_ws"*|/dev/*|/proc/self/*) ;; *) echo "Error: cannot access outside workspace: $a" >&2; return 1 ;; esac ;; esac; done; return 0; }} && '
                f'cat() {{ _check_path "$@" && command cat "$@"; }} && '
                f'head() {{ _check_path "$@" && command head "$@"; }} && '
                f'tail() {{ _check_path "$@" && command tail "$@"; }}'
            )

    # Claude tools
    bash_tool = BashTool()
    bash_tool.session = _SandboxedSession()
    bash_tool.register(env)
    EditTool().register(env)

    # OpenAI tools
    ShellTool(cwd=ws).register(env)
    ApplyPatchTool().register(env)

    # Gemini tools
    GeminiShellTool(base_directory=ws).register(env)
    GeminiEditTool(base_directory=ws).register(env)
    GeminiWriteTool(base_directory=ws).register(env)
    GeminiReadTool(base_path=ws).register(env)
    GeminiSearchTool(base_path=ws).register(env)
    GeminiGlobTool(base_path=ws).register(env)
    GeminiListTool(base_path=ws).register(env)
    GeminiReadManyTool(base_path=ws).register(env)


init_tools()


async def _grade(graders: list[dict[str, Any]]):
    """Build an EvaluationResult from a list of grader dicts.

    Each grader is either:
      - script-based: {name, script, args?, weight?, timeout?}
        Written to /tmp/{name}.py, command auto-built.
      - command-based: {name, command, weight?, timeout?}
        Runs as-is.
    """
    # Kill any leftover agent GPU processes so graders have full GPU access.
    os.system("pkill -9 -f torchrun 2>/dev/null; pkill -9 -f torchtitan.train 2>/dev/null; sleep 1")
    for g in graders:
        if "script" in g:
            stem = g.pop("_script_stem", g["name"])
            script = g.pop("script")
            args = g.pop("args", "")
            Path(f"/tmp/{stem}.py").write_text(script)
            g.setdefault("command", f"python /tmp/{stem}.py {args}")
    total = sum(g.get("weight", 1) for g in graders)
    return await Grade.gather(*[
        BashGrader.grade(
            name=g.get("name"),
            weight=g.get("weight", 1) / total,
            command=g["command"],
            timeout_seconds=g.get("timeout", 10),
        )
        for g in graders
    ])


_STAGED_VENV = "/opt/venv"


def _setup_workspace(setup_command: str | None = None):
    """Set up workspace and optionally run a command to stage assets.

    Copies torchtitan source, tests, the standard launcher script,
    and a relocatable Python venv. Task-specific assets are downloaded
    by the setup_command.
    """
    shutil.rmtree(_workspace, ignore_errors=True)
    os.makedirs(_workspace, exist_ok=True)

    bash(f"cp -r {_src_dir}/torchtitan {_workspace}/torchtitan")
    bash(f"cp -r {_src_dir}/tests {_workspace}/tests")
    if os.path.exists(f"{_src_dir}/run_train.sh"):
        bash(f"cp {_src_dir}/run_train.sh {_workspace}/run_train.sh")
        bash(f"chmod +x {_workspace}/run_train.sh")

    if os.path.isdir(_STAGED_VENV):
        bash(f"ln -s {_STAGED_VENV} {_workspace}/.venv")

    if setup_command:
        bash(setup_command)

    if os.getuid() == 0:
        bash(f"chown -R 1000:1000 {_workspace}")

    os.chdir(_workspace)



def _apply_patches(patches: list[str] | None = None) -> None:
    """Apply inline patch strings to the workspace."""
    if not patches:
        return

    for i, patch_content in enumerate(patches):
        patch_path = Path(_workspace) / f".patch_{i}"
        patch_path.write_text(patch_content)
        bash(f"cd {_workspace} && patch --no-backup -p1 < {patch_path} && rm {patch_path}")


def _write_tmp_json(name: str, payload: dict[str, Any]) -> None:
    Path(f"/tmp/{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


# ===========================================================================
# Scenarios
# ===========================================================================


@env.scenario(name="train_to_target")
async def train_to_target(
    prompt: str,
    graders: list[dict[str, Any]],
    setup_command: str | None = None,
):
    """Train forward from a clean or staged workspace."""
    _setup_workspace(setup_command)
    yield prompt
    yield await _grade(graders)


@env.scenario(name="repair_degraded_recipe")
async def repair_degraded_recipe(
    prompt: str,
    graders: list[dict[str, Any]],
    patches: list[str],
    setup_command: str | None = None,
):
    """Stage a degraded recipe via patches, then let the agent repair it."""
    _setup_workspace(setup_command)
    _apply_patches(patches)
    yield prompt
    yield await _grade(graders)


@env.scenario(name="audit_training_data")
async def audit_training_data(
    prompt: str,
    graders: list[dict[str, Any]],
    contamination: str = "label_noise",
    train_file: str = "data/scifact.jsonl",
    val_file: str = "data/val.jsonl",
    noise_rate: float = 0.3,
    leak_rate: float = 0.2,
    setup_command: str | None = None,
):
    """Inject training-data contamination, then let the agent audit and clean it."""
    _setup_workspace(setup_command)
    bash(
        f"python -m mutations data {contamination} {_workspace}"
        f" --train-file {train_file} --val-file {val_file}"
        f" --noise-rate {noise_rate} --leak-rate {leak_rate}"
    )
    yield prompt
    yield await _grade(graders)


@env.scenario(name="audit_evaluation_signal")
async def audit_evaluation_signal(
    prompt: str,
    graders: list[dict[str, Any]],
    eval_mutation: str = "eval_leakage",
    eval_file: str = "data/val.jsonl",
    train_file: str = "data/scifact.jsonl",
    leak_rate: float = 0.25,
    setup_command: str | None = None,
):
    """Corrupt the visible evaluation signal and require the agent to audit it."""
    _setup_workspace(setup_command)
    bash(
        f"python -m mutations eval {eval_mutation} {_workspace}"
        f" --eval-file {eval_file} --train-file {train_file}"
        f" --leak-rate {leak_rate}"
    )
    yield prompt
    yield await _grade(graders)


@env.scenario(name="compose_multi_stage_pipeline")
async def compose_multi_stage_pipeline(
    prompt: str,
    graders: list[dict[str, Any]],
    expected_stages: list[str] | None = None,
    setup_command: str | None = None,
):
    """Encourage staged training pipelines with intermediate artifacts."""
    _setup_workspace(setup_command)
    if expected_stages is not None:
        _write_tmp_json("pipeline_spec", {"expected_stages": expected_stages})
    yield prompt
    yield await _grade(graders)


@env.scenario(name="certify_reliability")
async def certify_reliability(
    prompt: str,
    graders: list[dict[str, Any]],
    reliability_matrix: list[dict[str, Any]],
    patches: list[str] | None = None,
    setup_command: str | None = None,
):
    """Require evidence that a recipe is stable across a small run matrix."""
    _setup_workspace(setup_command)
    _apply_patches(patches)
    _write_tmp_json("reliability_spec", {"matrix": reliability_matrix})
    yield prompt
    yield await _grade(graders)


@env.scenario(name="optimize_under_constraints")
async def optimize_under_constraints(
    prompt: str,
    graders: list[dict[str, Any]],
    constraints: dict[str, Any],
    setup_command: str | None = None,
):
    """Expose explicit resource or experiment-budget constraints to the agent."""
    _setup_workspace(setup_command)
    _write_tmp_json("constraint_spec", constraints)
    yield prompt
    yield await _grade(graders)


@env.scenario(name="adapt_without_forgetting")
async def adapt_without_forgetting(
    prompt: str,
    graders: list[dict[str, Any]],
    base_checkpoint: str,
    adapt_train_files: list[str],
    retain_eval_files: list[str],
    forbidden_train_files: list[str] | None = None,
    setup_command: str | None = None,
):
    """Stage a base checkpoint and require adaptation to new data without forgetting."""
    _setup_workspace(setup_command)
    for rel_path in forbidden_train_files or []:
        forbidden_path = Path(_workspace) / rel_path
        if forbidden_path.exists():
            forbidden_path.unlink()
    yield prompt
    yield await _grade(graders)


@env.scenario(name="targeted_failure_recovery")
async def targeted_failure_recovery(
    prompt: str,
    graders: list[dict[str, Any]],
    failure_manifest: dict[str, Any] | None = None,
    setup_command: str | None = None,
):
    """Stage failing artifacts or subsets and require targeted recovery ig."""
    _setup_workspace(setup_command)
    yield prompt
    yield await _grade(graders)


@env.scenario(name="restore_reference_parity")
async def restore_reference_parity(
    prompt: str,
    graders: list[dict[str, Any]],
    reference_spec: dict[str, Any],
    patches: list[str] | None = None,
    setup_command: str | None = None,
):
    """Stage a reference artifact or spec and require parity restoration."""
    _setup_workspace(setup_command)
    _apply_patches(patches)
    _write_tmp_json("reference_spec", reference_spec)
    yield prompt
    yield await _grade(graders)


SCENARIOS = {
    name: value
    for name, value in globals().items()
    if not name.startswith("_") and hasattr(value, "task")
}
