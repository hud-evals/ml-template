"""Sanity tests for all task definitions.

Auto-discovers tasks the same way tasks/__init__.py does, then validates
structural invariants per scenario family.
"""

import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

import pytest

from hud.eval.task import Task
from env import SCENARIOS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TASKS_PKG = REPO_ROOT / "tasks"
GRADERS_DIR = TASKS_PKG / "graders"
GRADER_SCRIPTS = {p.stem for p in GRADERS_DIR.glob("*.py") if p.name != "__init__.py"}
MAX_SLUG_LENGTH = 20

# ---------------------------------------------------------------------------
# Auto-discover every task package (mirrors tasks/__init__.py logic)
# ---------------------------------------------------------------------------

_SKIP_PKGS = {"calibration", "graders", "tests"}
_discovered: dict[str, Task] = {}

import tasks as _tasks_pkg  # noqa: E402

for _info in pkgutil.iter_modules(_tasks_pkg.__path__, _tasks_pkg.__name__ + "."):
    if not _info.ispkg:
        continue
    _pkg = _info.name.rsplit(".", 1)[-1]
    if _pkg in _SKIP_PKGS:
        continue
    for _target in [_info.name, f"{_info.name}.task"]:
        try:
            _mod = importlib.import_module(_target)
        except Exception:
            continue
        for _attr in vars(_mod).values():
            if isinstance(_attr, Task):
                _discovered[_pkg] = _attr
                break
        if _pkg in _discovered:
            break

ALL_SLUGS = list(_discovered.keys())


# ---------------------------------------------------------------------------
# Basic task invariants
# ---------------------------------------------------------------------------


class TestTaskDiscovery:
    def test_at_least_one_task(self):
        assert len(_discovered) > 0, "No tasks discovered"

    @pytest.mark.parametrize("slug", ALL_SLUGS)
    def test_has_slug(self, slug: str):
        assert _discovered[slug].slug == slug

    @pytest.mark.parametrize("slug", ALL_SLUGS)
    def test_slug_length(self, slug: str):
        assert len(slug) <= MAX_SLUG_LENGTH, (
            f"'{slug}' is {len(slug)} chars (max {MAX_SLUG_LENGTH})"
        )

    @pytest.mark.parametrize("slug", ALL_SLUGS)
    def test_has_scenario(self, slug: str):
        assert _discovered[slug].id in SCENARIOS


# ---------------------------------------------------------------------------
# Grader integrity
# ---------------------------------------------------------------------------


def _graders_for(slug: str) -> list[dict]:
    return _discovered[slug].args.get("graders", [])


def _script_graders(slug: str) -> list[dict]:
    return [g for g in _graders_for(slug) if "_script_stem" in g]


class TestGraderIntegrity:
    @pytest.mark.parametrize("slug", ALL_SLUGS)
    def test_has_graders(self, slug: str):
        assert len(_graders_for(slug)) > 0, f"{slug}: no graders defined"

    @pytest.mark.parametrize("slug", ALL_SLUGS)
    def test_grader_scripts_exist(self, slug: str):
        """Every script-based grader must reference a file in tasks/graders/."""
        for g in _script_graders(slug):
            stem = g["_script_stem"]
            assert stem in GRADER_SCRIPTS, (
                f"{slug}: grader references '{stem}.py' which doesn't exist in tasks/graders/"
            )

    @pytest.mark.parametrize("slug", ALL_SLUGS)
    def test_grader_weights_sum_to_one(self, slug: str):
        total = sum(g.get("weight", 1) for g in _graders_for(slug))
        assert abs(total - 1.0) < 0.02, (
            f"{slug}: grader weights sum to {total:.3f}, expected ~1.0"
        )

    @pytest.mark.parametrize("slug", ALL_SLUGS)
    def test_grader_names_unique(self, slug: str):
        names = [g["name"] for g in _graders_for(slug)]
        assert len(names) == len(set(names)), (
            f"{slug}: duplicate grader names: {[n for n in names if names.count(n) > 1]}"
        )


# ---------------------------------------------------------------------------
# Scenario-specific invariants
# ---------------------------------------------------------------------------


def _slugs_for_scenario(name: str) -> list[str]:
    return [s for s in ALL_SLUGS if _discovered[s].id == name]


def _repair_slugs():
    return _slugs_for_scenario("repair_degraded_recipe")


def _audit_training_slugs():
    return _slugs_for_scenario("audit_training_data")


def _reliability_slugs():
    return _slugs_for_scenario("certify_reliability")


def _pipeline_slugs():
    return _slugs_for_scenario("compose_multi_stage_pipeline")


def _audit_eval_slugs():
    return _slugs_for_scenario("audit_evaluation_signal")


def _constraint_slugs():
    return _slugs_for_scenario("optimize_under_constraints")


def _adaptation_slugs():
    return _slugs_for_scenario("adapt_without_forgetting")


def _targeted_recovery_slugs():
    return _slugs_for_scenario("targeted_failure_recovery")


def _parity_slugs():
    return _slugs_for_scenario("restore_reference_parity")


class TestGraderArgContracts:
    """Validate that grader args match the expected format of each grader script."""

    @pytest.mark.parametrize("slug", ALL_SLUGS)
    def test_eval_has_valid_type(self, slug: str):
        """Unified eval grader must have a valid type as first arg."""
        valid_types = {"emb", "vlm", "flux", "moe"}
        for g in _script_graders(slug):
            if g["_script_stem"] != "eval":
                continue
            args = g["args"].split()
            assert len(args) >= 2, (
                f"{slug}: eval grader needs '<type> ...' args, got: {g['args']}"
            )
            assert args[0] in valid_types, (
                f"{slug}: eval type '{args[0]}' not in {valid_types}"
            )

    @pytest.mark.parametrize("slug", ALL_SLUGS)
    def test_threshold_cache_files_have_producers(self, slug: str):
        """Every cache file read by check_threshold must be written by an earlier grader."""
        graders = _graders_for(slug)
        produced_cache_files: set[str] = set()

        def _extract_cache_file(eval_type: str, args: list[str]) -> str | None:
            if eval_type == "emb":
                # emb mteb <tasks> <cache> <ws> or emb local <data> <cache> <ws>
                return args[2] if len(args) >= 4 else None
            elif eval_type in ("vlm", "flux"):
                # vlm <cache> <ws> or flux <cache> <ws>
                return args[0] if args else None
            elif eval_type == "moe":
                # moe <slug> <cache> <ws>
                return args[1] if len(args) >= 3 else None
            return None

        for g in graders:
            stem = g.get("_script_stem", "")
            if stem == "eval":
                args = g.get("args", "").split()
                if args:
                    cache = _extract_cache_file(args[0], args[1:])
                    if cache:
                        produced_cache_files.add(cache)

            if stem == "check_threshold":
                args = g.get("args", "").split()
                if args:
                    cache_file = args[0]
                    assert cache_file in produced_cache_files, (
                        f"{slug}: check_threshold reads '{cache_file}' but no "
                        f"preceding eval grader produces it "
                        f"(produced: {produced_cache_files})"
                    )


class TestRepairTasks:
    @pytest.mark.parametrize("slug", _repair_slugs())
    def test_has_patches(self, slug: str):
        patches = _discovered[slug].args.get("patches", [])
        assert len(patches) > 0, f"{slug}: repair task must have patches"

    @pytest.mark.parametrize("slug", _repair_slugs())
    def test_patch_files_exist(self, slug: str):
        task_dir = TASKS_PKG / slug
        patch_files = sorted(task_dir.glob("*.patch"))
        assert len(patch_files) > 0, f"{slug}: no .patch files in {task_dir}"

    @pytest.mark.parametrize("slug", _repair_slugs())
    def test_patches_match_files(self, slug: str):
        """Number of loaded patches matches the number of .patch files on disk."""
        patches = _discovered[slug].args.get("patches", [])
        task_dir = TASKS_PKG / slug
        patch_files = sorted(task_dir.glob("*.patch"))
        assert len(patches) == len(patch_files), (
            f"{slug}: loaded {len(patches)} patches but found "
            f"{len(patch_files)} .patch files"
        )

    @pytest.mark.parametrize("slug", _repair_slugs())
    def test_patches_apply(self, slug: str):
        """Every patch must apply cleanly against the source tree."""
        _assert_patches_apply(slug)


def _slugs_with_patches() -> list[str]:
    return [s for s in ALL_SLUGS if _discovered[s].args.get("patches")]


class TestPatchesApply:
    @pytest.mark.parametrize("slug", _slugs_with_patches())
    def test_patches_apply(self, slug: str):
        """Every patch in any task must apply cleanly against the source tree."""
        _assert_patches_apply(slug)


def _assert_patches_apply(slug: str) -> None:
    import subprocess

    patches = _discovered[slug].args.get("patches", [])
    for i, patch_content in enumerate(patches):
        result = subprocess.run(
            ["patch", "-p1", "--dry-run"],
            input=patch_content,
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, (
            f"{slug}: patch {i} failed to apply:\n{result.stderr}"
        )


class TestAuditTrainingTasks:
    @pytest.mark.parametrize("slug", _audit_training_slugs())
    def test_has_contamination_type(self, slug: str):
        contamination = _discovered[slug].args.get("contamination")
        assert contamination, f"{slug}: audit task must specify contamination type"

    @pytest.mark.parametrize("slug", _audit_training_slugs())
    def test_contamination_type_is_valid(self, slug: str):
        from tasks.mutations.data import VALID_DATA_MUTATIONS
        contamination = _discovered[slug].args["contamination"]
        assert contamination in VALID_DATA_MUTATIONS, (
            f"{slug}: unknown contamination type '{contamination}'"
        )

    @pytest.mark.parametrize("slug", _audit_training_slugs())
    def test_has_data_cleaned_grader(self, slug: str):
        stems = {g["_script_stem"] for g in _script_graders(slug)}
        assert "check_data_cleaned" in stems, (
            f"{slug}: audit task should have a check_data_cleaned grader"
        )

    @pytest.mark.parametrize("slug", _audit_training_slugs())
    def test_has_audit_provenance_grader(self, slug: str):
        stems = {g["_script_stem"] for g in _script_graders(slug)}
        assert "check_audit_provenance" in stems, (
            f"{slug}: audit task should have a check_audit_provenance grader"
        )

    @pytest.mark.parametrize("slug", _audit_training_slugs())
    def test_rates_in_range(self, slug: str):
        args = _discovered[slug].args
        for key in ("noise_rate", "leak_rate"):
            if key in args:
                assert 0 < args[key] < 1, (
                    f"{slug}: {key}={args[key]} out of (0, 1)"
                )


class TestAuditEvaluationTasks:
    @pytest.mark.parametrize("slug", _audit_eval_slugs())
    def test_has_eval_mutation(self, slug: str):
        eval_mutation = _discovered[slug].args.get("eval_mutation")
        assert eval_mutation, f"{slug}: audit-eval task must specify eval_mutation"

    @pytest.mark.parametrize("slug", _audit_eval_slugs())
    def test_eval_mutation_is_valid(self, slug: str):
        from tasks.mutations.eval import VALID_EVAL_MUTATIONS

        eval_mutation = _discovered[slug].args["eval_mutation"]
        assert eval_mutation in VALID_EVAL_MUTATIONS, (
            f"{slug}: unknown eval mutation '{eval_mutation}'"
        )

    @pytest.mark.parametrize("slug", _audit_eval_slugs())
    def test_has_eval_cleaned_grader(self, slug: str):
        stems = {g["_script_stem"] for g in _script_graders(slug)}
        assert "check_eval_cleaned" in stems, (
            f"{slug}: audit-eval task should have a check_eval_cleaned grader"
        )

class TestReliabilityTasks:
    @pytest.mark.parametrize("slug", _reliability_slugs())
    def test_has_reliability_matrix(self, slug: str):
        matrix = _discovered[slug].args.get("reliability_matrix", [])
        assert matrix, f"{slug}: reliability task must define a reliability_matrix"

    @pytest.mark.parametrize("slug", _reliability_slugs())
    def test_reliability_matrix_rows_are_named(self, slug: str):
        matrix = _discovered[slug].args.get("reliability_matrix", [])
        assert all(isinstance(row.get("name"), str) and row["name"] for row in matrix), (
            f"{slug}: each reliability_matrix row should have a non-empty name"
        )


class TestPipelineTasks:
    @pytest.mark.parametrize("slug", _pipeline_slugs())
    def test_has_expected_stages(self, slug: str):
        expected_stages = _discovered[slug].args.get("expected_stages", [])
        assert expected_stages, f"{slug}: pipeline task should declare expected_stages"


class TestConstraintTasks:
    @pytest.mark.parametrize("slug", _constraint_slugs())
    def test_has_constraints(self, slug: str):
        constraints = _discovered[slug].args.get("constraints", {})
        assert constraints, f"{slug}: constraint task should declare constraints"

    @pytest.mark.parametrize("slug", _constraint_slugs())
    def test_has_budget_enforcement_grader(self, slug: str):
        stems = {g["_script_stem"] for g in _script_graders(slug)}
        assert "check_step_budget" in stems, (
            f"{slug}: constraint task should have a check_step_budget grader"
        )


class TestAdaptationTasks:
    @pytest.mark.parametrize("slug", _adaptation_slugs())
    def test_has_base_checkpoint(self, slug: str):
        base_checkpoint = _discovered[slug].args.get("base_checkpoint")
        assert base_checkpoint, f"{slug}: adaptation task should declare base_checkpoint"

    @pytest.mark.parametrize("slug", _adaptation_slugs())
    def test_has_adapt_train_files(self, slug: str):
        adapt_train_files = _discovered[slug].args.get("adapt_train_files", [])
        assert adapt_train_files, f"{slug}: adaptation task should declare adapt_train_files"

    @pytest.mark.parametrize("slug", _adaptation_slugs())
    def test_has_retain_eval_files(self, slug: str):
        retain_eval_files = _discovered[slug].args.get("retain_eval_files", [])
        assert retain_eval_files, f"{slug}: adaptation task should declare retain_eval_files"


class TestTargetedRecoveryTasks:
    @pytest.mark.parametrize("slug", _targeted_recovery_slugs())
    def test_has_failure_manifest(self, slug: str):
        failure_manifest = _discovered[slug].args.get("failure_manifest", {})
        assert failure_manifest, f"{slug}: targeted recovery task should declare failure_manifest"


class TestParityTasks:
    @pytest.mark.parametrize("slug", _parity_slugs())
    def test_has_reference_spec(self, slug: str):
        reference_spec = _discovered[slug].args.get("reference_spec", {})
        assert reference_spec, f"{slug}: parity task should declare reference_spec"


_TRAIN_AUDIT_FIXED = {"prompt", "graders", "setup_command", "contamination"}
_TRAIN_CONTAMINATE_FIXED = {"workspace", "mutation"}
_EVAL_AUDIT_FIXED = {"prompt", "graders", "setup_command", "eval_mutation"}
_EVAL_CONTAMINATE_FIXED = {"workspace", "mutation"}


class TestAuditScenarioContracts:
    def test_training_audit_kwargs_match_contaminate(self):
        from env import audit_training_data
        from tasks.mutations.data import contaminate

        scenario_params = inspect.signature(audit_training_data).parameters
        contaminate_params = inspect.signature(contaminate).parameters

        scenario_kwargs = {
            k: v.default
            for k, v in scenario_params.items()
            if k not in _TRAIN_AUDIT_FIXED
        }
        contaminate_kwargs = {
            k: v.default
            for k, v in contaminate_params.items()
            if k not in _TRAIN_CONTAMINATE_FIXED
        }
        assert "contamination" in scenario_params
        assert "mutation" in contaminate_params
        assert contaminate_params["mutation"].default is inspect._empty
        assert scenario_kwargs == contaminate_kwargs, (
            f"audit_training_data kwargs {scenario_kwargs} != "
            f"contaminate() kwargs {contaminate_kwargs}"
        )

    def test_eval_audit_kwargs_match_contaminate(self):
        from env import audit_evaluation_signal
        from tasks.mutations.eval import contaminate_eval_signal

        scenario_params = inspect.signature(audit_evaluation_signal).parameters
        contaminate_params = inspect.signature(contaminate_eval_signal).parameters

        scenario_kwargs = {
            k: v.default
            for k, v in scenario_params.items()
            if k not in _EVAL_AUDIT_FIXED
        }
        contaminate_kwargs = {
            k: v.default
            for k, v in contaminate_params.items()
            if k not in _EVAL_CONTAMINATE_FIXED
        }
        assert "eval_mutation" in scenario_params
        assert "mutation" in contaminate_params
        assert contaminate_params["mutation"].default is inspect._empty
        assert scenario_kwargs == contaminate_kwargs, (
            f"audit_evaluation_signal kwargs {scenario_kwargs} != "
            f"contaminate_eval_signal() kwargs {contaminate_kwargs}"
        )


class TestDockerfileCompleteness:
    """Verify the Dockerfile copies all files that runtime code references."""

    def test_setup_fixtures_dependencies_in_dockerfile(self):
        """Files imported by setup_fixtures must be copied into the Docker image."""
        import ast
        import re

        dockerfile = (REPO_ROOT / "Dockerfile.hud").read_text()
        fixtures_src = (REPO_ROOT / "tasks" / "utils" / "setup_fixtures.py").read_text()

        # Extract file paths referenced in setup_fixtures
        # Look for paths like "scripts/checkpoint_conversion/convert_to_hf.py"
        path_refs = re.findall(
            r'["\']([a-zA-Z_/]+\.py)["\']', fixtures_src
        )
        # Filter to paths that look like repo-relative scripts
        script_refs = [p for p in path_refs if p.startswith("scripts/")]

        for script in script_refs:
            # Check if the Dockerfile copies this file or its parent directory
            script_dir = "/".join(script.split("/")[:-1])
            assert (
                script in dockerfile
                or script_dir in dockerfile
                or f"./{script}" in dockerfile
                or f"./{script_dir}" in dockerfile
            ), (
                f"setup_fixtures.py references '{script}' but it's not "
                f"copied in Dockerfile.hud"
            )

    def test_all_copied_source_files_exist(self):
        """Every file/directory COPYed in the Dockerfile must exist in the repo."""
        import re

        dockerfile = (REPO_ROOT / "Dockerfile.hud").read_text()
        for match in re.finditer(r"COPY\s+\./(\S+)", dockerfile):
            src = match.group(1)
            path = REPO_ROOT / src
            assert path.exists(), (
                f"Dockerfile copies './{src}' but it doesn't exist at {path}"
            )

    def test_tasks_copied_as_package(self):
        """Template Dockerfile should not hardcode individual task packages."""
        import re

        dockerfile = (REPO_ROOT / "Dockerfile.hud").read_text()
        task_copies = [
            match.group(1)
            for match in re.finditer(r"COPY\s+\./(tasks/?\S*)", dockerfile)
        ]

        assert task_copies, "Dockerfile must copy the tasks package"
        assert task_copies == ["tasks/"], (
            "Dockerfile should copy './tasks/' once, not hardcode individual "
            f"task subdirectories: {task_copies}"
        )


class TestSourceFilesClean:
    """Verify source files haven't been left in a patched state."""

    @pytest.mark.parametrize("slug", _slugs_with_patches())
    def test_patches_apply_cleanly(self, slug: str):
        """Patches must apply -- source files must not be in a patched state."""
        import subprocess

        patches = _discovered[slug].args.get("patches", [])
        for i, patch_content in enumerate(patches):
            result = subprocess.run(
                ["patch", "-p1", "--dry-run"],
                input=patch_content,
                text=True,
                capture_output=True,
                cwd=REPO_ROOT,
            )
            assert result.returncode == 0, (
                f"{slug}: patch {i} doesn't apply. "
                f"Source file may be in a patched state.\n{result.stdout}"
            )

    @pytest.mark.parametrize("slug", _slugs_with_patches())
    def test_no_orig_files(self, slug: str):
        """No .orig backup files should exist for patched source files."""
        import re

        patches = _discovered[slug].args.get("patches", [])
        for patch_content in patches:
            for match in re.finditer(r"^--- a/(.+)$", patch_content, re.MULTILINE):
                src_file = REPO_ROOT / match.group(1)
                orig_file = src_file.with_suffix(src_file.suffix + ".orig")
                assert not orig_file.exists(), (
                    f"Backup file exists: {orig_file} -- "
                    f"source was likely patched and not fully reverted"
                )


class TestScenarioSignatures:
    def test_train_to_target_signature(self):
        from env import train_to_target

        params = inspect.signature(train_to_target).parameters
        assert list(params) == ["prompt", "graders", "setup_command"]
        assert params["setup_command"].default is None

    def test_optimize_under_constraints_signature(self):
        from env import optimize_under_constraints

        params = inspect.signature(optimize_under_constraints).parameters
        assert "constraints" in params
        assert params["constraints"].default is inspect._empty
        assert params["setup_command"].default is None

    def test_adapt_without_forgetting_signature(self):
        from env import adapt_without_forgetting

        params = inspect.signature(adapt_without_forgetting).parameters
        for required in ("base_checkpoint", "adapt_train_files", "retain_eval_files"):
            assert required in params
            assert params[required].default is inspect._empty
        assert params["forbidden_train_files"].default is None
        assert params["setup_command"].default is None

    def test_targeted_failure_recovery_signature(self):
        from env import targeted_failure_recovery

        params = inspect.signature(targeted_failure_recovery).parameters
        assert "failure_manifest" in params
        assert params["failure_manifest"].default is None
        assert params["setup_command"].default is None

    def test_restore_reference_parity_signature(self):
        from env import restore_reference_parity

        params = inspect.signature(restore_reference_parity).parameters
        assert "reference_spec" in params
        assert params["reference_spec"].default is inspect._empty
        assert params["patches"].default is None
        assert params["setup_command"].default is None
