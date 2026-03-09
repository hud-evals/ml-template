"""Smoke test for the ML training environment.

Validates grading scripts, mutations, data contamination, and optionally
the training pipeline before deploying to Modal.

Usage:
    PYTHONPATH=. uv run python smoke_test.py                  # Quick (no GPU)
    PYTHONPATH=. uv run python smoke_test.py --training       # Full (needs GPU)
    PYTHONPATH=. uv run python smoke_test.py -k test_mutation # Run specific tests
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_check_script(script_content: str, args: list[str] | None = None) -> subprocess.CompletedProcess:
    """Write a check script to a temp file and execute it."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_content)
        f.flush()
        cmd = [sys.executable, f.name] + (args or [])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        os.unlink(f.name)
        return result


def _make_workspace(tmp_path: str) -> str:
    ws = os.path.join(tmp_path, "workspace")
    os.makedirs(os.path.join(ws, "data"), exist_ok=True)
    return ws


def _make_checkpoint(ws: str, name: str, stage: str, extra_meta: dict | None = None) -> str:
    ckpt_dir = os.path.join(ws, "checkpoints", name)
    os.makedirs(ckpt_dir, exist_ok=True)
    # Fake safetensors file
    with open(os.path.join(ckpt_dir, "model.safetensors"), "wb") as f:
        f.write(b"\x00" * 64)
    meta = {"stage": stage, "model_name": "test", **(extra_meta or {})}
    with open(os.path.join(ckpt_dir, "training_metadata.json"), "w") as f:
        json.dump(meta, f)
    return ckpt_dir


def _copy_source_file(ws: str, relative_path: str):
    """Copy a source file from the repo into a mock workspace."""
    src = os.path.join(REPO_ROOT, relative_path)
    dst = os.path.join(ws, relative_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def _make_training_data(ws: str, filename: str = "scifact.jsonl", n: int = 50) -> str:
    """Create minimal training data for testing."""
    path = os.path.join(ws, "data", filename)
    with open(path, "w") as f:
        for i in range(n):
            pair = {
                "instruction": "test",
                "query": f"query {i}",
                "positive": f"positive text {i}",
                "negatives": [f"negative {i}-{j}" for j in range(3)],
            }
            f.write(json.dumps(pair) + "\n")
    return path


# ===========================================================================
# Test: Check script generation (all scripts compile)
# ===========================================================================


class TestCheckScriptGeneration:
    """Verify all grading check scripts are valid Python."""

    def test_grader_eval_script(self):
        from grading.checks import grader_eval_script
        code = grader_eval_script("/code")
        compile(code, "<grader_eval>", "exec")

    def test_ndcg_check_script(self):
        from grading.checks import ndcg_check_script
        code = ndcg_check_script()
        compile(code, "<ndcg_check>", "exec")

    def test_metadata_check_script(self):
        from grading.checks import metadata_check_script
        code = metadata_check_script()
        compile(code, "<metadata_check>", "exec")

    def test_resume_check_script(self):
        from grading.checks import resume_check_script
        code = resume_check_script()
        compile(code, "<resume_check>", "exec")

    def test_merge_check_script(self):
        from grading.checks import merge_check_script
        code = merge_check_script()
        compile(code, "<merge_check>", "exec")

    def test_vlm_metadata_check_script(self):
        from grading.checks import vlm_metadata_check_script
        code = vlm_metadata_check_script()
        compile(code, "<vlm_metadata_check>", "exec")

    def test_vlm_checkpoint_check_script(self):
        from grading.checks import vlm_checkpoint_check_script
        code = vlm_checkpoint_check_script()
        compile(code, "<vlm_checkpoint_check>", "exec")

    def test_vlm_eval_script(self):
        from grading.checks import vlm_eval_script
        code = vlm_eval_script("/code")
        compile(code, "<vlm_eval>", "exec")

    def test_vlm_loss_check_script(self):
        from grading.checks import vlm_loss_check_script
        code = vlm_loss_check_script()
        compile(code, "<vlm_loss_check>", "exec")

    def test_code_fix_check_scripts(self):
        from grading.checks import code_fix_check_script
        for mutation in ["buggy_loss", "bad_pooling", "buggy_projector", "bad_label_mask"]:
            code = code_fix_check_script(mutation)
            compile(code, f"<code_fix_{mutation}>", "exec")

    def test_data_cleaned_check_script(self):
        from grading.checks import data_cleaned_check_script
        code = data_cleaned_check_script()
        compile(code, "<data_cleaned_check>", "exec")


# ===========================================================================
# Test: Metadata checks with mock workspaces
# ===========================================================================


class TestMetadataChecks:
    """Verify metadata check scripts pass/fail correctly with mock data."""

    def test_metadata_check_pass(self, tmp_path):
        from grading.checks import metadata_check_script
        ws = _make_workspace(str(tmp_path))
        _make_checkpoint(ws, "epoch_1", "finetune")

        result = _run_check_script(metadata_check_script(), ["finetune", ws])
        assert result.returncode == 0

    def test_metadata_check_fail_wrong_stage(self, tmp_path):
        from grading.checks import metadata_check_script
        ws = _make_workspace(str(tmp_path))
        _make_checkpoint(ws, "epoch_1", "pretrain")

        result = _run_check_script(metadata_check_script(), ["finetune", ws])
        assert result.returncode == 1

    def test_metadata_check_fail_no_checkpoint(self, tmp_path):
        from grading.checks import metadata_check_script
        ws = _make_workspace(str(tmp_path))

        result = _run_check_script(metadata_check_script(), ["finetune", ws])
        assert result.returncode == 1

    def test_resume_check_pass(self, tmp_path):
        from grading.checks import resume_check_script
        ws = _make_workspace(str(tmp_path))
        _make_checkpoint(ws, "epoch_1", "finetune", {"resume_from": "checkpoints/stage1"})

        result = _run_check_script(resume_check_script(), [ws])
        assert result.returncode == 0

    def test_resume_check_fail(self, tmp_path):
        from grading.checks import resume_check_script
        ws = _make_workspace(str(tmp_path))
        _make_checkpoint(ws, "epoch_1", "finetune")

        result = _run_check_script(resume_check_script(), [ws])
        assert result.returncode == 1

    def test_merge_check_pass(self, tmp_path):
        from grading.checks import merge_check_script
        ws = _make_workspace(str(tmp_path))
        merge_dir = os.path.join(ws, "checkpoints", "merged")
        os.makedirs(merge_dir, exist_ok=True)
        with open(os.path.join(merge_dir, "merge_metadata.json"), "w") as f:
            json.dump({"source_checkpoints": ["ckpt1", "ckpt2"]}, f)

        result = _run_check_script(merge_check_script(), [ws])
        assert result.returncode == 0

    def test_merge_check_fail_one_source(self, tmp_path):
        from grading.checks import merge_check_script
        ws = _make_workspace(str(tmp_path))
        merge_dir = os.path.join(ws, "checkpoints", "merged")
        os.makedirs(merge_dir, exist_ok=True)
        with open(os.path.join(merge_dir, "merge_metadata.json"), "w") as f:
            json.dump({"source_checkpoints": ["ckpt1"]}, f)

        result = _run_check_script(merge_check_script(), [ws])
        assert result.returncode == 1


# ===========================================================================
# Test: nDCG threshold checks
# ===========================================================================


class TestNDCGChecks:
    def test_ndcg_above_threshold(self, tmp_path):
        from grading.checks import ndcg_check_script
        ws = _make_workspace(str(tmp_path))
        cache = {"ndcg@10": 0.45, "best_dir": "/fake"}
        with open(os.path.join(ws, ".grader_eval.json"), "w") as f:
            json.dump(cache, f)

        result = _run_check_script(ndcg_check_script(), ["0.30", ws])
        assert result.returncode == 0

    def test_ndcg_below_threshold(self, tmp_path):
        from grading.checks import ndcg_check_script
        ws = _make_workspace(str(tmp_path))
        cache = {"ndcg@10": 0.05, "best_dir": "/fake"}
        with open(os.path.join(ws, ".grader_eval.json"), "w") as f:
            json.dump(cache, f)

        result = _run_check_script(ndcg_check_script(), ["0.30", ws])
        assert result.returncode == 1


# ===========================================================================
# Test: VLM checks
# ===========================================================================


class TestVLMChecks:
    def test_vlm_metadata_pass(self, tmp_path):
        from grading.checks import vlm_metadata_check_script
        ws = _make_workspace(str(tmp_path))
        ckpt = os.path.join(ws, "checkpoints", "vlm")
        os.makedirs(ckpt, exist_ok=True)
        with open(os.path.join(ckpt, "training_metadata.json"), "w") as f:
            json.dump({"task": "vlm_finetune", "steps": 100}, f)

        result = _run_check_script(vlm_metadata_check_script(), [ws])
        assert result.returncode == 0

    def test_vlm_metadata_fail(self, tmp_path):
        from grading.checks import vlm_metadata_check_script
        ws = _make_workspace(str(tmp_path))

        result = _run_check_script(vlm_metadata_check_script(), [ws])
        assert result.returncode == 1

    def test_vlm_checkpoint_safetensors(self, tmp_path):
        from grading.checks import vlm_checkpoint_check_script
        ws = _make_workspace(str(tmp_path))
        ckpt = os.path.join(ws, "checkpoints", "vlm")
        os.makedirs(ckpt, exist_ok=True)
        with open(os.path.join(ckpt, "model.safetensors"), "wb") as f:
            f.write(b"\x00" * 64)

        result = _run_check_script(vlm_checkpoint_check_script(), [ws])
        assert result.returncode == 0

    def test_vlm_checkpoint_dcp(self, tmp_path):
        from grading.checks import vlm_checkpoint_check_script
        ws = _make_workspace(str(tmp_path))
        ckpt = os.path.join(ws, "checkpoints", "vlm", "checkpoint", "step-100")
        os.makedirs(ckpt, exist_ok=True)
        with open(os.path.join(ckpt, ".__0_0.distcp"), "wb") as f:
            f.write(b"\x00" * 64)

        result = _run_check_script(vlm_checkpoint_check_script(), [ws])
        assert result.returncode == 0

    def test_vlm_checkpoint_none(self, tmp_path):
        from grading.checks import vlm_checkpoint_check_script
        ws = _make_workspace(str(tmp_path))

        result = _run_check_script(vlm_checkpoint_check_script(), [ws])
        assert result.returncode == 1

    def test_vlm_loss_pass(self, tmp_path):
        from grading.checks import vlm_loss_check_script
        ws = _make_workspace(str(tmp_path))
        with open(os.path.join(ws, ".vlm_eval.json"), "w") as f:
            json.dump({"val_loss": 1.5, "output_dir": "/fake"}, f)

        result = _run_check_script(vlm_loss_check_script(), ["2.0", ws])
        assert result.returncode == 0

    def test_vlm_loss_fail(self, tmp_path):
        from grading.checks import vlm_loss_check_script
        ws = _make_workspace(str(tmp_path))
        with open(os.path.join(ws, ".vlm_eval.json"), "w") as f:
            json.dump({"val_loss": 3.0, "output_dir": "/fake"}, f)

        result = _run_check_script(vlm_loss_check_script(), ["2.0", ws])
        assert result.returncode == 1


# ===========================================================================
# Test: Mutations
# ===========================================================================


class TestMutations:
    """Verify code mutations are applied correctly."""

    def _setup_embedding_workspace(self, tmp_path):
        ws = _make_workspace(str(tmp_path))
        for f in [
            "torchtitan/experiments/embedding/losses.py",
            "torchtitan/experiments/embedding/embedding_trainer.py",
            "torchtitan/experiments/embedding/datasets.py",
        ]:
            _copy_source_file(ws, f)
        return ws

    def _setup_vlm_workspace(self, tmp_path):
        ws = _make_workspace(str(tmp_path))
        for f in [
            "torchtitan/experiments/vlm/model/model.py",
            "torchtitan/experiments/vlm/datasets/mm_datasets.py",
        ]:
            _copy_source_file(ws, f)
        return ws

    def test_buggy_loss_mutation(self, tmp_path):
        ws = self._setup_embedding_workspace(tmp_path)
        losses_path = os.path.join(ws, "torchtitan/experiments/embedding/losses.py")

        original = open(losses_path).read()
        assert "F.normalize" in original

        # Apply mutation (inline to avoid workspace path issues)
        from env import _apply_code_mutation
        old_ws = os.environ.get("_SMOKE_WS")
        # Monkey-patch WORKSPACE for mutation
        import env
        saved = env.WORKSPACE
        env.WORKSPACE = ws
        _apply_code_mutation("buggy_loss")
        env.WORKSPACE = saved

        mutated = open(losses_path).read()
        assert "# Embeddings used as-is (no normalization)" in mutated
        assert "F.normalize(query_embeds" not in mutated

    def test_bad_pooling_mutation(self, tmp_path):
        ws = self._setup_embedding_workspace(tmp_path)
        trainer_path = os.path.join(ws, "torchtitan/experiments/embedding/embedding_trainer.py")

        original = open(trainer_path).read()
        assert "seq_lengths = attention_mask.sum" in original

        import env
        saved = env.WORKSPACE
        env.WORKSPACE = ws
        from env import _apply_code_mutation
        _apply_code_mutation("bad_pooling")
        env.WORKSPACE = saved

        mutated = open(trainer_path).read()
        assert "# Mean pooling over all tokens" in mutated

    def test_buggy_projector_mutation(self, tmp_path):
        ws = self._setup_vlm_workspace(tmp_path)
        model_path = os.path.join(ws, "torchtitan/experiments/vlm/model/model.py")

        original = open(model_path).read()
        assert "nn.functional.silu" in original

        import env
        saved = env.WORKSPACE
        env.WORKSPACE = ws
        from env import _apply_vlm_code_mutation
        _apply_vlm_code_mutation("buggy_projector")
        env.WORKSPACE = saved

        mutated = open(model_path).read()
        assert "# Linear projection (no nonlinearity)" in mutated
        assert "nn.functional.silu" not in mutated

    def test_bad_label_mask_mutation(self, tmp_path):
        ws = self._setup_vlm_workspace(tmp_path)
        ds_path = os.path.join(ws, "torchtitan/experiments/vlm/datasets/mm_datasets.py")

        original = open(ds_path).read()
        assert "Mask special tokens in labels" in original

        import env
        saved = env.WORKSPACE
        env.WORKSPACE = ws
        from env import _apply_vlm_code_mutation
        _apply_vlm_code_mutation("bad_label_mask")
        env.WORKSPACE = saved

        mutated = open(ds_path).read()
        assert "# Labels include all tokens (no masking)" in mutated
        assert "torch.isin(labels, special_token_ids)" not in mutated


# ===========================================================================
# Test: Code fix checks (verify they catch unfixed bugs)
# ===========================================================================


class TestCodeFixChecks:
    """Verify code fix check scripts detect fixed vs unfixed code."""

    def test_buggy_loss_unfixed_fails(self, tmp_path):
        from grading.checks import code_fix_check_script
        ws = _make_workspace(str(tmp_path))
        _copy_source_file(ws, "torchtitan/experiments/embedding/losses.py")

        # Apply the mutation
        import env
        saved = env.WORKSPACE
        env.WORKSPACE = ws
        from env import _apply_code_mutation
        _apply_code_mutation("buggy_loss")
        env.WORKSPACE = saved

        result = _run_check_script(code_fix_check_script("buggy_loss"), [ws])
        assert result.returncode == 1, f"Should fail on unfixed code: {result.stdout}"

    def test_buggy_loss_fixed_passes(self, tmp_path):
        from grading.checks import code_fix_check_script
        ws = _make_workspace(str(tmp_path))
        _copy_source_file(ws, "torchtitan/experiments/embedding/losses.py")

        result = _run_check_script(code_fix_check_script("buggy_loss"), [ws])
        assert result.returncode == 0, f"Should pass on original code: {result.stdout}"

    def test_bad_pooling_unfixed_fails(self, tmp_path):
        from grading.checks import code_fix_check_script
        ws = _make_workspace(str(tmp_path))
        _copy_source_file(ws, "torchtitan/experiments/embedding/embedding_trainer.py")

        import env
        saved = env.WORKSPACE
        env.WORKSPACE = ws
        from env import _apply_code_mutation
        _apply_code_mutation("bad_pooling")
        env.WORKSPACE = saved

        result = _run_check_script(code_fix_check_script("bad_pooling"), [ws])
        assert result.returncode == 1, f"Should fail on unfixed code: {result.stdout}"

    def test_bad_pooling_fixed_passes(self, tmp_path):
        from grading.checks import code_fix_check_script
        ws = _make_workspace(str(tmp_path))
        _copy_source_file(ws, "torchtitan/experiments/embedding/embedding_trainer.py")

        result = _run_check_script(code_fix_check_script("bad_pooling"), [ws])
        assert result.returncode == 0, f"Should pass on original code: {result.stdout}"

    def test_buggy_projector_unfixed_fails(self, tmp_path):
        from grading.checks import code_fix_check_script
        ws = _make_workspace(str(tmp_path))
        _copy_source_file(ws, "torchtitan/experiments/vlm/model/model.py")

        import env
        saved = env.WORKSPACE
        env.WORKSPACE = ws
        from env import _apply_vlm_code_mutation
        _apply_vlm_code_mutation("buggy_projector")
        env.WORKSPACE = saved

        result = _run_check_script(code_fix_check_script("buggy_projector"), [ws])
        assert result.returncode == 1, f"Should fail on unfixed code: {result.stdout}"

    def test_buggy_projector_fixed_passes(self, tmp_path):
        from grading.checks import code_fix_check_script
        ws = _make_workspace(str(tmp_path))
        _copy_source_file(ws, "torchtitan/experiments/vlm/model/model.py")

        result = _run_check_script(code_fix_check_script("buggy_projector"), [ws])
        assert result.returncode == 0, f"Should pass on original code: {result.stdout}"

    def test_bad_label_mask_unfixed_fails(self, tmp_path):
        from grading.checks import code_fix_check_script
        ws = _make_workspace(str(tmp_path))
        _copy_source_file(ws, "torchtitan/experiments/vlm/datasets/mm_datasets.py")

        import env
        saved = env.WORKSPACE
        env.WORKSPACE = ws
        from env import _apply_vlm_code_mutation
        _apply_vlm_code_mutation("bad_label_mask")
        env.WORKSPACE = saved

        result = _run_check_script(code_fix_check_script("bad_label_mask"), [ws])
        assert result.returncode == 1, f"Should fail on unfixed code: {result.stdout}"

    def test_bad_label_mask_fixed_passes(self, tmp_path):
        from grading.checks import code_fix_check_script
        ws = _make_workspace(str(tmp_path))
        _copy_source_file(ws, "torchtitan/experiments/vlm/datasets/mm_datasets.py")

        result = _run_check_script(code_fix_check_script("bad_label_mask"), [ws])
        assert result.returncode == 0, f"Should pass on original code: {result.stdout}"


# ===========================================================================
# Test: Data contamination
# ===========================================================================


class TestDataContamination:
    """Verify data contamination injection and detection."""

    def test_label_noise_modifies_data(self, tmp_path):
        from grading.mutations import inject_label_noise

        ws = _make_workspace(str(tmp_path))
        data_path = _make_training_data(ws, n=100)
        original = open(data_path).read()

        inject_label_noise(data_path, data_path, noise_rate=0.3)
        modified = open(data_path).read()

        assert original != modified, "Label noise should modify the data"

    def test_data_leakage_adds_samples(self, tmp_path):
        from grading.mutations import inject_data_leakage

        ws = _make_workspace(str(tmp_path))
        train_path = _make_training_data(ws, "train.jsonl", n=50)
        val_path = _make_training_data(ws, "val.jsonl", n=20)

        original_count = sum(1 for _ in open(train_path))
        output_path = os.path.join(ws, "data", "combined.jsonl")
        inject_data_leakage(train_path, val_path, output_path, leak_rate=0.5)

        new_count = sum(1 for _ in open(output_path))
        assert new_count > original_count, "Leakage should add samples"

    def test_duplicates_adds_copies(self, tmp_path):
        from grading.mutations import inject_duplicates

        ws = _make_workspace(str(tmp_path))
        data_path = _make_training_data(ws, n=50)
        original_count = sum(1 for _ in open(data_path))

        inject_duplicates(data_path, data_path, dup_rate=0.3)
        new_count = sum(1 for _ in open(data_path))

        assert new_count > original_count, "Duplicates should increase sample count"

    def test_data_cleaned_check_detects_unchanged(self, tmp_path):
        from grading.checks import data_cleaned_check_script

        ws = _make_workspace(str(tmp_path))
        data_path = _make_training_data(ws, n=50)

        # Simulate contamination info saved during setup
        info = {
            "type": "label_noise",
            "hash": hashlib.md5(open(data_path, "rb").read()).hexdigest(),
            "line_count": 50,
        }
        with open("/tmp/contamination_info.json", "w") as f:
            json.dump(info, f)

        result = _run_check_script(data_cleaned_check_script(), [ws])
        assert result.returncode == 1, f"Should fail when data unchanged: {result.stdout}"

    def test_data_cleaned_check_passes_on_modified(self, tmp_path):
        from grading.checks import data_cleaned_check_script

        ws = _make_workspace(str(tmp_path))
        data_path = _make_training_data(ws, n=50)

        info = {
            "type": "label_noise",
            "hash": "deadbeef00000000",  # Different from actual
            "line_count": 50,
        }
        with open("/tmp/contamination_info.json", "w") as f:
            json.dump(info, f)

        result = _run_check_script(data_cleaned_check_script(), [ws])
        assert result.returncode == 0, f"Should pass when data modified: {result.stdout}"


# ===========================================================================
# Test: Task weight sanity
# ===========================================================================


class TestTaskWeights:
    """Verify all task grading weights sum to ~1.0."""

    def _load_task_graders(self, task_name: str) -> list[dict]:
        """Extract grader weights from a task file."""
        import re
        task_path = os.path.join(REPO_ROOT, "tasks", task_name, "task.py")
        with open(task_path) as f:
            content = f.read()

        weights = []

        # Zip-spread: zip(CHECKS, [0.10, 0.15, 0.15])
        for match in re.findall(r'zip\(.*?,\s*\[([\d.,\s]+)\]', content):
            weights.extend(float(w.strip()) for w in match.split(",") if w.strip())

        # Spread with fixed weight: {**c, "weight": 0.05} for c in VARNAME
        # Count how many items VARNAME has by finding its definition
        for w_str, varname in re.findall(
            r'\{\*\*c,\s*"weight":\s*([\d.]+)\}\s+for\s+c\s+in\s+(\w+)', content
        ):
            count = self._count_list_var(content, task_name, varname)
            weights.extend([float(w_str)] * count)

        # Direct inline weights (not inside a spread comprehension)
        for line in content.split("\n"):
            if "for c in" in line or "for c," in line:
                continue
            for w in re.findall(r'"weight":\s*([\d.]+)', line):
                weights.append(float(w))

        return weights

    def _count_list_var(self, content: str, task_name: str, varname: str) -> int:
        """Count items in a list variable, resolving imports if needed."""
        import re

        # Check local definition (greedy to handle nested brackets)
        pattern = rf'^{varname}\s*=\s*\[(.*?)\n\]'
        match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
        if match:
            return len(re.findall(r'\{"name"', match.group(1)))

        # Check imports: from tasks.X.task import CHECKS as VARNAME
        import_match = re.search(
            rf'from\s+([\w.]+)\s+import\s+.*?(\w+)\s+as\s+{varname}', content
        )
        if not import_match:
            import_match = re.search(
                rf'from\s+([\w.]+)\s+import\s+.*?{varname}', content
            )
        if import_match:
            module_path = import_match.group(1).replace(".", "/")
            imported_file = os.path.join(REPO_ROOT, module_path + ".py")
            if os.path.exists(imported_file):
                with open(imported_file) as f:
                    imported_content = f.read()
                return self._count_list_var(imported_content, "", varname)

        return 1  # fallback

    @pytest.mark.parametrize("task_name", [
        "pretrain_embedding",
        "finetune_embedding",
        "merge_and_evaluate",
        "vlm_finetune",
        "debug_embedding_loss",
        "debug_embedding_pooling",
        "debug_vlm_labels",
        "debug_vlm_projector",
        "data_audit_leakage",
        "data_audit_noise",
    ])
    def test_weights_sum_to_one(self, task_name):
        weights = self._load_task_graders(task_name)
        assert len(weights) > 0, f"No weights found in {task_name}"
        total = sum(weights)
        assert abs(total - 1.0) < 0.02, f"{task_name}: weights sum to {total}, expected ~1.0"


# ===========================================================================
# Test: Training pipeline (requires GPU)
# ===========================================================================


def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.skipif(not _has_gpu(), reason="No GPU available")
class TestEmbeddingTraining:
    """Minimal embedding training smoke test (requires GPU)."""

    def test_embedding_train_finetune(self, tmp_path):
        ws = str(tmp_path / "workspace")
        os.makedirs(f"{ws}/data", exist_ok=True)

        # Create tiny training data
        _make_training_data(ws, n=20)

        env = {**os.environ, "PYTHONPATH": REPO_ROOT}
        result = subprocess.run(
            [
                sys.executable, "-m", "torchtitan.experiments.embedding.train",
                "--stage", "finetune",
                "--train_data", f"{ws}/data/scifact.jsonl",
                "--output_dir", f"{ws}/checkpoints/stage1",
                "--epochs", "1",
                "--batch_size", "2",
                "--max_seq_length", "64",
                "--num_hard_negatives", "2",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, f"Training failed:\n{result.stderr[-2000:]}"

        # Verify checkpoint was created
        ckpts = list((tmp_path / "workspace" / "checkpoints" / "stage1").glob("**/model.safetensors"))
        assert len(ckpts) > 0, "No checkpoint produced"

        # Verify metadata
        meta_files = list((tmp_path / "workspace" / "checkpoints" / "stage1").glob("**/training_metadata.json"))
        assert len(meta_files) > 0, "No training metadata produced"
        meta = json.loads(meta_files[0].read_text())
        assert meta["stage"] == "finetune"


@pytest.mark.skipif(not _has_gpu(), reason="No GPU available")
class TestVLMTraining:
    """Minimal VLM training smoke test (requires GPU)."""

    def test_vlm_train(self, tmp_path):
        ws = str(tmp_path / "workspace")
        os.makedirs(f"{ws}/data/cc12m", exist_ok=True)

        # Copy test assets
        shutil.copytree(
            os.path.join(REPO_ROOT, "tests/assets/cc12m_test"),
            f"{ws}/data/cc12m",
            dirs_exist_ok=True,
        )
        shutil.copytree(
            os.path.join(REPO_ROOT, "tests/assets/tokenizer"),
            f"{ws}/tokenizer",
        )

        env = {**os.environ, "PYTHONPATH": ws}
        # Copy torchtitan to workspace for PYTHONPATH
        shutil.copytree(
            os.path.join(REPO_ROOT, "torchtitan"),
            f"{ws}/torchtitan",
        )

        result = subprocess.run(
            [
                sys.executable, "-m", "torchtitan.experiments.vlm.train",
                "--dataset", "cc12m-test",
                "--data_path", f"{ws}/data/cc12m",
                "--tokenizer_path", f"{ws}/tokenizer",
                "--output_dir", f"{ws}/checkpoints/vlm",
                "--steps", "5",
                "--batch_size", "2",
            ],
            cwd=ws,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, f"VLM training failed:\n{result.stderr[-2000:]}"

        # Verify metadata
        meta_path = os.path.join(ws, "checkpoints/vlm/training_metadata.json")
        assert os.path.exists(meta_path), "No VLM training metadata"
        meta = json.loads(open(meta_path).read())
        assert meta["task"] == "vlm_finetune"


# ===========================================================================
# Entrypoint
# ===========================================================================


if __name__ == "__main__":
    args = sys.argv[1:]

    # Default: skip GPU tests unless --training flag
    if "--training" in args:
        args.remove("--training")
    else:
        if "-k" not in args:
            args.extend(["-k", "not Training"])

    pytest.main([__file__, "-v", "--tb=short"] + args)
