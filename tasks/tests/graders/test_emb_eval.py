import json
import os
import subprocess
import sys
from pathlib import Path

from ..conftest import GRADERS_DIR, make_checkpoint, make_training_data, make_workspace


def _fake_src_tree(root: Path) -> None:
    evaluate_py = root / "torchtitan" / "experiments" / "embedding" / "evaluate.py"
    evaluate_py.parent.mkdir(parents=True, exist_ok=True)
    evaluate_py.write_text(
        "def evaluate_local(model_dir, eval_data, max_seq_length=512):\n"
        "    if eval_data.endswith('nq_val.jsonl'):\n"
        "        return {'ndcg@10': 0.42}\n"
        "    return {'ndcg@10': 0.31}\n"
        "\n"
        "def evaluate_mteb(model_dir, tasks, max_seq_length=512):\n"
        "    return {'ndcg@10': 0.55}\n"
    )
    for pkg in [
        root / "torchtitan" / "__init__.py",
        root / "torchtitan" / "experiments" / "__init__.py",
        root / "torchtitan" / "experiments" / "embedding" / "__init__.py",
    ]:
        pkg.parent.mkdir(parents=True, exist_ok=True)
        pkg.write_text("")


class TestCheckpointDiscovery:
    def test_excludes_assets_dir(self, tmp_path):
        """Checkpoints in assets/ should not be discovered."""
        ws = make_workspace(str(tmp_path))
        assets_ckpt = os.path.join(ws, "assets", "hf", "SomeModel")
        os.makedirs(assets_ckpt, exist_ok=True)
        with open(os.path.join(assets_ckpt, "model.safetensors"), "wb") as f:
            f.write(b"\x00" * 64)

        fake_src = tmp_path / "src"
        _fake_src_tree(fake_src)
        result = subprocess.run(
            [sys.executable, str(GRADERS_DIR / "eval.py"), "emb", "mteb", "SciFact", ".emb_eval.json", ws],
            env={**os.environ, "SRC_DIR": str(fake_src)},
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0
        assert "No model checkpoints found" in result.stdout

    def test_finds_hf_checkpoint(self, tmp_path):
        """HF checkpoints (model.safetensors not in assets/) should be discovered."""
        ws = make_workspace(str(tmp_path))
        make_checkpoint(ws, "epoch_1", "finetune")
        fake_src = tmp_path / "src"
        _fake_src_tree(fake_src)

        result = subprocess.run(
            [sys.executable, str(GRADERS_DIR / "eval.py"), "emb", "mteb", "SciFact", ".emb_eval.json", ws],
            env={**os.environ, "SRC_DIR": str(fake_src)},
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "No model checkpoints found" not in result.stdout

    def test_finds_dcp_checkpoint(self, tmp_path):
        """DCP checkpoints (step-N/.metadata) should be discovered."""
        ws = make_workspace(str(tmp_path))
        dcp_dir = os.path.join(ws, "outputs", "checkpoint", "step-100")
        os.makedirs(dcp_dir, exist_ok=True)
        with open(os.path.join(dcp_dir, ".metadata"), "w") as f:
            f.write("{}")

        fake_src = tmp_path / "src"
        _fake_src_tree(fake_src)
        result = subprocess.run(
            [sys.executable, str(GRADERS_DIR / "eval.py"), "emb", "mteb", "SciFact", ".emb_eval.json", ws],
            env={**os.environ, "SRC_DIR": str(fake_src)},
            capture_output=True, text=True, timeout=30,
        )
        # DCP is discovered (conversion attempted). Conversion may fail
        # locally without GPU/triton, but that's expected.
        assert "DCP conversion failed" in result.stdout or "Converted DCP" in result.stdout


class TestEmbEval:
    def test_mteb_writes_cache(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        make_checkpoint(ws, "epoch_1", "finetune")
        fake_src = tmp_path / "src"
        _fake_src_tree(fake_src)

        result = subprocess.run(
            [sys.executable, str(GRADERS_DIR / "eval.py"), "emb", "mteb", "SciFact", ".emb_eval.json", ws],
            env={**os.environ, "SRC_DIR": str(fake_src)},
            capture_output=True, text=True, timeout=30,
        )

        assert result.returncode == 0, result.stderr
        cache = json.loads(Path(ws, ".emb_eval.json").read_text())
        assert cache["ndcg@10"] == 0.55
        assert cache["candidates"][0]["eval_kind"] == "mteb"

    def test_local_writes_cache(self, tmp_path):
        ws = make_workspace(str(tmp_path))
        make_training_data(ws, "nq_val.jsonl", n=10)
        make_checkpoint(ws, "epoch_1", "finetune")
        fake_src = tmp_path / "src"
        _fake_src_tree(fake_src)

        result = subprocess.run(
            [sys.executable, str(GRADERS_DIR / "eval.py"), "emb", "local", "data/nq_val.jsonl", ".nq_eval.json", ws],
            env={**os.environ, "SRC_DIR": str(fake_src)},
            capture_output=True, text=True, timeout=30,
        )

        assert result.returncode == 0, result.stderr
        cache = json.loads(Path(ws, ".nq_eval.json").read_text())
        assert cache["ndcg@10"] == 0.42
        assert cache["candidates"][0]["eval_kind"] == "local"
