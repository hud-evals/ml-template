"""ML training environment -- torchtitan fork with embedding experiments."""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from sdlc import BashGrader, CodingEnvironment, Grade, bash
from sdlc.mcp.coding import CodingService


logger = logging.getLogger(__name__)
MCP_TESTING_MODE = os.environ.get("MCP_TESTING_MODE") in ["1", "true"]

_REPO_ROOT = Path(__file__).parent
if Path("/mcp_server/torchtitan").exists():
    SRC_DIR = "/mcp_server"
    WORKSPACE = "/home/ubuntu/workspace"
elif Path("/code/torchtitan").exists():
    SRC_DIR = "/code"
    WORKSPACE = "/workspace"
else:
    SRC_DIR = str(_REPO_ROOT)
    WORKSPACE = str(_REPO_ROOT / "workspace")

# Ensure SRC_DIR is on sys.path so grading imports 
import sys
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

env = CodingEnvironment("coding")

AGENT_CONFIG = {
    "system_prompt": (
        "You are an expert ML engineer specializing in embedding model training.\n\n"
        "Environment:\n"
        "  Working directory: /workspace\n"
        "  Framework: torchtitan/ with embedding experiment at torchtitan/experiments/embedding/\n"
        "    (InfoNCE loss, hard negative mining, Matryoshka loss, asymmetric formatting, SLERP model merging)\n"
        "  Base model: Qwen/Qwen3-0.6B (general-purpose LM, NOT an embedding model)\n"
        "  All models and datasets are pre-cached locally -- no network access.\n"
        "  Always use absolute paths.\n\n"
        "Pre-staged data in data/:\n"
        "  data/synthetic.jsonl -- 5000 synthetic training pairs\n"
        "  data/scifact.jsonl   -- ~400 labeled SciFact pairs\n"
        "  data/nq.jsonl        -- 5000 labeled Natural Questions pairs\n"
        "  data/val.jsonl       -- held-out SciFact pairs (validation only, do NOT train on this)\n\n"
        "CLI tools:\n"
        "  python -m torchtitan.experiments.embedding.train --stage <pretrain|finetune> --train_data <path> --output_dir <path> [--resume_from <ckpt>] [--epochs N] [--batch_size N] [--gradient_accumulation_steps N]\n"
        "  python -m torchtitan.experiments.embedding.evaluate --model <ckpt> --eval_data data/val.jsonl\n"
        "  python -m torchtitan.experiments.embedding.merge --checkpoints <ckpt1> <ckpt2> --output_dir <path>\n"
        "  python -m torchtitan.experiments.embedding.prepare_data merge --inputs <f1> <f2> --output <out>\n"
        "  python -m torchtitan.experiments.embedding.prepare_data filter --input <path> --output <path> --model <ckpt> --min_similarity 0.7\n\n"
        "Constraints:\n"
        "  - GPU memory is limited. Use batch_size=4 with gradient_accumulation_steps=8 for effective batch of 32.\n"
        "  - Bash session times out after 120s with no output. For training/evaluation:\n"
        "    nohup cmd > log.log 2>&1 & echo PID:$! -- then poll with: tail -20 log.log\n"
        "  - Delete intermediate checkpoints to save disk and speed up grading.\n"
        "  - Write metrics.json into your best checkpoint directory."
    ),
}

VLM_AGENT_CONFIG = {
    "system_prompt": (
        "You are an expert ML engineer specializing in vision-language model training.\n\n"
        "Environment:\n"
        "  Working directory: /workspace\n"
        "  Framework: torchtitan/ with VLM experiment at torchtitan/experiments/vlm/\n"
        "  Model: Qwen3 decoder + SigLIP2 vision encoder\n"
        "    Available flavors: debugmodel (~7M), qwen3_0.6B (~700M, Qwen3-0.6B + SigLIP2-Base),\n"
        "                       qwen3_1.7B (~2B, Qwen3-1.7B + SigLIP2-Large)\n"
        "  Uses FlexAttention for both encoder and decoder.\n"
        "  Always use absolute paths.\n\n"
        "Pre-staged data:\n"
        "  data/cc12m/       -- CC12M image-caption pairs (webdataset .tar format)\n"
        "  tokenizer/        -- Llama3 tokenizer (for debugmodel only, vocab_size=2048)\n"
        "  tokenizer_qwen3/  -- Qwen3 tokenizer with VLM special tokens (for qwen3_* flavors)\n\n"
        "Training:\n"
        "  PYTHONPATH=/workspace python -m torchtitan.experiments.vlm.train \\\n"
        "    --flavor qwen3_0.6B --dataset cc12m-test --data_path /workspace/data/cc12m \\\n"
        "    --tokenizer_path /workspace/tokenizer_qwen3 \\\n"
        "    --output_dir /workspace/checkpoints/vlm --steps 200 --batch_size 4\n\n"
        "Constraints:\n"
        "  - Use qwen3_0.6B for a proper training run (fits on a single H100)\n"
        "  - Dataset loops infinitely (set --steps to control training length)\n"
        "  - Loss should decrease steadily; expect convergence within 100-200 steps\n"
        "  - Bash session times out after 120s with no output. For training:\n"
        "    nohup cmd > log.log 2>&1 & echo PID:$! -- then poll with: tail -20 log.log\n"
        "  - Delete intermediate checkpoints to save disk and speed up grading.\n"
        "  - Write training_metadata.json into your checkpoint directory."
    ),
}

_tools_initialized = False


def init_tools(workspace: str | None = None):
    """Initialize coding tools with workspace sandboxing. Call after WORKSPACE is set."""
    global _tools_initialized
    if _tools_initialized:
        return
    _tools_initialized = True

    import asyncio as _aio

    from hud.tools.coding.bash import ClaudeBashSession
    from hud.tools.coding.shell import ShellTool
    from hud.tools.coding.utils import get_demote_preexec_fn

    ws = workspace or WORKSPACE

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
            # Restrict: HOME to workspace, cd only allows workspace subtree
            await self.run(
                f'export HOME="{ws}" && '
                f'cd() {{ local t="${{1:-.}}"; local r=$(realpath -m "$t" 2>/dev/null || echo "$t"); '
                f'case "$r" in "{ws}"*) builtin cd "$t" ;; '
                f'*) echo "Error: cannot navigate outside workspace" >&2; return 1 ;; esac; }}'
            )

    coding_service = CodingService()
    coding_service.bash_tool.session = _SandboxedSession()
    ShellTool(cwd=ws).register(coding_service.server)
    env.connect_server(coding_service.server)


def check_weights(stage: str) -> str:
    """Bash command that finds a checkpoint for *stage* and verifies it has model weights."""
    find = f"python /tmp/check_metadata.py {stage} {WORKSPACE}"
    return (
        f"DIR=$({find}) && "
        f"(ls $DIR/model.safetensors 2>/dev/null || ls $DIR/pytorch_model.bin 2>/dev/null)"
    )

def check_merge_weights() -> str:
    """Bash command that finds a merge checkpoint and verifies it has model weights."""
    find = f"python /tmp/check_merge.py {WORKSPACE}"
    return (
        f"DIR=$({find}) && "
        f"(ls $DIR/model.safetensors 2>/dev/null || ls $DIR/pytorch_model.bin 2>/dev/null)"
    )



def _grade(graders: list[dict[str, Any]]) -> Grade:
    """Build a Grade from a list of grader dicts: [{name, command, weight?, timeout?}, ...]."""
    total = sum(g.get("weight", 1) for g in graders)
    return Grade.from_subscores([
        BashGrader.grade(
            name=g.get("name"),
            weight=g.get("weight", 1) / total,
            command=g["command"],
            timeout=g.get("timeout", 10),
        )
        for g in graders
    ])


# ---------------------------------------------------------------------------
# Workspace setup helpers
# ---------------------------------------------------------------------------

def _build_scifact_val_set(val_path: str, n_candidates: int = 200):
    """Build a SciFact validation set from test qrels (held out from training)."""
    import random as _rng

    from datasets import load_dataset

    _rng.seed(42)

    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus", trust_remote_code=True)
    queries = load_dataset("BeIR/scifact", "queries", split="queries", trust_remote_code=True)
    qrels = load_dataset("BeIR/scifact-qrels", split="test", trust_remote_code=True)

    corpus_map = {str(row["_id"]): row["text"] for row in corpus}
    corpus_texts = list(corpus_map.values())
    query_map = {str(row["_id"]): row["text"] for row in queries}

    qrel_map: dict[str, list[str]] = {}
    for row in qrels:
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        if row["score"] > 0:
            qrel_map.setdefault(qid, []).append(cid)

    count = 0
    with open(val_path, "w") as f:
        for qid, pos_ids in qrel_map.items():
            if qid not in query_map or not pos_ids:
                continue
            pos_text = corpus_map.get(pos_ids[0], "")
            if not pos_text:
                continue

            pool = [t for t in _rng.sample(corpus_texts, min(n_candidates + 10, len(corpus_texts))) if t != pos_text]
            candidates = pool[:n_candidates]

            f.write(json.dumps({
                "instruction": "Given a scientific claim, retrieve evidence that supports or refutes it",
                "query": query_map[qid],
                "positive": pos_text,
                "candidates": candidates,
            }) + "\n")
            count += 1

    logger.info("Built SciFact val set: %d queries (%d candidates each) at %s", count, n_candidates, val_path)


def _clear_dataset_cache(dataset_name: str):
    """Remove a HuggingFace dataset from the local cache."""
    from huggingface_hub import scan_cache_dir

    cache_info = scan_cache_dir()
    for repo in cache_info.repos:
        if repo.repo_id == dataset_name:
            for revision in repo.revisions:
                strategy = cache_info.delete_revisions(revision.commit_hash)
                strategy.execute()
            logger.info("Cleared HF cache for %s", dataset_name)
            return
    logger.info("No HF cache found for %s", dataset_name)


def _setup_workspace():
    shutil.rmtree(WORKSPACE, ignore_errors=True)
    os.makedirs(WORKSPACE, exist_ok=True)

    bash(f"cp -r {SRC_DIR}/torchtitan {WORKSPACE}/torchtitan")

    os.makedirs(f"{WORKSPACE}/data", exist_ok=True)
    os.chdir(WORKSPACE)

    build = f"PYTHONPATH={SRC_DIR} python -m data.build_datasets"
    bash(
        f"{build} synthetic "
        f"--corpus_dataset scifact --output {WORKSPACE}/data/synthetic.jsonl --max_samples 5000"
    )
    bash(
        f"{build} download "
        f"--dataset scifact --output {WORKSPACE}/data/scifact.jsonl --max_samples 500"
    )
    bash(
        f"{build} download "
        f"--dataset nq --output {WORKSPACE}/data/nq.jsonl --max_samples 5000"
    )

    _build_scifact_val_set(f"{WORKSPACE}/data/val.jsonl")
    _clear_dataset_cache("BeIR/scifact-qrels")

    if os.getuid() == 0:
        bash(f"chmod -R 777 {WORKSPACE}")

        agent_cache = "/home/ubuntu/.cache"
        os.makedirs(f"{agent_cache}/huggingface", exist_ok=True)

        hub_dir = "/root/.cache/huggingface/hub"
        if os.path.exists(hub_dir):
            bash(f"chmod -R 777 {hub_dir}")
            hub_link = f"{agent_cache}/huggingface/hub"
            if not os.path.exists(hub_link):
                os.symlink(hub_dir, hub_link)

        torch_dir = "/root/.cache/torch"
        if os.path.exists(torch_dir):
            bash(f"chmod -R 777 {torch_dir}")
            torch_link = f"{agent_cache}/torch"
            if not os.path.exists(torch_link):
                os.symlink(torch_dir, torch_link)

        bash(f"chown -R 1000:1000 {agent_cache}")

        bashrc = "/home/ubuntu/.bashrc"
        with open(bashrc, "a") as f:
            f.write("\nexport USER=ubuntu\nexport LOGNAME=ubuntu\n")
        bash(f"chown 1000:1000 {bashrc}")

        for d in [SRC_DIR, "/code"]:
            if os.path.exists(d):
                bash(f"chmod 700 {d}")


def _write_check_scripts():
    from grading.checks import (
        grader_eval_script,
        merge_check_script,
        metadata_check_script,
        ndcg_check_script,
        resume_check_script,
    )

    scripts = {
        "grader_eval": grader_eval_script(SRC_DIR),
        "check_ndcg": ndcg_check_script(),
        "check_metadata": metadata_check_script(),
        "check_resume": resume_check_script(),
        "check_merge": merge_check_script(),
    }
    for name, content in scripts.items():
        with open(f"/tmp/{name}.py", "w") as f:
            f.write(content)


def _setup_vlm_workspace():
    """Set up workspace for VLM training tasks."""
    shutil.rmtree(WORKSPACE, ignore_errors=True)
    os.makedirs(WORKSPACE, exist_ok=True)

    bash(f"cp -r {SRC_DIR}/torchtitan {WORKSPACE}/torchtitan")
    bash(f"cp -r {SRC_DIR}/tests {WORKSPACE}/tests")

    os.makedirs(f"{WORKSPACE}/data/cc12m", exist_ok=True)
    bash(f"cp {SRC_DIR}/tests/assets/cc12m_test/*.tar {WORKSPACE}/data/cc12m/")
    bash(f"cp -r {SRC_DIR}/tests/assets/tokenizer {WORKSPACE}/tokenizer")
    if os.path.exists(f"{SRC_DIR}/tests/assets/tokenizer_qwen3"):
        bash(f"cp -r {SRC_DIR}/tests/assets/tokenizer_qwen3 {WORKSPACE}/tokenizer_qwen3")

    os.chdir(WORKSPACE)

    if os.getuid() == 0:
        bash(f"chmod -R 777 {WORKSPACE}")

        agent_cache = "/home/ubuntu/.cache"
        os.makedirs(f"{agent_cache}/huggingface", exist_ok=True)

        for cache_dir in ["/root/.cache/huggingface/hub", "/root/.cache/torch"]:
            if os.path.exists(cache_dir):
                bash(f"chmod -R 777 {cache_dir}")
                link = f"{agent_cache}/{os.path.basename(os.path.dirname(cache_dir))}/{os.path.basename(cache_dir)}"
                os.makedirs(os.path.dirname(link), exist_ok=True)
                if not os.path.exists(link):
                    os.symlink(cache_dir, link)

        bash(f"chown -R 1000:1000 {agent_cache}")

        bashrc = "/home/ubuntu/.bashrc"
        with open(bashrc, "a") as f:
            f.write("\nexport USER=ubuntu\nexport LOGNAME=ubuntu\n")
        bash(f"chown 1000:1000 {bashrc}")

        for d in [SRC_DIR, "/code"]:
            if os.path.exists(d):
                bash(f"chmod 700 {d}")


def _write_code_fix_check_scripts(mutation_type: str):
    from grading.checks import code_fix_check_script

    with open("/tmp/check_code_fix.py", "w") as f:
        f.write(code_fix_check_script(mutation_type))


def _write_data_audit_check_scripts():
    from grading.checks import data_cleaned_check_script

    with open("/tmp/check_data_cleaned.py", "w") as f:
        f.write(data_cleaned_check_script())


def _write_vlm_check_scripts():
    from grading.checks import (
        vlm_checkpoint_check_script,
        vlm_eval_script,
        vlm_loss_check_script,
        vlm_metadata_check_script,
    )

    scripts = {
        "vlm_check_metadata": vlm_metadata_check_script(),
        "vlm_check_checkpoint": vlm_checkpoint_check_script(),
        "vlm_eval": vlm_eval_script(SRC_DIR),
        "vlm_check_loss": vlm_loss_check_script(),
    }
    for name, content in scripts.items():
        with open(f"/tmp/{name}.py", "w") as f:
            f.write(content)


# ---------------------------------------------------------------------------
# Mutation helpers
# ---------------------------------------------------------------------------

def _apply_code_mutation(mutation_type: str):
    """Apply a code mutation to the workspace embedding experiment code."""
    losses_path = f"{WORKSPACE}/torchtitan/experiments/embedding/losses.py"
    trainer_path = f"{WORKSPACE}/torchtitan/experiments/embedding/embedding_trainer.py"

    if mutation_type == "buggy_loss":
        with open(losses_path) as f:
            code = f.read()
        code = code.replace(
            "    query_embeds = F.normalize(query_embeds, dim=-1)\n"
            "    positive_embeds = F.normalize(positive_embeds, dim=-1)\n"
            "    negative_embeds = F.normalize(negative_embeds, dim=-1)\n",
            "    # Embeddings used as-is (no normalization)\n"
        )
        with open(losses_path, "w") as f:
            f.write(code)

    elif mutation_type == "bad_pooling":
        with open(trainer_path) as f:
            code = f.read()
        code = code.replace(
            "    seq_lengths = attention_mask.sum(dim=-1) - 1\n"
            "    batch_indices = torch.arange(hidden.size(0), device=hidden.device)\n"
            "    last_hidden = hidden[batch_indices, seq_lengths]\n",
            "    # Mean pooling over all tokens\n"
            "    last_hidden = (hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=-1, keepdim=True)\n"
        )
        with open(trainer_path, "w") as f:
            f.write(code)

    elif mutation_type == "wrong_negatives":
        datasets_path = f"{WORKSPACE}/torchtitan/experiments/embedding/datasets.py"
        with open(datasets_path) as f:
            code = f.read()
        code = code.replace(
            "        negs = pair.negatives[: self.num_hard_negatives]\n"
            "        while len(negs) < self.num_hard_negatives:\n"
            "            other_idx = random.randint(0, len(self.pairs) - 1)\n"
            "            if other_idx != idx:\n"
            "                negs.append(self.pairs[other_idx].positive)\n",
            "        # Use first negative repeated (bug: no diversity)\n"
            "        first_neg = pair.negatives[0] if pair.negatives else pair.positive\n"
            "        negs = [first_neg] * self.num_hard_negatives\n"
        )
        with open(datasets_path, "w") as f:
            f.write(code)


def _apply_vlm_code_mutation(mutation_type: str):
    """Apply a code mutation to the workspace VLM code."""
    model_path = f"{WORKSPACE}/torchtitan/experiments/vlm/model/model.py"
    datasets_path = f"{WORKSPACE}/torchtitan/experiments/vlm/datasets/mm_datasets.py"

    if mutation_type == "buggy_projector":
        with open(model_path) as f:
            code = f.read()
        code = code.replace(
            "        x_NLD = self.w1(x_NLD)\n"
            "        x_NLD = nn.functional.silu(x_NLD)\n"
            "        x_NLD = self.w2(x_NLD)\n",
            "        # Linear projection (no nonlinearity)\n"
            "        x_NLD = self.w2(self.w1(x_NLD))\n"
        )
        with open(model_path, "w") as f:
            f.write(code)

    elif mutation_type == "bad_label_mask":
        with open(datasets_path) as f:
            code = f.read()
        code = code.replace(
            "        # Mask special tokens in labels\n"
            "        special_token_ids = torch.tensor(\n"
            "            [special_tokens.boi_id, special_tokens.eoi_id, special_tokens.img_id]\n"
            "        )\n"
            "        labels = torch.where(\n"
            "            torch.isin(labels, special_token_ids), special_tokens.ignore_id, labels\n"
            "        )\n",
            "        # Labels include all tokens (no masking)\n"
        )
        with open(datasets_path, "w") as f:
            f.write(code)


# ===========================================================================
# Scenarios -- setup + grading from task-provided config
# ===========================================================================


@env.scenario(name="multistage_retrieval")
async def multistage_retrieval(prompt: str, graders: list[dict[str, Any]]):
    _setup_workspace()
    yield prompt
    _write_check_scripts()
    yield _grade(graders)


@env.scenario(name="pretrain_embedding")
async def pretrain_embedding(prompt: str, graders: list[dict[str, Any]]):
    _setup_workspace()
    yield prompt
    _write_check_scripts()
    yield _grade(graders)


@env.scenario(name="finetune_embedding")
async def finetune_embedding(prompt: str, graders: list[dict[str, Any]]):
    _setup_workspace()
    yield prompt
    _write_check_scripts()
    yield _grade(graders)


@env.scenario(name="merge_and_evaluate")
async def merge_and_evaluate(prompt: str, graders: list[dict[str, Any]]):
    _setup_workspace()
    yield prompt
    _write_check_scripts()

    _MULTI_CKPT_CHECK = f"""
import glob, os, sys
ws = "{WORKSPACE}"
ckpts = set()
for mf in glob.glob(f"{{ws}}/**/model.safetensors", recursive=True):
    ckpts.add(os.path.dirname(mf))
for mf in glob.glob(f"{{ws}}/**/pytorch_model.bin", recursive=True):
    ckpts.add(os.path.dirname(mf))
ckpts = [c for c in ckpts if "merge" not in c.lower()]
print(f"Found {{len(ckpts)}} non-merged checkpoints")
sys.exit(0 if len(ckpts) >= 2 else 1)
"""
    with open("/tmp/check_multi_ckpt.py", "w") as f:
        f.write(_MULTI_CKPT_CHECK)

    yield _grade(graders)


@env.scenario(name="debug_embedding")
async def debug_embedding(
    prompt: str,
    graders: list[dict[str, Any]],
    mutation_type: str = "buggy_loss",
):
    _setup_workspace()
    _apply_code_mutation(mutation_type)
    yield prompt
    _write_check_scripts()
    _write_code_fix_check_scripts(mutation_type)
    yield _grade(graders)


@env.scenario(name="data_audit_embedding")
async def data_audit_embedding(
    prompt: str,
    graders: list[dict[str, Any]],
    contamination: str = "label_noise",
    noise_rate: float = 0.3,
    leak_rate: float = 0.2,
):
    _setup_workspace()

    import hashlib

    from grading.mutations import inject_data_leakage, inject_duplicates, inject_label_noise

    train_path = f"{WORKSPACE}/data/scifact.jsonl"
    contaminated_path = f"{WORKSPACE}/data/scifact.jsonl"

    if contamination == "label_noise":
        inject_label_noise(train_path, contaminated_path, noise_rate=noise_rate)
    elif contamination == "data_leakage":
        val_path = f"{WORKSPACE}/data/val.jsonl"
        inject_data_leakage(train_path, val_path, contaminated_path, leak_rate=leak_rate)
    elif contamination == "duplicates":
        inject_duplicates(train_path, contaminated_path, dup_rate=noise_rate)

    contamination_info = {
        "type": contamination,
        "hash": hashlib.md5(open(contaminated_path, "rb").read()).hexdigest(),
        "line_count": sum(1 for _ in open(contaminated_path)),
    }
    with open("/tmp/contamination_info.json", "w") as f:
        json.dump(contamination_info, f)

    yield prompt
    _write_check_scripts()
    _write_data_audit_check_scripts()
    yield _grade(graders)


@env.scenario(name="vlm_finetune")
async def vlm_finetune(prompt: str, graders: list[dict[str, Any]]):
    _setup_vlm_workspace()
    yield prompt
    _write_vlm_check_scripts()
    yield _grade(graders)


@env.scenario(name="debug_vlm")
async def debug_vlm(
    prompt: str,
    graders: list[dict[str, Any]],
    mutation_type: str = "buggy_projector",
):
    _setup_vlm_workspace()
    _apply_vlm_code_mutation(mutation_type)
    yield prompt
    _write_vlm_check_scripts()
    _write_code_fix_check_scripts(mutation_type)
    yield _grade(graders)
