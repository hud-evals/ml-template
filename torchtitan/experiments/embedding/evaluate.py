"""Evaluation script for embedding models using MTEB/BEIR benchmarks."""

import argparse
import json
import logging
import math
import os
import sys

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _dcg(relevances: list[float], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def _ndcg_at_k(relevances: list[float], k: int) -> float:
    dcg = _dcg(relevances, k)
    ideal = _dcg(sorted(relevances, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


def _encode_texts(texts: list[str], model, tokenizer, device, max_seq_length: int, batch_size: int = 64) -> torch.Tensor:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, max_length=max_seq_length, padding=True,
            truncation=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**enc).last_hidden_state
            seq_lens = enc["attention_mask"].sum(dim=-1) - 1
            idx = torch.arange(out.size(0), device=device)
            embs = F.normalize(out[idx, seq_lens], dim=-1)
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def evaluate_local(
    model_path: str,
    eval_path: str,
    max_seq_length: int = 512,
    output_dim: int | None = None,
    batch_size: int = 64,
) -> dict:
    """Evaluate on a local JSONL eval file."""
    from transformers import AutoModel, AutoTokenizer

    from .datasets import format_document_input, format_query_input

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    queries = []
    with open(eval_path) as f:
        for line in f:
            queries.append(json.loads(line))

    doc_to_idx: dict[str, int] = {}
    for item in queries:
        positive = item["positive"]
        candidates = item.get("candidates", [])
        if positive not in candidates:
            candidates = [positive] + candidates
        for c in candidates:
            if c not in doc_to_idx:
                doc_to_idx[c] = len(doc_to_idx)

    all_doc_texts = [""] * len(doc_to_idx)
    for text, idx in doc_to_idx.items():
        all_doc_texts[idx] = format_document_input(text, tokenizer.eos_token)
    logger.info("Encoding %d unique candidates...", len(all_doc_texts))
    all_doc_embs = _encode_texts(all_doc_texts, model, tokenizer, device, max_seq_length, batch_size)
    if output_dim:
        all_doc_embs = F.normalize(all_doc_embs[:, :output_dim], dim=-1)

    q_texts = [
        format_query_input(item.get("instruction", ""), item["query"], tokenizer.eos_token)
        for item in queries
    ]
    logger.info("Encoding %d queries...", len(q_texts))
    all_q_embs = _encode_texts(q_texts, model, tokenizer, device, max_seq_length, batch_size)
    if output_dim:
        all_q_embs = F.normalize(all_q_embs[:, :output_dim], dim=-1)

    ndcg_scores = []
    mrr_scores = []
    recall_at = {1: [], 5: [], 10: []}

    for i, item in enumerate(queries):
        positive = item["positive"]
        candidates = item.get("candidates", [])
        if positive not in candidates:
            candidates = [positive] + candidates
        positive_idx = candidates.index(positive)

        cand_indices = torch.tensor([doc_to_idx[c] for c in candidates], device=device)
        cand_embs = all_doc_embs[cand_indices]
        scores = cand_embs @ all_q_embs[i]
        ranked = scores.argsort(descending=True).tolist()

        relevances = [1.0 if idx == positive_idx else 0.0 for idx in ranked]
        ndcg_scores.append(_ndcg_at_k(relevances, 10))

        rank = ranked.index(positive_idx) + 1
        mrr_scores.append(1.0 / rank)

        for k in recall_at:
            recall_at[k].append(1.0 if positive_idx in ranked[:k] else 0.0)

    metrics = {
        "ndcg@10": sum(ndcg_scores) / len(ndcg_scores),
        "mrr": sum(mrr_scores) / len(mrr_scores),
    }
    for k, vals in recall_at.items():
        metrics[f"recall@{k}"] = sum(vals) / len(vals)
    metrics["num_queries"] = len(queries)

    return metrics


def evaluate_mteb(
    model_path: str,
    tasks: list[str],
    batch_size: int = 64,
    max_seq_length: int = 512,
) -> dict:
    """Evaluate using MTEB benchmark tasks."""
    import mteb
    import numpy as np
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    class HFEmbedder:
        def __init__(self, model, tokenizer, device, max_length, batch_size):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.max_length = max_length
            self.batch_size = batch_size

        def encode(self, sentences: list[str], **kwargs) -> np.ndarray:
            all_embs = []
            for i in range(0, len(sentences), self.batch_size):
                batch = sentences[i : i + self.batch_size]
                enc = self.tokenizer(
                    batch, max_length=self.max_length, padding=True,
                    truncation=True, return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    out = self.model(**enc).last_hidden_state
                    seq_lens = enc["attention_mask"].sum(dim=-1) - 1
                    idx = torch.arange(out.size(0), device=self.device)
                    embs = F.normalize(out[idx, seq_lens], dim=-1)
                all_embs.append(embs.cpu().float().numpy())
            return np.concatenate(all_embs, axis=0)

    embedder = HFEmbedder(model, tokenizer, device, max_seq_length, batch_size)

    mteb_tasks = mteb.get_tasks(tasks=tasks)
    evaluation = mteb.MTEB(tasks=mteb_tasks)
    results = evaluation.run(embedder, output_folder=None, verbosity=0)

    metrics = {}
    ndcg_scores = []
    for task_result in results:
        task_name = task_result.task_name
        for split, split_scores in task_result.scores.items():
            for score_dict in split_scores:
                ndcg = score_dict.get("ndcg_at_10", score_dict.get("cos_sim_spearman", 0))
                metrics[f"{task_name}/{split}/ndcg@10"] = ndcg
                ndcg_scores.append(ndcg)

    if ndcg_scores:
        metrics["ndcg@10"] = sum(ndcg_scores) / len(ndcg_scores)
    metrics["num_tasks"] = len(tasks)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate embedding model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval_data", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dim", type=int, default=None)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--threshold_ndcg", type=float, default=None)
    args = parser.parse_args()

    if args.eval_data:
        metrics = evaluate_local(args.model, args.eval_data, args.max_seq_length, args.output_dim)
    elif args.tasks:
        metrics = evaluate_mteb(args.model, args.tasks, args.batch_size, args.max_seq_length)
    else:
        parser.error("Provide either --eval_data or --tasks")

    output = json.dumps(metrics, indent=2)
    print(output)

    if os.path.isdir(args.model):
        metrics_path = os.path.join(args.model, "metrics.json")
        with open(metrics_path, "w") as f:
            f.write(output)
        logger.info("Wrote %s", metrics_path)

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output)

    if args.threshold_ndcg is not None and metrics.get("ndcg@10", 0) < args.threshold_ndcg:
        logger.error("nDCG@10 %.4f below threshold %.4f", metrics["ndcg@10"], args.threshold_ndcg)
        sys.exit(1)


if __name__ == "__main__":
    main()
