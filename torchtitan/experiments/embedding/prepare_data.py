"""Data preparation utilities for the embedding pipeline (filter + merge)."""

import argparse
import json
import logging
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def filter_by_similarity(
    input_path: str,
    output_path: str,
    model_path: str,
    min_similarity: float = 0.7,
    batch_size: int = 64,
    max_seq_length: int = 512,
):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    from .datasets import format_document_input, format_query_input

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    pairs = []
    with open(input_path) as f:
        for line in f:
            pairs.append(json.loads(line))

    logger.info("Filtering %d pairs with min_similarity=%.2f using %s", len(pairs), min_similarity, model_path)
    kept = []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]

        q_texts = [format_query_input(p.get("instruction", ""), p["query"], tokenizer.eos_token) for p in batch]
        d_texts = [format_document_input(p["positive"], tokenizer.eos_token) for p in batch]

        with torch.no_grad():
            q_enc = tokenizer(q_texts, max_length=max_seq_length, padding=True, truncation=True, return_tensors="pt").to(device)
            q_out = model(**q_enc).last_hidden_state
            q_lens = q_enc["attention_mask"].sum(dim=-1) - 1
            idx = torch.arange(q_out.size(0), device=device)
            q_emb = F.normalize(q_out[idx, q_lens], dim=-1)

            d_enc = tokenizer(d_texts, max_length=max_seq_length, padding=True, truncation=True, return_tensors="pt").to(device)
            d_out = model(**d_enc).last_hidden_state
            d_lens = d_enc["attention_mask"].sum(dim=-1) - 1
            idx = torch.arange(d_out.size(0), device=device)
            d_emb = F.normalize(d_out[idx, d_lens], dim=-1)

            sims = (q_emb * d_emb).sum(dim=-1)

        for pair, sim in zip(batch, sims.tolist()):
            if sim >= min_similarity:
                kept.append(pair)

    logger.info("Kept %d/%d pairs (%.1f%%)", len(kept), len(pairs), 100 * len(kept) / max(len(pairs), 1))

    with open(output_path, "w") as f:
        for pair in kept:
            f.write(json.dumps(pair) + "\n")

    logger.info("Wrote %s", output_path)


def merge_datasets(
    inputs: list[str],
    output: str,
    ratios: list[float] | None = None,
    max_samples: int | None = None,
    seed: int = 42,
):
    random.seed(seed)

    all_pairs: list[list[dict]] = []
    for path in inputs:
        pairs = []
        with open(path) as f:
            for line in f:
                pairs.append(json.loads(line))
        all_pairs.append(pairs)
        logger.info("Loaded %d pairs from %s", len(pairs), path)

    if ratios is None:
        ratios = [1.0 / len(inputs)] * len(inputs)

    ratio_sum = sum(ratios)
    ratios = [r / ratio_sum for r in ratios]

    total = max_samples or sum(len(p) for p in all_pairs)

    merged = []
    for pairs, ratio in zip(all_pairs, ratios):
        n = min(int(total * ratio), len(pairs))
        sampled = random.sample(pairs, n) if n < len(pairs) else pairs
        merged.extend(sampled)

    random.shuffle(merged)
    if max_samples and len(merged) > max_samples:
        merged = merged[:max_samples]

    with open(output, "w") as f:
        for pair in merged:
            f.write(json.dumps(pair) + "\n")

    logger.info("Merged %d pairs into %s", len(merged), output)


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for embedding pipeline")
    subparsers = parser.add_subparsers(dest="command")

    filt = subparsers.add_parser("filter")
    filt.add_argument("--input", required=True)
    filt.add_argument("--output", required=True)
    filt.add_argument("--model", required=True)
    filt.add_argument("--min_similarity", type=float, default=0.7)
    filt.add_argument("--batch_size", type=int, default=64)
    filt.add_argument("--max_seq_length", type=int, default=512)

    mg = subparsers.add_parser("merge")
    mg.add_argument("--inputs", nargs="+", required=True)
    mg.add_argument("--output", required=True)
    mg.add_argument("--ratios", type=float, nargs="*", default=None)
    mg.add_argument("--max_samples", type=int, default=None)
    mg.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "filter":
        filter_by_similarity(args.input, args.output, args.model, args.min_similarity, args.batch_size, args.max_seq_length)

    elif args.command == "merge":
        merge_datasets(args.inputs, args.output, args.ratios, args.max_samples, args.seed)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
