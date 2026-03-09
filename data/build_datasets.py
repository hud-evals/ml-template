"""Download and generate training datasets for the embedding pipeline.

This script runs during scenario setup (NOT copied to the agent workspace).
"""

import argparse
import json
import logging
import random
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_msmarco(max_samples: int, num_negatives: int) -> list[dict]:
    from datasets import load_dataset

    logger.info("Loading MS MARCO passage ranking...")
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True, trust_remote_code=True)

    pairs = []
    for item in ds:
        if len(pairs) >= max_samples:
            break

        passages = item.get("passages", {})
        is_selected = passages.get("is_selected", [])
        texts = passages.get("passage_text", [])

        positives = [t for t, s in zip(texts, is_selected) if s == 1]
        negatives = [t for t, s in zip(texts, is_selected) if s == 0]

        if not positives:
            continue

        while len(negatives) < num_negatives:
            negatives.append(random.choice(texts))

        pairs.append({
            "instruction": "Given a web search query, retrieve relevant passages that answer the query",
            "query": item["query"],
            "positive": positives[0],
            "negatives": negatives[:num_negatives],
        })

    return pairs


def load_nq(max_samples: int, num_negatives: int) -> list[dict]:
    from datasets import load_dataset

    logger.info("Loading Natural Questions...")
    ds = load_dataset(
        "sentence-transformers/natural-questions", split="train", streaming=True,
    )

    pairs = []
    buffer = []
    for item in ds:
        buffer.append(item)
        if len(buffer) >= max_samples * 2:
            break

    random.shuffle(buffer)
    for item in buffer[:max_samples]:
        negatives = []
        attempts = 0
        while len(negatives) < num_negatives and attempts < num_negatives * 3:
            other = random.choice(buffer)
            if other["answer"] != item["answer"]:
                negatives.append(other["answer"])
            attempts += 1

        pairs.append({
            "instruction": "Given a question, retrieve the passage that contains the answer",
            "query": item["query"],
            "positive": item["answer"],
            "negatives": negatives[:num_negatives],
        })

    return pairs


def load_hotpotqa(max_samples: int, num_negatives: int) -> list[dict]:
    from datasets import load_dataset

    logger.info("Loading HotpotQA...")
    ds = load_dataset("hotpot_qa", "fullwiki", split="train", streaming=True, trust_remote_code=True)

    pairs = []
    buffer = []
    for item in ds:
        context_titles = item.get("context", {}).get("title", [])
        context_sents = item.get("context", {}).get("sentences", [])
        supporting = set(item.get("supporting_facts", {}).get("title", []))

        pos_parts = []
        neg_parts = []
        for title, sents in zip(context_titles, context_sents):
            text = " ".join(sents)
            if title in supporting:
                pos_parts.append(text)
            else:
                neg_parts.append(text)

        if pos_parts:
            buffer.append({
                "query": item["question"],
                "positive": " ".join(pos_parts),
                "neg_parts": neg_parts,
            })
        if len(buffer) >= max_samples:
            break

    for item in buffer:
        negatives = item["neg_parts"]
        while len(negatives) < num_negatives:
            other = random.choice(buffer)
            negatives.append(other["positive"])
        pairs.append({
            "instruction": "Given a multi-hop question, retrieve the passages needed to answer it",
            "query": item["query"],
            "positive": item["positive"],
            "negatives": negatives[:num_negatives],
        })

    return pairs


def load_nli(max_samples: int, num_negatives: int) -> list[dict]:
    from datasets import load_dataset

    logger.info("Loading AllNLI (SNLI + MultiNLI)...")
    ds = load_dataset("sentence-transformers/all-nli", "triplet", split="train", streaming=True)

    pairs = []
    neg_buffer = []
    for item in ds:
        if len(pairs) >= max_samples:
            break
        neg_buffer.append(item["negative"])
        negatives = [item["negative"]]
        while len(negatives) < num_negatives and len(neg_buffer) > num_negatives:
            negatives.append(random.choice(neg_buffer))
        pairs.append({
            "instruction": "Retrieve semantically similar sentences",
            "query": item["anchor"],
            "positive": item["positive"],
            "negatives": negatives[:num_negatives],
        })

    return pairs


def load_scifact(max_samples: int, num_negatives: int) -> list[dict]:
    from datasets import load_dataset

    logger.info("Loading SciFact...")
    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus", trust_remote_code=True)
    queries = load_dataset("BeIR/scifact", "queries", split="queries", trust_remote_code=True)

    corpus_map = {str(row["_id"]): row["text"] for row in corpus}

    # Hold out corpus docs referenced by test qrels to prevent eval leakage
    test_qrels = load_dataset("BeIR/scifact-qrels", split="test", trust_remote_code=True)
    holdout_ids = {str(row["corpus-id"]) for row in test_qrels if row["score"] > 0}
    logger.info("Holding out %d test corpus documents from training", len(holdout_ids))

    train_corpus = {cid: text for cid, text in corpus_map.items() if cid not in holdout_ids}
    train_texts = list(train_corpus.values())

    qrels = load_dataset("BeIR/scifact-qrels", split="train", trust_remote_code=True)
    qrel_map: dict[str, list[str]] = {}
    for row in qrels:
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        if row["score"] > 0 and cid not in holdout_ids:
            qrel_map.setdefault(qid, []).append(cid)

    query_map = {str(row["_id"]): row["text"] for row in queries}

    pairs = []
    for qid, pos_ids in qrel_map.items():
        if qid not in query_map or not pos_ids:
            continue
        pos_text = train_corpus.get(pos_ids[0], "")
        if not pos_text:
            continue

        negatives = random.sample(train_texts, min(num_negatives * 2, len(train_texts)))
        negatives = [n for n in negatives if n != pos_text][:num_negatives]

        pairs.append({
            "instruction": "Given a scientific claim, retrieve evidence that supports or refutes it",
            "query": query_map[qid],
            "positive": pos_text,
            "negatives": negatives,
        })
        if len(pairs) >= max_samples:
            break

    return pairs


def load_fiqa(max_samples: int, num_negatives: int) -> list[dict]:
    from datasets import load_dataset

    logger.info("Loading FiQA...")
    corpus = load_dataset("BeIR/fiqa", "corpus", split="corpus", trust_remote_code=True)
    queries = load_dataset("BeIR/fiqa", "queries", split="queries", trust_remote_code=True)

    corpus_map = {str(row["_id"]): row["text"] for row in corpus}
    corpus_texts = list(corpus_map.values())

    qrels = load_dataset("BeIR/fiqa-qrels", split="test", trust_remote_code=True)
    qrel_map: dict[str, list[str]] = {}
    for row in qrels:
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        if row["score"] > 0:
            qrel_map.setdefault(qid, []).append(cid)

    query_map = {str(row["_id"]): row["text"] for row in queries}

    pairs = []
    for qid, pos_ids in qrel_map.items():
        if qid not in query_map or not pos_ids:
            continue
        pos_text = corpus_map.get(pos_ids[0], "")
        if not pos_text:
            continue

        negatives = random.sample(corpus_texts, min(num_negatives * 2, len(corpus_texts)))
        negatives = [n for n in negatives if n != pos_text][:num_negatives]

        pairs.append({
            "instruction": "Given a financial question, retrieve relevant answers",
            "query": query_map[qid],
            "positive": pos_text,
            "negatives": negatives,
        })
        if len(pairs) >= max_samples:
            break

    return pairs


LOADERS = {
    "msmarco": load_msmarco,
    "nq": load_nq,
    "hotpotqa": load_hotpotqa,
    "nli": load_nli,
    "scifact": load_scifact,
    "fiqa": load_fiqa,
}


_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "nor", "not", "so", "yet", "both", "either", "neither", "each",
    "every", "all", "any", "few", "more", "most", "other", "some", "such",
    "no", "only", "own", "same", "than", "too", "very", "just", "because",
    "this", "that", "these", "those", "it", "its",
}

_INSTRUCTIONS = [
    "Given a web search query, retrieve relevant passages that answer the query",
    "Given a question, retrieve the passage that contains the answer",
    "Retrieve semantically similar sentences",
    "Given a topic, retrieve relevant documents",
    "Find passages that are related to the given text",
]


def _extract_synthetic_query(passage: str) -> str:
    sentences = re.split(r'[.!?]+', passage)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if not sentences:
        return ""

    sent = random.choice(sentences[:3])
    words = sent.split()
    key_words = [w for w in words if w.lower().strip(".,;:!?") not in _STOP_WORDS]
    if len(key_words) < 3:
        return sent

    n = random.randint(3, min(8, len(key_words)))
    return " ".join(key_words[:n])


def generate_synthetic(
    corpus_dataset: str,
    max_samples: int,
    num_negatives: int,
    seed: int = 42,
) -> list[dict]:
    random.seed(seed)

    loader = LOADERS.get(corpus_dataset)
    if loader is None:
        raise ValueError(f"Unknown corpus dataset: {corpus_dataset}. Choose from {list(LOADERS.keys())}")

    logger.info("Loading corpus from %s for synthetic pair generation...", corpus_dataset)
    source_pairs = loader(max_samples * 2, 0)

    passages = []
    for p in source_pairs:
        passages.append(p["positive"])
        if p.get("query"):
            passages.append(p["query"])
    passages = list(set(passages))
    random.shuffle(passages)

    logger.info("Generating synthetic pairs from %d passages...", len(passages))
    pairs = []
    for passage in passages:
        if len(pairs) >= max_samples:
            break

        query = _extract_synthetic_query(passage)
        if not query:
            continue

        negs = random.sample(passages, min(num_negatives + 1, len(passages)))
        negs = [n for n in negs if n != passage][:num_negatives]

        pairs.append({
            "instruction": random.choice(_INSTRUCTIONS),
            "query": query,
            "positive": passage,
            "negatives": negs,
        })

    logger.info("Generated %d synthetic pairs", len(pairs))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Build training datasets")
    subparsers = parser.add_subparsers(dest="command")

    dl = subparsers.add_parser("download")
    dl.add_argument("--dataset", required=True, choices=list(LOADERS.keys()))
    dl.add_argument("--output", required=True)
    dl.add_argument("--max_samples", type=int, default=50000)
    dl.add_argument("--num_negatives", type=int, default=7)
    dl.add_argument("--seed", type=int, default=42)

    syn = subparsers.add_parser("synthetic")
    syn.add_argument("--corpus_dataset", required=True, choices=list(LOADERS.keys()))
    syn.add_argument("--output", required=True)
    syn.add_argument("--max_samples", type=int, default=100000)
    syn.add_argument("--num_negatives", type=int, default=7)
    syn.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "download":
        random.seed(args.seed)
        pairs = LOADERS[args.dataset](args.max_samples, args.num_negatives)
        logger.info("Loaded %d training pairs from %s", len(pairs), args.dataset)
        with open(args.output, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
        logger.info("Wrote %s", args.output)

    elif args.command == "synthetic":
        pairs = generate_synthetic(args.corpus_dataset, args.max_samples, args.num_negatives, args.seed)
        with open(args.output, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
        logger.info("Wrote %s", args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
