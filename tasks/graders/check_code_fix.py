"""Verify the agent actually fixed the injected code bug.

Usage: python check_code_fix.py <mutation_type> <workspace>
"""
import os, sys

mutation_type = sys.argv[1]
workspace = sys.argv[2]

MUTATIONS = {
    "buggy_loss": {
        "description": "Positive scores must be temperature-scaled like negatives",
        "files": [
            {
                "file": "torchtitan/experiments/embedding/losses.py",
                "forbidden_snippets": [
                    "pos_scores = (query_embeds * positive_embeds).sum(dim=-1)\n",
                ],
                "required_snippets": [
                    "pos_scores = (query_embeds * positive_embeds).sum(dim=-1) / temperature",
                ],
            }
        ],
    },
    "bad_pooling": {
        "description": "Should use last-token pooling, not first-token pooling",
        "files": [
            {
                "file": "torchtitan/experiments/embedding/embedding_trainer.py",
                "forbidden_snippets": [
                    "last_hidden = hidden[:, 0, :]",
                ],
                "required_snippets": [
                    "seq_lengths = attention_mask.sum(dim=-1) - 1",
                    "batch_indices = torch.arange(hidden.size(0), device=hidden.device)",
                    "last_hidden = hidden[batch_indices, seq_lengths]",
                ],
            }
        ],
    },
    "buggy_projector": {
        "description": "Projector should have a non-linearity between linear layers",
        "files": [
            {
                "file": "torchtitan/experiments/vlm/model/model.py",
                "forbidden_snippets": [
                    "x_NLD = self.w2(self.w1(x_NLD[:, :1, :]))",
                    "x_NLD = x_NLD.expand(-1, seq_len, -1)",
                ],
                "required_snippets": [
                    "x_NLD = self.w1(x_NLD)",
                    "x_NLD = nn.functional.silu(x_NLD)",
                    "x_NLD = self.w2(x_NLD)",
                ],
            }
        ],
    },
    "bad_label_mask": {
        "description": "Special tokens should be masked in labels",
        "files": [
            {
                "file": "torchtitan/experiments/vlm/datasets/mm_datasets.py",
                "forbidden_snippets": [
                    "torch.isin(labels, special_token_ids), labels, special_tokens.ignore_id",
                ],
                "required_snippets": [
                    "special_token_ids = torch.tensor(",
                    "torch.isin(labels, special_token_ids)",
                    "special_tokens.ignore_id",
                ],
            }
        ],
    },
    "flux_causal_attn": {
        "description": "Flux attention blocks should disable causal masking",
        "files": [
            {
                "file": "torchtitan/models/flux/model/layers.py",
                "forbidden_snippets": [
                    "attn = self.inner_attention(q, k, v)\n",
                ],
                "required_snippets": [
                    "attn = self.inner_attention(q, k, v, is_causal=False)\n        attn = rearrange(attn, \"B H L D -> B L (H D)\")\n\n        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]",
                    "q, k = apply_rope(q, k, pe)\n        attn = self.inner_attention(q, k, v, is_causal=False)\n        attn = rearrange(attn, \"B H L D -> B L (H D)\")",
                ],
            }
        ],
    },
    "flux_cfg_dropout": {
        "description": "Flux should drop T5 and CLIP guidance independently",
        "files": [
            {
                "file": "torchtitan/models/flux/flux_datasets.py",
                "forbidden_snippets": [
                    "drop_text = torch.rand(1).item() < dropout_prob",
                    "if drop_text:\n                    sample_dict[\"t5_tokens\"] = self._t5_empty_token\n                    sample_dict[\"clip_tokens\"] = self._clip_empty_token",
                ],
                "required_snippets": [
                    "if torch.rand(1).item() < dropout_prob:\n                    sample_dict[\"t5_tokens\"] = self._t5_empty_token\n                if torch.rand(1).item() < dropout_prob:\n                    sample_dict[\"clip_tokens\"] = self._clip_empty_token",
                ],
            }
        ],
    },
    "flux_zero_timestep": {
        "description": "Flux timestep conditioning should use the real timesteps",
        "files": [
            {
                "file": "torchtitan/models/flux/model/model.py",
                "forbidden_snippets": [
                    "vec = self.time_in(timestep_embedding(torch.zeros_like(timesteps), 256))",
                ],
                "required_snippets": [
                    "vec = self.time_in(timestep_embedding(timesteps, 256))",
                ],
            }
        ],
    },
    "fp16_logits": {
        "description": "Logits should not be cast to float16 before cross-entropy",
        "files": [
            {
                "file": "torchtitan/experiments/embedding/losses.py",
                "forbidden_snippets": [
                    "logits = logits.half()",
                    "logits.half()",
                ],
                "required_snippets": [
                    "return F.cross_entropy(logits, labels)",
                ],
            }
        ],
    },
    "unstable_loss": {
        "description": "Should use F.cross_entropy with standard temperature, not manual log-sum-exp",
        "files": [
            {
                "file": "torchtitan/experiments/embedding/losses.py",
                "forbidden_snippets": [
                    "temperature: float = 0.005",
                    "torch.log(torch.exp(logits).sum(",
                ],
                "required_snippets": [
                    "temperature: float = 0.02",
                    "return F.cross_entropy(logits, labels)",
                ],
            }
        ],
    },
    "quantized_weights": {
        "description": "Weights should not be fp16-quantized between training steps",
        "files": [
            {
                "file": "torchtitan/trainer.py",
                "forbidden_snippets": [
                    "p.data.copy_(p.data.half().float())",
                ],
                "required_snippets": [],
            }
        ],
    },
    "moe_load_balance": {
        "description": "MoE debugmodel should have load balancing enabled",
        "files": [
            {
                "file": "torchtitan/models/deepseek_v3/__init__.py",
                "forbidden_snippets": [
                    "load_balance_coeff=None",
                ],
                "required_snippets": [],
            }
        ],
    },
}

check = MUTATIONS.get(mutation_type)
if not check:
    print(f"Unknown mutation: {mutation_type}")
    sys.exit(1)

for file_check in check["files"]:
    filepath = os.path.join(workspace, file_check["file"])
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    with open(filepath) as f:
        code = f.read()

    present = [s for s in file_check["forbidden_snippets"] if s in code]
    if present:
        print(f"Bug NOT fixed: {check['description']}")
        print(f"File: {file_check['file']}")
        print("Found forbidden snippets:")
        for snippet in present:
            print(f"  - {snippet}")
        sys.exit(1)

    missing = [s for s in file_check["required_snippets"] if s not in code]
    if missing:
        print(f"Fix not detected: {check['description']}")
        print(f"File: {file_check['file']}")
        print("Missing required snippets:")
        for snippet in missing:
            print(f"  - {snippet}")
        sys.exit(1)

print(f"Code fix verified: {check['description']}")
sys.exit(0)
