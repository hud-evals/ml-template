"""Contrastive losses for embedding model training."""

import torch
import torch.nn.functional as F


def infonce_loss(
    query_embeds: torch.Tensor,
    positive_embeds: torch.Tensor,
    negative_embeds: torch.Tensor,
    temperature: float = 0.02,
    false_neg_threshold: float = 0.1,
) -> torch.Tensor:
    """InfoNCE contrastive loss with in-batch negatives and false negative masking."""
    B, D = query_embeds.shape

    query_embeds = F.normalize(query_embeds, dim=-1)
    positive_embeds = F.normalize(positive_embeds, dim=-1)
    negative_embeds = F.normalize(negative_embeds, dim=-1)

    pos_scores = (query_embeds * positive_embeds).sum(dim=-1) / temperature

    hard_neg_scores = (
        torch.bmm(negative_embeds, query_embeds.unsqueeze(-1)).squeeze(-1) / temperature
    )

    in_batch_scores = torch.mm(query_embeds, positive_embeds.t()) / temperature

    with torch.no_grad():
        pos_sim = (query_embeds * positive_embeds).sum(dim=-1, keepdim=True)
        in_batch_sim = torch.mm(query_embeds, positive_embeds.t())
        false_neg_mask = in_batch_sim > pos_sim + false_neg_threshold
        false_neg_mask.fill_diagonal_(True)

    in_batch_scores = in_batch_scores.masked_fill(false_neg_mask, float("-inf"))

    all_neg_scores = torch.cat([hard_neg_scores, in_batch_scores], dim=-1)
    logits = torch.cat([pos_scores.unsqueeze(-1), all_neg_scores], dim=-1)

    labels = torch.zeros(B, dtype=torch.long, device=query_embeds.device)
    return F.cross_entropy(logits, labels)


def matryoshka_loss(
    query_embeds: torch.Tensor,
    positive_embeds: torch.Tensor,
    negative_embeds: torch.Tensor,
    dims: list[int],
    temperature: float = 0.02,
    false_neg_threshold: float = 0.1,
) -> torch.Tensor:
    """Matryoshka Representation Learning loss."""
    total_loss = torch.tensor(0.0, device=query_embeds.device)

    for dim in dims:
        q = query_embeds[:, :dim]
        p = positive_embeds[:, :dim]
        n = negative_embeds[:, :, :dim]
        total_loss = total_loss + infonce_loss(
            q, p, n, temperature, false_neg_threshold
        )

    return total_loss / len(dims)


def build_infonce_loss(compile_config=None, parallel_dims=None):
    """Builder matching torchtitan's LossFunctionBuilder signature."""
    return infonce_loss
