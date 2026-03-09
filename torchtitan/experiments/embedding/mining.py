"""Hard negative mining and false negative filtering."""

import torch
import torch.nn.functional as F


def mine_hard_negatives(
    query_embeds: torch.Tensor,
    corpus_embeds: torch.Tensor,
    positive_indices: list[int],
    num_negatives: int = 7,
) -> list[list[int]]:
    query_embeds = F.normalize(query_embeds, dim=-1)
    corpus_embeds = F.normalize(corpus_embeds, dim=-1)

    sim_matrix = torch.mm(query_embeds, corpus_embeds.t())

    for i, pos_idx in enumerate(positive_indices):
        sim_matrix[i, pos_idx] = float("-inf")

    _, topk_indices = sim_matrix.topk(num_negatives, dim=-1)
    return topk_indices.tolist()


def filter_false_negatives(
    query_embeds: torch.Tensor,
    positive_embeds: torch.Tensor,
    negative_embeds: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    query_embeds = F.normalize(query_embeds, dim=-1)
    positive_embeds = F.normalize(positive_embeds, dim=-1)
    negative_embeds = F.normalize(negative_embeds, dim=-1)

    pos_sim = (query_embeds * positive_embeds).sum(dim=-1)

    neg_sim = torch.bmm(
        negative_embeds, query_embeds.unsqueeze(-1)
    ).squeeze(-1)

    return neg_sim > (pos_sim.unsqueeze(-1) + threshold)
