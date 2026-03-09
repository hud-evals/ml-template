# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import einops as E
import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.llama3 import Llama3Model as Llama3
from torchtitan.models.qwen3 import Qwen3Model

from .args import Siglip2Config, SpecialTokens
from .siglip2 import VisionTransformer


def _scatter_img_tokens(h_BSD, tokens_BS, i_NLD, i_mask_NL, img_id):
    B, S, D = h_BSD.shape
    # Where are the image tokens in LLM input, make broadcastable with h_BSD
    img_mask_h_BSD = E.repeat(tokens_BS == img_id, "b s -> b s 1")
    # Only get valid (non-padded) tokens, result are flatten
    i_flatten = torch.masked_select(i_NLD, mask=i_mask_NL.unsqueeze(-1))

    assert i_flatten.numel() // D == img_mask_h_BSD.sum(), (
        f"Different number of visual embeddings {i_flatten.numel() // D} "
        f"with placeholder in input token embeddings {img_mask_h_BSD.sum()}"
    )
    h_BSD.masked_scatter_(mask=img_mask_h_BSD, source=i_flatten.to(h_BSD.dtype))
    return h_BSD


class Projector(nn.Module):
    """Project the Encoder embedding to the LLM embedding."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_dim, in_dim)
        self.w2 = nn.Linear(in_dim, out_dim)
        self.init_weights()

    def forward(self, x_NLD: torch.Tensor):
        x_NLD = self.w1(x_NLD)
        x_NLD = nn.functional.silu(x_NLD)
        x_NLD = self.w2(x_NLD)
        return x_NLD

    def init_weights(self):
        nn.init.xavier_uniform_(self.w1.weight)
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        nn.init.xavier_uniform_(self.w2.weight)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)


class _VLMMixin:
    """Shared VLM logic for encoder + projector on any Decoder backbone."""

    def _vlm_init(self, config):
        self.encoder = VisionTransformer(config.encoder)
        self.projector = Projector(in_dim=config.encoder.dim, out_dim=config.dim)

    def _vlm_init_weights(self):
        if self.encoder is not None:
            self.encoder.init_weights()
        if self.projector is not None:
            self.projector.init_weights()

    def _vlm_get_attention_masks(self, super_masks, input_batch, tokenizer, extra_inputs):
        assert isinstance(super_masks, BlockMask)
        encoder_masks = self.encoder.get_attention_masks(
            input_batch, tokenizer, extra_inputs
        )
        assert isinstance(encoder_masks, BlockMask)
        return {"llm_masks": super_masks, "encoder_masks": encoder_masks}

    def _vlm_forward(self, tokens, pixel_values, grid_thw, special_tokens, attention_masks):
        h_BSD = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        if self.encoder is not None:
            assert attention_masks is not None, (
                "encoder only allows FlexAttention, so the LLM must use FlexAttention as well."
            )
            grid_hw = grid_thw[:, :, 1:]
            pixel_masks = E.reduce(grid_hw != -1, "n l hw -> n l", reduction="all")
            i_NLD = self.encoder(
                pixel_values, pixel_masks, grid_hw, attention_masks["encoder_masks"]
            )
            i_NLD = self.projector(i_NLD)
            h_BSD = _scatter_img_tokens(
                h_BSD, tokens, i_NLD, pixel_masks, special_tokens.img_id
            )

        for layer in self.layers.values():
            h_BSD = layer(h_BSD, self.freqs_cis, attention_masks["llm_masks"])

        h_BSD = self.norm(h_BSD) if self.norm else h_BSD
        output = self.output(h_BSD) if self.output else h_BSD
        return output


class Llama3Siglip2Transformer(_VLMMixin, Llama3):
    @dataclass(kw_only=True, slots=True)
    class Config(Llama3.Config):
        encoder: Siglip2Config = field(default_factory=Siglip2Config)

    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config
        self._vlm_init(config)

    def init_weights(self, buffer_device=None):
        super().init_weights(buffer_device=buffer_device)
        self._vlm_init_weights()

    def get_attention_masks(self, input_batch, tokenizer, extra_inputs=None):
        masks = super().get_attention_masks(input_batch, tokenizer, extra_inputs)
        return self._vlm_get_attention_masks(masks, input_batch, tokenizer, extra_inputs)

    def forward(self, tokens, pixel_values, grid_thw, special_tokens, attention_masks=None):
        return self._vlm_forward(tokens, pixel_values, grid_thw, special_tokens, attention_masks)


class Qwen3Siglip2Transformer(_VLMMixin, Qwen3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(Qwen3Model.Config):
        encoder: Siglip2Config = field(default_factory=Siglip2Config)

    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config
        self._vlm_init(config)

    def init_weights(self, buffer_device=None):
        super().init_weights(buffer_device=buffer_device)
        self._vlm_init_weights()

    def get_attention_masks(self, input_batch, tokenizer, extra_inputs=None):
        masks = super().get_attention_masks(input_batch, tokenizer, extra_inputs)
        return self._vlm_get_attention_masks(masks, input_batch, tokenizer, extra_inputs)

    def forward(self, tokens, pixel_values, grid_thw, special_tokens, attention_masks=None):
        return self._vlm_forward(tokens, pixel_values, grid_thw, special_tokens, attention_masks)
