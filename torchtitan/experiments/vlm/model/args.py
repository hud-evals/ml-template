# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.components.tokenizer import HuggingFaceTokenizer


@dataclass
class VLMTokenNames:
    """Token string names the VLM pipeline uses for each role.

    Different tokenizer families ship different token names for equivalent
    concepts.  Set these per-model in the config registry so the mapping is
    explicit rather than discovered at runtime.
    """

    img: str = "<|image_pad|>"
    boi: str = "<|vision_start|>"
    eoi: str = "<|vision_end|>"
    pad: str = "<|vision_pad|>"


@dataclass
class SpecialTokens:
    img_token: str
    img_id: int
    boi_token: str
    boi_id: int
    eoi_token: str
    eoi_id: int
    pad_token: str
    pad_id: int
    ignore_id: int = -100  # Pytorch F.cross_entropy default

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: HuggingFaceTokenizer,
        names: VLMTokenNames,
    ) -> "SpecialTokens":
        added_tokens = tokenizer.tokenizer.get_added_tokens_decoder()
        token_to_id = {tok.content: tok_id for tok_id, tok in added_tokens.items()}

        def _resolve(role: str, name: str) -> tuple[str, int]:
            if name not in token_to_id:
                raise KeyError(
                    f"VLM special token '{role}' ({name!r}) not found in "
                    f"tokenizer added tokens. Available: "
                    f"{sorted(t.content for t in added_tokens.values())}"
                )
            return name, token_to_id[name]

        img_tok, img_id = _resolve("img", names.img)
        boi_tok, boi_id = _resolve("boi", names.boi)
        eoi_tok, eoi_id = _resolve("eoi", names.eoi)
        pad_tok, pad_id = _resolve("pad", names.pad)

        return cls(
            img_token=img_tok, img_id=img_id,
            boi_token=boi_tok, boi_id=boi_id,
            eoi_token=eoi_tok, eoi_id=eoi_id,
            pad_token=pad_tok, pad_id=pad_id,
        )


@dataclass
class Siglip2Config:
    dim: int = 768
    ffn_dim: int = 3072
    n_layers: int = 12
    n_heads: int = 12

    n_pos_embs: int = 16  # Number of positional embeddings per h&w
    n_channels: int = 3  # RGB channels
    patch_size: int = 16
    spatial_merge_size: int = 1

    layer_norm_eps: float = 1e-6
    attn_backend: str = "flex"
    attn_mask_type: str = "causal"
