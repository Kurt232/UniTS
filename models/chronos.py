import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
)
from transformers.models.t5.modeling_t5 import T5Stack

from typing import Tuple, Optional, Dict, Any, Literal
from dataclasses import dataclass
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from models.units import MLPBlock, Mlp, CrossAttention

class CLSHead(nn.Module):
    def __init__(self, d_model, head_dropout=0):
        super().__init__()
        d_mid = d_model
        self.proj_in = nn.Linear(d_model, d_mid)
        self.cross_att = CrossAttention(d_mid)

        self.mlp = MLPBlock(dim=d_mid, mlp_ratio=8, mlp_layer=Mlp,
                            proj_drop=head_dropout, init_values=None, drop_path=0.0,
                            act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                            prefix_token_length=None)

    def forward(self, x, category_token=None, return_feature=False):
        x = self.proj_in(x)
        B, V, C = x.shape
        cls_token = x[:, -1:] # [B*V, 1, C]
        cls_token = self.cross_att(x, query=cls_token) # [B, 1, C]

        cls_token = self.mlp(cls_token)
        if return_feature:
            return cls_token
        m = category_token.shape[-2]
        cls_token = cls_token.expand(B, m, C)
        distance = torch.einsum('nkc,nmc->nm', cls_token, category_token)

        return distance

class Chronos(nn.Module):
    def __init__(self, num_class, config):
        super().__init__()
        self.d_model = config.d_model

        config.is_decoder = False
        config.use_cache = False
        config.is_encoder_decoder = False
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(config, self.shared)

        self.category_tokens = nn.Parameter(torch.zeros(1, num_class, config.d_model))
        nn.init.normal_(self.category_tokens, std=0.02)

        self.cls_head = CLSHead(config.d_model, head_dropout=0.1)

        chronos_config = ChronosConfig(**config.chronos_config)
        self.tokenizer = MeanScaleUniformBins(
            **chronos_config.tokenizer_kwargs,
            config=chronos_config,
        )

    def forward(self, x, attention_mask=None):
        N, V, L = x.shape
        x = x.reshape(N*V, L)
        embeddings: BaseModelOutputWithPastAndCrossAttentions = self.encoder(
            input_ids=x,
            attention_mask=attention_mask,
        )
        x = embeddings.last_hidden_state
        x = x.reshape(N, -1, self.d_model)

        distance = self.cls_head(x, self.category_tokens) # [N, num_class]
        return distance


@dataclass
class ChronosConfig:
    """
    This class holds all the configuration parameters to be used
    by ``ChronosTokenizer`` and ``ChronosModel``.
    """

    tokenizer_class: str
    tokenizer_kwargs: Dict[str, Any]
    context_length: int
    prediction_length: int
    n_tokens: int
    n_special_tokens: int
    pad_token_id: int
    eos_token_id: int
    use_eos_token: bool
    model_type: Literal["causal", "seq2seq"]
    num_samples: int
    temperature: float
    top_k: int
    top_p: float

    def __post_init__(self):
        assert (
            self.pad_token_id < self.n_special_tokens
            and self.eos_token_id < self.n_special_tokens
        ), f"Special token id's must be smaller than {self.n_special_tokens=}"


class MeanScaleUniformBins():
    def __init__(
        self, low_limit: float, high_limit: float, config: ChronosConfig
    ) -> None:
        self.config = config
        self.centers = torch.linspace(
            low_limit,
            high_limit,
            config.n_tokens - config.n_special_tokens - 1,
        )
        self.boundaries = torch.concat(
            (
                torch.tensor([-1e20], device=self.centers.device),
                (self.centers[1:] + self.centers[:-1]) / 2,
                torch.tensor([1e20], device=self.centers.device),
            )
        )

    def _input_transform(
        self, context: torch.Tensor, scale: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = context.to(dtype=torch.float32)
        attention_mask = ~torch.isnan(context)

        if scale is None:
            scale = torch.nansum(
                torch.abs(context) * attention_mask, dim=-1
            ) / torch.nansum(attention_mask, dim=-1)
            scale[~(scale > 0)] = 1.0

        scaled_context = context / scale.unsqueeze(dim=-1)
        token_ids = (
            torch.bucketize(
                input=scaled_context,
                boundaries=self.boundaries,
                # buckets are open to the right, see:
                # https://pytorch.org/docs/2.1/generated/torch.bucketize.html#torch-bucketize
                right=True,
            )
            + self.config.n_special_tokens
        )

        token_ids.clamp_(0, self.config.n_tokens - 1)

        token_ids[~attention_mask] = self.config.pad_token_id

        return token_ids, attention_mask, scale

    def _append_eos_token(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = token_ids.shape[0]
        eos_tokens = torch.full((batch_size, 1), fill_value=self.config.eos_token_id)
        token_ids = torch.concat((token_ids, eos_tokens), dim=1)
        eos_mask = torch.full((batch_size, 1), fill_value=True)
        attention_mask = torch.concat((attention_mask, eos_mask), dim=1)

        return token_ids, attention_mask

    def context_input_transform(
        self, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = context.shape[-1]

        if length > self.config.context_length:
            context = context[..., -self.config.context_length :]

        token_ids, attention_mask, scale = self._input_transform(context=context)

        if self.config.use_eos_token and self.config.model_type == "seq2seq":
            token_ids, attention_mask = self._append_eos_token(
                token_ids=token_ids, attention_mask=attention_mask
            )

        return token_ids, attention_mask, scale

    def label_input_transform(
        self, label: torch.Tensor, scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        length = label.shape[-1]

        assert length == self.config.prediction_length
        token_ids, attention_mask, _ = self._input_transform(context=label, scale=scale)

        if self.config.use_eos_token:
            token_ids, attention_mask = self._append_eos_token(
                token_ids=token_ids, attention_mask=attention_mask
            )

        return token_ids, attention_mask

    def output_transform(
        self, samples: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        scale_unsqueezed = scale.unsqueeze(-1).unsqueeze(-1)
        indices = torch.clamp(
            samples - self.config.n_special_tokens - 1,
            min=0,
            max=len(self.centers) - 1,
        )
        return self.centers[indices] * scale_unsqueezed

if __name__ == "__main__":
    device = 'cuda'
    config = AutoConfig.from_pretrained(
    "amazon/chronos-t5-small",
    )
    model = Chronos(7, config).to(device)
    encoder_weights = torch.load('chronos-t5-small-encoder.pth', map_location='cpu')
    msg = model.load_state_dict(encoder_weights, strict=False)
    print(msg)

    # test
    state_dict = model.state_dict()
    # print(state_dict['shared.weight'])
    # print(state_dict['encoder.block.0.layer.0.SelfAttention.q.weight'])

    imu_input = torch.randn(1, 100, 6)
    tokenizer = model.tokenizer
    N, L, V = imu_input.shape
    imu_input = imu_input.permute(0, 2, 1).reshape(N*V, L)
    input_ids, mask, _ = tokenizer.context_input_transform(imu_input)
    input_ids = input_ids.reshape(N, V, -1).to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    output = model(input_ids, mask)
    print(output.shape)

    # num of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_params/1e6:.4f}M') # 26.4942M small

