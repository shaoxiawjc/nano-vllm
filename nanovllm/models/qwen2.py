from .qwen3 import Qwen3MLP
import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen2Config, Qwen3Config
from .qwen3 import Qwen3ForCausalLM


class Qwen2ForCausalLM(Qwen3ForCausalLM):

    def __init__(self, config: Qwen2Config):
        config_ = Qwen3Config(
            attention_dropout=config.attention_dropout,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            head_dim=getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads),
            hidden_act=config.hidden_act,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            max_window_layers=config.max_window_layers,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            rope_scaling=getattr(config, "rope_scaling", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            sliding_window = getattr(config, "sliding_window", False),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", True),
            use_sliding_window=getattr(config, "use_sliding_window", False),
            vocab_size=config.vocab_size,
            torch_dtype=config.torch_dtype,
            attention_bias=getattr(config, 'attention_bias', False),
        )
        super().__init__(config_)


