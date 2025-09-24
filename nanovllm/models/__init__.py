from .qwen3 import Qwen3ForCausalLM
from .qwen2 import Qwen2ForCausalLM

model_cls = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen2ForCausalLM": Qwen2ForCausalLM,
}


def get_model(hf_config):
    return model_cls[hf_config.architectures[0]](hf_config)