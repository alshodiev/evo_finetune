from transformers import AutoConfig, AutoModelForCausalLM, AdamW
import torch
from flash_attn.modules.mha import MHA
from positional_embeddings import swap_mha_rope
from evo.evo.models import Evo



def load_model(model_name, revision="1.1_fix", device='cuda:0'):
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, revision=revision)
    model_config.use_cache = True
    # Ensure you load the model properly, and use the swap_mha_rope as necessary
    model = Evo.from_pretrained(
        model_name,
        config=model_config,
        trust_remote_code=True,
        revision=revision
    )
    model.to(device)
    return model

def setup_optimizer(model, learning_rate=0.001):
    return AdamW(model.parameters(), lr=learning_rate)
