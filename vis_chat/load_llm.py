from typing import Optional

import torch
import guidance
from guidance.llms.transformers import LLaMA
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel


class Vicuna(LLaMA):
    """ A HuggingFace transformers version of the Vicuna language model with Guidance support.
    """

    llm_name: str = "vicuna"

    default_system_prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )

    @staticmethod
    def role_start(role):
        if role == 'user':
            return 'USER: '
        elif role == 'assistant':
            return 'ASSISTANT: '
        else:
            return ''

    @staticmethod
    def role_end(role):
        if role == 'user':
            return ' '
        elif role == 'assistant':
            return '</s>'
        elif role == 'system':
            return '\n\n'
        else:
            return ''


def load_llm(
    load_8bit: bool = True,
    base_model: str = "lmsys/vicuna-7b-v1.3",
    lora_weights: Optional[str] = None
):
    # we use Vicuna here, but any LLaMA-style chat model will do
    tokenizer = LlamaTokenizer.from_pretrained(base_model, legacy=False)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if lora_weights is not None:
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    if 'vicuna' not in base_model:
        llm = guidance.llms.Transformers(model=model, tokenizer=tokenizer, device=None)
    else:
        llm = Vicuna(model=model, tokenizer=tokenizer, device=None)
    return llm
