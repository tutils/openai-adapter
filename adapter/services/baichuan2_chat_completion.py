from typing import List, Dict, Iterator, Literal
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.generation.utils import GenerationConfig
from adapter.chat_completion import ChatCompletion, ChatMessage


class Baichuan2ChatCompletion(ChatCompletion):
    def __init__(self, model_path: str) -> None:
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.generation_config = GenerationConfig.from_pretrained(
            model_path)
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True,
        )
        super().__init__(model, tokenizer)

    def _build_input(self, messages: List[ChatMessage]) -> List[Dict[Literal["role", "content"], str]]:
        msgs: List[Dict[Literal["role", "content"], str]] = []
        for message in messages:
            msgs.append({
                "role": message.role,
                "content": message.content,
            })
        return msgs

    def chat(self, messages: List[ChatMessage]) -> str:
        msgs = self._build_input(messages)
        completion = self._model.chat(
            self._tokenizer,
            msgs,
            stream=False,
        )
        # print(completion)
        return completion

    def chat_stream(self, messages: List[ChatMessage]) -> Iterator[str]:
        msgs = self._build_input(messages)
        completion_gen = self._model.chat(
            self._tokenizer,
            msgs,
            stream=True,
        )
        position = 0
        for completion in completion_gen:
            delta = completion[position:]
            # print(delta, end='', flush=True)
            position = len(completion)
            yield delta
        # print()
