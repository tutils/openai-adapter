from typing import List, Dict, Iterator, Literal
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.generation.utils import GenerationConfig
from adapter.chat_completion import ChatCompletion, ChatMessage, register_chat_completion_service


SERVICE_NAME = "baichuan2"


class Baichuan2ChatCompletion(ChatCompletion):
    _overwrite_system: bool

    def __init__(self, model_path: str, overwrite_system: bool = False) -> None:
        self._overwrite_system = overwrite_system
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
        system = ""
        wait_user = True
        user, assistant = "", ""
        for message in messages:
            if message.role == "system":
                if system == "" or self._overwrite_system:
                    system = message.content
                else:
                    system += "\n" + message.content
                continue

            if wait_user:
                # appending user
                if message.role == "user":
                    if user == "":
                        user = message.content
                    else:
                        user += "\n" + message.content
                else:
                    wait_user = False
                    assistant = message.content
            else:
                # appending assistant
                if message.role == "user":
                    msgs.extend([{
                        "role": "user",
                        "content": user,
                    }, {
                        "role": "assistant",
                        "content": assistant,
                    }])
                    wait_user = True
                    user, assistant = message.content, ""
                else:
                    if assistant == "":
                        assistant = message.content
                    else:
                        assistant += "\n" + message.content

        if wait_user:
            msgs.append({
                "role": "user",
                "content": user,
            })
        else:
            msgs.extend([{
                "role": "user",
                "content": user,
            }, {
                "role": "assistant",
                "content": assistant,
            }])

        if system != "":
            msgs.insert(0, {
                "role": "system",
                "content": system,
            })

        # print(f"<msgs>\n{msgs}")
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


register_chat_completion_service(SERVICE_NAME, Baichuan2ChatCompletion)
