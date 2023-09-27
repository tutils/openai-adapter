from typing import List, Tuple, Iterator
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.generation.utils import GenerationConfig
from adapter.chat_completion import ChatCompletion, ChatMessage, register_chat_completion_service


SERVICE_NAME = "qwen"


class QwenChatCompletion(ChatCompletion):
    _overwrite_system: bool

    def __init__(self, model_path: str, overwrite_system: bool = False) -> None:
        self._overwrite_system = overwrite_system
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        model.generation_config = GenerationConfig.from_pretrained(
            model_path, trust_remote_code=True)
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        super().__init__(model, tokenizer)

    def _build_input(self, messages: List[ChatMessage]) -> Tuple[str, List[Tuple[str, str]], str]:
        query = ""
        history: List[Tuple[str, str]] = []
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
                    history.append((user, assistant))
                    wait_user = True
                    user, assistant = message.content, ""
                else:
                    if assistant == "":
                        assistant = message.content
                    else:
                        assistant += "\n" + message.content

        if wait_user:
            query = user
        else:
            history.append((user, assistant))

        if system == "":
            system = "You are a helpful assistant."

        # print(f"<query>\n{query}\n<history>\n{history}\n<system>\n{system}")
        return query, history, system

    def chat(self, messages: List[ChatMessage]) -> str:
        query, history, system = self._build_input(messages)
        completion, _ = self._model.chat(
            self._tokenizer,
            query,
            history=history,
            system=system,
            append_history=False,
        )
        # print(completion)
        return completion

    def chat_stream(self, messages: List[ChatMessage]) -> Iterator[str]:
        query, history, system = self._build_input(messages)
        completion_gen = self._model.chat_stream(
            self._tokenizer,
            query,
            history=history,
            system=system,
        )
        position = 0
        for completion in completion_gen:
            delta = completion[position:]
            # print(delta, end='', flush=True)
            position = len(completion)
            yield delta
        # print()


register_chat_completion_service(SERVICE_NAME, QwenChatCompletion)
