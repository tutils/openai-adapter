from typing import List, Tuple, Dict, Iterator, Literal
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from adapter.chat_completion import ChatCompletion, ChatMessage, register_chat_completion_service


SERVICE_NAME = "chatglm2"


class ChatGLM2ChatCompletion(ChatCompletion):
    def __init__(self, model_path: str) -> None:
        model: PreTrainedModel = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
        ).cuda().eval()
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        super().__init__(model, tokenizer)

    def _build_input(self, messages: List[ChatMessage]) -> Tuple[str, List[Tuple[str, str]]]:
        query = ""
        history: List[Tuple[str, str]] = []
        wait_user = True
        user, assistant = "", ""
        for message in messages:
            if wait_user:
                # appending user
                if message.role in ["user", "system"]:
                    if user == "":
                        user = message.content
                    else:
                        user += "\n" + message.content
                else:
                    wait_user = False
                    assistant = message.content
            else:
                # appending assistant
                if message.role in ["user", "system"]:
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

        # print(f"<query>\n{query}\n<history>\n{history}")
        return query, history

    def chat(self, messages: List[ChatMessage]) -> str:
        query, history = self._build_input(messages)
        completion, _ = self._model.chat(
            self._tokenizer,
            query,
            history,
        )
        # print(completion)
        return completion

    def chat_stream(self, messages: List[ChatMessage]) -> Iterator[str]:
        query, history = self._build_input(messages)
        completion_gen = self._model.stream_chat(
            self._tokenizer,
            query,
            history,
        )
        position = 0
        for completion, _ in completion_gen:
            delta = completion[position:]
            # print(delta, end='', flush=True)
            position = len(completion)
            yield delta
        # print()


register_chat_completion_service(SERVICE_NAME, ChatGLM2ChatCompletion)
