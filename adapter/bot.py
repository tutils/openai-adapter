from typing import List, Iterator
from adapter.chat_completion import ChatCompletion, ChatMessage


class ChatBot:
    _service: ChatCompletion
    _system: str
    _messages: List[ChatMessage]

    def __init__(self, service: ChatCompletion, system: str = "", history: List[ChatMessage] = []) -> None:
        self._service = service
        self._system = system
        if system == "":
            self._messages = []
        else:
            self._messages = [ChatMessage(
                role="system",
                content=system,
            )]
        if history:
            self._messages = history
            if not self._system and history[0].role == "system":
                self._system = history[0].content

    def chat(self, message: str) -> str:
        self._messages.append(ChatMessage(
            role="user",
            content=message,
        ))
        completion = self._service.chat(self._messages)
        self._messages.append(ChatMessage(
            role="assistant",
            content=completion,
        ))
        return completion

    def chat_stream(self, message: str) -> Iterator[str]:
        self._messages.append(ChatMessage(
            role="user",
            content=message,
        ))
        completion_gen = self._service.chat_stream(self._messages)
        delta_compl_list: List[str] = []
        for delta_compl in completion_gen:
            delta_compl_list.append(delta_compl)
            yield delta_compl
        self._messages.append(ChatMessage(
            role="assistant",
            content="".join(delta_compl_list),
        ))

    def clear(self):
        if self._system == "":
            self._messages = []
        else:
            self._messages = [ChatMessage(
                role="system",
                content=self._system,
            )]

    @property
    def history(self) -> List[ChatMessage]:
        return self._messages
