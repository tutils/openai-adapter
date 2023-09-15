from typing import List
from typing import List, Iterator, Literal
from abc import ABC, abstractmethod
from transformers import PreTrainedModel, PreTrainedTokenizer
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletion(ABC):
    _model: PreTrainedModel
    _tokenizer: PreTrainedTokenizer

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self._model = model
        self._tokenizer = tokenizer

    @abstractmethod
    def chat(self, messages: List[ChatMessage]) -> str:
        pass

    @abstractmethod
    def chat_stream(self, messages: List[ChatMessage]) -> Iterator[str]:
        pass

    def num_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def num_tokens_from_messages(self, messages: List[ChatMessage]) -> int:
        res: int = 0
        for message in messages:
            res += self.num_tokens(message.role)
            res += self.num_tokens(message.content)
        return res
