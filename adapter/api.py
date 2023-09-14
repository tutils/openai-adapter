import time
from typing import List, Optional
from typing import Any, Dict, List, Iterator, Optional, Union, Callable, Literal
from pydantic import BaseModel
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from adapter.chat_completion import ChatCompletion, ChatMessage


class ChatCompletionFunctionCall(BaseModel):
    name: str
    arguments: str  # Arguments in JSON format


class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function"]
    content: Optional[str] = None  # content can be a string or null
    name: Optional[str] = None
    function_call: Optional[ChatCompletionFunctionCall] = None


class ChatCompletionFunctionParam(BaseModel):
    type: str
    properties: Dict[str, Any]
    required: Optional[List[str]]


class ChatCompletionFunction(BaseModel):
    name: str
    description: Optional[str]
    parameters: Union[ChatCompletionFunctionParam, Any]  # JSON Schema object


class ChatCompletionRequest(BaseModel):
    """
    see https://platform.openai.com/docs/api-reference/chat/create
    """
    model: str
    messages: List[ChatCompletionMessage]
    functions: Optional[List[ChatCompletionFunction]] = None
    function_call: Optional[str] = None
    temperature: Optional[float] = 1.0  # defaults to 1.0
    top_p: Optional[float] = 1.0  # defaults to 1.0
    n: Optional[int] = 1  # defaults to 1
    stream: Optional[bool] = False  # defaults to False
    stop: Optional[Union[str, list]] = None  # defaults to None
    max_tokens: Optional[int] = 4096  # defaults to inf
    presence_penalty: Optional[float] = 0  # defaults to 0
    frequency_penalty: Optional[float] = 0  # defaults to 0
    logit_bias: Optional[Dict[int, int]] = None  # defaults to None
    user: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionMessageRoleOnly(BaseModel):
    role: Literal["assistant", "function"]


class ChatCompletionMessageContentOnly(BaseModel):
    content: Optional[str] = None  # content can be a string or null


class ChatCompletionMessageEmpty(BaseModel):
    pass


class ChatCompletionChoiceDelta(BaseModel):
    index: int
    delta: Union[ChatCompletionMessageRoleOnly,
                 ChatCompletionMessageContentOnly, ChatCompletionMessageEmpty]
    finish_reason: Optional[str]


class ChatCompletionStreamingResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChoiceDelta]


class Model(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[Model]


_service: ChatCompletion = None

_service_loader: Callable[[], ChatCompletion] = None


def set_service_loader(loader: Callable[[], ChatCompletion]) -> None:
    assert isinstance(
        loader, Callable), f"invalid service_loader: {loader}"
    global _service_loader
    _service_loader = loader


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    assert isinstance(
        _service_loader, Callable), f"set service loader by calling set_service_loader(<loader>) first"
    _service = _service_loader()
    assert isinstance(
        _service, ChatCompletion), f"service_loader() return invalid serivce: {_service}"

    yield

    # collects GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cur_req_id: int = 1000


def gen_req_id() -> str:
    global _cur_req_id
    id = _cur_req_id
    _cur_req_id += 1
    return f"chatcmpl-{id}"


@app.get("/v1/models")
async def models() -> ModelsResponse:
    rsp = ModelsResponse(
        data=[
            Model(
                id="gpt-3.5-turbo",
                created=1686935002,
                owned_by="openai",
            ),
        ],
    )
    return rsp


def build_chat_compl_resp(id: str, model: str, messages) -> ChatCompletionResponse:
    completion = _service.chat(messages)

    prompt_tokens = _service.num_tokens_from_messages(messages)
    completion_tokens = _service.num_tokens(completion)
    total_tokens = prompt_tokens+completion_tokens

    rsp = ChatCompletionResponse(
        id=id,
        created=int(time.time()),
        model=model,
        choices=[ChatCompletionChoice(
            index=0,
            message=ChatCompletionMessage(
                role="assistant",
                content=completion,
            ),
            finish_reason="stop",
        )],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )
    return rsp


def build_chat_compl_streaming_resp(id: str, model: str, messages) -> Iterator[str]:
    completion_gen = _service.chat_stream(messages)
    rsp = ChatCompletionStreamingResponse(
        id=id,
        created=int(time.time()),
        model=model,
        choices=[ChatCompletionChoiceDelta(
            index=0,
            delta=ChatCompletionMessageRoleOnly(
                role="assistant",
            ),
            finish_reason=None,
        )],
    )
    yield rsp.model_dump_json()

    for delta_compl in completion_gen:
        if delta_compl != "":
            delta = ChatCompletionMessageContentOnly(
                content=delta_compl,
            )
        else:
            delta = ChatCompletionMessageEmpty()

        rsp = ChatCompletionStreamingResponse(
            id=id,
            created=int(time.time()),
            model=model,
            choices=[ChatCompletionChoiceDelta(
                index=0,
                delta=delta,
                finish_reason=None,
            )],
        )
        yield rsp.model_dump_json()

    yield "[DONE]"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> Union[ChatCompletionResponse, str]:
    id = gen_req_id()
    model = req.model
    messages: List[ChatMessage] = []

    print(f"<|request: {id}, promt|>")
    for message in req.messages:
        messages.append(ChatMessage(
            role=message.role,
            content=message.content,
        ))
        print(f"<|role: {message.role}|>")
        print(f"{message.content}\n")
    print(f"<|response: {id}, completion|>")

    if req.stream:
        return EventSourceResponse(
            build_chat_compl_streaming_resp(id, model, messages))

    return build_chat_compl_resp(id, model, messages)
