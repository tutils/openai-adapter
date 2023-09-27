import adapter
from adapter import ChatCompletion, create_chat_completion_service
import uvicorn


# use = "baichuan2"
# use = "chatglm2"
use = "qwen"

service_args = {
    "baichuan2": ["../Baichuan2/baichuan-inc/Baichuan2-13B-Chat-4bits", True],
    "chatglm2": ["../ChatGLM2-6B/THUDM/chatglm2-6b"],
    "qwen": ["../Qwen/Qwen/Qwen-14B-Chat-Int4", True],
}


def service_loader() -> ChatCompletion:
    print("init service ...")
    service = create_chat_completion_service(use, *service_args[use])
    print("init service done")
    return service


adapter.set_service_loader(service_loader)
app = adapter.app

if __name__ == "__main__":
    """
    run on the shell:
    PYTHONPATH=/this/repo/path python api_server.py
    or
    PYTHONPATH=/this/repo/path uvicorn api_server:app --host 0.0.0.0 --reload
    """
    uvicorn.run(app, host="0.0.0.0")
