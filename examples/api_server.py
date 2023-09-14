import adapter
from adapter import ChatCompletion, Baichuan2ChatCompletion, ChatGLM2ChatCompletion
import uvicorn

use = "baichuan2"
# use = "chatglm2"


def service_loader() -> ChatCompletion:
    print("init service ...")
    if use == "baichuan2":
        serivce = Baichuan2ChatCompletion(
            "../Baichuan2/baichuan-inc/Baichuan2-13B-Chat-4bits")
    elif use == "chatglm2":
        serivce = ChatGLM2ChatCompletion(
            "../ChatGLM2-6B/THUDM/chatglm2-6b")
    print("init service done")
    return serivce


adapter.set_service_loader(service_loader)
app = adapter.app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
