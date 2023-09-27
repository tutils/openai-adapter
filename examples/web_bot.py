import os
import torch
import streamlit as st
from adapter import ChatBot, ChatCompletion, create_chat_completion_service


use_service = os.environ.get("USE_SERVICE")
service_params = {
    "baichuan2": ["../Baichuan2/baichuan-inc/Baichuan2-13B-Chat-4bits", True],
    "chatglm2": ["../ChatGLM2-6B/THUDM/chatglm2-6b-int4"],
    "qwen": ["../Qwen/Qwen/Qwen-14B-Chat-Int4", True],
}


def service_loader() -> ChatCompletion:
    print("init service ...")
    service = create_chat_completion_service(
        use_service, *service_params[use_service])
    print("init service done")
    return service


st.set_page_config(page_title="Bot")
st.title("Bot")


@st.cache_resource
def create_bot(system: str = "") -> ChatBot:
    print("create_bot()")
    service = service_loader()
    bot = ChatBot(service, system=system)
    return bot


def clear_chat_history() -> None:
    bot: ChatBot = st.session_state.bot
    bot.clear()


def init_chat_history() -> None:
    bot: ChatBot = st.session_state.bot
    history = bot.history

    with st.chat_message("bot", avatar='🤖'):
        st.markdown(f"当前大模型：{use_service}")

    for message in history:
        if not message.role in ["user", "assistant"]:
            continue
        avatar = '🧑‍💻' if message.role == "user" else '🤖'
        with st.chat_message(message.role, avatar=avatar):
            st.markdown(message.content)


def main(system=""):
    bot = create_bot(system=system)
    st.session_state.bot = bot
    init_chat_history()

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)

        print(f"[user] {prompt}", flush=True)
        with st.chat_message("bot", avatar='🤖'):
            placeholder = st.empty()
            print(f"[bot] ", flush=True, end="")
            completion_gen = bot.chat_stream(prompt)
            completion = ""
            for delta_compl in completion_gen:
                print(delta_compl, flush=True, end="")
                completion += delta_compl
                placeholder.markdown(completion)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            print()

    st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    """
    run on the shell:
    PYTHONPATH=/this/repo/path USE_SERVICE=<baichuan2|chatglm2|qwen> streamlit run web_bot.py
    """
    main()
