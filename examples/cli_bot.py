
import os
import platform
import subprocess
from colorama import Fore, Style
from tempfile import NamedTemporaryFile
from adapter import ChatBot, ChatCompletion, create_chat_completion_service


# use = "baichuan2"
# use = "chatglm2"
use = "qwen"

service_params = {
    "baichuan2": "../Baichuan2/baichuan-inc/Baichuan2-13B-Chat-4bits",
    "chatglm2": "../ChatGLM2-6B/THUDM/chatglm2-6b",
    "qwen": "../Qwen/Qwen/Qwen-14B-Chat-Int4",
}


def service_loader() -> ChatCompletion:
    print("init service ...")
    service = create_chat_completion_service(use, service_params[use])
    print("init service done")
    return service


def clear_screen() -> None:
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    print(Fore.YELLOW + Style.BRIGHT +
          f"当前大模型：{use}，输入进行对话，vim 多行输入，clear 清空历史，CTRL+C 中断生成，stream 开关流式生成，exit 结束。")


def vim_input() -> str:
    with NamedTemporaryFile() as tempfile:
        tempfile.close()
        subprocess.call(['vim', '+star', tempfile.name])
        text = open(tempfile.name, encoding='utf8').read()
    return text


def main(system="") -> None:
    stream = True
    service = service_loader()
    bot = ChatBot(service, system=system)
    clear_screen()

    while True:
        message = input(Fore.GREEN + Style.BRIGHT + "\nUser：" + Style.NORMAL)
        if message.strip() == "exit":
            break

        if message.strip() == "clear":
            clear_screen()
            bot.clear()
            continue

        if message.strip() == 'vim':
            message = vim_input()
            print(message)

        print(Fore.CYAN + Style.BRIGHT + "\nBot：" + Style.NORMAL, end='')
        if message.strip() == "stream":
            stream = not stream
            print(Fore.YELLOW + "({}流式生成)\n".format("开启" if stream else "关闭"), end='')
            continue

        if stream:
            try:
                completion_gen = bot.chat_stream(message)
                for delta_compl in completion_gen:
                    print(delta_compl, end='', flush=True)
            except KeyboardInterrupt:
                pass
            print()
        else:
            completion = bot.chat(message)
            print(completion)

    print(Style.RESET_ALL)


if __name__ == "__main__":
    """
    run on the shell:
    PYTHONPATH=/this/repo/path python cli_bot.py
    """
    main()
