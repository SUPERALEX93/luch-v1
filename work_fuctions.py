import subprocess
from colorama import Fore, Back, Style
from ddgs import DDGS
import sys
import pyautogui


def web_search(request):
    with DDGS() as ddgs:
        return(list(ddgs.text(request, max_results=10)))

def terminal(command):
    result = subprocess.run(command.split(), shell=True, capture_output=True, text=True)
    return result.stdout

def exit():
    print(Fore.GREEN + "EXIT" + Style.RESET_ALL)
    sys.exit(0)

def lock_pc():
    subprocess.run(["hyprlock"])
    return "BLOCK PC SUCCESSFULLY"

def print_text(text):
    subprocess.run(['wtype', text])
    return "TEXT PRINTED SUCCESSFULLY"

commands_ai = {
    "terminal": {"func": terminal,"icon": "🖥️"},
    "web_search": {"func": web_search,"icon": "🌐"},
    "lock_pc": {"func": lock_pc, "icon": "🔒"},
    "print_text": {"func": print_text, "icon": "✍️"},

}

commands_ai_and_args = """
    
    1. terminal - выполнить команду в терминале.
    Пример: {"command": "terminal", "args": {"command": "ls"}}.В аргументах нужно указать команду для терминала в виде строки в аргументе command.
    
    2. web_search - использовать веб поиск по запросу.
    Пример: {"command": "web_search", "args": {"request": "что такое ии"}}.В аргументах нужно указать запрос для поиска в аргументе request.
    
    3. lock_pc - заблокировать пк.
    Пример: {"command": "lock_pc", "args": {}}.В аргументах не нужно ничего указывать.
    
    4. print_text - сэмитировать печать текста с клавиатуры.
    Пример: {"command": "print_text", "args": {"text":"тут текст"}}.В аргументах нужно указать текст который надо написать в виде строки в аргументе text.
    
    
    """