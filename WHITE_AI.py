import os
import json
import tkinter as tk
from tkinter import scrolledtext, Menu
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests

# Настройка папки для логирования
logs_dir = "LOGS"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_dialog(dialog, session_id):
    ensure_dir(logs_dir)
    filename = f"Session_{session_id}.json"
    with open(os.path.join(logs_dir, filename), 'w', encoding='utf-8') as f:
        json.dump(dialog, f, ensure_ascii=False, indent=4)

# Инициализация модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Объявление глобальных переменных
dialog = []
chat_history_ids = None

def get_new_session_id():
    """Возвращает новый идентификатор сессии, основываясь на существующих файлах."""
    files = os.listdir(logs_dir) if os.path.exists(logs_dir) else []
    session_ids = [int(f.split('_')[1].split('.')[0]) for f in files if f.startswith('Session_')]
    return max(session_ids) + 1 if session_ids else 1

# Инициализация session_id после определения функции get_new_session_id
session_id = get_new_session_id()


def send():
    global chat_history_ids, dialog, session_id
    user_input = entry_widget.get()
    entry_widget.delete(0, tk.END)  # Очистить поле ввода
    if user_input.strip():  # Проверка на пустую строку
        if user_input.startswith("define:"):
            word = user_input.split("define:")[1].strip()
            response = fetch_data_from_api(word)
        else:
            text_widget.config(state='normal')
            text_widget.insert(tk.END, "Вы: " + user_input + "\n")
            text_widget.config(state='disabled')

            new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids],
                                      dim=1) if chat_history_ids is not None else new_user_input_ids
            chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        text_widget.config(state='normal')
        text_widget.insert(tk.END, "WHITY: " + response + "\n")
        text_widget.config(state='disabled')
        text_widget.yview(tk.END)  # Прокрутка текста вниз

        dialog.append({"question": user_input, "answer": response})

        # Сохранение после каждого сообщения
        save_dialog(dialog, session_id)


def load_sessions():
    """Загружает существующие сессии из файла и добавляет их в меню."""
    ensure_dir(logs_dir)
    files = os.listdir(logs_dir)
    session_ids = sorted([int(f.split('_')[1].split('.')[0]) for f in files if f.startswith('Session_')], reverse=True)
    for sid in session_ids:
        session_menu.add_command(label=f"Сессия {sid}", command=lambda sid=sid: load_session(sid))

def load_session(sid):
    """Загружает и отображает диалог из указанной сессии."""
    global dialog
    dialog = []
    filename = os.path.join(logs_dir, f"Session_{sid}.json")
    with open(filename, 'r', encoding='utf-8') as f:
        dialog = json.load(f)
    text_widget.config(state='normal')
    text_widget.delete(1.0, tk.END)
    for entry in dialog:
        text_widget.insert(tk.END, f"Вы: {entry['question']}\nWHITY: {entry['answer']}\n")
    text_widget.config(state='disabled')

def new_session():
    """Создает новую сессию, сохраняет текущую и очищает интерфейс."""
    global session_id, dialog
    save_dialog(dialog, session_id)
    session_id = get_new_session_id()
    dialog = []
    text_widget.config(state='normal')
    text_widget.delete(1.0, tk.END)
    text_widget.insert(tk.END, "Новая сессия создана.\n")
    text_widget.config(state='disabled')
    session_menu.add_command(label=f"Сессия {session_id}", command=lambda sid=session_id: load_session(sid))

def get_new_session_id():
    """Возвращает новый идентификатор сессии, основываясь на существующих файлах."""
    files = os.listdir(logs_dir) if os.path.exists(logs_dir) else []
    session_ids = [int(f.split('_')[1].split('.')[0]) for f in files if f.startswith('Session_')]
    return max(session_ids) + 1 if session_ids else 1

def new_session():
    """Создает новую сессию, сохраняет текущую и очищает интерфейс."""
    global session_id, dialog
    save_dialog(dialog, session_id)
    session_id = get_new_session_id()
    dialog = []
    text_widget.config(state='normal')
    text_widget.delete(1.0, tk.END)
    text_widget.insert(tk.END, "Новая сессия создана.\n")
    text_widget.config(state='disabled')
    session_menu.add_command(label=f"Сессия {session_id}", command=lambda sid=session_id: load_session(sid))
    load_sessions()  # Обновить список сессий в меню
    
root = tk.Tk()
root.title("Чат с WHITY")
root.configure(bg='white')

menu = Menu(root)
root.config(menu=menu)
session_menu = Menu(menu, tearoff=0)
menu.add_cascade(label="Сессии", menu=session_menu)

load_sessions()  # Загрузка существующих сессий

text_widget = scrolledtext.ScrolledText(root, state='disabled', height=20, width=70, font=("Arial", 10), bg="#F0F0F0", fg="black", borderwidth=2, relief="solid")
text_widget.pack(padx=10, pady=10)

entry_frame = tk.Frame(root, bg='white')
entry_frame.pack(fill='x', padx=10)

entry_widget = tk.Entry(entry_frame, font=("Arial", 12), bg="white", fg="black", borderwidth=2, relief="solid")
entry_widget.pack(fill='x', side='left', expand=True, pady=10)

send_button = tk.Button(entry_frame, text="Отправить", command=send, font=("Arial", 12), bg="#4C8BF5", fg="white", relief="raised", borderwidth=2)
send_button.pack(side='right', padx=10)

new_session_button = tk.Button(entry_frame, text="Создать сессию", command=new_session, font=("Arial", 12), bg="#4C8BF5", fg="white", relief="raised", borderwidth=2)
new_session_button.pack(side='right', padx=10)

root.mainloop()
