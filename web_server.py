import os
import socket
import subprocess
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import settings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_instance = None
server_instance = None

class QueryModel(BaseModel):
    text: str

class CommandModel(BaseModel):
    cmd: str   # полная команда, начинающаяся с '/'

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

@app.get("/")
def index():
    html_path = Path("index.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Файл index.html не найден")
    return FileResponse(html_path)

@app.post("/api/ask")
def ask_ai(data: QueryModel):
    global ai_instance
    if not ai_instance:
        raise HTTPException(status_code=500, detail="Экземпляр AI не инициализирован")

    ai_instance.response_ready_event.clear()
    ai_instance.user_prompt = data.text

    is_ready = ai_instance.response_ready_event.wait(timeout=60)
    if not is_ready:
        raise HTTPException(status_code=504, detail="Превышено время ожидания ответа ИИ")

    return {
        "status": "ok",
        "response": ai_instance.format_response(ai_instance.response),
        "play_audio": getattr(ai_instance, 'ai_speak', False)
    }

@app.post("/api/command")
def execute_command(data: CommandModel):
    global ai_instance
    if not ai_instance:
        raise HTTPException(status_code=500, detail="AI не инициализирован")
    try:
        result = ai_instance.process_command(data.cmd)
        return {"status": "ok", "output": result}
    except Exception as e:
        return {"status": "error", "output": str(e)}

@app.post("/api/voice")
async def ask_ai_voice(file: UploadFile = File(...)):
    global ai_instance
    if not ai_instance:
        raise HTTPException(status_code=500, detail="Экземпляр AI не инициализирован")

    temp_raw = "temp_mobile_raw"
    target_wav = settings.PATHS.get("temp_wav_file", "temp.wav")

    with open(temp_raw, "wb") as f:
        f.write(await file.read())

    subprocess.run([
        "ffmpeg", "-y", "-i", temp_raw,
        "-ar", "16000", "-ac", "1", target_wav
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(temp_raw):
        os.remove(temp_raw)

    ai_instance.response_ready_event.clear()
    ai_instance.pending_voice = True

    is_ready = ai_instance.response_ready_event.wait(timeout=60)
    if not is_ready:
        raise HTTPException(status_code=504, detail="Превышено время ожидания ответа ИИ")

    # Проверка стоп-слова
    if getattr(ai_instance, 'stop_audio_triggered', False):
        ai_instance.stop_audio_triggered = False
        return {
            "status": "stop_audio",
            "user_text": "",
            "response": "",
            "play_audio": False
        }

    # Игнорирование (чужой голос или нет триггера)
    if getattr(ai_instance, 'ignore_response', False):
        ai_instance.ignore_response = False
        return {
            "status": "ignored",
            "user_text": "",
            "response": "",
            "play_audio": False
        }

    return {
        "status": "ok",
        "user_text": ai_instance.user_prompt or "Голосовой запрос",
        "response": ai_instance.format_response(ai_instance.response),
        "play_audio": getattr(ai_instance, 'ai_speak', False) or getattr(ai_instance, 'is_voice_success', False)
    }

@app.get("/api/audio")
def get_audio():
    return FileResponse(settings.PATHS["response_path"], media_type="audio/wav")

def stop_server():
    global server_instance
    if server_instance:
        server_instance.should_exit = True

def start_server_in_thread(ai_obj, port=1337):
    global ai_instance, server_instance
    ai_instance = ai_obj
    ip = get_local_ip()

    for i in range(20):
        print(f" http://{ip}:{port}")

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="error")
    server_instance = uvicorn.Server(config)
    server_instance.run()