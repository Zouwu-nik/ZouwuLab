"""Zouwu Lab — локальный веб-интерфейс для Qwen-Image 2.1 (картинки) и MiniMax H3 (видео).

Запускает движок (ядро ComfyUI без его интерфейса) из папки engine/ как фоновый процесс,
а сам отдаёт простой веб-интерфейс на http://127.0.0.1:7860
Все модели, выходные файлы, Python и библиотеки лежат внутри одной папки программы.
"""
import asyncio
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from urllib.parse import unquote, urlparse, quote

from aiohttp import web, ClientSession, ClientTimeout, WSMsgType

import workflow

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
MODELS = ROOT / "models"
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
ENGINE = ROOT / "engine" / "ComfyUI"
LOGS = ROOT / "logs"
CONFIG_FILE = ROOT / "config.json"
HISTORY_FILE = OUTPUT / "images" / "history.jsonl"
FOLDERS = ["diffusion_models", "text_encoders", "vae", "loras", "latent_upscale_models", "checkpoints", "llm"]
MODEL_EXT = {".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

DEFAULT_CONFIG = {
    "port": 7860,
    "engine_port": 8189,
    "open_browser": True,
    "hf_token": "",
    "civitai_token": "",
    "engine_args": "",
    "civitai_domain": "civitai.red",
    "vram_headroom": 2.0,   # ГБ видеопамяти, которые движок держит свободными (запас под вычисления видео)
}

for d in [*(MODELS / f for f in FOLDERS), MODELS / "face", INPUT, OUTPUT / "images", LOGS]:
    d.mkdir(parents=True, exist_ok=True)


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        try:
            cfg.update(json.loads(CONFIG_FILE.read_text("utf-8")))
        except Exception:
            pass
    return cfg


def save_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), "utf-8")


CONFIG = load_config()


def _arg(name):
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


# Одна программа для обеих нейросетей: Qwen-Image 2.1 (картинки) и MiniMax H3 (видео).
# Ключ --family из старых ярлыков (Start Flux / Start Qwen) больше ничего не ограничивает.
FAMILY = ""
FAMILY_NAME = ""
FAMILIES = ("qwen", "krea2", "flux3", "minimax", "ltx")
VIDEO_FAMILIES = ("minimax", "ltx")
VIDEO_EXT = {".mp4", ".webm", ".mkv", ".mov"}


def _port_free(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _reserve_ui_port(pref):
    """Сразу занимаем порт интерфейса (сокет держим и отдаём веб-серверу) — две копии программы,
    запущенные одновременно (Start Flux + Start Qwen), не схватят один и тот же порт."""
    import socket
    for p in range(int(pref), int(pref) + 20):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            s.bind(("127.0.0.1", p))
            return p, s
        except OSError:
            s.close()
    return int(pref), None


def _free_from(start, count, taken):
    for p in range(start, start + count):
        if p not in taken and _port_free(p):
            taken.add(p)
            return p
    return start


UI_PORT, UI_SOCK = (_reserve_ui_port(CONFIG["port"]) if __name__ == "__main__" else (int(CONFIG["port"]), None))
# у каждой копии программы свой блок из 10 портов: движок — первые 5, помощник — следующие 5
_base = int(CONFIG["engine_port"]) + 10 * max(0, UI_PORT - int(CONFIG["port"]))
_taken = {UI_PORT}
RUNTIME = {"port": UI_PORT}
RUNTIME["engine_port"] = _free_from(_base, 5, _taken)
RUNTIME["llm_port"] = _free_from(_base + 5, 5, _taken)
SUFFIX = ""
CATALOG = json.loads((APP_DIR / "catalog.json").read_text("utf-8"))["items"]


def human(n):
    n = float(n or 0)
    for unit in ["Б", "КБ", "МБ", "ГБ", "ТБ"]:
        if n < 1024 or unit == "ТБ":
            return f"{n:.1f} {unit}" if unit != "Б" else f"{int(n)} Б"
        n /= 1024


def safe_name(name: str) -> str:
    name = os.path.basename(unquote(name or "")).strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    return name or f"file_{uuid.uuid4().hex[:8]}.safetensors"


_FAM_CACHE = {}


def model_family(p: Path):
    """Для какой нейросети файл модели — по именам слоёв (дообученные модели часто названы как угодно)."""
    if p.suffix.lower() != ".safetensors":
        return ""
    key = (str(p), p.stat().st_mtime)
    if key not in _FAM_CACHE:
        fam = ""
        try:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                if n < 200_000_000:
                    j = " ".join(k for k in json.loads(f.read(n)) if k != "__metadata__")[:400000]
                    if "video_patch_proj" in j and "audio_patch_proj" in j:
                        fam = "minimax"
                    elif "audio_adaln_single" in j or ("adaln_single" in j and "audio_attn" in j):
                        fam = "ltx"
                    elif "txtfusion" in j:
                        fam = "krea2"
                    elif "content_mode_blocks" in j or "single_stream_modulations" in j:
                        fam = "flux3"
                    elif "double_blocks" in j or "single_blocks" in j:
                        fam = "flux"
                    elif "transformer_blocks" in j and "img_mlp" in j:
                        fam = "qwen"
        except Exception:
            fam = ""
        _FAM_CACHE[key] = fam
    return _FAM_CACHE[key]


_FUSED_CACHE = {}


def mlp_fused(unet):
    """Склеены ли у модели Qwen 2.1 половины MLP в один слой gate_up (так у официальных файлов).
    У части дообученных моделей они лежат раздельно — тогда LoRA для gate_up надо разрезать."""
    p = MODELS / "diffusion_models" / (unet or "")
    try:
        key = (str(p), p.stat().st_mtime)
    except OSError:
        return True
    if key in _FUSED_CACHE:
        return _FUSED_CACHE[key]
    fused = True
    try:
        if p.suffix.lower() == ".safetensors":
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                if n < 200_000_000:
                    ks = [k for k in json.loads(f.read(n)) if "img_mlp" in k]
                    if ks:
                        fused = any("gate_up" in k for k in ks)
        elif p.suffix.lower() == ".gguf":
            import gguf
            names = [t.name for t in gguf.GGUFReader(str(p)).tensors if "img_mlp" in t.name]
            if names:
                fused = any("gate_up" in k for k in names)
    except Exception:
        fused = True
    _FUSED_CACHE[key] = fused
    return fused


def list_models(folder):
    base = MODELS / folder
    out = []
    for p in sorted(base.rglob("*"), key=lambda x: str(x).lower()):
        if p.is_file() and p.suffix.lower() in MODEL_EXT and "_qwen_fixed" not in p.parts:
            rel = p.relative_to(base).as_posix()
            item = {"name": rel, "size": p.stat().st_size, "size_h": human(p.stat().st_size)}
            if folder == "diffusion_models":
                item["fam"] = model_family(p)
            out.append(item)
    return out


# ---------------------------------------------------------------- движок (ядро ComfyUI)
class Engine:
    def __init__(self):
        self.proc = None
        self.state = "stopped"  # stopped | starting | ready | error | missing
        self.message = ""
        self.log_path = LOGS / f"engine{SUFFIX}.log"
        self.client_id = uuid.uuid4().hex
        self.jobs = {}
        self.ws_task = None
        self.system = {}

    @property
    def url(self):
        return f"http://127.0.0.1:{RUNTIME['engine_port']}"

    def start(self):
        if not (ENGINE / "main.py").exists():
            self.state, self.message = "missing", "Движок не установлен. Запустите install.bat"
            return
        if self.proc and self.proc.poll() is None:
            return
        args = [sys.executable, "-s", str(ENGINE / "main.py"),
                "--listen", "127.0.0.1", "--port", str(RUNTIME["engine_port"]),
                "--models-directory", str(MODELS),
                "--input-directory", str(INPUT),
                "--output-directory", str(OUTPUT),
                "--user-directory", str(ROOT / "engine" / "user"),
                "--temp-directory", str(ROOT / "engine" / "temp"),
                "--disable-auto-launch", "--preview-method", "latent2rgb"]
        (ROOT / "engine" / "user").mkdir(parents=True, exist_ok=True)
        extra = (CONFIG.get("engine_args") or "").split()
        prof = [a for a in mem_profile_args() if a.split("=")[0] not in " ".join(extra)]
        args += prof + extra
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"  # журнал пишется сразу — по нему проверяем, применились ли LoRA
        env["HF_HOME"] = str(ROOT / "cache" / "huggingface")
        self.log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.proc = subprocess.Popen(args, cwd=str(ENGINE), stdout=self.log, stderr=subprocess.STDOUT,
                                     env=env, creationflags=flags)
        self.state, self.message = "starting", "Запуск движка…"

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except Exception:
                self.proc.kill()
        self.proc = None
        self.state = "stopped"

    def log_tail(self, n=60):
        try:
            lines = self.log_path.read_text("utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n:])
        except Exception:
            return ""

    async def watchdog(self, app):
        async with ClientSession(timeout=ClientTimeout(total=3)) as s:
            while True:
                if self.proc and self.proc.poll() is not None and self.state != "stopped":
                    self.state = "error"
                    self.message = "Движок завершился с ошибкой — см. журнал"
                elif self.proc and self.state in ("starting", "error", "ready"):
                    try:
                        async with s.get(self.url + "/system_stats") as r:
                            if r.status == 200:
                                self.system = await r.json()
                                if self.state != "ready":
                                    self.state, self.message = "ready", "Движок готов"
                                    if not self.ws_task or self.ws_task.done():
                                        self.ws_task = asyncio.create_task(self.ws_loop())
                        self.fails = 0
                    except Exception:
                        # под тяжёлой нагрузкой движок может не ответить за 3 с — это ещё не падение
                        self.fails = getattr(self, "fails", 0) + 1
                        if self.state == "ready" and self.fails >= 5:
                            self.state = "starting"
                await mem_guard(self)
                await asyncio.sleep(2)

    # ---- websocket от движка: прогресс, превью, результаты
    async def ws_loop(self):
        while self.state == "ready":
            try:
                async with ClientSession() as s:
                    async with s.ws_connect(f"{self.url}/ws?clientId={self.client_id}", max_msg_size=0) as ws:
                        async for msg in ws:
                            if msg.type == WSMsgType.TEXT:
                                self.on_message(json.loads(msg.data))
                            elif msg.type == WSMsgType.BINARY:
                                self.on_binary(msg.data)
            except Exception:
                await asyncio.sleep(1)

    def current_job(self):
        for j in self.jobs.values():
            if j["status"] == "running":
                return j
        return None

    def on_binary(self, data: bytes):
        if len(data) < 8:
            return
        event = struct.unpack(">I", data[:4])[0]
        img = None
        if event == 1:
            img = data[8:]
        elif event == 4:  # превью с метаданными
            mlen = struct.unpack(">I", data[4:8])[0]
            img = data[8 + mlen:]
        job = self.current_job()
        if img and job:
            job["preview"] = img
            job["preview_n"] += 1

    def on_message(self, m):
        t, d = m.get("type"), m.get("data", {}) or {}
        job = self.jobs.get(d.get("prompt_id")) if isinstance(d, dict) else None
        if t == "status":
            return
        if not job:
            return
        if t == "execution_start":
            job["status"], job["stage"] = "running", "Старт"
            job["started"] = time.time()
        elif t == "executing":
            node = d.get("node")
            if node is None:
                self.finish(job)
            else:
                job["status"] = "running"
                ct = job["graph"].get(node, {}).get("class_type", "")
                job["stage"] = STAGE_NAMES.get(ct, ct)
        elif t == "progress":
            job["status"] = "running"
            job["value"], job["max"] = d.get("value", 0), d.get("max", 0)
        elif t == "executed":
            o = d.get("output") or {}
            for im in o.get("images", []) + o.get("3d", []) + o.get("audio", []):
                if im.get("type") == "output":
                    sub = im.get("subfolder", "")
                    path = f"{sub}/{im['filename']}" if sub else im["filename"]
                    url = "/output/" + quote(path)
                    if url not in job["images"]:
                        job["images"].append(url)
        elif t == "execution_success":
            self.finish(job)
        elif t == "execution_error":
            job["status"] = "error"
            msg = (d.get("exception_message") or "").strip()
            if "out of memory" in msg.lower() or "exceed allowed memory" in msg.lower() or d.get("exception_type", "").endswith("OutOfMemoryError"):
                job["error"] = on_gpu_oom(job)
            else:
                job["error"] = f"{d.get('node_type', '')}: {msg}"
        elif t == "execution_interrupted":
            job["status"] = "cancelled"
            job["error"] = "Остановлено"

    def finish(self, job):
        if job["status"] in ("done", "error", "cancelled", "finishing"):
            return
        job["status"], job["stage"] = "finishing", "Сохранение"
        asyncio.get_event_loop().create_task(self.finalize(job))

    async def finalize(self, job):
        if not job["images"]:  # результат мог быть взят из кэша — берём из истории движка
            try:
                async with ClientSession() as s:
                    async with s.get(f"{self.url}/history/{job['id']}") as r:
                        h = (await r.json()).get(job["id"], {})
                for out in h.get("outputs", {}).values():
                    for im in out.get("images", []) + out.get("3d", []) + out.get("audio", []):
                        if im.get("type") == "output":
                            sub = im.get("subfolder", "")
                            path = f"{sub}/{im['filename']}" if sub else im["filename"]
                            url = "/output/" + quote(path)
                            if url not in job["images"]:
                                job["images"].append(url)
            except Exception:
                pass
        if job.get("post"):  # доработка результата (например, вклейка правки обратно в исходник)
            try:
                job["images"] = await job.pop("post")(job["images"])
            except Exception as e:
                job["status"], job["error"] = "error", "Ошибка при сохранении правки: " + str(e)
                return
        if job.get("saved"):
            return
        job["saved"] = True
        try:  # подпись программы в данных PNG (картинка не меняется; настройки генерации остаются)
            await asyncio.get_running_loop().run_in_executor(None, tag_outputs, list(job["images"]))
        except Exception:
            pass
        try:  # движок пишет «lora key not loaded», если слои LoRA не подошли к модели
            with open(self.log_path, "rb") as f:
                f.seek(job.get("log_pos", 0))
                tail = f.read().decode("utf-8", errors="replace")
            miss = tail.count("lora key not loaded")
            if miss and any(l.get("enabled", True) for l in job["params"].get("loras", [])):
                job["warnings"].append(f"Движок не смог применить LoRA: {miss} слоёв не подошли к модели. "
                                       "Скорее всего, LoRA сделана для другой модели или версии — она почти не подействовала.")
        except Exception:
            pass
        job["status"], job["stage"] = "done", "Готово"
        job["finished"] = time.time()
        job["preview"] = None
        entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "images": job["images"],
                 "params": job["params"], "seconds": round(job["finished"] - job.get("started", job["created"]), 1),
                 "warnings": job.get("warnings") or []}
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        try:
            if not any(Path(unquote(u.split("?")[0])).suffix.lower() in VIDEO_EXT | {".glb", ".mp3", ".flac", ".wav", ".opus"} for u in job["images"]):
                SCORER["on_job_done"](job)   # автооценка новых картинок (в фоне, на процессоре)
        except Exception:
            pass


SOFTWARE_TAG = "Zouwu Lab by Zouwu (t.me/Zouwu_N)"


def tag_outputs(urls):
    """Строка «Software» в данных готовых PNG: по файлу видно, где он сделан."""
    from PIL import Image, PngImagePlugin
    for u in urls:
        p = OUTPUT / unquote(u.split("?")[0]).replace("/output/", "", 1)
        if p.suffix.lower() != ".png" or not p.is_file():
            continue
        with Image.open(p) as im:
            if im.info.get("Software") == SOFTWARE_TAG:
                continue
            meta = PngImagePlugin.PngInfo()
            for k, v in im.info.items():
                if isinstance(v, str):
                    meta.add_text(k, v)
            meta.add_text("Software", SOFTWARE_TAG)
            im.load()
            tmp = p.with_suffix(".tagging.png")
            im.save(tmp, pnginfo=meta, compress_level=4)
        os.replace(tmp, p)


STAGE_NAMES = {
    "UNETLoader": "Загрузка модели", "UnetLoaderGGUF": "Загрузка модели (GGUF)", "LoraLoader": "Применение LoRA",
    "KSampler": "Генерация", "TextEncodeQwenImage21": "Кодирование промпта и картинок", "CLIPLoader": "Загрузка текстового энкодера",
    "VAELoader": "Загрузка VAE", "LoraLoaderModelOnly": "Применение LoRA",
    "CLIPTextEncode": "Кодирование промпта", "VAEEncode": "Кодирование изображения",
    "SamplerCustomAdvanced": "Генерация", "VAEDecode": "Декодирование", "SaveImage": "Сохранение",
    "LoadImage": "Загрузка изображения", "BasicScheduler": "Расписание шагов",
    "MiniMaxH3ImageToVideo": "Кодирование промпта и кадров", "VAEDecodeAudio": "Звук",
    "CreateVideo": "Сборка видео", "SaveVideo": "Сохранение видео",
    "TextEncodeQwenImageEditPlus": "Кодирование промпта и референсов", "LTXVConditioning": "Кодирование промпта",
    "LTXVImgToVideoInplace": "Вшивание исходного кадра", "LTXVAddGuide": "Ключевые кадры",
    "LTXVLatentUpsampler": "Апскейл ×2", "LatentUpscaleModelLoader": "Загрузка апскейлера",
    "VAEDecodeTiled": "Декодирование видео", "LTXVAudioVAEDecode": "Звук", "LTXVPreprocess": "Подготовка кадра",
    "SamplerCustom": "Генерация", "ImageScale": "Подгонка кадра",
    "ImageOnlyCheckpointLoader": "Загрузка модели 3D", "CLIPVisionEncode": "Разбор ракурсов",
    "Hunyuan3Dv2ConditioningMultiView": "Разбор ракурсов", "Hunyuan3Dv2Conditioning": "Разбор картинки",
    "VAEDecodeHunyuan3D": "Сборка объёма", "VoxelToMesh": "Построение сетки", "MeshSmoothNormals": "Сглаживание",
    "SaveGLB": "Сохранение модели",
    "DualCLIPLoader": "Загрузка текстовых энкодеров", "TextEncodeAceStepAudio1.5": "Сочиняю структуру песни",
    "MiniMaxMusic3TextEncode": "Сочиняю структуру песни", "YuE2GenerateMusic": "Сочиняю мелодию и вокал",
    "VAEDecodeAudio": "Сборка звука", "SaveAudioMP3": "Сохранение MP3", "CheckpointLoaderSimple": "Загрузка модели",
}

ENGINE_ = Engine()


# ---------------------------------------------------------------- память: профили и предохранитель
MEM_PROFILES = {
    # движок сам выгружает в ОЗУ то, что не влезает в видеопамять (dynamic VRAM); профили ограничивают аппетит к ОЗУ
    "default": [],
    "ram_safe": ["--disable-pinned-memory", "--cache-ram", "4", "--vram-headroom", "1"],
    "ram_min": ["--disable-pinned-memory", "--cache-none", "--fast-disk", "--vram-headroom", "1.5"],
}
MEM = {"last_stop": "", "stopped_at": 0}


def _engine_knows(flag):
    try:
        return flag in (ENGINE / "comfy" / "cli_args.py").read_text("utf-8", errors="replace")
    except OSError:
        return False


def auto_profile():
    """«Авто»: профиль по объёму ОЗУ этого компьютера (решается при каждом запуске — папку можно переносить)."""
    gb = (ram_info().get("total") or 0) / 1024 ** 3
    if not gb:
        return "ram_safe"
    if gb < 24:
        return "ram_min"
    if gb < 48:
        return "ram_safe"
    return "default"


def effective_profile():
    p = CONFIG.get("mem_profile") or "auto"
    return auto_profile() if p == "auto" else p


def mem_profile_args():
    """Аргументы движка по выбранному профилю памяти (только те, что знает установленная версия движка)."""
    out, args = [], MEM_PROFILES.get(effective_profile(), [])
    i = 0
    while i < len(args):
        flag = args[i]
        vals = []
        while i + 1 < len(args) and not args[i + 1].startswith("--"):
            vals.append(args[i + 1])
            i += 1
        if _engine_knows(flag):
            out += [flag] + vals
        i += 1
    # запас видеопамяти: не меньше, чем в профиле, и не меньше настройки (растёт сама после нехватки памяти)
    try:
        want = float(CONFIG.get("vram_headroom") or 0)
    except (TypeError, ValueError):
        want = 0.0
    if want > 0 and _engine_knows("--vram-headroom"):
        if "--vram-headroom" in out:
            k = out.index("--vram-headroom")
            out[k + 1] = str(max(float(out[k + 1]), want))
        else:
            out += ["--vram-headroom", str(want)]
    return out


VRAM_HEADROOM_MAX = 6.0


def on_gpu_oom(job):
    """Нехватка видеопамяти: увеличиваем запас, перезапускаем движок и объясняем, что делать."""
    old = float(CONFIG.get("vram_headroom") or 0)
    new = min(VRAM_HEADROOM_MAX, max(old, 2.0) + 1.5)
    video = job.get("params", {}).get("family") in VIDEO_FAMILIES
    tips = ("Если повторится: включите «⚡ Турбо», уменьшите длительность или разрешение (например 1024×576), "
            "закройте другие программы, занимающие видеокарту." if video else
            "Если повторится: уменьшите разрешение или число картинок за раз, закройте другие программы на видеокарте.")
    if new > old:
        CONFIG["vram_headroom"] = new
        save_config(CONFIG)

        def _restart():
            if not any(j.get("status") in ("queued", "running") for j in ENGINE_.jobs.values()):
                ENGINE_.stop()
                ENGINE_.start()
        try:
            asyncio.get_event_loop().call_later(1.5, _restart)
        except RuntimeError:
            pass
        return (f"Не хватило видеопамяти. Программа увеличила запас видеопамяти до {new:g} ГБ (часть модели будет "
                f"держаться в ОЗУ) и перезапускает движок — через полминуты нажмите «Сгенерировать» ещё раз. " + tips)
    return "Не хватило видеопамяти даже с максимальным запасом. " + tips


def ram_info():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {"total": vm.total, "available": vm.available}
    except Exception:
        return {}


async def mem_guard(engine):
    """Предохранитель: если свободная оперативная память почти кончилась — остановить генерацию,
    пока Windows не начала задыхаться (зависание, вылет, перезагрузка)."""
    if not CONFIG.get("mem_guard", True):
        return
    job = engine.current_job()
    if not job:
        return
    r = ram_info()
    if not r:
        return
    limit = float(CONFIG.get("mem_guard_gb", 1.5)) * 1024 ** 3
    if r["available"] >= limit:
        return
    if time.time() - MEM["stopped_at"] < 10:
        return
    MEM["stopped_at"] = time.time()
    try:
        await LLM["stop_async"]()
    except Exception:
        pass
    try:
        async with ClientSession(timeout=ClientTimeout(total=5)) as s:
            await s.post(engine.url + "/interrupt", json={})
            await s.post(engine.url + "/free", json={"unload_models": True, "free_memory": True})
    except Exception:
        pass
    msg = (f"Остановлено предохранителем: свободной оперативной памяти осталось {human(r['available'])} из {human(r['total'])}. "
           "Включите профиль «Бережный к ОЗУ» или «Минимум памяти» (Настройки → Память), возьмите энкодер w4a8/nvfp4 "
           "или модель поменьше (GGUF), уменьшите разрешение, закройте браузерные вкладки и другие программы.")
    job["status"], job["error"] = "error", msg
    MEM["last_stop"] = msg
    logging_note(msg)


# ---------------------------------------------------------------- загрузки и импорт файлов
class Transfers:
    def __init__(self):
        self.tasks = {}
        self.cancel = set()

    def new(self, kind, folder, filename, source):
        tid = uuid.uuid4().hex[:10]
        self.tasks[tid] = {"id": tid, "kind": kind, "folder": folder, "filename": filename, "source": source,
                           "total": 0, "done": 0, "speed": 0, "status": "queued", "error": "", "t": time.time()}
        return self.tasks[tid]

    def public(self):
        return sorted(({k: v for k, v in t.items() if k != "after"} for t in self.tasks.values()),
                      key=lambda t: -t["t"])

    async def download(self, task, url):
        """Качает файл; для Civitai пробует запасные адреса (civitai.red / civitai.com)."""
        urls = [url] + list(task.get("alt_urls") or [])
        for i, u in enumerate(urls):
            task["error"] = ""
            await self._dl(task, u)
            if task["status"] != "error" or i == len(urls) - 1:
                break
        if task["status"] == "done" and task.get("after"):
            try:
                await task.pop("after")()
            except Exception:
                pass

    async def _dl(self, task, url):
        headers = {"User-Agent": "ZouwuLab/1.0"}
        host = urlparse(url).netloc.lower()
        if "huggingface.co" in host and CONFIG.get("hf_token"):
            headers["Authorization"] = f"Bearer {CONFIG['hf_token'].strip()}"
        is_civ = "civitai." in host or "/api/download/models/" in url
        if is_civ and CONFIG.get("civitai_token"):
            url += ("&" if "?" in url else "?") + "token=" + CONFIG["civitai_token"].strip()
        task["status"] = "downloading"
        try:
            async with ClientSession(timeout=ClientTimeout(total=None, sock_read=120)) as s:
                # имя файла заранее может быть неизвестно (например, Civitai)
                part = None
                start = 0
                if task["filename"]:
                    part = MODELS / task["folder"] / (task["filename"] + ".part")
                    if part.exists():
                        start = part.stat().st_size
                h = dict(headers)
                if start:
                    h["Range"] = f"bytes={start}-"
                async with s.get(url, headers=h, allow_redirects=True) as r:
                    if r.status in (401, 403) and is_civ:
                        raise RuntimeError("Civitai не дал скачать (401/403): укажите API‑ключ Civitai в Настройках. "
                                           "Для 18+ LoRA в аккаунте Civitai должен быть включён показ взрослого контента.")
                    if r.status in (401, 403):
                        raise RuntimeError("Доступ запрещён (401/403). Для закрытых моделей укажите HF‑токен в Настройках "
                                           "и нажмите «Agree» на странице модели на huggingface.co. Для Civitai — API‑ключ.")
                    if r.status == 416:  # уже докачано
                        start = 0
                    elif r.status not in (200, 206):
                        raise RuntimeError(f"HTTP {r.status}")
                    if not task["filename"]:
                        cd = r.headers.get("Content-Disposition", "")
                        m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", cd)
                        task["filename"] = safe_name(m.group(1) if m else urlparse(str(r.url)).path)
                        part = MODELS / task["folder"] / (task["filename"] + ".part")
                    final = MODELS / task["folder"] / task["filename"]
                    if r.status == 200:
                        start = 0
                    length = int(r.headers.get("Content-Length", 0) or 0)
                    task["total"] = start + length if length else 0
                    task["done"] = start
                    mode = "ab" if (start and r.status == 206) else "wb"
                    t0, b0 = time.time(), task["done"]
                    with open(part, mode) as f:
                        async for chunk in r.content.iter_chunked(1 << 20):
                            if task["id"] in self.cancel:
                                raise asyncio.CancelledError()
                            f.write(chunk)
                            task["done"] += len(chunk)
                            dt = time.time() - t0
                            if dt > 1:
                                task["speed"] = (task["done"] - b0) / dt
                                t0, b0 = time.time(), task["done"]
                if final.exists():
                    final.unlink()
                part.rename(final)
                task["status"] = "done"
        except asyncio.CancelledError:
            task["status"] = "cancelled"
            task["error"] = "Отменено (недокачанный файл .part сохранён — при повторе докачка продолжится)"
        except Exception as e:
            task["status"] = "error"
            task["error"] = str(e) or e.__class__.__name__

    def copy_thread(self, task, src: Path, mode: str):
        dst = MODELS / task["folder"] / task["filename"]
        task["status"] = "copying"
        try:
            task["total"] = src.stat().st_size
            if dst.exists():
                raise RuntimeError("Файл с таким именем уже есть в папке")
            if mode == "move":
                try:
                    os.replace(src, dst)
                    task["done"] = task["total"]
                    task["status"] = "done"
                    return
                except OSError:
                    pass  # другой диск — копируем и удаляем
            if mode == "link":
                try:
                    os.link(src, dst)
                    task["done"] = task["total"]
                    task["status"] = "done"
                    return
                except OSError:
                    pass  # другой диск — обычное копирование
            tmp = dst.with_name(dst.name + ".part")
            t0, b0 = time.time(), 0
            with open(src, "rb") as fi, open(tmp, "wb") as fo:
                while True:
                    if task["id"] in self.cancel:
                        raise RuntimeError("Отменено")
                    buf = fi.read(8 << 20)
                    if not buf:
                        break
                    fo.write(buf)
                    task["done"] += len(buf)
                    if time.time() - t0 > 1:
                        task["speed"] = (task["done"] - b0) / (time.time() - t0)
                        t0, b0 = time.time(), task["done"]
            tmp.rename(dst)
            if mode == "move":
                src.unlink()
            task["status"] = "done"
        except Exception as e:
            task["status"] = "error"
            task["error"] = str(e)


TRANSFERS = Transfers()


# ---------------------------------------------------------------- HTTP API
routes = web.RouteTableDef()


def jerr(msg, code=400):
    return web.json_response({"error": msg}, status=code)


@routes.get("/")
async def index(_):
    """index.html с номером версии в ссылках на скрипт и стили — браузер не возьмёт старые файлы из кэша после обновления."""
    st = APP_DIR / "static"
    ver = int(max((st / f).stat().st_mtime for f in ("app.js", "style.css", "index.html")))
    html = (st / "index.html").read_text("utf-8")
    html = html.replace('/static/app.js"', f'/static/app.js?v={ver}"').replace('/static/style.css"', f'/static/style.css?v={ver}"')
    lock = json.dumps(FAMILY)
    html = html.replace("</title>", f"</title><script>window.FAMILY_LOCK={lock};</script>", 1)
    return web.Response(text=html, content_type="text/html", headers={"Cache-Control": "no-store"})


@web.middleware
async def no_cache_static(request, handler):
    resp = await handler(request)
    if request.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


@routes.get("/api/state")
async def state(_):
    e = ENGINE_
    dev = {}
    try:
        d = e.system.get("devices", [{}])[0]
        dev = {"name": d.get("name", ""), "vram_total": human(d.get("vram_total", 0)),
               "vram_free": human(d.get("vram_free", 0))}
    except Exception:
        pass
    r = ram_info()
    if r:
        dev["ram_total"], dev["ram_free"] = human(r["total"]), human(r["available"])
        dev["ram_free_gb"] = round(r["available"] / 1024 ** 3, 1)
    return web.json_response({"engine": e.state, "message": e.message, "device": dev,
                              "mem_profile": CONFIG.get("mem_profile") or "auto", "mem_profile_effective": effective_profile(), "mem_guard": CONFIG.get("mem_guard", True),
                              "qwen_supported": (ENGINE / "comfy" / "ldm" / "qwen_image21").exists(),
                              "flux3_supported": (ENGINE / "comfy" / "ldm" / "flux3").exists(),
                              "family_lock": FAMILY, "gguf": GGUF_STATE(),
                              "models": {f: list_models(f) for f in FOLDERS}})


@routes.get("/api/engine/log")
async def engine_log(_):
    return web.json_response({"log": ENGINE_.log_tail(200)})


@routes.post("/api/engine/restart")
async def engine_restart(_):
    ENGINE_.stop()
    ENGINE_.start()
    return web.json_response({"ok": True})


@routes.post("/api/engine/free")
async def engine_free(_):
    async with ClientSession() as s:
        async with s.post(ENGINE_.url + "/free", json={"unload_models": True, "free_memory": True}) as r:
            return web.json_response({"ok": r.status == 200})


# ---------------------------------------------------------------- поддержка .gguf (ComfyUI-GGUF)
import gguf_support  # noqa: E402
GGUF = {"running": False, "error": "", "log": "", "done": 0, "total": 0}


def GGUF_STATE():
    return {"installed": gguf_support.installed(ENGINE), "running": GGUF["running"], "error": GGUF["error"],
            "done": GGUF["done"], "total": GGUF["total"]}


@routes.get("/api/gguf")
async def gguf_state(_):
    return web.json_response(GGUF_STATE())


@routes.post("/api/gguf/install")
async def gguf_install(_):
    if GGUF["running"]:
        return jerr("Уже устанавливается")
    GGUF.update(running=True, error="", log="", done=0, total=0)

    def work():
        try:
            gguf_support.install(ROOT, ENGINE, log=lambda m: GGUF.__setitem__("log", GGUF["log"] + m + "\n"),
                                 progress=lambda d, t: GGUF.update(done=d, total=t))
        except Exception as e:
            GGUF["error"] = str(e)
        finally:
            GGUF["running"] = False

    def after(fut):
        if not GGUF["error"]:  # новые узлы движок видит только после перезапуска
            ENGINE_.stop()
            ENGINE_.start()
    fut = asyncio.get_running_loop().run_in_executor(None, work)
    fut.add_done_callback(after)
    return web.json_response({"ok": True})


@routes.get("/api/catalog")
async def catalog(_):
    items = []
    for it in CATALOG:
        x = dict(it)
        x["installed"] = (MODELS / it["folder"] / it["filename"]).exists()
        x["partial"] = (MODELS / it["folder"] / (it["filename"] + ".part")).exists()
        items.append(x)
    return web.json_response({"items": items, "has_hf_token": bool(CONFIG.get("hf_token"))})


@routes.post("/api/download")
async def download(req):
    b = await req.json()
    if "civitai." in (b.get("url") or "") and "/models/" in (b.get("url") or ""):
        try:  # ссылка на страницу Civitai — качаем вместе с описанием и картинкой
            t = await CIVITAI["download_link"](b["url"])
        except Exception as e:
            return jerr(str(e))
        return web.json_response({k: v for k, v in t.items() if k != "after"})
    if b.get("id"):
        it = next((i for i in CATALOG if i["id"] == b["id"]), None)
        if not it:
            return jerr("Нет такой записи в каталоге")
        url, folder, filename = it["url"], it["folder"], it["filename"]
    else:
        url, folder = (b.get("url") or "").strip(), b.get("folder")
        filename = safe_name(b["filename"]) if b.get("filename") else ""
        if not url.startswith("http"):
            return jerr("Укажите ссылку http(s)")
        if "huggingface.co" in url and "/blob/" in url:
            url = url.replace("/blob/", "/resolve/")
        if not filename and "huggingface.co" in url:
            filename = safe_name(urlparse(url).path)
    if folder not in FOLDERS:
        return jerr("Неверная папка")
    for t in TRANSFERS.tasks.values():
        if t["filename"] and t["filename"] == filename and t["status"] in ("queued", "downloading"):
            return jerr("Этот файл уже скачивается")
    task = TRANSFERS.new("download", folder, filename, url)
    asyncio.create_task(TRANSFERS.download(task, url))
    return web.json_response(task)


@routes.get("/api/transfers")
async def transfers(_):
    return web.json_response({"items": TRANSFERS.public()})


@routes.post("/api/transfers/cancel")
async def transfers_cancel(req):
    b = await req.json()
    TRANSFERS.cancel.add(b.get("id"))
    return web.json_response({"ok": True})


@routes.post("/api/transfers/clear")
async def transfers_clear(_):
    for k in [k for k, t in TRANSFERS.tasks.items() if t["status"] not in ("queued", "downloading", "copying")]:
        del TRANSFERS.tasks[k]
    return web.json_response({"ok": True})


@routes.post("/api/import_path")
async def import_path(req):
    """Взять файл модели из любого места на диске: связать (hardlink), скопировать или переместить."""
    b = await req.json()
    folder, mode = b.get("folder"), b.get("mode", "link")
    raw = (b.get("path") or "").strip().strip('"')
    if folder not in FOLDERS:
        return jerr("Неверная папка")
    src = Path(raw)
    if not src.exists():
        return jerr("Файл не найден: " + raw)
    files = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in MODEL_EXT] if src.is_dir() else [src]
    if not files:
        return jerr("В папке нет файлов моделей")
    created = []
    for f in files:
        task = TRANSFERS.new("copy", folder, safe_name(f.name), str(f))
        threading.Thread(target=TRANSFERS.copy_thread, args=(task, f, mode), daemon=True).start()
        created.append(task)
    return web.json_response({"items": created})


@routes.post("/api/upload_model")
async def upload_model(req):
    """Загрузка файла модели через браузер (drag&drop / выбор файла)."""
    folder = req.query.get("folder")
    if folder not in FOLDERS:
        return jerr("Неверная папка")
    reader = await req.multipart()
    saved = []
    async for part in reader:
        if part.name != "file" or not part.filename:
            continue
        name = safe_name(part.filename)
        dst = MODELS / folder / name
        tmp = dst.with_name(name + ".part")
        with open(tmp, "wb") as f:
            while True:
                chunk = await part.read_chunk(4 << 20)
                if not chunk:
                    break
                f.write(chunk)
        if dst.exists():
            dst.unlink()
        tmp.rename(dst)
        saved.append(name)
    return web.json_response({"saved": saved})


@routes.post("/api/delete_model")
async def delete_model(req):
    b = await req.json()
    folder = b.get("folder")
    if folder not in FOLDERS:
        return jerr("Неверная папка")
    p = (MODELS / folder / b.get("name", "")).resolve()
    if not str(p).startswith(str((MODELS / folder).resolve())) or not p.is_file():
        return jerr("Файл не найден")
    p.unlink()
    return web.json_response({"ok": True})


@routes.post("/api/open_folder")
async def open_folder(req):
    b = await req.json()
    target = {"output": OUTPUT / "images", "voice": OUTPUT / "voice", "video": OUTPUT / "video", "mesh": OUTPUT / "mesh",
              "music": OUTPUT / "music", "speech": OUTPUT / "speech",
              "root": ROOT}.get(b.get("folder"), MODELS / b.get("folder", ""))
    if b.get("folder") in ("mesh", "music", "speech"):
        target.mkdir(parents=True, exist_ok=True)
    if b.get("folder") not in FOLDERS + ["output", "root", "voice", "video", "mesh", "music", "speech"]:
        return jerr("Неверная папка")
    try:
        if os.name == "nt":
            os.startfile(str(target))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except Exception as e:
        return jerr(str(e))
    return web.json_response({"ok": True, "path": str(target)})


@routes.get("/api/settings")
async def get_settings(_):
    c = dict(CONFIG)
    c["root"] = str(ROOT)
    return web.json_response(c)


@routes.post("/api/settings")
async def set_settings(req):
    b = await req.json()
    restart = (b.get("engine_args", CONFIG["engine_args"]) != CONFIG["engine_args"]
               or b.get("mem_profile", CONFIG.get("mem_profile") or "auto") != (CONFIG.get("mem_profile") or "auto")
               or float(b.get("vram_headroom", CONFIG.get("vram_headroom") or 0) or 0) != float(CONFIG.get("vram_headroom") or 0))
    if "mem_guard_gb" in b:
        try:
            b["mem_guard_gb"] = max(0.5, min(16.0, float(b["mem_guard_gb"])))
        except (TypeError, ValueError):
            b.pop("mem_guard_gb")
    for k in ("hf_token", "civitai_token", "engine_args", "open_browser", "civitai_domain", "mem_profile", "mem_guard",
              "mem_guard_gb", "llm_unload_always", "vram_headroom"):
        if k in b:
            CONFIG[k] = b[k]
    save_config(CONFIG)
    if restart:
        ENGINE_.stop()
        ENGINE_.start()
    return web.json_response({"ok": True, "restarted": restart})


@routes.post("/api/input_upload")
async def input_upload(req):
    reader = await req.multipart()
    names = []
    async for part in reader:
        if part.name != "image" or not part.filename:
            continue
        ext = Path(part.filename).suffix.lower()
        if ext not in IMG_EXT:
            continue
        name = f"in_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}{ext}"
        with open(INPUT / name, "wb") as f:
            while True:
                c = await part.read_chunk(1 << 20)
                if not c:
                    break
                f.write(c)
        names.append(name)
    return web.json_response({"names": names})


SUBMIT = {"lock": None, "recent": {}}


def _fingerprint(p):
    return json.dumps(p, sort_keys=True, ensure_ascii=False)


def recent_duplicate(p):
    """Тот же запрос за последние секунды, пока прошлый ещё в работе, — вернуть его, а не ставить второй раз."""
    fp, now = _fingerprint(p), time.time()
    for k in [k for k, (t, _) in SUBMIT["recent"].items() if now - t > 30]:
        SUBMIT["recent"].pop(k, None)
    hit = SUBMIT["recent"].get(fp)
    if hit and now - hit[0] < 8:
        j = ENGINE_.jobs.get(hit[1])
        if j and j["status"] in ("queued", "running", "finishing"):
            return j
    return None


# ---------------------------------------------------------------- 3D: Hunyuan3D 2 (модель по ракурсам)
H3D_LABEL = {"front": "спереди", "left": "слева", "back": "сзади", "right": "справа"}


def prep_view(name, remove_bg, tag):
    """Ракурс для Hunyuan3D: объект на белом фоне, по центру квадрата 1024 с полями (так модель обучали)."""
    from PIL import Image
    import numpy as np
    with Image.open(INPUT / name) as im:
        im.load()
        rgba = im.convert("RGBA")
    a = np.asarray(rgba, np.float32)
    alpha = a[..., 3] / 255.0
    if remove_bg and alpha.min() > 0.99:          # непрозрачная картинка — вырезаем фон нейросетью
        m = MATTING["matte"](rgba)
        if m is not None:
            alpha = m
    rgb = a[..., :3] * alpha[..., None] + 255.0 * (1 - alpha[..., None])
    ys, xs = np.where(alpha > 0.5)
    if len(xs) > 16 and (alpha.min() < 0.99):
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        rgb = rgb[y0:y1, x0:x1]
    img = Image.fromarray(rgb.clip(0, 255).astype("uint8"))
    side = int(max(img.size) / 0.85)              # объект занимает ~85% кадра
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    canvas = canvas.resize((1024, 1024), Image.LANCZOS)
    out = f"3d_{tag}.png"
    canvas.save(INPUT / out)
    return out


@routes.post("/api/3d/generate")
async def gen_3d(req):
    p = await req.json()
    if ENGINE_.state != "ready":
        return jerr("Движок ещё не готов: " + ENGINE_.message)
    model = p.get("model") or ""
    if not model or not (MODELS / "checkpoints" / model).exists():
        return jerr("Не выбрана модель Hunyuan3D. Скачайте её: «Модели» → «3D».")
    views = {k: v for k, v in (p.get("views") or {}).items() if k in H3D_LABEL and v}
    for k, v in views.items():
        if not (INPUT / v).exists():
            return jerr(f"Не найдена картинка «{H3D_LABEL[k]}» — добавьте её заново.")
    multi = workflow.is_multiview_3d(model)
    if not views:
        return jerr("Добавьте хотя бы картинку «спереди».")
    if not multi:
        if "front" not in views:
            return jerr("Эта модель строит 3D по одной картинке — добавьте ракурс «спереди» "
                        "(для нескольких ракурсов скачайте Hunyuan3D‑2mv).")
        views = {"front": views["front"]}
    elif "front" not in views:
        return jerr("Ракурс «спереди» обязателен — остальные по желанию.")
    tag = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
    rb = bool(p.get("remove_bg", True))
    if rb and not MATTING["installed"]():
        rb = False
    try:
        prepared = {}
        for k, v in views.items():
            prepared[k] = await asyncio.get_running_loop().run_in_executor(None, prep_view, v, rb, f"{tag}_{k}")
    except Exception as e:
        return jerr("Не удалось подготовить картинки: " + str(e))
    low = model.lower()
    fast = "turbo" in low or "fast" in low
    seed = int(p.get("seed", -1))
    if seed < 0:
        seed = random.randint(0, 2**50)
    q = {"high": (512, 200000), "mid": (384, 50000)}.get(p.get("quality"), (256, 8000))
    g = {"family": "3d", "model": model.replace("/", os.sep), "multiview": multi, "views": prepared, "seed": seed,
         "steps": int(p.get("steps") or (10 if "turbo" in low else 30)), "cfg": 1.0 if fast else float(p.get("cfg") or 5.0),
         "octree": q[0], "num_chunks": q[1], "threshold": float(p.get("threshold", 0.6)),
         "latent_res": 4096 if multi or "2.1" in low else 3072}
    params = {**g, "views_src": views, "remove_bg": rb, "quality": p.get("quality", "fast"), "prompt": "", "loras": [],
              "width": 1024, "height": 1024, "batch": 1}
    return await queue_graph(workflow.build(g), params)


# ---------------------------------------------------------------- МУЗЫКА: ACE‑Step 1.5, MiniMax Music 3, YuE2, Stable Audio 3
MUSIC_ENGINES = {
    # движок: (папка модели, признак файла модели, (папка, признак) спутников..., шаги, cfg, макс. длительность)
    "ace": {"folder": "diffusion_models", "match": "acestep", "clip1": "qwen_0.6b_ace", "clip2": "qwen_4b_ace", "vae": "ace_1.5_vae", "max": 600},
    "mm3": {"folder": "diffusion_models", "match": "minimax_music3", "clip1": "minimax_music3_text_encoder", "vae": "minimax_music3_dav", "max": 300},
    "yue2": {"folder": "checkpoints", "match": "yue2", "max": 360},
    "sa3": {"folder": "checkpoints", "match": "stable_audio", "clip1": "t5gemma", "max": 380},
}
MUSIC_NAMES = {"ace": "ACE‑Step 1.5", "mm3": "MiniMax Music 3", "yue2": "YuE2", "sa3": "Stable Audio 3"}


def music_files(eng):
    """Файлы движка: модели и найденные спутники (энкодеры, VAE)."""
    c = MUSIC_ENGINES[eng]
    models = [x["name"] for x in list_models(c["folder"]) if c["match"] in x["name"].lower()]
    out = {"models": models}
    for k, folder in (("clip1", "text_encoders"), ("clip2", "text_encoders"), ("vae", "vae")):
        if k in c:
            out[k] = next((x["name"] for x in list_models(folder) if c[k] in x["name"].lower()), None)
    return out


@routes.get("/api/music/state")
async def music_state(_):
    return web.json_response({e: music_files(e) for e in MUSIC_ENGINES})


@routes.post("/api/music/generate")
async def gen_music(req):
    p = await req.json()
    if ENGINE_.state != "ready":
        return jerr("Движок ещё не готов: " + ENGINE_.message)
    eng = p.get("engine")
    if eng not in MUSIC_ENGINES:
        return jerr("Выберите нейросеть")
    if not (ENGINE / "comfy_extras" / "nodes_minimax_music.py").exists():
        return jerr("Движок устарел и не знает музыкальных нейросетей. Закройте программу и запустите update_engine.bat.")
    f = music_files(eng)
    model = p.get("model") if p.get("model") in f["models"] else (f["models"][0] if f["models"] else None)
    miss = [n for n, v in (("модель", model), ("текстовый энкодер", f.get("clip1", 1)), ("второй энкодер", f.get("clip2", 1)),
                           ("VAE", f.get("vae", 1))) if not v]
    if miss:
        return jerr(f"Для {MUSIC_NAMES[eng]} не хватает: {', '.join(miss)}. Откройте «Модели» → «Музыка» и скачайте файлы со звёздочкой.")
    tags, lyrics = (p.get("tags") or "").strip(), (p.get("lyrics") or "").strip()
    if not tags and not lyrics:
        return jerr("Опишите стиль музыки (жанр, инструменты, настроение) или впишите текст песни.")
    low = model.lower()
    steps, cfg = {"ace": (8, 1.0), "mm3": (30, 1.7), "yue2": (32, 1.0), "sa3": (8, 1.0)}[eng]
    if eng == "ace" and ("sft" in low or "base" in low):
        steps, cfg = 50, 7.0
    if p.get("steps"):
        steps = int(p["steps"])
    seed = int(p.get("seed", -1))
    if seed < 0:
        seed = random.randint(0, 2**50)
    dur = max(5.0, min(float(p.get("duration") or 120), MUSIC_ENGINES[eng]["max"]))
    g = {"engine": eng, "model": model.replace("/", os.sep), "clip1": f.get("clip1"), "clip2": f.get("clip2"), "vae": f.get("vae"),
         "tags": tags, "lyrics": "" if p.get("instrumental") and eng != "ace" else lyrics,
         "duration": dur, "seed": seed, "steps": steps, "cfg": cfg,
         "bpm": int(p.get("bpm") or 120), "keyscale": p.get("keyscale") or "C major", "timesig": str(p.get("timesig") or "4"),
         "language": p.get("language") or "ru"}
    if eng == "ace" and p.get("instrumental"):
        g["lyrics"] = "[Instrumental]"
    params = {**g, "family": "music", "prompt": tags, "prompt_user": tags, "loras": [], "width": 0, "height": 0, "batch": 1,
              "instrumental": bool(p.get("instrumental"))}
    return await queue_graph(workflow.build_music(g), params)


@routes.post("/api/generate")
async def generate(req):
    p = await req.json()
    if SUBMIT["lock"] is None:
        SUBMIT["lock"] = asyncio.Lock()
    async with SUBMIT["lock"]:
        fp = _fingerprint(p)
        dup = recent_duplicate(p)
        if dup:
            return web.json_response({"id": dup["id"], "seed": dup["params"].get("seed"), "duplicate": True})
        resp = await _generate(p)
        try:
            data = json.loads(resp.body)
            if data.get("id"):
                SUBMIT["recent"][fp] = (time.time(), data["id"])
        except Exception:
            pass
        return resp


async def _generate(p):
    if ENGINE_.state != "ready":
        return jerr("Движок ещё не готов: " + ENGINE_.message)
    for key, label in (("unet", "модель"), ("clip", "текстовый энкодер"), ("vae", "VAE")):
        if re.search(r"music|ace[_-]?step|yue|stable[_-]?audio", (p.get(key) or "").lower()):
            return jerr(f"В поле «{label}» выбран файл вкладки «Музыка» ({p[key]}) — он не подходит для картинок и видео. "
                        "Выберите в этом поле слева файл нужной нейросети (для MiniMax H3 энкодер — qwen3vl_32b_minimax_h3…). "
                        "Обновите страницу (Ctrl+F5), чтобы программа сама подставила правильный.")
    for key, folder, label in (("unet", "diffusion_models", "модель"), ("clip", "text_encoders", "текстовый энкодер"),
                               ("vae", "vae", "VAE")):
        if not p.get(key) or not (MODELS / folder / p[key]).exists():
            return jerr(f"Не выбран или не найден файл: {label}. Откройте «Модели».")
    for l in p.get("loras", []):
        if l.get("enabled", True) and not (MODELS / "loras" / l.get("name", "")).exists():
            return jerr(f"LoRA не найдена: {l.get('name')}")
    if gguf_error(p):
        return jerr(gguf_error(p))
    if p.get("family") == "minimax":
        return await _generate_video(p)
    if p.get("family") == "ltx":
        return await _generate_ltx(p)
    if p.get("family") == "flux3":
        p["character"] = None
        if not (ENGINE / "comfy" / "ldm" / "flux3").exists():
            return jerr("Движок устарел и не знает FLUX.3. Закройте программу и запустите update_engine.bat "
                        "в папке программы (модели и настройки не пострадают).")
    if p.get("family") == "krea2":
        p["character"] = None
        if p.get("use_image") and p.get("image_mode") == "reference" and not p.get("style_lora_off"):
            p["style_lora"] = find_krea_style_lora()
            p["style_strength"] = float(p.get("style_strength", 1.0))
    if p.get("family") == "qwen":
        p["turbo_sigmas"] = qwen_is_turbo(p)
    if not p.get("use_text") and not p.get("use_image") and not p.get("character"):
        return jerr("Включите текст, изображение или оба")
    if p.get("character"):
        refs, cm = CHARS["refs_for"](p["character"])
        if not refs:
            return jerr("Персонаж не найден или у него нет фото")
        p["char_refs"] = refs
        p["character_name"] = cm.get("name", "")
        p["prompt_user"] = p.get("prompt") or ""
        if p.get("char_phrase", True):
            fam = p.get("family", "qwen")
            n_ref = min(4, len(p.get("images") or [])) if (p.get("use_image") and p.get("image_mode", "reference") == "reference") else 0
            idx = n_ref + 1  # номер фото персонажа среди референсов (<imageN> для Qwen)
            p["prompt"] = ((p.get("prompt") or "").rstrip() + " " + CHARS["phrase_gen"](fam, idx)).strip()
            p["use_text"] = True
    if p.get("use_image"):
        imgs = [n for n in p.get("images", []) if (INPUT / n).exists()]
        if not imgs:
            return jerr("Режим изображения включён, но картинка не загружена")
        p["images"] = imgs
    seed = int(p.get("seed", -1))
    if seed < 0:
        seed = random.randint(0, 2**50)
    p["seed"] = seed
    return await queue_graph(workflow.build(await engine_params(p)), p)


def is_ref_model(name):
    """Файл модели H3 для режима референсов (Ref2VA) — по имени."""
    return bool(re.search(r"ref2?v", (name or "").lower()))


def is_turbo_lora(name):
    n = (name or "").lower()
    return "turbo" in n and ("h3" in n or "minimax" in n)


def find_turbo_lora(ref=False):
    """Турбо-LoRA MiniMax H3 из папки loras: для кадров — fl2v (8 шагов предпочтительнее 4), для референсов — ref2v."""
    names = [x["name"] for x in list_models("loras")
             if "turbo" in x["name"].lower() and ("h3" in x["name"].lower() or "minimax" in x["name"].lower())]
    names = [n for n in names if bool(re.search(r"ref2?v", n.lower())) == ref]
    names.sort(key=lambda n: (0 if "8step" in n.lower() else 1, n))
    return names[0] if names else None


def turbo_steps(lora):
    m = re.search(r"(\d+)\s*step", (lora or "").lower())
    return int(m.group(1)) if m else None


REF_TAG = re.compile(r"@img\s*(\d+)", re.I)


def mm_ref_prompt(text, refs):
    """Промпт для Ref2VA. @img1… → теги модели. Если это не готовый промпт по формату MiniMax (subject_definitions …),
    программа сама собирает минимальную структуру: <Subject i> — человек/место/предмет с <Picture i>."""
    text = (text or "").strip()
    used = sorted({int(m.group(1)) for m in REF_TAG.finditer(text)})
    bad = [i for i in used if i < 1 or i > len(refs)]
    if bad:
        raise ValueError(f"В промпте есть @img{bad[0]}, а референсов всего {len(refs)}.")
    if "subject_definitions" in text.lower():
        return REF_TAG.sub(lambda m: f"<Picture {m.group(1)}>", text)
    body = REF_TAG.sub(lambda m: f"<Subject {m.group(1)}>", text) or "The subjects from the reference pictures in a natural, lively scene."
    defs, keep = [], []
    for i, r in enumerate(refs, 1):
        label = (r.get("label") or "").strip()
        what = label if label else "the person, object or place"
        defs.append(f"<Subject {i}> is {what} in <Picture {i}>.")
        keep.append(f"<Subject {i}> (appears in [Shot 1]): fully_preserved - its appearance from <Picture {i}> is retained.")
    return ("subject_definitions:\n" + "\n".join(defs) + "\n\nsummary:\n[reference generation] " + body
            + "\n\nretention_analysis:\n" + "\n".join(keep) + "\n\ndetailed_description:\n[Shot 1] " + body
            + "\n\noverall_soundscape:\nNatural ambient sound that matches the scene and the actions.\n\nnon_diegetic_music:\nN/A")


async def _generate_video(p):
    """MiniMax H3: текст → видео, первый кадр → видео, первый + последний кадр → видео (со звуком);
    режим референсов (Ref2VA): до 9 картинок, в промпте @img1…@img9."""
    if not p.get("avae") or not (MODELS / "vae" / p["avae"]).exists():
        return jerr("Не выбран аудио‑VAE MiniMax H3 (minimax_h3_audio_vae_fp32). Скачайте его в «Модели» — он нужен всегда.")
    ref_mode = p.get("mm_mode") == "ref"
    if ref_mode:
        refs = [r for r in (p.get("refs") or []) if isinstance(r, dict) and r.get("name")][:9]
        for r in refs:
            if not (INPUT / r["name"]).exists():
                return jerr(f"Не найден референс {r['name']} — добавьте картинку заново.")
        if not refs:
            return jerr("Режим «Референсы»: добавьте хотя бы одну картинку (до 9).")
        if not is_ref_model(p.get("unet")):
            return jerr("Для референсов нужна отдельная модель MiniMax H3 Ref2VA (в имени файла «Ref2VA»). "
                        "Скачайте её в «Модели» → MiniMax H3 и выберите в поле «Модель (референсы)».")
        try:
            p["prompt_user"] = p.get("prompt") or ""
            p["prompt"] = mm_ref_prompt(p.get("prompt"), refs)
        except ValueError as e:
            return jerr(str(e))
        p["refs_meta"] = refs
        p["refs"] = [r["name"] for r in refs]
        p["first_frame"] = p["last_frame"] = None
    else:
        p["refs"] = None
        if is_ref_model(p.get("unet")):
            return jerr("Выбрана модель Ref2VA (для референсов). Для кадров и текста выберите модель FL2VA "
                        "или переключитесь на «🧩 Референсы».")
        for k, label in (("first_frame", "первый кадр"), ("last_frame", "последний кадр")):
            if p.get(k) and not (INPUT / p[k]).exists():
                return jerr(f"Не найден {label} — добавьте картинку заново.")
            if not p.get(k):
                p[k] = None
        p["prompt"] = (p.get("prompt") or "").strip()
        if not p["prompt"] and not p.get("first_frame") and not p.get("last_frame"):
            return jerr("Напишите, что должно происходить в видео, или добавьте первый кадр.")
    p["turbo_lora"] = None
    if p.get("turbo"):
        mine = [l for l in (p.get("loras") or []) if l.get("enabled", True) and is_turbo_lora(l.get("name"))
                and bool(re.search(r"ref2?v", l["name"].lower())) == ref_mode and (MODELS / "loras" / l["name"]).exists()]
        p["turbo_lora"] = mine[0]["name"] if mine else find_turbo_lora(ref_mode)
        p["turbo_strength"] = float(mine[0].get("strength", 1.0)) if mine else 1.0
        if not p["turbo_lora"]:
            return jerr("Не найдена турбо‑LoRA MiniMax H3 " + ("для референсов (ref2v turbo 4 step)" if ref_mode else "(fl2v turbo 8 step)")
                        + ". Скачайте её: окно LoRA → «★ Официальные», или снимите галочку «⚡ Турбо».")
        p["steps"] = turbo_steps(p["turbo_lora"]) or p.get("steps", 8)  # турбо‑LoRA обучена на своё число шагов
    # в стеке остаётся только подходящая турбо‑LoRA (чужая — от другого режима — к модели не подходит)
    p["loras"] = [l for l in (p.get("loras") or []) if not is_turbo_lora(l.get("name")) or l.get("name") == p["turbo_lora"]]
    # холст: по пропорциям первого (или последнего) кадра, иначе как задано
    frame = p.get("first_frame") or p.get("last_frame")
    if frame and p.get("fit_frame", True):
        try:
            from PIL import Image as _I
            with _I.open(INPUT / frame) as im:
                p["width"], p["height"] = workflow.h3_auto_canvas(*im.size)
        except Exception:
            pass
    p["width"], p["height"] = workflow.h3_canvas(p.get("width", 1344), p.get("height", 768))
    p["frames"] = workflow.h3_frames(p.get("duration", 5))
    p["use_text"], p["use_image"] = True, bool(frame or p.get("refs"))
    p["images"] = list(p["refs"]) if p.get("refs") else [x for x in (p.get("first_frame"), p.get("last_frame")) if x]
    p["batch"] = 1
    seed = int(p.get("seed", -1))
    if seed < 0:
        seed = random.randint(0, 2**50)
    p["seed"] = seed
    return await queue_graph(workflow.build(await engine_params(p)), p)


def find_krea_style_lora():
    for x in list_models("loras"):
        n = x["name"].lower()
        if "krea" in n and "style" in n and "ref" in n:
            return x["name"]
    return None


def qwen_is_turbo(p):
    """Qwen‑Image 2.1 Turbo (дистиллят на 8 шагов): отдельная модель или турбо‑LoRA в стеке."""
    if "turbo" in str(p.get("unet", "")).lower():
        return True
    return any(l.get("enabled", True) and "turbo" in str(l.get("name", "")).lower() and "qwen" in str(l.get("name", "")).lower()
               for l in p.get("loras") or [])


def find_ltx_upscaler():
    names = [x["name"] for x in list_models("latent_upscale_models") if "spatial" in x["name"].lower()]
    names.sort(key=lambda n: (0 if "2.5" in n else 1, n))
    return names[0] if names else None


async def _generate_ltx(p):
    """LTX‑2.5: текст → видео, первый кадр → видео, первый + последний кадр (со звуком).
    Апскейл ×2 латентным апскейлером; при наличии исходника он вшивается и после апскейла («упор на исходник»)."""
    if not p.get("avae") or not (MODELS / "vae" / p["avae"]).exists():
        return jerr("Не выбран аудио‑VAE LTX‑2.5 (ltx-2.5-audio-vae). Скачайте его в «Модели» → LTX‑2.5 — он нужен всегда.")
    for k, label in (("first_frame", "первый кадр"), ("last_frame", "последний кадр")):
        if p.get(k) and not (INPUT / p[k]).exists():
            return jerr(f"Не найден {label} — добавьте картинку заново.")
        if not p.get(k):
            p[k] = None
    p["refs"] = None
    p["prompt"] = (p.get("prompt") or "").strip()
    if not p["prompt"]:
        return jerr("Напишите, что должно происходить в видео (LTX нужен промпт и для режима с кадром).")
    p["upscaler"] = None
    if p.get("two_stage", True):
        up = p.get("upscaler_name")
        if up and (MODELS / "latent_upscale_models" / up).exists():
            p["upscaler"] = up
        else:
            p["upscaler"] = find_ltx_upscaler()
        if not p["upscaler"] and not p.get("last_frame"):
            return jerr("Для апскейла нужен файл ltx-2.5-latent-spatial-upscaler-x2 (≈1 ГБ). Скачайте его в «Модели» → LTX‑2.5 "
                        "или выключите «Апскейл ×2».")
    frame = p.get("first_frame") or p.get("last_frame")
    two = bool(p["upscaler"]) and not p.get("last_frame")
    if frame and p.get("fit_frame", True):
        try:
            from PIL import Image as _I
            with _I.open(INPUT / frame) as im:
                w, h = im.size
            area = float(p.get("width", 1280)) * float(p.get("height", 704))
            k = (area / (w * h)) ** 0.5
            p["width"], p["height"] = w * k, h * k
        except Exception:
            pass
    p["width"], p["height"] = workflow.ltx_canvas(p.get("width", 1280), p.get("height", 704), two)
    p["frames"] = workflow.ltx_frames(p.get("duration", 5))
    p["steps"] = 8 + (3 if two else 0)
    p["cfg"] = float(p.get("cfg") or 1.0)
    p["use_text"], p["use_image"] = True, bool(frame)
    p["images"] = [x for x in (p.get("first_frame"), p.get("last_frame")) if x]
    p["batch"] = 1
    seed = int(p.get("seed", -1))
    if seed < 0:
        seed = random.randint(0, 2**50)
    p["seed"] = seed
    return await queue_graph(workflow.build(await engine_params(p)), p)


def gguf_error(p):
    if str(p.get("unet", "")).lower().endswith(".gguf") and not gguf_support.installed(ENGINE):
        return ("Выбрана модель .gguf — для неё нужна поддержка GGUF в движке. "
                "Нажмите «Установить поддержку GGUF» под выбором модели (около 1 МБ, один раз).")
    if str(p.get("clip", "")).lower().endswith(".gguf"):
        return "Текстовый энкодер в формате .gguf не поддерживается — выберите .safetensors (например, Qwen3‑VL int8 или heretic w4a8)."
    return None


def check_models(p):
    """Текст ошибки или None."""
    if ENGINE_.state != "ready":
        return "Движок ещё не готов: " + ENGINE_.message
    if gguf_error(p):
        return gguf_error(p)
    for key, folder, label in (("unet", "diffusion_models", "модель"), ("clip", "text_encoders", "текстовый энкодер"),
                               ("vae", "vae", "VAE")):
        if not p.get(key) or not (MODELS / folder / p[key]).exists():
            return f"Не выбран или не найден файл: {label}. Откройте «Модели»."
    for l in p.get("loras", []):
        if l.get("enabled", True) and not (MODELS / "loras" / l.get("name", "")).exists():
            return f"LoRA не найдена: {l.get('name')}"
    # Редактор и замена лица работают только на Qwen-Image 2.1
    fam = model_family(MODELS / "diffusion_models" / p["unet"]) or ""
    names = {"krea2": "Krea 2", "flux3": "FLUX.3 Klein", "ltx": "LTX‑2.5", "minimax": "MiniMax H3"}
    if fam in names:
        return (f"Редактор и замена лица работают только на Qwen‑Image 2.1, а выбрана модель {names[fam]}. "
                "Переключитесь вверху на «Qwen‑Image 2.1» (Картинки) и выберите её модель. "
                "FLUX.3 Klein тоже умеет правку — во вкладке «Генерация»: включите «Изображение» → «Референс / правка».")
    return None


async def engine_params(p):
    """Копия параметров для движка: пути в подпапках — с системным разделителем (на Windows «\\»),
    LoRA для Qwen 2.1 — при необходимости исправленные копии."""
    g = dict(p)
    for k in ("unet", "clip", "vae"):
        if g.get(k):
            g[k] = g[k].replace("/", os.sep)
    loras = []
    for l in p.get("loras", []):
        l = dict(l)
        if l.get("enabled", True) and l.get("name") and CIVITAI["lora_arch"](l["name"]) == "checkpoint":
            continue  # целая модель в стеке LoRA — движку не отдаём (только зря грузит гигабайты)
        if l.get("enabled", True) and l.get("name"):
            if p.get("family") == "qwen":
                try:
                    l["te"] = bool(CIVITAI["lora_has_te"](l["name"]))  # есть часть для текстового энкодера
                except Exception:
                    l["te"] = False
                try:
                    fused = await asyncio.get_running_loop().run_in_executor(None, mlp_fused, p.get("unet"))
                    l["name"] = await asyncio.get_running_loop().run_in_executor(None, CIVITAI["prepare_qwen_lora"], l["name"], fused)
                except Exception as e:
                    logging_note(f"не удалось подготовить LoRA {l['name']}: {e}")
            l["name"] = l["name"].replace("/", os.sep)
        loras.append(l)
    g["loras"] = loras
    return g


def logging_note(msg):
    try:
        with open(LOGS / "app.log", "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    except Exception:
        pass


def lora_warnings(p):
    """Предупреждения, если LoRA сделана не для выбранной нейросети."""
    want = {"minimax": "minimax", "krea2": "krea2", "ltx": "ltx", "flux3": "flux3"}.get(p.get("family"), "qwen21")
    fam_name = {"minimax": "MiniMax H3", "krea2": "Krea 2", "ltx": "LTX‑2.5", "flux3": "FLUX.3 Klein"}.get(want, "Qwen‑Image 2.1")
    out = []
    for l in p.get("loras", []):
        if not l.get("enabled", True) or not l.get("name"):
            continue
        a = CIVITAI["lora_arch"](l["name"])
        if a in ("unknown", want):
            continue
        if a == "checkpoint":
            out.append(f"«{l['name']}» — это не LoRA, а целая модель (дообученная {fam_name}). "
                       "Как LoRA она не работает и была пропущена. Нажмите «Перенести в модели» на её карточке, "
                       "затем выберите её в поле «Модель (diffusion)» вместо обычной модели.")
            continue
        extra = " (LoRA от прежнего Qwen‑Image 20B с новой 2.1 несовместимы)" if a == "qwen_old" else ""
        out.append(f"LoRA «{l['name']}» сделана для {CIVITAI['arch_label'].get(a, a)}, а выбрана {fam_name} — "
                   f"она не подействует{extra}.")
    return out


async def queue_graph(graph, p, post=None, extra=None):
    try:  # процесс озвучки (вкладка «Голос») без дела держит видеопамять — отпускаем
        await VOICE["release"]()
    except Exception:
        pass
    try:  # и процесс синтеза речи (вкладка «Речь»)
        await SPEECH["release"]()
    except Exception:
        pass
    await LLM["stop_gpu"]()  # помощник промптов не должен занимать видеопамять во время генерации
    async with ClientSession() as s:
        async with s.post(ENGINE_.url + "/prompt", json={"prompt": graph, "client_id": ENGINE_.client_id}) as r:
            data = await r.json(content_type=None)
    if "prompt_id" not in data:
        errs = []
        if isinstance(data.get("error"), dict):
            errs.append(data["error"].get("message", "") + " " + data["error"].get("details", ""))
        for nid, ne in (data.get("node_errors") or {}).items():
            for e in ne.get("errors", []):
                errs.append(f"{ne.get('class_type', nid)}: {e.get('message')} {e.get('details', '')}")
        return jerr("Движок отклонил задание: " + "; ".join(errs))
    pid = data["prompt_id"]
    try:
        log_pos = ENGINE_.log_path.stat().st_size
    except OSError:
        log_pos = 0
    extra = dict(extra or {})
    warns = lora_warnings(p) + list(extra.pop("warnings", None) or [])
    ENGINE_.jobs[pid] = {"id": pid, "status": "queued", "stage": "В очереди", "value": 0, "max": 0,
                         "images": [], "error": "", "preview": None, "preview_n": 0, "graph": graph,
                         "params": p, "created": time.time(), "post": post, "log_pos": log_pos, "warnings": warns}
    return web.json_response({"id": pid, "seed": p.get("seed"), "warnings": warns, **(extra or {})})


@routes.get("/api/job/{id}")
async def job(req):
    j = ENGINE_.jobs.get(req.match_info["id"])
    if not j:
        return jerr("Нет задания", 404)
    return web.json_response({k: v for k, v in j.items() if k not in ("preview", "graph", "params", "post", "log_pos")})


@routes.get("/api/job/{id}/preview")
async def job_preview(req):
    j = ENGINE_.jobs.get(req.match_info["id"])
    if not j or not j.get("preview"):
        return web.Response(status=204)
    return web.Response(body=j["preview"], content_type="image/jpeg", headers={"Cache-Control": "no-store"})


@routes.post("/api/interrupt")
async def interrupt(_):
    async with ClientSession() as s:
        await s.post(ENGINE_.url + "/queue", json={"clear": True})
        await s.post(ENGINE_.url + "/interrupt", json={})
    for j in ENGINE_.jobs.values():
        if j["status"] in ("queued", "running"):
            j["status"], j["error"] = "cancelled", "Остановлено"
    return web.json_response({"ok": True})


@routes.get("/api/gallery")
async def gallery(_):
    items = []
    if HISTORY_FILE.exists():
        for line in HISTORY_FILE.read_text("utf-8", errors="replace").splitlines()[-300:]:
            try:
                e = json.loads(line)
                e["images"] = [u for u in e["images"] if (OUTPUT / unquote(u[len("/output/"):])).exists()]
                if e["images"]:
                    items.append(e)
            except Exception:
                pass
    return web.json_response({"items": list(reversed(items))})


@routes.post("/api/gallery/delete")
async def gallery_delete(req):
    b = await req.json()
    u = b.get("url", "")
    p = (OUTPUT / unquote(u[len("/output/"):])).resolve()
    if str(p).startswith(str(OUTPUT.resolve())) and p.is_file():
        p.unlink()
    return web.json_response({"ok": True})


async def on_startup(app):
    ENGINE_.start()
    app["watchdog"] = asyncio.create_task(ENGINE_.watchdog(app))
    app["labsync"] = asyncio.create_task(LAB["sync_loop"]())
    app["scoreidle"] = asyncio.create_task(SCORER["idle_loop"]())
    app["mattidle"] = asyncio.create_task(MATTING["idle_loop"]())


async def on_shutdown(app):
    LLM["stop"]()
    ENGINE_.stop()


import civitai  # noqa: E402
CIVITAI = civitai.register(routes, sys.modules[__name__])
import llm  # noqa: E402
LLM = llm.register(routes, sys.modules[__name__], CIVITAI)
import chars  # noqa: E402
CHARS = chars.register(routes, sys.modules[__name__])
import edit  # noqa: E402
EDIT = edit.register(routes, sys.modules[__name__], LLM)
import faceswap  # noqa: E402
FACE = faceswap.register(routes, sys.modules[__name__], EDIT)
import scorer  # noqa: E402
SCORER = scorer.register(routes, sys.modules[__name__])
import lab  # noqa: E402
LAB = lab.register(routes, sys.modules[__name__], LLM, CIVITAI)
import matting  # noqa: E402
MATTING = matting.register(sys.modules[__name__])
import voice  # noqa: E402
VOICE = voice.register(routes, sys.modules[__name__])
import speech  # noqa: E402
SPEECH = speech.register(routes, sys.modules[__name__])


def main():
    app = web.Application(client_max_size=1024**4, middlewares=[no_cache_static])
    app.add_routes(routes)
    app.router.add_static("/static", APP_DIR / "static")
    app.router.add_static("/output", OUTPUT)
    app.router.add_static("/input", INPUT)
    app.router.add_static("/loras", MODELS / "loras")
    app.router.add_static("/characters", ROOT / "characters")
    app.router.add_static("/journal/thumbs", LAB["thumbs"])
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    port = int(RUNTIME["port"])
    url = f"http://127.0.0.1:{port}"
    print(f"\n  Zouwu Lab (Qwen-Image 2.1 · Krea 2 · FLUX.3 · MiniMax H3 · LTX-2.5 · Hunyuan3D · Higgs TTS 3): {url}\n  Папка программы: {ROOT}\n  Закройте это окно, чтобы остановить.\n")
    if port != int(CONFIG["port"]):
        print(f"  (порт {CONFIG['port']} занят — наверное, уже открыта другая копия программы; эта работает на {port})\n")
    if CONFIG.get("open_browser", True) and "--no-browser" not in sys.argv:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    try:
        if UI_SOCK is not None:
            web.run_app(app, sock=UI_SOCK, print=None)
        else:
            web.run_app(app, host="127.0.0.1", port=port, print=None)
    finally:
        ENGINE_.stop()


if __name__ == "__main__":
    main()
