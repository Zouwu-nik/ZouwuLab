"""Установка движка и библиотек в папку программы (вызывается из install.bat).

Всё ставится локально: Python (python/), окружение (venv/), кэш пакетов (cache/),
движок генерации (engine/ComfyUI — только ядро, его интерфейс не используется).
"""
import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ENGINE_VERSION = "v0.42.3"  # версия ядра ComfyUI: Qwen-Image 2.1, Krea 2, FLUX.3 Klein, MiniMax H3, LTX-2.5, Hunyuan3D
ENGINE_ZIP = f"https://github.com/Comfy-Org/ComfyUI/archive/refs/tags/{ENGINE_VERSION}.zip"

ROOT = Path(__file__).resolve().parent.parent
PY = Path(sys.executable)
UV = ROOT / "tools" / ("uv.exe" if os.name == "nt" else "uv")
ENGINE_DIR = ROOT / "engine" / "ComfyUI"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def say(msg):
    print(f"\n=== {msg}", flush=True)


def run(args):
    print("  >", " ".join(str(a) for a in args), flush=True)
    r = subprocess.run([str(a) for a in args])
    if r.returncode != 0:
        raise SystemExit(f"\nОшибка: команда завершилась с кодом {r.returncode}")


def uv_pip(*args):
    run([UV, "pip", "install", "--python", PY, *args])


def detect_gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return out.splitlines()[0] if out else ""
    except Exception:
        return ""


def download(url, dest: Path):
    req = urllib.request.Request(url, headers={"User-Agent": "ZouwuLab-setup"})
    with urllib.request.urlopen(req) as r:
        total = int(r.headers.get("Content-Length", 0) or 0)
        buf = io.BytesIO()
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            buf.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.1f} / {total / 1e6:.1f} МБ", end="", flush=True)
            else:
                print(f"\r  {done / 1e6:.1f} МБ", end="", flush=True)
        print()
        dest.write_bytes(buf.getvalue())


def install_engine(force=False):
    if (ENGINE_DIR / "main.py").exists() and not force:
        print("  Движок уже установлен:", ENGINE_DIR)
        return
    say(f"Скачиваю ядро движка ComfyUI {ENGINE_VERSION}")
    (ROOT / "engine").mkdir(exist_ok=True)
    tmp = ROOT / "engine" / "engine.zip"
    download(ENGINE_ZIP, tmp)
    keep = ROOT / "engine" / "_keep_nodes"   # установленные нами узлы (GGUF) переживают обновление движка
    shutil.rmtree(keep, ignore_errors=True)
    nodes = ENGINE_DIR / "custom_nodes"
    if nodes.exists():
        keep.mkdir(parents=True)
        for d in nodes.iterdir():
            if d.is_dir() and d.name == "ComfyUI-GGUF":
                shutil.move(str(d), str(keep / d.name))
    staging = ROOT / "engine" / "_unpack"
    shutil.rmtree(staging, ignore_errors=True)
    with zipfile.ZipFile(tmp) as z:
        z.extractall(staging)
    inner = next(staging.iterdir())
    if ENGINE_DIR.exists():
        # сохраняем пользовательские данные движка, если были
        shutil.rmtree(ENGINE_DIR)
    shutil.move(str(inner), str(ENGINE_DIR))
    shutil.rmtree(staging, ignore_errors=True)
    tmp.unlink(missing_ok=True)
    if keep.exists():
        (ENGINE_DIR / "custom_nodes").mkdir(exist_ok=True)
        for d in keep.iterdir():
            shutil.move(str(d), str(ENGINE_DIR / "custom_nodes" / d.name))
        shutil.rmtree(keep, ignore_errors=True)


def install_gguf():
    say("Поддержка моделей .gguf (узлы ComfyUI-GGUF)")
    try:
        import gguf_support
        if gguf_support.installed(ENGINE_DIR) and gguf_support.installed_commit(ENGINE_DIR) == gguf_support.GGUF_COMMIT:
            print("  уже установлена")
            return
        gguf_support.install(ROOT, ENGINE_DIR, log=print)
        print("  готово")
    except Exception as e:
        print("  не удалось (можно установить позже из интерфейса):", e)


NEED_GB = 12   # библиотеки ~7 ГБ + временные архивы во время установки (модели — отдельно)


def check_folder():
    """Проверки места установки. Диск любой (C:, D:, внешний) — программа целиком живёт в своей папке."""
    root = str(ROOT)
    say("Папка программы: " + root)
    try:
        free = shutil.disk_usage(root).free / 1024 ** 3
        print(f"  свободно на диске: {free:.0f} ГБ (нужно ~{NEED_GB} ГБ на установку + место под модели)")
        if free < NEED_GB:
            raise SystemExit(f"\nОшибка: на диске меньше {NEED_GB} ГБ свободного места. Освободите место "
                             "или перенесите папку программы на другой диск и запустите install.bat снова.")
    except OSError:
        pass
    warn = []
    if any(ord(ch) > 127 for ch in root):
        warn.append("в пути есть русские (не латинские) буквы — некоторые библиотеки с этим работают плохо")
    if " " in root:
        warn.append("в пути есть пробелы")
    if len(root) > 80:
        warn.append(f"путь длинный ({len(root)} символов) — в Windows длинные пути внутри библиотек могут не создаться")
    low = root.lower()
    if "onedrive" in low or "dropbox" in low or "яндекс" in low or "google drive" in low:
        warn.append("папка лежит в облачной синхронизации — она будет пытаться выгрузить десятки гигабайт")
    if warn:
        print("  ВНИМАНИЕ: " + "; ".join(warn) + ".")
        print("  Установка продолжится. Если будут ошибки — перенесите папку в короткий путь без пробелов\n"
              "  и русских букв (например C:\\ZouwuLab или D:\\ZouwuLab) и запустите install.bat снова.")


def main():
    args = sys.argv[1:]
    if "--update-engine" not in args:
        check_folder()
    for d in ["models/diffusion_models", "models/text_encoders", "models/vae", "models/loras", "input", "output", "logs", "cache"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    if "--update-engine" in args:
        install_engine(force=True)
        index = installed_torch_index(args)
        say("Библиотеки движка (PyTorch остаётся прежним)")
        torch_extras(index)
        uv_pip("-r", ENGINE_DIR / "requirements.txt", "-c", ROOT / "cache" / "constraints.txt",
               "--extra-index-url", index, "--index-strategy", "unsafe-best-match")
        install_gguf()
        say("Движок обновлён до " + ENGINE_VERSION)
        return

    gpu = detect_gpu()
    say("Видеокарта: " + (gpu or "NVIDIA не найдена"))
    index = torch_index(args, gpu)

    say(f"Устанавливаю PyTorch ({index.rsplit('/', 1)[-1]}) — это самый большой пакет, ~2.5 ГБ")
    uv_pip("torch", "torchvision", "--index-url", index)

    torch_extras(index)

    install_engine()

    say("Устанавливаю библиотеки движка")
    uv_pip("-r", ENGINE_DIR / "requirements.txt", "-c", ROOT / "cache" / "constraints.txt",
           "--extra-index-url", index, "--index-strategy", "unsafe-best-match")

    install_gguf()

    say("Движок помощника промптов (llama.cpp)")
    try:
        import llm_install
        if not (ROOT / "engine" / "llama" / llm_install.EXE).exists():
            v = llm_install.install(ROOT / "engine" / "llama", None,
                                    lambda n, d, t: print(f"\r  {n}: {d / 1e6:.0f}/{t / 1e6:.0f} МБ", end="", flush=True))
            print("\n  установлен вариант:", v)
        else:
            print("  уже установлен")
    except Exception as e:
        print("  не удалось (можно установить позже из интерфейса):", e)

    say("Проверка")
    run([PY, "-c", "import torch;print('  CUDA доступна:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"])
    say("Готово! Запускайте start.bat")


def pyver(mod):
    return subprocess.run([PY, "-c", f"import {mod};print({mod}.__version__)"], capture_output=True, text=True).stdout.strip()


def installed_torch_index(args):
    """Индекс PyTorch, соответствующий уже установленной сборке (cu128, cpu, …)."""
    import re
    m = re.search(r"\+(cu\d+|cpu|rocm[\d.]+|xpu)", pyver("torch"))
    return "https://download.pytorch.org/whl/" + m.group(1) if m else torch_index(args)


def torch_extras(index):
    """torchaudio той же сборки, что и torch (новому ядру он нужен), и фиксация версий."""
    ver = pyver("torch")
    (ROOT / "cache").mkdir(exist_ok=True)
    cons = ROOT / "cache" / "constraints.txt"
    cons.write_text(f"torch=={ver}\n", "utf-8")
    try:
        uv_pip("torchaudio", "--index-url", index, "-c", cons)
    except SystemExit:
        print("  torchaudio не установился — не страшно, он нужен только для звуковых узлов движка")
    lines = [f"torch=={ver}"]
    for mod in ("torchvision", "torchaudio"):
        v = pyver(mod)
        if v:
            lines.append(f"{mod}=={v}")
    cons.write_text("\n".join(lines) + "\n", "utf-8")
    print("  " + ", ".join(lines))


def torch_index(args, gpu=None):
    if "--cpu" in args:
        return "https://download.pytorch.org/whl/cpu"
    for a in args:
        if a.startswith("--cuda="):
            return "https://download.pytorch.org/whl/" + a.split("=", 1)[1]
    if gpu is None:
        gpu = detect_gpu()
    if not gpu:
        print("\n  ВНИМАНИЕ: nvidia-smi не найден. Ставлю версию для NVIDIA (CUDA 12.8) всё равно.\n"
              "  Если видеокарты NVIDIA нет — запустите:  install.bat --cpu  (будет очень медленно)")
    return "https://download.pytorch.org/whl/cu128"


if __name__ == "__main__":
    main()
