"""Поддержка моделей .gguf (квантованные Qwen-Image 2.1 / FLUX от unsloth, leejet и др.).

Ставит в движок узлы ComfyUI-GGUF (версия leejet — именно она знает Qwen-Image 2.1) и библиотеку gguf.
Вызывается кнопкой в интерфейсе или из setup.py. Всё остаётся в папке программы.
"""
import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

GGUF_REPO = "leejet/ComfyUI-GGUF"
GGUF_COMMIT = "373048b8403a7820620065210a691263d4da0a61"  # 2026-09-25: поддержка qwen_image21
GGUF_ZIPS = [f"https://codeload.github.com/{GGUF_REPO}/zip/{GGUF_COMMIT}",
             f"https://github.com/{GGUF_REPO}/archive/{GGUF_COMMIT}.zip"]
NODE_DIR_NAME = "ComfyUI-GGUF"


def node_dir(engine_dir: Path) -> Path:
    return Path(engine_dir) / "custom_nodes" / NODE_DIR_NAME


def installed(engine_dir: Path) -> bool:
    d = node_dir(engine_dir)
    if not (d / "nodes.py").exists():
        return False
    try:
        import importlib
        import importlib.util
        importlib.invalidate_caches()
        return importlib.util.find_spec("gguf") is not None
    except Exception:
        return False


def installed_commit(engine_dir: Path) -> str:
    f = node_dir(engine_dir) / ".flux2studio_commit"
    return f.read_text().strip() if f.exists() else ""


def _download(url, progress=None):
    req = urllib.request.Request(url, headers={"User-Agent": "ZouwuLab"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length", 0) or 0)
        buf, done = io.BytesIO(), 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            buf.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
        return buf.getvalue()


def _pip(root: Path, *pkgs, log=print):
    uv = root / "tools" / ("uv.exe" if os.name == "nt" else "uv")
    if uv.exists():
        cmd = [str(uv), "pip", "install", "--python", sys.executable, *pkgs]
    else:
        cmd = [sys.executable, "-m", "pip", "install", *pkgs]
    env = dict(os.environ)
    env.setdefault("UV_CACHE_DIR", str(root / "cache" / "uv"))
    env.setdefault("PIP_CACHE_DIR", str(root / "cache" / "pip"))
    env.setdefault("UV_LINK_MODE", "copy")
    log("  > " + " ".join(cmd))
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, creationflags=flags)
    if r.returncode != 0:
        raise RuntimeError("Не удалось установить библиотеку gguf:\n" + (r.stderr or r.stdout)[-1500:])


def install(root: Path, engine_dir: Path, log=print, progress=None):
    """Скачать узлы ComfyUI-GGUF (нужной версии) и поставить библиотеку gguf."""
    engine_dir = Path(engine_dir)
    if not (engine_dir / "main.py").exists():
        raise RuntimeError("Движок не установлен — сначала install.bat")
    data, last = None, ""
    for u in GGUF_ZIPS:
        try:
            log("  Скачиваю " + u)
            data = _download(u, progress)
            break
        except Exception as e:
            last = str(e)
    if data is None:
        raise RuntimeError("Не удалось скачать ComfyUI-GGUF: " + last)
    dst = node_dir(engine_dir)
    tmp = dst.with_name(NODE_DIR_NAME + "_new")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        top = z.namelist()[0].split("/")[0] + "/"
        for n in z.namelist():
            if not n.startswith(top) or n.endswith("/"):
                continue
            out = tmp / n[len(top):]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(n))
    (tmp / ".flux2studio_commit").write_text(GGUF_COMMIT)
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    _pip(Path(root), "gguf>=0.13.0", log=log)
    return GGUF_COMMIT
