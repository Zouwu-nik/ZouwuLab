"""Скачивание готовой сборки llama.cpp (llama-server) в папку программы: engine/llama.
Только стандартная библиотека — используется и сервером (кнопка в интерфейсе), и setup.py.
"""
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

LLAMA_TAG = "b11138"  # проверенная версия llama.cpp
BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/"
EXE = "llama-server.exe" if os.name == "nt" else "llama-server"


def detect_variant():
    """cuda-13.4 для свежих драйверов NVIDIA (≥580), cuda-12.4 для старых, vulkan для AMD/Intel, иначе cpu."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        if out:
            major = int(out.splitlines()[0].split(".")[0])
            return "cuda-13.4" if major >= 580 else "cuda-12.4"
    except Exception:
        pass
    return "vulkan" if os.name == "nt" else "cpu"


def asset_names(variant):
    if os.name == "nt":
        names = [f"llama-{LLAMA_TAG}-bin-win-{variant}-x64.zip"]
        if variant.startswith("cuda"):
            names.append(f"cudart-llama-bin-win-{variant}-x64.zip")
        return names
    if variant == "vulkan":
        return [f"llama-{LLAMA_TAG}-bin-ubuntu-vulkan-x64.tar.gz"]
    return [f"llama-{LLAMA_TAG}-bin-ubuntu-x64.tar.gz"]


def _download(url, dest: Path, progress):
    req = urllib.request.Request(url, headers={"User-Agent": "ZouwuLab"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0) or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            progress(dest.name, done, total)


def install(dest: Path, variant=None, progress=lambda *a: None):
    variant = variant or detect_variant()
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "_dl"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        for name in asset_names(variant):
            arch = tmp / name
            _download(BASE + name, arch, progress)
            out = tmp / (name + "_x")
            if name.endswith(".zip"):
                with zipfile.ZipFile(arch) as z:
                    z.extractall(out)
            else:
                with tarfile.open(arch) as t:
                    t.extractall(out)
            # файлы могут лежать в подпапке — переносим всё из папки, где лежит exe/dll, в корень
            dirs = {p.parent for p in out.rglob("*") if (p.is_file() or p.is_symlink()) and p.suffix.lower() in (".exe", ".dll", ".so", "")}
            for d in dirs or {out}:
                for f in d.iterdir():
                    if f.is_file() or f.is_symlink():
                        target = dest / f.name
                        if target.exists() or target.is_symlink():
                            target.unlink()
                        shutil.move(str(f), str(target))
        exe = dest / EXE
        if not exe.exists():
            raise RuntimeError("В архиве не найден " + EXE)
        if os.name != "nt":
            for f in dest.iterdir():
                if f.is_file() and not f.suffix:
                    f.chmod(0o755)
        (dest / "variant.txt").write_text(f"{LLAMA_TAG} {variant}\n")
        return variant
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    v = sys.argv[1] if len(sys.argv) > 1 else None

    def pr(name, done, total):
        print(f"\r  {name}: {done / 1e6:.0f}/{total / 1e6:.0f} МБ", end="", flush=True)
    print("Вариант:", install(root / "engine" / "llama", v, pr))
