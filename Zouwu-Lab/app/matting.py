"""ИИ-вырезка фона: нейросеть ISNet (DIS, Apache-2.0), файл из набора rembg, ~171 МБ.

Скачивается только по кнопке, работает на процессоре через onnxruntime (~2–3 с на картинку,
меньше 1 ГБ ОЗУ), через 5 минут без дела выгружается. Видеокарту не трогает.
"""
import asyncio
import threading
import time

import numpy as np
from PIL import Image

URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx"
FILE = "isnet-general-use.onnx"
SIZE = "171 МБ"

ST = {"sess": None, "used": 0.0, "lock": threading.Lock(), "installing": False, "error": ""}


def deps():
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def register(S):
    DIR = S.MODELS / "matting"
    DIR.mkdir(parents=True, exist_ok=True)
    path = DIR / FILE

    def installed():
        return path.is_file() and deps()

    def state():
        t = next((t for t in S.TRANSFERS.tasks.values() if t.get("folder") == "matting"
                  and t["status"] in ("queued", "downloading")), None)
        return {"installed": installed(), "deps": deps(), "file": path.is_file(), "size": SIZE,
                "installing": ST["installing"] or bool(t), "error": ST["error"],
                "progress": round(100 * t["done"] / t["total"]) if t and t.get("total") else 0}

    async def install():
        if ST["installing"]:
            return
        ST["installing"], ST["error"] = True, ""
        try:
            if not deps():
                import gguf_support
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        None, lambda: gguf_support._pip(S.ROOT, "onnxruntime>=1.18", log=lambda *_: None))
                except Exception as e:
                    raise RuntimeError("Не удалось установить onnxruntime: " + str(e).split("\n", 1)[-1][-600:])
                import importlib
                importlib.invalidate_caches()
                if not deps():
                    raise RuntimeError("onnxruntime установлен, но не загружается — перезапустите программу.")
            if not path.is_file():
                task = S.TRANSFERS.new("download", "matting", FILE, URL)
                await S.TRANSFERS.download(task, URL)
                if task["status"] != "done":
                    raise RuntimeError(task.get("error") or "Скачивание не удалось")
        except Exception as e:
            ST["error"] = str(e)
        finally:
            ST["installing"] = False

    def session():
        if ST["sess"] is None:
            import onnxruntime as ort
            o = ort.SessionOptions()
            o.intra_op_num_threads = 4
            ST["sess"] = ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])
        ST["used"] = time.time()
        return ST["sess"]

    def matte(img):
        """Маска объекта 0..1 размером с картинку (None, если вырезка не установлена)."""
        if not installed():
            return None
        with ST["lock"]:
            s = session()
            im = img.convert("RGB").resize((1024, 1024), Image.LANCZOS)
            a = np.asarray(im, np.float32)
            a = a / max(float(a.max()), 1e-6) - 0.5
            x = a.transpose(2, 0, 1)[None].astype(np.float32)
            p = s.run(None, {s.get_inputs()[0].name: x})[0][0, 0]
            ST["used"] = time.time()
        p = (p - p.min()) / max(float(p.max() - p.min()), 1e-6)
        m = Image.fromarray((p * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
        return np.asarray(m, np.float32) / 255.0

    async def idle_loop():
        while True:
            await asyncio.sleep(60)
            if ST["sess"] is not None and time.time() - ST["used"] > 300:
                with ST["lock"]:
                    ST["sess"] = None

    return {"installed": installed, "state": state, "install": install, "matte": matte, "idle_loop": idle_loop}
