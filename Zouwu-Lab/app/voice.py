"""Вкладка «Голос»: переозвучка роликов голосом эталона.

1. Эталон — видео/звук, где персонаж говорит нужным голосом: берётся отрезок 3–10 с, чистится от фона,
   текст эталона распознаётся сам (Whisper large-v3-turbo).
2. Ролик, который надо переозвучить: звук отделяется от фона, речь распознаётся по фразам с таймингом.
3. Пользователь правит текст фраз (и при желании — их время) и нажимает «Озвучить».
4. Каждая фраза синтезируется голосом эталона ровно в длину своей фразы (OmniVoice, duration),
   фраза ставится на своё место, фон возвращается, звук вклеивается в то же видео.

Модели и процесс работы — в отдельном постоянном процессе voice_worker.py (команды через stdin).
Установка — только по кнопке; уже установленные библиотеки движка не меняются.
"""
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

from aiohttp import web

OMNIVOICE_PKG = "omnivoice==0.2.1"
SEP_MODEL = "mel_band_roformer_kim_ft_unwa.ckpt"
SEP_SIZE = 913_100_690
SEP_URL = "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/"
PIP = ["audio-separator==0.47.0", "onnxruntime", "librosa", "audioread", "requests", "imageio-ffmpeg", "pydub",
       "accelerate", "num2words", "soundfile"]
REPOS = {  # папка → репозиторий Hugging Face
    "omnivoice": "k2-fsa/OmniVoice",
    "whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
}
TOKENIZER_REPO = "eustlb/higgs-audio-v2-tokenizer"   # если в репозитории OmniVoice нет audio_tokenizer/
IMPORT_CHECK = ("import audio_separator.separator, librosa, audioread, torchaudio, pydub;"
                "from audio_separator.separator.uvr_lib_v5 import spec_utils;"
                "from audio_separator.separator.architectures import mdxc_separator;"
                "from omnivoice import OmniVoice;from transformers import pipeline, HiggsAudioV2TokenizerModel")
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus"}
IDLE_UNLOAD = 300   # через 5 минут без дела процесс озвучки закрывается и освобождает видеопамять


def _wanted(path, size):
    p = path.lower()
    if p.startswith(".") or "/." in p or p.endswith((".md", ".png", ".jpg", ".jpeg", ".gif", ".gitattributes")):
        return False
    if any(x in p for x in ("flax_model", "tf_model", ".msgpack", ".h5", "onnx/", ".onnx", "openvino", "/gguf", ".gguf")):
        return False
    return True


def register(routes, S):
    ROOT, MODELS, INPUT, OUTPUT = S.ROOT, S.MODELS, S.INPUT, S.OUTPUT
    VDIR = MODELS / "voice"
    SEP = VDIR / "separator"
    BIN = VDIR / "bin"
    UP = INPUT / "voice"
    OUT = OUTPUT / "voice"
    TMP = ROOT / "engine" / "temp" / "voice"
    MARK = VDIR / "installed_v2.json"
    for d in (UP, OUT):
        d.mkdir(parents=True, exist_ok=True)
    FF = BIN / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")

    ST = {"running": False, "step": "", "done": 0, "total": 0, "error": "", "log": []}
    W = {"proc": None, "reqs": {}, "last": 0.0, "lock": asyncio.Lock()}

    def err(msg, code=400):
        return web.json_response({"error": str(msg)}, status=code)

    def log(msg):
        ST["log"].append(time.strftime("%H:%M:%S ") + msg)
        ST["log"] = ST["log"][-60:]

    def models_ok():
        ov = VDIR / "omnivoice"
        return ((ov / "config.json").exists() and ((ov / "audio_tokenizer" / "config.json").exists())
                and (VDIR / "whisper-large-v3-turbo" / "config.json").exists())

    def sep_ok():
        f = SEP / SEP_MODEL
        return f.exists() and f.stat().st_size == SEP_SIZE

    def installed():
        return MARK.exists() and models_ok() and FF.exists() and sep_ok()

    def state():
        return {"installed": installed(), "running": ST["running"], "step": ST["step"], "done": ST["done"],
                "total": ST["total"], "error": ST["error"], "log": ST["log"][-12:],
                "worker": bool(W["proc"] and W["proc"].returncode is None)}

    # ------------------------------------------------------------ установка
    def run_cmd(args, what):
        log(what)
        r = subprocess.run([str(a) for a in args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           cwd=str(ROOT))
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-6:]
            raise RuntimeError(what + ": " + " | ".join(tail))
        return r.stdout

    def http_get(url, dest: Path, label, size=0):
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        have = part.stat().st_size if part.exists() else 0
        hdr = {"User-Agent": "ZouwuLab/1.0"}
        tok = (S.CONFIG.get("hf_token") or "").strip()
        if tok and "huggingface.co" in url:
            hdr["Authorization"] = "Bearer " + tok
        if have:
            hdr["Range"] = f"bytes={have}-"
        with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=60) as r:
            if have and r.status != 206:
                have = 0
            ST["step"] = label
            with open(part, "ab" if have else "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    ST["done"] += len(b)
        part.replace(dest)

    def py(code):
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace")

    def repo_files(repo):
        from huggingface_hub import HfApi
        tok = (S.CONFIG.get("hf_token") or "").strip() or None
        files = []
        for it in HfApi().list_repo_tree(repo, recursive=True, token=tok):
            size = getattr(it, "size", None)
            if size is None or not hasattr(it, "size"):
                continue
            if _wanted(it.path, size):
                files.append((it.path, int(size)))
        names = {p for p, _ in files}
        if any(p.endswith(".safetensors") for p in names):   # веса в двух форматах — берём safetensors
            files = [(p, s) for p, s in files if not p.endswith((".bin", ".pt", ".pth")) or "audio_tokenizer" in p]
        return files

    def install_sync():
        uv = ROOT / "tools" / ("uv.exe" if os.name == "nt" else "uv")
        ST["step"] = "Библиотеки (OmniVoice, audio-separator…)"
        if uv.exists():
            frz = run_cmd([uv, "pip", "freeze", "--python", sys.executable], "Список установленных библиотек")
            TMP.mkdir(parents=True, exist_ok=True)
            pins = [l for l in frz.splitlines() if re.match(r"^[A-Za-z0-9_.\-]+==", l)]
            cons = TMP / "constraints.txt"
            cons.write_text("\n".join(pins) + "\n", "utf-8")
            # OmniVoice нужен transformers ≥ 5.3 — если стоит старее, разрешаем обновить только его
            tv = next((l.split("==")[1] for l in pins if l.lower().startswith("transformers==")), "")
            old_tf = bool(tv) and tuple(int(x) for x in re.findall(r"\d+", tv)[:2]) < (5, 3)
            if old_tf:
                cons.write_text("\n".join(l for l in pins if not l.lower().startswith(("transformers==", "tokenizers==",
                                                                                         "huggingface-hub==", "safetensors=="))) + "\n", "utf-8")
            extra = ["transformers>=5.3"] if old_tf else []
            run_cmd([uv, "pip", "install", "--python", sys.executable, "-c", cons, *PIP, *extra], "Устанавливаю библиотеки")
            run_cmd([uv, "pip", "install", "--python", sys.executable, "--no-deps", OMNIVOICE_PKG], "Устанавливаю OmniVoice")
            if py("import torchaudio.functional").returncode != 0:
                import torch
                m = re.search(r"\+(cu\d+|cpu)", torch.__version__)
                idx = "https://download.pytorch.org/whl/" + (m.group(1) if m else "cpu")
                run_cmd([uv, "pip", "install", "--python", sys.executable, "torchaudio", "--index-url", idx, "-c", cons],
                        "Устанавливаю torchaudio")
        else:
            run_cmd([sys.executable, "-m", "pip", "install", *PIP], "Устанавливаю библиотеки (pip)")
            run_cmd([sys.executable, "-m", "pip", "install", "--no-deps", OMNIVOICE_PKG], "Устанавливаю OmniVoice (pip)")
        ST["step"] = "Проверка библиотек"
        r = py(IMPORT_CHECK)
        if r.returncode != 0:
            raise RuntimeError("Библиотеки не загружаются: " + " | ".join(r.stderr.strip().splitlines()[-3:]))
        # модели: список файлов и общий размер — чтобы показывать честный прогресс
        ST["step"] = "Список файлов моделей"
        plan = []
        for folder, repo in REPOS.items():
            for p, s in repo_files(repo):
                plan.append((repo, p, VDIR / folder / p, s))
        if not any("audio_tokenizer/" in str(x[1]) for x in plan if x[0] == REPOS["omnivoice"]):
            for p, s in repo_files(TOKENIZER_REPO):
                plan.append((TOKENIZER_REPO, p, VDIR / "omnivoice" / "audio_tokenizer" / p, s))
        todo = [x for x in plan if not (x[2].exists() and x[2].stat().st_size == x[3])]
        ST["done"], ST["total"] = 0, sum(x[3] for x in todo) + (0 if sep_ok() else SEP_SIZE)
        log(f"Скачиваю модели: {len(todo)} файлов, {ST['total'] / 1e9:.1f} ГБ")
        for repo, p, dest, s in todo:
            start = ST["done"]
            http_get(f"https://huggingface.co/{repo}/resolve/main/{p}", dest, f"{repo.split('/')[-1]} · {Path(p).name}")
            ST["done"] = start + s
        # ffmpeg — из пакета imageio-ffmpeg
        if not FF.exists():
            exe = py("import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())").stdout.strip()
            if not exe or not Path(exe).exists():
                raise RuntimeError("Не найден ffmpeg из пакета imageio-ffmpeg")
            BIN.mkdir(parents=True, exist_ok=True)
            shutil.copy(exe, FF)
        # модель разделения голоса и фона: качаем сами (с докачкой и проверкой размера),
        # затем audio-separator докачивает свои мелкие служебные файлы
        SEP.mkdir(parents=True, exist_ok=True)
        ck = SEP / SEP_MODEL
        if ck.exists() and ck.stat().st_size != SEP_SIZE:
            log("Файл модели разделения повреждён (недокачан) — качаю заново")
            ck.unlink()
        if not ck.exists():
            http_get(SEP_URL + SEP_MODEL, ck, "Модель разделения голоса и фона (MelBand-Roformer)")
            if ck.stat().st_size != SEP_SIZE:
                ck.unlink()
                raise RuntimeError("Модель разделения скачалась не полностью — нажмите «Установить» ещё раз")
        yml = SEP / ("config_" + SEP_MODEL.replace(".ckpt", ".yaml"))
        if not yml.exists():
            http_get(SEP_URL + yml.name, yml, "Настройки модели разделения")
        ST["step"] = "Служебные файлы модели разделения"
        env = dict(os.environ, PATH=str(BIN) + os.pathsep + os.environ.get("PATH", ""))
        code = ("import logging;from audio_separator.separator import Separator;"
                f"s=Separator(log_level=logging.WARNING,model_file_dir={str(SEP)!r},output_dir={str(TMP)!r});"
                f"s.load_model(model_filename={SEP_MODEL!r})")
        r2 = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r2.returncode != 0:
            raise RuntimeError("Модель разделения не загружается: " + " | ".join((r2.stderr or "").strip().splitlines()[-3:]))
        ST["step"] = "Проверка"
        r = py(IMPORT_CHECK)
        if r.returncode != 0:
            raise RuntimeError("Проверка не прошла: " + " | ".join(r.stderr.strip().splitlines()[-4:]))
        if not models_ok():
            raise RuntimeError("Не все файлы моделей скачались — нажмите «Установить» ещё раз (докачает недостающее)")
        MARK.write_text(json.dumps({"omnivoice": OMNIVOICE_PKG, "sep": SEP_MODEL, "time": time.time()}), "utf-8")

    async def install_task():
        if ST["running"]:
            return
        ST.update(running=True, error="", step="Начинаю", done=0, total=0, log=[])
        try:
            await asyncio.get_running_loop().run_in_executor(None, install_sync)
            ST["step"] = "Готово"
            log("Голосовой модуль установлен")
        except Exception as e:
            ST["error"] = str(e)
            log("Ошибка: " + str(e))
        finally:
            ST["running"] = False

    @routes.get("/api/voice/state")
    async def v_state(_):
        return web.json_response(state())

    @routes.post("/api/voice/install")
    async def v_install(_):
        asyncio.create_task(install_task())
        await asyncio.sleep(0.2)
        return web.json_response(state())

    # ------------------------------------------------------------ файлы
    def probe(path: Path):
        info = {"duration": 0.0, "video": False, "audio": False}
        try:
            import av
            with av.open(str(path)) as c:
                info["duration"] = float(c.duration) / 1e6 if c.duration else 0.0
                info["video"] = any(s.type == "video" for s in c.streams)
                info["audio"] = any(s.type == "audio" for s in c.streams)
        except Exception:
            info["video"] = path.suffix.lower() in VIDEO_EXT
            info["audio"] = True
        return info

    def in_up(name):
        p = (UP / (name or "")).resolve()
        return p if name and str(p).startswith(str(UP.resolve())) and p.is_file() else None

    @routes.post("/api/voice/upload")
    async def v_upload(req):
        rd = await req.multipart()
        out = []
        async for part in rd:
            if not part.filename:
                continue
            ext = Path(part.filename).suffix.lower()
            if ext not in VIDEO_EXT | AUDIO_EXT:
                continue
            stem = re.sub(r"[^\w\-. ]+", "_", Path(part.filename).stem)[:60].strip() or "clip"
            name = f"{stem}_{uuid.uuid4().hex[:5]}{ext}"
            with open(UP / name, "wb") as f:
                while True:
                    ch = await part.read_chunk(1 << 20)
                    if not ch:
                        break
                    f.write(ch)
            out.append({"name": name, "title": part.filename, **probe(UP / name)})
        if not out:
            return err("Нужен файл видео (mp4, mov, mkv, webm) или звука (wav, mp3, m4a…)")
        return web.json_response({"files": out})

    @routes.post("/api/voice/from_output")
    async def v_from_output(req):
        b = await req.json()
        rel = (b.get("url") or "").split("?")[0]
        if not rel.startswith("/output/"):
            return err("Нужна ссылка на файл из output")
        from urllib.parse import unquote
        src = (OUTPUT / unquote(rel[len("/output/"):])).resolve()
        if not str(src).startswith(str(OUTPUT.resolve())) or not src.is_file():
            return err("Файл не найден")
        name = f"{src.stem[:60]}_{uuid.uuid4().hex[:5]}{src.suffix.lower()}"
        shutil.copy(src, UP / name)
        return web.json_response({"files": [{"name": name, "title": src.name, **probe(UP / name)}]})

    # ------------------------------------------------------------ процесс озвучки
    async def free_gpu():
        try:
            from aiohttp import ClientSession
            async with ClientSession() as s:
                async with s.post(S.ENGINE_.url + "/free", json={"unload_models": True, "free_memory": True}) as r:
                    await r.read()
        except Exception:
            pass
        try:
            await S.LLM["stop_gpu"]()
        except Exception:
            pass
        try:  # процесс синтеза речи (вкладка «Речь») без дела — отпускаем видеопамять
            await S.SPEECH["release"]()
        except Exception:
            pass

    async def reader(proc):
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            s = line.decode("utf-8", errors="replace").strip()
            if not s.startswith("{"):
                continue
            try:
                e = json.loads(s)
            except ValueError:
                continue
            q = W["reqs"].get(e.get("id"))
            if not q:
                continue
            ev = e.pop("ev", "")
            e.pop("id", None)
            if ev == "stage":
                q.update(status="running", msg=e.get("msg", ""), p=e.get("p", q.get("p", 0)))
            elif ev == "result":
                q.update(status="done", msg="Готово", p=1, result=e, finished=time.time())
            elif ev == "error":
                q.update(status="error", msg=e.get("msg", "Ошибка"), finished=time.time())
            W["last"] = time.time()
        rc = await proc.wait()
        for q in W["reqs"].values():
            if q["status"] in ("queued", "running"):
                q.update(status="error", msg=q.get("cancel_msg") or f"Процесс озвучки завершился (код {rc}). Подробности — logs\\voice.log")
        if W["proc"] is proc:
            W["proc"] = None

    async def ensure_worker():
        if W["proc"] and W["proc"].returncode is None:
            return W["proc"]
        await free_gpu()
        TMP.mkdir(parents=True, exist_ok=True)
        cfg = {"ffmpeg": str(FF), "tmp_dir": str(TMP), "out_dir": str(OUT), "sep_dir": str(SEP), "sep_model": SEP_MODEL,
               "omnivoice_dir": str(VDIR / "omnivoice"), "whisper_dir": str(VDIR / "whisper-large-v3-turbo")}
        cf = TMP / "worker.json"
        cf.write_text(json.dumps(cfg, ensure_ascii=False), "utf-8")
        env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1", PATH=str(BIN) + os.pathsep + os.environ.get("PATH", ""),
                   HF_HOME=str(ROOT / "cache" / "huggingface"))
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        logf = open(S.LOGS / "voice.log", "w", encoding="utf-8", errors="replace")
        proc = await asyncio.create_subprocess_exec(sys.executable, str(S.APP_DIR / "voice_worker.py"), str(cf),
                                                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                    stderr=logf, cwd=str(TMP), env=env, creationflags=flags)
        W["proc"], W["last"] = proc, time.time()
        asyncio.create_task(reader(proc))
        asyncio.create_task(idle_watch(proc))
        return proc

    async def idle_watch(proc):
        while proc.returncode is None:
            await asyncio.sleep(20)
            busy = any(q["status"] in ("queued", "running") for q in W["reqs"].values())
            if not busy and time.time() - W["last"] > IDLE_UNLOAD:
                await stop_worker(graceful=True)
                break

    async def stop_worker(graceful=False, msg="Остановлено"):
        p = W["proc"]
        if not p or p.returncode is not None:
            return
        for q in W["reqs"].values():
            if q["status"] in ("queued", "running"):
                q["cancel_msg"] = msg
        try:
            if graceful:
                p.stdin.write(b'{"cmd":"quit"}\n')
                await p.stdin.drain()
                await asyncio.wait_for(p.wait(), 10)
            else:
                p.kill()
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    async def submit(cmd):
        if not installed():
            raise RuntimeError("Сначала установите голосовой модуль (кнопка вверху вкладки).")
        async with W["lock"]:
            proc = await ensure_worker()
            rid = uuid.uuid4().hex[:10]
            W["reqs"][rid] = {"cmd": cmd["cmd"], "status": "queued", "msg": "в очереди", "p": 0, "created": time.time()}
            if len(W["reqs"]) > 200:
                for k in sorted(W["reqs"], key=lambda k: W["reqs"][k]["created"])[:50]:
                    W["reqs"].pop(k, None)
            proc.stdin.write((json.dumps({**cmd, "id": rid}, ensure_ascii=False) + "\n").encode("utf-8"))
            await proc.stdin.drain()
            W["last"] = time.time()
            return rid

    def lang_of(b):
        l = (b.get("lang") or "ru").strip().lower()
        return l if re.match(r"^[a-z]{2,3}$|^auto$", l) else "ru"

    @routes.post("/api/voice/ref")
    async def v_ref(req):
        b = await req.json()
        p = in_up(b.get("name"))
        if not p:
            return err("Загрузите эталон — видео или звук с нужным голосом")
        start = max(0.0, float(b.get("start") or 0))
        end = float(b.get("end") or 0)
        if end and end - start > 20:
            end = start + 20
        try:
            rid = await submit({"cmd": "ref", "path": str(p), "start": start, "end": end or None,
                                "separate": bool(b.get("separate", True)), "lang": lang_of(b)})
        except Exception as e:
            return err(e)
        return web.json_response({"id": rid})

    @routes.post("/api/voice/asr")
    async def v_asr(req):
        b = await req.json()
        p = in_up(b.get("name"))
        if not p:
            return err("Файл ролика не найден — добавьте его заново")
        try:
            rid = await submit({"cmd": "asr", "path": str(p), "separate": bool(b.get("separate", True)), "lang": lang_of(b)})
        except Exception as e:
            return err(e)
        return web.json_response({"id": rid})

    @routes.post("/api/voice/tts")
    async def v_tts(req):
        b = await req.json()
        p = in_up(b.get("name"))
        if not p:
            return err("Файл ролика не найден — добавьте его заново")
        refs = b.get("refs") or ([b["ref"]] if b.get("ref") else [])
        good = []
        for ref in refs[:8]:
            rw = Path((ref or {}).get("wav") or "")
            ok = rw.is_file() and str(rw.resolve()).startswith(str(TMP.resolve()))
            good.append({"wav": str(rw), "text": ((ref or {}).get("text") or "").strip()[:1000]} if ok else None)
        if not good or not good[0]:
            return err("Сначала подготовьте эталон (кнопка «Взять голос»)")
        segs = []
        for s in b.get("segments") or []:
            try:
                a, e = float(s.get("start")), float(s.get("end"))
            except (TypeError, ValueError):
                continue
            t = (s.get("text") or "").strip()
            keep = bool(s.get("keep"))
            ri = int(s.get("ref") or 0)
            if not keep and t and (ri >= len(good) or not good[ri]):
                return err(f"Эталон «{s.get('ref_label') or ri + 1}» не подготовлен — откройте его и нажмите «Взять голос»")
            if (t or keep) and e - a >= 0.05:
                segs.append({"start": a, "end": e, "text": t[:600], "keep": keep, "ref": ri,
                             "gain": max(-18.0, min(18.0, float(s.get("gain") or 0)))})
        if not any(not s["keep"] for s in segs):
            return err("Нет фраз для озвучки — распознайте текст или впишите фразы (у всех стоит «оригинал»)")
        o = b.get("opts") or {}
        stem = re.sub(r"_[0-9a-f]{5}$", "", p.stem)
        info = probe(p)
        try:
            rid = await submit({"cmd": "tts", "path": str(p), "video": info["video"], "segments": segs,
                                "refs": good,
                                "lang": lang_of(b), "separate": bool(b.get("separate", True)),
                                "out_name": f"{stem}_voice_{time.strftime('%H%M%S')}",
                                "opts": {"steps": max(8, min(64, int(o.get("steps", 32)))),
                                         "guidance": max(0.5, min(5.0, float(o.get("guidance", 2.0)))),
                                         "seed": int(o.get("seed", -1)),
                                         "keep_outside": bool(o.get("keep_outside", True))}})
        except Exception as e:
            return err(e)
        return web.json_response({"id": rid})

    @routes.post("/api/voice/say")
    async def v_say(req):
        """Просто текст (без видео) → звук голосом эталона."""
        b = await req.json()
        text = (b.get("text") or "").strip()
        if not text:
            return err("Впишите текст")
        good = []
        for ref in (b.get("refs") or [])[:8]:
            rw = Path((ref or {}).get("wav") or "")
            ok = rw.is_file() and str(rw.resolve()).startswith(str(TMP.resolve()))
            good.append({"wav": str(rw), "text": ((ref or {}).get("text") or "").strip()[:1000]} if ok else None)
        ri = int(b.get("ref") or 0)
        if ri >= len(good) or not good[ri]:
            return err("Эталон не подготовлен — нажмите «Взять голос»")
        o = b.get("opts") or {}
        try:
            dur = max(0.0, min(120.0, float(b.get("duration") or 0)))
            spd = max(0.5, min(2.0, float(b.get("speed") or 1)))
        except (TypeError, ValueError):
            dur, spd = 0.0, 1.0
        try:
            rid = await submit({"cmd": "say", "text": text[:3000], "refs": good, "ref": ri, "lang": lang_of(b),
                                "duration": dur, "speed": spd,
                                "opts": {"steps": max(8, min(64, int(o.get("steps", 32)))),
                                         "guidance": max(0.5, min(5.0, float(o.get("guidance", 2.0)))),
                                         "seed": int(o.get("seed", -1)),
                                         "gain": max(-18.0, min(18.0, float(o.get("gain") or 0)))}})
        except Exception as e:
            return err(e)
        return web.json_response({"id": rid})

    @routes.get("/api/voice/req/{rid}")
    async def v_req(req):
        q = W["reqs"].get(req.match_info["rid"])
        if not q:
            return err("Задание не найдено (программа перезапускалась?)", 404)
        out = {k: v for k, v in q.items() if k not in ("cancel_msg",)}
        r = out.get("result") or {}
        for k in ("wav", "voice", "video"):
            if r.get(k):
                r[k + "_url"] = "/output/voice/" + r[k]
        return web.json_response(out)

    @routes.post("/api/voice/stop")
    async def v_stop(_):
        await stop_worker(graceful=False, msg="Остановлено")
        return web.json_response({"ok": True})

    @routes.post("/api/voice/unload")
    async def v_unload(_):
        await stop_worker(graceful=True, msg="Выгружено")
        return web.json_response({"ok": True})

    async def release():
        """Перед генерацией картинок/видео: процесс озвучки без дела — закрыть, видеопамять движку."""
        if W["proc"] and W["proc"].returncode is None and not any(q["status"] in ("queued", "running") for q in W["reqs"].values()):
            await stop_worker(graceful=True, msg="Выгружено")

    return {"state": state, "installed": installed, "stop": stop_worker, "release": release}
