"""Вкладка «Речь»: генератор голоса на Higgs TTS 3 (Boson AI, 4B, 100+ языков, включая русский).

Текст (с тегами эмоций, подачи, пауз и звуков) → речь. Голос — случайный (seed) или свой:
из куска видео/звука (клонирование, нужен эталон 5–15 с) или из удачного случайного результата.
Диалог: строки «Имя: текст» озвучиваются голосами из библиотеки с таким же именем.

Модель и процесс работы — в отдельном постоянном процессе speech_worker.py (команды через stdin).
Установка — только по кнопке. Лицензия модели — некоммерческая (Boson Higgs TTS 3 Research and
Non‑Commercial License); авторам роликов разрешено бесплатно с упоминанием Higgs Audio (Boson AI).
"""
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

from aiohttp import web

REPOS = ["bosonai/higgs-tts-3-4b", "bosonai/higgs-audio-v3-tts-4b"]   # второе имя — на случай переименования
MODEL_SIZE_GB = 9.3
MIN_TF = (5, 5)
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus"}
IDLE_UNLOAD = 900   # 15 минут без дела — процесс закрывается и отдаёт видеопамять
IMPORT_CHECK = ("import av, soundfile, safetensors, tokenizers, torchaudio;"
                "from transformers import Qwen3Model, Qwen3Config, PreTrainedTokenizerFast, "
                "HiggsAudioV2TokenizerModel, HiggsAudioV2TokenizerConfig")
TAG_RE = re.compile(r"<\|(?:[a-z_]+:[a-z_]+|reset)\|>")


def register(routes, S):
    ROOT, MODELS, INPUT, OUTPUT = S.ROOT, S.MODELS, S.INPUT, S.OUTPUT
    MDIR = MODELS / "voice" / "higgs-tts-3"
    UP = INPUT / "speech"
    VOICES = UP / "voices"
    OUT = OUTPUT / "speech"
    TMP = ROOT / "engine" / "temp" / "speech"
    MARK = MDIR / "zouwu_installed.json"
    WHISPER = MODELS / "voice" / "whisper-large-v3-turbo"
    for d in (UP, VOICES, OUT):
        d.mkdir(parents=True, exist_ok=True)

    ST = {"running": False, "step": "", "done": 0, "total": 0, "error": "", "log": []}
    W = {"proc": None, "reqs": {}, "last": 0.0, "lock": asyncio.Lock(), "cur": None}

    def err(msg, code=400):
        return web.json_response({"error": str(msg)}, status=code)

    def log(msg):
        ST["log"].append(time.strftime("%H:%M:%S ") + msg)
        ST["log"] = ST["log"][-60:]

    # ------------------------------------------------------------ установка
    def model_ok():
        if not (MDIR / "config.json").exists() or not (MDIR / "tokenizer.json").exists():
            return False
        idx = MDIR / "model.safetensors.index.json"
        if idx.exists():
            try:
                shards = set(json.loads(idx.read_text("utf-8"))["weight_map"].values())
            except Exception:
                return False
            return all((MDIR / s).exists() for s in shards)
        return (MDIR / "model.safetensors").exists()

    def installed():
        return MARK.exists() and model_ok()

    def voices():
        out = []
        for j in sorted(VOICES.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                v = json.loads(j.read_text("utf-8"))
            except Exception:
                continue
            if (VOICES / (j.stem + ".wav")).exists():
                v["id"] = j.stem
                v["url"] = f"/input/speech/voices/{j.stem}.wav?t={int((VOICES / (j.stem + '.wav')).stat().st_mtime)}"
                out.append(v)
        return out

    def state():
        return {"installed": installed(), "running": ST["running"], "step": ST["step"], "done": ST["done"],
                "total": ST["total"], "error": ST["error"], "log": ST["log"][-12:], "size_gb": MODEL_SIZE_GB,
                "worker": bool(W["proc"] and W["proc"].returncode is None), "whisper": (WHISPER / "config.json").exists(),
                "voices": voices()}

    def run_cmd(args, what):
        log(what)
        r = subprocess.run([str(a) for a in args], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-6:]
            raise RuntimeError(what + ": " + " | ".join(tail))
        return r.stdout

    def py(code):
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace")

    def tf_version():
        r = py("import transformers;print(transformers.__version__)")
        m = re.match(r"(\d+)\.(\d+)", r.stdout.strip()) if r.returncode == 0 else None
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    def http_get(url, dest: Path, label):
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        have = part.stat().st_size if part.exists() else 0
        hdr = {"User-Agent": "ZouwuLab/1.0"}
        tok = (S.CONFIG.get("hf_token") or "").strip()
        if tok:
            hdr["Authorization"] = "Bearer " + tok
        if have:
            hdr["Range"] = f"bytes={have}-"
        with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=60) as r:
            if have and r.status != 206:
                ST["done"] -= have
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

    def repo_plan():
        from huggingface_hub import HfApi
        tok = (S.CONFIG.get("hf_token") or "").strip() or None
        last = None
        for repo in REPOS:
            try:
                files = []
                for it in HfApi().list_repo_tree(repo, recursive=True, token=tok):
                    size = getattr(it, "size", None)
                    if size is None:
                        continue
                    p = it.path
                    if p.startswith((".", "assets/")) or p.endswith((".png", ".jpg", ".gif", ".wav", ".mp3", ".gitattributes")):
                        continue
                    files.append((p, int(size)))
                if files:
                    return repo, files
            except Exception as e:
                last = e
        msg = str(last or "репозиторий не найден")
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            msg = ("Hugging Face не даёт скачать модель без согласия с лицензией. Откройте huggingface.co/bosonai/higgs-tts-3-4b, "
                   "примите условия, создайте токен (Settings → Access Tokens) и вставьте его в «Настройки» программы.")
        raise RuntimeError(msg)

    def install_sync():
        ST["step"] = "Проверяю библиотеки"
        if tf_version() < MIN_TF or py(IMPORT_CHECK).returncode != 0:
            uv = ROOT / "tools" / ("uv.exe" if os.name == "nt" else "uv")
            extra = ["soundfile", "safetensors"]
            if tf_version() < MIN_TF:
                extra.append("transformers>=%d.%d" % MIN_TF)
            if uv.exists():
                frz = run_cmd([uv, "pip", "freeze", "--python", sys.executable], "Список установленных библиотек")
                TMP.mkdir(parents=True, exist_ok=True)
                pins = [l for l in frz.splitlines() if re.match(r"^[A-Za-z0-9_.\-]+==", l)]
                if tf_version() < MIN_TF:   # обновить разрешено только transformers и его прямые зависимости
                    pins = [l for l in pins if not l.lower().startswith(("transformers==", "tokenizers==", "huggingface-hub==",
                                                                          "huggingface_hub==", "safetensors=="))]
                cons = TMP / "constraints.txt"
                cons.write_text("\n".join(pins) + "\n", "utf-8")
                run_cmd([uv, "pip", "install", "--python", sys.executable, "-c", cons, *extra], "Устанавливаю библиотеки")
                if py("import torchaudio.functional").returncode != 0:   # кодеку Higgs нужен torchaudio под ту же сборку torch
                    import torch
                    m = re.search(r"\+(cu\d+|cpu)", torch.__version__)
                    idx = "https://download.pytorch.org/whl/" + (m.group(1) if m else "cpu")
                    run_cmd([uv, "pip", "install", "--python", sys.executable, "torchaudio", "--index-url", idx, "-c", cons],
                            "Устанавливаю torchaudio")
            else:
                run_cmd([sys.executable, "-m", "pip", "install", *extra], "Устанавливаю библиотеки (pip)")
        r = py(IMPORT_CHECK)
        if r.returncode != 0:
            raise RuntimeError("Библиотеки не загружаются: " + " | ".join(r.stderr.strip().splitlines()[-3:]))
        ST["step"] = "Список файлов модели"
        repo, files = repo_plan()
        todo = [(p, s) for p, s in files if not ((MDIR / p).exists() and (MDIR / p).stat().st_size == s)]
        ST["done"], ST["total"] = 0, sum(s for _, s in todo)
        for p, s in todo:   # недокачанные куски (.part) учитываем в прогрессе
            part = MDIR / (p + ".part")
            if part.exists():
                ST["done"] += part.stat().st_size
        log(f"Скачиваю {repo}: {len(todo)} файлов, {ST['total'] / 1e9:.1f} ГБ")
        for p, s in todo:
            http_get(f"https://huggingface.co/{repo}/resolve/main/{p}", MDIR / p, f"Higgs TTS 3 · {Path(p).name}")
            if (MDIR / p).stat().st_size != s:
                (MDIR / p).unlink()
                raise RuntimeError(f"Файл {p} скачался не полностью — нажмите «Установить» ещё раз")
        if not model_ok():
            raise RuntimeError("Не все файлы модели скачались — нажмите «Установить» ещё раз (докачает недостающее)")
        MARK.write_text(json.dumps({"repo": repo, "time": time.time()}), "utf-8")

    async def install_task():
        if ST["running"]:
            return
        ST.update(running=True, error="", step="Начинаю", done=0, total=0, log=[])
        try:
            await asyncio.get_running_loop().run_in_executor(None, install_sync)
            ST["step"] = "Готово"
            log("Higgs TTS 3 установлен")
        except Exception as e:
            ST["error"] = str(e)
            log("Ошибка: " + str(e))
        finally:
            ST["running"] = False

    @routes.get("/api/speech/state")
    async def s_state(_):
        return web.json_response(state())

    @routes.post("/api/speech/install")
    async def s_install(_):
        asyncio.create_task(install_task())
        await asyncio.sleep(0.2)
        return web.json_response(state())

    # ------------------------------------------------------------ процесс синтеза
    async def free_gpu():
        try:
            from aiohttp import ClientSession
            async with ClientSession() as s:
                async with s.post(S.ENGINE_.url + "/free", json={"unload_models": True, "free_memory": True}) as r:
                    await r.read()
        except Exception:
            pass
        for fn in (lambda: S.LLM["stop_gpu"](), lambda: S.VOICE["release"]()):
            try:
                await fn()
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
                if q.get("on_done"):
                    try:
                        q["on_done"](e)
                    except Exception as ex:
                        log("Ошибка сохранения: " + str(ex))
            elif ev == "error":
                q.update(status="error", msg=e.get("msg", "Ошибка"), finished=time.time())
            W["last"] = time.time()
        rc = await proc.wait()
        for q in W["reqs"].values():
            if q["status"] in ("queued", "running"):
                q.update(status="error", msg=q.get("cancel_msg") or f"Процесс синтеза завершился (код {rc}). Подробности — logs\\speech.log")
        if W["proc"] is proc:
            W["proc"] = None
            W["warm"] = False

    async def ensure_worker():
        if W["proc"] and W["proc"].returncode is None:
            return W["proc"]
        await free_gpu()
        TMP.mkdir(parents=True, exist_ok=True)
        cfg = {"tmp_dir": str(TMP), "out_dir": str(OUT), "model_dir": str(MDIR),
               "codec_cfg": str(S.APP_DIR / "higgs_codec_config.json"),
               "codec_alt_dir": str(MODELS / "voice" / "omnivoice" / "audio_tokenizer"),
               "whisper_dir": str(WHISPER)}
        cf = TMP / "worker.json"
        cf.write_text(json.dumps(cfg, ensure_ascii=False), "utf-8")
        env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1", HF_HOME=str(ROOT / "cache" / "huggingface"),
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        logf = open(S.LOGS / "speech.log", "w", encoding="utf-8", errors="replace")
        proc = await asyncio.create_subprocess_exec(sys.executable, str(S.APP_DIR / "speech_worker.py"), str(cf),
                                                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                    stderr=logf, cwd=str(TMP), env=env, creationflags=flags)
        W["proc"], W["last"] = proc, time.time()
        asyncio.create_task(reader(proc))
        asyncio.create_task(idle_watch(proc))
        return proc

    def busy():
        return any(q["status"] in ("queued", "running") for q in W["reqs"].values())

    async def idle_watch(proc):
        while proc.returncode is None:
            await asyncio.sleep(20)
            if not busy() and time.time() - W["last"] > IDLE_UNLOAD:
                await stop_worker(graceful=True)
                break

    async def send(obj):
        p = W["proc"]
        if p and p.returncode is None:
            p.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            await p.stdin.drain()

    async def stop_worker(graceful=False, msg="Остановлено"):
        p = W["proc"]
        if not p or p.returncode is not None:
            return
        for q in W["reqs"].values():
            if q["status"] in ("queued", "running"):
                q["cancel_msg"] = msg
        try:
            if graceful:
                await send({"cmd": "quit"})
                await asyncio.wait_for(p.wait(), 10)
            else:
                p.kill()
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    async def submit(cmd, on_done=None, need_model=True):
        if need_model and not installed():
            raise RuntimeError("Сначала установите Higgs TTS 3 (кнопка вверху вкладки).")
        async with W["lock"]:
            await ensure_worker()
            rid = uuid.uuid4().hex[:10]
            W["reqs"][rid] = {"cmd": cmd["cmd"], "status": "queued", "msg": "в очереди", "p": 0, "created": time.time(),
                              "on_done": on_done}
            if len(W["reqs"]) > 200:
                for k in sorted(W["reqs"], key=lambda k: W["reqs"][k]["created"])[:50]:
                    W["reqs"].pop(k, None)
            await send({**cmd, "id": rid})
            W["last"] = time.time()
            return rid

    # ------------------------------------------------------------ файлы и голоса
    def probe(path: Path):
        info = {"duration": 0.0, "video": path.suffix.lower() in VIDEO_EXT}
        try:
            import av
            with av.open(str(path)) as c:
                info["duration"] = round(float(c.duration) / 1e6, 2) if c.duration else 0.0
                info["video"] = any(s.type == "video" for s in c.streams)
        except Exception:
            pass
        return info

    def in_dir(base: Path, name):
        p = (base / (name or "")).resolve()
        return p if name and str(p).startswith(str(base.resolve())) and p.is_file() else None

    @routes.post("/api/speech/upload")
    async def s_upload(req):
        rd = await req.multipart()
        out = []
        async for part in rd:
            if not part.filename:
                continue
            from urllib.parse import unquote
            fname = unquote(part.filename)
            ext = Path(fname).suffix.lower()
            if ext not in VIDEO_EXT | AUDIO_EXT:
                continue
            stem = re.sub(r"[^\w\-. ]+", "_", Path(fname).stem)[:60].strip() or "voice"
            name = f"{stem}_{uuid.uuid4().hex[:5]}{ext}"
            with open(UP / name, "wb") as f:
                while True:
                    ch = await part.read_chunk(1 << 20)
                    if not ch:
                        break
                    f.write(ch)
            out.append({"name": name, "title": fname, "url": "/input/speech/" + name, **probe(UP / name)})
        if not out:
            return err("Нужен звук (wav, mp3, m4a, flac…) или видео (mp4, mov, mkv, webm)")
        return web.json_response({"files": out})

    @routes.post("/api/speech/from_output")
    async def s_from_output(req):
        """Файл из output (видео из вкладки «Видео», ролик из «Голоса») → источник голоса."""
        import shutil
        from urllib.parse import unquote
        b = await req.json()
        rel = (b.get("url") or "").split("?")[0]
        if not rel.startswith("/output/"):
            return err("Нужна ссылка на файл из output")
        src = (OUTPUT / unquote(rel[len("/output/"):])).resolve()
        if not str(src).startswith(str(OUTPUT.resolve())) or not src.is_file():
            return err("Файл не найден")
        name = f"{src.stem[:60]}_{uuid.uuid4().hex[:5]}{src.suffix.lower()}"
        shutil.copy(src, UP / name)
        return web.json_response({"files": [{"name": name, "title": src.name, "url": "/input/speech/" + name, **probe(UP / name)}]})

    def clean_label(s, fallback="Голос"):
        s = re.sub(r"[\[\]:\n\r\t<>|]+", " ", (s or "")).strip()[:40]
        return s or fallback

    def unique_label(label, skip=None):
        names = {v["name"].lower() for v in voices() if v.get("id") != skip}
        base, k = label, 2
        while label.lower() in names:
            label, k = f"{base} {k}", k + 1
        return label

    def save_meta(vid, **kw):
        j = VOICES / (vid + ".json")
        meta = json.loads(j.read_text("utf-8")) if j.exists() else {}
        meta.update(kw)
        j.write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
        return meta

    @routes.post("/api/speech/voice_add")
    async def s_voice_add(req):
        """Кусок файла → голос в библиотеке (эталон 24 кГц + его текст)."""
        b = await req.json()
        p = in_dir(UP, b.get("name"))
        if not p:
            return err("Загрузите файл с голосом")
        try:
            start = max(0.0, float(b.get("start") or 0))
            end = float(b.get("end") or 0)
        except (TypeError, ValueError):
            start, end = 0.0, 0.0
        if end and end - start > 30:
            end = start + 30
        if not end:
            end = start + 20
        label = unique_label(clean_label(b.get("label")))
        vid = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:4]
        text = (b.get("text") or "").strip()[:1500]

        def done(res):
            save_meta(vid, name=label, text=res.get("text") or text, duration=res.get("duration"),
                      source=p.name, start=start, end=end, created=time.time())
        try:
            rid = await submit({"cmd": "ref", "path": str(p), "start": start, "end": end, "dest": str(VOICES / (vid + ".wav")),
                                "text": text, "transcribe": bool(b.get("transcribe", True)), "lang": (b.get("lang") or "auto")},
                               on_done=done, need_model=False)
        except Exception as e:
            return err(e)
        return web.json_response({"id": rid, "voice": vid})

    @routes.post("/api/speech/voice_update")
    async def s_voice_update(req):
        b = await req.json()
        vid = re.sub(r"[^\w\-]", "", b.get("id") or "")
        if not vid or not (VOICES / (vid + ".json")).exists():
            return err("Голос не найден")
        kw = {}
        if "name" in b:
            kw["name"] = unique_label(clean_label(b.get("name")), skip=vid)
        if "text" in b:
            kw["text"] = (b.get("text") or "").strip()[:1500]
        if "note" in b:
            kw["note"] = (b.get("note") or "").strip()[:200]
        save_meta(vid, **kw)
        return web.json_response({"ok": True, "voices": voices()})

    @routes.post("/api/speech/voice_del")
    async def s_voice_del(req):
        b = await req.json()
        vid = re.sub(r"[^\w\-]", "", b.get("id") or "")
        for ext in (".wav", ".json"):
            f = VOICES / (vid + ext)
            if vid and f.exists():
                f.unlink()
        return web.json_response({"ok": True, "voices": voices()})

    @routes.post("/api/speech/voice_from_result")
    async def s_voice_from_result(req):
        """Удачный случайный голос из результата → в библиотеку."""
        import shutil
        b = await req.json()
        meta = in_dir(OUT, (b.get("file") or "").replace(".wav", ".json"))
        if not meta:
            return err("Результат не найден")
        m = json.loads(meta.read_text("utf-8"))
        ref = in_dir(OUT, m.get("ref_wav") or "")
        if not ref:
            return err("У этого результата нет образца голоса (он озвучен голосом из библиотеки)")
        label = unique_label(clean_label(b.get("name"), "Голос seed " + str(m.get("seed", ""))))
        vid = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:4]
        shutil.copy(ref, VOICES / (vid + ".wav"))
        save_meta(vid, name=label, text=m.get("ref_text") or "", seed=m.get("seed"), source=m.get("wav"), created=time.time())
        return web.json_response({"ok": True, "voices": voices(), "id": vid})

    # ------------------------------------------------------------ синтез
    def voice_obj(vid):
        vid = re.sub(r"[^\w\-]", "", vid or "")
        j, w = VOICES / (vid + ".json"), VOICES / (vid + ".wav")
        if not vid or not j.exists() or not w.exists():
            return None
        m = json.loads(j.read_text("utf-8"))
        return {"wav": str(w), "text": m.get("text") or "", "name": m.get("name") or vid, "id": vid}

    def parse_dialog(text, default):
        """«Имя: реплика» / «[Имя] реплика» → куски с голосами из библиотеки; прочие строки — голосом по умолчанию."""
        lib = {v["name"].lower(): v["id"] for v in voices()}
        rx_br = re.compile(r"^\s*\[([^\]\n]{1,40})\]\s*:?\s*(.*)$")          # [Имя] реплика — всегда говорящий
        rx_col = re.compile(r"^\s*([^:\n<>\[\]|]{1,40}?)\s*:\s*(.*)$")       # Имя: реплика — если такой голос есть
        segs, cur_voice, cur_spk, buf = [], default, "", []
        speakers = set()

        def flush():
            if any(x.strip() for x in buf):
                segs.append({"text": "\n".join(buf).strip(), "voice": cur_voice, "speaker": cur_spk})
            buf.clear()
        for line in text.splitlines():
            name, rest = None, ""
            m = rx_br.match(line)
            if m:
                name, rest = m.group(1).strip(), m.group(2)
            else:
                m = rx_col.match(line)
                if m and m.group(1).strip().lower() in lib:
                    name, rest = m.group(1).strip(), m.group(2)
            if not name:
                buf.append(line)
                continue
            flush()
            cur_spk = name
            speakers.add(name)
            vid = lib.get(name.lower())
            cur_voice = voice_obj(vid) if vid else {"rand_key": "spk:" + name.lower()}
            buf.append(rest.strip())
        flush()
        return segs, speakers

    @routes.post("/api/speech/say")
    async def s_say(req):
        b = await req.json()
        text = (b.get("text") or "").strip()
        if not TAG_RE.sub("", text).strip():
            return err("Впишите текст")
        if len(text) > 20000:
            return err("Слишком длинный текст (больше 20 000 символов) — разбейте на части")
        vid = b.get("voice") or ""
        default = voice_obj(vid) if vid else {"rand_key": "rand"}
        if vid and not default:
            return err("Выбранный голос не найден — выберите другой")
        segs, speakers = parse_dialog(text, default) if b.get("dialog", True) else ([{"text": text, "voice": default}], set())
        o = b.get("opts") or {}
        try:
            opts = {"temperature": max(0.0, min(1.5, float(o.get("temperature", 0.8)))),
                    "top_p": max(0.1, min(1.0, float(o.get("top_p", 0.8)))),
                    "top_k": max(0, min(1026, int(o.get("top_k", 30)))),
                    "seed": int(o.get("seed", -1)),
                    "gap": max(0.0, min(3.0, float(o.get("gap", 0.25)))),
                    "para_gap": max(0.0, min(5.0, float(o.get("para_gap", 0.6)))),
                    "gain": max(-18.0, min(18.0, float(o.get("gain") or 0)))}
        except (TypeError, ValueError):
            return err("Неверные параметры")
        clean = TAG_RE.sub(" ", text)
        first = re.sub(r"^\s*(?:\[[^\]\n]{1,40}\]\s*:?|[^:\n]{1,40}:)\s*", "", clean.strip().splitlines()[0])
        name = re.sub(r"[^\w\- ]+", "", first)
        name = re.sub(r"\s+", "_", name.strip())[:28] or "speech"
        stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{name}"
        voice_name = default.get("name") if default and default.get("name") else "случайный"

        def done(res):
            meta = {**res, "text": text, "voice": vid, "voice_name": voice_name, "speakers": sorted(speakers),
                    "opts": opts, "time": time.strftime("%d.%m %H:%M"), "created": time.time()}
            (OUT / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
        try:
            rid = await submit({"cmd": "say", "segments": segs, "opts": opts, "out_name": stem}, on_done=done)
        except Exception as e:
            return err(e)
        return web.json_response({"id": rid, "speakers": sorted(speakers)})

    @routes.post("/api/speech/autotag")
    async def s_autotag(req):
        """Встроенный помощник расставляет теги эмоций, пауз и звуков (как «Enhance» у ElevenLabs)."""
        import speech_tags
        b = await req.json()
        text = (b.get("text") or "").strip()
        if not TAG_RE.sub("", text).strip():
            return err("Впишите текст")
        if len(text) > 12000:
            return err("Слишком длинный текст для помощника (больше 12 000 символов) — размечайте частями")
        try:
            if not S.LLM["available"]():
                return err("Помощник не установлен: Модели → Помощник (LLM) → скачайте модель.")
        except Exception:
            pass
        if busy():
            return err("Дождитесь окончания озвучки")
        await release()   # видеопамять — помощнику
        level = b.get("level") if b.get("level") in ("light", "medium", "strong") else "medium"

        async def chat(msgs, max_tokens, temperature, hold=False):
            return await S.LLM["chat"](msgs, max_tokens=max_tokens, temperature=temperature, hold=hold)
        try:
            out, n = await speech_tags.auto_tag(text, chat, level)
        except Exception as e:
            return err(e)
        if not n:
            return err("Помощник не предложил ни одного тега — попробуйте ещё раз или уровень «ярко».")
        return web.json_response({"text": out, "tags": n})

    @routes.get("/api/speech/req/{rid}")
    async def s_req(req):
        q = W["reqs"].get(req.match_info["rid"])
        if not q:
            return err("Задание не найдено (программа перезапускалась?)", 404)
        out = {k: v for k, v in q.items() if k not in ("cancel_msg", "on_done")}
        if out.get("status") == "queued":   # пока модель грузится заранее — показываем её прогресс
            w = next((x for x in W["reqs"].values() if x["cmd"] == "warm" and x["status"] == "running"), None)
            if w:
                out["msg"], out["p"] = w.get("msg", ""), w.get("p", 0)
        r = dict(out.get("result") or {})
        for k in ("wav", "ref_wav"):
            if r.get(k) and not str(r[k]).startswith(("/", "\\")) and ":" not in str(r[k]):
                r[k + "_url"] = "/output/speech/" + r[k]
        if out.get("cmd") == "ref" and out.get("status") == "done":
            out["voices"] = voices()
        out["result"] = r
        return web.json_response(out)

    @routes.get("/api/speech/list")
    async def s_list(_):
        items = []
        for j in sorted(OUT.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:80]:
            try:
                m = json.loads(j.read_text("utf-8"))
            except Exception:
                continue
            if not (OUT / (m.get("wav") or "")).exists():
                continue
            m["url"] = "/output/speech/" + m["wav"]
            m["has_ref"] = bool(m.get("ref_wav") and (OUT / m["ref_wav"]).exists())
            items.append(m)
        return web.json_response({"items": items})

    @routes.post("/api/speech/del_result")
    async def s_del_result(req):
        b = await req.json()
        meta = in_dir(OUT, (b.get("file") or "").replace(".wav", ".json"))
        if meta:
            m = json.loads(meta.read_text("utf-8"))
            for k in ("wav", "ref_wav"):
                f = in_dir(OUT, m.get(k) or "")
                if f:
                    f.unlink()
            meta.unlink()
        return web.json_response({"ok": True})

    @routes.post("/api/speech/warm")
    async def s_warm(_):
        """Вкладка открыта — заранее грузим модель, чтобы первая озвучка не ждала загрузки."""
        if not installed() or busy() or ST["running"]:
            return web.json_response({"ok": False})
        if W.get("warm") and W["proc"] and W["proc"].returncode is None:
            return web.json_response({"ok": True, "already": True})
        try:
            await submit({"cmd": "warm"})
            W["warm"] = True
        except Exception as e:
            return err(e)
        return web.json_response({"ok": True})

    @routes.post("/api/speech/cancel")
    async def s_cancel(_):
        (TMP / "cancel.flag").touch()   # процесс синтеза проверяет этот файл во время генерации
        # во время загрузки модели отмена не срабатывает — через 3 с просто закрываем процесс
        await asyncio.sleep(3)
        if any(q["status"] == "running" and str(q.get("msg", "")).startswith(("Загружаю", "Жду")) for q in W["reqs"].values()):
            await stop_worker(graceful=False, msg="Остановлено")
        return web.json_response({"ok": True})

    @routes.post("/api/speech/unload")
    async def s_unload(_):
        await stop_worker(graceful=True, msg="Выгружено")
        return web.json_response({"ok": True})

    async def release():
        """Перед генерацией картинок/видео/озвучкой: процесс синтеза без дела — закрыть, видеопамять другим."""
        if W["proc"] and W["proc"].returncode is None and not busy():
            await stop_worker(graceful=True, msg="Выгружено")

    return {"state": state, "installed": installed, "stop": stop_worker, "release": release}
