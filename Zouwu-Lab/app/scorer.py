"""Автооценка картинок — вторая «пара глаз» рядом с вашими звёздами.

Две независимые лёгкие модели (обе грузятся через transformers, который уже стоит вместе с движком):
  • PickScore (yuvalkirstain/PickScore_v1, CLIP ViT-H) — какую картинку люди скорее выберут для ЭТОГО промпта
    (учитывает и соответствие промпту, и общее качество). Обычные значения ~17–24, больше — лучше.
  • Эстетика LAION (CLIP ViT-L/14 + маленькая голова improved-aesthetic-predictor, Apache-2.0) — насколько
    картинка красива сама по себе, без учёта промпта. Шкала ~1–10.
По умолчанию работают на процессоре — не занимают видеопамять и не мешают генерации.
Оценки хранятся в journal/journal.db (таблица scores) — их можно сравнивать с вашими.
"""
import asyncio
import sqlite3
import threading
import time
from pathlib import Path
from urllib.parse import unquote

from aiohttp import web

HF = "https://huggingface.co"
MODELS = {
    "pick": {"title": "PickScore — выбор людей для промпта", "repo": "yuvalkirstain/PickScore_v1", "dir": "pickscore",
             "files": ["config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                       "special_tokens_map.json", "vocab.json", "merges.txt"], "size": "3.9 ГБ"},
    "aes": {"title": "Эстетика LAION — красота картинки, 1–10", "repo": "openai/clip-vit-large-patch14", "dir": "clip-vit-large-patch14",
            "files": ["config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                      "special_tokens_map.json", "vocab.json", "merges.txt"], "size": "1.7 ГБ"},
}
IDLE_UNLOAD = 600  # выгрузить модели из памяти после 10 минут без работы


def register(routes, S):
    BASE = S.MODELS / "scorers"
    BASE.mkdir(parents=True, exist_ok=True)
    MLP = S.APP_DIR / "scorers" / "laion_aesthetic_mlp.safetensors"
    DB = S.ROOT / "journal" / "journal.db"
    st = {"loaded": {}, "lock": threading.Lock(), "last_use": 0, "queue": [], "running": False, "done": 0, "error": ""}

    def db():
        c = sqlite3.connect(DB, timeout=20)
        c.row_factory = sqlite3.Row
        return c

    (S.ROOT / "journal").mkdir(exist_ok=True)
    with db() as c:
        c.execute("CREATE TABLE IF NOT EXISTS scores(image TEXT PRIMARY KEY, pick REAL, aes REAL, prompt TEXT, time TEXT)")

    def cfg(k, d=None):
        v = S.CONFIG.get(k)
        return d if v in (None, "") else v

    def mdir(key):
        return BASE / MODELS[key]["dir"]

    def installed(key):
        return all((mdir(key) / f).exists() for f in MODELS[key]["files"])

    def enabled_keys():
        return [k for k in MODELS if installed(k) and cfg(f"score_{k}", True)]

    # ------------------------------------------------ модели
    def device():
        want = cfg("score_device", "cpu")
        if want == "gpu":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                pass
        return "cpu"

    def load(key):
        if key in st["loaded"]:
            return st["loaded"][key]
        import torch
        from transformers import CLIPModel, CLIPProcessor
        dev = device()
        dtype = torch.float16 if dev == "cuda" else torch.float32
        model = CLIPModel.from_pretrained(str(mdir(key)), torch_dtype=dtype).to(dev).eval()
        proc = CLIPProcessor.from_pretrained(str(mdir(key)))
        extra = None
        if key == "aes":
            from safetensors.torch import load_file
            sd = load_file(str(MLP))
            head = torch.nn.Sequential(torch.nn.Linear(768, 1024), torch.nn.Dropout(0.2), torch.nn.Linear(1024, 128),
                                       torch.nn.Dropout(0.2), torch.nn.Linear(128, 64), torch.nn.Dropout(0.1),
                                       torch.nn.Linear(64, 16), torch.nn.Linear(16, 1))
            head.load_state_dict({k.replace("layers.", ""): v for k, v in sd.items()})
            extra = head.to(dev).eval().float()
        st["loaded"][key] = (model, proc, extra, dev, dtype)
        return st["loaded"][key]

    def unload():
        with st["lock"]:
            st["loaded"].clear()
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _features(out):
        # transformers 4.x возвращает тензор, 5.x может вернуть объект с pooler_output
        if hasattr(out, "norm"):
            return out
        for a in ("image_embeds", "text_embeds", "pooler_output"):
            v = getattr(out, a, None)
            if v is not None:
                return v
        return out[0]

    def score_image(path: Path, prompt: str):
        """{'pick': float|None, 'aes': float|None} — считает включённые модели."""
        import torch
        from PIL import Image, ImageOps
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        res = {}
        with st["lock"]:
            for key in enabled_keys():
                model, proc, head, dev, dtype = load(key)
                with torch.no_grad():
                    px = proc(images=img, return_tensors="pt")["pixel_values"].to(dev, dtype)
                    im = _features(model.get_image_features(pixel_values=px)).float()
                    im = im / im.norm(dim=-1, keepdim=True)
                    if key == "aes":
                        res["aes"] = round(float(head(im)[0, 0]), 3)
                    else:
                        t = proc(text=[prompt or ""], padding=True, truncation=True, max_length=77, return_tensors="pt")
                        tx = _features(model.get_text_features(input_ids=t["input_ids"].to(dev),
                                                               attention_mask=t["attention_mask"].to(dev))).float()
                        tx = tx / tx.norm(dim=-1, keepdim=True)
                        res["pick"] = round(float(model.logit_scale.exp().float() * (im @ tx.T)[0, 0]), 3)
            st["last_use"] = time.time()
        return res

    # ------------------------------------------------ очередь
    def save(image, prompt, r):
        with db() as c:
            old = c.execute("SELECT pick, aes FROM scores WHERE image=?", (image,)).fetchone()
            pick = r.get("pick", old["pick"] if old else None)
            aes = r.get("aes", old["aes"] if old else None)
            c.execute("INSERT OR REPLACE INTO scores(image,pick,aes,prompt,time) VALUES(?,?,?,?,?)",
                      (image, pick, aes, (prompt or "")[:2000], time.strftime("%Y-%m-%d %H:%M:%S")))

    def resolve(image):
        """URL картинки → путь к файлу (для чужих картинок из журнала — миниатюра)."""
        if image.startswith("/output/"):
            p = (S.OUTPUT / unquote(image[len("/output/"):])).resolve()
            if str(p).startswith(str(S.OUTPUT.resolve())) and p.is_file():
                return p
        with db() as c:
            r = c.execute("SELECT thumb FROM entries WHERE image=?", (image,)).fetchone() if \
                c.execute("SELECT name FROM sqlite_master WHERE name='entries'").fetchone() else None
        if r and r["thumb"] and (S.ROOT / "journal" / "thumbs" / r["thumb"]).exists():
            return S.ROOT / "journal" / "thumbs" / r["thumb"]
        return None

    def enqueue(image, prompt, force=False):
        if not enabled_keys() or (not force and not cfg("score_auto", True)):
            return
        if any(q[0] == image for q in st["queue"]):
            return
        st["queue"].append((image, prompt))
        if not st["running"]:
            try:
                asyncio.get_running_loop().create_task(worker())
            except RuntimeError:
                pass

    def engine_busy():
        return any(j["status"] in ("queued", "running", "finishing") for j in S.ENGINE_.jobs.values())

    async def worker():
        if st["running"]:
            return
        st["running"] = True
        loop = asyncio.get_running_loop()
        try:
            while st["queue"]:
                if device() == "cuda":
                    while engine_busy():   # на видеокарте — только между генерациями
                        await asyncio.sleep(2)
                image, prompt = st["queue"].pop(0)
                path = resolve(image)
                if not path:
                    continue
                try:
                    r = await loop.run_in_executor(None, score_image, path, prompt)
                    if r:
                        save(image, prompt, r)
                        st["done"] += 1
                    st["error"] = ""
                except Exception as e:
                    st["error"] = f"{type(e).__name__}: {e}"[:300]
        finally:
            st["running"] = False

    async def idle_loop():
        while True:
            await asyncio.sleep(60)
            if st["loaded"] and not st["running"] and time.time() - st["last_use"] > IDLE_UNLOAD:
                unload()

    def on_job_done(job):
        """После каждой генерации/правки — оценить новые картинки."""
        p = job.get("params") or {}
        prompt = p.get("prompt_user") or p.get("prompt") or p.get("user_instruction") or ""
        for url in job.get("images") or []:
            enqueue(url.split("?")[0], prompt)

    def scores_for(image):
        with db() as c:
            r = c.execute("SELECT pick, aes FROM scores WHERE image=?", ((image or "").split("?")[0],)).fetchone()
            if not r:
                return None
            out = {"pick": r["pick"], "aes": r["aes"]}
            for k in ("pick", "aes"):  # место среди всех оценённых картинок (процентиль)
                if out[k] is not None:
                    n = c.execute(f"SELECT COUNT(*) FROM scores WHERE {k} IS NOT NULL").fetchone()[0]
                    lower = c.execute(f"SELECT COUNT(*) FROM scores WHERE {k} < ?", (out[k],)).fetchone()[0]
                    out[k + "_pct"] = round(100 * lower / max(1, n - 1)) if n > 1 else None
            return out

    def agreement(family=None):
        """Насколько автооценки согласны с вашими звёздами (ранговая корреляция Спирмена)."""
        with db() as c:
            if not c.execute("SELECT name FROM sqlite_master WHERE name='entries'").fetchone():
                return {}
            q = ("SELECT e.rating, s.pick, s.aes FROM entries e JOIN scores s ON s.image=e.image "
                 "WHERE e.rating IS NOT NULL AND COALESCE(e.excluded,0)=0" + (" AND e.family=?" if family and family != "all" else ""))
            rows = c.execute(q, (family,) if family and family != "all" else ()).fetchall()

        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            i = 0
            while i < len(v):
                j = i
                while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                    j += 1
                for k in range(i, j + 1):
                    r[order[k]] = (i + j) / 2
                i = j + 1
            return r

        def spearman(a, b):
            if len(a) < 5:
                return None
            ra, rb = rank(a), rank(b)
            ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
            cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
            va = sum((x - ma) ** 2 for x in ra) ** 0.5
            vb = sum((y - mb) ** 2 for y in rb) ** 0.5
            return round(cov / (va * vb), 2) if va and vb else None

        out = {}
        for k in ("pick", "aes"):
            pairs = [(r["rating"], r[k]) for r in rows if r[k] is not None]
            out[k] = {"n": len(pairs), "rho": spearman([p[0] for p in pairs], [p[1] for p in pairs]),
                      "by_rating": {str(s): round(sum(p[1] for p in pairs if p[0] == s) / max(1, sum(1 for p in pairs if p[0] == s)), 2)
                                    for s in range(1, 6) if any(p[0] == s for p in pairs)}}
        return out

    # ------------------------------------------------ HTTP
    def err(m, code=400):
        return web.json_response({"error": str(m)}, status=code)

    def state():
        dl = {}
        for k in MODELS:
            ts = [t for t in S.TRANSFERS.tasks.values() if t.get("folder", "").startswith(f"scorers/{MODELS[k]['dir']}")
                  and t["status"] in ("queued", "downloading")]
            if ts:
                dl[k] = {"done": sum(t["done"] for t in ts), "total": sum(t["total"] for t in ts)}
        return {"models": {k: {"title": v["title"], "size": v["size"], "installed": installed(k), "enabled": bool(cfg(f"score_{k}", True)),
                               "downloading": dl.get(k)} for k, v in MODELS.items()},
                "device": cfg("score_device", "cpu"), "auto": bool(cfg("score_auto", True)), "loaded": list(st["loaded"]),
                "queue": len(st["queue"]), "running": st["running"], "done": st["done"], "error": st["error"]}

    @routes.get("/api/score/state")
    async def score_state(req):
        s = state()
        s["agreement"] = agreement(req.query.get("family"))
        return web.json_response(s)

    @routes.post("/api/score/install")
    async def score_install(req):
        b = await req.json()
        key = b.get("model")
        if key not in MODELS:
            return err("Неизвестная модель")
        started = []
        for f in MODELS[key]["files"]:
            if (mdir(key) / f).exists():
                continue
            mdir(key).mkdir(parents=True, exist_ok=True)
            url = f"{HF}/{MODELS[key]['repo']}/resolve/main/{f}"
            task = S.TRANSFERS.new("download", f"scorers/{MODELS[key]['dir']}", f, url)
            task["title"] = f"Автооценка: {MODELS[key]['title']} — {f}"
            asyncio.create_task(S.TRANSFERS.download(task, url))
            started.append(f)
        return web.json_response({"ok": True, "files": started})

    @routes.post("/api/score/config")
    async def score_config(req):
        b = await req.json()
        for k in ("score_pick", "score_aes", "score_auto"):
            if k in b:
                S.CONFIG[k] = bool(b[k])
        if b.get("score_device") in ("cpu", "gpu"):
            if b["score_device"] != cfg("score_device", "cpu"):
                unload()
            S.CONFIG["score_device"] = b["score_device"]
        S.save_config(S.CONFIG)
        return web.json_response(state())

    @routes.get("/api/score")
    async def score_get(req):
        return web.json_response({"scores": scores_for(req.query.get("image")), "pending": any(
            q[0] == (req.query.get("image") or "").split("?")[0] for q in st["queue"])})

    @routes.post("/api/score/now")
    async def score_now(req):
        """Оценить одну картинку сейчас (например, открытую из галереи)."""
        b = await req.json()
        image = (b.get("image") or "").split("?")[0]
        if not enabled_keys():
            return err("Автооценка не установлена")
        st["queue"] = [q for q in st["queue"] if q[0] != image]
        st["queue"].insert(0, (image, b.get("prompt") or ""))
        if not st["running"]:
            asyncio.create_task(worker())
        return web.json_response({"ok": True})

    @routes.post("/api/score/backfill")
    async def score_backfill(_):
        """Оценить все картинки из журнала, у которых ещё нет автооценки."""
        if not enabled_keys():
            return err("Сначала скачайте модель автооценки")
        import json as _json
        with db() as c:
            rows = c.execute("SELECT e.image, e.params FROM entries e LEFT JOIN scores s ON s.image=e.image "
                             "WHERE s.image IS NULL OR (s.pick IS NULL AND ?) OR (s.aes IS NULL AND ?)",
                             (installed("pick") and bool(cfg("score_pick", True)), installed("aes") and bool(cfg("score_aes", True)))).fetchall()
        n = 0
        for r in rows:
            try:
                p = _json.loads(r["params"] or "{}")
            except Exception:
                p = {}
            if not any(q[0] == r["image"] for q in st["queue"]):
                st["queue"].append((r["image"], p.get("prompt_user") or p.get("prompt") or p.get("user_instruction") or ""))
                n += 1
        if n and not st["running"]:
            asyncio.create_task(worker())
        return web.json_response({"queued": n})

    return {"on_job_done": on_job_done, "scores_for": scores_for, "agreement": agreement, "idle_loop": idle_loop,
            "enqueue": enqueue, "unload": unload}
