"""Мастерская: журнал оценок генераций, разговор с помощником о результатах и свод правил.

Всё хранится в папке journal/ (её не трогают обновления программы):
    journal/journal.db   — база SQLite: оценки с полными параметрами, правила, разговоры
    journal/thumbs/      — уменьшенные копии оценённых картинок (остаются, даже если картинку удалили из output)
    journal/reports/     — сохранённые отчёты (база при этом НЕ очищается)
    journal/backup/      — ежедневные копии базы (последние 14)
Параметры генерации записываются автоматически — в комментарии их писать не нужно.
"""
import asyncio
import base64
import hashlib
import io
import json
import socket
import zipfile
import re
import shutil
import sqlite3
import statistics
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, quote

from aiohttp import web
from PIL import Image

TAGS_GOOD = ["отлично", "лицо ок", "сходство ок", "стиль ок", "LoRA в самый раз"]
TAGS_BAD = ["лицо поплыло", "не похож персонаж", "руки / пальцы", "анатомия", "пережжено / пересвет", "мыло / размыто",
            "артефакты", "не тот стиль", "не по промпту", "LoRA не сработала", "LoRA пережала", "композиция", "цвет / кожа"]

LAB_RULES = """Ты — ассистент-исследователь в программе Zouwu Lab (генерация картинок нейросетями {fam}).
Пользователь тестирует настройки, оценивает результаты и обсуждает их с тобой, чтобы накопить свод правил:
что работает хорошо, а что плохо. Позже этот свод станет системным промптом для автоматической генерации.

Что ты получаешь автоматически (пользователь это не пишет): текущие настройки программы, параметры обсуждаемой
картинки (модель, LoRA и их сила, промпт, CFG, шаги, разрешение, семплер, seed), его оценку 1–5, метки и
комментарий, краткие выжимки LoRA, принятые правила и сводку по журналу оценок.{vision}

Как отвечать:
- по-русски, коротко и конкретно, ссылаясь на реальные параметры (сила LoRA, CFG, шаги, разрешение, формулировки промпта);
- объясни вероятную причину проблемы и предложи 1–3 конкретных изменения для следующей попытки;
- когда появляется вывод, который стоит проверить или запомнить, запиши его ОТДЕЛЬНОЙ строкой в виде
  «ПРАВИЛО: <коротко, с конкретными значениями и условием, когда применять>»;
- предлагай правило только если оно следует из оценок/комментариев пользователя или из статистики; если данных мало —
  допиши «(гипотеза)»; не повторяй уже принятые правила;
- если пользователь сам формулирует вывод («запиши, что…») — оформи его строкой «ПРАВИЛО: …».
Только взрослые персонажи (18+); с несовершеннолетними ничего не обсуждай."""

PATTERN_RULES = """Ты анализируешь журнал оценок генераций пользователя (оценки 1–5) в программе Zouwu Lab.
Тебе дана статистика: средняя оценка по группам параметров и отклонение от общей средней, частота меток,
уже принятые правила. Найди закономерности, которые подтверждаются числами (учитывай размер групп: при n < 5
это только гипотеза). Ответ по-русски: 3–8 коротких наблюдений, затем правила — каждое отдельной строкой
«ПРАВИЛО: <конкретно, с числами и условием>»; для слабых данных добавляй «(гипотеза)». Не повторяй принятые правила."""


def register(routes, S, LLM, CIV):
    J = S.ROOT / "journal"
    THUMBS, REPORTS, BACKUP, ARCHIVE = J / "thumbs", J / "reports", J / "backup", J / "archive"
    for d in (J, THUMBS, REPORTS, BACKUP, ARCHIVE):
        d.mkdir(parents=True, exist_ok=True)
    DB = J / "journal.db"

    def db():
        c = sqlite3.connect(DB, timeout=20)  # базу могут открывать обе копии программы (Flux и Qwen)
        c.row_factory = sqlite3.Row
        return c

    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS entries(id TEXT PRIMARY KEY, time TEXT, updated TEXT, family TEXT, image TEXT UNIQUE,
            thumb TEXT, params TEXT, rating INTEGER, tags TEXT, comment TEXT);
        CREATE TABLE IF NOT EXISTS rules(id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, family TEXT, text TEXT,
            status TEXT, source TEXT, evidence TEXT, note TEXT);
        CREATE TABLE IF NOT EXISTS chats(id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, family TEXT, entry_id TEXT,
            role TEXT, content TEXT);
        CREATE INDEX IF NOT EXISTS i_entries_family ON entries(family);
        CREATE INDEX IF NOT EXISTS i_chats_entry ON chats(entry_id);
        """)
        cols = {r[1] for r in c.execute("PRAGMA table_info(entries)")}
        if "excluded" not in cols:  # «не учитывать в статистике» (добавлено позже — старые базы дополняем)
            c.execute("ALTER TABLE entries ADD COLUMN excluded INTEGER DEFAULT 0")
        # с какого компьютера оценка/разговор (для объединения журналов с разных устройств)
        for t in ("entries", "chats", "rules"):
            if "device" not in {r[1] for r in c.execute(f"PRAGMA table_info({t})")}:
                c.execute(f"ALTER TABLE {t} ADD COLUMN device TEXT")
        if "uid" not in {r[1] for r in c.execute("PRAGMA table_info(chats)")}:
            c.execute("ALTER TABLE chats ADD COLUMN uid TEXT")

    # ------------------------------------------------ этот компьютер (у каждой копии журнала своё имя)
    DEV_FILE = J / "device.json"
    try:
        DEV = json.loads(DEV_FILE.read_text("utf-8"))
    except Exception:
        DEV = {}
    if not DEV.get("id"):
        DEV = {"id": uuid.uuid4().hex[:12], "name": (socket.gethostname() or "Мой ПК")[:40]}
        DEV_FILE.write_text(json.dumps(DEV, ensure_ascii=False), "utf-8")

    def chat_uid(dev, t, eid, role, content):
        return hashlib.md5(f"{dev}|{t}|{eid}|{role}|{content}".encode("utf-8")).hexdigest()

    with db() as c:  # старые записи — с этого компьютера
        c.execute("UPDATE entries SET device=? WHERE device IS NULL", (DEV["id"],))
        c.execute("UPDATE rules SET device=? WHERE device IS NULL", (DEV["id"],))
        c.execute("UPDATE chats SET device=? WHERE device IS NULL", (DEV["id"],))
        for r in c.execute("SELECT id, time, entry_id, role, content, device FROM chats WHERE uid IS NULL").fetchall():
            c.execute("UPDATE chats SET uid=? WHERE id=?", (chat_uid(r["device"], r["time"], r["entry_id"], r["role"], r["content"]), r["id"]))
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS u_chats_uid ON chats(uid)")
        c.execute("""CREATE TABLE IF NOT EXISTS devices(id TEXT PRIMARY KEY, name TEXT, seen TEXT)""")
        c.execute("INSERT OR REPLACE INTO devices(id, name, seen) VALUES(?,?,?)", (DEV["id"], DEV["name"], time.strftime("%Y-%m-%d %H:%M:%S")))

    def backup():
        """Копия базы раз в день (хранится 14 последних) — журнал копится неделями, терять его нельзя."""
        try:
            dst = BACKUP / f"journal_{time.strftime('%Y%m%d')}.db"
            if not dst.exists():
                with db() as src, sqlite3.connect(dst) as out:
                    src.backup(out)
            for old in sorted(BACKUP.glob("journal_*.db"))[:-14]:
                old.unlink()
        except Exception:
            pass
    backup()

    def now():
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def err(m, code=400):
        return web.json_response({"error": str(m)}, status=code)

    def cfg(k, d=None):
        v = S.CONFIG.get(k)
        return d if v in (None, "") else v

    # ------------------------------------------------ параметры картинки из истории генераций
    def out_path(url):
        u = unquote((url or "").split("?")[0])
        if not u.startswith("/output/"):
            return None
        p = (S.OUTPUT / u[len("/output/"):]).resolve()
        return p if str(p).startswith(str(S.OUTPUT.resolve())) and p.is_file() else None

    def history_params(url):
        url = (url or "").split("?")[0]
        try:
            lines = S.HISTORY_FILE.read_text("utf-8", errors="replace").splitlines()
        except OSError:
            return None, None
        for line in reversed(lines):
            if url not in line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if url in e.get("images", []):
                return e.get("params") or {}, e
        return None, None

    def clean_params(p, e=None):
        """Только то, что важно для анализа (без служебных полей)."""
        keep = ["family", "mode", "unet", "clip", "vae", "weight_dtype", "prompt", "prompt_user", "prompt_src", "negative",
                "width", "height", "steps", "cfg", "sampler", "scheduler", "seed", "batch", "denoise", "use_text", "use_image",
                "image_mode", "ref_megapixels", "character", "character_name", "user_instruction", "face_area"]
        q = {k: p.get(k) for k in keep if p.get(k) not in (None, "")}
        q["loras"] = [{"name": l.get("name"), "strength": l.get("strength")} for l in p.get("loras") or []
                      if l.get("enabled", True) and l.get("name")]
        if p.get("images"):
            q["n_input_images"] = len(p["images"])
        if e and e.get("seconds"):
            q["seconds"] = e["seconds"]
        if e and e.get("warnings"):
            q["warnings"] = list(e["warnings"])[:6]
        return q

    def make_thumb(src: Path, eid):
        try:
            im = Image.open(src).convert("RGB")
            im.thumbnail((640, 640))
            name = f"{eid}.jpg"
            im.save(THUMBS / name, "JPEG", quality=86)
            return name
        except Exception:
            return ""

    def row_entry(r):
        if not r:
            return None
        d = dict(r)
        d["params"] = json.loads(d.get("params") or "{}")
        d["tags"] = json.loads(d.get("tags") or "[]")
        d["thumb_url"] = f"/journal/thumbs/{quote(d['thumb'])}" if d.get("thumb") else ""
        return d

    def entry_by_image(url):
        with db() as c:
            e = row_entry(c.execute("SELECT * FROM entries WHERE image=?", ((url or "").split("?")[0],)).fetchone())
        if e:
            e["auto_excluded"] = lora_problem(e["params"])
        return e

    def entry_by_id(eid):
        with db() as c:
            return row_entry(c.execute("SELECT * FROM entries WHERE id=?", (eid,)).fetchone())

    def lora_problem(p):
        """Почему оценка не годится для статистики: LoRA на самом деле не работали. Пустая строка — всё в порядке."""
        want = {"minimax": "minimax", "krea2": "krea2", "ltx": "ltx", "flux3": "flux3"}.get(p.get("family"), "qwen21")
        for w in p.get("warnings") or []:
            if "не смог применить LoRA" in w or "не LoRA" in w or "не подействует" in w:
                return "движок предупредил, что LoRA не подошла"
        for l in p.get("loras") or []:
            name = l.get("name") or ""
            if not (S.MODELS / "loras" / name).exists():
                if (S.MODELS / "diffusion_models" / Path(name).name).exists():
                    return f"«{Path(name).stem}» — целая модель, стояла как LoRA"
                continue
            try:
                a = CIV["lora_arch"](name)
            except Exception:
                continue
            if a == "checkpoint":
                return f"«{Path(name).stem}» — целая модель, стояла как LoRA"
            if a not in ("unknown", want):
                return f"«{Path(name).stem}» — LoRA для другой модели"
        return ""

    def all_entries(family=None, include_excluded=False):
        with db() as c:
            if family and family != "all":
                rows = c.execute("SELECT * FROM entries WHERE family=? AND rating IS NOT NULL ORDER BY time", (family,)).fetchall()
            else:
                rows = c.execute("SELECT * FROM entries WHERE rating IS NOT NULL ORDER BY time").fetchall()
        out = []
        for r in rows:
            e = row_entry(r)
            e["auto_excluded"] = lora_problem(e["params"])
            if include_excluded or not (e.get("excluded") or e["auto_excluded"]):
                out.append(e)
        return out

    def excluded_summary(family):
        es = all_entries(family, include_excluded=True)
        manual = [e for e in es if e.get("excluded")]
        auto = [e for e in es if not e.get("excluded") and e["auto_excluded"]]
        reasons = Counter(e["auto_excluded"] for e in auto)
        return {"manual": len(manual), "auto": len(auto), "reasons": [{"reason": r, "n": n} for r, n in reasons.most_common()]}

    def rules(family=None, status=None):
        q, a = "SELECT * FROM rules WHERE 1=1", []
        if family and family != "all":
            q += " AND (family=? OR family='all')"
            a.append(family)
        if status:
            q += " AND status=?"
            a.append(status)
        with db() as c:
            out = [dict(r) for r in c.execute(q + " ORDER BY id", a).fetchall()]
        for r in out:
            r["evidence"] = json.loads(r.get("evidence") or "[]")
        return out

    def accepted_rules_text(family, limit=40):
        rs = rules(family, "accepted")[-limit:]
        return [r["text"] for r in rs]

    # ------------------------------------------------ статистика (считает код, а не модель)
    def bins(v, edges, labels):
        for e, l in zip(edges, labels):
            if v < e:
                return l
        return labels[-1]

    def features(p):
        f = {}
        f["модель"] = [Path(str(p.get("unet") or "?")).name]
        f["режим"] = [{"edit": "редактор", "faceswap": "замена лица"}.get(p.get("mode"), "генерация")]
        loras = p.get("loras") or []
        f["LoRA"] = [Path(l["name"]).stem for l in loras] or ["без LoRA"]
        f["LoRA и сила"] = [f"{Path(l['name']).stem} @ " + bins(float(l.get("strength") or 1), [0.45, 0.75, 0.95, 1.15, 1.45],
                                                                 ["<0.45", "0.45–0.75", "0.75–0.95", "0.95–1.15", "1.15–1.45", "≥1.45"])
                           for l in loras]
        f["кол-во LoRA"] = [str(min(len(loras), 4)) + ("+" if len(loras) >= 4 else "")]
        if p.get("cfg") is not None:
            f["CFG"] = [f"{float(p['cfg']):g}"]
        if p.get("steps") is not None:
            f["шаги"] = [bins(int(p["steps"]), [5, 9, 17, 26, 36, 46], ["1–4", "5–8", "9–16", "17–25", "26–35", "36–45", "46+"])]
        if p.get("width") and p.get("height"):
            w, h = int(p["width"]), int(p["height"])
            r = w / h
            f["формат"] = ["вертикальный" if r < 0.9 else "горизонтальный" if r > 1.1 else "квадрат"]
            f["мегапиксели"] = [bins(w * h / 1048576, [0.8, 1.2, 1.7, 2.2, 3.0], ["<0.8", "0.8–1.2", "1.2–1.7", "1.7–2.2", "2.2–3", "3+"])]
        if p.get("sampler"):
            f["семплер/планировщик"] = [f"{p.get('sampler')}/{p.get('scheduler')}"]
        f["персонаж"] = ["да" if p.get("character") else "нет"]
        f["промпт"] = [{"helper": "от помощника", "helper_edited": "от помощника, исправлен"}.get(p.get("prompt_src"), "свой")]
        ln = len((p.get("prompt") or "").split())
        f["длина промпта"] = [bins(ln, [25, 60, 120, 200], ["<25 слов", "25–60", "60–120", "120–200", "200+"])]
        if p.get("use_image"):
            f["вход-картинка"] = [p.get("image_mode") or "reference"]
        if p.get("_device_name"):
            f["компьютер"] = [p["_device_name"]]
        if p.get("gpu"):
            f["видеокарта"] = [p["gpu"]]
        return f

    def compute_stats(family):
        es = all_entries(family)
        if not es:
            return {"n": 0, "groups": [], "tags": [], "family": family, "excluded": excluded_summary(family)}
        ratings = [e["rating"] for e in es]
        mean = statistics.mean(ratings)
        groups = defaultdict(list)
        names = device_names()
        for e in es:
            e["params"]["_device_name"] = names.get(e.get("device") or "", "")
            for k, vals in features(e["params"]).items():
                for v in vals:
                    groups[(k, v)].append(e["rating"])
        rows = []
        for (k, v), rs in groups.items():
            if len(rs) < 2 or len(rs) == len(es):  # группа из всех картинок ничего не говорит
                continue
            m = statistics.mean(rs)
            rows.append({"feature": k, "value": v, "n": len(rs), "avg": round(m, 2), "delta": round(m - mean, 2),
                         "low": sum(1 for x in rs if x <= 2), "high": sum(1 for x in rs if x >= 4)})
        rows.sort(key=lambda r: (-abs(r["delta"]) * min(r["n"], 10), r["feature"]))
        tagc = Counter()
        tagr = defaultdict(list)
        for e in es:
            for t in e["tags"]:
                tagc[t] += 1
                tagr[t].append(e["rating"])
        tags = [{"tag": t, "n": n, "avg": round(statistics.mean(tagr[t]), 2)} for t, n in tagc.most_common()]
        week = time.strftime("%Y-%m-%d", time.localtime(time.time() - 7 * 86400))
        return {"n": len(es), "mean": round(mean, 2), "dist": {str(i): ratings.count(i) for i in range(1, 6)},
                "excluded": excluded_summary(family),
                "last7": sum(1 for e in es if e["time"] >= week), "first": es[0]["time"], "last": es[-1]["time"],
                "groups": rows, "tags": tags, "family": family}

    def stats_text(st, limit=45):
        if not st["n"]:
            return "Оценок пока нет."
        out = [f"Всего оценок: {st['n']}, средняя {st['mean']} (1–5), распределение {st['dist']}, период {st['first'][:10]} — {st['last'][:10]}.",
               "Группы параметров (n — сколько картинок, avg — средняя оценка, Δ — отличие от общей средней, low/high — оценок ≤2 и ≥4):"]
        for r in st["groups"][:limit]:
            out.append(f"- {r['feature']} = {r['value']}: n={r['n']}, avg={r['avg']}, Δ={r['delta']:+}, low={r['low']}, high={r['high']}")
        if st["tags"]:
            out.append("Метки: " + "; ".join(f"{t['tag']} ×{t['n']} (avg {t['avg']})" for t in st["tags"][:20]))
        return "\n".join(out)

    def params_text(p):
        loras = ", ".join(f"{Path(l['name']).stem} ×{l.get('strength')}" for l in p.get("loras") or []) or "без LoRA"
        bits = [f"нейросеть: {p.get('family', 'flux')}", f"режим: {p.get('mode') or 'генерация'}", f"модель: {p.get('unet')}",
                f"энкодер: {p.get('clip')}", f"LoRA: {loras}", f"разрешение: {p.get('width')}×{p.get('height')}",
                f"шаги: {p.get('steps')}, CFG: {p.get('cfg')}, семплер: {p.get('sampler')}/{p.get('scheduler')}, seed: {p.get('seed')}"]
        if p.get("character"):
            bits.append(f"персонаж: {p.get('character_name') or p.get('character')}")
        if p.get("use_image"):
            bits.append(f"входные картинки: {p.get('n_input_images', '?')} ({p.get('image_mode')})")
        if p.get("denoise") is not None and p.get("image_mode") == "img2img":
            bits.append(f"denoise: {p.get('denoise')}")
        bits.append(f"промпт ({ {'helper': 'написан помощником', 'helper_edited': 'от помощника, исправлен'}.get(p.get('prompt_src'), 'написан пользователем')}): {p.get('prompt_user') or p.get('prompt') or p.get('user_instruction') or ''}")
        if p.get("negative"):
            bits.append(f"негатив: {p['negative']}")
        return "\n".join(bits)

    def lora_digests(p):
        out = []
        for l in p.get("loras") or []:
            try:
                d = CIV["digest_of"](l["name"])
            except Exception:
                continue
            st = d.get("strength") or {}
            s = f"- {Path(l['name']).stem}: " + (d.get("summary") or "")
            if st.get("min") is not None:
                s += f" (рекомендуемая сила {st['min']}–{st['max']})"
            if d.get("rules"):
                s += " Правила автора: " + "; ".join(d["rules"])
            if d.get("summary") or st.get("min") is not None or d.get("rules"):
                out.append(s)
        return "\n".join(out)

    def parse_rules(text):
        return [m.strip() for m in re.findall(r"^\s*[-*]?\s*ПРАВИЛО\s*:\s*(.+)$", text or "", flags=re.M | re.I) if m.strip()]

    def image_data_url(path: Path):
        im = Image.open(path).convert("RGB")
        im.thumbnail((1024, 1024))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    # ------------------------------------------------ HTTP: оценки
    @routes.post("/api/lab/rate")
    async def rate(req):
        b = await req.json()
        url = (b.get("image") or "").split("?")[0]
        src = out_path(url)
        if not src:
            return err("Картинка не найдена в output")
        rating = b.get("rating")
        rating = int(rating) if rating not in (None, "") else None
        if rating is not None and not 1 <= rating <= 5:
            return err("Оценка 1–5")
        old = entry_by_image(url)
        tags = [str(t)[:40] for t in (b.get("tags") if b.get("tags") is not None else (old or {}).get("tags", []))][:20]
        comment = str(b.get("comment") if b.get("comment") is not None else (old or {}).get("comment") or "")[:2000]
        if old:
            with db() as c:
                c.execute("UPDATE entries SET rating=COALESCE(?, rating), tags=?, comment=?, updated=? WHERE id=?",
                          (rating, json.dumps(tags, ensure_ascii=False), comment, now(), old["id"]))
            sync_soon()
            return web.json_response(entry_by_image(url))
        if rating is None:
            return err("Сначала поставьте оценку (звёзды)")
        p, e = history_params(url)
        from_ui = p is None
        if from_ui:
            p = b.get("params") or {}  # картинки без записи в истории — берём текущие настройки интерфейса
        params = clean_params(p, e)
        if from_ui:
            params["params_from_ui"] = True
        eid = time.strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:4]
        thumb = await asyncio.get_running_loop().run_in_executor(None, make_thumb, src, eid)
        with db() as c:
            params["gpu"] = gpu_name()
            try:
                if not S.SCORER["scores_for"](url):
                    S.SCORER["enqueue"](url, params.get("prompt_user") or params.get("prompt") or "", True)
            except Exception:
                pass
            c.execute("INSERT INTO entries(id,time,updated,family,image,thumb,params,rating,tags,comment,device) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                      (eid, now(), now(), params.get("family") or b.get("family") or "qwen", url, thumb,
                       json.dumps(params, ensure_ascii=False), rating, json.dumps(tags, ensure_ascii=False), comment, DEV["id"]))
        sync_soon()
        return web.json_response(entry_by_image(url))

    @routes.get("/api/lab/entry")
    async def get_entry(req):
        e = entry_by_image(req.query.get("image"))
        if not e:
            p, h = history_params(req.query.get("image"))
            return web.json_response({"entry": None, "params": clean_params(p, h) if p else None})
        return web.json_response({"entry": e, "params": e["params"]})

    @routes.get("/api/lab/ratings")
    async def ratings(_):
        with db() as c:
            return web.json_response({r["image"]: r["rating"] for r in c.execute("SELECT image, rating FROM entries WHERE rating IS NOT NULL")})

    @routes.post("/api/lab/exclude")
    async def exclude(req):
        b = await req.json()
        with db() as c:
            c.execute("UPDATE entries SET excluded=? WHERE image=?", (1 if b.get("value") else 0, (b.get("image") or "").split("?")[0]))
        return web.json_response(entry_by_image(b.get("image")) or {})

    @routes.post("/api/lab/reset")
    async def reset(req):
        """Начать статистику заново. Перед очисткой вся база копируется в journal/archive — ничего не теряется."""
        b = await req.json()
        fam = b.get("family") or "all"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        arch = ARCHIVE / f"journal_before_reset_{fam}_{stamp}.db"
        with db() as src, sqlite3.connect(arch) as out:
            src.backup(out)
        where, args = ("", ()) if fam == "all" else (" WHERE family=?", (fam,))
        with db() as c:
            n = c.execute("SELECT COUNT(*) FROM entries" + where, args).fetchone()[0]
            thumbs = [r[0] for r in c.execute("SELECT thumb FROM entries" + where, args) if r[0]]
            c.execute("DELETE FROM entries" + where, args)
            c.execute("DELETE FROM chats" + where, args)
            if b.get("rules") == "all":
                c.execute("DELETE FROM rules" + ("" if fam == "all" else " WHERE family=?"), args)
            elif b.get("rules") == "unconfirmed":
                c.execute("DELETE FROM rules WHERE status!='accepted'" + ("" if fam == "all" else " AND family=?"), args)
        for t in thumbs:  # миниатюры тоже в архив — вдруг пригодятся
            try:
                (ARCHIVE / "thumbs").mkdir(exist_ok=True)
                shutil.move(str(THUMBS / t), str(ARCHIVE / "thumbs" / t))
            except OSError:
                pass
        return web.json_response({"ok": True, "removed": n, "archive": str(arch.relative_to(S.ROOT))})

    @routes.get("/api/lab/tags")
    async def tags(_):
        return web.json_response({"good": TAGS_GOOD, "bad": TAGS_BAD})

    # ------------------------------------------------ HTTP: правила
    @routes.get("/api/lab/rules")
    async def get_rules(req):
        return web.json_response({"items": rules(req.query.get("family")),
                                  "in_prompt": bool(cfg("lab_rules_in_prompt", True))})

    @routes.post("/api/lab/rules")
    async def edit_rules(req):
        b = await req.json()
        act = b.get("action")
        with db() as c:
            if act == "add":
                text = (b.get("text") or "").strip()
                if not text:
                    return err("Пустое правило")
                ev = [x for x in (b.get("evidence") or []) if x]
                fam = b.get("family") or "all"
                same = c.execute("SELECT id, evidence FROM rules WHERE lower(trim(text))=lower(?) AND family=?", (text[:600], fam)).fetchone()
                if same:  # такое правило уже есть — добавляем доказательство, а не дубль
                    old = json.loads(same["evidence"] or "[]")
                    c.execute("UPDATE rules SET status=?, evidence=? WHERE id=?",
                              (b.get("status") or "accepted", json.dumps(old + [x for x in ev if x not in old]), same["id"]))
                else:
                    c.execute("INSERT INTO rules(time,family,text,status,source,evidence,note,device) VALUES(?,?,?,?,?,?,?,?)",
                              (now(), fam, text[:600], b.get("status") or "accepted", b.get("source") or "manual", json.dumps(ev), "", DEV["id"]))
            elif act in ("accept", "reject"):
                c.execute("UPDATE rules SET status=? WHERE id=?", ("accepted" if act == "accept" else "rejected", int(b["id"])))
            elif act == "edit":
                c.execute("UPDATE rules SET text=?, family=COALESCE(?, family) WHERE id=?", ((b.get("text") or "").strip()[:600], b.get("family"), int(b["id"])))
            elif act == "delete":
                c.execute("DELETE FROM rules WHERE id=?", (int(b["id"]),))
            elif act == "in_prompt":
                S.CONFIG["lab_rules_in_prompt"] = bool(b.get("value"))
                S.save_config(S.CONFIG)
            else:
                return err("Неизвестное действие")
        return web.json_response({"ok": True})

    # ------------------------------------------------ HTTP: разговор
    @routes.post("/api/lab/chat")
    async def lab_chat(req):
        b = await req.json()
        fam = b.get("family") or "qwen"
        url = (b.get("image") or "").split("?")[0]
        entry = entry_by_image(url) if url else None
        img_params = entry["params"] if entry else None
        if url and img_params is None:
            p, h = history_params(url)
            img_params = clean_params(p, h) if p else None
        vision = bool(LLM["vision"]())
        sp = LAB_RULES.format(fam={"minimax": "MiniMax H3 (видео)", "ltx": "LTX-2.5 (видео)", "krea2": "Krea 2", "flux3": "FLUX.3 Klein"}.get(fam, "Qwen-Image 2.1"),
                              vision=" Ты видишь саму картинку." if vision else " Картинку ты НЕ видишь (у помощника нет модели со зрением) — опирайся на слова пользователя.")
        ctx = []
        cur = b.get("current") or {}
        if cur:
            ctx.append("ТЕКУЩИЕ НАСТРОЙКИ ПРОГРАММЫ:\n" + params_text(clean_params(cur)))
        if img_params:
            ctx.append("ОБСУЖДАЕМАЯ КАРТИНКА — её параметры:\n" + params_text(img_params))
            dg = lora_digests(img_params)
            if dg:
                ctx.append("ВЫЖИМКИ LoRA:\n" + dg)
        if entry:
            ctx.append(f"ОЦЕНКА ПОЛЬЗОВАТЕЛЯ: {entry['rating']}/5; метки: {', '.join(entry['tags']) or 'нет'}; комментарий: {entry['comment'] or 'нет'}")
        try:
            sc = S.SCORER["scores_for"](url) if url else None
        except Exception:
            sc = None
        if sc:
            bits = []
            if sc.get("pick") is not None:
                bits.append(f"PickScore {sc['pick']:.1f} (выше, чем у {sc.get('pick_pct') or 0}% картинок пользователя)")
            if sc.get("aes") is not None:
                bits.append(f"эстетика LAION {sc['aes']:.1f}/10")
            ctx.append("АВТООЦЕНКА (модели, обученные на вкусе многих людей, не этого пользователя): " + "; ".join(bits))
        acc = accepted_rules_text(fam)
        ctx.append("ПРИНЯТЫЕ ПРАВИЛА:\n" + ("\n".join("- " + r for r in acc) if acc else "пока нет"))
        st = compute_stats(fam)
        ctx.append("СВОДКА ЖУРНАЛА:\n" + stats_text(st, 15))
        sp += "\n\n" + "\n\n".join(ctx)
        msgs = [{"role": "system", "content": sp}]
        hist = [m for m in (b.get("messages") or []) if m.get("role") in ("user", "assistant") and m.get("content")][-12:]
        src = out_path(url) if url else None
        for i, m in enumerate(hist):
            content = str(m["content"])[:4000]
            if vision and src and i == len(hist) - 1 and m["role"] == "user":
                data = await asyncio.get_running_loop().run_in_executor(None, image_data_url, src)
                msgs.append({"role": "user", "content": [{"type": "text", "text": content}, {"type": "image_url", "image_url": {"url": data}}]})
            else:
                msgs.append({"role": m["role"], "content": content})
        try:
            out = await LLM["chat"](msgs, max_tokens=900, temperature=0.5, need_vision=vision and bool(src))
        except Exception as e:
            return err(e)
        if out.strip().upper().startswith("REFUSED"):
            out = "Не могу помочь с этим: запрос касается несовершеннолетних."
        with db() as c:
            eid_ = entry["id"] if entry else None
            msgs_ = ([("user", str(hist[-1]["content"])[:4000])] if hist and hist[-1]["role"] == "user" else []) + [("assistant", out)]
            for role, content in msgs_:
                t = now()
                c.execute("INSERT OR IGNORE INTO chats(time,family,entry_id,role,content,device,uid) VALUES(?,?,?,?,?,?,?)",
                          (t, fam, eid_, role, content, DEV["id"], chat_uid(DEV["id"], t, eid_, role, content)))
        return web.json_response({"reply": out, "rules": parse_rules(out), "vision": vision,
                                  "entry_id": entry["id"] if entry else None})

    # ------------------------------------------------ HTTP: статистика, закономерности, отчёт
    @routes.get("/api/lab/stats")
    async def stats(req):
        return web.json_response(compute_stats(req.query.get("family") or "all"))

    @routes.post("/api/lab/patterns")
    async def patterns(req):
        b = await req.json()
        fam = b.get("family") or "all"
        st = compute_stats(fam)
        if st["n"] < 5:
            return err(f"Пока мало оценок ({st['n']}) — закономерности появятся хотя бы после 20–30 оценённых картинок.")
        acc = accepted_rules_text(fam)
        low = [e for e in all_entries(fam) if e["rating"] <= 2 and e["comment"]][-8:]
        high = [e for e in all_entries(fam) if e["rating"] >= 4 and e["comment"]][-8:]
        user = (stats_text(st) + "\n\nПринятые правила:\n" + ("\n".join("- " + r for r in acc) or "нет")
                + "\n\nКомментарии к плохим картинкам:\n" + ("\n".join(f"- [{e['rating']}] {e['comment']}" for e in low) or "нет")
                + "\n\nКомментарии к хорошим картинкам:\n" + ("\n".join(f"- [{e['rating']}] {e['comment']}" for e in high) or "нет"))
        try:
            out = await LLM["chat"]([{"role": "system", "content": PATTERN_RULES}, {"role": "user", "content": user}],
                                    max_tokens=1000, temperature=0.3)
        except Exception as e:
            return err(e)
        found = parse_rules(out)
        with db() as c:
            for r in found:
                if c.execute("SELECT 1 FROM rules WHERE lower(trim(text))=lower(?)", (r[:600],)).fetchone():
                    continue
                c.execute("INSERT INTO rules(time,family,text,status,source,evidence,note) VALUES(?,?,?,?,?,?,?)",
                          (now(), fam, r[:600], "proposed", "stats", "[]", f"по {st['n']} оценкам"))
        return web.json_response({"reply": out, "rules": found, "stats": st})

    def report_md(fam):
        st = compute_stats(fam)
        es = all_entries(fam)
        name = {"flux": "FLUX.2 Klein", "qwen": "Qwen-Image 2.1", "krea2": "Krea 2", "flux3": "FLUX.3 Klein", "minimax": "MiniMax H3", "ltx": "LTX-2.5"}.get(fam, "все нейросети")
        L = [f"# Отчёт Мастерской Zouwu Lab — {name}", "", f"Сформирован: {now()}. База после отчёта не очищается.", ""]
        if not st["n"]:
            return "\n".join(L + ["Оценок пока нет."]), st
        L += [f"Оценок: **{st['n']}**, средняя **{st['mean']}**, распределение {st['dist']}, период {st['first'][:10]} — {st['last'][:10]}.", ""]
        devs = device_counts()
        if len(devs) > 1:
            L += ["Источники: " + ", ".join(f"{d['name']} — {d['n']}" for d in devs) + ".", ""]
        ex_ = st.get("excluded") or {}
        if ex_.get("manual") or ex_.get("auto"):
            L += [f"Не учтены: вручную {ex_['manual']}, автоматически {ex_['auto']} ("
                  + "; ".join(f"{r['reason']} ×{r['n']}" for r in ex_.get("reasons", [])) + ").", ""]
        for title, status in (("Принятые правила", "accepted"), ("Предложенные, ещё не проверенные", "proposed")):
            rs = rules(fam, status)
            L += [f"## {title} ({len(rs)})", ""] + [f"- {r['text']}  _(источник: {r['source']}, доказательств: {len(r['evidence'])}, {r['time'][:10]})_" for r in rs] + [""]
        try:
            ag = S.SCORER["agreement"](fam)
        except Exception:
            ag = {}
        lines = []
        for k, n in (("pick", "PickScore"), ("aes", "Эстетика LAION")):
            x = ag.get(k) or {}
            if x.get("n"):
                lines.append(f"- {n}: согласие с оценками пользователя ρ = {x.get('rho')} (по {x['n']} картинкам); "
                             "среднее по звёздам: " + ", ".join(f"★{r} → {v}" for r, v in x.get("by_rating", {}).items()))
        if lines:
            L += ["## Автооценка против оценок пользователя", ""] + lines + [""]
        L += ["## Статистика по параметрам", "", "| параметр | значение | n | средняя | Δ | ≤2 | ≥4 |", "|---|---|---|---|---|---|---|"]
        for r in st["groups"][:80]:
            L.append(f"| {r['feature']} | {r['value']} | {r['n']} | {r['avg']} | {r['delta']:+} | {r['low']} | {r['high']} |")
        L += ["", "## Метки", ""] + [f"- {t['tag']}: {t['n']} раз, средняя {t['avg']}" for t in st["tags"]] + [""]

        def ex(e):
            p = e["params"]
            loras = ", ".join(f"{Path(l['name']).stem} ×{l.get('strength')}" for l in p.get("loras") or []) or "—"
            return (f"- **{e['rating']}/5** · {e['time'][:16]} · {p.get('family')} · {Path(str(p.get('unet'))).name} · LoRA: {loras} · "
                    f"{p.get('width')}×{p.get('height')} · шаги {p.get('steps')} · CFG {p.get('cfg')} · метки: {', '.join(e['tags']) or '—'}"
                    f"\n  - промпт: {(p.get('prompt_user') or p.get('prompt') or p.get('user_instruction') or '')[:600]}"
                    + (f"\n  - комментарий: {e['comment']}" if e["comment"] else "")
                    + (f"\n  - миниатюра: journal/thumbs/{e['thumb']}" if e.get("thumb") else ""))
        best = sorted(es, key=lambda e: (-e["rating"], e["time"]))[:15]
        worst = sorted(es, key=lambda e: (e["rating"], e["time"]))[:15]
        L += ["## Лучшие", ""] + [ex(e) for e in best] + ["", "## Худшие", ""] + [ex(e) for e in worst] + [""]
        with db() as c:
            n_chat = c.execute("SELECT COUNT(*) FROM chats" + ("" if fam == "all" else " WHERE family=?"), (() if fam == "all" else (fam,))).fetchone()[0]
        L += [f"Сообщений в разговорах Мастерской: {n_chat} (полностью — в выгрузке «Все данные»)."]
        return "\n".join(L), st

    @routes.get("/api/lab/report")
    async def report(req):
        fam = req.query.get("family") or "all"
        fmt = req.query.get("fmt", "md")
        stamp = time.strftime("%Y%m%d_%H%M")
        if fmt == "jsonl":
            lines = [json.dumps({"type": "entry", **e}, ensure_ascii=False) for e in all_entries(fam, include_excluded=True)]
            lines += [json.dumps({"type": "rule", **r}, ensure_ascii=False) for r in rules(fam)]
            with db() as c:
                q = "SELECT * FROM chats" + ("" if fam == "all" else " WHERE family=?") + " ORDER BY id"
                lines += [json.dumps({"type": "chat", **dict(r)}, ensure_ascii=False) for r in c.execute(q, (() if fam == "all" else (fam,)))]
            text, name = "\n".join(lines) + "\n", f"zouwulab_data_{fam}_{stamp}.jsonl"
        else:
            text, _ = report_md(fam)
            name = f"zouwulab_report_{fam}_{stamp}.md"
        (REPORTS / name).write_text(text, "utf-8")  # копия остаётся в journal/reports
        return web.Response(text=text, content_type="text/plain", charset="utf-8",
                            headers={"Content-Disposition": f"attachment; filename=\"{name}\""})

    # ------------------------------------------------ объединение журналов с разных компьютеров
    # Пакет журнала (.f2journal = zip): manifest.json, entries.jsonl, rules.jsonl, chats.jsonl, devices.jsonl, thumbs/*.
    # Слияние без дублей: оценки — по id, разговоры — по uid, правила — по тексту (доказательства объединяются).
    SYNC = {"last": "", "result": "", "error": "", "running": False, "seen": {}, "timer": None}
    SYNC_STATE = J / "sync_state.json"
    try:
        SYNC["seen"] = json.loads(SYNC_STATE.read_text("utf-8"))
    except Exception:
        SYNC["seen"] = {}

    def gpu_name():
        try:
            d = S.ENGINE_.system.get("devices", [{}])[0].get("name", "")
            return re.sub(r"^cuda:\d+\s*", "", d).split(":")[0].strip()[:60]
        except Exception:
            return ""

    def device_names():
        with db() as c:
            return {r["id"]: r["name"] for r in c.execute("SELECT id, name FROM devices")}

    def device_counts():
        names = device_names()
        with db() as c:
            rows = c.execute("SELECT device, COUNT(*) n FROM entries WHERE rating IS NOT NULL GROUP BY device").fetchall()
        return [{"id": r["device"], "name": names.get(r["device"], r["device"] or "?"), "n": r["n"], "me": r["device"] == DEV["id"]}
                for r in rows]

    def build_pack(only_mine=False):
        """Пакет журнала в памяти (bytes)."""
        buf = io.BytesIO()
        with db() as c, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            w = " WHERE device=?" if only_mine else ""
            a = (DEV["id"],) if only_mine else ()
            ents = [dict(r) for r in c.execute("SELECT * FROM entries" + w, a)]
            z.writestr("entries.jsonl", "\n".join(json.dumps(e, ensure_ascii=False) for e in ents))
            z.writestr("rules.jsonl", "\n".join(json.dumps(dict(r), ensure_ascii=False) for r in c.execute("SELECT * FROM rules" + w, a)))
            z.writestr("chats.jsonl", "\n".join(json.dumps(dict(r), ensure_ascii=False) for r in c.execute("SELECT * FROM chats" + w, a)))
            z.writestr("devices.jsonl", "\n".join(json.dumps(dict(r), ensure_ascii=False) for r in c.execute("SELECT * FROM devices")))
            z.writestr("manifest.json", json.dumps({"format": "flux2studio-journal", "version": 1, "device": DEV,
                                                    "exported": now(), "entries": len(ents)}, ensure_ascii=False))
            for e in ents:
                t = e.get("thumb")
                if t and (THUMBS / t).exists():
                    z.write(THUMBS / t, "thumbs/" + t)
        return buf.getvalue()

    def import_pack(data: bytes):
        """Добавить чужой журнал. Возвращает счётчики новых записей."""
        res = {"entries": 0, "chats": 0, "rules": 0, "rules_merged": 0, "device": ""}
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = set(z.namelist())
            if "manifest.json" not in names:
                raise ValueError("Это не журнал Zouwu Lab")
            man = json.loads(z.read("manifest.json"))
            if man.get("format") != "flux2studio-journal":
                raise ValueError("Это не журнал Zouwu Lab")
            src = man.get("device") or {}
            res["device"] = src.get("name") or "?"
            rows = lambda n: [json.loads(l) for l in z.read(n).decode("utf-8").splitlines() if l.strip()] if n in names else []
            with db() as c:
                for d in rows("devices.jsonl"):
                    if d.get("id") and d["id"] != DEV["id"]:
                        c.execute("INSERT INTO devices(id,name,seen) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name",
                                  (d["id"], d.get("name") or "?", d.get("seen") or ""))
                for e in rows("entries.jsonl"):
                    if not e.get("id") or c.execute("SELECT 1 FROM entries WHERE id=?", (e["id"],)).fetchone():
                        continue
                    dev = e.get("device") or src.get("id") or "?"
                    image = e.get("image") or ""
                    if dev != DEV["id"] and not image.startswith("ext:"):
                        image = f"ext:{dev}:{image}"   # картинки с другого компьютера здесь нет — только миниатюра
                    if c.execute("SELECT 1 FROM entries WHERE image=?", (image,)).fetchone():
                        continue
                    t = e.get("thumb") or ""
                    if t and "thumbs/" + t in names and not (THUMBS / t).exists():
                        (THUMBS / t).write_bytes(z.read("thumbs/" + t))
                    c.execute("INSERT INTO entries(id,time,updated,family,image,thumb,params,rating,tags,comment,excluded,device) "
                              "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                              (e["id"], e.get("time"), e.get("updated"), e.get("family"), image, t, e.get("params") or "{}",
                               e.get("rating"), e.get("tags") or "[]", e.get("comment") or "", e.get("excluded") or 0, dev))
                    res["entries"] += 1
                for m in rows("chats.jsonl"):
                    dev = m.get("device") or src.get("id") or "?"
                    uid = m.get("uid") or chat_uid(dev, m.get("time"), m.get("entry_id"), m.get("role"), m.get("content"))
                    cur = c.execute("INSERT OR IGNORE INTO chats(time,family,entry_id,role,content,device,uid) VALUES(?,?,?,?,?,?,?)",
                                    (m.get("time"), m.get("family"), m.get("entry_id"), m.get("role"), m.get("content"), dev, uid))
                    res["chats"] += cur.rowcount
                for r in rows("rules.jsonl"):
                    text = (r.get("text") or "").strip()
                    if not text:
                        continue
                    ev = json.loads(r.get("evidence") or "[]") if isinstance(r.get("evidence"), str) else (r.get("evidence") or [])
                    same = c.execute("SELECT id, evidence, status FROM rules WHERE lower(trim(text))=lower(?) AND family=?",
                                     (text, r.get("family") or "all")).fetchone()
                    if same:
                        old = json.loads(same["evidence"] or "[]")
                        new_ev = old + [x for x in ev if x and x not in old]
                        status = same["status"]
                        if status == "proposed" and r.get("status") in ("accepted", "rejected"):
                            status = r["status"]   # решение, принятое на другом компьютере, важнее «на проверке»
                        if new_ev != old or status != same["status"]:
                            c.execute("UPDATE rules SET evidence=?, status=? WHERE id=?", (json.dumps(new_ev), status, same["id"]))
                            res["rules_merged"] += 1
                    else:
                        c.execute("INSERT INTO rules(time,family,text,status,source,evidence,note,device) VALUES(?,?,?,?,?,?,?,?)",
                                  (r.get("time"), r.get("family") or "all", text, r.get("status") or "proposed", r.get("source") or "import",
                                   json.dumps(ev), r.get("note") or "", r.get("device") or src.get("id")))
                        res["rules"] += 1
        return res

    def pack_name():
        safe = re.sub(r"[^\w\-]+", "_", DEV["name"], flags=re.U).strip("_") or "pc"
        return f"zouwulab_journal_{safe}_{DEV['id'][:6]}.f2journal"

    def sync_now():
        """Общая папка (OneDrive, Яндекс Диск, сетевая): каждый компьютер пишет туда только СВОЙ файл,
        а чужие файлы забирает — так нет конфликтов одновременной записи."""
        d = (cfg("lab_sync_dir") or "").strip().strip('"')
        if not d:
            return None
        folder = Path(d)
        if not folder.is_dir():
            raise ValueError(f"Папка не найдена: {d}")
        SYNC["running"] = True
        try:
            mine = folder / pack_name()
            tmp = mine.with_name(mine.name + ".part")
            tmp.write_bytes(build_pack(only_mine=True))
            tmp.replace(mine)
            total = {"entries": 0, "chats": 0, "rules": 0, "rules_merged": 0, "files": 0}
            for f in sorted(folder.glob("*.f2journal")):
                if f.name == mine.name:
                    continue
                key, mt = f.name, f.stat().st_mtime
                if SYNC["seen"].get(key) == mt:
                    continue
                r = import_pack(f.read_bytes())
                for k in ("entries", "chats", "rules", "rules_merged"):
                    total[k] += r[k]
                total["files"] += 1
                SYNC["seen"][key] = mt
            SYNC_STATE.write_text(json.dumps(SYNC["seen"]), "utf-8")
            SYNC.update(last=now(), error="",
                        result=f"новых оценок: {total['entries']}, разговоров: {total['chats']}, правил: {total['rules']}"
                               + (f" (обновлено {total['rules_merged']})" if total["rules_merged"] else "")
                               + f"; прочитано файлов: {total['files']}")
            return total
        finally:
            SYNC["running"] = False

    async def sync_async():
        try:
            await asyncio.get_running_loop().run_in_executor(None, sync_now)
        except Exception as e:
            SYNC["error"] = str(e)

    def sync_soon():
        """После оценки — синхронизировать через минуту (несколько оценок подряд — одна выгрузка)."""
        if not cfg("lab_sync_dir"):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if SYNC["timer"]:
            SYNC["timer"].cancel()
        SYNC["timer"] = loop.call_later(60, lambda: asyncio.ensure_future(sync_async()))

    async def sync_loop():
        await asyncio.sleep(5)
        while True:
            if cfg("lab_sync_dir"):
                await sync_async()
            await asyncio.sleep(600)

    @routes.get("/api/lab/devices")
    async def devices(_):
        return web.json_response({"me": DEV, "devices": device_counts(), "sync_dir": cfg("lab_sync_dir") or "",
                                  "sync": {k: SYNC[k] for k in ("last", "result", "error", "running")}})

    @routes.post("/api/lab/device")
    async def set_device(req):
        b = await req.json()
        name = (b.get("name") or "").strip()[:40]
        if name:
            DEV["name"] = name
            DEV_FILE.write_text(json.dumps(DEV, ensure_ascii=False), "utf-8")
            with db() as c:
                c.execute("INSERT OR REPLACE INTO devices(id,name,seen) VALUES(?,?,?)", (DEV["id"], name, now()))
        if "sync_dir" in b:
            d = (b.get("sync_dir") or "").strip().strip('"')
            if d and not Path(d).is_dir():
                return err("Папка не найдена: " + d)
            S.CONFIG["lab_sync_dir"] = d
            S.save_config(S.CONFIG)
            if d:
                await sync_async()
        return web.json_response({"ok": True, "me": DEV, "sync": {k: SYNC[k] for k in ("last", "result", "error")}})

    @routes.post("/api/lab/sync")
    async def sync_route(_):
        if not cfg("lab_sync_dir"):
            return err("Не указана общая папка")
        await sync_async()
        if SYNC["error"]:
            return err(SYNC["error"])
        return web.json_response({"ok": True, "sync": {k: SYNC[k] for k in ("last", "result", "error")}})

    @routes.get("/api/lab/pack")
    async def export_pack(req):
        data = await asyncio.get_running_loop().run_in_executor(None, build_pack, req.query.get("mine") == "1")
        name = pack_name()
        ascii_name = f"zouwulab_journal_{DEV['id'][:6]}.f2journal"
        return web.Response(body=data, content_type="application/zip",
                            headers={"Content-Disposition": f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name)}"})

    @routes.post("/api/lab/import")
    async def import_route(req):
        reader = await req.multipart()
        total = {"entries": 0, "chats": 0, "rules": 0, "rules_merged": 0, "devices": []}
        async for part in reader:
            if part.name != "file":
                continue
            data = await part.read(decode=False)
            try:
                r = await asyncio.get_running_loop().run_in_executor(None, import_pack, data)
            except zipfile.BadZipFile:
                return err(f"«{part.filename}» — не файл журнала")
            except ValueError as e:
                return err(f"«{part.filename}»: {e}")
            for k in ("entries", "chats", "rules", "rules_merged"):
                total[k] += r[k]
            total["devices"].append(r["device"])
        return web.json_response(total)

    # ------------------------------------------------ для помощника промптов: принятые правила
    def prompt_rules(family):
        if not cfg("lab_rules_in_prompt", True):
            return []
        return accepted_rules_text(family, 25)

    return {"prompt_rules": prompt_rules, "thumbs": THUMBS, "dir": J, "compute_stats": compute_stats, "sync_loop": sync_loop}
