"""Персонажи: сохранённые лица, которые FLUX.2 Klein держит от генерации к генерации.

Klein умеет принимать фото как референс (ReferenceLatent) — это его родной способ сохранять
личность без FaceID/PuLID и без дополнительных библиотек. Персонаж = 1–4 фото (обычно
кадрированы по лицу/голове) в папке characters/<id>/. При генерации они копируются во input
и подаются как референсы, а к промпту добавляется фраза о сходстве.
"""
import json
import shutil
import time
import uuid
from pathlib import Path
from urllib.parse import quote, unquote

from aiohttp import web
from PIL import Image, ImageOps

MAX_PHOTOS = 4
PHRASE_GEN = ("The person is the same person as in the reference photo: keep the identical face, facial structure, "
              "eyes, nose, lips, skin tone and overall likeness.")
PHRASE_EDIT = "Use the face and identity of the person shown in the character reference photo."


def phrase_gen(family="flux", index=1):
    if family == "qwen":
        return (f"The person is the same person as in <image{index}>: keep the identical face, facial structure, "
                "eyes, nose, lips, skin tone and overall likeness.")
    return PHRASE_GEN


def phrase_edit(family="flux", index=2):
    if family == "qwen":
        return f"Give the person the face and identity of the person shown in <image{index}>, keep everything else in <image1> as it is."
    return PHRASE_EDIT


def register(routes, S):
    ROOT = S.ROOT
    CH = ROOT / "characters"
    CH.mkdir(exist_ok=True)
    CACHE = S.INPUT / "char_cache"
    CACHE.mkdir(parents=True, exist_ok=True)

    def err(m, code=400):
        return web.json_response({"error": str(m)}, status=code)

    def meta_of(cid):
        p = CH / cid / "meta.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            return None

    def save_meta(cid, m):
        (CH / cid / "meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), "utf-8")

    def public(cid, m):
        photos = [f"/characters/{cid}/{quote(ph)}?v={int((CH / cid / ph).stat().st_mtime)}"
                  for ph in m.get("photos", []) if (CH / cid / ph).exists()]
        return {"id": cid, "name": m.get("name", ""), "note": m.get("note", ""), "photos": photos,
                "created": m.get("created", "")}

    def crop_and_save(img: Image.Image, crop, dest: Path):
        img = ImageOps.exif_transpose(img).convert("RGB")
        if crop and len(crop) == 4:
            x0, y0, x1, y1 = [int(round(v)) for v in crop]
            x0, x1 = sorted((max(0, x0), min(img.width, x1)))
            y0, y1 = sorted((max(0, y0), min(img.height, y1)))
            if x1 - x0 >= 32 and y1 - y0 >= 32:
                img = img.crop((x0, y0, x1, y1))
        # не храним огромные файлы: до ~2 Мп
        if img.width * img.height > 2_100_000:
            k = (2_000_000 / (img.width * img.height)) ** 0.5
            img = img.resize((int(img.width * k), int(img.height * k)), Image.LANCZOS)
        img.save(dest)

    def new_char(name):
        cid = time.strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:4]
        (CH / cid).mkdir(parents=True)
        m = {"name": name.strip() or "Персонаж", "note": "", "photos": [], "created": time.strftime("%Y-%m-%d %H:%M")}
        save_meta(cid, m)
        return cid, m

    def add_photo(cid, m, img, crop):
        if len(m["photos"]) >= MAX_PHOTOS:
            raise ValueError(f"У персонажа уже {MAX_PHOTOS} фото — удалите лишнее")
        fn = f"ref_{uuid.uuid4().hex[:6]}.png"
        crop_and_save(img, crop, CH / cid / fn)
        m["photos"].append(fn)
        save_meta(cid, m)

    # ------------------------------------------------ для генерации/редактора
    def refs_for(cid):
        """Скопировать фото персонажа в input и вернуть имена для LoadImage."""
        m = meta_of(cid or "")
        if not m:
            return [], None
        names = []
        for ph in m.get("photos", [])[:MAX_PHOTOS]:
            src = CH / cid / ph
            if not src.exists():
                continue
            dst = CACHE / f"{cid}_{ph}"
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                shutil.copyfile(src, dst)
            names.append(f"char_cache/{dst.name}")
        return names, m

    # ------------------------------------------------ HTTP
    @routes.get("/api/chars")
    async def list_chars(_):
        out = []
        for d in sorted(CH.iterdir(), reverse=True):
            m = meta_of(d.name) if d.is_dir() else None
            if m:
                out.append(public(d.name, m))
        return web.json_response({"items": out})

    @routes.post("/api/chars")
    async def create(req):
        """multipart: name, crops (json-список рамок), file × N — или JSON {name, url, crop} из результата."""
        if req.content_type == "application/json":
            b = await req.json()
            cid, m = (b["id"], meta_of(b["id"])) if b.get("id") else new_char(b.get("name", ""))
            if not m:
                return err("Персонаж не найден")
            url = (b.get("url") or "").split("?")[0]
            base = S.OUTPUT if url.startswith("/output/") else S.INPUT if url.startswith("/input/") else None
            if not base:
                return err("Неверная картинка")
            p = (base / unquote(url.split("/", 2)[2])).resolve()
            if not str(p).startswith(str(base.resolve())) or not p.is_file():
                return err("Картинка не найдена")
            try:
                add_photo(cid, m, Image.open(p), b.get("crop"))
            except ValueError as e:
                return err(e)
            return web.json_response(public(cid, m))
        reader = await req.multipart()
        name, crops, cid, files = "", [], None, []
        async for part in reader:
            if part.name == "name":
                name = (await part.text())[:80]
            elif part.name == "id":
                cid = (await part.text()).strip()
            elif part.name == "crops":
                try:
                    crops = json.loads(await part.text())
                except Exception:
                    crops = []
            elif part.name == "file" and part.filename:
                files.append(await part.read(decode=False))
        if not files:
            return err("Добавьте хотя бы одно фото")
        if cid:
            m = meta_of(cid)
            if not m:
                return err("Персонаж не найден")
        else:
            cid, m = new_char(name)
        import io
        try:
            for i, data in enumerate(files):
                add_photo(cid, m, Image.open(io.BytesIO(data)), crops[i] if i < len(crops) else None)
        except ValueError as e:
            return err(e)
        except Exception as e:
            return err("Не удалось прочитать фото: " + str(e))
        return web.json_response(public(cid, m))

    @routes.post("/api/chars/{id}/edit")
    async def edit(req):
        cid = req.match_info["id"]
        m = meta_of(cid)
        if not m:
            return err("Персонаж не найден", 404)
        b = await req.json()
        for k in ("name", "note"):
            if k in b:
                m[k] = str(b[k])[:500]
        if "remove_photo" in b:
            i = int(b["remove_photo"])
            if 0 <= i < len(m["photos"]):
                f = CH / cid / m["photos"].pop(i)
                if f.exists():
                    f.unlink()
        save_meta(cid, m)
        return web.json_response(public(cid, m))

    @routes.post("/api/chars/{id}/delete")
    async def delete(req):
        cid = req.match_info["id"]
        if not meta_of(cid):
            return err("Персонаж не найден", 404)
        shutil.rmtree(CH / cid, ignore_errors=True)
        return web.json_response({"ok": True})

    return {"refs_for": refs_for, "PHRASE_GEN": PHRASE_GEN, "PHRASE_EDIT": PHRASE_EDIT, "dir": CH,
            "phrase_gen": phrase_gen, "phrase_edit": phrase_edit}
