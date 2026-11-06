"""Замена лица: лицо с другого фото (или персонажа из библиотеки) переносится на картинку.

Как это работает
 1. Лица находит детектор YuNet (из библиотеки kornia, которая уже стоит вместе с движком; нужен
    только файл весов 0.4 МБ — models/face/yunet_final.pth). Можно и без него: обвести лицо кистью.
 2. На картинке строится маска-овал по найденному лицу (или берётся маска пользователя), вокруг —
    вырезка с запасом контекста; она масштабируется до ~1 Мп, чтобы лицо рисовалось в полном разрешении.
 3. Нейросеть (FLUX.2 Klein или Qwen-Image 2.1) перерисовывает только область маски. Ей подаются:
    исходник с размытым старым лицом (чтобы видела позу, свет, причёску, но не старую личность) и
    фото нового лица, вырезанное по голове.
 4. Результат вклеивается обратно с мягкой растушёвкой — всё вне маски остаётся пиксель в пиксель.
"""
import asyncio
import threading
import time
import uuid
from pathlib import Path

from aiohttp import web, ClientSession, ClientTimeout
from PIL import Image, ImageDraw, ImageFilter, ImageOps

YUNET_URLS = ["https://github.com/kornia/data/raw/main/yunet_final.pth",
              "https://raw.githubusercontent.com/kornia/data/main/yunet_final.pth"]

INSTR = {
    "flux": ("Face swap: give the person in the image the face of the person from the reference photo — exactly the same "
             "identity: face shape, eyes, eyebrows, nose, lips, jawline and skin tone. Keep the original head angle, gaze "
             "direction{expr}, lighting, shadows, color grading, grain and sharpness, {hair} so the new face blends "
             "seamlessly with the neck and body."),
    "qwen": ("Replace the face of the person in <image1> with the face of the person in <image2>: exactly the same identity — "
             "face shape, eyes, eyebrows, nose, lips, jawline and skin tone of <image2>. Keep the head angle, gaze direction"
             "{expr}, lighting, shadows and color grading of <image1>, {hair} blend it seamlessly with the neck; "
             "everything outside the face stays exactly as in <image1>."),
}


def register(routes, S, EDIT):
    FACE_DIR = S.MODELS / "face"
    FACE_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS = FACE_DIR / "yunet_final.pth"
    INPUT = S.INPUT
    TMP = EDIT["TMP"]
    st = {"det": None, "error": "", "downloading": False}
    det_lock = threading.Lock()
    cache = {}

    def err(m, code=400):
        return web.json_response({"error": str(m)}, status=code)

    # ------------------------------------------------ детектор
    def detector():
        with det_lock:
            if st["det"] is None:
                if not WEIGHTS.exists():
                    raise RuntimeError("Детектор лиц не скачан")
                import torch
                from unittest import mock
                from kornia.contrib import FaceDetector
                sd = torch.load(str(WEIGHTS), map_location="cpu")
                # kornia сама качает веса из интернета — подсовываем локальный файл
                with mock.patch("torch.hub.load_state_dict_from_url", return_value=sd):
                    st["det"] = FaceDetector(confidence_threshold=0.5)
            return st["det"]

    def detect(img: Image.Image, min_score=0.6):
        """Список лиц [{box:[x0,y0,x1,y1], score, kps:{...}}] в пикселях исходной картинки, крупные — первыми."""
        import numpy as np
        import torch
        from kornia.contrib import FaceDetectorResult, FaceKeypoint
        det = detector()
        img = ImageOps.exif_transpose(img).convert("RGB")
        W, H = img.size
        k = min(1.0, 1600 / max(W, H))           # огромные картинки — уменьшаем
        if max(W, H) < 480:
            k = 480 / max(W, H)                  # маленькие — увеличиваем, чтобы найти мелкие лица
        work = img.resize((max(1, int(W * k)), max(1, int(H * k))), Image.LANCZOS) if k != 1.0 else img
        x = torch.from_numpy(np.asarray(work).copy()).permute(2, 0, 1).float()[None]
        with det_lock, torch.no_grad():
            res = det(x)[0]
        faces = []
        names = {"eye_l": FaceKeypoint.EYE_LEFT, "eye_r": FaceKeypoint.EYE_RIGHT, "nose": FaceKeypoint.NOSE,
                 "mouth_l": FaceKeypoint.MOUTH_LEFT, "mouth_r": FaceKeypoint.MOUTH_RIGHT}
        for r in res:
            fr = FaceDetectorResult(r)
            sc = float(fr.score.item())
            if sc < min_score:
                continue
            x0, y0 = [float(v) / k for v in fr.top_left.tolist()]
            x1, y1 = [float(v) / k for v in fr.bottom_right.tolist()]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            kps = {n: [round(float(v) / k, 1) for v in fr.get_keypoint(kp).tolist()] for n, kp in names.items()}
            faces.append({"box": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)], "score": round(sc, 3), "kps": kps})
        faces.sort(key=lambda f: -(f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
        return faces

    def detect_file(path: Path):
        key = (str(path), path.stat().st_mtime)
        if key not in cache:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)
                cache[key] = {"faces": detect(im), "w": im.width, "h": im.height}
        return cache[key]

    # ------------------------------------------------ маски и вырезки
    def face_ellipse(box, area, W, H):
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if area == "head":   # лицо + волосы
            ew, eh, cy = w * 1.55, h * 1.8, cy - h * 0.2
        else:                # только лицо: от лба до подбородка, уши не трогаем
            ew, eh, cy = w * 1.0, h * 1.18, cy - h * 0.04
        return [max(0, cx - ew / 2), max(0, cy - eh / 2), min(W, cx + ew / 2), min(H, cy + eh / 2)]

    def make_mask(W, H, ell):
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).ellipse(ell, fill=255)
        return m

    def blurred_ref(img: Image.Image, mask: Image.Image, radius):
        """Копия картинки, где старое лицо размыто: модель видит позу и свет, но не прежнюю личность."""
        soft = mask.filter(ImageFilter.GaussianBlur(max(2, radius / 3)))
        return Image.composite(img.filter(ImageFilter.GaussianBlur(radius)), img, soft)

    def head_crop(img: Image.Image, box):
        """Вырезка головы с запасом (волосы, подбородок, немного шеи) — фото нового лица для модели."""
        W, H = img.size
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        side = max(w, h) * 2.1
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - h * 0.12
        l, t = max(0, cx - side / 2), max(0, cy - side / 2)
        r, b = min(W, cx + side / 2), min(H, cy + side / 2)
        c = img.crop((int(l), int(t), int(r), int(b)))
        if c.width * c.height > 1_100_000:
            k = (1_048_576 / (c.width * c.height)) ** 0.5
            c = c.resize((int(c.width * k), int(c.height * k)), Image.LANCZOS)
        elif max(c.size) < 512:  # мелкое лицо — увеличим, чтобы модель разглядела черты
            k = 512 / max(c.size)
            c = c.resize((int(c.width * k), int(c.height * k)), Image.LANCZOS)
        return c

    def prepare_donor(path: Path, face_index=None):
        """Фото донора → вырезка головы. Возвращает (имя в input, нашлось ли лицо)."""
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        faces = []
        if WEIGHTS.exists():
            try:
                faces = detect_file(path)["faces"]
            except Exception:
                faces = []
        tag = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:5]
        name = f"edit_tmp/{tag}_donor.png"
        if faces:
            i = face_index if isinstance(face_index, int) and 0 <= face_index < len(faces) else 0
            head_crop(img, faces[i]["box"]).save(INPUT / name)
            return name, True
        if img.width * img.height > 1_100_000:
            k = (1_048_576 / (img.width * img.height)) ** 0.5
            img = img.resize((int(img.width * k), int(img.height * k)), Image.LANCZOS)
        img.save(INPUT / name)
        return name, False

    def prepare_target(src: Path, box, user_mask: Path, area, keep_expr):
        """Маска области замены + референс с размытым лицом. Возвращает (путь маски, путь референса, ширина лица)."""
        img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        W, H = img.size
        tag = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:5]
        if user_mask:
            m = Image.open(user_mask).convert("L").resize((W, H), Image.NEAREST).point(lambda v: 255 if v > 127 else 0)
            bb = m.getbbox()
            if not bb:
                raise ValueError("Маска пустая")
            face_w = bb[2] - bb[0]
        else:
            ell = face_ellipse(box, area, W, H)
            m = make_mask(W, H, ell)
            face_w = box[2] - box[0]
        mask_path = TMP / f"{tag}_fmask.png"
        m.save(mask_path)
        # сильнее размываем, если выражение лица не нужно сохранять
        radius = max(3.0, face_w / (22 if keep_expr else 9))
        ref_path = TMP / f"{tag}_fref.png"
        blurred_ref(img, m, radius).save(ref_path)
        return mask_path, ref_path, face_w

    # ------------------------------------------------ HTTP
    @routes.get("/api/face/state")
    async def face_state(_):
        return web.json_response({"installed": WEIGHTS.exists(), "downloading": st["downloading"], "error": st["error"]})

    @routes.post("/api/face/install")
    async def face_install(_):
        if WEIGHTS.exists():
            return web.json_response({"ok": True})
        st.update(downloading=True, error="")
        last = ""
        try:
            for u in YUNET_URLS:
                try:
                    async with ClientSession(timeout=ClientTimeout(total=120)) as s:
                        async with s.get(u, headers={"User-Agent": "ZouwuLab/1.0"}) as r:
                            if r.status != 200:
                                last = f"HTTP {r.status}"
                                continue
                            data = await r.read()
                    if len(data) < 100_000:  # вместо файла пришла страница с ошибкой
                        last = "получен не тот файл"
                        continue
                    tmp = WEIGHTS.with_name(WEIGHTS.name + ".part")
                    tmp.write_bytes(data)
                    tmp.replace(WEIGHTS)
                    return web.json_response({"ok": True})
                except Exception as e:
                    last = str(e) or e.__class__.__name__
            st["error"] = "Не удалось скачать детектор лиц: " + last
            return err(st["error"] + ". Можно скачать вручную по ссылке " + YUNET_URLS[0] + " и положить в models\\face")
        finally:
            st["downloading"] = False

    @routes.post("/api/face/detect")
    async def face_detect(req):
        b = await req.json()
        path = EDIT["in_input"](b.get("image"))
        if not path:
            return err("Картинка не найдена")
        if not WEIGHTS.exists():
            return err("Детектор лиц не скачан", 409)
        try:
            r = await asyncio.get_running_loop().run_in_executor(None, detect_file, path)
        except Exception as e:
            return err("Детектор лиц не сработал: " + str(e) + ". Обведите лицо кистью — замена работает и так.")
        return web.json_response(r)

    @routes.post("/api/faceswap")
    async def faceswap(req):
        p = await req.json()
        e = S.check_models(p)
        if e:
            return err(e)
        src = EDIT["in_input"](p.get("image"))
        if not src:
            return err("Откройте картинку, на которой нужно заменить лицо")
        fam = p["family"] = "qwen"
        loop = asyncio.get_running_loop()
        # --- новое лицо
        warns = []
        if p.get("character"):
            refs, cm = S.CHARS["refs_for"](p["character"])
            if not refs:
                return err("Персонаж не найден или у него нет фото")
            donor_refs = refs[:3]
        else:
            dp = EDIT["in_input"](p.get("donor"))
            if not dp:
                return err("Добавьте фото нового лица или выберите персонажа")
            try:
                name, found = await loop.run_in_executor(None, prepare_donor, dp, p.get("donor_face"))
            except Exception as ex:
                return err("Не удалось прочитать фото лица: " + str(ex))
            if not found:
                warns.append("На фото нового лица детектор не нашёл лицо — модели подано всё фото. "
                             "Лучше взять фото, где лицо крупно и хорошо видно.")
            donor_refs = [name]
        # --- где менять
        user_mask = EDIT["in_input"](p.get("mask")) if p.get("mask") else None
        box = p.get("face")
        if not user_mask and not box:
            if not WEIGHTS.exists():
                return err("Детектор лиц не скачан — скачайте его (кнопка вверху) или обведите лицо кистью")
            try:
                faces = (await loop.run_in_executor(None, detect_file, src))["faces"]
            except Exception as ex:
                return err("Детектор лиц не сработал: " + str(ex))
            if not faces:
                return err("На картинке не найдено лицо. Обведите его кистью — замена сработает по маске.")
            box = faces[0]["box"]
        area = "head" if p.get("area") == "head" else "face"
        keep_expr = p.get("keep_expr", True)
        try:
            mask_path, ref_path, face_w = await loop.run_in_executor(
                None, prepare_target, src, box, user_mask, area, keep_expr)
        except Exception as ex:
            return err("Не удалось подготовить картинку: " + str(ex))
        # --- инструкция
        expr = ", facial expression" if keep_expr else ""
        if fam == "qwen":
            hair = "use the hairstyle and hair color of <image2>," if area == "head" else "keep the original hair,"
        else:
            hair = "use the hairstyle and hair color from the reference photo," if area == "head" else "keep the original hair,"
        instruction = INSTR["qwen" if fam == "qwen" else "flux"].format(expr=expr, hair=hair)
        extra_txt = (p.get("extra") or "").strip()
        if extra_txt:
            instruction += " " + extra_txt
        q = {**p, "char_refs": donor_refs, "char_phrase": False, "character": p.get("character"),
             "ref_path": str(ref_path), "mask": str(mask_path.relative_to(INPUT).as_posix()),
             "grow": int(max(3, face_w * 0.03)), "feather": int(max(6, face_w * 0.12)),
             "context": float(p.get("context", 1.0)), "min_crop": 512,
             "megapixels": float(p.get("megapixels") or 1.0), "denoise": float(p.get("strength", 1.0))}
        resp = await EDIT["core"](q, src, mask_path, instruction,
                                  extra={"instruction": instruction, "warnings": warns},
                                  mode="faceswap", hist_extra={"face_area": area})
        return resp

    return {"detect_file": detect_file, "weights": WEIGHTS}
