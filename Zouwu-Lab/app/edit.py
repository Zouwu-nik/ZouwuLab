"""Редактор: правка картинки FLUX.2 Klein по текстовой инструкции, по желанию — только в области маски.

Без маски: картинка подаётся как референс, модель перерисовывает её по инструкции, результат
возвращается к исходному размеру.
С маской (inpaint с «crop & stitch»): вырезается область вокруг маски с запасом контекста,
масштабируется до ~1 Мп (так мелкие детали получают полное разрешение модели), правится только
внутри маски, затем вклеивается обратно в оригинал с мягкой растушёвкой. Всё вне маски остаётся
пиксель в пиксель как было.
"""
import asyncio
import json
import math
import random
import shutil
import time
import uuid
from pathlib import Path
from urllib.parse import quote, unquote

from aiohttp import web
from PIL import Image, ImageFilter, ImageOps

import workflow


def _snap(v, m=16):
    return max(4 * m, int(round(v / float(m))) * m)


def _fit_mp(w, h, mp, m=16):
    """Размер ~mp мегапикселей с сохранением пропорций, кратный m (16 для FLUX, 32 для Qwen)."""
    k = math.sqrt(mp * 1024 * 1024 / (w * h))
    return _snap(w * k, m), _snap(h * k, m)


def register(routes, S, LLM):
    INPUT, OUTPUT = S.INPUT, S.OUTPUT
    TMP = INPUT / "edit_tmp"
    TMP.mkdir(parents=True, exist_ok=True)
    for f in TMP.iterdir():  # временные файлы старше 2 дней
        try:
            if time.time() - f.stat().st_mtime > 2 * 86400:
                f.unlink()
        except OSError:
            pass

    def err(msg, code=400):
        return web.json_response({"error": str(msg)}, status=code)

    def in_input(name):
        p = (INPUT / (name or "")).resolve()
        return p if name and str(p).startswith(str(INPUT.resolve())) and p.is_file() else None

    # ------------------------------------------------ подготовка
    def prepare(src_path: Path, mask_path, o):
        """Готовит файлы для движка и возвращает (params, info для вклейки)."""
        im = ImageOps.exif_transpose(Image.open(src_path)).convert("RGB")
        W, H = im.size
        tag = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:5]
        info = {"src": str(src_path), "W": W, "H": H, "mode": "full"}

        mask = None
        if mask_path:
            m = Image.open(mask_path).convert("L").resize((W, H), Image.NEAREST).point(lambda v: 255 if v > 127 else 0)
            if m.getbbox():
                mask = m

        ref_im = None  # отдельная картинка-референс (для замены лица — с размытым старым лицом), режется так же
        if o.get("ref_path"):
            ref_im = ImageOps.exif_transpose(Image.open(o["ref_path"])).convert("RGB").resize((W, H), Image.LANCZOS)

        if mask is None:
            mp = min(float(o.get("full_megapixels") or 0) or (W * H / 1048576), 2.5)
            mp = max(mp, 0.25)
            tw, th = _fit_mp(W, H, mp, 32 if o.get("family") == "qwen" else 16)
            src_name = f"edit_tmp/{tag}_src.png"
            im.resize((tw, th), Image.LANCZOS).save(INPUT / src_name)
            out = {"src": src_name, "mask": None, "width": tw, "height": th}
            if ref_im is not None:
                out["ref_src"] = f"edit_tmp/{tag}_ref.png"
                ref_im.resize((tw, th), Image.LANCZOS).save(INPUT / out["ref_src"])
            return out, info

        # --- расширение маски (grow) и растушёвка (feather) в пикселях оригинала
        grow = max(0, int(o.get("grow", 12)))
        feather = max(0, int(o.get("feather", 16)))
        mg = mask
        if grow:
            mg = mask.filter(ImageFilter.GaussianBlur(grow / 2)).point(lambda v: 255 if v > 8 else 0)
        # --- область вырезки: рамка маски + контекст вокруг
        x0, y0, x1, y1 = mg.getbbox()
        bw, bh = x1 - x0, y1 - y0
        ctx = float(o.get("context", 0.6))  # доля размера маски, добавляемая с каждой стороны
        side_min = int(o.get("min_crop", 512))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        cw = max(bw * (1 + 2 * ctx), side_min)
        ch = max(bh * (1 + 2 * ctx), side_min)
        cw, ch = min(cw, W), min(ch, H)
        left = int(max(0, min(cx - cw / 2, W - cw)))
        top = int(max(0, min(cy - ch / 2, H - ch)))
        box = (left, top, int(left + cw), int(top + ch))
        crop = im.crop(box)
        cmask = mg.crop(box)
        mp = float(o.get("megapixels") or 1.0)
        tw, th = _fit_mp(crop.width, crop.height, mp, 32 if o.get("family") == "qwen" else 16)
        src_name, mask_name = f"edit_tmp/{tag}_src.png", f"edit_tmp/{tag}_mask.png"
        crop.resize((tw, th), Image.LANCZOS).save(INPUT / src_name)
        cmask.resize((tw, th), Image.LANCZOS).point(lambda v: 255 if v > 127 else 0).save(INPUT / mask_name)
        extra = {}
        if ref_im is not None:
            extra["ref_src"] = f"edit_tmp/{tag}_ref.png"
            ref_im.crop(box).resize((tw, th), Image.LANCZOS).save(INPUT / extra["ref_src"])
        # маска вклейки: расширенная и растушёванная
        paste = mg.filter(ImageFilter.GaussianBlur(feather / 2)) if feather else mg
        paste_name = f"edit_tmp/{tag}_paste.png"
        paste.save(INPUT / paste_name)
        info.update(mode="mask", box=box, paste=str(INPUT / paste_name))
        return {"src": src_name, "mask": mask_name, "width": tw, "height": th, **extra}, info

    def compose(result_path: Path, info) -> Image.Image:
        res = Image.open(result_path).convert("RGB")
        orig = ImageOps.exif_transpose(Image.open(info["src"])).convert("RGB")
        if info["mode"] == "full":
            return res.resize((info["W"], info["H"]), Image.LANCZOS)
        box = info["box"]
        patch = res.resize((box[2] - box[0], box[3] - box[1]), Image.LANCZOS)
        layer = orig.copy()
        layer.paste(patch, box[:2])
        paste = Image.open(info["paste"]).convert("L")
        return Image.composite(layer, orig, paste)

    # ------------------------------------------------ HTTP
    @routes.post("/api/edit/source")
    async def source(req):
        """Сделать исходником для редактора готовую картинку из output (результат генерации/правки)."""
        b = await req.json()
        url = b.get("url", "").split("?")[0]
        if not url.startswith("/output/"):
            return err("Можно взять только картинку из результатов")
        p = (OUTPUT / unquote(url[len("/output/"):])).resolve()
        if not str(p).startswith(str(OUTPUT.resolve())) or not p.is_file():
            return err("Картинка не найдена")
        name = f"edsrc_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:5]}{p.suffix.lower()}"
        shutil.copyfile(p, INPUT / name)
        with Image.open(INPUT / name) as im:
            w, h = im.size
        return web.json_response({"name": name, "w": w, "h": h})

    @routes.post("/api/edit/refine")
    async def refine(req):
        b = await req.json()
        src = in_input(b.get("image"))
        if not src:
            return err("Нет исходной картинки")
        try:
            text, vision = await LLM["refine_edit"](src, in_input(b.get("mask")), b.get("instruction", ""),
                                                    [l for l in b.get("loras", []) if l.get("enabled", True)],
                                                    family="qwen", character=bool(b.get("character")))
        except Exception as e:
            return err(e)
        return web.json_response({"instruction": text, "vision": vision})

    @routes.post("/api/edit")
    async def run_edit(req):
        p = await req.json()
        e = S.check_models(p)
        if e:
            return err(e)
        src = in_input(p.get("image"))
        if not src:
            return err("Загрузите картинку, которую нужно править")
        instruction = (p.get("instruction") or "").strip()
        if not instruction:
            return err("Напишите, что нужно изменить")
        mask = in_input(p.get("mask")) if p.get("mask") else None
        refined = None
        if p.get("refine") and LLM["available"]():
            try:
                refined, _ = await LLM["refine_edit"](src, mask, instruction,
                                                      [l for l in p.get("loras", []) if l.get("enabled", True)],
                                                      family="qwen", character=bool(p.get("character")))
                instruction = refined or instruction
            except Exception as ex:
                return err("Помощник не смог уточнить инструкцию: " + str(ex) +
                           "\nСнимите галочку «Уточнить инструкцию ИИ», чтобы отправить текст как есть.")
        if p.get("character"):
            refs, cm = S.CHARS["refs_for"](p["character"])
            if not refs:
                return err("Персонаж не найден или у него нет фото")
            p["char_refs"] = refs
            if p.get("char_phrase", True):
                instruction = instruction.rstrip() + " " + S.CHARS["phrase_edit"]("qwen", 2)
        return await core(p, src, mask, instruction, extra={"instruction": instruction, "refined": refined is not None})

    async def core(p, src, mask, instruction, extra=None, mode="edit", hist_extra=None):
        p["family"] = "qwen"  # правка и замена лица — только Qwen-Image 2.1
        """Общая часть правки: подготовка (crop & stitch), граф, очередь, вклейка результата.
        p["char_refs"] — фото-референсы (персонаж или донор лица), p["ref_path"] — отдельный референс вместо исходника."""
        try:
            gp, info = await asyncio.get_running_loop().run_in_executor(None, prepare, src, mask, p)
        except Exception as ex:
            return err("Не удалось подготовить картинку: " + str(ex))
        seed = int(p.get("seed", -1))
        if seed < 0:
            seed = random.randint(0, 2**50)
        params = {**p, **gp, "instruction": instruction, "seed": seed}
        graph = workflow.build_edit(await S.engine_params(params))
        hist = {**{k: p.get(k) for k in ("family", "unet", "clip", "vae", "weight_dtype", "steps", "cfg", "sampler", "scheduler", "loras")},
                "character": p.get("character"), "mode": mode, "edit_source": p.get("image"), "edit_mask": p.get("mask"),
                "prompt": instruction, "user_instruction": p.get("instruction"), "seed": seed,
                "width": info["W"], "height": info["H"], **(hist_extra or {})}

        async def post(urls):
            out = []
            for u in urls:
                rp = OUTPUT / unquote(u.split("?")[0][len("/output/"):])
                img = await asyncio.get_running_loop().run_in_executor(None, compose, rp, info)
                name = f"edit_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}.png"
                img.save(OUTPUT / "images" / name)
                try:
                    rp.unlink()
                except OSError:
                    pass
                out.append("/output/images/" + quote(name))
            return out

        return await S.queue_graph(graph, hist, post=post,
                                   extra={**(extra or {}), "seed": seed, "crop": list(info.get("box") or [])})

    # ------------------------------------------------ вырезка изменений: «до» и «после» → только изменённое на прозрачном
    CUT = OUTPUT / "cutouts"
    CUT.mkdir(parents=True, exist_ok=True)

    def cutout_job(before, after, method, sens, fill):
        import alpha
        import numpy as np
        b = ImageOps.exif_transpose(Image.open(before)).convert("RGB")
        a = ImageOps.exif_transpose(Image.open(after)).convert("RGB")
        matte, used = None, "threshold"
        if method == "ai" and S.MATTING["installed"]():
            matte = S.MATTING["matte"](a)
            used = "ai" if matte is not None else "threshold"
        layer, frac = alpha.diff_layer(b, a, matte=matte, sens=sens, fill=fill)
        if layer is None:
            return None
        tag = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
        layer.save(CUT / f"cut_{tag}.png")
        m = np.asarray(layer.getchannel("A")) > 127
        Image.fromarray((m * 255).astype("uint8")).filter(ImageFilter.MaxFilter(5)).save(CUT / f"mask_{tag}.png")
        box = layer.getchannel("A").getbbox()
        return {"url": "/output/cutouts/" + quote(f"cut_{tag}.png"), "mask_url": "/output/cutouts/" + quote(f"mask_{tag}.png"),
                "method": used, "area": round(frac * 100, 1), "box": list(box) if box else None,
                "w": layer.width, "h": layer.height}

    @routes.post("/api/edit/cutout")
    async def cutout(req):
        b = await req.json()
        before, after = in_input(b.get("before")), in_input(b.get("after"))
        if not before or not after:
            return err("Нужны две версии картинки: до правки и после")
        method = b.get("method") if b.get("method") in ("ai", "threshold") else ("ai" if S.MATTING["installed"]() else "threshold")
        sens = max(0.0, min(1.0, float(b.get("sens", 0.5))))
        try:
            r = await asyncio.get_running_loop().run_in_executor(None, cutout_job, before, after, method, sens, bool(b.get("fill", True)))
        except Exception as ex:
            return err("Не удалось вырезать: " + str(ex))
        if not r:
            return err("Изменений почти нет — вырезать нечего. Прибавьте чувствительность или сравните с более ранней версией.")
        return web.json_response(r)

    @routes.get("/api/matting")
    async def matting_state(_):
        return web.json_response(S.MATTING["state"]())

    @routes.post("/api/matting/install")
    async def matting_install(_):
        asyncio.create_task(S.MATTING["install"]())
        await asyncio.sleep(0.2)
        return web.json_response(S.MATTING["state"]())

    return {"core": core, "prepare": prepare, "in_input": in_input, "TMP": TMP}
