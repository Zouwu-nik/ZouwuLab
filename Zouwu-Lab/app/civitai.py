"""Civitai (civitai.red / civitai.com): поиск LoRA, скачивание с описанием и картинкой,
опознание уже лежащих LoRA по хэшу. Метаданные хранятся рядом с файлом:
    models/loras/<имя>.civitai.json   — название с сайта, описание, триггер-слова, ссылка
    models/loras/<имя>.preview.jpg    — первая картинка со страницы
"""
import asyncio
import hashlib
import json
import os
import re
import shutil
import struct
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote

from aiohttp import web, ClientSession, ClientTimeout

HOSTS = ["civitai.red", "civitai.com"]
BASE_FILTERS = {
    "klein9": (["Flux.2 Klein 9B", "Flux.2 Klein 9B-base"], r"klein.*9\s*b"),
    "klein4": (["Flux.2 Klein 4B", "Flux.2 Klein 4B-base"], r"klein.*4\s*b"),
    "klein": (["Flux.2 Klein 9B", "Flux.2 Klein 9B-base", "Flux.2 Klein 4B", "Flux.2 Klein 4B-base"], r"klein"),
    "flux2": ([], r"flux\.?\s*2"),
    # на Civitai база новой модели называется «Qwen 2» (в тегах — «qwen 2.1»)
    "qwen21": (["Qwen 2", "Qwen 2.1", "Qwen-Image 2.1"], r"qwen[\s_-]*(image[\s_-]*)?2(\.\d)?\b"),
    "qwen": ([], r"qwen"),
    # MiniMax H3 (видео): точное название базы на Civitai пока неизвестно — пробуем варианты, затем ищем словом
    "minimax": (["MiniMax H3", "Minimax H3", "MiniMax-H3"], r"minimax|hailuo|\bh3\b"),
    # Krea 2 (картинки) и LTX‑2.5 (видео; большинство LoRA от LTX‑2.3 к ней подходят)
    "krea2": (["Krea 2", "Krea2", "Krea-2"], r"krea[\s_-]*2"),
    "flux3": (["Flux.3 Klein", "Flux.3 Image Klein", "Flux.3", "FLUX.3 Klein"], r"flux[\s._-]*3"),
    "ltx": (["LTXV 2.5", "LTXV 2.3", "LTXV2", "LTX 2.5", "LTX-2"], r"ltx[\s_-]*v?[\s_-]*2"),
    "any": ([], ""),
}
FALLBACK_QUERY = {"flux3": "flux 3 klein", "krea2": "krea 2", "ltx": "ltx 2", "minimax": "minimax h3", "qwen21": "qwen", "qwen": "qwen", "klein9": "klein", "klein4": "klein", "klein": "klein", "flux2": "flux"}
BAD_BASE_NAMES = set()   # названия баз, которые API Civitai не принимает (узнаём на лету)
SORTS = {"downloads": "Most Downloaded", "rated": "Highest Rated", "new": "Newest"}
PREVIEW_EXT = [".jpg", ".jpeg", ".png", ".webp", ".mp4", ".webm"]


class CivError(Exception):
    pass


def register(routes, S):
    """S — модуль server (CONFIG, MODELS, TRANSFERS, safe_name, list_models, jerr)."""
    LORAS = S.MODELS / "loras"

    def domain():
        d = (S.CONFIG.get("civitai_domain") or "civitai.red").strip()
        return d if d in HOSTS else "civitai.red"

    def hosts():
        return [domain()] + [h for h in HOSTS if h != domain()]

    def site(host):
        # FLUX2_CIVITAI_SITE — только для тестов (подмена сайта локальным сервером)
        return os.environ.get("FLUX2_CIVITAI_SITE") or f"https://{host}"

    def auth_headers():
        h = {"User-Agent": "ZouwuLab/1.0", "Accept": "application/json"}
        tok = (S.CONFIG.get("civitai_token") or "").strip()
        if tok:
            h["Authorization"] = f"Bearer {tok}"
        return h

    async def api(path, params=None):
        last = "нет ответа"
        for host in hosts():
            try:
                async with ClientSession(timeout=ClientTimeout(total=40)) as s:
                    async with s.get(f"{site(host)}/api/v1/{path}", params=params, headers=auth_headers()) as r:
                        if r.status == 200:
                            return await r.json(content_type=None)
                        txt = (await r.text())[:300]
                        last = f"HTTP {r.status}: {txt}"
                        if r.status in (400, 404):
                            raise CivError(last)
            except CivError:
                raise
            except Exception as e:
                last = f"{host}: {e or e.__class__.__name__}"
        raise CivError("Civitai недоступен — " + last)

    # ------------------------------------------------ нормализация ответа API
    def thumb(url, w=450):
        if not url:
            return url
        if re.search(r"/(width=\d+|original=true)[^/]*/", url):
            return re.sub(r"/(width=\d+|original=true)[^/]*/", f"/width={w}/", url, count=1)
        head, _, tail = url.rpartition("/")
        return f"{head}/width={w}/{tail}" if "image.civitai.com" in url else url

    def first_image(images):
        imgs = [i for i in (images or []) if i.get("url")]
        for i in imgs:
            if (i.get("type") or "image") == "image":
                return i
        return imgs[0] if imgs else None

    def page_url(mid, vid=None):
        return f"https://{domain()}/models/{mid}" + (f"?modelVersionId={vid}" if vid else "")

    def primary_file(files):
        files = [f for f in (files or []) if (f.get("type") or "Model") == "Model"] or list(files or [])
        for f in files:
            if f.get("primary"):
                return f
        for f in files:
            if str(f.get("name", "")).lower().endswith(".safetensors"):
                return f
        return files[0] if files else None

    def version_card(model, v):
        img = first_image(v.get("images"))
        f = primary_file(v.get("files"))
        mid = model.get("id") or v.get("modelId")
        return {
            "model_id": mid, "version_id": v.get("id"),
            "name": model.get("name") or (v.get("model") or {}).get("name") or "",
            "version_name": v.get("name", ""), "base_model": v.get("baseModel", ""),
            "trained_words": v.get("trainedWords") or [],
            "type": model.get("type") or (v.get("model") or {}).get("type") or "",
            "nsfw": bool(model.get("nsfw") or (v.get("model") or {}).get("nsfw")),
            "image": {"url": img["url"], "thumb": thumb(img["url"]), "type": img.get("type") or "image",
                      "nsfw_level": img.get("nsfwLevel") or 0,
                      "width": img.get("width"), "height": img.get("height")} if img else None,
            "file": {"name": f.get("name"), "size_kb": f.get("sizeKB"), "url": f.get("downloadUrl"),
                     "sha256": ((f.get("hashes") or {}).get("SHA256") or "").lower()} if f else None,
            "page": page_url(mid, v.get("id")),
            "creator": (model.get("creator") or {}).get("username", ""),
            "downloads": (model.get("stats") or {}).get("downloadCount", 0),
            "likes": (model.get("stats") or {}).get("thumbsUpCount", 0),
        }

    LORA_TYPES = {"lora", "locon", "lycoris", "dora"}

    def model_card(model, rx=""):
        versions = [v for v in model.get("modelVersions") or []
                    if not rx or re.search(rx, v.get("baseModel") or "", re.I)]
        if not versions:
            return None
        card = version_card(model, versions[0])
        card["versions"] = [{"id": v.get("id"), "name": v.get("name"), "base_model": v.get("baseModel"),
                             "installed": is_installed(primary_file(v.get("files")))} for v in versions]
        card["installed"] = card["versions"][0]["installed"]
        return card

    def is_installed(f):
        return bool(f and f.get("name") and (LORAS / S.safe_name(f["name"])).exists())

    # ------------------------------------------------ локальные метаданные
    def meta_path(fname):
        p = LORAS / fname
        return p.with_name(p.stem + ".civitai.json")

    def preview_of(fname):
        p = LORAS / fname
        for ext in PREVIEW_EXT:
            q = p.with_name(p.stem + ".preview" + ext)
            if q.exists():
                return q
        return None

    def read_meta(fname):
        mp = meta_path(fname)
        if mp.exists():
            try:
                return json.loads(mp.read_text("utf-8"))
            except Exception:
                return None
        return None

    def write_meta(fname, meta):
        old = read_meta(fname) or {}
        for k in ("title", "note", "strength", "trained_words_custom", "digest"):  # правки пользователя и выжимку не затираем
            if old.get(k) is not None and not meta.get(k):
                meta[k] = old[k]
        meta_path(fname).write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")

    async def save_preview(fname, image):
        if not image or not image.get("url"):
            return
        url = image["url"] if image.get("type") == "video" else thumb(image["url"], 768)
        ext = ".mp4" if image.get("type") == "video" else (Path(urlparse(url).path).suffix.lower() or ".jpg")
        if ext not in PREVIEW_EXT:
            ext = ".jpg"
        old = preview_of(fname)
        try:
            async with ClientSession(timeout=ClientTimeout(total=60)) as s:
                async with s.get(url, headers={"User-Agent": "ZouwuLab/1.0"}) as r:
                    if r.status != 200:
                        return
                    data = await r.read()
            if old:
                old.unlink()
            p = LORAS / fname
            p.with_name(p.stem + ".preview" + ext).write_bytes(data)
        except Exception:
            pass

    async def fetch_full(mid, vid):
        """Полные данные: модель (описание) + версия."""
        model = await api(f"models/{mid}") if mid else None
        if model:
            v = next((x for x in model.get("modelVersions") or [] if x.get("id") == vid), None)
            if v is None and vid:
                v = await api(f"model-versions/{vid}")
            if v is None:
                v = (model.get("modelVersions") or [None])[0]
        else:
            v = await api(f"model-versions/{vid}")
            model = {"id": v.get("modelId"), **(v.get("model") or {})}
        if not v:
            raise CivError("Версия не найдена")
        return model, v

    def build_meta(model, v):
        card = version_card(model, v)
        return {
            "source": "civitai", "model_id": card["model_id"], "version_id": card["version_id"],
            "name": card["name"], "version_name": card["version_name"], "base_model": card["base_model"],
            "trained_words": card["trained_words"], "nsfw": card["nsfw"],
            "description": model.get("description") or "", "version_description": v.get("description") or "",
            "tags": model.get("tags") or [], "creator": card["creator"],
            "page": f"https://civitai.red/models/{card['model_id']}?modelVersionId={card['version_id']}",
            "page_com": f"https://civitai.com/models/{card['model_id']}?modelVersionId={card['version_id']}",
            "image": card["image"],
            "examples": example_prompts(v),
            "file_hashes": sorted({str(h).lower() for h in ((primary_file(v.get("files")) or {}).get("hashes") or {}).values() if h}),
            "file_name": (primary_file(v.get("files")) or {}).get("name") or "",
            "example_stats": example_stats(v.get("images") or [], v.get("id"), meta_names(model, v)),
        }

    def meta_names(model, v):
        names = {norm(model.get("name") or ""), norm((primary_file(v.get("files")) or {}).get("name") or "")}
        return {n for n in names if len(n) >= 3}

    def norm(t):
        t = re.sub(r"\.(safetensors|pt|ckpt)$", "", str(t or "").lower())
        return re.sub(r"[^a-z0-9а-я]+", "", t)

    def example_stats(images, vid, names, hashes=()):
        """С какой силой LoRA и какими настройками автор/пользователи делали картинки на странице."""
        weights, cfg, steps, samplers = [], [], [], []
        hashes = {h.lower() for h in hashes or ()}
        for im in images or []:
            m = im.get("meta") or {}
            if not isinstance(m, dict):
                continue
            w = None
            for r in m.get("civitaiResources") or []:
                if r.get("modelVersionId") == vid and (r.get("weight") is not None or r.get("strength") is not None):
                    w = r.get("weight", r.get("strength"))
            if w is None:
                for r in (m.get("resources") or []) + (m.get("additionalResources") or []):
                    if str(r.get("type", "")).lower() not in ("lora", "locon", "lycoris", ""):
                        continue
                    ok = (r.get("hash") and str(r["hash"]).lower() in hashes) or (norm(r.get("name")) in names)
                    if ok and (r.get("weight") is not None or r.get("strength") is not None):
                        w = r.get("weight", r.get("strength"))
            if w is None:
                for nm, val in re.findall(r"<lora:([^:>]+):([-\d.]+)", m.get("prompt") or ""):
                    if norm(nm) in names:
                        w = val
            try:
                if w is not None and 0.05 <= abs(float(w)) <= 3:
                    weights.append(round(float(w), 2))
            except (TypeError, ValueError):
                pass
            for key, arr in (("cfgScale", cfg), ("steps", steps)):
                try:
                    if m.get(key) is not None:
                        arr.append(float(m[key]))
                except (TypeError, ValueError):
                    pass
            if m.get("sampler"):
                samplers.append(str(m["sampler"]))
        return {"weights": weights, "cfg": cfg, "steps": steps, "samplers": samplers}

    async def fetch_image_meta(vid, names, hashes=()):
        """Картинки пользователей с этой LoRA (их параметры генерации) — для рекомендаций по силе и промптам."""
        items = []
        for nsfw in ("X", None):
            try:
                params = [("modelVersionId", str(vid)), ("limit", "40"), ("sort", "Most Reactions")]
                if nsfw:
                    params.append(("nsfw", nsfw))
                items = (await api("images", params)).get("items") or []
                break
            except Exception:
                continue
        stats = example_stats(items, vid, names, hashes)
        prompts = []
        for im in items:
            pr = (((im.get("meta") or {}) if isinstance(im.get("meta"), dict) else {}).get("prompt") or "").strip()
            if pr and pr not in prompts:
                prompts.append(pr[:700])
        return stats, prompts[:8]

    def example_prompts(v, limit=5):
        """Промпты, с которыми автор делал картинки на странице LoRA."""
        out = []
        for im in v.get("images") or []:
            pr = ((im.get("meta") or {}).get("prompt") or "").strip()
            if pr and pr not in out:
                out.append(pr[:700])
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------ теги из самого файла LoRA (kohya: ss_tag_frequency)
    _tags_cache = {}

    def file_tags(fname, top=10):
        return [t for t, _ in file_tag_counts(fname, top)]

    def file_tag_counts(fname, top=10):
        p = LORAS / fname
        try:
            key = (str(p), p.stat().st_mtime)
        except OSError:
            return []
        if key in _tags_cache:
            return _tags_cache[key]
        tags = []
        try:
            if p.suffix.lower() == ".safetensors":
                with open(p, "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    if n < 50_000_000:
                        md = json.loads(f.read(n)).get("__metadata__") or {}
                        tf = md.get("ss_tag_frequency")
                        if isinstance(tf, str):
                            tf = json.loads(tf)
                        cnt = Counter()
                        for ds in (tf or {}).values():
                            for t, c in (ds or {}).items():
                                t = t.strip()
                                if t:
                                    cnt[t] += int(c or 0)
                        tags = cnt.most_common(top)
        except Exception:
            tags = []
        _tags_cache[key] = tags
        return tags

    # ------------------------------------------------ для какой модели сделана LoRA (по именам слоёв в файле)
    _arch_cache = {}
    ARCH_LABEL = {"flux3": "FLUX.3 Klein", "krea2": "Krea 2", "ltx": "LTX‑2.x", "minimax": "MiniMax H3", "flux": "FLUX", "qwen21": "Qwen‑Image 2.1", "qwen_old": "старый Qwen‑Image (20B)",
                  "sdxl": "SD / SDXL", "other": "другая модель", "unknown": "не определено",
                  "checkpoint": "целая модель (не LoRA)"}

    def lora_arch(fname):
        p = LORAS / fname
        try:
            key = (str(p), p.stat().st_mtime)
        except OSError:
            return "unknown"
        if key in _arch_cache:
            return _arch_cache[key]
        arch = "unknown"
        try:
            if p.suffix.lower() == ".safetensors":
                with open(p, "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    if n < 200_000_000:
                        keys = [k.lower() for k in json.loads(f.read(n)) if k != "__metadata__"]
                        j = " ".join(keys[:4000])
                        blocks = [int(m) for m in re.findall(r"transformer_blocks[._](\d+)[._]", j)]
                        is_lora = any(t in j for t in ("lora", "lokr", "hada_", "loha", "oft_", ".alpha", "dora_scale"))
                        if keys and not is_lora and sum(k.endswith(".weight") for k in keys) > 50:
                            arch = "checkpoint"   # целая модель (дообученный чекпойнт), а не LoRA
                        elif "audio_attn" in j or "video_to_audio" in j or "audio_to_video" in j \
                                or (blocks and ("attn1" in j or "attn2" in j) and "img_mlp" not in j
                                    and "down_blocks" not in j and "input_blocks" not in j and "lora_te" not in j) \
                                or ("ltx" in p.name.lower() and "attn1" in j):
                            arch = "ltx"
                        elif "content_mode_blocks" in j or "single_stream_modulations" in j \
                                or (re.search(r"flux[._-]?3", p.name.lower()) and ("single_blocks" in j or "double_blocks" in j)):
                            arch = "flux3"
                        elif "txtfusion" in j or re.search(r"attn[._]w[qkvo][._]", j) or "krea" in p.name.lower() and "blocks" in j:
                            arch = "krea2"
                        elif "to_gate_compress" in j or ("qkv_proj" in j and "blocks." in j and "double_blocks" not in j) \
                                or "audio_patch_proj" in j or ("minimax" in p.name.lower() and "h3" in p.name.lower()):
                            arch = "minimax"
                        elif "double_blocks" in j or "single_blocks" in j or "single_transformer_blocks" in j:
                            arch = "flux"
                        elif "add_q_proj" in j or "txt_mlp" in j or "img_mod" in j or "img_mlp_net" in j or "img_mlp.net" in j \
                                or (blocks and max(blocks) >= 32):
                            arch = "qwen_old"
                        elif blocks and ("img_mlp" in j or "to_q" in j or "gate_up" in j):
                            arch = "qwen21"
                        elif "down_blocks" in j or "input_blocks" in j or "lora_te" in j or "up_blocks" in j:
                            arch = "sdxl"
                        elif keys:
                            arch = "other"
        except Exception:
            arch = "unknown"
        _arch_cache[key] = arch
        return arch

    # ------------------------------------------------ совместимость LoRA с Qwen-Image 2.1
    # В движке слои MLP Qwen 2.1 хранятся «сшитыми» (gate_up), а LoRA обычно адресуют половины
    # (gate_layer / proj). Движок разрезает их только для ключей с префиксом «transformer.» или без
    # префикса; в популярных форматах «diffusion_model.…» и kohya «lora_unet_…» эти слои молча
    # отбрасываются, и LoRA почти не действует. Делаем копию с переименованными ключами.
    FIX_DIR = "_qwen_fixed"
    KOHYA_PARTS = {"attn_to_q": "attn.to_q", "attn_to_k": "attn.to_k", "attn_to_v": "attn.to_v",
                   "attn_to_out_0": "attn.to_out.0", "img_mlp_gate_layer": "img_mlp.gate_layer",
                   "img_mlp_proj": "img_mlp.proj", "img_mlp_out": "img_mlp.out", "img_mlp_gate_up": "img_mlp.gate_up"}

    def _qwen_key(k):
        for pre in ("model.diffusion_model.", "diffusion_model.", "base_model.model."):
            if k.startswith(pre):
                return "transformer." + k[len(pre):]
        m = re.match(r"^lora_unet_transformer_blocks_(\d+)_([a-z0-9_]+?)(\.[a-z_]+(?:\.weight)?|\.alpha)$", k)
        if m and m.group(2) in KOHYA_PARTS:
            return f"transformer.transformer_blocks.{m.group(1)}.{KOHYA_PARTS[m.group(2)]}{m.group(3)}"
        return k

    def _needs_qwen_fix(keys):
        for k in keys:
            if re.search(r"img_mlp[._](gate_layer|proj)", k) and not (k.startswith("transformer.") or k.startswith("transformer_blocks.")):
                return True
        return False

    def lora_has_te(fname):
        """Есть ли в LoRA веса для текстового энкодера (тогда её надо подключать и к энкодеру)."""
        p = LORAS / fname
        if p.suffix.lower() != ".safetensors":
            return False
        try:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                keys = [k for k in json.loads(f.read(n)) if k != "__metadata__"]
        except Exception:
            return False
        return any(k.startswith(("text_encoder", "lora_te", "te_", "te1", "te2")) for k in keys)

    GATE_UP = re.compile(r"img_mlp[._]gate_up")

    def _split_gate_up(k, t):
        """Кусок LoRA для склеенного слоя gate_up → два куска для раздельных gate_layer и proj
        (у дообученных моделей, где эти половины хранятся отдельно). Порядок половин — как в движке: gate, потом proj."""
        a, b = k.replace("gate_up", "gate_layer"), k.replace("gate_up", "proj")
        kl = k.lower()
        # выход слоя (строки) делится пополам; вход и alpha — общие
        if re.search(r"(lora_b|lora_up)\.weight$", kl) or kl.endswith(".diff") or kl.endswith(".diff_b") or kl.endswith("dora_scale"):
            h = t.shape[0] // 2
            return [(a, t[:h].contiguous()), (b, t[h:].contiguous())]
        return [(a, t), (b, t.clone())]

    def prepare_qwen_lora(fname, fused=True):
        """Имя LoRA, которое надо отдать движку для Qwen 2.1 (при необходимости — исправленная копия).
        fused=False — у выбранной модели слои MLP хранятся раздельно (часть дообученных моделей): куски LoRA
        для склеенного gate_up разрезаются на gate_layer и proj."""
        src = LORAS / fname
        if src.suffix.lower() != ".safetensors" or lora_arch(fname) != "qwen21":
            return fname
        try:
            with open(src, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                keys = [k for k in json.loads(f.read(n)) if k != "__metadata__"]
        except Exception:
            return fname
        split = not fused and any(GATE_UP.search(k) for k in keys)
        if not _needs_qwen_fix(keys) and not split:
            return fname
        out = LORAS / FIX_DIR / (Path(fname).stem + ("__qwenfix_split" if split else "__qwenfix") + ".safetensors")
        if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
            from safetensors import safe_open
            from safetensors.torch import save_file
            out.parent.mkdir(parents=True, exist_ok=True)
            tensors, meta = {}, {}
            with safe_open(str(src), framework="pt") as f:
                meta = f.metadata() or {}
                for k in f.keys():
                    nk, t = _qwen_key(k), f.get_tensor(k)
                    if split and GATE_UP.search(nk):
                        for kk, tt in _split_gate_up(nk, t):
                            tensors[kk] = tt
                    else:
                        tensors[nk] = t
            tmp = out.with_name(out.name + ".part")
            save_file(tensors, str(tmp), metadata=meta)
            tmp.replace(out)
        return str(Path(FIX_DIR) / out.name)

    def triggers(fname, m=None):
        """(слова, источник): ручные → Civitai → теги из файла."""
        m = m if m is not None else (read_meta(fname) or {})
        if m.get("trained_words_custom") is not None:
            return list(m["trained_words_custom"]), "manual"
        if m.get("trained_words"):
            return list(m["trained_words"]), "civitai"
        d = m.get("digest") or {}
        if d.get("made_by") in ("llm", "user") and d.get("triggers"):  # помощник нашёл их в тексте описания
            return list(d["triggers"]), "digest"
        ft = file_tag_counts(fname)
        if not ft:
            return [], ""
        # слово-триггер при обучении стоит в каждой подписи — берём самые частые теги (с тем же счётом, что и первый)
        mx = ft[0][1]
        return [t for t, c in ft if c >= mx][:3], "file"

    def html_to_text(h, limit=900):
        h = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", h or "", flags=re.S | re.I)
        h = re.sub(r"<(br|/p|/li|/h\d)[^>]*>", "\n", h or "", flags=re.I)
        h = re.sub(r"<[^>]+>", "", h)
        h = re.sub(r"&nbsp;", " ", h)
        h = re.sub(r"&amp;", "&", h)
        h = re.sub(r"\n\s*\n+", "\n", h).strip()
        return h[:limit]

    NUM = r"(\d(?:[.,]\d{1,2})?)"
    STR_WORDS = r"(?:lora\s*)?(?:weight|strength|scale|intensity|вес|сила|влияни[ея]|强度|权重)"

    def parse_strength(text):
        """Рекомендуемая сила из текста описания: (min, max) или None."""
        t = html_to_text(text, 20000).lower()
        m = re.search(STR_WORDS + r"[^0-9\n]{0,30}?" + NUM + r"\s*(?:-|–|—|~|to|до|and)\s*" + NUM, t)
        if m:
            a, b = float(m.group(1).replace(",", ".")), float(m.group(2).replace(",", "."))
            if 0.05 <= a <= 3 and 0.05 <= b <= 3:
                return (min(a, b), max(a, b))
        m = re.search(STR_WORDS + r"[^0-9\n]{0,30}?" + NUM, t)
        if m:
            a = float(m.group(1).replace(",", "."))
            if 0.05 <= a <= 3:
                return (a, a)
        raw = (text or "").replace("&lt;", "<").replace("&gt;", ">").lower()
        vals = [float(x) for x in re.findall(r"<lora:[^:>]+:([\d.]+)", raw) if 0.05 <= float(x) <= 3]
        if vals:
            return (min(vals), max(vals))
        return None

    def _q(vals, q):
        v = sorted(vals)
        return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]

    def auto_digest(fname, m=None):
        """Выжимка без языковой модели: триггеры, сила (из описания и из примеров на странице)."""
        m = m if m is not None else (read_meta(fname) or {})
        words, _ = triggers(fname, m)
        st = {}
        rng = parse_strength((m.get("description") or "") + "\n" + (m.get("version_description") or ""))
        ws = (m.get("example_stats") or {}).get("weights") or []
        if rng:
            st = {"min": rng[0], "max": rng[1], "rec": round((rng[0] + rng[1]) / 2, 2), "source": "описание"}
        elif len(ws) >= 2:
            st = {"min": _q(ws, 0.2), "max": _q(ws, 0.8), "rec": _q(ws, 0.5), "source": f"примеры ({len(ws)})"}
        elif len(ws) == 1:
            st = {"min": ws[0], "max": ws[0], "rec": ws[0], "source": "пример"}
        return {"made_by": "auto", "triggers": words, "strength": st, "summary": "", "rules": [], "tips_en": "", "settings": ""}

    def digest_of(fname, m=None):
        m = m if m is not None else (read_meta(fname) or {})
        d = m.get("digest")
        if d and d.get("made_by") in ("llm", "user"):
            return d
        return auto_digest(fname, m)

    def update_meta(fname, **fields):
        m = read_meta(fname) or {}
        m.update(fields)
        meta_path(fname).write_text(json.dumps(m, ensure_ascii=False, indent=2), "utf-8")
        return m

    def lora_context(fname):
        """Всё, что помощнику промптов стоит знать о LoRA."""
        m = read_meta(fname) or {}
        words, src = triggers(fname, m)
        return {
            "file": fname, "title": m.get("title") or m.get("name") or Path(fname).stem,
            "triggers": words, "triggers_source": src,
            "file_tags": file_tags(fname),
            "description": html_to_text(m.get("description")),
            "version_description": html_to_text(m.get("version_description"), 400),
            "examples": m.get("examples") or [], "note": m.get("note") or "",
            "base_model": m.get("base_model") or "", "digest": digest_of(fname, m),
            "example_stats": m.get("example_stats") or {},
        }

    def parse_link(text):
        text = (text or "").strip()
        if text.isdigit():
            return int(text), None
        u = urlparse(text)
        m = re.search(r"/models/(\d+)", u.path)
        vid = parse_qs(u.query).get("modelVersionId", [None])[0]
        m2 = re.search(r"/api/download/models/(\d+)", u.path)
        if m2:
            return None, int(m2.group(1))
        if not m:
            raise CivError("Не похоже на ссылку Civitai вида https://civitai.red/models/12345")
        return int(m.group(1)), int(vid) if vid and vid.isdigit() else None

    async def start_download(mid, vid):
        model, v = await fetch_full(mid, vid)
        meta = build_meta(model, v)
        f = primary_file(v.get("files"))
        if not f:
            raise CivError("У этой версии нет файла для скачивания")
        fname = S.safe_name(f.get("name") or f"civitai_{v['id']}.safetensors")
        vid = v["id"]
        urls = [f"{site(h)}/api/download/models/{vid}" for h in hosts()]
        if f.get("downloadUrl") and f["downloadUrl"] not in urls:
            urls.append(f["downloadUrl"])
        mtype = str(model.get("type") or (v.get("model") or {}).get("type") or "").lower()
        if mtype == "checkpoint":  # целая модель — сразу в папку моделей, а не в LoRA
            task = S.TRANSFERS.new("download", "diffusion_models", fname, meta["name"] or fname)
            task["title"] = meta["name"] + " (модель)"
            if (S.MODELS / "diffusion_models" / fname).exists():
                task["status"], task["error"] = "done", "Эта модель уже скачана"
                return task
            task["alt_urls"] = urls[1:]
            asyncio.create_task(S.TRANSFERS.download(task, urls[0]))
            return task
        task = S.TRANSFERS.new("download", "loras", fname, meta["name"] or fname)
        task["title"] = meta["name"]
        if (LORAS / fname).exists():
            write_meta(fname, meta)
            await save_preview(fname, meta["image"])
            task["status"], task["error"] = "done", "Файл уже был — обновлены описание и картинка"
            return task
        write_meta(fname, meta)
        asyncio.create_task(save_preview(fname, meta["image"]))
        task["alt_urls"] = urls[1:]
        asyncio.create_task(enrich(fname, meta))

        async def after():
            if not preview_of(fname):
                await save_preview(fname, meta["image"])
            for hook in HOOKS["downloaded"]:  # например, помощник разберёт описание
                try:
                    await hook(fname)
                except Exception:
                    pass
        task["after"] = after
        asyncio.create_task(S.TRANSFERS.download(task, urls[0]))
        return task

    async def enrich(fname, meta):
        """Дополнить описание параметрами картинок пользователей (сила LoRA, CFG, шаги, промпты)."""
        try:
            names = {norm(meta.get("name")), norm(meta.get("file_name")), norm(Path(fname).stem)}
            stats, prompts = await fetch_image_meta(meta["version_id"], {n for n in names if len(n) >= 3}, meta.get("file_hashes") or [])
            m = read_meta(fname) or meta
            old = m.get("example_stats") or {}
            for k in ("weights", "cfg", "steps", "samplers"):
                stats[k] = (old.get(k) or []) + stats.get(k, [])
            ex = list(m.get("examples") or [])
            for pr in prompts:
                if pr not in ex and len(ex) < 8:
                    ex.append(pr)
            update_meta(fname, example_stats=stats, examples=ex)
        except Exception:
            pass

    HOOKS = {"downloaded": []}

    def public_task(t):
        return {k: v for k, v in t.items() if k not in ("after",)}

    # ------------------------------------------------ HTTP
    def err(e, code=400):
        return web.json_response({"error": str(e)}, status=code)

    @routes.get("/api/civitai/search")
    async def search(req):
        q = req.query
        base = q.get("base", "klein")
        names, rx = BASE_FILTERS.get(base, BASE_FILTERS["any"])
        params = [("types", "LORA"), ("limit", "30"), ("sort", SORTS.get(q.get("sort"), "Most Downloaded")),
                  ("period", q.get("period", "AllTime")), ("nsfw", "true" if q.get("nsfw", "1") == "1" else "false")]
        if q.get("q"):
            params.append(("query", q["q"]))
        if q.get("cursor"):
            params.append(("cursor", q["cursor"]))
        data = None
        last_params = params
        valid = [n for n in names if n not in BAD_BASE_NAMES]
        if valid:
            try:
                last_params = params + [("baseModels", n) for n in valid]
                data = await api("models", last_params)
            except CivError as e:
                if "HTTP 400" not in str(e):
                    return err(e, 502)
                # какое-то название базы API не знает — проверяем по одному и оставляем рабочие
                ok = []
                for n in valid:
                    try:
                        await api("models", [("types", "LORA"), ("limit", "1"), ("baseModels", n)])
                        ok.append(n)
                    except CivError as e2:
                        if "HTTP 400" in str(e2):
                            BAD_BASE_NAMES.add(n)
                if ok:
                    try:
                        last_params = params + [("baseModels", n) for n in ok]
                        data = await api("models", last_params)
                    except CivError as e3:
                        return err(e3, 502)
        if data is None:
            # фильтр по базе недоступен — ищем по ключевому слову и отбираем версии нужной базы сами
            fb = list(params)
            if base in FALLBACK_QUERY and not q.get("q"):
                fb.append(("query", FALLBACK_QUERY[base]))
            try:
                last_params = fb
                data = await api("models", fb)
            except CivError as e:
                return err(e, 502)
        # Civitai иногда подмешивает в выдачу модели, воркфлоу и прочее, хотя просили только LoRA, — отсеиваем сами.
        # Если после отсева осталось мало, догружаем следующие страницы (не больше трёх).
        def lora_only(d):
            return [c for c in (model_card(m, rx) for m in d.get("items") or []
                                if str(m.get("type") or "").lower() in LORA_TYPES) if c]
        items = lora_only(data)
        cursor = (data.get("metadata") or {}).get("nextCursor")
        extra = 0
        while len(items) < 12 and cursor and extra < 3:
            extra += 1
            nxt = [(k, v) for k, v in last_params if k != "cursor"] + [("cursor", cursor)]
            try:
                d2 = await api("models", nxt)
            except CivError:
                break
            items += lora_only(d2)
            cursor = (d2.get("metadata") or {}).get("nextCursor")
        return web.json_response({"items": items, "cursor": cursor, "domain": domain()})

    @routes.get("/api/civitai/resolve")
    async def resolve(req):
        try:
            mid, vid = parse_link(req.query.get("url"))
            model, v = await fetch_full(mid, vid)
        except CivError as e:
            return err(e)
        card = model_card(model) or version_card(model, v)
        card.update(version_card(model, v))
        card["installed"] = is_installed(primary_file(v.get("files")))
        card["description"] = model.get("description") or ""
        return web.json_response(card)

    @routes.post("/api/civitai/download")
    async def download(req):
        b = await req.json()
        try:
            if b.get("url"):
                mid, vid = parse_link(b["url"])
            else:
                mid, vid = b.get("model_id"), b.get("version_id")
            task = await start_download(mid, vid)
        except CivError as e:
            return err(e)
        return web.json_response(public_task(task))

    @routes.get("/api/loras")
    async def loras(_):
        out = []
        for f in S.list_models("loras"):
            m = read_meta(f["name"]) or {}
            pv = preview_of(f["name"])
            out.append({
                "file": f["name"], "size_h": f["size_h"],
                "title": m.get("title") or m.get("name") or "",
                "civitai_name": m.get("name") or "", "version_name": m.get("version_name") or "",
                "base_model": m.get("base_model") or "", "trained_words": triggers(f["name"], m)[0],
                "triggers_source": triggers(f["name"], m)[1],
                "arch": lora_arch(f["name"]), "arch_label": ARCH_LABEL.get(lora_arch(f["name"]), ""),
                "page": (m.get("page") if domain() == "civitai.red" else m.get("page_com")) or m.get("page") or "",
                "note": m.get("note") or "", "has_meta": bool(m.get("source")),
                "nsfw": bool(m.get("nsfw")) or ((m.get("image") or {}).get("nsfw_level") or 0) >= 4,
                "default_strength": m.get("strength"),
                "digest": {k: v for k, v in digest_of(f["name"], m).items() if k in ("made_by", "strength", "summary", "rules", "settings")},
                "needs_digest": bool((m.get("description") or m.get("examples")) and (m.get("digest") or {}).get("made_by") not in ("llm", "user")),
                "preview": ("/loras/" + quote(pv.relative_to(LORAS).as_posix()) + f"?v={int(pv.stat().st_mtime)}") if pv else "",
                "preview_type": "video" if pv and pv.suffix in (".mp4", ".webm") else "image",
            })
        return web.json_response({"items": out})

    @routes.get("/api/loras/info")
    async def lora_info(req):
        name = req.query.get("file", "")
        if not (LORAS / name).is_file():
            return err("Нет такого файла", 404)
        m = read_meta(name) or {}
        words, src = triggers(name, m)
        return web.json_response({"file": name, **m, "digest": digest_of(name, m), "triggers": words, "triggers_source": src,
                                  "file_tags": file_tags(name), "civitai_words": m.get("trained_words") or []})

    @routes.post("/api/loras/edit")
    async def lora_edit(req):
        b = await req.json()
        name = b.get("file", "")
        if not (LORAS / name).is_file():
            return err("Нет такого файла", 404)
        m = read_meta(name) or {}
        for k in ("title", "note", "strength"):
            if k in b:
                m[k] = b[k]
        if "digest" in b:  # правка выжимки вручную (None — сбросить, пусть помощник разберёт заново)
            if b["digest"] is None:
                m.pop("digest", None)
            else:
                d = dict(digest_of(name, m))
                d.update({k: v for k, v in b["digest"].items() if k in ("summary", "rules", "strength", "settings", "tips_en")})
                d["made_by"] = "user"
                m["digest"] = d
        if "trained_words" in b:  # None — вернуть слова с Civitai/из файла
            tw = b["trained_words"]
            if tw is None:
                m.pop("trained_words_custom", None)
            else:
                m["trained_words_custom"] = [w.strip() for w in tw if w and w.strip()]
        meta_path(name).write_text(json.dumps(m, ensure_ascii=False, indent=2), "utf-8")
        return web.json_response({"ok": True})

    @routes.post("/api/loras/preview_from")
    async def preview_from(req):
        """Сделать превью LoRA из своей картинки (например, из результата генерации)."""
        b = await req.json()
        name, src = b.get("file", ""), b.get("url", "")
        if not (LORAS / name).is_file() or not src.startswith("/output/"):
            return err("Неверные данные")
        from urllib.parse import unquote
        sp = (S.OUTPUT / unquote(src[len("/output/"):])).resolve()
        if not sp.is_file():
            return err("Картинка не найдена")
        old = preview_of(name)
        if old:
            old.unlink()
        p = LORAS / name
        (p.with_name(p.stem + ".preview" + sp.suffix.lower())).write_bytes(sp.read_bytes())
        return web.json_response({"ok": True})

    IDENT = {"running": False, "done": 0, "total": 0, "found": 0, "current": "", "errors": []}

    def sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    async def identify_one(name):
        loop = asyncio.get_running_loop()
        digest = await loop.run_in_executor(None, sha256, LORAS / name)
        try:
            v = await api(f"model-versions/by-hash/{digest}")
        except CivError as e:
            if "404" in str(e):
                return False
            raise
        model, v = await fetch_full(v.get("modelId"), v.get("id"))
        meta = build_meta(model, v)
        write_meta(name, meta)
        await save_preview(name, meta["image"])
        await enrich(name, meta)
        return True

    @routes.post("/api/loras/identify")
    async def identify(req):
        b = await req.json()
        if IDENT["running"]:
            return err("Уже идёт поиск")
        files = [b["file"]] if b.get("file") else [f["name"] for f in S.list_models("loras") if not (read_meta(f["name"]) or {}).get("source")]
        IDENT.update(running=True, done=0, total=len(files), found=0, current="", errors=[])

        async def run():
            try:
                for n in files:
                    IDENT["current"] = n
                    try:
                        if await identify_one(n):
                            IDENT["found"] += 1
                        else:
                            IDENT["errors"].append(f"{n}: не найдена на Civitai (возможно, удалена или изменена)")
                    except Exception as e:
                        IDENT["errors"].append(f"{n}: {e}")
                    IDENT["done"] += 1
            finally:
                IDENT["running"] = False
                IDENT["current"] = ""
        asyncio.create_task(run())
        return web.json_response({"total": len(files)})

    @routes.get("/api/loras/identify")
    async def identify_state(_):
        return web.json_response(IDENT)

    @routes.post("/api/loras/delete")
    async def lora_delete(req):
        b = await req.json()
        name = b.get("file", "")
        p = (LORAS / name).resolve()
        if not str(p).startswith(str(LORAS.resolve())) or not p.is_file():
            return err("Файл не найден")
        pv, mp = preview_of(name), meta_path(name)
        p.unlink()
        for x in (pv, mp):
            if x and x.exists():
                x.unlink()
        return web.json_response({"ok": True})

    @routes.post("/api/loras/to_model")
    async def lora_to_model(req):
        """Файл лежит в LoRA, но это целая модель — перенести в models/diffusion_models."""
        b = await req.json()
        name = b.get("file", "")
        src = (LORAS / name).resolve()
        if not str(src).startswith(str(LORAS.resolve())) or not src.is_file():
            return err("Файл не найден")
        dst_dir = S.MODELS / "diffusion_models"
        dst = dst_dir / src.name
        if dst.exists():
            return err("В папке моделей уже есть файл с таким именем")
        extra = [x for x in (preview_of(name), meta_path(name)) if x and x.exists()]
        try:
            await asyncio.get_running_loop().run_in_executor(None, shutil.move, str(src), str(dst))
            for x in extra:
                shutil.move(str(x), str(dst_dir / x.name))
        except Exception as e:
            return err("Не удалось перенести: " + str(e))
        return web.json_response({"ok": True, "model": dst.name})

    async def download_link(url):
        mid, vid = parse_link(url)
        return await start_download(mid, vid)

    return {"download_link": download_link, "lora_context": lora_context, "lora_arch": lora_arch, "arch_label": ARCH_LABEL,
            "prepare_qwen_lora": prepare_qwen_lora, "lora_has_te": lora_has_te, "fix_dir": FIX_DIR,
            "read_meta": read_meta, "update_meta": update_meta, "auto_digest": auto_digest, "digest_of": digest_of,
            "html_to_text": html_to_text, "triggers": triggers, "file_tags": file_tags, "enrich": enrich,
            "hooks": HOOKS, "lora_arch": lora_arch}
