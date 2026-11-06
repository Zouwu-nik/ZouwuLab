"""Сборка графов (API-формат ComfyUI): Qwen-Image 2.1 (картинки, правка) и MiniMax H3 (видео со звуком)."""
import re


class Graph:
    def __init__(self):
        self.nodes = {}
        self._n = 0

    def add(self, class_type, **inputs):
        self._n += 1
        nid = str(self._n)
        self.nodes[nid] = {"class_type": class_type, "inputs": inputs}
        return nid


def build(p: dict) -> dict:
    fam = p.get("family")
    if fam == "minimax":
        return build_minimax(p)
    if fam == "ltx":
        return build_ltx(p)
    if fam == "krea2":
        return build_krea2(p)
    if fam == "3d":
        return build_3d(p)
    if fam == "flux3":
        return build_flux3(p)
    return build_qwen(p)


def build_edit(p: dict) -> dict:
    return build_edit_qwen(p)


# ============================================================ QWEN-IMAGE 2.1
# Схема повторяет официальные шаблоны ComfyUI (image_qwen_image_2_1_t2i / _image_edit):
# UNETLoader → (LoRA) → KSampler(euler/simple, cfg 1); CLIPLoader(type=qwen_image, Qwen3-VL 8B);
# TextEncodeQwenImage21 — текст + до 10 картинок-референсов (<image1>, <image2>…), сам кодирует их через VAE.
QWEN_MAX_REFS = 10


def _snap32(v):
    v = int(round(float(v) / 32.0)) * 32
    return max(256, min(v, 4096))


WEIGHT_DTYPES = ("default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2")


def unet_loader(g, p):
    """Файл .gguf грузится узлом ComfyUI-GGUF, остальные — штатным UNETLoader."""
    if str(p["unet"]).lower().endswith(".gguf"):
        return g.add("UnetLoaderGGUF", unet_name=p["unet"])
    return g.add("UNETLoader", unet_name=p["unet"], weight_dtype=p.get("weight_dtype") if p.get("weight_dtype") in WEIGHT_DTYPES else "default")


def _qwen_loaders(g, p):
    unet = unet_loader(g, p)
    clip = g.add("CLIPLoader", clip_name=p["clip"], type="qwen_image", device=p.get("clip_device", "default"))
    vae = g.add("VAELoader", vae_name=p["vae"])
    model, clip = [unet, 0], [clip, 0]
    for lora in p.get("loras", []):
        if not lora.get("enabled", True) or not lora.get("name"):
            continue
        st = float(lora.get("strength", 1.0))
        if lora.get("te"):  # LoRA обучала и текстовый энкодер — подключаем её к обоим
            nid = g.add("LoraLoader", model=model, clip=clip, lora_name=lora["name"],
                        strength_model=st, strength_clip=st)
            model, clip = [nid, 0], [nid, 1]
        else:
            nid = g.add("LoraLoaderModelOnly", model=model, lora_name=lora["name"], strength_model=st)
            model = [nid, 0]
    return model, clip, vae


def _qwen_encode(g, clip, vae, prompt, negative, images, resolution):
    kw = {f"images.image_{i + 1}": [g.add("LoadImage", image=name), 0] for i, name in enumerate(images[:QWEN_MAX_REFS])}
    return g.add("TextEncodeQwenImage21", clip=clip, prompt=prompt, negative_prompt=negative or "",
                 vae=[vae, 0], resolution=int(resolution), **kw)


# Qwen-Image-2.1-Turbo: 8 шагов, CFG 1, собственное расписание шума из модели (Kijai: близко к linear_quadratic)
QWEN_TURBO_SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0"


def _qwen_sample(g, p, model, enc, latent, vae, denoise, prefix):
    steps = max(1, int(p["steps"]))
    cfg = float(p.get("cfg", 1.0))
    sched = p.get("scheduler", "simple")
    if sched == "flux2":
        sched = "simple"
    if p.get("turbo_sigmas") and float(denoise) >= 0.999:
        sigmas = g.add("ManualSigmas", sigmas=QWEN_TURBO_SIGMAS)
        smp = g.add("KSamplerSelect", sampler_name=p.get("sampler", "euler"))
        ks = g.add("SamplerCustom", model=model, add_noise=True, noise_seed=int(p["seed"]), cfg=cfg,
                   positive=[enc, 0], negative=[enc, 1], sampler=[smp, 0], sigmas=[sigmas, 0], latent_image=latent)
    else:
        if p.get("turbo_sigmas"):
            sched = "linear_quadratic"
        ks = g.add("KSampler", model=model, seed=int(p["seed"]), steps=steps, cfg=cfg,
                   sampler_name=p.get("sampler", "euler"), scheduler=sched,
                   positive=[enc, 0], negative=[enc, 1], latent_image=latent, denoise=float(denoise))
    dec = g.add("VAEDecode", samples=[ks, 0], vae=[vae, 0])
    g.add("SaveImage", images=[dec, 0], filename_prefix=prefix)


def build_qwen(p: dict) -> dict:
    g = Graph()
    width, height = _snap32(p["width"]), _snap32(p["height"])
    batch = max(1, min(int(p.get("batch", 1)), 16))
    model, clip, vae = _qwen_loaders(g, p)
    use_text = bool(p.get("use_text", True))
    prompt = p.get("prompt", "") if use_text else ""
    negative = (p.get("negative") or "") if use_text else ""

    use_image = bool(p.get("use_image")) and bool(p.get("images"))
    mode = p.get("image_mode", "reference")
    refs = []
    latent, denoise = None, 1.0
    if use_image and mode == "reference":
        refs += list(p["images"][:4])
    elif use_image and mode == "img2img":
        img = g.add("LoadImage", image=p["images"][0])
        scaled = g.add("ImageScale", image=[img, 0], upscale_method="lanczos", width=width, height=height, crop="center")
        latent = [g.add("VAEEncode", pixels=[scaled, 0], vae=[vae, 0]), 0]
        if batch > 1:
            latent = [g.add("RepeatLatentBatch", samples=latent, amount=batch), 0]
        denoise = max(0.01, min(float(p.get("denoise", 0.6)), 1.0))
    refs += list(p.get("char_refs") or [])[:4]

    enc = _qwen_encode(g, clip, vae, prompt, negative, refs, p.get("ref_resolution", 1024))
    if latent is None:
        latent = [g.add("EmptyLatentImage", width=width, height=height, batch_size=batch), 0]
    _qwen_sample(g, p, model, enc, latent, vae, denoise, p.get("filename_prefix", "images/qwen"))
    return g.nodes


def build_edit_qwen(p: dict) -> dict:
    """Правка Qwen-Image 2.1: <image1> — картинка для правки (сервер уже подогнал её размер под кратный 32),
    дальше — фото персонажа. Без маски холст берётся из самого кодировщика (размер <image1>),
    с маской — латент <image1> + SetLatentNoiseMask, т.е. меняется только область маски."""
    g = Graph()
    model, clip, vae = _qwen_loaders(g, p)
    refs = [p.get("ref_src") or p["src"]] + list(p.get("char_refs") or [])[:4]
    enc = _qwen_encode(g, clip, vae, p.get("instruction", ""), p.get("negative") or "", refs, 0)
    denoise = 1.0
    if p.get("mask"):
        img = g.add("LoadImage", image=p["src"])
        lat = g.add("VAEEncode", pixels=[img, 0], vae=[vae, 0])
        m = g.add("LoadImageMask", image=p["mask"], channel="red")
        latent = [g.add("SetLatentNoiseMask", samples=[lat, 0], mask=[m, 0]), 0]
        denoise = max(0.05, min(float(p.get("denoise", 1.0)), 1.0))
    else:
        latent = [enc, 2]
    _qwen_sample(g, p, model, enc, latent, vae, denoise, "edit_tmp/edit")
    return g.nodes


# ============================================================ MINIMAX H3 (видео + звук)
# Как официальный шаблон ComfyUI «Image to Video (MiniMax H3)»:
# UNETLoader → (турбо-LoRA) → CLIPLoader(type=minimax, Qwen3-VL 32B) → MiniMaxH3ImageToVideo (текст + первый/последний кадр)
# → RandomNoise + KSamplerSelect(res_multistep) + BasicScheduler(simple) + BasicGuider → SamplerCustomAdvanced
# → VAEDecode (кадры) + VAEDecodeAudio (звук) → CreateVideo(24 fps) → SaveVideo (mp4).
H3_FPS = 24
H3_MAX_PIXELS = 768 * 1344


def h3_canvas(width, height):
    """Размер кадра H3: кратно 32, площадь не больше 768×1344 (как в узле MiniMaxH3ImageToVideo)."""
    w, h = max(64, float(width)), max(64, float(height))
    if w * h > H3_MAX_PIXELS:
        k = (H3_MAX_PIXELS / (w * h)) ** 0.5
        w, h = w * k, h * k
    return max(256, int(round(w / 32.0)) * 32), max(256, int(round(h / 32.0)) * 32)


def h3_auto_canvas(width, height):
    """Холст под пропорции картинки: короткая сторона 768, площадь не больше 768×1344 (родной размер H3)."""
    r = float(width) / float(height)
    w, h = (768 * r, 768) if r >= 1 else (768, 768 / r)
    return h3_canvas(w, h)


def h3_frames(seconds):
    """Длительность → число кадров на сетке модели 17k+5 (124 ≈ 5 с, 362 ≈ 15 с)."""
    n = max(5, int(round(float(seconds) * H3_FPS)))
    while n % 17 != 5:
        n += 1
    return n


def build_minimax(p: dict) -> dict:
    g = Graph()
    width, height = h3_canvas(p["width"], p["height"])
    length = h3_frames(p.get("duration", 5))
    unet = unet_loader(g, p)
    model = [unet, 0]
    if p.get("turbo_lora"):
        model = [g.add("LoraLoaderModelOnly", model=model, lora_name=p["turbo_lora"],
                       strength_model=float(p.get("turbo_strength", 1.0))), 0]
    for lora in p.get("loras", []):
        if not lora.get("enabled", True) or not lora.get("name") or lora["name"] == p.get("turbo_lora"):
            continue
        model = [g.add("LoraLoaderModelOnly", model=model, lora_name=lora["name"],
                       strength_model=float(lora.get("strength", 1.0))), 0]
    clip = g.add("CLIPLoader", clip_name=p["clip"], type="minimax", device=p.get("clip_device", "default"))
    vae = g.add("VAELoader", vae_name=p["vae"])
    avae = g.add("VAELoader", vae_name=p["avae"])
    if p.get("refs"):
        # Ref2VA: до 9 картинок-референсов, в промпте они — <Picture 1>…<Picture N> (порядок = порядок входов)
        kw = {f"ref_images.ref_image_{i}": [g.add("LoadImage", image=name), 0]
              for i, name in enumerate(p["refs"][:9])}
        i2v = g.add("MiniMaxH3ReferenceToVideo", clip=[clip, 0], vae=[vae, 0], audio_vae=[avae, 0],
                    prompt=p.get("prompt", ""), width=width, height=height, length=length,
                    ref_image_size="max" if p.get("ref_max") else "match", **kw)
    else:
        kw = {}
        if p.get("first_frame"):
            kw["first_frame"] = [g.add("LoadImage", image=p["first_frame"]), 0]
        if p.get("last_frame"):
            kw["last_frame"] = [g.add("LoadImage", image=p["last_frame"]), 0]
        i2v = g.add("MiniMaxH3ImageToVideo", clip=[clip, 0], vae=[vae, 0], prompt=p.get("prompt", ""),
                    width=width, height=height, length=length, **kw)
    noise = g.add("RandomNoise", noise_seed=int(p["seed"]))
    sampler = g.add("KSamplerSelect", sampler_name="res_multistep")   # официальные настройки H3
    sigmas = g.add("BasicScheduler", model=model, scheduler="simple",
                   steps=max(1, int(p["steps"])), denoise=1.0)
    guider = g.add("BasicGuider", model=model, conditioning=[i2v, 0])
    sample = g.add("SamplerCustomAdvanced", noise=[noise, 0], guider=[guider, 0], sampler=[sampler, 0],
                   sigmas=[sigmas, 0], latent_image=[i2v, 1])
    frames = g.add("VAEDecode", samples=[sample, 0], vae=[vae, 0])
    audio = g.add("VAEDecodeAudio", samples=[sample, 0], vae=[avae, 0])
    video = g.add("CreateVideo", images=[frames, 0], fps=float(H3_FPS), audio=[audio, 0])
    g.add("SaveVideo", video=[video, 0], filename_prefix=p.get("filename_prefix", "video/h3"), format="mp4",
          **{"format.codec": "auto"})
    return g.nodes


# ============================================================ KREA 2 (картинки)
# Как официальные шаблоны ComfyUI «Text to Image (Krea-2 Turbo)» и «Image Style Reference (Krea-2 Turbo)»:
# UNETLoader → (LoRA) → KSampler(8 шагов, CFG 1, euler/simple); CLIPLoader(type=krea2, Qwen3-VL 4B); VAE Qwen-Image.
# Негатив — обнулённый позитив (у турбо CFG 1). Референсы стиля: TextEncodeQwenImageEditPlus (до 3 картинок)
# + FluxKontextMultiReferenceLatentMethod(index_timestep_zero) + ModelSamplingFlux + LoRA krea2_style_reference.
def _loras_model_only(g, model, loras, skip=()):
    for lora in loras or []:
        if not lora.get("enabled", True) or not lora.get("name") or lora["name"] in skip:
            continue
        model = [g.add("LoraLoaderModelOnly", model=model, lora_name=lora["name"],
                       strength_model=float(lora.get("strength", 1.0))), 0]
    return model


def build_krea2(p: dict) -> dict:
    g = Graph()
    width, height = _snap32(p["width"]), _snap32(p["height"])
    batch = max(1, min(int(p.get("batch", 1)), 16))
    model = [unet_loader(g, p), 0]
    if p.get("style_lora"):
        model = [g.add("LoraLoaderModelOnly", model=model, lora_name=p["style_lora"],
                       strength_model=float(p.get("style_strength", 1.0))), 0]
    model = _loras_model_only(g, model, p.get("loras"), skip=(p.get("style_lora"),))
    clip = [g.add("CLIPLoader", clip_name=p["clip"], type="krea2", device=p.get("clip_device", "default")), 0]
    vae = g.add("VAELoader", vae_name=p["vae"])
    prompt = p.get("prompt", "") if p.get("use_text", True) else ""
    use_image = bool(p.get("use_image")) and bool(p.get("images"))
    mode = p.get("image_mode", "reference")
    latent, denoise = None, 1.0
    if use_image and mode == "reference":
        refs = {f"image{i + 1}": [g.add("LoadImage", image=name), 0] for i, name in enumerate(p["images"][:3])}
        pos = g.add("TextEncodeQwenImageEditPlus", clip=clip, prompt=prompt, vae=[vae, 0], **refs)
        pos = g.add("FluxKontextMultiReferenceLatentMethod", conditioning=[pos, 0], reference_latents_method="index_timestep_zero")
        model = [g.add("ModelSamplingFlux", model=model, max_shift=1.15, base_shift=0.5, width=width, height=height), 0]
    else:
        pos = g.add("CLIPTextEncode", clip=clip, text=prompt)
        if use_image and mode == "img2img":
            img = g.add("LoadImage", image=p["images"][0])
            scaled = g.add("ImageScale", image=[img, 0], upscale_method="lanczos", width=width, height=height, crop="center")
            latent = [g.add("VAEEncode", pixels=[scaled, 0], vae=[vae, 0]), 0]
            if batch > 1:
                latent = [g.add("RepeatLatentBatch", samples=latent, amount=batch), 0]
            denoise = max(0.01, min(float(p.get("denoise", 0.6)), 1.0))
    neg_text = (p.get("negative") or "").strip()
    if float(p.get("cfg", 1.0)) > 1.01 and neg_text and not (use_image and mode == "reference"):
        neg = g.add("CLIPTextEncode", clip=clip, text=neg_text)   # Raw с CFG: настоящий негативный промпт
    else:
        neg = g.add("ConditioningZeroOut", conditioning=[pos, 0])
    if latent is None:
        latent = [g.add("EmptyLatentImage", width=width, height=height, batch_size=batch), 0]
    ks = g.add("KSampler", model=model, seed=int(p["seed"]), steps=max(1, int(p["steps"])), cfg=float(p.get("cfg", 1.0)),
               sampler_name=p.get("sampler", "euler"), scheduler=p.get("scheduler", "simple") if p.get("scheduler") != "flux2" else "simple",
               positive=[pos, 0], negative=[neg, 0], latent_image=latent, denoise=float(denoise))
    dec = g.add("VAEDecode", samples=[ks, 0], vae=[vae, 0])
    g.add("SaveImage", images=[dec, 0], filename_prefix=p.get("filename_prefix", "images/krea2"))
    return g.nodes


# ============================================================ LTX-2.5 (видео со звуком)
# Как официальные шаблоны ComfyUI «Text/Image to Video (LTX-2.5)» и «First & Last Frame to Video (LTX-2.5)».
# Дистиллированная модель: фиксированное расписание (8 шагов), CFG 1, euler_ancestral.
# Двухэтапная схема с апскейлом: этап 1 — половинное разрешение; затем латентный апскейлер ×2 и короткая
# доводка (3 шага). Для i2v исходная картинка вшивается в первый кадр ДВАЖДЫ: на этапе 1 (сила ~0.7)
# и после апскейла (сила 1.0 — «упор на исходник»), поэтому детали исходника не теряются при увеличении.
LTX_FPS = 24
LTX_SIGMAS_1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
LTX_SIGMAS_2 = "0.85, 0.7250, 0.4219, 0.0"
LTX_NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly"


def ltx_frames(seconds):
    """Длительность → число кадров 8k+1 (121 ≈ 5 с при 24 к/с)."""
    n = max(9, int(round(float(seconds) * LTX_FPS)) + 1)
    while n % 8 != 1:
        n += 1
    return n


def ltx_canvas(width, height, two_stage=True):
    """Кратно 64 при двух этапах (половина — кратна 32), иначе кратно 32."""
    m = 64 if two_stage else 32
    w = max(m * 4, int(round(float(width) / m)) * m)
    h = max(m * 4, int(round(float(height) / m)) * m)
    return w, h


def build_ltx(p: dict) -> dict:
    g = Graph()
    first, last = p.get("first_frame"), p.get("last_frame")
    guides = bool(last)                                   # последний кадр (или оба) — через направляющие кадры
    two = bool(p.get("two_stage", True)) and bool(p.get("upscaler")) and not guides
    W, H = ltx_canvas(p["width"], p["height"], two)
    length = ltx_frames(p.get("duration", 5))
    model = [unet_loader(g, p), 0]
    model = _loras_model_only(g, model, p.get("loras"))
    clip = [g.add("CLIPLoader", clip_name=p["clip"], type="ltxv", device=p.get("clip_device", "default")), 0]
    vae = g.add("VAELoader", vae_name=p["vae"])
    avae = g.add("VAELoader", vae_name=p["avae"])
    pos = g.add("CLIPTextEncode", clip=clip, text=p.get("prompt", ""))
    neg = g.add("CLIPTextEncode", clip=clip, text=(p.get("negative") or "").strip() or LTX_NEGATIVE)
    cond = g.add("LTXVConditioning", positive=[pos, 0], negative=[neg, 0], frame_rate=float(LTX_FPS))
    positive, negative = [cond, 0], [cond, 1]
    W1, H1 = (W // 2, H // 2) if two else (W, H)
    vlat = [g.add("EmptyLTXVLatentVideo", width=W1, height=H1, length=length, batch_size=1), 0]
    alat = [g.add("LTXVEmptyLatentAudio", frames_number=length, frame_rate=LTX_FPS, batch_size=1, audio_vae=[avae, 0]), 0]

    def prep(name):
        img = g.add("LoadImage", image=name)
        sc = g.add("ImageScale", image=[img, 0], upscale_method="lanczos", width=W, height=H, crop="center")
        return [g.add("LTXVPreprocess", image=[sc, 0], img_compression=int(p.get("img_compression", 18))), 0]

    src = prep(first) if first else None
    s_strength = max(0.0, min(1.0, float(p.get("src_strength", 0.7))))
    if guides:
        lt = vlat
        if first:
            ag = g.add("LTXVAddGuide", positive=positive, negative=negative, vae=[vae, 0], latent=lt, image=src,
                       frame_idx=0, strength=s_strength)
            positive, negative, lt = [ag, 0], [ag, 1], [ag, 2]
        ag = g.add("LTXVAddGuide", positive=positive, negative=negative, vae=[vae, 0], latent=lt, image=prep(last),
                   frame_idx=-1, strength=s_strength)
        positive, negative, vlat = [ag, 0], [ag, 1], [ag, 2]
    elif src:
        vlat = [g.add("LTXVImgToVideoInplace", vae=[vae, 0], image=src, latent=vlat, strength=s_strength, bypass=False), 0]
    guider = g.add("LTXVDualCFGGuider", model=model, positive=positive, negative=negative,
                   video_cfg=float(p.get("cfg", 1.0)), audio_cfg=float(p.get("cfg", 1.0)))
    sampler = g.add("KSamplerSelect", sampler_name="euler_ancestral")
    av = g.add("LTXVConcatAVLatent", video_latent=vlat, audio_latent=alat)
    s1 = g.add("SamplerCustomAdvanced", noise=[g.add("RandomNoise", noise_seed=int(p["seed"])), 0], guider=[guider, 0],
               sampler=[sampler, 0], sigmas=[g.add("ManualSigmas", sigmas=LTX_SIGMAS_1), 0], latent_image=[av, 0])
    sep = g.add("LTXVSeparateAVLatent", av_latent=[s1, 0])
    if two:
        up = [g.add("LTXVLatentUpsampler", samples=[sep, 0], vae=[vae, 0],
                    upscale_model=[g.add("LatentUpscaleModelLoader", model_name=p["upscaler"]), 0]), 0]
        if src:   # упор на исходник: после увеличения первый кадр снова «пришивается» к картинке
            up = [g.add("LTXVImgToVideoInplace", vae=[vae, 0], image=src, latent=up,
                        strength=max(0.0, min(1.0, float(p.get("anchor", 1.0)))), bypass=False), 0]
        av2 = g.add("LTXVConcatAVLatent", video_latent=up, audio_latent=[sep, 1])
        s2 = g.add("SamplerCustomAdvanced", noise=[g.add("RandomNoise", noise_seed=int(p["seed"]) + 1), 0], guider=[guider, 0],
                   sampler=[sampler, 0], sigmas=[g.add("ManualSigmas", sigmas=LTX_SIGMAS_2), 0], latent_image=[av2, 0])
        sep = g.add("LTXVSeparateAVLatent", av_latent=[s2, 0])
    video_lat = [sep, 0]
    if guides:
        video_lat = [g.add("LTXVCropGuides", positive=positive, negative=negative, latent=video_lat), 2]
    frames = g.add("VAEDecodeTiled", samples=video_lat, vae=[vae, 0], tile_size=512, overlap=64, temporal_size=64, temporal_overlap=16)
    audio = g.add("LTXVAudioVAEDecode", samples=[sep, 1], audio_vae=[avae, 0])
    video = g.add("CreateVideo", images=[frames, 0], fps=float(LTX_FPS), audio=[audio, 0])
    g.add("SaveVideo", video=[video, 0], filename_prefix=p.get("filename_prefix", "video/ltx"), format="mp4",
          **{"format.codec": "auto"})
    return g.nodes


# ============================================================ HUNYUAN3D 2 (3D-модель по картинкам)
# Как официальные шаблоны ComfyUI «Image to Model (Hunyuan3D 2.1)» и «Hunyuan3D 2mv»: ImageOnlyCheckpointLoader (модель +
# CLIP Vision + VAE в одном файле) → CLIPVisionEncode каждого ракурса → Hunyuan3Dv2ConditioningMultiView (спереди / слева /
# сзади / справа) или Hunyuan3Dv2Conditioning (одна картинка) → KSampler → VAEDecodeHunyuan3D → VoxelToMesh → GLB.
H3D_VIEWS = ("front", "left", "back", "right")


def is_multiview_3d(name):
    """Модель Hunyuan3D‑2mv (несколько ракурсов) — по имени файла."""
    return bool(re.search(r"(^|[-_.])mv([-_.]|$)", str(name or "").lower().replace("\\", "/").split("/")[-1]))


def build_3d(p: dict) -> dict:
    g = Graph()
    ck = g.add("ImageOnlyCheckpointLoader", ckpt_name=p["model"])
    views = {k: v for k, v in (p.get("views") or {}).items() if k in H3D_VIEWS and v}
    enc = {k: [g.add("CLIPVisionEncode", clip_vision=[ck, 1], image=[g.add("LoadImage", image=v), 0], crop="center"), 0]
           for k, v in views.items()}
    if p.get("multiview"):
        cond = g.add("Hunyuan3Dv2ConditioningMultiView", **enc)
    else:
        cond = g.add("Hunyuan3Dv2Conditioning", clip_vision_output=enc.get("front") or next(iter(enc.values())))
    model = g.add("ModelSamplingAuraFlow", model=[ck, 0], shift=1.0)
    latent = g.add("EmptyLatentHunyuan3Dv2", resolution=int(p.get("latent_res", 4096)), batch_size=1)
    ks = g.add("KSampler", model=[model, 0], seed=int(p["seed"]), steps=max(1, int(p.get("steps", 30))),
               cfg=float(p.get("cfg", 5.0)), sampler_name="euler", scheduler="normal",
               positive=[cond, 0], negative=[cond, 1], latent_image=[latent, 0], denoise=1.0)
    vox = g.add("VAEDecodeHunyuan3D", samples=[ks, 0], vae=[ck, 2], num_chunks=int(p.get("num_chunks", 8000)),
                octree_resolution=int(p.get("octree", 256)))
    mesh = g.add("VoxelToMesh", voxel=[vox, 0], algorithm="surface net", threshold=float(p.get("threshold", 0.6)))
    mesh = g.add("MeshSmoothNormals", mesh=[mesh, 0], crease_angle=float(p.get("crease", 180.0)))
    g.add("SaveGLB", mesh=[mesh, 0], filename_prefix=p.get("filename_prefix", "mesh/zl3d"))
    return g.nodes


# ============================================================ FLUX.3 Image Klein (картинки и правка, Black Forest Labs)
# Как официальные шаблоны ComfyUI «image_flux3_klein_*»: UNETLoader → KSampler; CLIPLoader(type=flux3, Qwen3‑VL 4B);
# VAE от FLUX.2; латент EmptyFlux2LatentImage. Distilled — 16 шагов, CFG 1; Base — 50 шагов, CFG 3.5.
# Правка и референсы: ReferenceMultiLatent (картинки — <ref_image_0>, <ref_image_1>… в промпте), негатив — обнулённый позитив.
def build_flux3(p: dict) -> dict:
    g = Graph()
    width, height = _snap32(p["width"]), _snap32(p["height"])
    batch = max(1, min(int(p.get("batch", 1)), 16))
    model = _loras_model_only(g, [unet_loader(g, p), 0], p.get("loras"))
    clip = [g.add("CLIPLoader", clip_name=p["clip"], type="flux3", device=p.get("clip_device", "default")), 0]
    vae = g.add("VAELoader", vae_name=p["vae"])
    prompt = p.get("prompt", "") if p.get("use_text", True) else ""
    cfg = float(p.get("cfg", 1.0))
    pos = [g.add("CLIPTextEncode", clip=clip, text=prompt), 0]
    if cfg > 1.01:
        neg = [g.add("CLIPTextEncode", clip=clip, text=(p.get("negative") or "").strip()), 0]
    else:
        neg = [g.add("ConditioningZeroOut", conditioning=pos), 0]
    use_image = bool(p.get("use_image")) and bool(p.get("images"))
    mode = p.get("image_mode", "reference")
    latent, denoise = None, 1.0
    if use_image and mode == "reference":
        imgs = {f"images.image{i + 1}": [g.add("LoadImage", image=n), 0] for i, n in enumerate(p["images"][:8])}
        rml = g.add("ReferenceMultiLatent", positive=pos, negative=neg, vae=[vae, 0], resolution="resize",
                    **{"resolution.megapixels": float(p.get("ref_megapixels", 1.0))}, **imgs)
        pos, neg = [rml, 0], [rml, 1]
    elif use_image and mode == "img2img":
        img = g.add("LoadImage", image=p["images"][0])
        sc = g.add("ImageScale", image=[img, 0], upscale_method="lanczos", width=width, height=height, crop="center")
        latent = [g.add("VAEEncode", pixels=[sc, 0], vae=[vae, 0]), 0]
        if batch > 1:
            latent = [g.add("RepeatLatentBatch", samples=latent, amount=batch), 0]
        denoise = max(0.01, min(float(p.get("denoise", 0.6)), 1.0))
    if latent is None:
        latent = [g.add("EmptyFlux2LatentImage", width=width, height=height, batch_size=batch), 0]
    sched = p.get("scheduler", "simple")
    ks = g.add("KSampler", model=model, seed=int(p["seed"]), steps=max(1, int(p["steps"])), cfg=cfg,
               sampler_name=p.get("sampler", "euler"), scheduler="simple" if sched == "flux2" else sched,
               positive=pos, negative=neg, latent_image=latent, denoise=float(denoise))
    dec = g.add("VAEDecode", samples=[ks, 0], vae=[vae, 0])
    g.add("SaveImage", images=[dec, 0], filename_prefix=p.get("filename_prefix", "images/flux3"))
    return g.nodes


# ============================================================ МУЗЫКА (вкладка «Музыка»)
# Как официальные шаблоны ComfyUI: «Text to Audio (ACE-Step 1.5)», «Text to Music (MiniMax Music 3)»,
# «Text to Music (YuE2)», «Audio Generation (Stable Audio 3 Medium)». Результат — MP3 в output/music.
ACE_KEYS = [f"{r} {q}" for q in ("major", "minor") for r in ("C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B")]


def build_music(p: dict) -> dict:
    g = Graph()
    eng = p["engine"]
    seed = int(p["seed"])
    secs = float(p.get("duration", 120))
    tags, lyrics = (p.get("tags") or "").strip(), (p.get("lyrics") or "").strip()
    if eng == "ace":
        unet = g.add("UNETLoader", unet_name=p["model"], weight_dtype="default")
        clip = g.add("DualCLIPLoader", clip_name1=p["clip1"], clip_name2=p["clip2"], type="ace", device="default")
        vae = g.add("VAELoader", vae_name=p["vae"])
        key = p.get("keyscale") if p.get("keyscale") in ACE_KEYS else "C major"
        enc = g.add("TextEncodeAceStepAudio1.5", clip=[clip, 0], tags=tags, lyrics=lyrics or "[Instrumental]", seed=seed,
                    bpm=int(p.get("bpm", 120)), duration=secs, timesignature=str(p.get("timesig", "4")),
                    language=p.get("language", "en"), keyscale=key, generate_audio_codes=True,
                    cfg_scale=2.0, temperature=0.85, top_p=0.9, top_k=0, min_p=0.0)
        model = [g.add("ModelSamplingAuraFlow", model=[unet, 0], shift=3.0), 0]
        latent = [g.add("EmptyAceStep1.5LatentAudio", seconds=secs, batch_size=1), 0]
        pos, sampler, sched = [enc, 0], "euler", "simple"
    elif eng == "mm3":
        model = [g.add("UNETLoader", unet_name=p["model"], weight_dtype="default"), 0]
        clip = g.add("CLIPLoader", clip_name=p["clip1"], type="minimax", device="default")
        vae = g.add("VAELoader", vae_name=p["vae"])
        enc = g.add("MiniMaxMusic3TextEncode", clip=[clip, 0], caption=tags, lyrics=lyrics, seed=seed, max_duration=secs,
                    cfg_scale=1.7, top_k=50)
        latent = [g.add("EmptyMiniMaxMusic3LatentAudio", seconds=[enc, 1], batch_size=1), 0]
        pos, sampler, sched = [enc, 0], "euler", "simple"
    elif eng == "yue2":
        ck = g.add("CheckpointLoaderSimple", ckpt_name=p["model"])
        model, vae = [ck, 0], ck
        enc = g.add("YuE2GenerateMusic", clip=[ck, 1], style=tags, lyrics=lyrics, abc="", seed=seed, mode="full",
                    max_duration=secs, temperature=1.0, top_p=0.95, top_k=100, repetition_penalty=1.2)
        latent = [g.add("EmptyYuE2LatentAudio", seconds=[enc, 1], batch_size=1), 0]
        pos, sampler, sched = [enc, 0], "dpm_2", "sgm_uniform"
    else:  # sa3 — Stable Audio 3: инструментал, звуки
        ck = g.add("CheckpointLoaderSimple", ckpt_name=p["model"])
        model, vae = [ck, 0], ck
        clip = g.add("CLIPLoader", clip_name=p["clip1"], type="stable_audio", device="default")
        pos = [g.add("CLIPTextEncode", clip=[clip, 0], text=tags), 0]
        neg_sa = [g.add("CLIPTextEncode", clip=[clip, 0], text=""), 0]
        latent = [g.add("EmptyLatentAudio", seconds=secs, batch_size=1), 0]
        sampler, sched = "lcm", "simple"
    neg = neg_sa if eng == "sa3" else [g.add("ConditioningZeroOut", conditioning=pos), 0]
    ks = g.add("KSampler", model=model, seed=seed, steps=max(1, int(p["steps"])), cfg=float(p["cfg"]),
               sampler_name=sampler, scheduler=sched, positive=pos, negative=neg, latent_image=latent, denoise=1.0)
    audio = g.add("VAEDecodeAudio", samples=[ks, 0], vae=[vae, 2 if eng in ("yue2", "sa3") else 0])
    g.add("SaveAudioMP3", audio=[audio, 0], filename_prefix=p.get("filename_prefix", f"music/{eng}"), quality="V0")
    return g.nodes
