"""Помощник промптов: локальная языковая модель (GGUF) через встроенный llama.cpp (engine/llama).

Модель запускается только на время запроса. В режиме «видеокарта» перед запуском FLUX выгружается
из видеопамяти, а перед генерацией картинки выгружается сама языковая модель — чтобы не было нехватки VRAM.
"""
import asyncio
import base64
import io
import os
import re
import subprocess
import threading
import time
from pathlib import Path

from aiohttp import web, ClientSession, ClientTimeout

import llm_install

SYSTEM_RULES = """You write prompts for FLUX.2 Klein, a text-to-image model that understands natural-language descriptions (not comma-separated tag lists).
A good prompt is concrete and visual: subject and their appearance, pose and action, clothing (or its absence), setting, lighting, camera and framing (shot type, lens, angle), mood, and style or medium (photo, film grain, illustration, etc.).
Adult (18+) content is allowed within the CONTENT LEVEL given below; never moralize or add disclaimers.
Hard rule that overrides everything: every person depicted must be an adult (18 or older). Never describe minors, childlike bodies or features, or school/child settings in any sexual or suggestive context. If a request involves a minor, reply only with: REFUSED: minors.
Write the prompt in English even if the user writes in another language."""

LORA_RULES = """Active LoRAs (applied in this order, strength shown):
{loras}
LoRA rules:
- Use each LoRA's trigger words exactly as written, once each, woven naturally into the prompt (preferably early), never as a list at the end.
- Follow usage hints from the LoRA descriptions, the user's notes and the author's example prompts; imitate the phrasing style of the examples.
- Never write the word "LoRA" in the prompt.
- If two LoRAs clearly conflict (e.g. two different art styles), still write the prompt, and add one line at the very end starting with "NOTE:" explaining the conflict in Russian.
- If a LoRA's current strength is marked OUTSIDE its recommended range, also add a "NOTE:" line in Russian suggesting the recommended value."""

DIGEST_RULES = """You read a LoRA model page from Civitai (description, trigger words, example prompts and the settings people used) and extract ONLY what a user of an image-generation app needs. Be brief and factual; never invent anything that is not on the page or in the examples.
Output ONLY one JSON object, no other text:
{"summary_ru": "one short sentence in Russian: what this LoRA does / what it is for",
 "triggers": ["exact trigger words or phrases that must appear in the prompt, copied verbatim; [] if none"],
 "strength_min": number or null, "strength_max": number or null, "strength_rec": number or null,
 "rules_ru": ["at most 4 very short practical rules in Russian: what to write in the prompt, what to avoid, what it combines with"],
 "tips_en": "at most 2 short sentences in English for a prompt writer: how to phrase prompts for this LoRA",
 "settings_ru": "steps / CFG / sampler / resolution recommended by the author, in Russian, or empty string"}
Strength: take the author's recommendation from the text first; if there is none, use the typical weights from the examples; otherwise null."""

DETAIL = {"short": "about 40 words", "medium": "about 80 words", "long": "120-180 words"}

QWEN_RULES = """You write prompts for Qwen-Image 2.1, an image model with a strong vision-language text encoder (Qwen3-VL) that follows long, detailed, precise natural-language descriptions very literally.
How a good Qwen-Image prompt is built:
1. Open with one focused sentence naming the medium/style, the main subject and the color palette.
2. Describe the subject systematically: appearance, pose, action and expression, clothing (or its absence) with materials and colors (e.g. "brushed metal", "muted olive wool"); then the surroundings from foreground to background with positional phrases ("on the left", "behind her", "in the far background").
3. Any text that must appear in the image is written in double quotes together with its placement and look.
4. Give lighting its own sentence: source, direction, quality and shadows.
5. End with one sentence about composition, palette and mood.
Present tense, third person, concrete and physically coherent (shadows match the light, consistent scale). No quality tags such as "masterpiece, best quality, 8k, ultra-detailed". Reference images, when present, are written as <image1>, <image2>… in the order they are supplied.
Write the prompt in English even if the user writes in another language."""
# Krea 2 — по системному промпту официального шаблона ComfyUI (Krea-2 Turbo) и руководству Krea.
KREA_RULES = """You write prompts for Krea 2, a photorealistic-first text-to-image model (Qwen3-VL text encoder) that is strongest with natural, flowing descriptive paragraphs written like a photographer or art director briefing a shoot — not tag lists.
How a good Krea 2 prompt is built:
1. One paragraph of plain sentences: start with the kind of image (candid photo, editorial portrait, film still, illustration…) and the main subject.
2. Then concrete appearance, pose and expression, clothing with materials and colors, and the place around the subject.
3. Photographic language helps a lot: lens and distance ("shot on a 35mm lens at eye level", "85mm close-up, shallow depth of field"), film stock or look, grain, light source and its direction and color temperature, time of day.
4. Mention texture and imperfection for realism (skin texture, stray hairs, worn fabric, dust in the light).
5. Text that must appear in the image is written in double quotes with its placement.
6. End with the mood and palette in a few words.
No quality tags (masterpiece, 8k, best quality), no weights or brackets. When style reference images are supplied, describe the CONTENT of the new image; the references decide the style.
Write the prompt in English even if the user writes in another language."""
# FLUX.3 Image Klein — по руководству Black Forest Labs и шаблонам ComfyUI.
FLUX3_RULES = """You write prompts for FLUX.3 Image Klein, a text-to-image and image-editing model from Black Forest Labs with a Qwen3-VL text encoder. It follows long, precise natural-language descriptions and renders text in images well.
How a good FLUX.3 prompt is built:
1. Natural flowing prose, most important first: the kind of image (photo, illustration, 3D render…), the main subject and what it is doing.
2. Then concrete appearance, clothing with materials and colors, the setting from foreground to background with positional words.
3. Camera and light like a photographer: shot size, lens or angle, light source, direction and color temperature, time of day.
4. Text that must appear in the image is written in double quotes together with its placement and style.
5. End with mood and color palette in a few words.
No quality tags (masterpiece, 8k, best quality), no weights or brackets, no negations ("no X") — describe what IS there.
Reference images, when supplied, are called <ref_image_0>, <ref_image_1>… in the order given.
Write the prompt in English even if the user writes in another language."""
# LTX‑2.5 — по руководству Lightricks по промптам LTX‑2 (один связный абзац, хронология, звук).
LTX_RULES = """You write prompts for LTX-2.5, a model that generates a video TOGETHER WITH its audio (speech, sound effects, ambience, music) in one pass, 24 fps, usually 5-10 seconds.
The prompt is ONE flowing paragraph in English, present tense, written like a cinematographer describing a single continuous shot, in chronological order:
1. Start with the main action in one sentence.
2. Specific movements and gestures, in the order they happen; keep them physically plausible for the duration (a 5-second clip holds only one or two actions).
3. Appearance of the characters and objects: age group (adults only), hair, clothing with materials and colors.
4. Background and environment details.
5. Camera: shot size, angle and movement with speed ("slow dolly-in", "handheld tracking shot from behind", "static wide shot").
6. Lighting and colors, time of day.
7. Audio: the ambience and sounds of the actions; music if wanted. Spoken words go in double quotes right where they are said, with who says them and how ("she whispers, \"We're late.\""). Keep the user's spoken words verbatim in their language.
Avoid abrupt cuts, lists, headings, quality tags and negations ("no X") — describe what IS there. About 120-200 words.
When a first frame is supplied, the video starts exactly from that picture: describe what is in it and how it starts moving, without changing the look of the people and the place."""
DETAIL_QWEN = {"short": "about 60 words", "medium": "about 150 words", "long": "250-400 words"}

# MiniMax H3 — по официальному руководству MiniMax (VIDEO_PROMPT_WRITING_GUIDE_base_en.md в репозитории
# MiniMaxAI/MiniMax-H3) и руководству ComfyUI по промптам H3.
MINIMAX_RULES = """You write prompts for MiniMax H3, a model that generates a video TOGETHER WITH its soundtrack (speech, sound effects, music) in one pass: 24 fps, up to 15 seconds, native canvas 768 px on the short side.
The prompt is ALWAYS written in English in exactly this format — three fields separated by blank lines:
integrated_multimodal_description: [Shot 1] ...
overall_soundscape: ...
non_diegetic_music: ...
How to write integrated_multimodal_description:
1. Open [Shot 1] with the visual style and medium (e.g. "Live-action, cinematic, 35mm film look, natural daylight" or "3D animation, soft toon shading"), then the setting, the subjects with their appearance and clothing, and their positions.
2. Tell the action as a timeline of shots that fits the video duration. The first shot has no timestamp; every later shot starts like "[Shot 2] At 00:03.500, the camera cuts to ..." with increasing times inside the duration. For small changes prefer camera movement inside one shot instead of a cut. About 1-2 shots for 5 seconds, up to 4-5 for 15 seconds.
3. Every camera movement names type, amplitude and speed: "the camera pushes in with small amplitude at slow speed", "pans right with medium amplitude at moderate speed", "static locked-off camera".
4. Describe concrete physical actions in the order they happen; keep each character's appearance, clothing and position consistent across shots.
5. Speech: give each voice a stable speaker ID (S1), (S2)… and repeat it whenever that voice speaks. Who it is, the ID, the action and the manner of speaking stay OUTSIDE the tags; inside <d> … </d> only the language tag and the exact words, e.g.: The tall man in a grey coat (S1) says quietly: <d>[Russian] Пора идти.</d> Keep the user's spoken words verbatim in their language (Russian stays Russian with the [Russian] tag). Off-screen narration: "(S1) says in an off-screen voiceover:" and state that the visible people's lips stay closed. If speech is cut off by the end of the video, end it with <cutoff>.
6. Text that is visible in the picture (signs, titles, screens) goes in double quotes, verbatim, in its original language.
7. The model has no negative prompt: never write "no X" / "without X"; describe what IS there instead (e.g. "the wall is plain and blank").
8. No quality tags (masterpiece, 8k, best quality).
overall_soundscape: 1-4 sentences in one paragraph about ambience, sounds of the actions and non-verbal human sounds (breathing, footsteps, laughter) across the whole video — no speech, no singing, no music. Write N/A only for complete silence.
non_diegetic_music: 1-3 sentences about the score the audience hears (instruments, rhythm, tempo, mood, when it enters or changes), or N/A if there is none."""
# Режим референсов (Ref2VA) — по VIDEO_PROMPT_WRITING_GUIDE_ref_en.md из MiniMaxAI/MiniMax-H3.
MINIMAX_REF_RULES = """You write prompts for MiniMax H3 in REFERENCE mode (Ref2VA): the model receives up to nine reference pictures, labelled <Picture 1> … <Picture N> in the given order, and generates a NEW video with its soundtrack (speech, sound effects, music) in one pass, at 24 fps, up to 15 seconds, that reuses the people, places, objects or style from those pictures.
The prompt is ALWAYS in English and has exactly these six sections, each name on its own line followed by its text, sections separated by blank lines:
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:
subject_definitions — one line per reusable thing taken from the pictures: "<Subject N> is the <what it is> in <Picture N>, featuring <key visible features: face and hair, clothing, colours, materials, architecture…>." Number subjects 1, 2, 3… A subject may cite several pictures ("in <Picture 2> and <Picture 3>"). A picture that only defines a character, place, costume or style gets NO separate entry of its own — it is cited inside its subject's line.
summary — one paragraph that starts with the task type in square brackets, normally "[reference generation]", then says what happens in the target video using the <Subject N> labels.
retention_analysis — one line per subject: "<Subject N> (appears in [Shot 1], [Shot 2]): fully_preserved - <what is kept>." Markers: fully_preserved (identity/look kept exactly), partially_preserved (some features kept, others changed — say which), attribute_transfer (only a property is taken, e.g. the outfit, colour palette or art style), weak_reference (loose inspiration).
detailed_description — first one sentence on the overall visual style and lighting, then the shots: "[Shot 1] …", later shots "[Shot 2] At 00:03.000, the shot cuts to …" with timestamps inside the video duration (about 1-2 shots for 5 s, up to 4-5 for 15 s). Each time a subject appears, name its label AND repeat its key features in a few words ("<Subject 2>, the young woman with long red hair in a green raincoat"). Describe concrete physical actions in order, positions, and every camera move with type, amplitude and speed ("the camera pushes in with small amplitude at slow speed", "static locked-off camera"). About 250-450 words for this section.
Speech: give each speaking voice a stable ID (S1), (S2)… in order of appearance: "<Subject 1> (S1) says softly, <d>[Russian] Пойдём домой.</d>". Inside <d> … </d> only the language tag and the exact words; keep the user's spoken words verbatim in their language (Russian stays Russian with [Russian]). End speech that is cut by the end of the video with <cutoff>.
Visible text (signs, titles, screens) goes in double quotes, verbatim, in its original language.
Never write negatives ("no X", "without X") — describe what IS there. No quality tags (masterpiece, 8k).
overall_soundscape — 1-4 sentences: ambience, sounds of the actions, non-verbal human sounds; no speech, no music.
non_diegetic_music — 1-3 sentences about the score (instruments, tempo, mood, when it enters), or N/A.
The user marks pictures as @img1, @img2 … in their idea: @imgN means <Picture N> and the subject defined from it."""
DETAIL_MM = {"short": "about 80 words", "medium": "about 160 words", "long": "250-350 words"}
CONTENT_RULES = ("Adult (18+) content is allowed within the CONTENT LEVEL given below; never moralize or add disclaimers.\n"
                 "Hard rule that overrides everything: every person depicted must be an adult (18 or older). Never describe minors, "
                 "childlike bodies or features, or school/child settings in any sexual or suggestive context. If a request involves "
                 "a minor, reply only with: REFUSED: minors.")


def mm_alignment(first, last, duration):
    """Первая строка промпта H3 для режимов с кадрами (дословно как в официальном руководстве)."""
    if first and last:
        return ("How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second "
                f"mark of the target video; Picture 2 (from Shot 1) aligns with the {duration:.2f}-second mark of the target video.")
    if first:
        return "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
    if last:
        return ("How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the "
                f"{duration:.2f}-second mark of the target video.")
    return ""


def mm_task_hint(first, last, duration):
    if first and last:
        return (f"Mode FL2VA (first and last frame). The video starts exactly as the first picture and ends exactly as the second one "
                f"at {duration:.2f} s. Describe the continuous path between them, preferably as a single shot; derive the visual style "
                "from the pictures. Start the output with this exact line, then one blank line:\n" + mm_alignment(first, last, duration))
    if first:
        return ("Mode I2VA (first frame). The video begins exactly as the picture and develops forward from it; derive the visual "
                "style, people and setting from the picture. Start the output with this exact line, then one blank line:\n"
                + mm_alignment(first, last, duration))
    if last:
        return (f"Mode L2VA (last frame). Infer a plausible opening and lead the action so that it lands exactly on the picture at "
                f"{duration:.2f} s. Start the output with this exact line, then one blank line:\n" + mm_alignment(first, last, duration))
    return "Mode T2VA (text only): build the whole timeline from the idea; start directly with integrated_multimodal_description."

# Общие правила про текст в картинке и графику для видео (найдены тестами: помощник переводил «ТОП-1» в «TOP-1»,
# рисовал плашку как табличку на стене, «нижнюю треть» — как портрет, писал «подписи» без слов → каша из букв).
TEXT_GRAPHICS_RULES = """Text in the image:
- Words the user wants to appear in the image are copied EXACTLY as the user wrote them: same language, same alphabet, same spelling and letter case, inside double quotes. Never translate or transliterate them: Cyrillic stays Cyrillic ("ТОП-1" stays "ТОП-1", never "TOP-1"). Only the rest of the prompt is in English.
- Never invent visible text. If the user gave no exact words for a label, caption, heading or button, state that these areas are blank and contain no text; never write "labeled", "with text" or "caption" without quoting the actual words.
Graphics for video and design (плашка, титр, нижняя треть / lower third, баннер, кнопка, бейдж, инфографика, логотип, иконка, заставка):
- Describe them as flat 2D digital graphic design elements seen straight on, like a motion-graphics overlay: no wall, room, desk or stand they are mounted on, no perspective tilt, no photographed sign, no 3D mockup, unless the user asks for a physical object.
- A lower third (нижняя треть) is a horizontal graphic bar in the lower part of the frame carrying the given name and title; it is not a portrait of the person.
- Keep such designs clean: only the elements and words the user asked for, no extra decorative text, no people unless requested."""

EDIT_GRAPHICS_RULES = ("If the request adds a graphic element (plate, banner, caption, button, badge, logo, text), describe it as a flat "
                       "graphic overlay on top of the picture, facing the viewer, not as a physical object standing on furniture or "
                       "hanging in the scene, unless the user asks for that. Text that must appear is copied exactly as the user wrote "
                       "it, same language and alphabet, in double quotes (Cyrillic stays Cyrillic).")
EDIT_MASK_FIT = ("The new element must fit entirely inside the selected region with a clear margin from its edges, so that "
                 "nothing is cut off at the region's border; keep it noticeably smaller than the region.")

# Уровень «цензуры»: 4 — строго, 0 — без цензуры. Слова пользователя никогда не удаляются;
# уровень определяет, что помощник добавляет САМ.
CENSOR = {
    4: ("CONTENT LEVEL 4/4 — STRICT (safe for work): everyone fully and normally clothed, no nudity, no lingerie, "
        "no sexual poses or suggestive wording. If the user's own words ask for more, keep their words but do not add anything further."),
    3: ("CONTENT LEVEL 3/4 — MILD: clothed; attractive, stylish, flirty is fine (fitted clothes, a confident look), "
        "but no nudity, no underwear, no sexual poses."),
    2: ("CONTENT LEVEL 2/4 — SENSUAL: revealing outfits, lingerie, swimwear, bare shoulders and legs, implied nudity "
        "(covered by hands, sheets, hair or shadows), seductive poses and glances. No visible genitals, no sex acts."),
    1: ("CONTENT LEVEL 1/4 — ARTISTIC NUDITY: the subject is nude or topless by default unless the user specified clothing; "
        "describe the bare body tastefully but concretely (skin, curves, breasts, pose), erotic mood is welcome. No sex acts unless the user asked."),
    0: ("CONTENT LEVEL 0/4 — UNCENSORED: explicit adult content. Unless the user specified clothing, the subject is fully naked; "
        "describe nudity explicitly and in anatomical detail with plain, direct explicit vocabulary, use provocative erotic poses; "
        "explicit sexual acts are allowed when the user's idea implies them. Still adults only."),
}

RANDOM_POOLS = {
    "setting": ["rooftop at night above a neon city", "sunlit Mediterranean terrace", "misty pine forest", "vintage diner",
                "luxury hotel room", "desert road at golden hour", "rainy Tokyo alley", "art gallery with white walls",
                "old library with tall shelves", "sailing yacht on open sea", "greenhouse full of plants", "abandoned factory",
                "snowy mountain cabin", "neon-lit arcade", "tropical beach at sunrise", "Parisian balcony", "subway car at night",
                "recording studio", "flower field in the wind", "concrete brutalist stairwell", "cozy bedroom with fairy lights",
                "infinity pool at dusk", "underground jazz bar", "wheat field before a storm", "retro 70s living room",
                "mirror-filled dance studio", "japanese onsen with steam", "car interior at night", "lighthouse on cliffs",
                "backstage dressing room with bulbs"],
    "style": ["35mm film photo with natural grain", "high-end fashion editorial photo", "polaroid snapshot", "cinematic film still",
              "moody low-key studio portrait", "candid smartphone photo", "vintage 1970s film photo", "black and white fine-art photo",
              "glossy magazine cover shot", "soft dreamy pastel photo", "hyperreal 85mm portrait", "documentary street photo",
              "neon noir cinematic still", "oil painting", "anime illustration", "watercolor illustration", "digital concept art"],
    "light": ["golden hour backlight", "hard midday sun with deep shadows", "soft window light", "neon pink and blue light",
              "candlelight", "overcast diffused light", "flash photography at night", "rim light in the dark", "sunset glow",
              "blue hour twilight", "dappled light through leaves", "studio softbox light"],
    "camera": ["close-up portrait", "medium shot", "full-body shot", "low angle shot", "high angle shot", "over-the-shoulder view",
               "wide shot with environment", "mirror selfie", "top-down view", "dutch angle", "shallow depth of field, 85mm",
               "wide-angle 24mm"],
    "mood": ["playful", "mysterious", "melancholic", "confident", "dreamy", "energetic", "intimate", "serene", "rebellious", "romantic"],
}


def register(routes, S, civ):
    ROOT = S.ROOT
    LLM_DIR = S.MODELS / "llm"
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    BIN_DIR = ROOT / "engine" / "llama"
    EXE = BIN_DIR / llm_install.EXE
    PORT = int(S.RUNTIME["llm_port"])  # свой порт у каждой копии программы (Start Flux / Start Qwen)
    LOG = S.LOGS / f"llm{S.SUFFIX}.log"
    st = {"proc": None, "model": None, "mmproj": None, "device": None, "status": "stopped", "message": "",
          "install": {"running": False, "file": "", "done": 0, "total": 0, "error": "", "variant": ""}}
    lock = asyncio.Lock()

    def cfg(k, d=None):
        v = S.CONFIG.get(k)
        return d if v in (None, "") else v

    def gguf_files():
        files = sorted(p.relative_to(LLM_DIR).as_posix() for p in LLM_DIR.rglob("*.gguf"))
        models = [f for f in files if "mmproj" not in f.lower()]
        mmproj = [f for f in files if "mmproj" in f.lower()]
        return models, mmproj

    def auto_mmproj(model):
        """Подобрать файл зрения (mmproj) к модели: имя mmproj без слова «mmproj…» должно быть началом имени модели."""
        _, mm = gguf_files()
        if not model or not mm:
            return ""
        stem = Path(model).name.lower()
        best = ""
        for m in mm:
            base = Path(m).name.lower().split("mmproj")[0].rstrip("-._ ")
            if len(base) >= 8 and stem.startswith(base) and len(base) > len(best):
                best = m
        return best

    def detected():
        if "detected" not in st:
            st["detected"] = llm_install.detect_variant()
        return st["detected"]

    def running():
        return st["proc"] is not None and st["proc"].poll() is None

    def stop():
        p = st["proc"]
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(10)
            except Exception:
                p.kill()
        st.update(proc=None, model=None, mmproj=None, device=None, status="stopped")

    async def stop_async():
        """Выгрузка без блокировки сервера (ожидание завершения процесса — в отдельном потоке)."""
        p = st["proc"]
        st.update(proc=None, model=None, mmproj=None, device=None, status="stopped")
        if p and p.poll() is None:
            p.terminate()
            try:
                await asyncio.get_running_loop().run_in_executor(None, p.wait, 10)
            except Exception:
                p.kill()

    def log_tail(n=25):
        try:
            return "\n".join(LOG.read_text("utf-8", errors="replace").splitlines()[-n:])
        except Exception:
            return ""

    async def free_flux():
        try:
            async with ClientSession(timeout=ClientTimeout(total=30)) as s:
                await s.post(S.ENGINE_.url + "/free", json={"unload_models": True, "free_memory": True})
        except Exception:
            pass

    async def ensure(model, mmproj, device):
        if not EXE.exists():
            raise RuntimeError("Движок помощника не установлен — нажмите «Установить» в Настройках → Помощник промптов.")
        if not model or not (LLM_DIR / model).exists():
            raise RuntimeError("Не выбрана модель помощника. Скачайте её: Модели → «⬇ Скачать рекомендованные» → раздел «Помощник промптов».")
        if running() and st["model"] == model and st["mmproj"] == mmproj and st["device"] == device:
            return
        stop()
        if device == "gpu":
            await free_flux()
        args = [str(EXE), "-m", str(LLM_DIR / model), "--host", "127.0.0.1", "--port", str(PORT),
                "-c", str(int(cfg("llm_ctx", 8192))), "--jinja", "--no-webui",
                "-ngl", "999" if device == "gpu" else "0"]
        if mmproj and (LLM_DIR / mmproj).exists():
            args += ["--mmproj", str(LLM_DIR / mmproj)]
        log = open(LOG, "w", encoding="utf-8", errors="replace")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        st["proc"] = subprocess.Popen(args, cwd=str(BIN_DIR), stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        st.update(model=model, mmproj=mmproj, device=device, status="loading", message="Загрузка модели помощника…")
        t0 = time.time()
        async with ClientSession(timeout=ClientTimeout(total=3)) as s:
            while time.time() - t0 < 300:
                if st["proc"].poll() is not None:
                    st["status"] = "error"
                    raise RuntimeError("Модель помощника не запустилась:\n" + log_tail())
                try:
                    async with s.get(f"http://127.0.0.1:{PORT}/health") as r:
                        if r.status == 200:
                            st.update(status="ready", message="")
                            return
                except Exception:
                    pass
                await asyncio.sleep(0.5)
        stop()
        raise RuntimeError("Модель помощника слишком долго загружается")

    def current_model():
        model = cfg("llm_model") or (gguf_files()[0][0] if gguf_files()[0] else "")
        mmproj = cfg("llm_mmproj", "auto")
        if mmproj == "auto":
            mmproj = auto_mmproj(model)
        elif mmproj == "none":
            mmproj = ""
        if mmproj and not (LLM_DIR / mmproj).exists():
            mmproj = ""
        return model, mmproj

    def available():
        model, _ = current_model()
        return EXE.exists() and bool(model) and (LLM_DIR / model).exists()

    async def chat(messages, max_tokens=800, temperature=0.8, need_vision=False, hold=False):
        model, mmproj = current_model()
        if need_vision and not mmproj:
            raise RuntimeError("Для «Описать картинку» нужна модель со зрением (VL) и её файл mmproj. "
                               "Скачайте «Qwen3‑VL 8B abliterated» + «файл зрения» в Модели → Помощник промптов.")
        device = cfg("llm_device", "gpu")
        async with lock:
            await ensure(model, mmproj, device)
            st["status"] = "thinking"
            try:
                body = {"messages": messages, "temperature": temperature, "top_p": 0.9, "max_tokens": max_tokens,
                        "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
                async with ClientSession(timeout=ClientTimeout(total=600)) as s:
                    async with s.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body) as r:
                        data = await r.json(content_type=None)
                if "choices" not in data:
                    raise RuntimeError("Ошибка модели: " + str(data.get("error", data))[:400])
                text = data["choices"][0]["message"].get("content") or ""
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
                return text
            finally:
                if not cfg("llm_keep_loaded", False) and not hold:
                    stop()
                elif running():
                    st["status"] = "ready"

    # ------------------------------------------------ сборка контекста
    def lora_block(loras):
        lines = []
        for i, l in enumerate(loras, 1):
            c = civ["lora_context"](l["name"])
            d = c.get("digest") or {}
            smart = d.get("made_by") in ("llm", "user")   # есть краткая выжимка — длинное описание не нужно
            cur = float(l.get("strength", 1))
            part = [f"{i}. \"{c['title']}\" (strength {cur:.2f})"]
            if c["triggers"]:
                src = {"civitai": "", "manual": " (set by user)", "file": " (guessed from training tags)",
                       "digest": " (found in the description)"}.get(c["triggers_source"], "")
                part.append(f"   trigger words{src}: " + ", ".join(c["triggers"]))
            else:
                part.append("   trigger words: none known")
            if c["file_tags"] and not smart:
                part.append("   frequent training tags: " + ", ".join(c["file_tags"][:8]))
            if c["note"]:
                part.append("   user's note: " + c["note"])
            if d.get("summary"):
                part.append("   what it does: " + d["summary"])
            if d.get("tips_en"):
                part.append("   how to prompt it: " + d["tips_en"])
            if d.get("rules"):
                part.append("   rules: " + "; ".join(d["rules"][:4]))
            st = d.get("strength") or {}
            if st.get("min") is not None:
                rng = f"{st['min']:g}–{st['max']:g}" if st.get("max") != st.get("min") else f"{st['min']:g}"
                bad = cur < st["min"] - 0.05 or cur > st["max"] + 0.05
                part.append(f"   recommended strength: {rng}" + (f" — the current {cur:g} is OUTSIDE this range" if bad else ""))
            if not smart and c["description"]:
                part.append("   description: " + c["description"].replace("\n", " ")[:700])
            for ex in c["examples"][:1 if smart else 3]:
                part.append("   example prompt by author: " + ex.replace("\n", " ")[:400])
            lines.append("\n".join(part))
        return "\n".join(lines)

    def system_prompt(p):
        loras = [l for l in p.get("loras") or [] if l.get("enabled", True) and float(l.get("strength", 1)) != 0]
        fam = p.get("family") or "qwen"
        qwen = fam in ("qwen", "krea2", "flux3")
        # у каждой нейросети свои правила промптинга + общие строки про 18+ и «только взрослые»
        base = {"qwen": QWEN_RULES, "krea2": KREA_RULES, "flux3": FLUX3_RULES, "ltx": LTX_RULES}.get(fam) or (
            MINIMAX_REF_RULES if p.get("mm_mode") == "ref" else MINIMAX_RULES)
        sp = base + "\n" + CONTENT_RULES
        if qwen:
            sp += "\n\n" + TEXT_GRAPHICS_RULES
        lvl = p.get("censor")
        if lvl is not None and str(lvl) != "":
            sp += "\n\n" + CENSOR.get(max(0, min(4, int(lvl))), CENSOR[4])
            sp += "\nThe user's own words are never removed or softened; the level controls what you add yourself."
        else:
            sp += "\n\nDescribe nudity or sexual content only as far as the user or the image asks for it."
        if loras:
            sp += "\n\n" + LORA_RULES.format(loras=lora_block(loras))
        lab = getattr(S, "LAB", None)   # правила, которые пользователь принял в Мастерской
        learned = lab["prompt_rules"](p.get("family") or "qwen") if lab else []
        if learned:
            sp += ("\n\nRules learned from the user's own tests (in Russian; follow them when they apply to this request, "
                   "they override general habits):\n" + "\n".join("- " + r for r in learned))
        if p.get("character"):
            sp += ("\n\nIDENTITY LOCK: a character reference photo defines who the person is. You cannot see it, so you do not know "
                   "their looks. Refer to them only neutrally ('the woman', 'the same woman as in the reference photo'). "
                   "NEVER state or imply their skin tone or complexion, ethnicity or race, age, eye color, face shape, facial features, "
                   "hair color, hair length or hairstyle, or body type — the reference photo decides all of that. "
                   "Mention any of these ONLY if the user explicitly wrote it. Avoid words like fair, pale, tanned, porcelain, "
                   "blonde, brunette, Asian, European, African, etc. unless the user used them. Focus the prompt on pose, action, "
                   "expression, clothing, setting, lighting, camera and mood.")
        if p.get("edit_mode") and fam == "qwen":
            sp += ("\n\nMode: IMAGE EDITING with reference images. The images are <image1>, <image2>… in order. The prompt must be an "
                   "editing instruction, e.g. 'Keep the character and pose in <image1> unchanged, put the jacket from <image2> on her, "
                   "preserve the original face, hair, body shape, background and lighting', not a new scene from scratch.")
        elif p.get("edit_mode") and fam == "flux3":
            sp += ("\n\nMode: IMAGE EDITING with reference images <ref_image_0>, <ref_image_1>… in order. Write the RESULT, not the change: "
                   "describe the finished image as it should look, naming what comes from each reference, e.g. 'The woman from <ref_image_0> "
                   "wears the red leather jacket from <ref_image_1>, same face, hairstyle, pose and background, soft window light'.")
        elif p.get("edit_mode"):
            sp += ("\n\nMode: IMAGE EDITING. The user supplies reference image(s); the prompt must be an editing "
                   "instruction for the reference (what to change, what to keep: e.g. 'Replace the background with …, "
                   "keep her face, hairstyle and pose unchanged'), not a description of a new scene from scratch.")
        return sp, loras

    def strip_output(t):
        t = t.strip().strip('"').strip()
        t = re.sub(r"^(prompt|final prompt)\s*:\s*", "", t, flags=re.I)
        return t

    def split_note(t):
        note = ""
        m = re.search(r"\n?NOTE:\s*(.+)$", t, flags=re.S)
        if m:
            note, t = m.group(1).strip(), t[:m.start()].strip()
        return t, note

    def refused(t):
        return t.strip().upper().startswith("REFUSED")

    # ------------------------------------------------ выжимка из описания LoRA (один раз, хранится рядом с файлом)
    import hashlib
    import json as _json

    def page_sig(m):
        return hashlib.md5(((m.get("description") or "") + (m.get("version_description") or "") +
                            "|".join(m.get("examples") or [])).encode("utf-8")).hexdigest()[:12]

    def needs_digest(fname):
        m = civ["read_meta"](fname) or {}
        d = m.get("digest") or {}
        if d.get("made_by") == "user":
            return False
        has_text = bool((m.get("description") or "").strip() or m.get("examples") or m.get("version_description"))
        return has_text and not (d.get("made_by") == "llm" and d.get("sig") == page_sig(m))

    def parse_json(t):
        t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
        a, b = t.find("{"), t.rfind("}")
        if a < 0 or b <= a:
            raise ValueError("нет JSON")
        body = t[a:b + 1]
        try:
            return _json.loads(body)
        except Exception:
            return _json.loads(re.sub(r",\s*([}\]])", r"\1", body))

    def _num(v):
        try:
            v = float(str(v).replace(",", "."))
            return round(v, 2) if 0.05 <= v <= 3 else None
        except (TypeError, ValueError):
            return None

    def _stats_line(st):
        out = []
        ws = st.get("weights") or []
        if ws:
            v = sorted(ws)
            out.append(f"LoRA weights people used in {len(v)} example images: min {v[0]}, median {v[len(v) // 2]}, max {v[-1]}")
        for k, lab in (("cfg", "CFG"), ("steps", "steps")):
            v = sorted(st.get(k) or [])
            if v:
                out.append(f"typical {lab}: {v[len(v) // 2]:g}")
        if st.get("samplers"):
            from collections import Counter
            out.append("samplers: " + ", ".join(x for x, _ in Counter(st["samplers"]).most_common(2)))
        return "; ".join(out)

    async def digest_lora(fname, hold=False):
        """Прочитать страницу LoRA один раз и сохранить краткую выжимку (триггеры, сила, правила)."""
        m = civ["read_meta"](fname) or {}
        auto = civ["auto_digest"](fname, m)
        title = m.get("title") or m.get("name") or Path(fname).stem
        parts = [f"LoRA: {title}", f"Base model: {m.get('base_model') or 'unknown'}",
                 "Trigger words listed on the site: " + (", ".join(m.get("trained_words") or []) or "none")]
        tags = civ["file_tags"](fname)
        if tags:
            parts.append("Most frequent training caption tags in the file: " + ", ".join(tags[:10]))
        desc = civ["html_to_text"](m.get("description"), 5000)
        if desc:
            parts.append("Description:\n" + desc)
        vd = civ["html_to_text"](m.get("version_description"), 1500)
        if vd:
            parts.append("About this version:\n" + vd)
        for ex in (m.get("examples") or [])[:5]:
            parts.append("Example prompt: " + ex.replace("\n", " ")[:350])
        sl = _stats_line(m.get("example_stats") or {})
        if sl:
            parts.append("Settings from example images: " + sl)
        out = await chat([{"role": "system", "content": DIGEST_RULES}, {"role": "user", "content": "\n\n".join(parts)}],
                         max_tokens=600, temperature=0.2, hold=hold)
        j = parse_json(out)
        trig = [str(x).strip() for x in (j.get("triggers") or []) if str(x).strip()][:6]
        site = list(m.get("trained_words") or [])
        trig = site + [t for t in trig if t.lower() not in {x.lower() for x in site}] if site else trig
        smin, smax, srec = _num(j.get("strength_min")), _num(j.get("strength_max")), _num(j.get("strength_rec"))
        st = auto["strength"]
        if smin is not None or srec is not None:
            smin = smin if smin is not None else srec
            smax = smax if smax is not None else (srec if srec is not None else smin)
            smin, smax = min(smin, smax), max(smin, smax)
            st = {"min": smin, "max": smax, "rec": srec if srec is not None else round((smin + smax) / 2, 2), "source": "страница (ИИ)"}
        d = {"made_by": "llm", "sig": page_sig(m), "time": time.strftime("%Y-%m-%d %H:%M"),
             "summary": str(j.get("summary_ru") or "").strip()[:300],
             "triggers": trig[:8], "strength": st,
             "rules": [str(x).strip()[:160] for x in (j.get("rules_ru") or []) if str(x).strip()][:4],
             "tips_en": str(j.get("tips_en") or "").strip()[:400],
             "settings": str(j.get("settings_ru") or "").strip()[:200]}
        civ["update_meta"](fname, digest=d)
        return d

    async def ensure_digests(loras, limit=3):
        """Перед работой помощника: у включённых LoRA без выжимки — сделать её (один раз на LoRA)."""
        todo = [l["name"] for l in loras if needs_digest(l["name"])][:limit]
        for n in todo:
            try:
                await digest_lora(n, hold=True)
            except Exception:
                pass

    DIG = {"running": False, "queue": [], "done": 0, "total": 0, "current": "", "errors": []}

    def engine_busy():
        return any(j["status"] in ("queued", "running", "finishing") for j in S.ENGINE_.jobs.values())

    async def digest_worker():
        """Фоновая очередь: разбор описаний по одному, только пока не идёт генерация картинки."""
        if DIG["running"]:
            return
        DIG["running"] = True
        try:
            while DIG["queue"]:
                while engine_busy():
                    await asyncio.sleep(3)
                n = DIG["queue"].pop(0)
                DIG["current"] = n
                try:
                    if needs_digest(n):
                        await digest_lora(n, hold=bool(DIG["queue"]))
                except Exception as e:
                    DIG["errors"].append(f"{n}: {e}")
                DIG["done"] += 1
        finally:
            DIG.update(running=False, current="")
            if not cfg("llm_keep_loaded", False) and not engine_busy():
                stop()

    def queue_digests(files):
        new = [f for f in files if f not in DIG["queue"] and needs_digest(f)]
        if not DIG["running"]:
            DIG.update(done=0, total=0, errors=[])
        DIG["queue"] += new
        DIG["total"] += len(new)
        if new:
            asyncio.get_running_loop().create_task(digest_worker())
        return len(new)

    async def on_downloaded(fname):
        if available() and cfg("llm_auto_digest", True):
            queue_digests([fname])
    civ["hooks"]["downloaded"].append(on_downloaded)

    # ------------------------------------------------ HTTP
    def err(msg, code=400):
        return web.json_response({"error": str(msg)}, status=code)

    @routes.get("/api/llm/state")
    async def state(_):
        models, mm = gguf_files()
        return web.json_response({
            "engine_installed": EXE.exists(),
            "engine_variant": (BIN_DIR / "variant.txt").read_text().strip() if (BIN_DIR / "variant.txt").exists() else "",
            "install": st["install"], "models": models, "mmprojs": mm,
            "model": cfg("llm_model", models[0] if models else ""), "mmproj": cfg("llm_mmproj", "auto"),
            "mmproj_auto": auto_mmproj(cfg("llm_model", models[0] if models else "")),
            "device": cfg("llm_device", "gpu"), "keep_loaded": bool(cfg("llm_keep_loaded", False)),
            "auto_digest": bool(cfg("llm_auto_digest", True)),
            "status": st["status"] if running() or st["status"] == "error" else "stopped",
            "loaded": st["model"] if running() else "", "detected_variant": detected(),
        })

    @routes.post("/api/llm/config")
    async def set_config(req):
        b = await req.json()
        for k_in, k_cfg in (("model", "llm_model"), ("mmproj", "llm_mmproj"), ("device", "llm_device"),
                            ("keep_loaded", "llm_keep_loaded"), ("auto_digest", "llm_auto_digest")):
            if k_in in b:
                S.CONFIG[k_cfg] = b[k_in]
        S.save_config(S.CONFIG)
        if running() and (st["model"] != cfg("llm_model") or st["device"] != cfg("llm_device", "gpu")):
            stop()
        return web.json_response({"ok": True})

    @routes.post("/api/llm/install")
    async def install(req):
        b = await req.json()
        if st["install"]["running"]:
            return err("Уже устанавливается")
        stop()
        ins = st["install"]
        ins.update(running=True, error="", done=0, total=0, file="", variant="")

        def work():
            def pr(name, done, total):
                ins.update(file=name, done=done, total=total)
            try:
                ins["variant"] = llm_install.install(BIN_DIR, b.get("variant") or None, pr)
            except Exception as e:
                ins["error"] = str(e)
            finally:
                ins["running"] = False
        threading.Thread(target=work, daemon=True).start()
        return web.json_response({"ok": True})

    @routes.post("/api/llm/unload")
    async def unload(_):
        stop()
        return web.json_response({"ok": True})

    @routes.post("/api/loras/digest")
    async def digest_route(req):
        b = await req.json()
        name = b.get("file", "")
        if not (S.MODELS / "loras" / name).is_file():
            return err("Нет такого файла")
        if not available():
            d = civ["auto_digest"](name)
            return web.json_response({"digest": d, "note": "Помощник не установлен — выжимка сделана без него: только триггеры и сила из текста."})
        try:
            d = await digest_lora(name)
        except Exception as e:
            return err("Помощник не смог разобрать описание: " + str(e))
        return web.json_response({"digest": d})

    @routes.post("/api/loras/digest_all")
    async def digest_all(req):
        b = await req.json()
        if not available():
            return err("Помощник промптов не установлен — сначала скачайте его модель (⚙ у кнопок помощника)")
        files = b.get("files") or [f["name"] for f in S.list_models("loras")]
        n = queue_digests([f for f in files if (S.MODELS / "loras" / f).is_file()])
        return web.json_response({"queued": n, **DIG})

    @routes.get("/api/loras/digest_all")
    async def digest_all_state(_):
        return web.json_response({k: v for k, v in DIG.items() if k != "queue"} | {"left": len(DIG["queue"])})

    def frame_url(name):
        from PIL import Image
        path = (S.INPUT / (name or "")).resolve()
        if not name or not str(path).startswith(str(S.INPUT.resolve())) or not path.is_file():
            return None
        im = Image.open(path).convert("RGB")
        im.thumbnail((768, 768))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def mm_duration(p):
        try:
            sec = float(p.get("duration", 5))
        except (TypeError, ValueError):
            sec = 5.0
        n = max(5, int(round(sec * 24)))
        while n % 17 != 5:
            n += 1
        return n / 24.0, (n - 1) / 24.0

    async def enhance_refs(p, idea, n, sp):
        """Промпт MiniMax H3 Ref2VA: шесть разделов, субъекты из картинок-референсов (@img1… → <Picture N>)."""
        refs = [r for r in (p.get("refs") or []) if isinstance(r, dict) and r.get("name")][:9]
        total, _ = mm_duration(p)
        rnd = p.get("mode") == "random"
        lst = "\n".join(f"<Picture {i}> (@img{i})" + (f": the user calls it «{r['label']}»" if (r.get("label") or "").strip() else "")
                        for i, r in enumerate(refs, 1)) or "(no pictures yet — define subjects from the idea only)"
        task = (f"The video is {total:.1f} seconds long. Reference pictures, in order:\n{lst}\n"
                "Turn the idea into the six-section reference prompt. Every picture the user tagged must become a subject; "
                "use pictures the user did not mention only if they clearly fit. ")
        if rnd:
            import random as _r
            pick = {k: _r.choice(v) for k, v in RANDOM_POOLS.items()}
            task += ("Keep every detail the user specified; invent boldly the rest — action, camera, sound. "
                     f"Inspiration (adapt freely): style — {pick['style']}; light — {pick['light']}; camera — {pick['camera']}; mood — {pick['mood']}. ")
            n = 1
        if n == 1:
            task += "Output only the prompt (the six sections), nothing else."
        else:
            task += f"Write {n} clearly different prompts (different action, camera and sound), each complete. Start each with a line '### '. Output nothing else."
        if p.get("extra"):
            task += "\nExtra wishes from the user: " + p["extra"]
        user_text = (f"Idea (may be in Russian): {idea}" if idea else "Idea: (empty — invent a fitting short scene with these subjects)") + "\n\n" + task
        imgs = [u for u in (frame_url(r["name"]) for r in refs) if u]
        vision = bool(imgs) and bool(current_model()[1])
        if vision:
            user_text += "\nThe attached images are the reference pictures in the same order (Picture 1 first). Describe what you see in them precisely."
            content = [{"type": "text", "text": user_text}] + [{"type": "image_url", "image_url": {"url": u}} for u in imgs]
        else:
            content = user_text + ("\n(You cannot see the pictures; rely on the user's names for them and keep their appearance general.)" if imgs else "")
        try:
            out = await chat([{"role": "system", "content": sp}, {"role": "user", "content": content}],
                             max_tokens=1100 * n + 200, temperature=1.0 if rnd else (0.6 if n == 1 else 0.95), need_vision=vision)
        except Exception as e:
            return err(e)
        if refused(out):
            return err("Помощник отказался: запрос касается несовершеннолетних.")
        if n == 1:
            text, note = split_note(fix_ref(strip_output(out)))
            return web.json_response({"prompt": text, "note": note, "vision": vision})
        parts = [x for x in re.split(r"^\s*#{2,}\s*\d*[.)]?\s*", out, flags=re.M) if x.strip()]
        clean, notes = [], []
        for x in parts:
            tx, note = split_note(fix_ref(strip_output(x)))
            if note:
                notes.append(note)
            if tx:
                clean.append(tx)
        return web.json_response({"variants": clean[:n], "note": notes[0] if notes else "", "vision": vision})

    def fix_ref(text):
        """Подстраховка для Ref2VA: @imgN → <Picture N>, обязательные разделы на месте."""
        text = re.sub(r"@img\s*(\d+)", lambda m: f"<Picture {m.group(1)}>", text.strip(), flags=re.I)
        for sec, filler in (("overall_soundscape:", "Natural ambient sound that matches the scene and the actions."),
                            ("non_diegetic_music:", "N/A")):
            if sec not in text:
                text += f"\n\n{sec}\n{filler}"
        return text

    async def enhance_video(p, idea, n, sp, loras):
        """Промпт MiniMax H3: таймлайн шотов + звук + музыка, с учётом первого/последнего кадра."""
        if p.get("mm_mode") == "ref":
            return await enhance_refs(p, idea, n, sp)
        first, last = p.get("first_frame"), p.get("last_frame")
        total, last_t = mm_duration(p)
        detail = DETAIL_MM.get(p.get("detail"), DETAIL_MM["medium"])
        rnd = p.get("mode") == "random"
        task = (f"The video is {total:.1f} seconds long. {mm_task_hint(bool(first), bool(last), last_t)}\n")
        if rnd:
            import random as _r
            pick = {k: _r.choice(v) for k, v in RANDOM_POOLS.items()}
            task += ("Keep every detail the user specified; invent boldly everything else — action, setting, camera moves, sound. "
                     f"Inspiration (adapt freely, do not list): setting — {pick['setting']}; style — {pick['style']}; light — {pick['light']}; "
                     f"camera — {pick['camera']}; mood — {pick['mood']}. ")
            n = 1
        if n == 1:
            task += (f"Write one complete prompt with the visual description of {detail}. Output only the prompt in the required "
                     "format (the alignment line if required, then the three fields), nothing else.")
        else:
            task += (f"Write {n} clearly different prompts (different action, camera and sound), each in the full required format. "
                     "Start each prompt with a line '### '. Output nothing else.")
        if p.get("extra"):
            task += "\nExtra wishes from the user: " + p["extra"]
        user_text = (f"Idea (may be in Russian): {idea}" if idea else "Idea: (empty — invent a fitting short scene)") + "\n\n" + task
        imgs = [u for u in (frame_url(first), frame_url(last)) if u]
        vision = bool(imgs) and bool(current_model()[1])
        if vision:
            user_text += "\nThe attached picture(s) are the keyframe(s) in order: Picture 1" + (", Picture 2" if len(imgs) > 1 else "") + "."
            content = [{"type": "text", "text": user_text}] + [{"type": "image_url", "image_url": {"url": u}} for u in imgs]
        else:
            content = user_text + ("\n(You cannot see the keyframes; keep the description of what is in them general.)" if imgs else "")
        try:
            out = await chat([{"role": "system", "content": sp}, {"role": "user", "content": content}],
                             max_tokens=700 * n + 200, temperature=1.0 if rnd else (0.7 if n == 1 else 0.95), need_vision=vision)
        except Exception as e:
            return err(e)
        if refused(out):
            return err("Помощник отказался: запрос касается несовершеннолетних.")
        if n == 1:
            text, note = split_note(fix_mm(strip_output(out), first, last, last_t))
            return web.json_response({"prompt": text, "note": note, "vision": vision})
        parts = [x for x in re.split(r"^\s*#{2,}\s*\d*[.)]?\s*", out, flags=re.M) if x.strip()]
        clean, notes = [], []
        for x in parts:
            tx, note = split_note(fix_mm(strip_output(x), first, last, last_t))
            if note:
                notes.append(note)
            if tx:
                clean.append(tx)
        return web.json_response({"variants": clean[:n], "note": notes[0] if notes else "", "vision": vision})

    def fix_ltx(text):
        """LTX: один абзац, без служебных полей MiniMax."""
        text = re.sub(r"^(integrated_multimodal_description|overall_soundscape|non_diegetic_music)\s*:\s*", "", text.strip(), flags=re.M | re.I)
        text = re.sub(r"\[Shot \d+\]\s*", "", text)
        return re.sub(r"\s*\n+\s*", " ", text).strip()

    async def enhance_ltx(p, idea, n, sp, data_url=None):
        """Промпт LTX‑2.5: один связный абзац по хронологии, со звуком; с учётом первого/последнего кадра."""
        first, last = p.get("first_frame"), p.get("last_frame")
        total, _ = mm_duration(p)
        rnd = p.get("mode") == "random"
        task = f"The video is {total:.1f} seconds long. "
        if data_url or first:
            task += "It starts exactly from the first frame (Picture 1): describe what is in it and how it comes to life. "
        if last:
            task += "It must end on the last frame (" + ("Picture 2" if first else "Picture 1") + "): the action leads naturally into it. "
        if rnd:
            import random as _r
            pick = {k: _r.choice(v) for k, v in RANDOM_POOLS.items()}
            task += ("Keep every detail the user specified; invent boldly everything else — action, setting, camera move, sound. "
                     f"Inspiration (adapt freely, do not list): setting — {pick['setting']}; style — {pick['style']}; light — {pick['light']}; "
                     f"camera — {pick['camera']}; mood — {pick['mood']}. ")
            n = 1
        if n == 1:
            task += "Write one prompt. Output only the prompt paragraph, nothing else."
        else:
            task += f"Write {n} clearly different prompts (different action, camera and sound). Start each with a line '### '. Output nothing else."
        if p.get("extra"):
            task += "\nExtra wishes from the user: " + p["extra"]
        user_text = (f"Idea (may be in Russian): {idea}" if idea else "Idea: (empty — invent a fitting short scene)") + "\n\n" + task
        imgs = [data_url] if data_url else [u for u in (frame_url(first), frame_url(last)) if u]
        vision = bool(imgs) and (bool(data_url) or bool(current_model()[1]))
        if vision:
            content = [{"type": "text", "text": user_text}] + [{"type": "image_url", "image_url": {"url": u}} for u in imgs]
        else:
            content = user_text + ("\n(You cannot see the keyframes; keep the description of what is in them general.)" if imgs else "")
        try:
            out = await chat([{"role": "system", "content": sp}, {"role": "user", "content": content}],
                             max_tokens=450 * n + 200, temperature=1.0 if rnd else (0.7 if n == 1 else 0.95), need_vision=vision)
        except Exception as e:
            return err(e)
        if refused(out):
            return err("Помощник отказался: запрос касается несовершеннолетних.")
        if n == 1:
            text, note = split_note(strip_output(out))
            return web.json_response({"prompt": fix_ltx(text), "note": note, "vision": vision})
        parts = [x for x in re.split(r"^\s*#{2,}\s*\d*[.)]?\s*", out, flags=re.M) if x.strip()]
        clean, notes = [], []
        for x in parts:
            tx, note = split_note(strip_output(x))
            if note:
                notes.append(note)
            if tx:
                clean.append(fix_ltx(tx))
        return web.json_response({"variants": clean[:n], "note": notes[0] if notes else "", "vision": vision})

    def fix_mm(text, first, last, last_t):
        """Подстраховка: обязательная строка выравнивания кадров и три поля на месте."""
        text = text.strip()
        line = mm_alignment(bool(first), bool(last), last_t)
        if line and not text.lower().startswith(("for the target video", "how the reference pictures")):
            text = line + "\n\n" + text
        if "integrated_multimodal_description:" not in text:
            head, _, body = text.partition("\n\n") if line else ("", "", text)
            text = (head + "\n\n" if head else "") + "integrated_multimodal_description: [Shot 1] " + body.strip()
        if "overall_soundscape:" not in text:
            text += "\n\noverall_soundscape: N/A"
        if "non_diegetic_music:" not in text:
            text += "\n\nnon_diegetic_music: N/A"
        return text

    async def describe_video(p, data_url, sp):
        """«Описать» в режиме видео: картинка — первый кадр, помощник пишет промпт H3, как её оживить."""
        total, last_t = mm_duration(p)
        wish = (p.get("prompt") or "").strip()
        task = (f"This picture is the FIRST FRAME of a {total:.1f}-second video. {mm_task_hint(True, False, last_t)}\n"
                "Bring it to life: natural motion of the people and objects, a fitting camera move, the matching sound. "
                + (f"The user's wishes: {wish}. " if wish else "")
                + "Output only the prompt in the required format.")
        msgs = [{"role": "system", "content": sp},
                {"role": "user", "content": [{"type": "text", "text": task}, {"type": "image_url", "image_url": {"url": data_url}}]}]
        try:
            out = await chat(msgs, max_tokens=800, temperature=0.6, need_vision=True)
        except Exception as e:
            return err(e)
        if refused(out):
            return err("Помощник отказался: на картинке может быть несовершеннолетний.")
        text, note = split_note(fix_mm(strip_output(out), True, False, last_t))
        return web.json_response({"prompt": text, "note": note})

    @routes.post("/api/llm/enhance")
    async def enhance(req):
        p = await req.json()
        idea = (p.get("prompt") or "").strip()
        n = max(1, min(int(p.get("n", 1)), 6))
        sp, loras = system_prompt(p)
        if loras and any(needs_digest(l["name"]) for l in loras):
            await ensure_digests(loras)
            sp, loras = system_prompt(p)
        if p.get("family") == "minimax":
            return await enhance_video(p, idea, n, sp, loras)
        if p.get("family") == "ltx":
            return await enhance_ltx(p, idea, n, sp)
        D = DETAIL_QWEN
        detail = D.get(p.get("detail"), D["medium"])
        rnd = p.get("mode") == "random"
        if rnd:
            import random as _r
            pick = {k: _r.choice(v) for k, v in RANDOM_POOLS.items()}
            task = (f"RANDOM SCENE. Keep every detail the user specified exactly as given (do not change or contradict it). "
                    f"Invent EVERYTHING the user did not specify — boldly and unexpectedly: pose and action, what the person is doing, "
                    f"clothing (respecting the content level), location and background, time of day and weather, lighting, "
                    f"camera angle and framing, mood, and visual style. Use these randomly drawn inspirations as a starting point "
                    f"(adapt freely, do not list them): setting — {pick['setting']}; style — {pick['style']}; light — {pick['light']}; "
                    f"camera — {pick['camera']}; mood — {pick['mood']}. "
                    f"Write one prompt of {detail}. Output only the prompt itself, one paragraph, no quotes, no headings.")
            n = 1
        elif n == 1:
            task = (f"Rewrite the user's idea into one polished prompt of {detail}. "
                    "Output only the prompt itself, one paragraph, no quotes, no headings.")
        else:
            task = (f"Write {n} clearly different prompts ({detail} each) for the user's idea — vary composition, "
                    "camera, lighting and mood, keep the core idea. Put each prompt on its own line starting with '### '. "
                    "Output nothing else.")
        if p.get("extra"):
            task += "\nExtra wishes from the user: " + p["extra"]
        user = (f"Idea: {idea}" if idea else "Idea: (empty — invent a subject and a fitting scene" + (" for the active LoRAs" if loras else "") + ")") + "\n\n" + task
        try:
            out = await chat([{"role": "system", "content": sp}, {"role": "user", "content": user}],
                             max_tokens=350 * n + 200, temperature=1.0 if rnd else (0.8 if n == 1 else 0.95))
        except Exception as e:
            return err(e)
        if refused(out):
            return err("Помощник отказался: запрос касается несовершеннолетних.")
        if n == 1:
            text, note = split_note(strip_output(out))
            return web.json_response({"prompt": text, "note": note, "inspiration": pick if rnd else None})
        parts = [strip_output(x) for x in re.split(r"^\s*#{2,}\s*\d*[.)]?\s*", out, flags=re.M) if x.strip()]
        if len(parts) < 2:
            parts = [strip_output(x) for x in re.split(r"\n\s*\n", out) if x.strip()]
        notes = []
        clean = []
        for x in parts:
            t, note = split_note(x)
            if note:
                notes.append(note)
            if t:
                clean.append(t)
        return web.json_response({"variants": clean[:n], "note": notes[0] if notes else ""})

    MUSIC_FMT = {
        "ace": ("ACE-Step 1.5", "tags: a comma-separated list of 8-15 short tags — genre and subgenre, key instruments, vocal type "
                "(e.g. 'female vocals', 'male rap vocals'), mood, tempo feel, production style. "
                "lyrics: song lyrics with section tags on their own lines: [Intro], [Verse], [Pre-Chorus], [Chorus], [Bridge], [Outro]."),
        "mm3": ("MiniMax Music 3", "tags: one or two natural sentences describing genre, instruments, vocal timbre, mood and tempo. "
                "lyrics: song lyrics with section tags [Intro], [Verse], [Pre-Chorus], [Chorus], [Bridge], [Outro]; "
                "non-sung sounds may go in parentheses, e.g. (rain on the window)."),
        "yue2": ("YuE2", "tags: 5-8 space-separated style words — genre, instrument, mood, vocal gender and timbre "
                 "(e.g. 'pop piano uplifting female bright vocal'). lyrics: lyrics with section tags [verse], [chorus], [bridge], [outro]; "
                 "every section 4-8 lines. YuE2 sings best in English or Chinese."),
        "sa3": ("Stable Audio 3", "tags: a vivid instrumental description in 2-4 sentences — genre, instruments, rhythm, tempo (BPM), "
                "mood, how the track develops over time, production. lyrics: always an empty string (this model has no vocals)."),
    }

    @routes.post("/api/llm/music")
    async def music_helper(req):
        """Стиль и текст песни под конкретную музыкальную нейросеть. Ответ — JSON {tags, lyrics}."""
        p = await req.json()
        eng = p.get("engine") if p.get("engine") in MUSIC_FMT else "ace"
        name, fmt = MUSIC_FMT[eng]
        lang = {"ru": "Russian", "en": "English", "uk": "Ukrainian", "de": "German", "es": "Spanish", "fr": "French",
                "ja": "Japanese", "ko": "Korean", "zh": "Chinese"}.get(p.get("language") or "ru", "Russian")
        inst = bool(p.get("instrumental")) or eng == "sa3"
        sp = (f"You write inputs for {name}, a music generation model. {fmt}\n"
              "Tags/description are always in English. "
              + ("The track is INSTRUMENTAL: lyrics must be an empty string. " if inst else
                 f"Write the lyrics in {lang}; keep the user's own lines verbatim if they gave any. Lyrics must fit about "
                 f"{int(float(p.get('duration') or 120))} seconds of music. ")
              + "Never write anything sexual involving minors; no real artists' names or copyrighted lyrics — write original words.\n"
              'Answer ONLY with JSON: {"tags": "...", "lyrics": "..."}')
        user = ("Idea (may be in Russian): " + ((p.get("idea") or "").strip() or "(empty — invent an original catchy song)")
                + ("\nCurrent tags: " + p["tags"] if p.get("tags") else "") + ("\nCurrent lyrics:\n" + p["lyrics"] if p.get("lyrics") else ""))
        try:
            out = await chat([{"role": "system", "content": sp}, {"role": "user", "content": user}], max_tokens=1400, temperature=0.9)
        except Exception as e:
            return err(e)
        try:
            d = parse_json(out)
        except Exception:
            return err("Помощник ответил не в том формате — попробуйте ещё раз.")
        return web.json_response({"tags": str(d.get("tags") or "").strip(), "lyrics": "" if inst else str(d.get("lyrics") or "").strip()})

    @routes.post("/api/llm/chat")
    async def chat_route(req):
        p = await req.json()
        sp, loras = system_prompt(p)
        if loras and any(needs_digest(l["name"]) for l in loras):
            await ensure_digests(loras)
            sp, loras = system_prompt(p)
        sp += ("\n\nYou are chatting with the user in Russian about their image prompt. Current prompt in the app:\n"
               f"\"\"\"{(p.get('prompt') or '').strip()}\"\"\"\n"
               "Answer briefly in Russian. Whenever you propose a ready prompt, put it on separate lines starting with "
               "'PROMPT:' followed by the prompt in English.")
        msgs = [{"role": "system", "content": sp}]
        for m in (p.get("messages") or [])[-16:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                msgs.append({"role": m["role"], "content": str(m["content"])[:4000]})
        try:
            out = await chat(msgs, max_tokens=900, temperature=0.8)
        except Exception as e:
            return err(e)
        if refused(out):
            out = "Не могу помочь с этим: запрос касается несовершеннолетних."
        return web.json_response({"reply": out})

    @routes.post("/api/llm/describe")
    async def describe(req):
        p = await req.json()
        name = p.get("image", "")
        path = (S.INPUT / name).resolve()
        if not name or not str(path).startswith(str(S.INPUT.resolve())) or not path.is_file():
            return err("Сначала загрузите картинку в блок «Изображение»")
        try:
            from PIL import Image
            im = Image.open(path).convert("RGB")
            im.thumbnail((1024, 1024))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=90)
            data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            return err("Не удалось прочитать картинку: " + str(e))
        sp, loras = system_prompt({**p, "edit_mode": False})
        if p.get("family") == "minimax":
            return await describe_video(p, data_url, sp)
        if p.get("family") == "ltx":
            return await enhance_ltx(p, (p.get("prompt") or "").strip(), 1, sp, data_url)
        D = DETAIL_QWEN
        detail = D.get(p.get("detail"), D["long"])
        task = (f"Describe this image as a text-to-image prompt of {detail} so that the image model can recreate it: "
                "subject, exact appearance, pose, clothing, setting, lighting, camera framing and style. "
                + ("Also include the active LoRAs' trigger words naturally. " if loras else "")
                + (f"Apply these changes/wishes: {p['prompt'].strip()}. " if (p.get("prompt") or "").strip() else "")
                + "Output only the prompt.")
        msgs = [{"role": "system", "content": sp},
                {"role": "user", "content": [{"type": "text", "text": task},
                                             {"type": "image_url", "image_url": {"url": data_url}}]}]
        try:
            out = await chat(msgs, max_tokens=500, temperature=0.6, need_vision=True)
        except Exception as e:
            return err(e)
        if refused(out):
            return err("Помощник отказался: на картинке может быть несовершеннолетний.")
        text, note = split_note(strip_output(out))
        return web.json_response({"prompt": text, "note": note})

    async def refine_edit(img_path, mask_path, instruction, loras, family="qwen", character=False):
        family = "qwen"  # правка всегда идёт через Qwen-Image 2.1
        """Превратить просьбу пользователя (часто по-русски) в точную английскую инструкцию правки.
        Если есть модель со зрением — она смотрит на картинку, выделенная маской область подсвечена красным."""
        from PIL import Image, ImageFilter
        _, mmproj = current_model()
        vision = bool(mmproj)
        has_mask = False
        content = []
        if vision:
            im = Image.open(img_path).convert("RGB")
            im.thumbnail((1024, 1024))
            if mask_path:
                m = Image.open(mask_path).convert("L").resize(im.size).point(lambda v: 255 if v > 127 else 0)
                if m.getbbox():
                    has_mask = True
                    red = Image.new("RGB", im.size, (255, 0, 0))
                    im = Image.composite(Image.blend(im, red, 0.45), im, m)
                    edge = m.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
                    im.paste((255, 0, 0), (0, 0), edge)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=90)
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}})
        else:
            has_mask = bool(mask_path)
        sp, _ = system_prompt({"loras": loras, "family": family})
        if family == "qwen":
            text = ("The user wants to edit an existing image with Qwen-Image 2.1. In the editor the image to edit is <image1>"
                    + (" and the character's face photo is <image2>" if character else "") + ".\n"
                    + ("You can see the image. " if vision else "You cannot see the image; rely on the request. ")
                    + ("The region to change is highlighted with a semi-transparent RED overlay and outline — the red is only a selection marker, "
                       "not part of the image; only that region will be regenerated.\n" if has_mask and vision else
                       "Only a user-selected region will be regenerated.\n" if has_mask else "")
                    + f"User's request (may be in Russian): «{instruction.strip()}»\n"
                    + EDIT_GRAPHICS_RULES + (" " + EDIT_MASK_FIT if has_mask else "") + "\n"
                    "Write ONE precise editing instruction in English, 2-4 sentences, in the official Qwen-Image style: first say what to "
                    "change in <image1> and where, describe the new content concretely (type, material, color, texture, fit, how light and "
                    "shadows fall on it, consistent with the scene), then say what to keep unchanged (face, identity, hair, body shape, pose, "
                    "other clothing, background, lighting, framing). Example: 'Keep the woman and her pose in <image1> unchanged, replace her "
                    "white t-shirt with a black leather biker jacket with silver zippers, natural folds and realistic leather sheen, preserve "
                    "her face, hair and the original background and lighting.' Output only the instruction.")
            content.insert(0, {"type": "text", "text": text})
            out = await chat([{"role": "system", "content": sp}, {"role": "user", "content": content if vision else text}],
                             max_tokens=350, temperature=0.4, need_vision=vision)
            if refused(out):
                raise RuntimeError("Помощник отказался: запрос касается несовершеннолетних.")
            return split_note(strip_output(out))[0], vision
        text = ("The user wants to edit an existing image with FLUX.2 Klein, an instruction-based image editor that sees the original image.\n"
                + ("You can see the image. " if vision else "You cannot see the image; rely on the request. ")
                + ("The region to change is highlighted with a semi-transparent RED overlay and outline — the red is only a selection marker, "
                   "not part of the image; only that region will be regenerated.\n" if has_mask and vision else
                   "Only a user-selected region will be regenerated.\n" if has_mask else "")
                + f"User's request (may be in Russian): «{instruction.strip()}»\n"
                + EDIT_GRAPHICS_RULES + (" " + EDIT_MASK_FIT if has_mask else "") + "\n"
                "Write ONE precise editing instruction in English, 1-3 sentences: name what to change and where it is, describe the new "
                "content concretely (type, material, color, fit, texture, how light and shadows fall on it), and say what must stay "
                "unchanged (face, identity, pose, other clothing, background, framing). Output only the instruction.")
        content.insert(0, {"type": "text", "text": text})
        out = await chat([{"role": "system", "content": sp}, {"role": "user", "content": content if vision else text}],
                         max_tokens=300, temperature=0.4, need_vision=vision)
        if refused(out):
            raise RuntimeError("Помощник отказался: запрос касается несовершеннолетних.")
        return split_note(strip_output(out))[0], vision

    async def stop_gpu():
        """Перед генерацией картинки: помощник уходит из видеопамяти, а по умолчанию и из оперативной
        (на процессоре он держит в ОЗУ ещё 3–6 ГБ — Qwen и так на пределе)."""
        if running() and (st["device"] == "gpu" or cfg("llm_unload_always", True)):
            await stop_async()

    return {"stop_gpu": stop_gpu, "stop": stop, "stop_async": stop_async, "queue_digests": queue_digests, "digest": digest_lora,
            "chat": chat, "vision": lambda: bool(current_model()[1]),
            "refine_edit": refine_edit, "available": available}
