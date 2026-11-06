"""Переозвучка роликов голосом эталона (постоянный процесс; команды — строки JSON в stdin).

Команды:
  ref  — эталон: звук из видео/аудио, отрезок, очистка от фона, распознавание текста эталона;
  asr  — ролик: звук, отделение голоса от фона, распознавание речи по фразам с таймингом;
  tts  — ролик: каждая фраза озвучивается голосом эталона ровно в длину своей фразы (OmniVoice,
         параметр duration), фразы ставятся на свои места, фон возвращается, звук вклеивается в видео.
Ответы — строки JSON в stdout: {"id", "ev": "stage"|"result"|"error", ...}.
"""
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from functools import partial
from pathlib import Path

CFG = json.loads(Path(sys.argv[1]).read_text("utf-8"))
FFMPEG = CFG["ffmpeg"]
os.environ["PATH"] = str(Path(FFMPEG).parent) + os.pathsep + os.environ.get("PATH", "")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
TMP = Path(CFG["tmp_dir"])
TMP.mkdir(parents=True, exist_ok=True)
OUT = Path(CFG["out_dir"])
OUT.mkdir(parents=True, exist_ok=True)
LOCK = threading.Lock()
SR = 44100


def emit(**kw):
    with LOCK:
        sys.__stdout__.write(json.dumps(kw, ensure_ascii=False) + "\n")
        sys.__stdout__.flush()


sys.stdout = sys.stderr   # чужие print() из библиотек — в журнал, не в канал команд

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def ff(*args):
    r = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        msg = (r.stderr.strip().splitlines() or ["ошибка"])[-1]
        if "does not contain any stream" in msg or "Output file is empty" in msg:
            msg = "в файле нет звуковой дорожки"
        raise RuntimeError("ffmpeg: " + msg)


def extract(src, dst, start=None, end=None):
    pre = []
    if start:
        pre += ["-ss", f"{float(start):.3f}"]
    if end and float(end) > float(start or 0):
        pre += ["-to", f"{float(end):.3f}"]
    ff(*pre, "-i", src, "-vn", "-ac", "2", "-ar", SR, "-c:a", "pcm_f32le", dst)


def read(path):
    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return a, sr


def key_of(path, *extra):
    st = Path(path).stat()
    return hashlib.md5(f"{path}|{st.st_size}|{st.st_mtime}|{extra}".encode()).hexdigest()[:12]


# ------------------------------------------------------------------ разделение голос / фон
SEP = None


def separator():
    global SEP
    if SEP is None:
        import logging
        from audio_separator.separator import Separator
        SEP = Separator(log_level=logging.WARNING, model_file_dir=CFG["sep_dir"], output_dir=str(TMP),
                        output_format="WAV", use_soundfile=True, normalization_threshold=1.0)
        SEP.load_model(model_filename=CFG["sep_model"])
    return SEP


def split_voice(wav_path, tag):
    """(голос, фон, sr) в исходной громкости: голос подгоняется к смеси, фон = смесь − голос."""
    mix, sr = read(wav_path)
    out = separator().separate(str(wav_path), custom_output_names={"Vocals": f"{tag}_vox", "Other": f"{tag}_bg",
                                                                     "Instrumental": f"{tag}_bg"})
    vf = next((TMP / Path(o).name for o in out if "_vox" in Path(o).name), None)
    if not vf:
        raise RuntimeError("модель разделения не вернула дорожку голоса")
    vox, sr2 = read(vf)
    if sr2 != sr:
        import librosa
        vox = librosa.resample(vox.T, orig_sr=sr2, target_sr=sr).T
    n = min(len(mix), len(vox))
    vox = np.pad(vox[:n], ((0, len(mix) - n), (0, 0)))
    if vox.shape[1] != mix.shape[1]:
        vox = np.repeat(vox[:, :1], mix.shape[1], axis=1)
    g = float(np.sum(vox * mix) / max(1e-9, np.sum(vox * vox)))
    vox = vox * max(0.0, min(4.0, g))
    return vox, mix - vox, sr


CLIPS = {}   # кэш разобранных роликов: путь → {mix, vox, bg, sr}


def load_clip(path, separate, rid):
    k = key_of(path, separate)
    if k in CLIPS:
        return CLIPS[k]
    emit(id=rid, ev="stage", msg="Достаю звук из видео…", p=0.05)
    w = TMP / f"{k}.wav"
    extract(path, w)
    mix, sr = read(w)
    if separate:
        emit(id=rid, ev="stage", msg="Отделяю голос от музыки и шумов…", p=0.15)
        vox, bg, sr = split_voice(w, k)
    else:
        vox, bg = mix, np.zeros_like(mix)
    CLIPS[k] = {"mix": mix, "vox": vox, "bg": bg, "sr": sr}
    if len(CLIPS) > 12:
        CLIPS.pop(next(iter(CLIPS)))
    return CLIPS[k]


# ------------------------------------------------------------------ распознавание речи (Whisper)
ASR = None


def asr_pipe():
    global ASR
    if ASR is None:
        from transformers import pipeline
        dt = torch.float16 if DEV == "cuda" else torch.float32
        try:
            ASR = pipeline("automatic-speech-recognition", model=CFG["whisper_dir"], dtype=dt, device=DEV)
        except TypeError:
            ASR = pipeline("automatic-speech-recognition", model=CFG["whisper_dir"], torch_dtype=dt, device=DEV)
    return ASR


def to16k(mono, sr):
    import torchaudio
    return torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(mono)).float()[None], sr, 16000)[0].numpy()


def transcribe(mono, sr, lang, timestamps):
    kw = {"task": "transcribe"}
    if lang and lang != "auto":
        kw["language"] = lang
    out = asr_pipe()({"array": to16k(mono, sr), "sampling_rate": 16000}, return_timestamps=bool(timestamps),
                     chunk_length_s=30, batch_size=4, generate_kwargs=kw)
    return out


def voiced_bounds(mono, sr, a, b):
    """Уточняет границы фразы по громкости голоса (Whisper часто захватывает паузы)."""
    i0, i1 = max(0, int(a * sr)), min(len(mono), int(b * sr))
    if i1 - i0 < int(0.1 * sr):
        return a, b
    fr = int(0.02 * sr)
    seg = mono[i0:i1]
    k = len(seg) // fr
    if k < 3:
        return a, b
    e = np.sqrt(np.mean(seg[:k * fr].reshape(k, fr) ** 2, axis=1))
    thr = max(1e-3, float(np.max(e)) * 0.06)
    on = np.where(e > thr)[0]
    if not len(on):
        return a, b
    s = i0 + max(0, on[0] - 3) * fr
    t = i0 + min(k, on[-1] + 4) * fr
    return round(s / sr, 2), round(t / sr, 2)


def cmd_ref(r, rid):
    """Эталон: отрезок → (очистка от фона) → текст эталона."""
    w = TMP / f"ref_{key_of(r['path'], r.get('start'), r.get('end'))}.wav"
    emit(id=rid, ev="stage", msg="Готовлю эталон…", p=0.1)
    extract(r["path"], w, r.get("start"), r.get("end"))
    if r.get("separate", True):
        emit(id=rid, ev="stage", msg="Очищаю эталон от фона…", p=0.3)
        vox, _, sr = split_voice(w, "ref_" + w.stem)
    else:
        vox, sr = read(w)
    mono = vox.mean(axis=1)
    if float(np.sqrt(np.mean(mono ** 2))) < 1e-3:
        raise RuntimeError("В эталоне почти нет голоса — выберите отрезок, где персонаж говорит.")
    clean = TMP / f"{w.stem}_clean.wav"
    sf.write(str(clean), mono, sr, subtype="FLOAT")
    emit(id=rid, ev="stage", msg="Распознаю текст эталона…", p=0.7)
    text = transcribe(mono, sr, r.get("lang"), False)["text"].strip()
    emit(id=rid, ev="result", ref_wav=str(clean), ref_text=text, duration=round(len(mono) / sr, 2))


def cmd_asr(r, rid):
    c = load_clip(r["path"], r.get("separate", True), rid)
    mono = c["vox"].mean(axis=1)
    emit(id=rid, ev="stage", msg="Распознаю речь…", p=0.6)
    out = transcribe(mono, c["sr"], r.get("lang"), True)
    segs = []
    dur = len(mono) / c["sr"]
    for ch in out.get("chunks") or []:
        t = (ch.get("text") or "").strip()
        a, b = ch.get("timestamp") or (None, None)
        if not t:
            continue
        a = float(a or 0)
        b = float(b) if b is not None else dur
        a, b = voiced_bounds(mono, c["sr"], a, min(b, dur))
        segs.append({"start": a, "end": b, "text": t})
    if not segs and (out.get("text") or "").strip():
        a, b = voiced_bounds(mono, c["sr"], 0, dur)
        segs = [{"start": a, "end": b, "text": out["text"].strip()}]
    emit(id=rid, ev="result", segments=segs, duration=round(dur, 2))


# ------------------------------------------------------------------ синтез (OmniVoice)
TTS = None
PROMPTS = {}


def tts_model():
    global TTS
    if TTS is None:
        from omnivoice import OmniVoice
        dt = torch.float16 if DEV == "cuda" else torch.float32
        TTS = OmniVoice.from_pretrained(CFG["omnivoice_dir"], device_map=DEV + (":0" if DEV == "cuda" else ""), dtype=dt)
    return TTS


def voice_prompt(ref_wav, ref_text):
    k = (ref_wav, ref_text)
    if k not in PROMPTS:
        m = tts_model()
        a, sr = read(ref_wav)
        if len(PROMPTS) > 8:
            PROMPTS.clear()
        PROMPTS[k] = m.create_voice_clone_prompt(ref_audio=(torch.from_numpy(a.mean(axis=1)), sr), ref_text=ref_text or None)
    return PROMPTS[k]


def fit(x, n):
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def fade(x, sr, ms=12):
    n = min(len(x) // 2, int(sr * ms / 1000))
    if n > 1:
        x = x.copy()
        x[:n] *= np.linspace(0, 1, n)
        x[-n:] *= np.linspace(1, 0, n)
    return x


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return 0.0
    fr = 1024
    k = len(x) // fr
    if k < 1:
        return float(np.sqrt(np.mean(x * x)))
    e = np.sqrt(np.mean(x[:k * fr].reshape(k, fr) ** 2, axis=1))
    act = e[e > max(1e-4, e.max() * 0.1)]
    return float(np.sqrt(np.mean(act ** 2))) if len(act) else 0.0


PAUSE_RE = None


def split_pauses(s):
    """«Текст [пауза] текст [пауза 0.8] текст» → куски речи с тишиной между ними (по умолчанию 0.5 с).
    Время на речь делится между кусками пропорционально длине текста."""
    import re
    global PAUSE_RE
    if PAUSE_RE is None:
        PAUSE_RE = re.compile(r"\[\s*(?:пауза|pause)\s*([0-9]+(?:[.,][0-9]+)?)?\s*(?:с|сек|s|sec)?\s*\]", re.I)
    a, b = float(s["start"]), float(s["end"])
    parts = PAUSE_RE.split(s["text"])
    texts = [x.strip() for x in parts[0::2]]
    pauses = [float(x.replace(",", ".")) if x else 0.5 for x in parts[1::2]]
    extra = {k: v for k, v in s.items() if k not in ("start", "end", "text")}
    if not pauses:
        return [{**extra, "start": a, "end": b, "text": s["text"].strip()}]
    span = b - a
    # паузы по краям («[пауза] текст») — просто сдвиг начала/конца
    lead = pauses.pop(0) if not texts[0] else 0.0
    if not texts[0]:
        texts.pop(0)
    tail = pauses.pop() if texts and not texts[-1] else 0.0
    if texts and not texts[-1]:
        texts.pop()
    texts = [x for x in texts if x]
    if not texts:
        return []
    pauses = pauses[:max(0, len(texts) - 1)]
    pause_sum = lead + tail + sum(pauses)
    if pause_sum > span * 0.7:   # пауз больше, чем помещается — ужимаем их
        k = span * 0.7 / pause_sum
        lead, tail, pauses = lead * k, tail * k, [x * k for x in pauses]
        pause_sum = span * 0.7
    speech = span - pause_sum
    w = [max(1, len(x)) for x in texts]
    out, t = [], a + lead
    for i, x in enumerate(texts):
        d = speech * w[i] / sum(w)
        out.append({**extra, "start": t, "end": t + d, "text": x})
        t += d + (pauses[i] if i < len(pauses) else 0)
    return out


def cmd_tts(r, rid):
    import torchaudio
    c = load_clip(r["path"], r.get("separate", True), rid)
    sr = c["sr"]
    total = len(c["mix"])
    mono_old = c["vox"].mean(axis=1)
    emit(id=rid, ev="stage", msg="Загружаю модель озвучки (OmniVoice)…", p=0.2)
    m = tts_model()
    refs = r.get("refs") or [r.get("ref")]
    prompts = {}

    def prompt_for(i):
        i = i if 0 <= i < len(refs) and refs[i] else 0
        if i not in prompts:
            prompts[i] = voice_prompt(refs[i]["wav"], refs[i].get("text") or "")
        return prompts[i]
    o = r.get("opts") or {}
    lang = r.get("lang")
    lang = None if not lang or lang == "auto" else lang
    seed = int(o.get("seed", -1))
    if seed < 0:
        seed = int(time.time()) % 100000
    # маска оригинального голоса: 1 — звучит как в ролике, 0 — заменено новой озвучкой.
    # Вне фраз (вздохи, зевки, смех, которых нет в тексте) по умолчанию остаётся оригинал.
    keep_outside = bool(o.get("keep_outside", True))
    mask = np.ones(total, dtype=np.float32) if keep_outside else np.zeros(total, dtype=np.float32)
    ramp = max(1, int(0.03 * sr))

    def set_mask(a, b, v):
        i0, i1 = max(0, int(a * sr)), min(total, int(b * sr))
        if i1 > i0:
            mask[i0:i1] = v

    raw = [s for s in r.get("segments") or []]
    for s in raw:   # сначала «оригинал», потом заменяемые (они важнее)
        if s.get("keep"):
            set_mask(float(s["start"]), float(s["end"]), 1.0)
    for s in raw:
        if not s.get("keep") and (s.get("text") or "").strip():
            set_mask(float(s["start"]), float(s["end"]), 0.0)
    # плавные края (30 мс), чтобы на стыке старого и нового голоса не было щелчков
    cs = np.concatenate([[0.0], np.cumsum(np.pad(mask.astype(np.float64), (ramp // 2, ramp - ramp // 2 - 1), mode="edge"))])
    mask = ((cs[ramp:] - cs[:-ramp]) / ramp).astype(np.float32)[:total]
    clip_lvl = rms(mono_old)
    new = np.zeros(total, dtype=np.float32)
    segs = [x for s in raw if not s.get("keep") and (s.get("text") or "").strip() for x in split_pauses(s)]
    if not segs:
        raise RuntimeError("Нет текста для озвучки — распознайте ролик или впишите фразы")
    for i, s in enumerate(segs):
        a, b = max(0.0, float(s["start"])), min(total / sr, float(s["end"]))
        if b - a < 0.15:
            continue
        emit(id=rid, ev="stage", msg=f"Озвучиваю фразу {i + 1} из {len(segs)}…", p=0.25 + 0.65 * i / len(segs))
        torch.manual_seed(seed + i)
        wav = m.generate(text=s["text"].strip(), language=lang, voice_clone_prompt=prompt_for(int(s.get("ref") or 0)), duration=b - a,
                         num_step=int(o.get("steps", 32)), guidance_scale=float(o.get("guidance", 2.0)),
                         postprocess_output=False)[0]
        wav = np.asarray(wav, dtype=np.float32)
        if m.sampling_rate != sr:
            wav = torchaudio.functional.resample(torch.from_numpy(wav)[None], m.sampling_rate, sr)[0].numpy()
        i0 = int(a * sr)
        n = min(int((b - a) * sr), total - i0)
        wav = fade(fit(wav, n), sr)
        old = mono_old[i0:i0 + n]
        ro, rn = rms(old), rms(wav)
        # громкость: как была в этом месте ролика; если там было тихо (добавленная фраза, пауза, шёпот
        # вместо слов) — как у голоса в среднем по ролику, чтобы новая фраза не утонула
        target = ro if ro >= 0.5 * clip_lvl else clip_lvl
        if target > 1e-4 and rn > 1e-6:
            wav = wav * min(6.0, target / rn)
        if s.get("gain"):
            wav = wav * (10 ** (float(s["gain"]) / 20))
        new[i0:i0 + n] += wav
    emit(id=rid, ev="stage", msg="Сводим звук…", p=0.92)
    ch = c["mix"].shape[1]
    vox_keep = c["vox"] * mask[:, None]           # оригинальный голос там, где он остаётся
    res = c["bg"] + vox_keep + new[:, None] * np.ones((1, ch), dtype=np.float32)
    pk = float(np.max(np.abs(res))) if len(res) else 0
    if pk > 0.99:
        res *= 0.99 / pk
    stem = r["out_name"]
    wav_out = OUT / f"{stem}.wav"
    sf.write(str(wav_out), res, sr, subtype="PCM_16")
    vox_out = OUT / f"{stem}_голос.wav"
    sf.write(str(vox_out), vox_keep.mean(axis=1) + new, sr, subtype="PCM_16")
    result = {"wav": wav_out.name, "voice": vox_out.name, "seed": seed}
    if r.get("video"):
        emit(id=rid, ev="stage", msg="Собираю видео…", p=0.97)
        ext = Path(r["path"]).suffix.lower()
        vout = OUT / f"{stem}{ext if ext in ('.mp4', '.mov', '.mkv') else '.mp4'}"
        ff("-i", r["path"], "-i", wav_out, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
           "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", vout)
        result["video"] = vout.name
    emit(id=rid, ev="result", **result)


def cmd_say(r, rid):
    """Просто текст → звук голосом эталона (без видео). Длина — естественная или заданная;
    [пауза] / [пауза 1.2] — тишина между кусками."""
    import re as _re
    import torchaudio
    emit(id=rid, ev="stage", msg="Загружаю модель озвучки (OmniVoice)…", p=0.1)
    m = tts_model()
    refs = r.get("refs") or []
    ref = refs[int(r.get("ref") or 0)] if int(r.get("ref") or 0) < len(refs) and refs[int(r.get("ref") or 0)] else refs[0]
    prompt = voice_prompt(ref["wav"], ref.get("text") or "")
    o = r.get("opts") or {}
    lang = r.get("lang")
    lang = None if not lang or lang == "auto" else lang
    seed = int(o.get("seed", -1))
    if seed < 0:
        seed = int(time.time()) % 100000
    global PAUSE_RE
    if PAUSE_RE is None:
        split_pauses({"start": 0, "end": 1, "text": "x"})   # компилирует регулярку
    parts = PAUSE_RE.split(r["text"])
    texts = [x.strip() for x in parts[0::2]]
    pauses = [float(x.replace(",", ".")) if x else 0.5 for x in parts[1::2]]
    total_dur = float(r.get("duration") or 0)
    speed = float(r.get("speed") or 1.0)
    pieces = [(t_, (pauses[i] if i < len(pauses) else 0.0)) for i, t_ in enumerate(texts)]
    spoken = [t_ for t_, _ in pieces if t_]
    if not spoken:
        raise RuntimeError("Впишите текст")
    pause_sum = sum(p_ for _, p_ in pieces)
    speech_time = max(0.3 * len(spoken), total_dur - pause_sum - 0.3) if total_dur > 0 else 0   # 0.3 — тишина по краям
    sr = m.sampling_rate
    out = [np.zeros(int(0.15 * sr), dtype=np.float32)]
    w = [max(1, len(x)) for x in spoken]
    k = 0
    for i, (txt, pause) in enumerate(pieces):
        if txt:
            emit(id=rid, ev="stage", msg=f"Озвучиваю кусок {k + 1} из {len(spoken)}…", p=0.2 + 0.7 * k / len(spoken))
            torch.manual_seed(seed + i)
            kw = ({"duration": speech_time * w[k] / sum(w), "postprocess_output": False} if speech_time
                  else ({"speed": speed} if speed != 1.0 else {}))
            wav = m.generate(text=txt, language=lang, voice_clone_prompt=prompt, num_step=int(o.get("steps", 32)),
                             guidance_scale=float(o.get("guidance", 2.0)), **kw)[0]
            wav = np.asarray(wav, dtype=np.float32)
            if speech_time:
                wav = fit(wav, int(kw["duration"] * sr))
            out.append(fade(wav, sr))
            k += 1
        if pause:
            out.append(np.zeros(int(pause * sr), dtype=np.float32))
    out.append(np.zeros(int(0.15 * sr), dtype=np.float32))
    a = np.concatenate(out)
    pk = float(np.max(np.abs(a))) if len(a) else 0
    if pk > 0:
        a = a * (0.89 / pk)            # ровная громкость (≈ −1 дБ по пику)
    g = float(o.get("gain") or 0)
    if g:
        a = np.clip(a * (10 ** (g / 20)), -0.99, 0.99)
    name = _re.sub(r"[^\w\- ]+", "", _re.sub(r"\[[^\]]*\]", " ", r["text"]))
    name = _re.sub(r"\s+", "_", name.strip())[:30] or "text"
    f = OUT / f"text_{time.strftime('%H%M%S')}_{name}.wav"
    sf.write(str(f), a, sr, subtype="PCM_16")
    emit(id=rid, ev="result", wav=f.name, voice=f.name, seed=seed, duration=round(len(a) / sr, 2))


CMDS = {"ref": cmd_ref, "asr": cmd_asr, "tts": cmd_tts, "say": cmd_say}


def main():
    emit(ev="ready")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        rid = r.get("id")
        if r.get("cmd") == "quit":
            break
        fn = CMDS.get(r.get("cmd"))
        if not fn:
            emit(id=rid, ev="error", msg="неизвестная команда")
            continue
        try:
            fn(r, rid)
        except BaseException as e:   # библиотеки иногда вызывают sys.exit — процесс не должен падать
            if isinstance(e, KeyboardInterrupt):
                raise
            traceback.print_exc()
            msg = str(e) or e.__class__.__name__
            if "central directory" in msg or "corrupt" in msg.lower() or "failed finding" in msg:
                msg = "Файл модели повреждён (недокачан). Нажмите «Установить голосовой модуль» ещё раз — он перекачает его."
            if "out of memory" in msg.lower():
                msg = "Не хватило видеопамяти. Нажмите «Освободить VRAM» вверху и попробуйте ещё раз."
            emit(id=rid, ev="error", msg=msg)
        finally:
            if DEV == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
