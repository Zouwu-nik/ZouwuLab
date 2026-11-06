"""Синтез речи Higgs TTS 3 (Boson AI) — постоянный процесс; команды — строки JSON в stdin.

Модель: декодер Qwen3‑4B + общий эмбеддинг/голова на 8 кодбуков аудио (25 кадров/с, delay pattern),
кодек Higgs Audio Tokenizer (24 кГц) лежит внутри того же model.safetensors.
Реализация — на чистом PyTorch + transformers (Qwen3Model, HiggsAudioV2TokenizerModel), без SGLang/vLLM,
поэтому работает под Windows. Формат промпта и автомат кодбуков повторяют официальный движок
SGLang‑Omni (sglang_omni/models/higgs_tts) и порт MLX‑Audio.

Команды:
  ref    — фрагмент видео/звука → эталон голоса (24 кГц моно) + текст эталона (Whisper, если установлен);
  say    — текст (с тегами эмоций/звуков) → речь; несколько голосов (диалог), длинный текст режется по фразам;
  cancel — прервать текущий синтез;  unload_asr — выгрузить Whisper;  quit.
Ответы — строки JSON в stdout: {"id", "ev": "stage"|"result"|"error", ...}.
"""
import json
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

CFG = json.loads(Path(sys.argv[1]).read_text("utf-8"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
OUT = Path(CFG["out_dir"])
OUT.mkdir(parents=True, exist_ok=True)
TMP = Path(CFG["tmp_dir"])
TMP.mkdir(parents=True, exist_ok=True)
LOCK = threading.Lock()


T_START = time.time()


def log(msg):
    sys.stderr.write(time.strftime("%H:%M:%S ") + msg + "\n")
    sys.stderr.flush()


def emit(**kw):
    with LOCK:
        sys.__stdout__.write(json.dumps(kw, ensure_ascii=False) + "\n")
        sys.__stdout__.flush()


sys.stdout = sys.stderr   # чужие print() — в журнал, не в канал команд

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SR = 24000
FPS = 25
N_CB = 8          # кодбуков
CB_V = 1026       # словарь кодбука: 0..1023 — коды, 1024 — BOC, 1025 — EOC
BOC, EOC = 1024, 1025
PLACEHOLDER = -100
# Отмена — файлом-флагом, а не через stdin. На Windows нельзя держать поток, который ждёт чтения из канала stdin:
# пока такое чтение висит, любой запуск подпроцесса или запрос к этому дескриптору из библиотек (transformers,
# torch) блокируется навсегда — так «зависали» загрузка модели и распознавание текста.
CANCEL_FILE = TMP / "cancel.flag"


class _Cancel:
    def is_set(self):
        return CANCEL_FILE.exists()

    def set(self):
        CANCEL_FILE.touch()

    def clear(self):
        try:
            CANCEL_FILE.unlink()
        except FileNotFoundError:
            pass


CANCEL = _Cancel()


class Cancelled(Exception):
    pass


# ------------------------------------------------------------------ загрузка звука (PyAV — есть в движке)
def load_audio(path, start=None, end=None, sr=SR):
    """Любой видео/аудиофайл → моно float32 с частотой sr (отрезок start..end, с)."""
    import av
    out = []
    with av.open(str(path)) as c:
        st = next((s for s in c.streams if s.type == "audio"), None)
        if st is None:
            raise RuntimeError("в файле нет звуковой дорожки")
        if start:
            try:
                c.seek(int(float(start) * 1e6), any_frame=False, backward=True)
            except Exception:
                pass
        rs = av.AudioResampler(format="flt", layout="mono", rate=sr)
        t0 = None
        for fr in c.decode(st):
            for o in rs.resample(fr):
                a = o.to_ndarray().reshape(-1).astype(np.float32)
                ts = float(o.pts * o.time_base) if o.pts is not None else None
                if t0 is None:
                    t0 = ts if ts is not None else 0.0
                out.append(a)
            if end and fr.pts is not None and float(fr.pts * fr.time_base) > float(end) + 0.5:
                break
        for o in rs.resample(None):
            out.append(o.to_ndarray().reshape(-1).astype(np.float32))
    a = np.concatenate(out) if out else np.zeros(0, np.float32)
    t0 = t0 or 0.0
    i0 = max(0, int(round(((float(start) if start else 0.0) - t0) * sr)))
    i1 = int(round((float(end) - t0) * sr)) if end else len(a)
    return np.ascontiguousarray(a[i0:max(i0, i1)])


def active_rms(x, fr=480):
    x = np.asarray(x, np.float64)
    k = len(x) // fr
    if k < 1:
        return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
    e = np.sqrt(np.mean(x[:k * fr].reshape(k, fr) ** 2, axis=1))
    act = e[e > max(1e-4, e.max() * 0.1)]
    return float(np.sqrt(np.mean(act ** 2))) if len(act) else 0.0


def trim_silence(x, sr=SR, keep=0.06):
    """Обрезает тишину по краям фразы (оставляет keep с)."""
    fr = int(0.02 * sr)
    k = len(x) // fr
    if k < 3:
        return x
    e = np.sqrt(np.mean(x[:k * fr].reshape(k, fr) ** 2, axis=1))
    thr = max(2e-3, float(e.max()) * 0.02)
    on = np.where(e > thr)[0]
    if not len(on):
        return x
    a = max(0, on[0] * fr - int(keep * sr))
    b = min(len(x), (on[-1] + 1) * fr + int(keep * sr))
    return x[a:b]


def fades(x, sr=SR, fin=0.03, fout=0.015):
    x = x.copy()
    n1, n2 = min(len(x) // 2, int(fin * sr)), min(len(x) // 2, int(fout * sr))
    if n1 > 1:
        x[:n1] *= np.linspace(0, 1, n1, dtype=np.float32)
    if n2 > 1:
        x[-n2:] *= np.linspace(1, 0, n2, dtype=np.float32)
    return x


# ------------------------------------------------------------------ распознавание текста эталона (Whisper)
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


def unload_asr():
    global ASR
    if ASR is not None:
        ASR = None
        if DEV == "cuda":
            torch.cuda.empty_cache()


def transcribe(mono24, lang):
    import numpy as _np
    x = mono24
    # 24 кГц → 16 кГц для Whisper (линейная интерполяция достаточна для распознавания)
    n = int(len(x) * 16000 / SR)
    x16 = _np.interp(_np.linspace(0, len(x) - 1, n), _np.arange(len(x)), x).astype(_np.float32)
    kw = {"task": "transcribe"}
    if lang and lang != "auto":
        kw["language"] = lang
    return asr_pipe()({"array": x16, "sampling_rate": 16000}, generate_kwargs=kw)["text"].strip()


# ------------------------------------------------------------------ модель Higgs TTS 3
class Higgs:
    def __init__(self, mdir, codec_cfg, dev, progress=None):
        from safetensors import safe_open
        t0 = time.time()
        log(f"загрузка модели: импорт библиотек {time.time() - T_START:.1f} с от старта процесса")
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
        self.dev = torch.device(dev)
        self.dt = torch.bfloat16 if dev == "cuda" else torch.float32
        cfg = json.loads((Path(mdir) / "config.json").read_text("utf-8"))
        enc = cfg.get("audio_encoder_config") or {}
        if enc.get("encoder_type", "discrete") != "discrete":
            raise RuntimeError("Эта сборка Higgs не для синтеза речи (нужна TTS‑версия)")
        self.ncb = int(enc.get("num_codebooks", N_CB))
        self.cbv = int(enc.get("vocab_size", CB_V))
        tc = dict(cfg.get("text_config") or {})
        for k in ("architectures", "torch_dtype", "dtype", "transformers_version", "_name_or_path"):
            tc.pop(k, None)
        theta = tc.pop("rope_theta", None) or (tc.get("rope_parameters") or {}).get("rope_theta") or 1_000_000
        tc["rope_parameters"] = {"rope_type": "default", "rope_theta": float(theta)}
        tc.pop("rope_scaling", None)
        qc = Qwen3Config(**tc)
        qc.rope_theta = float(theta)   # для старых версий transformers, где параметр ещё отдельный
        try:
            qc._attn_implementation = "sdpa"
        except Exception:
            pass
        log("строю каркас Qwen3‑4B")
        with torch.device("meta"):
            body = Qwen3Model(qc)
        # сначала dtype на «пустом» (meta) устройстве — иначе на видеокарте временно окажется fp32‑копия (16 ГБ)
        body = body.to(self.dt).to_empty(device=self.dev)
        # буферы RoPE на meta не считаются — пересоздаём модуль на настоящем устройстве
        rcls = type(body.rotary_emb)
        try:
            rot = rcls(config=qc, device=self.dev)
        except TypeError:   # в новых transformers у RoPE нет параметра device
            rot = rcls(config=qc)
        body.rotary_emb = rot.to(self.dev)   # без смены dtype: частоты RoPE остаются fp32
        log(f"каркас на видеокарте за {time.time() - t0:.1f} с")
        hid = qc.hidden_size
        self.audio_w = torch.empty(self.ncb * self.cbv, int(enc.get("out_dim", hid)), device=self.dev, dtype=self.dt)
        # веса: имена чекпоинта Higgs → Qwen3Model
        params = dict(body.named_parameters())
        need = set(params)
        files = sorted({v for v in json.loads((Path(mdir) / "model.safetensors.index.json").read_text("utf-8"))["weight_map"].values()}) \
            if (Path(mdir) / "model.safetensors.index.json").exists() else ["model.safetensors"]
        got_audio = False
        codec_sd = {}
        total = sum((Path(mdir) / fn).stat().st_size for fn in files) or 1
        done_b, last = 0, 0.0
        log(f"каркас модели готов за {time.time() - t0:.1f} с; читаю веса {total / 1e9:.1f} ГБ")
        t1 = time.time()
        for fn in files:
            with safe_open(str(Path(mdir) / fn), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if progress and time.time() - last > 0.7:
                        last = time.time()
                        progress(done_b / total)
                        if int(done_b / 1e9) != int(getattr(self, "_lg", -1)):
                            self._lg = int(done_b / 1e9)
                            log(f"  прочитано {done_b / 1e9:.1f} ГБ за {time.time() - t1:.0f} с")
                    if k.startswith("tied.embedding.modality_embeddings.0.model."):
                        t = f.get_tensor(k)
                        done_b += t.numel() * t.element_size()
                        codec_sd[k[len("tied.embedding.modality_embeddings.0.model."):]] = t
                        continue
                    if k.startswith("tied.embedding.modality_embeddings.0.embedding."):
                        t = f.get_tensor(k)
                        done_b += t.numel() * t.element_size()
                        self.audio_w.copy_(t.to(self.dt))
                        got_audio = True
                        continue
                    if k.startswith("tied.embedding.text_embedding."):
                        name = "embed_tokens." + k[len("tied.embedding.text_embedding."):]
                    elif k.startswith("tied.head.text_head."):   # связана с эмбеддингами — запасной источник
                        name = "embed_tokens." + k[len("tied.head.text_head."):]
                        if name not in need:
                            continue
                    elif k.startswith("body.layers."):
                        name = "layers." + k[len("body.layers."):]
                    elif k.startswith("body.norm."):
                        name = "norm." + k[len("body.norm."):]
                    elif k.startswith(("model.", "body.")) and "embed_tokens" in k:
                        name = "embed_tokens." + k.split("embed_tokens.", 1)[1]
                    else:
                        continue   # прочие tied.head.* — голова связана с эмбеддингами
                    p = params.get(name)
                    if p is None:
                        continue
                    t = f.get_tensor(k)
                    done_b += t.numel() * t.element_size()
                    if tuple(t.shape) != tuple(p.shape):
                        raise RuntimeError(f"Форма веса {k}: {tuple(t.shape)}, ожидалось {tuple(p.shape)}")
                    with torch.no_grad():
                        p.copy_(t.to(self.dt))
                    need.discard(name)
        if need or not got_audio:
            miss = sorted(need)[:5] + ([] if got_audio else ["audio embedding"])
            raise RuntimeError("В файлах модели не хватает весов: " + ", ".join(miss) + " — нажмите «Установить» ещё раз")
        sec = time.time() - t1
        log(f"веса прочитаны за {sec:.1f} с ({total / 1e6 / max(sec, 0.01):.0f} МБ/с)"
            + (" — диск медленный: перенесите папку программы на SSD" if total / 1e6 / max(sec, 0.01) < 250 else ""))
        self.body = body.eval()
        for p in self.body.parameters():
            p.requires_grad_(False)
        self.offsets = (torch.arange(self.ncb, device=self.dev) * self.cbv).view(1, self.ncb)
        # токенизатор
        raw = Tokenizer.from_file(str(Path(mdir) / "tokenizer.json"))
        self.tok = PreTrainedTokenizerFast(tokenizer_object=raw)
        vocab = dict(self.tok.get_added_vocab())
        for t in ("<|tts|>", "<|ref_audio|>", "<|text|>", "<|audio|>"):
            if t not in vocab:
                raise RuntimeError(f"В токенизаторе нет служебного токена {t}")
        self.t_tts, self.t_ref_audio, self.t_text, self.t_audio = (vocab[t] for t in ("<|tts|>", "<|ref_audio|>", "<|text|>", "<|audio|>"))
        self.t_ref_text = vocab.get("<|ref_text|>")
        # кодек (24 кГц) — из того же чекпоинта
        t2 = time.time()
        self.codec = self._codec(codec_cfg, codec_sd)
        del codec_sd
        log(f"кодек готов за {time.time() - t2:.1f} с; модель целиком — {time.time() - t0:.1f} с")

    def _codec(self, codec_cfg, sd):
        from transformers import HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel
        model = None
        if sd:
            conf = HiggsAudioV2TokenizerConfig.from_json_file(codec_cfg)
            try:   # без случайной инициализации — веса всё равно загрузятся из файла
                from transformers.modeling_utils import no_init_weights
                with no_init_weights():
                    model = HiggsAudioV2TokenizerModel(conf)
            except Exception:
                model = HiggsAudioV2TokenizerModel(conf)
            model = model.float().eval()
            missing, _ = model.load_state_dict(sd, strict=False)
            if len(missing) > len(sd) // 2:
                model = None
        if model is None:   # запасной путь: кодек из голосового модуля (eustlb/higgs-audio-v2-tokenizer)
            alt = CFG.get("codec_alt_dir")
            if not alt or not (Path(alt) / "config.json").exists():
                raise RuntimeError("Не удалось загрузить аудио‑кодек Higgs из файлов модели")
            model = HiggsAudioV2TokenizerModel.from_pretrained(alt).float().eval()
        model = model.to(self.dev)
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    # --- коды ⇄ звук
    @torch.no_grad()
    def encode(self, mono24):
        x = torch.from_numpy(np.ascontiguousarray(mono24, np.float32)).view(1, 1, -1)
        if x.shape[-1] < SR:
            x = F.pad(x, (0, SR - x.shape[-1]))
        x = x.to(self.dev)
        ctx = torch.backends.cudnn.flags(enabled=False) if self.dev.type == "cuda" else _null()
        with ctx:
            codes = self.codec.encode(x).audio_codes        # [1, N, T]
        return codes[0].transpose(0, 1).to(torch.long)      # [T, N]

    @torch.no_grad()
    def decode(self, codes_TN):
        if codes_TN.shape[0] == 0:
            return np.zeros(0, np.float32)
        codes = torch.where(codes_TN >= self.cbv - 2, torch.zeros_like(codes_TN), codes_TN)
        wav = self.codec.decode(codes.transpose(0, 1).unsqueeze(0).to(self.dev)).audio_values
        return wav.reshape(-1).float().cpu().numpy()

    @staticmethod
    def delay(codes_TN):
        T, N = codes_TN.shape
        out = torch.full((T + N - 1, N), EOC, dtype=torch.long, device=codes_TN.device)
        idx = torch.arange(T + N - 1, device=codes_TN.device)
        for c in range(N):
            out[idx < c, c] = BOC
            out[c:c + T, c] = codes_TN[:, c]
        return out

    @staticmethod
    def undelay(delayed):
        L, N = delayed.shape
        rows = L - (N - 1)
        if rows <= 0:
            return delayed.new_empty((0, N))
        out = torch.empty((rows, N), dtype=delayed.dtype, device=delayed.device)
        for c in range(N):
            out[:, c] = delayed[c:c + rows, c]
        return out

    def embed_codes(self, codes_LN):
        return F.embedding(codes_LN + self.offsets, self.audio_w).sum(dim=-2)

    # --- генерация
    @torch.no_grad()
    def generate(self, text, refs=(), temperature=0.8, top_p=0.8, top_k=30, max_rows=2048, progress=None):
        """refs: [(delayed_codes_LN, ref_text|None)] → звук 24 кГц (np.float32)."""
        ids, segs = [self.t_tts], []
        for codes, rtext in refs:
            if rtext and self.t_ref_text is not None:
                ids.append(self.t_ref_text)
                ids += self.tok.encode(rtext, add_special_tokens=False)
            ids.append(self.t_ref_audio)
            segs.append((len(ids), codes))
            ids += [PLACEHOLDER] * int(codes.shape[0])
        ids.append(self.t_text)
        ids += self.tok.encode(text, add_special_tokens=False)
        ids.append(self.t_audio)
        t = torch.tensor(ids, dtype=torch.long, device=self.dev)
        emb = self.body.embed_tokens(torch.where(t == PLACEHOLDER, torch.zeros_like(t), t))
        for start, codes in segs:
            emb[start:start + codes.shape[0]] = self.embed_codes(codes.to(self.dev)).to(emb.dtype)
        out = self.body(inputs_embeds=emb[None], use_cache=True)
        cache, h = out.past_key_values, out.last_hidden_state[0, -1]
        N = self.ncb
        delay_count, eoc_left, done = 0, None, False
        rows = []
        cb = torch.arange(N, device=self.dev)
        for step in range(int(max_rows)):
            if step % 8 == 0 and CANCEL.is_set():
                raise Cancelled()
            logits = F.linear(h, self.audio_w).float().view(N, self.cbv)
            codes = sample(logits, temperature, top_p, top_k)
            if delay_count < N:                      # окно задержки: старшие кодбуки ещё «не начались»
                codes = torch.where(cb > delay_count, torch.full_like(codes, BOC), codes)
                delay_count += 1
            elif eoc_left is not None:               # кодбук 0 уже закончил — досчитываем хвост
                eoc_left -= 1
                if eoc_left <= 0:
                    done = True
            elif int(codes[0]) == EOC:
                if N <= 2:
                    done = True
                else:
                    eoc_left = N - 2
            rows.append(codes)
            if done:
                break
            if progress and step % 10 == 0:
                progress(step)
            e = self.embed_codes(codes.view(1, N)).to(self.dt)
            out = self.body(inputs_embeds=e.view(1, 1, -1), past_key_values=cache, use_cache=True)
            cache, h = out.past_key_values, out.last_hidden_state[0, -1]
        del cache
        delayed = torch.stack(rows, 0) if rows else torch.zeros((0, N), dtype=torch.long, device=self.dev)
        return self.decode(self.undelay(delayed)), done


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def sample(logits_NV, temperature, top_p, top_k):
    if temperature <= 1e-5:
        return logits_NV.argmax(-1)
    x = logits_NV / temperature
    if top_k and 0 < top_k < x.shape[-1]:
        kth = x.topk(int(top_k), dim=-1).values[:, -1:]
        x = torch.where(x < kth, torch.full_like(x, float("-inf")), x)
    if top_p and top_p < 1.0:
        sx, si = torch.sort(x, descending=True, dim=-1)
        cp = sx.softmax(-1).cumsum(-1)
        rm = cp > top_p
        rm[..., 1:] = rm[..., :-1].clone()
        rm[..., 0] = False
        mask = torch.zeros_like(rm).scatter_(-1, si, rm)
        x = torch.where(mask, torch.full_like(x, float("-inf")), x)
    return x.softmax(-1).multinomial(1).squeeze(-1)


MODEL = None


NEED_GB = 10.2   # веса 8.6 ГБ + кодек 0.8 ГБ + запас на генерацию


def vram_gate(rid):
    """Не начинаем загрузку, пока на видеокарте нет места: иначе Windows молча уводит модель в общую (системную)
    память, и всё «висит» десятки минут вместо ошибки. Движок освобождает память не мгновенно — ждём до 40 с."""
    if DEV != "cuda":
        return
    t0 = time.time()
    while True:
        free, total = torch.cuda.mem_get_info()
        log(f"видеопамять: свободно {free / 1e9:.1f} из {total / 1e9:.1f} ГБ")
        if free / 1e9 >= NEED_GB:
            return
        if time.time() - t0 > 40:
            raise RuntimeError(f"Не хватает видеопамяти: свободно {free / 1e9:.1f} ГБ из {total / 1e9:.1f}, а Higgs TTS 3 нужно "
                               f"~{NEED_GB:.0f} ГБ. Её держит что-то ещё: генерация картинок/видео, помощник (LLM), вкладка «Голос», "
                               "игры или браузер с видео. Закройте лишнее и нажмите «Озвучить» ещё раз.")
        if rid:
            emit(id=rid, ev="stage", msg=f"Жду, пока освободится видеопамять: свободно {free / 1e9:.1f} из нужных {NEED_GB:.0f} ГБ…", p=0.02)
        time.sleep(2)


def model(rid=None):
    global MODEL
    if MODEL is None:
        vram_gate(rid)
        size = 9.3

        def prog(f):
            if rid:
                emit(id=rid, ev="stage", msg=f"Загружаю Higgs TTS 3 в видеопамять: {f * size:.1f} из {size} ГБ…", p=round(0.02 + 0.03 * f, 3))
        MODEL = Higgs(CFG["model_dir"], CFG["codec_cfg"], DEV, progress=prog)
    return MODEL


def cmd_warm(r, rid):
    """Заранее загрузить модель (при открытии вкладки), чтобы «Озвучить» начиналось сразу."""
    model(rid)
    emit(id=rid, ev="result", ok=True)


# ------------------------------------------------------------------ разбиение текста
TAG_RE = re.compile(r"<\|(?:[a-z_]+:[a-z_]+|reset)\|>")
SENT_RE = re.compile(r"(?<=[.!?…。！？])\s+")


def chunks_of(text, limit=260):
    """Абзацы → предложения → куски не длиннее limit символов (теги не считаются)."""
    out = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = re.sub(r"\s*\n\s*", " ", para).strip()
        if not para:
            continue
        cur, first = "", True
        for s in SENT_RE.split(para):
            s = s.strip()
            if not s:
                continue
            ln = lambda z: len(TAG_RE.sub("", z))   # noqa: E731
            if cur and ln(cur) + 1 + ln(s) > limit:
                out.append((cur, first))
                cur, first = s, False
            else:
                cur = (cur + " " + s).strip()
            while ln(cur) > limit * 1.6:   # одно огромное предложение — режем по запятым/пробелам
                comma = max(cur.rfind(", ", 0, limit), cur.rfind("; ", 0, limit), cur.rfind(" — ", 0, limit))
                cut = comma if comma > limit // 2 else cur.rfind(" ", 0, limit)
                cut = cut if cut > limit // 3 else limit
                for tm in TAG_RE.finditer(cur):   # не резать посреди тега
                    if tm.start() < cut < tm.end():
                        cut = max(1, tm.start() - 1)
                out.append((cur[:cut + 1].strip(), first))
                cur, first = cur[cut + 1:].strip(), False
        if cur:
            out.append((cur, first))
    return out   # [(текст, начало_абзаца)]


# ------------------------------------------------------------------ команды
def cmd_ref(r, rid):
    emit(id=rid, ev="stage", msg="Достаю звук…", p=0.1)
    a = load_audio(r["path"], r.get("start"), r.get("end"))
    if len(a) < int(0.8 * SR):
        raise RuntimeError("Отрезок слишком короткий — нужно хотя бы 3 секунды речи (лучше 5–15 с)")
    a = a[:int(30 * SR)]
    if active_rms(a) < 1e-3:
        raise RuntimeError("В отрезке почти нет звука — выберите место, где человек говорит")
    a = trim_silence(a, keep=0.15)
    pk = float(np.max(np.abs(a))) or 1.0
    a = (a * min(4.0, 0.89 / pk)).astype(np.float32)
    dest = Path(r["dest"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), a, SR, subtype="PCM_16")
    text = (r.get("text") or "").strip()
    if not text and r.get("transcribe") and CFG.get("whisper_dir") and (Path(CFG["whisper_dir"]) / "config.json").exists():
        emit(id=rid, ev="stage", msg="Распознаю текст эталона (Whisper)…", p=0.5)
        text = transcribe(a, r.get("lang") or "auto")
        unload_asr()
    emit(id=rid, ev="result", wav=str(dest), text=text, duration=round(len(a) / SR, 2))


REF_CACHE = {}


def ref_codes(m, wav, text):
    st = Path(wav).stat()
    k = (wav, st.st_size, st.st_mtime)
    if k not in REF_CACHE:
        a, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        a = a.mean(axis=1)
        if sr != SR:
            n = int(len(a) * SR / sr)
            a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
        a = a[:int(30 * SR)]
        if len(REF_CACHE) > 16:
            REF_CACHE.clear()
        REF_CACHE[k] = m.delay(m.encode(a))
    return REF_CACHE[k], (text or "").strip() or None


RUN_RE = re.compile(r"(?:<\|(?:[a-z_]+:[a-z_]+|reset)\|>\s*)+")
DELIV_RE = re.compile(r"<\|(?:emotion|style|prosody):(?!pause\|>|long_pause\|>)[a-z_]+\|>")


def carry_state(text, state):
    """Теги подачи (эмоция, стиль, темп, тон, выразительность) действуют, пока их не сменит новая группа
    или <|reset|> (так работает Higgs TTS 3). Длинный текст мы режем на куски, и каждый кусок модель читает
    с чистого листа — поэтому подача, действовавшая в конце прошлого куска, переносится в начало следующего.
    Возвращает (текст для модели, состояние в конце куска)."""
    lead = re.match(r"^\s*((?:<\|(?:[a-z_]+:[a-z_]+|reset)\|>\s*)*)", text).group(1)
    own = bool(DELIV_RE.search(lead) or "<|reset|>" in lead)
    out = text if own or not state else state + text
    st = "" if own else state
    for run in RUN_RE.finditer(text):
        tags = run.group(0)
        d = "".join(DELIV_RE.findall(tags))
        if "<|reset|>" in tags:
            st = d
        elif d:
            st = d   # новая группа заменяет прежнюю целиком
    return out, st


def styled(text):
    return bool(re.search(r"<\|style:", text))


def cmd_say(r, rid):
    CANCEL.clear()
    unload_asr()
    if MODEL is None:
        emit(id=rid, ev="stage", msg="Загружаю Higgs TTS 3…", p=0.02)
    m = model(rid)
    o = r.get("opts") or {}
    temp = float(o.get("temperature", 0.8))
    top_p = float(o.get("top_p", 0.8))
    top_k = int(o.get("top_k", 30))
    seed = int(o.get("seed", -1))
    if seed < 0:
        seed = int(time.time() * 1000) % 1_000_000
    gap = float(o.get("gap", 0.25))
    pgap = float(o.get("para_gap", 0.6))
    # план: [(текст, голос, новый_абзац)]
    plan = []
    for seg in r.get("segments") or []:
        state = ""   # у каждой реплики/говорящего — своя подача
        for i, (t, first) in enumerate(chunks_of(seg["text"])):
            t2, state = carry_state(t, state)
            plan.append({"text": t2, "voice": seg.get("voice") or {}, "para": first, "speaker": seg.get("speaker") or ""})
    if not plan:
        raise RuntimeError("Впишите текст")
    est_rows = [max(40, int(len(TAG_RE.sub("", p["text"])) / 13 * FPS) + 20) for p in plan]
    total_est = float(sum(est_rows))
    done_est = 0.0
    pieces, auto_ref, first_piece, level = [], {}, None, None
    t0 = time.time()
    for i, p in enumerate(plan):
        v = p["voice"]
        if v.get("wav"):
            refs = [ref_codes(m, v["wav"], v.get("text"))]
            key = None
        else:   # случайный голос: первый кусок задаёт голос для всех следующих кусков этого «случайного» говорящего
            key = v.get("rand_key") or "rand"
            if key not in auto_ref and styled(p["text"]):
                # первый кусок — шёпот/крик/пение: голос по нему не снимаешь. Сначала тот же голос (тот же seed)
                # ровно произносит первую фразу — это образец для всех кусков, сам он в результат не идёт
                emit(id=rid, ev="stage", msg="Подбираю голос…", p=round(0.05 + 0.9 * done_est / total_est, 3))
                plain = re.sub(r"\s+", " ", TAG_RE.sub("", p["text"])).strip()
                plain = (SENT_RE.split(plain)[0] if len(plain) > 160 else plain)[:200] or plain[:200]
                torch.manual_seed(seed + i)
                if DEV == "cuda":
                    torch.cuda.manual_seed_all(seed + i)
                cw, _ = m.generate(plain, [], temp, top_p, top_k, max_rows=600)
                cw = trim_silence(cw)
                if len(cw) > SR:
                    auto_ref[key] = (m.delay(m.encode(cw[:int(25 * SR)])), plain)
                    if first_piece is None:
                        first_piece = (fades(cw.astype(np.float32)), plain)
            refs = [auto_ref[key]] if key in auto_ref else []

        def prog(step, base=done_est, est=est_rows[i], n=i):
            frac = (base + min(step, est * 0.98)) / total_est
            emit(id=rid, ev="stage", msg=f"Озвучиваю {n + 1} из {len(plan)}…", p=round(0.05 + 0.9 * frac, 3))
        prog(0)
        torch.manual_seed(seed + i)
        if DEV == "cuda":
            torch.cuda.manual_seed_all(seed + i)
        wav, finished = m.generate(p["text"], refs, temp, top_p, top_k, max_rows=int(o.get("max_rows", 2048)), progress=prog)
        done_est += est_rows[i]
        wav = trim_silence(wav)
        if key is not None and key not in auto_ref and len(wav) > SR:
            clip = wav[:int(25 * SR)]
            auto_ref[key] = (m.delay(m.encode(clip)), TAG_RE.sub("", p["text"]).strip())
        lv = active_rms(wav)
        if styled(p["text"]):
            pass   # шёпот должен остаться тихим, крик — громким
        elif level is None:
            level = lv
        elif lv > 1e-4:   # ровная громкость между кусками
            wav = wav * min(3.0, max(0.33, level / lv))
        wav = fades(wav.astype(np.float32))
        if i > 0:
            pieces.append(np.zeros(int((pgap if p["para"] else gap) * SR), np.float32))
        pieces.append(wav)
        if first_piece is None and not v.get("wav") and not styled(p["text"]):
            first_piece = (wav, TAG_RE.sub("", p["text"]).strip())
    a = np.concatenate([np.zeros(int(0.12 * SR), np.float32)] + pieces + [np.zeros(int(0.2 * SR), np.float32)])
    pk = float(np.max(np.abs(a))) if len(a) else 0
    if pk > 0:
        a = a * (0.89 / pk)
    g = float(o.get("gain") or 0)
    if g:
        a = np.clip(a * (10 ** (g / 20)), -0.99, 0.99)
    stem = r["out_name"]
    f = OUT / f"{stem}.wav"
    sf.write(str(f), a, SR, subtype="PCM_16")
    res = {"wav": f.name, "seed": seed, "duration": round(len(a) / SR, 2), "seconds": round(time.time() - t0, 1),
           "pieces": len(plan)}
    if first_piece is not None:   # для «Сохранить как голос»: начало речи случайного голоса + его текст
        fp = OUT / f"{stem}.ref.wav"
        sf.write(str(fp), first_piece[0][:int(20 * SR)], SR, subtype="PCM_16")
        res["ref_wav"], res["ref_text"] = fp.name, first_piece[1]
    emit(id=rid, ev="result", **res)


CMDS = {"ref": cmd_ref, "say": cmd_say, "warm": cmd_warm}
def main():
    log(f"процесс синтеза запущен: torch {torch.__version__}, устройство {DEV}"
        + (f" ({torch.cuda.get_device_name(0)})" if DEV == "cuda" else "") + f", импорт {time.time() - T_START:.1f} с")
    CANCEL.clear()
    emit(ev="ready")
    for line in sys.stdin:   # читаем stdin только в главном потоке и только когда ничего не считаем
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        rid = r.get("id")
        c = r.get("cmd")
        if c == "quit":
            break
        if c == "cancel":
            continue
        if c == "unload_asr":
            unload_asr()
            continue
        fn = CMDS.get(c)
        if not fn:
            emit(id=rid, ev="error", msg="неизвестная команда")
            continue
        try:
            fn(r, rid)
        except Cancelled:
            emit(id=rid, ev="error", msg="Остановлено")
        except BaseException as e:   # библиотеки иногда вызывают sys.exit — процесс не должен падать
            if isinstance(e, KeyboardInterrupt):
                raise
            traceback.print_exc()
            msg = str(e) or e.__class__.__name__
            if "out of memory" in msg.lower():
                msg = ("Не хватило видеопамяти (Higgs TTS 3 нужно ~10–11 ГБ). Закройте другие программы, занимающие "
                       "видеокарту (игры, браузер с видео, другие нейросети), и попробуйте ещё раз.")
            elif "header" in msg.lower() or "safetensor" in msg.lower() or "incomplete" in msg.lower():
                msg = "Файл модели повреждён (недокачан). Нажмите «Установить» ещё раз — он докачает его. (" + msg[:160] + ")"
            emit(id=rid, ev="error", msg=msg)
        finally:
            CANCEL.clear()
            if DEV == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
