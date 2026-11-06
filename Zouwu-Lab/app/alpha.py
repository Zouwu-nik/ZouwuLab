"""Прозрачность для готовых картинок: вырезка изменений после правки, вырезка с чёрного фона.

wants_alpha(prompt)  — нужен ли PNG с прозрачностью (слова «альфа», «прозрачность», «на чёрном фоне»…),
                       промпт без этих слов и подсказка, какой способ вырезки подойдёт.
black_key(img, mode) — картинка, сгенерированная на чёрном фоне → RGBA.
                       "object": фон убирается целиком, сам объект непрозрачный (плашки, кнопки, предметы);
                       "glow":   прозрачность по яркости (Unmult) — свечение, дым, огонь, блики, частицы.
diff_layer(orig, edited) — правка кадра → только то, что добавила нейросеть, на прозрачном фоне,
                       на том же месте. Положить поверх исходного кадра — получится результат правки.
"""
import re

import numpy as np
from PIL import Image, ImageFilter

try:
    from scipy import ndimage as ndi
except Exception:  # без scipy — упрощённо (без заливки дырок и чистки мусора)
    ndi = None

def E(p):
    return re.compile(p, re.I)

TRIGGERS = [E(r"альф"), E(r"прозрачн"), E(r"на\s+ч[её]рн\w*\s+фон"), E(r"без\s+фона"), E(r"вырез\w*\s+(по\s+)?альф"),
            E(r"\balpha\b"), E(r"transparen"), E(r"on\s+(a\s+)?(pure\s+|solid\s+)?black\s+background"), E(r"no\s+background")]
STRIP = [E(r"\s*,?\s*(с\s+|на\s+)?альф\w*[\s-]*канал\w*"), E(r"\s*,?\s*(с\s+|в\s+)?альф\w*"),
         E(r"\s*,?\s*(с\s+)?прозрачност\w*"), E(r"\s*,?\s*(на\s+)?прозрачн\w+\s+фон\w*"),
         E(r"\s*,?\s*(на\s+)?ч[её]рн\w*\s+фон\w*"), E(r"\s*,?\s*без\s+фона"),
         E(r"\s*,?\s*(with\s+)?(an\s+)?alpha(\s+channel)?\b"), E(r"\s*,?\s*(on\s+)?(a\s+)?transparent\s+background"),
         E(r"\s*,?\s*with\s+transparency"), E(r"\s*,?\s*(on\s+)?(a\s+)?(pure\s+|solid\s+)?black\s+background"),
         E(r"\s*,?\s*(in\s+)?png\b")]
GLOW = E(r"свечени|сияни|glow|блик|flare|огон|огн[яи]|пламя|fire|flame|дым|smoke|туман|fog|искр|spark|частиц|particle|"
         r"молни|lightning|неон|neon|луч|rays?\b|световой|light\s*leak|bokeh|боке|магическ|magic|энерги|energy|взрыв|explosion")

SUFFIX_NEW = ("Isolated on a solid pure black background (#000000): plain flat black everywhere around the subject, "
              "no floor, no shadow on the background, no scene, no frame, the whole subject fully visible.")
SUFFIX_STYLED = ("Match the colour palette, lighting and visual style of the reference image, but draw only this element, "
                 "isolated on a solid pure black background (#000000), no scene, the whole element fully visible.")
SUFFIX_EDIT = " Keep everything else in the image exactly unchanged."


def wants_alpha(prompt):
    """(нужна ли альфа, промпт без «альфа/прозрачность/…», "glow" | "object")."""
    t = (prompt or "").lower()
    need = any(r.search(t) for r in TRIGGERS)
    clean = prompt or ""
    if need:
        for r in STRIP:
            clean = r.sub("", clean)
        clean = re.sub(r"\s{2,}", " ", clean)
        clean = re.sub(r"\s+([,.;:!?])", r"\1", clean)
        clean = re.sub(r"([,;:])\s*([,.;:])", r"\2", clean).strip(" ,;:")
        clean = clean or (prompt or "")
    kind = "glow" if GLOW.search(t) else "object"
    return need, clean.strip(), kind


# ------------------------------------------------------------------ помощники
def _f(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _blur(a, sigma):
    if sigma <= 0:
        return a
    if ndi is not None:
        if a.ndim == 3:
            return ndi.gaussian_filter(a, (sigma, sigma, 0))
        return ndi.gaussian_filter(a, sigma)
    if a.ndim == 3:
        return np.stack([_blur(a[..., c], sigma) for c in range(a.shape[2])], -1)
    im = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(sigma)), dtype=np.float32) / 255.0


def _clean(mask, min_frac=0.0006, fill=True):
    """Убрать мелкие пятнышки, залить дырки внутри объектов."""
    if ndi is None or not mask.any():
        return mask
    lab, n = ndi.label(mask)
    if n:
        sizes = ndi.sum(mask, lab, range(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= max(16, min_frac * mask.size)
        if not keep.any():  # всё мелкое — оставить самое крупное
            keep[1 + int(np.argmax(sizes))] = True
        mask = keep[lab]
    if fill:
        mask = ndi.binary_fill_holes(mask)
    return mask


def _rgba(F, a):
    F = np.where((a < 0.004)[..., None], 0, F)
    out = np.dstack([np.clip(F, 0, 1), np.clip(a, 0, 1)[..., None]])
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGBA")


def _unmix(Eimg, Bg, a):
    """Цвет слоя F из E = a·F + (1−a)·Bg."""
    a3 = np.maximum(a, 1e-3)[..., None]
    F = (Eimg - (1 - a)[..., None] * Bg) / a3
    return np.where((a > 0.995)[..., None], Eimg, F)


# ------------------------------------------------------------------ картинка на чёрном фоне
def _k(sens):
    """Чувствительность 0..1 (0.5 — обычная) → множитель порога: больше чувствительность — ниже порог."""
    return 2 ** ((0.5 - float(sens)) * 3)


def _mthr(matte, sens):
    """Маска нейросети → прозрачность: внутри объекта — непрозрачно, мягкий только край."""
    thr = float(np.clip(0.5 * _k(sens) ** 0.5, 0.12, 0.88))
    return np.clip(np.asarray(matte, np.float32) / thr, 0, 1)


def _with_holes(core, a, fill):
    """Залитые дырки внутри объекта — непрозрачные."""
    if fill and ndi is not None:
        filled = ndi.binary_fill_holes(core)
        a = np.where(filled & ~core, 1.0, a)
        core = filled
    return core, a


def black_key(img, mode="object", matte=None, sens=0.5, fill=True):
    """matte — маска нейросети вырезки (0..1) или None (тогда — по порогу от цвета фона)."""
    f = _f(img)
    H, W = f.shape[:2]
    if mode == "glow":  # классический Unmult: прозрачность = яркость
        bp = max(0.0, (0.5 - float(sens)) * 0.3)  # меньше чувствительность — тёмная дымка уходит
        a = np.clip((f.max(axis=2) - bp) / (1 - bp), 0, 1)
        return _rgba(_unmix(f, np.zeros_like(f), a), a)
    # фон — цвет по краям кадра (обычно почти чёрный, но бывает тёмно-серый)
    b = max(2, min(H, W) // 100)
    border = np.concatenate([f[:b].reshape(-1, 3), f[-b:].reshape(-1, 3), f[:, :b].reshape(-1, 3), f[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    Bg = np.broadcast_to(bg, f.shape)
    if matte is not None:  # нейросеть нашла объект целиком — даже тёмные его части
        a = _mthr(matte, sens)
        core = _clean(a > 0.5, fill=False)
        near = _blur(core.astype(np.float32), 3) > 0.01
        a = np.where(core, a, a * near)
        core, a = _with_holes(core, a, fill)
        return _rgba(_unmix(f, Bg, a), a)
    d = np.abs(f - bg).max(axis=2)
    db = np.abs(border - bg).max(axis=1)
    n = float(np.percentile(db, 99)) + 0.01  # уровень шума фона
    t = max(0.05, n * 1.5 + 0.02) * _k(sens)
    t = max(t, n + 0.01)
    core = _clean(_blur(d, 0.8) > t, fill=False)
    soft = np.clip((_blur(d, 0.6) - n) / (t - n), 0, 1)  # сглаживание края и слабые ореолы вокруг объекта
    halo = soft * (_blur(core.astype(np.float32), 3) > 0.01)  # ореол — только рядом с объектом
    if ndi is not None:  # край объекта: прозрачность по яркости относительно соседей — без тёмной каймы
        inner = ndi.binary_erosion(core, iterations=2)
        ring = np.clip(d / np.maximum(ndi.maximum_filter(d, 7), 1e-3), 0, 1)
        a = np.where(inner, 1.0, np.where(core, ring, halo))
    else:
        a = np.where(core, 1.0, halo)
    core, a = _with_holes(core, a, fill)
    return _rgba(_unmix(f, Bg, a), a)


# ------------------------------------------------------------------ правка кадра → верхний слой
def diff_layer(orig, edited, matte=None, sens=0.5, fill=True):
    """Возвращает (RGBA того же размера, что edited, доля изменённой площади).
    matte — маска нейросети вырезки для edited: объекты, которые заметно изменились, берутся целиком
    (иначе от тёмных предметов на тёмном фоне остаются одни блики и контуры)."""
    if orig.size != edited.size:
        orig = orig.resize(edited.size, Image.LANCZOS)
    O, Ed = _f(orig), _f(edited)
    # нейросеть слегка сдвигает цвета всего кадра — убираем общий сдвиг перед сравнением
    shift = np.median(Ed - O, axis=(0, 1))
    Ob, Eb = _blur(O, 1.2), _blur(Ed - shift, 1.2)
    d = np.abs(Eb - Ob).max(axis=2)
    noise = float(np.median(d))
    t0 = (noise * 2.5 + 0.03) * _k(sens)
    t1 = t0 + 0.10
    a = np.clip((d - t0) / (t1 - t0), 0, 1)
    a = a * a * (3 - 2 * a)
    core = a > 0.5
    if ndi is not None:
        core = ndi.binary_closing(core, iterations=3)
    core = _clean(core, min_frac=0.001, fill=False)
    if not core.any():
        return None, 0.0
    near = _blur(core.astype(np.float32), 3) > 0.01
    a = np.where(core, 1.0, a * near)
    a = np.maximum(a, _blur(core.astype(np.float32), 0.7))
    if matte is not None and ndi is not None:
        M = matte > 0.5
        lab, n = ndi.label(M)
        if n:
            area = ndi.sum(M, lab, range(1, n + 1))
            new = ndi.sum(core, lab, range(1, n + 1))
            keep = np.zeros(n + 1, bool)
            keep[1:] = new / np.maximum(area, 1) > 0.12 * _k(sens)  # объект заметно изменился — он новый
            K = keep[lab]
            Kn = ndi.binary_dilation(K, iterations=2)
            am = _mthr(matte, sens) * Kn
            # куски разницы вне найденных объектов — только крупные (мелкие — перерисовка фона)
            dl, dn = ndi.label(core)
            if dn:
                zone = ndi.binary_dilation(K, iterations=6)
                dsz = ndi.sum(core, dl, range(1, dn + 1))
                din = ndi.sum(core & zone, dl, range(1, dn + 1))
                dkeep = np.zeros(dn + 1, bool)
                dkeep[1:] = (din / np.maximum(dsz, 1) > 0.5) | (dsz > 0.01 * core.size)
                dk = dkeep[dl]
                a = np.where(_blur(dk.astype(np.float32), 2) > 0.01, a, 0)
                core = dk
            a = np.maximum(a, am)
            core = core | (am > 0.5)
    core, a = _with_holes(a > 0.5, a, fill)
    if not core.any():
        return None, 0.0
    F = _unmix(Ed, O + shift, a)
    return _rgba(F, a), float(core.mean())
