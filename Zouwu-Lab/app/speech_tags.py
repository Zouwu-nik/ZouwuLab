"""Авторазметка текста тегами Higgs TTS 3 встроенным помощником (LLM) — как «Enhance» у ElevenLabs v3.

Помощник читает текст как режиссёр озвучки и вставляет теги эмоций, подачи, темпа, пауз и звуков.
Слова пользователя он менять не может: ответ сверяется с исходником по словам (difflib) — всё, что
помощник переписал или выбросил, возвращается как было; из добавленного остаются только теги и
междометия сразу после тега звука (<|sfx:laughter|>Ха-ха!). Неизвестные теги отбрасываются.
"""
import difflib
import re

EMOTIONS = ["elation", "amusement", "enthusiasm", "determination", "pride", "contentment", "affection", "relief",
            "contemplation", "confusion", "surprise", "awe", "longing", "arousal", "anger", "fear", "disgust",
            "bitterness", "sadness", "shame", "helplessness"]
STYLES = ["singing", "shouting", "whispering"]
PROSODY_SENT = ["speed_very_slow", "speed_slow", "speed_fast", "speed_very_fast", "pitch_low", "pitch_high",
                "expressive_high", "expressive_low"]
PROSODY_INLINE = ["pause", "long_pause"]
SFX = ["cough", "laughter", "crying", "screaming", "burping", "humming", "sigh", "sniff", "sneeze"]
ALLOWED = ({f"emotion:{x}" for x in EMOTIONS} | {f"style:{x}" for x in STYLES} | {f"prosody:{x}" for x in PROSODY_SENT + PROSODY_INLINE}
           | {f"sfx:{x}" for x in SFX} | {"reset"})
# междометия после тегов звуков (как в официальных примерах Boson: Haha, Ahem, Ahh, Hmm, Sff, Achoo, Burp);
# у crying и screaming междометия нет — тег ставится перед словами, которые произносятся сквозь плач / криком
SOUND_RU = {"laughter": "Ха-ха!", "sigh": "Ах…", "cough": "Кхм.", "humming": "Хм-м.", "sniff": "Шмыг.", "sneeze": "Апчхи!",
            "burping": "Ык."}
SOUND_EN = {"laughter": "Haha!", "sigh": "Ahh…", "cough": "Ahem.", "humming": "Hmm.", "sniff": "Sff.", "sneeze": "Achoo!",
            "burping": "Burp."}
TAG_RE = re.compile(r"<\|\s*(?:([a-z_]+)\s*:\s*([a-z_]+)|(reset))\s*\|>")
TOK_RE = re.compile(r"<\|\s*(?:[a-z_]+\s*:\s*[a-z_]+|reset)\s*\|>|\S+")
SPK_RE = re.compile(r"^\s*(\[[^\]\n]{1,40}\]\s*:?|[^:\n<>\[\]|]{1,40}?:)\s*")

LEVELS = {
    "light": "Restrained: tag only clear changes of feeling — roughly one delivery group per 3–5 sentences, a pause or a sound "
             "only where the text obviously asks for it. Plain narration stays untagged.",
    "medium": "Moderate: a delivery group wherever the feeling or manner changes (often every 1–3 sentences in emotional "
              "passages, rarely in calm narration), pauses at dramatic beats, sounds where the situation implies them.",
    "strong": "Theatrical, like an audiobook actor: give most emotional sentences a delivery group, use pauses for timing, add "
              "laughs, sighs and breaths wherever a real actor would — but never more than 3 tags in one group and never two "
              "pauses in a row.",
}


def system_prompt(level):
    return f"""You are a voice director preparing a script for the Higgs TTS 3 speech model (Boson AI). You insert its inline \
control tags so the text is performed like a real actor would. ONLY the tags listed below exist; anything else is read aloud \
or ruins the audio.

HOW THE MODEL READS TAGS (important)
A. DELIVERY tags set HOW the following text is spoken and STAY ACTIVE until the next delivery group or <|reset|>:
   <|emotion:X|>  X = {", ".join(EMOTIONS)}
   <|style:X|>    whispering (secret, intimacy, fear of being heard), shouting (calling across a room, rage, panic), singing (only for actual song lines)
   <|prosody:X|>  speed_very_slow (~0.65x), speed_slow (~0.85x), speed_fast (~1.2x), speed_very_fast (~1.4x),
                  pitch_low (~-3 semitones), pitch_high (~+2.5 semitones), expressive_high (vivid, animated), expressive_low (flat, deadpan, tired)
   - Put a delivery group at the START of a sentence (or right after the speaker name in a dialogue line).
   - Several delivery tags written together form ONE group (max 3). A new group REPLACES the previous one entirely:
     if you want to keep the anger but slow down, write both again: <|emotion:anger|><|prosody:speed_slow|>.
   - Because tags persist, do NOT repeat the same group on every sentence. Write a group only where the delivery changes.
   - When an emotional or whispered passage ends and plain narration returns, put <|reset|> at the start of that sentence.
     Never leave whispering/shouting/singing active longer than the text needs.
B. POSITIONAL tags act exactly where they stand:
   <|prosody:pause|> (~0.4–0.7 s) — dramatic beat, hesitation, before a reveal or punchline. Put it after the sentence-ending
     punctuation, before the next sentence, or at a real hesitation inside a sentence. Not instead of commas, not two in a row,
     never at the very start or end of a line.
   <|prosody:long_pause|> (~0.7–1.5 s) — scene change, heavy silence, suspense. Rare.
   <|sfx:X|> — a sound made in the speaker's own voice. Put it right before the place it happens and write the sound glued
     after it in the language of the text: laughter → <|sfx:laughter|>Ха-ха!  cough → <|sfx:cough|>Кхм.  sigh → <|sfx:sigh|>Ах…
     humming → <|sfx:humming|>Хм-м.  sniff → <|sfx:sniff|>Шмыг.  sneeze → <|sfx:sneeze|>Апчхи!  burping → <|sfx:burping|>Ык.
     (English text: Haha! / Ahem. / Ahh… / Hmm. / Sff. / Achoo! / Burp.)
     crying and screaming have NO written sound: put <|sfx:crying|> before the words spoken while crying (usually together with
     <|emotion:sadness|>), <|sfx:screaming|> before a panicked exclamation.

HOW TO DIRECT
1. Read the whole passage first: who speaks, to whom, what happened, what they want. Pick emotions from the situation and
   subtext, not from single words ("Отлично." can be bitterness). Narration (author's text) is usually neutral or calm —
   leave it untagged or use <|reset|> after a character's emotional line.
2. Follow the emotional arc: mark the turning points (calm → worry → fear → relief) instead of tagging everything.
3. Speed and pitch are seasoning: panic/hurry → speed_fast; grief, solemn or menacing lines → speed_slow; do not combine opposite tags.
4. Sounds only when the situation implies them (a joke → laughter, fatigue or relief → sigh, grief → crying). At most one sound per sentence.
5. {LEVELS.get(level, LEVELS["medium"])}
6. Avoid <|emotion:arousal|> (sexual desire) unless the text is explicitly an intimate scene between adults.

WHAT YOU MAY CHANGE IN THE TEXT
- NOTHING except: inserting tags, the sound after an sfx tag, and writing digits as words in the text's language with correct
  grammar ("в 2026 году" → "в две тысячи двадцать шестом году", "5 минут" → "пять минут"). Keep every other word, letter,
  punctuation mark, line break and dialogue prefix ([Name] / Name:) exactly as given. Do not translate.

OUTPUT
The same numbered lines in the format "N| text", one per input line, nothing else — no comments, no explanations.

EXAMPLE
Input:
1| Он открыл дверь и замер. На пороге стояла она.
2| [Аня] Привет. Я вернулась.
3| [Макс] Вернулась?! Через 3 года? Ну и ладно. Подумаешь.
4| Он отвернулся к окну.
Output:
1| Он открыл дверь и замер. <|prosody:pause|>На пороге стояла она.
2| [Аня] <|emotion:longing|><|prosody:speed_slow|>Привет. <|prosody:pause|>Я вернулась.
3| [Макс] <|emotion:surprise|><|prosody:expressive_high|>Вернулась?! <|emotion:anger|>Через три года? <|prosody:pause|><|emotion:bitterness|><|prosody:expressive_low|>Ну и ладно. <|sfx:sigh|>Ах… Подумаешь.
4| <|reset|>Он отвернулся к окну."""


def clean_tag(m):
    if m.group(3):
        return "<|reset|>"
    k = f"{m.group(1)}:{m.group(2)}"
    return f"<|{k}|>" if k in ALLOWED else ""


def tokens(s):
    return TOK_RE.findall(s)


def is_tag(t):
    return t.startswith("<|")


def norm(w):
    return re.sub(r"[^\w]+", "", w.lower().replace("ё", "е"))


def merge_line(orig, out, sounds):
    """Исходные слова + теги из ответа помощника. Слова всегда берутся из исходника."""
    ow = tokens(orig)
    # ответ: слова с прицепленными тегами перед ними
    words, pre, cur = [], [], []
    for t in tokens(TAG_RE.sub(lambda m: " " + clean_tag(m) + " ", out)):
        if is_tag(t):
            cur.append(t)
        else:
            words.append(t)
            pre.append(cur)
            cur = []
    tail = cur
    sm = difflib.SequenceMatcher(a=[norm(w) for w in ow], b=[norm(w) for w in words], autojunk=False)
    res, carry = [], []

    def put(tags, word=None):
        for tg in tags:
            if not res or res[-1] != tg:   # без двух одинаковых тегов подряд
                res.append(tg)
        if word is not None:
            res.append(word)
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == "equal":
            for i, j in zip(range(a0, a1), range(b0, b1)):
                put(carry + pre[j], ow[i])
                carry = []
        elif op == "insert":
            # допустимы только междометия сразу после тега звука
            j = b0
            while j < b1:
                tags = carry + pre[j]
                carry = []
                sfx = [tg for tg in tags if tg.startswith("<|sfx:")]
                if sfx and len(words[j]) <= 20:
                    put(tags, words[j])
                    j += 1
                    while j < b1 and not pre[j] and len(words[j]) <= 12 and re.search(r"[!.…?]$", words[j - 1]) is None:
                        put([], words[j])   # «Ха- ха!» из двух кусков
                        j += 1
                else:
                    carry = tags   # слово выбрасываем, его теги — к следующему слову
                    j += 1
        elif op == "replace" and any(re.search(r"\d", ow[i]) for i in range(a0, a1)) and b1 - b0 <= 6 * (a1 - a0) + 2 \
                and not any(re.search(r"\d", words[j]) for j in range(b0, b1)):
            # числа помощник пишет словами («3 года» → «три года») — это разрешено
            for j in range(b0, b1):
                put(carry + pre[j], words[j])
                carry = []
        else:   # replace / delete — слова исходника возвращаются, теги ответа переносятся на них
            tags = carry + [tg for j in range(b0, b1) for tg in pre[j]]
            carry = []
            for k, i in enumerate(range(a0, a1)):
                put(tags if k == 0 else [], ow[i])
            if a0 == a1:
                carry = tags
    put(carry + tail)
    # тег звука без междометия — подставляем стандартное
    fixed = []
    for i, t in enumerate(res):
        fixed.append(t)
        if t.startswith("<|sfx:"):
            nxt = res[i + 1] if i + 1 < len(res) else None
            if nxt is None or is_tag(nxt):
                fixed.append(sounds.get(t[6:-2], ""))
    s = ""
    for t in fixed:   # теги прилипают к следующему слову, слова — через пробел
        if not t:
            continue
        s += t if (s.endswith("|>") or not s) and is_tag(t) else ((" " if s and not s.endswith("|>") else "") + t)
    s = re.sub(r"\|>\s+(?=\S)", "|>", s)
    # теги перед именем говорящего — после него, иначе диалог не распознается
    m = re.match(r"^((?:<\|(?:[a-z_]+:[a-z_]+|reset)\|>)+)\s*(\[[^\]\n]{1,40}\]\s*:?|[^:\n<>\[\]|]{1,40}?:)\s*", s)
    if m and SPK_RE.match(orig):
        s = m.group(2).strip() + " " + m.group(1) + s[m.end():]
    return s.strip()


def split_blocks(lines, limit=1800):
    blocks, cur, n = [], [], 0
    for i, l in enumerate(lines):
        if cur and n + len(l) > limit:
            blocks.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += len(l) + 6
    if cur:
        blocks.append(cur)
    return blocks


def parse_numbered(out):
    got = {}
    for line in out.splitlines():
        m = re.match(r"^\s*(\d+)\s*[|:.)]\s?(.*)$", line)
        if m:
            got[int(m.group(1))] = m.group(2)
    return got


async def auto_tag(text, chat, level="medium"):
    """text → размеченный текст. chat(messages, max_tokens, temperature, hold) — встроенный помощник."""
    src = TAG_RE.sub("", text)
    lines = src.split("\n")
    cyr = len(re.findall(r"[а-яё]", src.lower())) >= len(re.findall(r"[a-z]", src.lower()))
    sounds = SOUND_RU if cyr else SOUND_EN
    idx = [i for i, l in enumerate(lines) if l.strip()]
    if not idx:
        return text, 0
    out_lines = list(lines)
    blocks = split_blocks([lines[i] for i in idx])
    added = 0
    for bi, blk in enumerate(blocks):
        nums = {k + 1: idx[p] for k, p in enumerate(blk)}
        user = "\n".join(f"{k}| {lines[i].strip()}" for k, i in nums.items())
        ans = await chat([{"role": "system", "content": system_prompt(level)}, {"role": "user", "content": user}],
                         max_tokens=min(4000, 400 + len(user) * 2), temperature=0.4, hold=bi < len(blocks) - 1)
        got = parse_numbered(ans)
        for k, i in nums.items():
            if k in got:
                lead = re.match(r"^\s*", lines[i]).group(0)
                new = merge_line(lines[i].strip(), got[k], sounds)
                added += len(TAG_RE.findall(new))
                out_lines[i] = lead + new
    return "\n".join(out_lines), added
