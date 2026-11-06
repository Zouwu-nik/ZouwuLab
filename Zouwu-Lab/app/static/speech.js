"use strict";
/* ================= Вкладка «Речь»: генератор голоса Higgs TTS 3 (Boson AI) =================
   текст (+ теги эмоций, подачи, пауз и звуков) → речь; голос — случайный (seed) или из библиотеки;
   новый голос — из куска записи (клонирование) или из удачного случайного результата; диалог «[Имя] реплика». */
const SP = { st: null, voices: [], src: null, job: null, items: [], newWav: null, loaded: false };
const SPKEY = "zl_speech";
const SP_TAGS = [
  { k: "emotion", t: "😊 Эмоция", sent: true, items: [
    ["elation", "восторг"], ["amusement", "веселье"], ["enthusiasm", "воодушевление"], ["determination", "решимость"], ["pride", "гордость"],
    ["contentment", "довольство"], ["affection", "нежность"], ["relief", "облегчение"], ["contemplation", "задумчивость"], ["confusion", "растерянность"],
    ["surprise", "удивление"], ["awe", "восхищение"], ["longing", "тоска"], ["arousal", "страсть"], ["anger", "гнев"], ["fear", "страх"],
    ["disgust", "отвращение"], ["bitterness", "горечь"], ["sadness", "грусть"], ["shame", "стыд"], ["helplessness", "беспомощность"]] },
  { k: "style", t: "🎭 Подача", sent: true, items: [["whispering", "шёпот"], ["shouting", "крик"], ["singing", "пение"]] },
  { k: "prosody", t: "⏱ Темп и тон", sent: true, items: [
    ["speed_very_slow", "очень медленно"], ["speed_slow", "медленнее"], ["speed_fast", "быстрее"], ["speed_very_fast", "очень быстро"],
    ["pitch_low", "ниже тон"], ["pitch_high", "выше тон"], ["expressive_high", "ярче, выразительнее"], ["expressive_low", "ровнее, спокойнее"]] },
  { k: "pause", t: "⏸ Пауза", items: [["pause", "пауза (~0.5 с)"], ["long_pause", "длинная пауза (~1 с)"]] },
  { k: "sfx", t: "🔊 Звуки", items: [
    ["laughter", "смех", "Ха-ха!"], ["sigh", "вздох", "Ах…"], ["cough", "кашель", "Кхм."], ["crying", "сквозь слёзы (перед словами)"],
    ["screaming", "с криком (перед словами)"], ["humming", "напев", "Хм-м."], ["sniff", "шмыганье", "Шмыг."], ["sneeze", "чих", "Апчхи!"], ["burping", "отрыжка", "Ык."]] },
];
const spFmt = s => { s = +s || 0; return s >= 60 ? `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}` : `${s.toFixed(1)} с`; };
const spOpt = () => ({ temperature: +$("#spTemp").value, gap: +$("#spGap").value, seed: +$("#spSeed").value, top_p: +$("#spTopP").value,
  top_k: +$("#spTopK").value, para_gap: +$("#spPGap").value, gain: +$("#spGain").value });

function spSave() {
  if (!SP.loaded) return;
  try { localStorage.setItem(SPKEY, JSON.stringify({ text: $("#spText").value, voice: $("#spVoice").value, opts: spOpt() })); } catch { }
}
function spLoad() {
  let o = null; try { o = JSON.parse(localStorage.getItem(SPKEY) || "null"); } catch { }
  if (o) {
    if (o.text) $("#spText").value = o.text;
    SP._voice = o.voice || "";
    const p = o.opts || {};
    for (const [id, k] of [["#spTemp", "temperature"], ["#spGap", "gap"], ["#spSeed", "seed"], ["#spTopP", "top_p"], ["#spTopK", "top_k"], ["#spPGap", "para_gap"], ["#spGain", "gain"]])
      if (p[k] !== undefined && p[k] !== null && !Number.isNaN(+p[k])) $(id).value = p[k];
  }
  SP.loaded = true; spSliders(); spCount();
}
function spSliders() { $("#spTempV").textContent = (+$("#spTemp").value).toFixed(2).replace(/0$/, ""); $("#spGapV").textContent = (+$("#spGap").value).toFixed(2); }
["#spTemp", "#spGap"].forEach(s => $(s).addEventListener("input", () => { spSliders(); spSave(); }));
["#spSeed", "#spTopP", "#spTopK", "#spPGap", "#spGain"].forEach(s => $(s).addEventListener("change", spSave));
$("#spSeedRnd").onclick = () => { $("#spSeed").value = -1; spSave(); };
function spErr(m) { $("#spErr").textContent = m || ""; $("#spErr").classList.toggle("hidden", !m); }

/* ---------- установка ---------- */
let spPoll = null;
async function spRefresh() {
  let st; try { st = await api("/api/speech/state"); } catch { return; }
  SP.st = st; SP.voices = st.voices || []; spInstall(st); spVoices();
  $("#spNewTextHint").textContent = st.whisper ? "(пусто — распознается сам)" : "(необязательно, но с ним голос похожее)";
  clearTimeout(spPoll);
  if (st.running) spPoll = setTimeout(spRefresh, 1000);
}
function spInstall(st) {
  const box = $("#spInstall");
  if (st.installed && !st.running) {
    box.innerHTML = `<div class="muted tiny">✓ Установлен. Русский и ещё 100+ языков, 21 эмоция, шёпот/крик/пение, смех и вздохи, клонирование голоса по 5–15 с записи. Видеопамять: ~10–11 ГБ.${st.worker ? ` <a href="#" id="spUnload">Выгрузить из видеопамяти</a>` : ""}</div>`;
    const u = $("#spUnload"); if (u) u.onclick = async e => { e.preventDefault(); await api("/api/speech/unload", {}); spRefresh(); };
    $("#spGo").disabled = !!SP.job;
    return;
  }
  $("#spGo").disabled = true;
  const pct = st.total ? Math.round(100 * st.done / st.total) : 0;
  box.innerHTML = st.running
    ? `<div class="tiny">${esc(st.step)}${st.total ? ` — ${fmtBytes(st.done)} из ${fmtBytes(st.total)}` : ""}</div><div class="vInstBar"><i style="width:${pct}%"></i></div><div class="vLog">${esc((st.log || []).join("\n"))}</div>`
    : `<p class="tiny">Нужно один раз скачать модель — около <b>${st.size_gb || 9.3} ГБ</b> (в models\\voice\\higgs-tts-3). Качается только по кнопке. Работает на видеокарте с 12+ ГБ.</p>
       <p class="tiny muted">Лицензия Boson AI — некоммерческая: личные проекты и ролики авторов — бесплатно с упоминанием «Higgs Audio (Boson AI)»; продажа и встраивание в сервисы — только по отдельной лицензии.</p>
       ${st.error ? `<div class="err">Ошибка: ${esc(st.error)}</div><div class="vLog">${esc((st.log || []).join("\n"))}</div>` : ""}
       <button class="primary" id="spInstallBtn">⬇ Установить Higgs TTS 3</button>`;
  const b = $("#spInstallBtn");
  if (b) b.onclick = async () => { b.disabled = true; try { await api("/api/speech/install", {}); } catch (e) { toast(e.message, 5000); } spRefresh(); };
}

/* ---------- библиотека голосов ---------- */
function spVoices(select) {
  const sel = $("#spVoice"), cur = select ?? (sel.value || SP._voice || "");
  sel.innerHTML = `<option value="">🎲 Случайный голос (по seed)</option>` + SP.voices.map(v => `<option value="${esc(v.id)}">🎙 ${esc(v.name)}</option>`).join("");
  sel.value = SP.voices.some(v => v.id === cur) ? cur : "";
  spVoiceBox();
}
function spVoiceBox() {
  const v = SP.voices.find(x => x.id === $("#spVoice").value);
  $("#spVoiceBox").classList.toggle("hidden", !v); $("#spRandHint").classList.toggle("hidden", !!v);
  if (!v) return;
  if ($("#spVoicePlay").dataset.id !== v.id) { $("#spVoicePlay").src = v.url; $("#spVoicePlay").dataset.id = v.id; }
  $("#spVName").value = v.name || ""; $("#spVText").value = v.text || "";
}
$("#spVoice").addEventListener("change", () => { spVoiceBox(); spSave(); });
$("#spVSave").onclick = async () => {
  const id = $("#spVoice").value; if (!id) return;
  try { const r = await api("/api/speech/voice_update", { id, name: $("#spVName").value, text: $("#spVText").value }); SP.voices = r.voices; spVoices(id); toast("Голос сохранён"); }
  catch (e) { toast(e.message, 4000); }
};
$("#spVDel").onclick = async () => {
  const id = $("#spVoice").value, v = SP.voices.find(x => x.id === id); if (!v) return;
  if (!confirm(`Удалить голос «${v.name}» из библиотеки?`)) return;
  try { const r = await api("/api/speech/voice_del", { id }); SP.voices = r.voices; spVoices(""); spSave(); } catch (e) { toast(e.message, 4000); }
};

/* ---------- новый голос из записи ---------- */
async function spUpload(files) {
  const f = files.find(x => /^(video|audio)\//.test(x.type) || /\.(mp4|mov|mkv|webm|avi|m4v|wav|mp3|m4a|flac|ogg|aac|opus)$/i.test(x.name));
  if (!f) { toast("Нужен файл звука или видео"); return; }
  const fd = new FormData(); fd.append("file", f, f.name);
  toast("Загружаю…", 1500);
  const r = await fetch("/api/speech/upload", { method: "POST", body: fd }).then(r => r.json()).catch(() => ({ error: "Не удалось загрузить файл" }));
  if (r.error) { toast(r.error, 4000); return; }
  spSetSrc(r.files[0]);
}
function spSetSrc(f) {
  SP.src = f || null;
  $("#spSrcBox").classList.toggle("hidden", !f); $("#spDrop").classList.toggle("hidden", !!f);
  if (!f) { $("#spSrcPlayer").removeAttribute("src"); return; }
  $("#spSrcName").textContent = f.title || f.name;
  $("#spSrcPlayer").src = f.url;
  $("#spStart").value = 0; $("#spEnd").value = Math.min(15, +f.duration || 15).toFixed(1);
  if (!$("#spNewName").value) $("#spNewName").value = (f.title || "").replace(/\.[^.]+$/, "").slice(0, 30);
  $("#spNewText").value = ""; $("#spNewSt").textContent = ""; spLen();
}
function spLen() {
  const a = +$("#spStart").value || 0, b = +$("#spEnd").value || 0, d = b - a;
  $("#spLen").textContent = d > 0 ? `Отрезок: ${d.toFixed(1)} с${d < 4 ? " — коротковато, лучше 5–15 с" : d > 20 ? " — длинновато, лучше до 15 с" : ""}` : "Укажите начало и конец отрезка";
}
(function spDrop() {
  const el = $("#spDrop"), input = $("#spFile");
  el.addEventListener("click", () => input.click());
  input.addEventListener("change", () => { const f = [...input.files]; input.value = ""; spUpload(f); });
  el.addEventListener("dragover", e => { e.preventDefault(); el.classList.add("over"); });
  el.addEventListener("dragleave", () => el.classList.remove("over"));
  el.addEventListener("drop", e => { e.preventDefault(); el.classList.remove("over"); spUpload([...e.dataTransfer.files]); });
})();
$("#spSrcX").onclick = () => spSetSrc(null);
$("#spSetStart").onclick = () => { $("#spStart").value = $("#spSrcPlayer").currentTime.toFixed(1); spLen(); };
$("#spSetEnd").onclick = () => { $("#spEnd").value = $("#spSrcPlayer").currentTime.toFixed(1); spLen(); };
["#spStart", "#spEnd"].forEach(s => $(s).addEventListener("input", spLen));
async function spTrack(id, onStep) {
  for (; ;) {
    await new Promise(r => setTimeout(r, 600));
    let q; try { q = await api("/api/speech/req/" + id); } catch (e) { if (/не найдено/.test(e.message)) throw e; continue; }
    if (q.status === "done") return q;
    if (q.status === "error") throw new Error(q.msg || "Ошибка");
    onStep && onStep(q);
  }
}
$("#spNewGo").onclick = async () => {
  if (!SP.src) return;
  const b = $("#spNewGo"), st = $("#spNewSt"); b.disabled = true; st.textContent = "Готовлю образец…";
  try {
    const r = await api("/api/speech/voice_add", { name: SP.src.name, start: +$("#spStart").value, end: +$("#spEnd").value,
      label: $("#spNewName").value, text: $("#spNewText").value, transcribe: true });
    const q = await spTrack(r.id, q => st.textContent = q.msg || "…");
    SP.voices = q.voices || SP.voices;
    spVoices(r.voice); spSave();
    const v = SP.voices.find(x => x.id === r.voice);
    st.innerHTML = `<span class="okc">✓ Голос «${esc(v ? v.name : "")}» добавлен и выбран.</span>${v && !v.text ? " Впишите текст образца в поле выше — так голос будет похожее." : " Проверьте текст образца выше."}`;
    spSetSrc(null); $("#spNewName").value = ""; $("#spNew").open = false;
  } catch (e) { st.innerHTML = `<span class="bad">${esc(e.message)}</span>`; }
  b.disabled = false;
};

/* ---------- теги ---------- */
(function spTagBar() {
  const bar = $("#spTags");
  bar.innerHTML = SP_TAGS.map((g, gi) => g.items.length <= 2 && g.k === "pause"
    ? g.items.map((it, ii) => `<button class="small ghost" data-g="${gi}" data-i="${ii}" title="Вставить в место курсора">${ii ? "⏸⏸" : "⏸"} ${esc(it[1])}</button>`).join("")
    : `<details class="spTag"><summary>${g.t}</summary><div class="m">${g.items.map((it, ii) => `<button class="small" data-g="${gi}" data-i="${ii}" title="&lt;|${g.k}:${it[0]}|&gt;">${esc(it[1])}</button>`).join("")}</div></details>`).join("")
    + `<span class="spAuto"><button class="small" id="spAuto" title="Встроенный помощник прочитает текст как режиссёр озвучки и сам расставит эмоции, паузы, вздохи и смех. Слова не меняются">✨ Расставить теги</button><select id="spAutoLvl" title="Сколько тегов ставить"><option value="light">сдержанно</option><option value="medium" selected>в меру</option><option value="strong">ярко</option></select><button class="small ghost hidden" id="spUndo" title="Вернуть текст как был до разметки">↶ Отменить</button></span>`
    + `<button class="small ghost" id="spReset" title="Вернуть обычную подачу: эмоции, шёпот, темп действуют до следующего тега — этот тег их сбрасывает">↺ Нейтрально</button>`
    + `<button class="small ghost" id="spDlg" title="Вставить реплику диалога: [Имя] текст">👥 Реплика</button><button class="small ghost" id="spUntag" title="Убрать из текста все теги">⌫ Без тегов</button>`;
  bar.addEventListener("click", e => {
    const b = e.target.closest("button[data-g]"); if (!b) return;
    const g = SP_TAGS[+b.dataset.g], it = g.items[+b.dataset.i];
    spInsert(`<|${g.k === "pause" ? "prosody" : g.k}:${it[0]}|>${it[2] || ""}`, !!g.sent);
    b.closest("details") && b.closest("details").removeAttribute("open");
  });
  document.addEventListener("click", e => { $$("#spTags details[open]").forEach(d => { if (!d.contains(e.target)) d.removeAttribute("open"); }); });
  $("#spReset").onclick = () => spInsert("<|reset|>", true);
  $("#spDlg").onclick = () => {
    const names = SP.voices.map(v => v.name);
    const ta = $("#spText"), pos = ta.selectionStart, before = ta.value.slice(0, pos);
    const line = (before && !before.endsWith("\n") ? "\n" : "") + `[${names[0] || "Голос 1"}] `;
    ta.setRangeText(line, pos, ta.selectionEnd, "end"); ta.focus(); spCount(); spSave();
  };
  $("#spAuto").onclick = async () => {
    const ta = $("#spText"), text = ta.value;
    if (!text.replace(/<\|(?:[a-z_]+:[a-z_]+|reset)\|>/g, "").trim()) { spErr("Впишите текст"); return; }
    const b = $("#spAuto"); b.disabled = true; spErr(""); b.textContent = "✨ Помощник читает…";
    try {
      const r = await api("/api/speech/autotag", { text, level: $("#spAutoLvl").value });
      SP.undo = text; ta.value = r.text; $("#spUndo").classList.remove("hidden");
      spCount(); spSave(); toast(`Помощник расставил тегов: ${r.tags}. Проверьте и поправьте, если нужно`, 3500);
      spWarm();   // помощник выгрузил модель речи — грузим её обратно, пока вы проверяете теги
    } catch (e) { spErr(e.message); }
    b.disabled = false; b.textContent = "✨ Расставить теги";
  };
  $("#spUndo").onclick = () => { if (SP.undo == null) return; $("#spText").value = SP.undo; SP.undo = null; $("#spUndo").classList.add("hidden"); spCount(); spSave(); };
  $("#spUntag").onclick = () => { $("#spText").value = $("#spText").value.replace(/<\|(?:[a-z_]+:[a-z_]+|reset)\|>/g, ""); spCount(); spSave(); };
})();
function spInsert(tag, sentence) {
  const ta = $("#spText"); let pos = ta.selectionStart;
  if (sentence) {   // теги эмоций/подачи/темпа — в начало текущего предложения
    const before = ta.value.slice(0, pos);
    const m = before.match(/(?:^|[.!?…]\s+|\n\s*|\]\s*)(?:<\|(?:[a-z_]+:[a-z_]+|reset)\|>)*[^.!?…\n\]]*$/);
    if (m) { pos = m.index + m[0].length - m[0].replace(/^(?:[.!?…]\s+|\n\s*|\]\s*)/, "").length; }
    ta.setRangeText(tag, pos, pos, "end");
  } else ta.setRangeText(tag, pos, ta.selectionEnd, "end");
  ta.focus(); spCount(); spSave();
}
function spCount() {
  const t = $("#spText").value, clean = t.replace(/<\|(?:[a-z_]+:[a-z_]+|reset)\|>/g, "").trim();
  const sec = clean.length / 14;
  $("#spCount").textContent = clean ? `${clean.length} симв. · примерно ${spFmt(sec)} речи` : "";
}
$("#spText").addEventListener("input", () => { spCount(); spSave(); });
$("#spClear").onclick = () => { $("#spText").value = ""; spCount(); spSave(); $("#spText").focus(); };

/* ---------- озвучка ---------- */
async function spGo() {
  if (SP.job || !(SP.st && SP.st.installed)) return;
  const text = $("#spText").value;
  if (!text.replace(/<\|(?:[a-z_]+:[a-z_]+|reset)\|>/g, "").trim()) { spErr("Впишите текст"); return; }
  spErr(""); SP.job = "pending"; $("#spGo").disabled = true;
  $("#spProg").classList.remove("hidden"); $("#spBar").style.width = "0"; $("#spTxt").textContent = "Отправка…";
  const t0 = Date.now();
  try {
    const r = await api("/api/speech/say", { text, voice: $("#spVoice").value, opts: spOpt() });
    SP.job = r.id;
    const q = await spTrack(r.id, q => { $("#spBar").style.width = Math.round(100 * (q.p || 0)) + "%"; $("#spTxt").textContent = `${q.msg || ""} · ${((Date.now() - t0) / 1000).toFixed(0)} с`; });
    $("#spBar").style.width = "100%";
    SP.newWav = q.result && q.result.wav;
    await spHistory();
    toast(`Готово: ${spFmt(q.result.duration)} речи за ${((Date.now() - t0) / 1000).toFixed(0)} с`);
  } catch (e) { spErr(e.message); }
  SP.job = null; $("#spGo").disabled = false; spRefresh();
  setTimeout(() => $("#spProg").classList.add("hidden"), 800);
}
$("#spGo").onclick = spGo;
$("#spStop").onclick = async () => { try { await api("/api/speech/cancel", {}); } catch { } };
document.addEventListener("keydown", e => { if (document.body.dataset.view === "speech" && e.ctrlKey && e.key === "Enter") { e.preventDefault(); spGo(); } });

/* ---------- результаты ---------- */
async function spHistory() {
  try { SP.items = (await api("/api/speech/list")).items || []; } catch { SP.items = []; }
  spRender();
}
function spRender() {
  const list = $("#spList");
  if (!SP.items.length) { list.innerHTML = `<div class="muEmpty"><div class="big">🗣</div>Здесь появятся озвучки. Впишите текст, добавьте эмоции кнопками сверху и нажмите «Озвучить».</div>`; return; }
  list.innerHTML = SP.items.map((m, i) => {
    const plain = (m.text || "").replace(/<\|(?:[a-z_]+:[a-z_]+|reset)\|>/g, "").replace(/\s+/g, " ").trim();
    const who = m.speakers && m.speakers.length ? "диалог: " + m.speakers.join(", ") : (m.voice_name || "случайный");
    return `<div class="muItem ${m.wav === SP.newWav ? "new" : ""}">
      <div class="h"><b>${esc(plain.slice(0, 70))}${plain.length > 70 ? "…" : ""}</b><span class="m">${esc(who)} · ${spFmt(m.duration)} · ${esc(m.time || "")}${m.seconds ? ` · за ${Math.round(m.seconds)} с` : ""} · seed ${m.seed}</span></div>
      <audio controls preload="none" src="${m.url}"></audio>
      <div class="acts"><a class="ghost small btnlink" href="${m.url}" download>⬇ WAV</a>
        ${m.has_ref ? `<button class="ghost small" data-keep="${i}" title="Сохранить этот случайный голос в библиотеку — дальше им можно озвучивать любой текст">💾 Сохранить голос</button>` : ""}
        <button class="ghost small" data-txt="${i}">Текст</button>
        <button class="ghost small" data-again="${i}" title="Вернуть текст, голос и настройки">↺ Повторить</button>
        <button class="ghost small" data-seed="${i}" title="Те же текст, голос, настройки и тот же seed">+ seed</button>
        <button class="ghost small" data-del="${i}" title="Удалить файл">🗑</button></div>
      <div class="lyr">${esc(m.text || "")}</div></div>`;
  }).join("");
  $$("#spList [data-txt]").forEach(b => b.onclick = () => b.closest(".muItem").classList.toggle("open"));
  const restore = (i, withSeed) => {
    const m = SP.items[i], o = m.opts || {};
    $("#spText").value = m.text || "";
    spVoices(SP.voices.some(v => v.id === m.voice) ? m.voice : "");
    for (const [id, k] of [["#spTemp", "temperature"], ["#spGap", "gap"], ["#spTopP", "top_p"], ["#spTopK", "top_k"], ["#spPGap", "para_gap"], ["#spGain", "gain"]]) if (o[k] !== undefined) $(id).value = o[k];
    $("#spSeed").value = withSeed ? m.seed : -1;
    spSliders(); spCount(); spSave(); toast(withSeed ? "Текст, настройки и seed восстановлены" : "Текст и настройки восстановлены");
  };
  $$("#spList [data-again]").forEach(b => b.onclick = () => restore(+b.dataset.again, false));
  $$("#spList [data-seed]").forEach(b => b.onclick = () => restore(+b.dataset.seed, true));
  $$("#spList [data-keep]").forEach(b => b.onclick = async () => {
    const m = SP.items[+b.dataset.keep];
    const name = prompt("Имя для этого голоса (по нему его можно звать в диалоге):", "Голос " + m.seed); if (name === null) return;
    try { const r = await api("/api/speech/voice_from_result", { file: m.wav, name }); SP.voices = r.voices; spVoices(r.id); spSave(); toast("Голос сохранён в библиотеку и выбран"); }
    catch (e) { toast(e.message, 4000); }
  });
  $$("#spList [data-del]").forEach(b => b.onclick = async () => {
    const m = SP.items[+b.dataset.del]; if (!confirm("Удалить эту озвучку?")) return;
    try { await api("/api/speech/del_result", { file: m.wav }); await spHistory(); } catch (e) { toast(e.message); }
  });
}

/* ---------- вход во вкладку ---------- */
function spWarm() { if (SP.st && SP.st.installed && !SP.job) api("/api/speech/warm", {}).catch(() => { }); }
window.spShow = async () => { await spRefresh(); spWarm(); if (!SP.items.length) await spHistory(); };
$("#spTab").onclick = () => setView("speech");
spLoad();
