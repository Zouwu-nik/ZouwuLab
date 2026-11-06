"use strict";
/* ================= Вкладка «Голос»: переозвучка роликов голосом эталона =================
   эталон → «Взять голос» (отрезок, очистка, текст эталона)
   ролик  → «Распознать» (фразы с таймингом) → правка текста → «Озвучить» (каждая фраза ровно в свою длину) */
const V = { view: "gen", refs: [{ label: "обычный" }], ri: 0, clips: [], inst: null, loaded: false, busy: 0 };
/* несколько эталонов = несколько эмоций (обычный, крик, шёпот…); V.ref — текущий открытый эталон с файлом */
Object.defineProperty(V, "ref", {
  get() { const R = V.refs[V.ri]; return R && R.name ? R : null; },
  set(v) { const R = V.refs[V.ri]; for (const k of Object.keys(R)) if (k !== "label") delete R[k]; if (v) Object.assign(R, v, { label: R.label }); },
});
const VKEY = "zl_voice2";
function vSave() {
  if (!V.loaded) return;
  try {
    localStorage.setItem(VKEY, JSON.stringify({
      view: V.view, refs: V.refs.map(R => ({ ...R, busy: undefined })), ri: V.ri, clips: V.clips.map(c => ({ ...c, req: undefined })),
      opts: { lang: $("#vLang").value, sep: $("#vSep").checked, keepOut: $("#vKeepOut").checked, steps: +$("#vSteps").value, guid: +$("#vGuid").value, seed: +$("#vSeed").value, refSep: $("#vRefSep").checked },
    }));
  } catch { }
}
(function vLoad() {
  let o = null; try { o = JSON.parse(localStorage.getItem(VKEY) || "null"); } catch { }
  if (!o) return;
  V.view = ["voice", "3d", "music", "speech"].includes(o.view) ? o.view : "gen"; V.refs = Array.isArray(o.refs) && o.refs.length ? o.refs : (o.ref ? [{ ...o.ref, label: "обычный" }] : V.refs); V.ri = Math.min(+o.ri || 0, V.refs.length - 1); V.clips = Array.isArray(o.clips) ? o.clips : [];
  V.clips.forEach(c => { if (c.state === "asr" || c.state === "tts") { c.state = ""; c.msg = ""; } });
  const op = o.opts || {};
  if (op.lang) $("#vLang").value = op.lang;
  if (op.sep === false) $("#vSep").checked = false;
  if (op.keepOut === false) $("#vKeepOut").checked = false;
  if (op.refSep === false) $("#vRefSep").checked = false;
  if (op.steps) $("#vSteps").value = op.steps;
  if (op.guid) $("#vGuid").value = op.guid;
  if (op.seed !== undefined) $("#vSeed").value = op.seed;
})();
const fmtS = s => (+s || 0).toFixed(2);
const isVid = n => /\.(mp4|mov|mkv|webm|avi|m4v)$/i.test(n || "");
const upUrl = n => "/input/voice/" + encodeURIComponent(n);
function vErr(m) { $("#vErr").textContent = m || ""; $("#vErr").classList.toggle("hidden", !m); }

function setView(v) {
  V.view = ["voice", "3d", "music", "speech"].includes(v) ? v : "gen";
  document.body.dataset.view = V.view;
  if (V.view === "voice") vRefreshState();
  if (V.view === "3d" && window.t3Show) window.t3Show();
  if (V.view === "music" && window.muShow) window.muShow();
  if (V.view === "speech" && window.spShow) window.spShow();
  vSave();
}
window.setView = setView;
$("#voiceTab").onclick = () => setView("voice");

/* ---------- установка ---------- */
let vPoll = null;
async function vRefreshState() {
  let st; try { st = await api("/api/voice/state"); } catch { return; }
  V.inst = st; renderInstall(st);
  clearTimeout(vPoll);
  if (st.running || (document.body.dataset.view === "voice" && !st.installed)) vPoll = setTimeout(vRefreshState, st.running ? 1000 : 5000);
}
function renderInstall(st) {
  const box = $("#vInstall");
  if (st.installed && !st.running) {
    box.innerHTML = `<div class="muted tiny">✓ Установлен: OmniVoice (озвучка голосом эталона, 600+ языков), Whisper large‑v3‑turbo (распознавание речи), MelBand‑Roformer (отделение голоса от фона).${st.worker ? ` <a href="#" id="vUnload">Выгрузить из видеопамяти</a>` : ""}</div>`;
    const u = $("#vUnload"); if (u) u.onclick = async e => { e.preventDefault(); await api("/api/voice/unload", {}); vRefreshState(); };
    return;
  }
  const pct = st.total ? Math.round(100 * st.done / st.total) : 0;
  box.innerHTML = st.running
    ? `<div class="tiny">${esc(st.step)}${st.total ? ` — ${fmtBytes(st.done)} из ${fmtBytes(st.total)}` : ""}</div><div class="vInstBar"><i style="width:${pct}%"></i></div><div class="vLog">${esc((st.log || []).join("\n"))}</div>`
    : `<p class="tiny">Нужно один раз скачать модели и библиотеки — около <b>5 ГБ</b>: OmniVoice (озвучка голосом эталона), Whisper large‑v3‑turbo (распознавание речи) и MelBand‑Roformer (отделение голоса от музыки). Качается только по кнопке; уже установленные библиотеки движка не меняются.</p>
       ${st.error ? `<div class="err">Ошибка: ${esc(st.error)}</div><div class="vLog">${esc((st.log || []).join("\n"))}</div>` : ""}
       <button class="primary" id="vInstallBtn">⬇ Установить голосовой модуль</button>`;
  const b = $("#vInstallBtn");
  if (b) b.onclick = async () => { b.disabled = true; try { await api("/api/voice/install", {}); } catch (e) { toast(e.message, 5000); } vRefreshState(); };
}

/* ---------- задания процесса озвучки ---------- */
async function track(id, onStep) {
  for (; ;) {
    await new Promise(r => setTimeout(r, 700));
    let q; try { q = await api("/api/voice/req/" + id); } catch (e) { throw e; }
    if (q.status === "done") return q.result || {};
    if (q.status === "error") throw new Error(q.msg || "Ошибка");
    onStep && onStep(q);
  }
}
function busy(d) { V.busy = Math.max(0, V.busy + d); $("#vStop").classList.toggle("hidden", !V.busy); }
$("#vStop").onclick = async () => { try { await api("/api/voice/stop", {}); toast("Остановлено"); } catch (e) { toast(e.message); } };
$("#vFolder").onclick = () => openFolder("voice");

/* ---------- загрузка файлов ---------- */
async function vUpload(files) {
  const fd = new FormData(); let n = 0;
  for (const f of files) if (/^(video|audio)\//.test(f.type) || /\.(mp4|mov|mkv|webm|avi|m4v|wav|mp3|m4a|flac|ogg|aac|opus)$/i.test(f.name)) { fd.append("file", f, f.name); n++; }
  if (!n) { toast("Нужны файлы видео или звука"); return []; }
  toast(`Загружаю ${n} файл(ов)…`, 1500);
  const r = await fetch("/api/voice/upload", { method: "POST", body: fd }).then(r => r.json());
  if (r.error) { toast(r.error, 4000); return []; }
  return r.files || [];
}
function dz(el, input, cb) {
  el.addEventListener("click", () => input.click());
  input.addEventListener("change", () => { const f = [...input.files]; input.value = ""; cb(f); });
  el.addEventListener("dragover", e => { e.preventDefault(); el.classList.add("over"); });
  el.addEventListener("dragleave", () => el.classList.remove("over"));
  el.addEventListener("drop", e => { e.preventDefault(); el.classList.remove("over"); cb([...e.dataTransfer.files]); });
}

/* ---------- эталон ---------- */
dz($("#vRefDrop"), $("#vRefFile"), async files => {
  const r = await vUpload(files.slice(0, 1)); if (!r.length) return;
  const f = r[0];
  if (!f.audio) { toast("В этом файле нет звука", 4000); return; }
  V.ref = { name: f.name, title: f.title, duration: f.duration, start: 0, end: Math.min(f.duration || 0, 10) };
  renderRef(); vSave();
});
function renderRefTabs() {
  const box = $("#vRefTabs");
  box.innerHTML = V.refs.map((R, i) => `<button class="small${i === V.ri ? " on" : ""}" data-ri="${i}" title="${R.wav ? "голос взят" : R.name ? "файл есть, голос ещё не взят" : "пусто"}">${R.wav ? "✓ " : ""}${esc(R.label || "эталон " + (i + 1))}</button>`).join("")
    + `<button class="small ghost" id="vRefAdd" title="Ещё один эталон того же персонажа с другой эмоцией (крик, шёпот, смех) — у каждой фразы можно выбрать, каким озвучивать">＋ эмоция</button>`;
  $$("[data-ri]", box).forEach(b => b.onclick = () => { V.ri = +b.dataset.ri; renderRef(); vSave(); });
  $("#vRefAdd").onclick = () => { const names = ["крик", "шёпот", "смех", "злость", "грусть"]; V.refs.push({ label: names[V.refs.length - 1] || "эмоция " + V.refs.length }); V.ri = V.refs.length - 1; renderRef(); renderClips(); vSave(); };
  $("#vRefLabel").value = V.refs[V.ri].label || "";
  $("#vRefDel").classList.toggle("hidden", V.refs.length < 2);
}
function renderRef() {
  renderRefTabs();
  const has = !!V.ref;
  $("#vRefDrop").classList.toggle("hidden", has); $("#vRefBox").classList.toggle("hidden", !has);
  if (!has) { $("#vRefPlayer").removeAttribute("src"); return; }
  $("#vRefName").textContent = V.ref.title || V.ref.name;
  const src = upUrl(V.ref.name);
  if ($("#vRefPlayer").getAttribute("src") !== src) $("#vRefPlayer").src = src;
  $("#vRefPlayer").style.maxHeight = isVid(V.ref.name) ? "220px" : "54px";
  $("#vRefStart").value = (+V.ref.start || 0).toFixed(1); $("#vRefEnd").value = (+V.ref.end || 0).toFixed(1);
  const len = Math.max(0, (+V.ref.end || V.ref.duration || 0) - (+V.ref.start || 0));
  $("#vRefLen").innerHTML = `Отрезок: <b>${len.toFixed(1)} с</b>` + (len < 3 ? " — маловато, лучше 3–10 с" : len > 20 ? " — возьмётся первые 20 с" : len > 10 ? " — длинновато: 3–10 с похожее и быстрее" : " — хорошо");
  const ready = !!V.ref.wav;
  $("#vRefTextBox").classList.toggle("hidden", !ready);
  if (ready && $("#vRefText").value !== (V.ref.text || "")) $("#vRefText").value = V.ref.text || "";
  if (!V.ref.busy) $("#vRefSt").innerHTML = ready ? `<span class="okc">✓ Голос взят (${(+V.ref.refDur || 0).toFixed(1)} с). Можно озвучивать ролики.</span>` : `<span class="muted">Выберите отрезок и нажмите «Взять голос».</span>`;
}
const refChanged = () => { if (V.ref) { V.ref.wav = null; V.ref.text = ""; } renderRef(); vSave(); };
$("#vRefX").onclick = () => { V.ref = null; renderRef(); vSave(); };
$("#vRefLabel").oninput = () => { V.refs[V.ri].label = $("#vRefLabel").value.trim() || "эталон " + (V.ri + 1); renderRefTabs(); renderClips(); vSave(); };
$("#vRefDel").onclick = () => {
  if (V.refs.length < 2) return;
  const gone = V.ri; V.refs.splice(gone, 1); V.ri = Math.max(0, gone - 1);
  V.clips.forEach(c => (c.segs || []).forEach(s => { const k = +s.ref || 0; s.ref = k === gone ? 0 : k > gone ? k - 1 : k; }));
  renderRef(); renderClips(); vSave();
};
$("#vRefStart").onchange = () => { V.ref.start = Math.max(0, +$("#vRefStart").value || 0); refChanged(); };
$("#vRefEnd").onchange = () => { V.ref.end = Math.max(0, +$("#vRefEnd").value || 0); refChanged(); };
$("#vRefSetStart").onclick = () => { V.ref.start = +$("#vRefPlayer").currentTime.toFixed(1); refChanged(); };
$("#vRefSetEnd").onclick = () => { V.ref.end = +$("#vRefPlayer").currentTime.toFixed(1); refChanged(); };
$("#vRefSep").onchange = () => { refChanged(); };
$("#vRefText").oninput = () => { if (V.ref) { V.ref.text = $("#vRefText").value; vSave(); } };
function playSpan(pl, a, b) {
  pl.currentTime = a; pl.play();
  const stop = () => { if (pl.currentTime >= b) { pl.pause(); pl.removeEventListener("timeupdate", stop); } };
  pl.addEventListener("timeupdate", stop);
}
$("#vRefPlaySel").onclick = () => playSpan($("#vRefPlayer"), +V.ref.start || 0, +V.ref.end || $("#vRefPlayer").duration);
async function takeVoice(R) {
  R = R || V.refs[V.ri];
  if (!R || !R.name) throw new Error(`Добавьте эталон «${(R && R.label) || "обычный"}» — видео или звук с нужным голосом (слева)`);
  const cur = () => R === V.refs[V.ri];
  R.busy = true; busy(1);
  if (cur()) $("#vRefSt").textContent = "Запускаю…";
  try {
    const { id } = await api("/api/voice/ref", { name: R.name, start: +R.start || 0, end: +R.end || 0, separate: $("#vRefSep").checked, lang: $("#vLang").value });
    const r = await track(id, q => { if (cur()) $("#vRefSt").textContent = q.msg || "…"; });
    Object.assign(R, { wav: r.ref_wav, text: r.ref_text, refDur: r.duration });
  } finally { R.busy = false; busy(-1); renderRef(); vSave(); }
}
$("#vRefGo").onclick = async () => { vErr(""); try { await takeVoice(); } catch (e) { $("#vRefSt").innerHTML = `<span class="bad">${esc(e.message)}</span>`; } };

/* ---------- ролики ---------- */
dz($("#vClipDrop"), $("#vClipFile"), async files => {
  const r = await vUpload(files);
  for (const f of r) {
    if (!f.audio) { toast(`${f.title}: в файле нет звука — пропущен`, 4000); continue; }
    V.clips.push({ name: f.name, title: f.title, duration: f.duration, video: f.video, segs: [] });
  }
  renderClips(); vSave();
});
function clipHtml(c, i) {
  const r = c.res || {}, st = c.state || "";
  const segs = c.segs || [];
  const media = (u, v, cls) => v ? `<video class="${cls}" src="${u}" controls preload="metadata"></video>` : `<audio class="${cls}" src="${u}" controls preload="metadata"></audio>`;
  return `<div class="vItem" data-i="${i}">
    <div class="vMedia">${media(upUrl(c.name), c.video, "orig")}<div class="muted tiny">исходник · ${(+c.duration || 0).toFixed(1)} с</div>
      ${r.wav ? `<div class="res">${media(r.video || r.wav, !!r.video, "new")}
        <div class="ab"><button class="small on" data-ab="new">новый голос</button><button class="small" data-ab="old">как было</button></div>
        <div class="links">${r.video ? `<a href="${r.video}" download>⬇ видео</a>` : ""}<a href="${r.wav}" download>⬇ звук</a><a href="${r.voice}" download title="Только новый голос, без фона — для монтажа">⬇ только голос</a></div></div>` : ""}
    </div>
    <div class="vBody">
      <div class="vh"><b>${esc(c.title || c.name)}</b><button class="icon" data-vx title="Убрать из списка">✕</button></div>
      <div class="vBtns">
        <button class="small" data-asr ${st ? "disabled" : ""}>📝 ${segs.length ? "Распознать заново" : "Распознать текст"}</button>
        <button class="small primary" data-tts ${st || !segs.some(s => !s.keep) ? "disabled" : ""}>🎙 Озвучить</button>
        <span class="st ${c.err ? "bad" : st ? "" : r.wav ? "okc" : ""}">${esc(c.msg || "")}</span>
      </div>
      ${st ? `<div class="pb"><i style="width:${Math.round(100 * (c.p || 0))}%"></i></div>` : ""}
      ${segs.length ? `<div class="segs">${segs.map((s, k) => `<div class="vseg${s.keep ? " keep" : ""}" data-k="${k}">
          <div class="tm"><input type="number" step="0.05" min="0" value="${fmtS(s.start)}" data-f="start" title="Начало фразы, с"><span>–</span><input type="number" step="0.05" min="0" value="${fmtS(s.end)}" data-f="end" title="Конец фразы, с"><b>${(s.end - s.start).toFixed(2)} с</b>${V.refs.length > 1 && !s.keep ? `<select class="rs" data-sref title="Каким эталоном (эмоцией) озвучить фразу">${V.refs.map((R, j) => `<option value="${j}"${(+s.ref || 0) === j ? " selected" : ""}>${esc(R.label)}</option>`).join("")}</select>` : ""}
          ${!s.keep ? `<label class="gn" title="Громкость фразы относительно автоматической, дБ (+6 — примерно вдвое громче)">громк.<input type="number" step="1" min="-18" max="18" value="${+s.gain || 0}" data-sgain></label>` : ""}
          <label class="kp" title="Этот кусок остаётся как в ролике (оригинальный голос, зевок, крик) — не переозвучивается"><input type="checkbox" data-keep ${s.keep ? "checked" : ""}> оригинал</label></div>
          <textarea rows="2" data-f="text" placeholder="${s.keep ? "остаётся как в ролике" : "текст фразы"}" ${s.keep ? "disabled" : ""}>${esc(s.text)}</textarea>
          <div class="sb"><button class="icon small" data-play title="Прослушать этот кусок исходника">▶</button><button class="icon small" data-pause title="Вставить паузу в место курсора: [пауза] = 0.5 с, [пауза 1] = 1 с">⏸</button><button class="icon small" data-del title="Убрать фразу из списка (на её месте останется то, что было в ролике, если включено «звуки вне фраз — из оригинала»)">✕</button></div>
        </div>`).join("")}
        <button class="small" data-add>＋ фраза</button><button class="small" data-addkeep title="Отметить кусок ролика, который должен остаться как был (зевок, крик, смех) — время поправьте полями">＋ оригинальный кусок</button>
        <span class="muted tiny">Длина фразы = время на её озвучку. Если слов стало больше — сдвиньте «конец» позже, иначе речь будет быстрой. Пауза внутри фразы — ⏸ или вписать <b>[пауза]</b> (0.5 с) / <b>[пауза 1.2]</b>; многоточие точной паузы не даёт.</span></div>`
      : `<p class="muted tiny">Нажмите «Распознать текст» — программа выпишет фразы с таймингом, вы их поправите.</p>`}
    </div></div>`;
}
function renderClips() {
  const L = $("#vList");
  if (!V.clips.length) { L.innerHTML = `<p class="muted tiny">Пока пусто. Добавьте ролики — или нажмите «🎙 → Голос» у готового видео MiniMax.</p>`; return; }
  L.innerHTML = V.clips.map(clipHtml).join("");
  $$(".vItem", L).forEach(el => bindClip(el, V.clips[+el.dataset.i]));
}
function rerender(c) {
  const i = V.clips.indexOf(c), el = $(`.vItem[data-i="${i}"]`);
  if (!el) return renderClips();
  const keep = $("video.new, audio.new", el), t = keep ? keep.currentTime : 0;
  el.outerHTML = clipHtml(c, i);
  const ne = $(`.vItem[data-i="${i}"]`); bindClip(ne, c);
}
function bindClip(el, c) {
  $("[data-vx]", el).onclick = () => { V.clips.splice(V.clips.indexOf(c), 1); renderClips(); vSave(); };
  $("[data-asr]", el).onclick = () => runAsr(c).catch(() => { });
  $("[data-tts]", el).onclick = () => runTts(c).catch(() => { });
  const orig = $(".orig", el);
  $$(".vseg", el).forEach(sg => {
    const s = c.segs[+sg.dataset.k];
    $$("[data-f]", sg).forEach(inp => inp.oninput = () => {
      const f = inp.dataset.f;
      s[f] = f === "text" ? inp.value : Math.max(0, +inp.value || 0);
      if (f !== "text") $(".tm b", sg).textContent = (s.end - s.start).toFixed(2) + " с";
      vSave();
    });
    $("[data-play]", sg).onclick = () => playSpan(orig, s.start, s.end);
    $("[data-keep]", sg).onchange = e => { s.keep = e.target.checked; rerender(c); vSave(); };
    const sr = $("[data-sref]", sg); if (sr) sr.onchange = () => { s.ref = +sr.value; vSave(); };
    const sgn = $("[data-sgain]", sg); if (sgn) sgn.onchange = () => { s.gain = Math.max(-18, Math.min(18, +sgn.value || 0)); vSave(); };
    $("[data-pause]", sg).onclick = () => {
      const ta = $("textarea", sg), v = ta.value, a = ta.selectionStart ?? v.length, b = ta.selectionEnd ?? a;
      const pre = v.slice(0, a).replace(/\s+$/, ""), post = v.slice(b).replace(/^\s+/, "");
      ta.value = (pre ? pre + " " : "") + "[пауза]" + (post ? " " + post : "");
      s.text = ta.value; ta.focus(); const pos = (pre ? pre.length + 1 : 0) + 7; ta.setSelectionRange(pos, pos); vSave();
    };
    $("[data-del]", sg).onclick = () => { c.segs.splice(c.segs.indexOf(s), 1); rerender(c); vSave(); };
  });
  const ak = $("[data-addkeep]", el);
  if (ak) ak.onclick = () => {
    const t0 = Math.max(0, +(orig.currentTime || 0).toFixed(2));
    c.segs.push({ start: t0, end: +Math.min(+c.duration || t0 + 1, t0 + 1).toFixed(2), text: "", keep: true });
    c.segs.sort((x, y) => x.start - y.start); rerender(c); vSave();
  };
  const add = $("[data-add]", el);
  if (add) add.onclick = () => {
    const last = c.segs[c.segs.length - 1], a = last ? Math.min(+c.duration || 0, last.end + 0.2) : 0;
    c.segs.push({ start: +a.toFixed(2), end: +Math.min(+c.duration || a + 2, a + 2).toFixed(2), text: "" }); rerender(c); vSave();
  };
  $$("[data-ab]", el).forEach(b => b.onclick = () => {
    const pl = $(".new", el), t = pl.currentTime, play = !pl.paused;
    pl.src = b.dataset.ab === "new" ? (c.res.video || c.res.wav) : upUrl(c.name);
    pl.currentTime = t; if (play) pl.play();
    $$("[data-ab]", el).forEach(x => x.classList.toggle("on", x === b));
  });
}
async function runAsr(c) {
  c.state = "asr"; c.msg = "Запускаю…"; c.p = 0; c.err = false; rerender(c); busy(1);
  try {
    const { id } = await api("/api/voice/asr", { name: c.name, separate: $("#vSep").checked, lang: $("#vLang").value });
    const r = await track(id, q => { c.msg = q.msg; c.p = q.p; rerender(c); });
    c.segs = (r.segments || []).map(s => ({ start: +s.start, end: +s.end, text: s.text }));
    if (r.duration) c.duration = r.duration;
    c.msg = c.segs.length ? `Найдено фраз: ${c.segs.length}. Поправьте текст и нажмите «Озвучить».` : "Речь не найдена — впишите фразы вручную (＋ фраза)";
  } catch (e) { c.msg = e.message; c.err = true; throw e; }
  finally { c.state = ""; busy(-1); rerender(c); vSave(); }
}
async function runTts(c) {
  // нужные фразам эталоны готовим сами, если «Взять голос» ещё не нажимали
  const need = new Set([0, ...(c.segs || []).filter(s => !s.keep).map(s => Math.min(+s.ref || 0, V.refs.length - 1))]);
  for (const i of need) {
    const R = V.refs[i];
    if (!R.name) { const m = `Эталон «${R.label}» пустой — добавьте для него видео слева или выберите у фраз другой эталон.`; vErr(m); throw new Error(m); }
    if (!R.wav) await takeVoice(R);
  }
  c.state = "tts"; c.msg = "Запускаю…"; c.p = 0; c.err = false; rerender(c); busy(1);
  try {
    const { id } = await api("/api/voice/tts", {
      name: c.name, segments: c.segs.map(s => ({ ...s, ref: Math.min(+s.ref || 0, V.refs.length - 1), ref_label: (V.refs[+s.ref || 0] || {}).label })),
      refs: V.refs.map(R => R.wav ? { wav: R.wav, text: R.text || "" } : null), lang: $("#vLang").value, separate: $("#vSep").checked,
      opts: { steps: +$("#vSteps").value, guidance: +$("#vGuid").value, seed: +$("#vSeed").value, keep_outside: $("#vKeepOut").checked },
    });
    const r = await track(id, q => { c.msg = q.msg; c.p = q.p; rerender(c); });
    c.res = { video: r.video_url, wav: r.wav_url, voice: r.voice_url, seed: r.seed };
    c.msg = `Готово (вариант ${r.seed}). Не понравилось — поправьте текст или нажмите 🎲 и озвучьте ещё раз.`;
  } catch (e) { c.msg = e.message; c.err = true; throw e; }
  finally { c.state = ""; busy(-1); rerender(c); vSave(); }
}
$("#vAsrAll").onclick = async () => { vErr(""); for (const c of V.clips.filter(c => !(c.segs || []).length)) { try { await runAsr(c); } catch { } } };
$("#vRunAll").onclick = async () => {
  vErr("");
  if (!V.clips.length) return vErr("Добавьте ролики для переозвучки.");
  for (const c of V.clips) {
    try { if (!(c.segs || []).length) await runAsr(c); if ((c.segs || []).length) await runTts(c); }
    catch (e) { if (/эталон/.test(e.message)) { vErr(e.message); break; } }
  }
};

/* ---------- настройки ---------- */
[["#vSteps", "#vStepsV", v => v], ["#vGuid", "#vGuidV", v => (+v).toFixed(2)]].forEach(([s, o, f]) => {
  const upd = () => { $(o).textContent = f($(s).value); vSave(); };
  $(s).addEventListener("input", upd); upd();
});
["#vSep", "#vLang", "#vSeed", "#vKeepOut"].forEach(s => $(s).addEventListener("change", vSave));
$("#vSeedRnd").onclick = () => { $("#vSeed").value = Math.floor(Math.random() * 99999); vSave(); };

/* видео из генерации → в список роликов */
$("#toVoice").onclick = async () => {
  const u = ($("#mainVideo").getAttribute("src") || "").split("?")[0];
  if (!u.startsWith("/output/")) return toast("Сначала выберите готовое видео");
  try {
    const r = await api("/api/voice/from_output", { url: u });
    for (const f of r.files) V.clips.push({ name: f.name, title: f.title, duration: f.duration, video: f.video, segs: [] });
    renderClips(); setView("voice"); toast("Видео добавлено во вкладку «Голос»");
  } catch (e) { toast(e.message, 4000); }
};

/* ---------- просто текст (без видео) ---------- */
V.say = V.say || { text: "", res: [] };
try { const s = JSON.parse(localStorage.getItem("zl_voice_say") || "null"); if (s) V.say = s; } catch { }
const saySave = () => { try { localStorage.setItem("zl_voice_say", JSON.stringify(V.say)); } catch { } };
function renderSayRefs() {
  const sel = $("#vSayRef"), cur = sel.value;
  sel.innerHTML = V.refs.map((R, i) => `<option value="${i}">${esc(R.label || "эталон " + (i + 1))}${R.name ? "" : " (пусто)"}</option>`).join("");
  if (cur && +cur < V.refs.length) sel.value = cur;
}
function renderSay() {
  $("#vSayText").value = V.say.text || "";
  $("#vSayRes").innerHTML = (V.say.res || []).map((r, i) => `<div class="vSayItem"><audio src="${r.url}" controls preload="metadata"></audio>
    <div class="tiny"><b>${esc(r.label || "")}</b> · ${(+r.duration || 0).toFixed(1)} с · вариант ${r.seed} — ${esc((r.text || "").slice(0, 120))}</div>
    <div class="links"><a href="${r.url}" download>⬇ WAV</a><a href="#" data-sx="${i}">✕</a></div></div>`).join("");
  $$("[data-sx]", $("#vSayRes")).forEach(a => a.onclick = e => { e.preventDefault(); V.say.res.splice(+a.dataset.sx, 1); renderSay(); saySave(); });
}
$("#vSayText").oninput = () => { V.say.text = $("#vSayText").value; saySave(); };
$("#vSayText").addEventListener("keydown", e => { if (e.key === "Enter" && e.ctrlKey) { e.preventDefault(); $("#vSayGo").click(); } });
$("#vSayGo").onclick = async () => {
  const text = $("#vSayText").value.trim();
  if (!text) return toast("Впишите текст");
  const ri = +$("#vSayRef").value || 0, R = V.refs[ri];
  const st = $("#vSaySt"), btn = $("#vSayGo");
  if (!R || !R.name) { st.innerHTML = `<span class="bad">Эталон «${esc((R && R.label) || "")}» пустой — добавьте для него видео слева.</span>`; return; }
  btn.disabled = true; busy(1);
  try {
    if (!R.wav) { st.textContent = "Готовлю эталон…"; await takeVoice(R); }
    const { id } = await api("/api/voice/say", {
      text, ref: ri, refs: V.refs.map(x => x.wav ? { wav: x.wav, text: x.text || "" } : null), lang: $("#vLang").value,
      duration: +$("#vSayDur").value || 0, speed: +$("#vSaySpeed").value || 1,
      opts: { steps: +$("#vSteps").value, guidance: +$("#vGuid").value, seed: +$("#vSeed").value, gain: +$("#vSayGain").value || 0 },
    });
    const r = await track(id, q => st.textContent = q.msg || "…");
    V.say.res.unshift({ url: r.wav_url, seed: r.seed, duration: r.duration, text, label: R.label });
    V.say.res = V.say.res.slice(0, 12);
    st.innerHTML = `<span class="okc">Готово. Не понравилось — 🎲 (вариант) в настройках слева и ещё раз.</span>`;
    renderSay(); saySave();
  } catch (e) { st.innerHTML = `<span class="bad">${esc(e.message)}</span>`; }
  finally { btn.disabled = false; busy(-1); }
};
const _renderRefTabs = renderRefTabs;
renderRefTabs = function () { _renderRefTabs(); renderSayRefs(); };
renderSayRefs(); renderSay();

renderRef(); renderClips(); V.loaded = true;
document.body.dataset.view = V.view;
if (V.view === "3d") setTimeout(() => window.t3Show && window.t3Show(), 0);
if (V.view === "music") setTimeout(() => window.muShow && window.muShow(), 0);
if (V.view === "voice") vRefreshState(); else setTimeout(vRefreshState, 1500);
