"use strict";
/* ================= Вкладка «Музыка»: ACE‑Step 1.5, MiniMax Music 3, YuE2, Stable Audio 3 ================= */
const MU = { eng: "ace", files: {}, tracks: [], job: null, newUrl: null };
const MU_INFO = {
  ace: { name: "ACE‑Step 1.5", max: 600, hint: "Быстрая (8 шагов), поёт на 50+ языках, включая русский, до 10 минут. Задаются язык, темп, тональность и размер.",
    tags: "теги через запятую: жанр, инструменты, вокал, настроение", ph: "indie pop, female vocals, acoustic guitar, warm piano, melancholic, autumn, 100 BPM" },
  mm3: { name: "MiniMax Music 3", max: 300, hint: "Чистый сведённый звук и точное следование разметке текста. До 5 минут; модель может закончить песню раньше. Медленнее ACE‑Step.",
    tags: "1–2 предложения: жанр, инструменты, голос, настроение", ph: "A warm indie pop ballad with acoustic guitar and soft piano, intimate female vocal, melancholic autumn mood, mid tempo." },
  yue2: { name: "YuE2", max: 360, hint: "Сильный выразительный вокал, лучше всего по‑английски и по‑китайски. Медленная, нужно 16+ ГБ видео. Только некоммерческое использование.",
    tags: "5–8 слов через пробел: жанр, инструмент, настроение, голос", ph: "indie pop acoustic melancholic female warm vocal" },
  sa3: { name: "Stable Audio 3", max: 380, hint: "Инструментал, фоновая музыка и звуковые эффекты — без вокала. 8 шагов, до 6 минут.",
    tags: "описание звучания: жанр, инструменты, темп, развитие трека", ph: "Lo-fi hip hop beat at 85 BPM with dusty vinyl drums, mellow Rhodes chords and a soft bass line, relaxed late-night mood." },
};
const MUKEY = "zl_music";
(function muInit() {
  const keys = [];
  for (const q of ["minor", "major"]) for (const r of ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]) keys.push(`${r} ${q}`);
  $("#muKey").innerHTML = keys.map(k => `<option>${k}</option>`).join("");
  $("#muKey").value = "A minor";
  let o = null; try { o = JSON.parse(localStorage.getItem(MUKEY) || "null"); } catch { }
  if (!o) return;
  MU.eng = MU_INFO[o.eng] ? o.eng : "ace"; MU._model = o.model;
  for (const [id, k] of [["#muIdea", "idea"], ["#muTags", "tags"], ["#muLyr", "lyr"], ["#muDur", "dur"], ["#muLang", "lang"], ["#muBpm", "bpm"], ["#muKey", "key"], ["#muSig", "sig"]]) if (o[k] !== undefined) $(id).value = o[k];
  $("#muInst").checked = !!o.inst;
})();
function muSave() {
  try {
    localStorage.setItem(MUKEY, JSON.stringify({ eng: MU.eng, model: $("#muModel").value, idea: $("#muIdea").value, tags: $("#muTags").value, lyr: $("#muLyr").value,
      dur: $("#muDur").value, lang: $("#muLang").value, bpm: $("#muBpm").value, key: $("#muKey").value, sig: $("#muSig").value, inst: $("#muInst").checked }));
  } catch { }
}
["#muIdea", "#muTags", "#muLyr", "#muDur", "#muLang", "#muBpm", "#muKey", "#muSig", "#muInst", "#muModel"].forEach(s => $(s).addEventListener("change", muSave));
["#muTags", "#muLyr"].forEach(s => $(s).addEventListener("input", muSave));

const fmtT = s => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
function muDurShow() { $("#muDurV").textContent = fmtT(+$("#muDur").value); }
$("#muDur").addEventListener("input", () => { muDurShow(); muSave(); });

/* ---------- выбор нейросети ---------- */
function muSetEng(e) {
  MU.eng = e; const I = MU_INFO[e];
  $$("#muEng [data-e]").forEach(b => b.classList.toggle("on", b.dataset.e === e));
  $("#muDur").max = I.max; if (+$("#muDur").value > I.max) $("#muDur").value = I.max; muDurShow();
  $("#muTagsHint").textContent = "— " + I.tags; $("#muTags").placeholder = I.ph;
  $$(".muAce").forEach(x => x.classList.toggle("hidden", e !== "ace"));
  $("#muLyrBox").classList.toggle("hidden", e === "sa3" || $("#muInst").checked);
  $("#muInstBox").classList.toggle("hidden", e === "sa3");
  muFiles(); muSave();
}
$$("#muEng [data-e]").forEach(b => b.onclick = () => muSetEng(b.dataset.e));
$("#muInst").addEventListener("change", () => $("#muLyrBox").classList.toggle("hidden", MU.eng === "sa3" || $("#muInst").checked));
async function muFiles() {
  try { MU.files = await api("/api/music/state"); } catch { MU.files = {}; }
  const f = MU.files[MU.eng] || { models: [] }, sel = $("#muModel"), cur = sel.value || MU._model;
  sel.innerHTML = f.models.length ? f.models.map(n => `<option value="${esc(n)}">${esc(n)}</option>`).join("") : `<option value="">— нет модели —</option>`;
  if (cur && f.models.includes(cur)) sel.value = cur;
  $("#muModelBox").classList.toggle("hidden", f.models.length < 2);
  const miss = [];
  if (!f.models.length) miss.push("модель");
  if ("clip1" in f && !f.clip1) miss.push("текстовый энкодер");
  if ("clip2" in f && !f.clip2) miss.push("второй энкодер");
  if ("vae" in f && !f.vae) miss.push("VAE");
  const I = MU_INFO[MU.eng];
  $("#muEngHint").innerHTML = esc(I.hint) + (miss.length ? `<br><span class="warnTxt">Не хватает: <b>${miss.join(", ")}</b>. <a href="#" id="muGet">Скачать в «Модели» → «Музыка»</a></span>` : "");
  const g = $("#muGet"); if (g) g.onclick = ev => { ev.preventDefault(); S.catFam = "music"; openModels("catalog"); };
}
$("#muModels").onclick = () => { S.catFam = "music"; openModels("catalog"); };

/* ---------- помощник: стиль и текст ---------- */
$("#muAi").onclick = async () => {
  const b = $("#muAi"); b.disabled = true; $("#muAiSt").textContent = "Помощник сочиняет…";
  try {
    const r = await api("/api/llm/music", { engine: MU.eng, idea: $("#muIdea").value, tags: $("#muTags").value, lyrics: $("#muLyr").value,
      language: $("#muLang").value, instrumental: $("#muInst").checked, duration: +$("#muDur").value });
    if (r.tags) $("#muTags").value = r.tags;
    if (r.lyrics !== undefined && !$("#muInst").checked && MU.eng !== "sa3") $("#muLyr").value = r.lyrics;
    $("#muAiSt").textContent = ""; muSave();
  } catch (e) { $("#muAiSt").textContent = e.message; }
  b.disabled = false;
};

/* ---------- генерация ---------- */
function muErr(m) { $("#muErr").textContent = m || ""; $("#muErr").classList.toggle("hidden", !m); }
async function muGo() {
  if (MU.job) return;
  muErr("");
  const body = { engine: MU.eng, model: $("#muModel").value, tags: $("#muTags").value, lyrics: $("#muLyr").value, instrumental: $("#muInst").checked,
    duration: +$("#muDur").value, language: $("#muLang").value, bpm: +$("#muBpm").value, keyscale: $("#muKey").value, timesig: $("#muSig").value,
    steps: +$("#muSteps").value || 0, seed: +$("#muSeed").value };
  MU.job = "pending"; $("#muGo").disabled = true;
  $("#muProg").classList.remove("hidden"); $("#muBar").style.width = "0"; $("#muTxt").textContent = "Отправка…";
  let r;
  try { r = await api("/api/music/generate", body); }
  catch (e) { muErr(e.message); MU.job = null; $("#muGo").disabled = false; $("#muProg").classList.add("hidden"); return; }
  MU.job = r.id;
  const t0 = Date.now();
  while (MU.job === r.id) {
    await new Promise(res => setTimeout(res, 600));
    let j; try { j = await api("/api/job/" + r.id); } catch { continue; }
    const pct = j.max ? Math.round(100 * j.value / j.max) : 0;
    $("#muBar").style.width = (j.status === "done" ? 100 : pct) + "%";
    $("#muTxt").textContent = `${j.stage || ""}${j.max ? ` ${j.value}/${j.max}` : ""} · ${((Date.now() - t0) / 1000).toFixed(0)} с`;
    if (j.status === "done") {
      MU.newUrl = (j.images || []).find(u => /\.(mp3|flac|wav|opus)(\?|$)/i.test(u)) || null;
      if (MU.newUrl) { await muHistory(); toast(`Трек готов за ${((Date.now() - t0) / 1000).toFixed(0)} с`); }
      else muErr("Движок не вернул звук");
      break;
    }
    if (j.status === "error" || j.status === "cancelled") { muErr(j.error || "Ошибка"); break; }
  }
  MU.job = null; $("#muGo").disabled = false;
  setTimeout(() => $("#muProg").classList.add("hidden"), 800);
}
$("#muGo").onclick = muGo;
$("#muStop").onclick = async () => { try { await api("/api/interrupt", {}); } catch { } };
$("#muSeedR").onclick = () => { $("#muSeed").value = -1; };
document.addEventListener("keydown", e => { if (document.body.dataset.view === "music" && e.ctrlKey && e.key === "Enter") { e.preventDefault(); muGo(); } });

/* ---------- список треков ---------- */
async function muHistory() {
  let r; try { r = await api("/api/gallery"); } catch { return; }
  MU.tracks = [];
  for (const e of r.items) for (const u of e.images) if (/\.(mp3|flac|wav|opus)(\?|$)/i.test(u) && e.params.family === "music") MU.tracks.push({ url: u, e });
  muRender();
}
function muRender() {
  const list = $("#muList");
  if (!MU.tracks.length) { list.innerHTML = `<div class="muEmpty"><div class="big">🎵</div>Здесь появятся треки. Опишите стиль, впишите текст (или нажмите «✨ Придумать») и нажмите «Сочинить».</div>`; return; }
  list.innerHTML = MU.tracks.slice(0, 60).map((t, i) => {
    const p = t.e.params, name = (p.tags || "").split(/[,.]/)[0].slice(0, 60) || "Трек";
    return `<div class="muItem ${t.url === MU.newUrl ? "new" : ""}" data-i="${i}">
      <div class="h"><b>${esc(name)}</b><span class="m">${esc(MU_INFO[p.engine]?.name || p.engine)} · ${fmtT(+p.duration || 0)} · ${esc(t.e.time)}${t.e.seconds ? ` · за ${t.e.seconds} с` : ""} · seed ${p.seed}</span></div>
      <audio controls preload="none" src="${t.url}"></audio>
      <div class="acts"><a class="ghost small btnlink" href="${t.url}" download>⬇ MP3</a>
        ${p.lyrics && p.lyrics !== "[Instrumental]" ? `<button class="ghost small" data-lyr="${i}">Текст</button>` : `<span class="m">инструментал</span>`}
        <button class="ghost small" data-again="${i}" title="Вернуть стиль, текст и настройки этого трека">↺ Настройки</button>
        <button class="ghost small" data-seed="${i}" title="Те же настройки и тот же seed">+ seed</button></div>
      <div class="lyr">${esc(p.lyrics || "")}</div></div>`;
  }).join("");
  $$("#muList [data-lyr]").forEach(b => b.onclick = () => b.closest(".muItem").classList.toggle("open"));
  const restore = (i, withSeed) => {
    const p = MU.tracks[i].e.params;
    muSetEng(p.engine); $("#muTags").value = p.tags || ""; $("#muInst").checked = !!p.instrumental;
    $("#muLyr").value = p.lyrics === "[Instrumental]" ? "" : (p.lyrics || ""); $("#muDur").value = p.duration || 120; muDurShow();
    if (p.engine === "ace") { $("#muLang").value = p.language || "ru"; $("#muBpm").value = p.bpm || 120; $("#muKey").value = p.keyscale || "A minor"; $("#muSig").value = p.timesig || "4"; }
    $("#muSeed").value = withSeed ? p.seed : -1;
    $("#muLyrBox").classList.toggle("hidden", p.engine === "sa3" || $("#muInst").checked);
    muSave(); toast(withSeed ? "Настройки и seed восстановлены" : "Настройки восстановлены");
  };
  $$("#muList [data-again]").forEach(b => b.onclick = () => restore(+b.dataset.again, false));
  $$("#muList [data-seed]").forEach(b => b.onclick = () => restore(+b.dataset.seed, true));
}

/* ---------- вход во вкладку ---------- */
window.muShow = async () => { muSetEng(MU.eng); if (!MU.tracks.length) await muHistory(); else muRender(); };
$("#muTab").onclick = () => setView("music");
