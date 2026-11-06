"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const api = async (url, body, method) => {
  const opt = body !== undefined ? { method: method || "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const r = await fetch(url, opt);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
  return j;
};
const toast = (msg, ms = 2600) => { const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden"); clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add("hidden"), ms); };
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtBytes = n => { n = +n || 0; const u = ["Б", "КБ", "МБ", "ГБ", "ТБ"]; let i = 0; while (n >= 1024 && i < 4) { n /= 1024; i++; } return (i ? n.toFixed(1) : n) + " " + u[i]; };

const SAMPLERS = ["euler", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde", "dpmpp_3m_sde", "res_multistep", "er_sde", "lms", "deis", "ipndm", "uni_pc", "ddim", "lcm", "gradient_estimation", "sa_solver"];
const SCHEDULERS = [["simple", "simple"], ["flux2", "flux2 (родной)"], ["beta", "beta"], ["normal", "normal"], ["sgm_uniform", "sgm_uniform"], ["karras", "karras"], ["exponential", "exponential"], ["kl_optimal", "kl_optimal"], ["linear_quadratic", "linear_quadratic"], ["ddim_uniform", "ddim_uniform"]];
const RATIOS = [["1:1", 1, 1], ["4:5", 4, 5], ["3:4", 3, 4], ["2:3", 2, 3], ["9:16", 9, 16], ["5:4", 5, 4], ["4:3", 4, 3], ["3:2", 3, 2], ["16:9", 16, 9], ["21:9", 21, 9]];
const FIELDS = ["avae", "upscaler", "ltxTwo", "ltxSrc", "ltxAnchor", "duration", "turbo", "refMax", "unetRef", "unet", "clip", "vae", "weight_dtype", "prompt", "negative", "width", "height", "steps", "cfg", "sampler", "scheduler", "batch", "seed", "denoise", "refMp", "useText", "useImage", "civBlur", "civNsfw", "civBase", "civSort", "civPeriod", "aiCensor"];

const LOCK = "";  // одна программа для Qwen (картинки) и MiniMax H3 (видео)
const ARCH_OF = { qwen: "qwen21", krea2: "krea2", flux3: "flux3", minimax: "minimax", ltx: "ltx" };
let S = { models: {}, loras: [], images: [], imageMode: "reference", lastSeed: null, job: null, results: [], engine: "", family: "qwen", fam: {}, frames: { first: null, last: null }, mm: "frames", refs: [] };

/* ---------------- нейросети (семейства) ---------------- */
const FAM = {
  minimax: { name: "MiniMax H3", video: true, defaults: { steps: 8, cfg: 1, sampler: "res_multistep", scheduler: "simple", width: 1344, height: 768, weight_dtype: "default", negative: "", prompt: "", batch: 1, seed: -1, denoise: 0.6, refMp: 1, duration: 5, turbo: true } },
  ltx: { name: "LTX‑2.5", video: true, defaults: { steps: 8, cfg: 1, sampler: "euler_ancestral", scheduler: "simple", width: 1280, height: 704, weight_dtype: "default", negative: "", prompt: "", batch: 1, seed: -1, denoise: 0.6, refMp: 1, duration: 5, ltxSrc: 0.7, ltxAnchor: 1 } },
  flux3: { name: "FLUX.3 Klein", defaults: { steps: 16, cfg: 1, sampler: "er_sde", scheduler: "simple", width: 1024, height: 1024, weight_dtype: "default", negative: "", prompt: "", batch: 1, seed: -1, denoise: 0.6, refMp: 1 } },
  krea2: { name: "Krea 2", defaults: { steps: 8, cfg: 1, sampler: "euler", scheduler: "simple", width: 1024, height: 1024, weight_dtype: "default", negative: "", prompt: "", batch: 1, seed: -1, denoise: 0.6, refMp: 1 } },
  qwen: { name: "Qwen‑Image 2.1", defaults: { steps: 30, cfg: 1, sampler: "euler", scheduler: "simple", width: 1024, height: 1536, weight_dtype: "default", negative: "", prompt: "", batch: 1, seed: -1, denoise: 0.6, refMp: 1 } },
};
const FAM_KEYS = ["unet", "clip", "vae", "weight_dtype", "steps", "cfg", "sampler", "scheduler", "width", "height", "negative",
  "prompt", "batch", "seed", "denoise", "refMp", "duration", "avae", "unetRef", "upscaler", "ltxSrc", "ltxAnchor"];
const KINDS = { image: ["qwen", "krea2", "flux3"], video: ["minimax", "ltx"] };
const kindOf = f => FAM[f] && FAM[f].video ? "video" : "image";
const isVidFam = (f = S.family) => !!(FAM[f] && FAM[f].video);

/* ---------------- init ---------------- */
function fillSelect(sel, items, keep) {
  const cur = keep ?? sel.value;
  sel.innerHTML = items.map(i => Array.isArray(i) ? `<option value="${esc(i[0])}">${esc(i[1])}</option>` : `<option value="${esc(i)}">${esc(i)}</option>`).join("");
  if (cur && [...sel.options].some(o => o.value === cur)) sel.value = cur;
}
fillSelect($("#sampler"), SAMPLERS, "euler");
fillSelect($("#scheduler"), SCHEDULERS, "simple");
$("#ratios").innerHTML = RATIOS.map(r => `<button data-r="${r[1]}:${r[2]}">${r[0]}</button>`).join("");

function saveUI() {
  const o = {};
  for (const f of FIELDS) { const el = $("#" + f); if (!el) continue; o[f] = el.type === "checkbox" ? el.checked : el.value; }
  o.loras = S.loras; o.imageMode = S.imageMode; o.lastSeed = S.lastSeed; o.character = S.character || null; o.charPhrase = $("#charPhrase").checked; o.family = S.family; o.fam = S.fam; o.promptSrc = S.promptSrc || "user"; o.frames = S.frames; o.mm = S.mm; o.refs = S.refs; o.ratio = S.ratio || null; o.lockAR = S.lockAR; o.lastVid = S.lastVid; o.lastImg = S.lastImg;
  try { localStorage.setItem("flux2studio", JSON.stringify(o)); } catch { }
}
function loadUI() {
  let o = null;
  try { o = JSON.parse(localStorage.getItem("flux2studio") || "null"); } catch { }
  if (!o) return;
  for (const f of FIELDS) { const el = $("#" + f); if (!el || o[f] === undefined) continue; if (el.type === "checkbox") el.checked = o[f]; else el.value = o[f]; }
  S.loras = o.loras || []; S.imageMode = o.imageMode || "reference"; S.lastSeed = o.lastSeed ?? null;
  S.character = o.character || null; if (o.charPhrase === false) $("#charPhrase").checked = false;
  S.family = FAM[o.family] ? o.family : "qwen"; S.fam = o.fam || {}; delete S.fam.flux; S.promptSrc = o.promptSrc || "user";
  S.frames = o.frames || { first: null, last: null }; S.mm = o.mm === "ref" ? "ref" : "frames"; S.refs = Array.isArray(o.refs) ? o.refs : []; S.ratio = o.ratio || null; S.lockAR = o.lockAR !== false; S.lastVid = FAM[o.lastVid] ? o.lastVid : null; S.lastImg = FAM[o.lastImg] ? o.lastImg : null;
  S._pending = { unet: o.unet, clip: o.clip, vae: o.vae, avae: o.avae, unetRef: o.unetRef, upscaler: o.upscaler };
}
loadUI();

/* ---------------- state polling ---------------- */
async function refreshState() {
  try {
    const st = await api("/api/state");
    const sig = JSON.stringify(st.models);
    const changed = sig !== S._msig; S._msig = sig;
    S.models = st.models; S.engine = st.engine; S.qwenSupported = st.qwen_supported; S.flux3Supported = st.flux3_supported; S.gguf = st.gguf || {}; famWarn(); ggufWarn();
    const el = $("#engineStatus");
    el.className = "status " + st.engine;
    const dev = (st.device && st.device.name ? ` · ${st.device.name.replace(/^cuda:\d+\s*/, "").split(":")[0]} · видео свободно ${st.device.vram_free} из ${st.device.vram_total}` : "")
      + (st.device && st.device.ram_total ? ` · ОЗУ свободно ${st.device.ram_free} из ${st.device.ram_total}` : "");
    el.classList.toggle("lowram", !!(st.device && st.device.ram_free_gb !== undefined && st.device.ram_free_gb < 3));
    el.title = st.device && st.device.ram_free_gb < 3 ? "Оперативной памяти почти не осталось — закройте лишние программы или включите профиль «Бережный к ОЗУ» (Настройки → Память)" : "";
    $(".txt", el).textContent = ({ ready: "Готов", starting: "Движок запускается…", error: "Ошибка движка", missing: "Движок не установлен", stopped: "Остановлен" }[st.engine] || st.engine) + dev;
    if (changed || S._pending) updateModelSelects();
  } catch (e) {
    const el = $("#engineStatus"); el.className = "status error"; $(".txt", el).textContent = "Нет связи с сервером";
  }
}
/* какой нейросети принадлежит файл — по имени */
function fileFamily(folder, name) {
  const n = (name || "").toLowerCase();
  // файлы вкладки «Музыка» (MiniMax Music 3, ACE‑Step, YuE, Stable Audio) — не для картинок и видео,
  // хотя в имени бывает «minimax»: иначе энкодер MiniMax Music 3 подставлялся в видео MiniMax H3
  if (/music|ace[_-]?step|yue|stable[_-]?audio/.test(n)) return "music";
  if (folder === "diffusion_models") {
    const it = (S.models.diffusion_models || []).find(x => x.name === name);  // сервер определил по слоям файла
    if (it && it.fam) return it.fam;
    if (/minimax|h3[_-]?(fl2va|ref2va)/.test(n)) return "minimax";
    if (/ltx/.test(n)) return "ltx";
    if (/krea/.test(n)) return "krea2";
    if (/flux[._-]?3/.test(n)) return "flux3";
    return /qwen[\s_.-]*image|qwen[\s_.-]*2\.?1|qw21/.test(n) ? "qwen" : "other";
  }
  if (folder === "text_encoders") return /minimax/.test(n) ? "minimax" : /ltx|gemma/.test(n) ? "ltx" : /upsampler/.test(n) ? "other" : /flux[._-]?3/.test(n) ? "flux3" : /qwen3[\s_.-]*vl[\s_.-]*4b.*fp8|krea/.test(n) ? "krea2" : /qwen3[\s_.-]*vl[\s_.-]*4b/.test(n) ? "flux3" : /qwen3[\s_.-]*vl/.test(n) ? "qwen" : "other";
  if (folder === "vae") return /minimax|h3_(video|audio)/.test(n) ? "minimax" : /ltx/.test(n) ? "ltx" : /flux[._-]?[23]/.test(n) ? "flux3" : /krea|qwen_image_vae\./.test(n) ? "krea2" : /qwen/.test(n) ? "qwen" : "other";
  if (folder === "latent_upscale_models") return /ltx/.test(n) ? "ltx" : "other";
  return "any";
}
/* файл подходит нейросети: Qwen3‑VL 4B — общий энкодер Krea 2 и FLUX.3 */
const fileFits = (folder, name, fam) => {
  const f = fileFamily(folder, name);
  if (f === fam) return true;
  return folder === "text_encoders" && ["krea2", "flux3"].includes(fam) && ["krea2", "flux3"].includes(f);
};
const isAudioVae = n => /audio/i.test(n || "");
const isRefModel = n => /ref2?v/i.test(n || "");
function fillModelSelect(sel, folder, arr, keep) {
  const alien = n => { const f = fileFamily(folder, n); return !!FAM[f] && !fileFits(folder, n, S.family); };
  const cur = (keep && !alien(keep) ? keep : "") || (fileFits(folder, sel.value, S.family) ? sel.value : "");
  const own = x => fileFamily(folder, x.name) === S.family ? 0 : 1;   // свои файлы — первыми, общие — после
  const mine = arr.filter(x => fileFits(folder, x.name, S.family)).sort((a, b) => own(a) - own(b)), other = arr.filter(x => !fileFits(folder, x.name, S.family));
  const o = x => `<option value="${esc(x.name)}">${esc(x.name)}  (${x.size_h})</option>`;
  sel.innerHTML = (mine.length ? mine.map(o).join("") : `<option value="">— нет файлов для ${FAM[S.family].name} —</option>`)
    + (other.length && !LOCK ? `<optgroup label="Файлы другой нейросети">${other.map(o).join("")}</optgroup>` : "");
  if (cur && [...sel.options].some(x => x.value === cur)) sel.value = cur;
  else if (mine.length) sel.value = mine[0].name;
}
function updateModelSelects() {
  const m = S.models, p = S._pending || {};
  const dm = m.diffusion_models || [];
  fillModelSelect($("#unet"), "diffusion_models", S.family === "minimax" ? dm.filter(x => !isRefModel(x.name)) : dm, p.unet);
  const ups = (m.latent_upscale_models || []).filter(x => /spatial/i.test(x.name) || !/temporal/i.test(x.name));
  $("#upscaler").innerHTML = ups.length ? ups.map(x => `<option value="${esc(x.name)}">${esc(x.name)}  (${x.size_h})</option>`).join("") : `<option value="">— нет апскейлера: скачайте в «Модели» → LTX‑2.5 —</option>`;
  if (p.upscaler && ups.some(x => x.name === p.upscaler)) $("#upscaler").value = p.upscaler;
  fillModelSelect($("#unetRef"), "diffusion_models", dm.filter(x => isRefModel(x.name)), p.unetRef || $("#unetRef").value);
  if (!dm.some(x => isRefModel(x.name))) $("#unetRef").innerHTML = `<option value="">— нет модели Ref2VA: скачайте в «Модели» → MiniMax H3 · Референсы —</option>`;
  fillModelSelect($("#clip"), "text_encoders", m.text_encoders || [], p.clip);
  const vaes = m.vae || [];
  fillModelSelect($("#vae"), "vae", isVidFam() ? vaes.filter(x => !isAudioVae(x.name)) : vaes, p.vae);
  fillModelSelect($("#avae"), "vae", vaes.filter(x => isAudioVae(x.name)), p.avae || $("#avae").value);
  S._pending = null;
  const loras = (m.loras || []).map(x => x.name);
  if (JSON.stringify(loras) !== S._lsig) { S._lsig = JSON.stringify(loras); syncTurbo(false); loadLoraLib(); }
  checkPair();
  const missing = [], has = (f, arr) => (arr || []).some(x => fileFits(f, x.name, S.family));
  if (!has("diffusion_models", m.diffusion_models)) missing.push("модель");
  if (!has("text_encoders", m.text_encoders)) missing.push("текстовый энкодер");
  if (!has("vae", (m.vae || []).filter(x => !isAudioVae(x.name)))) missing.push(isVidFam() ? "видео‑VAE" : "VAE");
  if (isVidFam() && !has("vae", (m.vae || []).filter(x => isAudioVae(x.name)))) missing.push("аудио‑VAE");
  if (S.family === "minimax" && S.mm === "ref" && !dm.some(x => isRefModel(x.name))) missing.push("модель Ref2VA (для референсов)");
  if (S.family === "ltx" && $("#ltxTwo").checked && !(m.latent_upscale_models || []).length) missing.push("апскейлер ×2 (или выключите «Апскейл ×2»)");
  $("#setupHint").innerHTML = missing.length ? `Для ${FAM[S.family].name} не хватает: <b>${missing.join(", ")}</b>. Откройте <a href="#" id="openModelsLink" style="color:var(--accent)">Модели</a> — скачайте рекомендованные или добавьте свои файлы.` : "";
  const l = $("#openModelsLink"); if (l) l.onclick = e => { e.preventDefault(); S.catFam = S.family; openModels("catalog"); };
}
function checkPair() {
  const u = ($("#unet").value || "").toLowerCase(), c = ($("#clip").value || "").toLowerCase();
  let w = "";
  const wrong = [["diffusion_models", "#unet", "модель"], ["text_encoders", "#clip", "энкодер"], ["vae", "#vae", "VAE"]]
    .filter(([f, id]) => $(id).value && !fileFits(f, $(id).value, S.family)).map(x => x[2]);
  if (wrong.length) w = `Выбраны файлы другой нейросети (${wrong.join(", ")}) — для ${FAM[S.family].name} выберите её файлы.`;
  if (S.family === "minimax") {
    if (!$("#turbo").checked && +$("#steps").value < 14) w += (w ? " " : "") + "Без турбо‑LoRA MiniMax H3 нужно ~20 шагов (ниже 15 качество падает).";
    $("#pairWarn").textContent = w; $("#pairWarn").classList.toggle("hidden", !w);
    return;
  }
  if (S.family === "ltx") {
    if (S.frames.last && $("#ltxTwo").checked) w += (w ? " " : "") + "С последним кадром LTX генерирует в один этап — апскейл ×2 будет пропущен.";
    $("#pairWarn").textContent = w; $("#pairWarn").classList.toggle("hidden", !w);
    return;
  }
  if (S.family === "flux3") {
    const base = /base/.test(u);
    if (base && (+$("#cfg").value < 2 || +$("#steps").value < 30)) w += (w ? " " : "") + "FLUX.3 Klein Base — неускоренная модель: ей нужно ~50 шагов и CFG ≈3.5 (кнопка «Base: 50 шагов»).";
    if (!base && u && +$("#cfg").value > 1.5) w += (w ? " " : "") + "FLUX.3 Klein (distilled) работает с CFG 1 и 4–16 шагами — больше CFG пережигает.";
    $("#pairWarn").textContent = w; $("#pairWarn").classList.toggle("hidden", !w);
    return;
  }
  if (S.family === "krea2") {
    const fast = /turbo/.test(u) || kreaTurboLora();
    if (fast && (+$("#steps").value > 14 || +$("#cfg").value > 1.5)) w += (w ? " " : "") + "Krea 2 Turbo (и Raw с турбо‑LoRA) рассчитана на 8 шагов и CFG 1 — больше шагов не улучшает, а CFG > 1 пережигает.";
    if (!fast && /raw/.test(u) && (+$("#steps").value < 20 || +$("#cfg").value < 2)) w += (w ? " " : "") + "Krea 2 Raw — базовая модель: ей нужно 28–52 шага и CFG ≈3.5 (кнопка «Raw: 52 шага» ниже). Для 8 шагов добавьте турбо‑LoRA.";
    $("#pairWarn").textContent = w; $("#pairWarn").classList.toggle("hidden", !w);
    return;
  }
  if (S.family === "qwen" && qwenTurbo()) {
    if (+$("#cfg").value > 1.2) w += (w ? " " : "") + "Турбо‑режим Qwen работает с CFG 1.";
    $("#pairWarn").textContent = w; $("#pairWarn").classList.toggle("hidden", !w);
    return;
  }
  if (S.family === "qwen") {
    if (+$("#cfg").value > 1.5 && !$("#negative").value.trim()) w += (w ? " " : "") + "Qwen‑Image 2.1 работает с CFG 1; CFG больше 1 имеет смысл только вместе с негативным промптом.";
    if (+$("#steps").value < 16) w += (w ? " " : "") + "Для Qwen‑Image 2.1 нужно 25–50 шагов (официально 40–50).";
    $("#pairWarn").textContent = w; $("#pairWarn").classList.toggle("hidden", !w);
    return;
  }
  const is4 = /4b/.test(u), is9 = /9b/.test(u);
  const c4 = /qwen.?3.?4b/.test(c), c8 = /qwen.?3.?8b/.test(c);
  if (is9 && c4) w = "Модель 9B требует энкодер Qwen3 8B (qwen_3_8b…), а выбран 4B.";
  if (is4 && c8) w = "Модель 4B требует энкодер Qwen3 4B (qwen_3_4b…), а выбран 8B.";
  if (/base/.test(u) && +$("#steps").value < 12) w += (w ? " " : "") + "Base‑модели нужно ~20–50 шагов и CFG ~4 (кнопка «Base» ниже).";
  if (u && !/base/.test(u) && +$("#cfg").value > 2) w += (w ? " " : "") + "Это distilled‑модель: ей нужен CFG ≈ 1 и ~4 шага, с большим CFG картинка пережигается (кнопка «Distilled» ниже).";
  $("#pairWarn").textContent = w; $("#pairWarn").classList.toggle("hidden", !w);
}
/* Qwen‑Image 2.1 Turbo: отдельная модель («turbo» в имени) или турбо‑LoRA в стеке → 8 шагов, CFG 1 */
const isQwenTurboLora = n => /turbo/i.test(n || "") && /qwen/i.test(n || "");
const qwenTurbo = () => S.family === "qwen" && (/turbo/i.test($("#unet").value || "") || S.loras.some(l => l.enabled && isQwenTurboLora(l.name)));
function qwenTurboSteps() {
  if (!qwenTurbo()) return;
  if (+$("#steps").value !== 8) { $("#steps").value = 8; $("#cfg").value = 1; toast("Qwen Turbo: 8 шагов, CFG 1 (турбо‑расписание ставится само)", 3500); }
}
$("#unet").addEventListener("change", () => {
  if (S.family === "qwen") qwenTurboSteps();
  if (S.family === "flux3") {
    const base = /base/i.test($("#unet").value || "");
    if (base && +$("#steps").value < 30) { $("#steps").value = 50; $("#cfg").value = 3.5; toast("FLUX.3 Klein Base: 50 шагов, CFG 3.5", 3500); }
    if (!base && +$("#cfg").value > 1.5) { $("#steps").value = 16; $("#cfg").value = 1; toast("FLUX.3 Klein: 16 шагов, CFG 1", 3000); }
  }
  if (S.family === "krea2" && !kreaTurboLora()) {
    const raw = /raw/i.test($("#unet").value || "");
    if (raw && +$("#steps").value < 20) { $("#steps").value = 52; $("#cfg").value = 3.5; toast("Krea 2 Raw: 52 шага, CFG 3.5 (базовая модель, медленнее турбо)", 4000); }
    if (!raw && /turbo/i.test($("#unet").value || "") && +$("#steps").value > 14) { $("#steps").value = 8; $("#cfg").value = 1; toast("Krea 2 Turbo: 8 шагов, CFG 1", 3000); }
  }
  checkPair();
});
$("#turbo").addEventListener("change", () => {
  if (S.family !== "minimax") return;
  if (!$("#turbo").checked && +$("#steps").value < 14) $("#steps").value = 20;
  syncTurbo();
});
["#unet", "#unetRef", "#clip", "#steps", "#cfg", "#turbo"].forEach(s => $(s).addEventListener("change", () => { checkPair(); ggufWarn(); saveUI(); }));
/* ---- поддержка .gguf ---- */
function ggufWarn() {
  const box = $("#ggufBox"); if (!box) return;
  const u = ($("#unet").value || "").toLowerCase(), g = S.gguf || {};
  const need = u.endsWith(".gguf") && !g.installed;
  box.classList.toggle("hidden", !need && !g.running && !g.error);
  if (g.running) box.innerHTML = `Устанавливаю поддержку GGUF… ${g.total ? Math.round(100 * g.done / g.total) + "%" : ""}`;
  else if (need) box.innerHTML = `Модель .gguf требует поддержку GGUF в движке (узлы ComfyUI‑GGUF, ~1 МБ, один раз; движок перезапустится).${g.error ? `<div class="e">Ошибка: ${esc(g.error)}</div>` : ""} <button class="small primary" id="ggufInstall">Установить поддержку GGUF</button>`;
  else box.innerHTML = g.error ? `Поддержка GGUF: ошибка — ${esc(g.error)}` : "";
  const b = $("#ggufInstall"); if (b) b.onclick = async () => { b.disabled = true; try { await api("/api/gguf/install", {}); toast("Устанавливаю поддержку GGUF…"); } catch (e) { toast(e.message, 5000); } refreshState(); };
}

/* ---------------- text / image switches ---------------- */
function updateMode() {
  const t = $("#useText").checked, i = $("#useImage").checked;
  $("#textBlock").style.opacity = t ? 1 : .4;
  $("#prompt").disabled = !t; $("#negative").disabled = !t;
  $("#imageBlock").classList.toggle("hidden", !i);
  $$("#imageMode button").forEach(b => b.classList.toggle("on", b.dataset.v === S.imageMode));
  $("#refOpts").classList.toggle("hidden", S.imageMode !== "reference");
  $("#i2iOpts").classList.toggle("hidden", S.imageMode !== "img2img");
  $("#imgLimit").textContent = S.imageMode === "reference" ? (S.family === "flux3" ? "до 8 картинок — в промпте <ref_image_0>, <ref_image_1>… · Ctrl+V" : S.family === "krea2" ? "до 3 картинок стиля (Krea 2 переносит их стиль) · Ctrl+V" : "до 4 референсов · можно вставить из буфера Ctrl+V") : "используется 1 картинка · можно вставить Ctrl+V";
  let h = "";
  if (isVidFam()) {
    const f = S.frames.first, l = S.frames.last;
    $("#modeHint").textContent = S.family === "minimax" && S.mm === "ref" ? `Референсы (${S.refs.length} из 9) → видео со звуком`
      : f && l ? "Первый + последний кадр → видео со звуком" : f ? "Картинка → видео со звуком" : l ? "Последний кадр → видео со звуком" : "Текст → видео со звуком";
    saveUI(); return;
  }
  if (S.family === "krea2" && t && i && S.imageMode === "reference") h = "Текст + картинки стиля → новое изображение в их стиле";
  else if (S.family === "flux3" && t && i && S.imageMode === "reference") h = "Текст + картинки: правка и референсы — опишите готовый результат, картинки — <ref_image_0>, <ref_image_1>…";
  else if (t && !i) h = "Текст → изображение";
  else if (!t && i) h = S.imageMode === "reference" ? "Только изображение: вариации по референсу без промпта" : "img2img без промпта: перерисовка картинки";
  else if (t && i) h = S.imageMode === "reference" ? "Текст + изображение: редактирование по инструкции / по референсам" : "Текст + изображение: img2img по промпту";
  else h = "⚠ Включите хотя бы один вход";
  $("#modeHint").textContent = h;
  $("#fromImage").disabled = !(i && S.images.length);
  saveUI();
}
$("#useText").onchange = updateMode; $("#useImage").onchange = updateMode;
$$("#imageMode button").forEach(b => b.onclick = () => { S.imageMode = b.dataset.v; updateMode(); });
$("#denoise").oninput = () => { $("#denoiseV").textContent = (+$("#denoise").value).toFixed(2); saveUI(); };
$("#denoiseV").textContent = (+$("#denoise").value).toFixed(2);

/* ---------------- input images ---------------- */
async function addImages(files) {
  files = [...files].filter(f => f.type.startsWith("image/"));
  if (!files.length) return;
  const max = S.imageMode === "reference" ? (S.family === "krea2" ? 3 : S.family === "flux3" ? 8 : 4) : 1;
  if (S.imageMode === "img2img") S.images = [];
  const fd = new FormData();
  const take = files.slice(0, Math.max(0, max - S.images.length) || 1);
  take.forEach(f => fd.append("image", f, f.name || "paste.png"));
  try {
    const r = await fetch("/api/input_upload", { method: "POST", body: fd }).then(r => r.json());
    for (const name of r.names) {
      const dims = await new Promise(res => { const im = new Image(); im.onload = () => res([im.naturalWidth, im.naturalHeight]); im.onerror = () => res([0, 0]); im.src = "/input/" + encodeURIComponent(name); });
      S.images.push({ name, w: dims[0], h: dims[1] });
    }
    S.images = S.images.slice(-max);
    if (!$("#useImage").checked) $("#useImage").checked = true;
    renderThumbs(); updateMode();
  } catch (e) { toast("Не удалось загрузить картинку: " + e.message); }
}
function renderThumbs() {
  $("#imgThumbs").innerHTML = S.images.map((im, i) => `<div class="thumb"><img src="/input/${encodeURIComponent(im.name)}"><button data-rm="${i}">✕</button><span>${im.w}×${im.h}</span></div>`).join("");
  $$("#imgThumbs [data-rm]").forEach(b => b.onclick = e => { e.stopPropagation(); S.images.splice(+b.dataset.rm, 1); renderThumbs(); updateMode(); });
}
function dropzone(el, input, cb) {
  el.addEventListener("click", e => { if (e.target.closest("button")) return; input.click(); });
  input.addEventListener("change", () => { cb(input.files); input.value = ""; });
  el.addEventListener("dragover", e => { e.preventDefault(); el.classList.add("over"); });
  el.addEventListener("dragleave", () => el.classList.remove("over"));
  el.addEventListener("drop", e => { e.preventDefault(); el.classList.remove("over"); cb(e.dataTransfer.files); });
}
dropzone($("#imgDrop"), $("#imgFile"), addImages);
document.addEventListener("paste", e => {
  if (e.target.matches("input,textarea")) return;
  const files = [...(e.clipboardData?.files || [])];
  if (!files.length) return;
  if (S.family === "minimax" && S.mm === "ref") refUpload(files);
  else if (isVidFam()) frameUpload(S.frames.first ? (S.frames.last ? "first" : "last") : "first", files);
  else addImages(files);
});

/* ---------------- resolution ---------------- */
const snap = v => Math.max(128, Math.round(v / 32) * 32);
/* MiniMax H3: короткая сторона 768, площадь не больше 768×1344 */
function h3Canvas(a, c) {
  const r = a / c; let w = r >= 1 ? 768 * r : 768, h = r >= 1 ? 768 : 768 / r;
  const max = 768 * 1344; if (w * h > max) { const k = Math.sqrt(max / (w * h)); w *= k; h *= k; }
  return [Math.max(256, Math.round(w / 32) * 32), Math.max(256, Math.round(h / 32) * 32)];
}
/* LTX‑2.5: стороны кратны 64 (с апскейлом ×2) или 32; площадь около 1280×704 */
function ltxCanvas(a, c) {
  const m = $("#ltxTwo").checked ? 64 : 32, area = 1280 * 704;
  const w = Math.sqrt(area * a / c), h = w * c / a;
  return [Math.max(m * 4, Math.round(w / m) * m), Math.max(m * 4, Math.round(h / m) * m)];
}
function updateMp() {
  const w = +$("#width").value, h = +$("#height").value;
  $("#mpInfo").textContent = `${(w * h / 1048576).toFixed(2)} Мп`;
  $$("#ratios button").forEach(b => { const [a, c] = b.dataset.r.split(":").map(Number); b.classList.toggle("on", Math.abs(w / h - a / c) < 0.02); });
  saveUI();
}
function setRatio(a, c) {
  S.ratio = [a, c];
  if (S.family === "minimax") { const [w, h] = h3Canvas(a, c); $("#width").value = w; $("#height").value = h; updateMp(); return; }
  if (S.family === "ltx") { const [w, h] = ltxCanvas(a, c); $("#width").value = w; $("#height").value = h; updateMp(); return; }
  const mp = +$("#mpSel").value * 1048576;
  const w = Math.sqrt(mp * a / c), h = w * c / a;
  $("#width").value = snap(w); $("#height").value = snap(h); updateMp();
}
$$("#ratios button").forEach(b => b.onclick = () => { const [a, c] = b.dataset.r.split(":").map(Number); setRatio(a, c); });
$("#mpSel").onchange = () => { setRatio(+$("#width").value, +$("#height").value); };
$("#swapWH").onclick = () => { const w = $("#width").value; $("#width").value = $("#height").value; $("#height").value = w; if (S.ratio) S.ratio = [S.ratio[1], S.ratio[0]]; updateMp(); };
/* пропорции: вписали ширину — высота пересчитывается по выбранному соотношению (и наоборот) */
S.lockAR = S.lockAR !== false;
const lockShow = () => { $("#lockAR").classList.toggle("on", S.lockAR); $("#lockAR").textContent = S.lockAR ? "🔒" : "🔓"; $("#lockAR").title = S.lockAR ? "Пропорции закреплены: меняете ширину — высота пересчитывается сама (и наоборот). Нажмите, чтобы задавать стороны независимо" : "Стороны задаются независимо. Нажмите, чтобы закрепить текущие пропорции"; };
lockShow();
$("#lockAR").onclick = () => { S.lockAR = !S.lockAR; if (S.lockAR) S.ratio = [+$("#width").value, +$("#height").value]; lockShow(); saveUI(); };
const curRatio = () => S.ratio && S.ratio[0] > 0 && S.ratio[1] > 0 ? S.ratio : [+$("#width").value || 1, +$("#height").value || 1];
function fitOther(src, final) {
  const isW = src === "#width", v = +$(src).value; if (!v) return;
  const [a, c] = curRatio();
  let w = isW ? v : v * a / c, h = isW ? v * c / a : v;
  if (final && S.family === "minimax") {   // MiniMax: площадь не больше 768×1344
    const max = 768 * 1344;
    if (w * h > max * 1.001) { const k = Math.sqrt(max / (w * h)); w *= k; h *= k; toast("MiniMax H3: больше 768×1344 по площади нельзя — размер уменьшен с теми же пропорциями", 3500); }
  }
  const sn = S.family === "ltx" && $("#ltxTwo").checked ? (v => Math.max(256, Math.round(v / 64) * 64)) : snap;
  if (final) { $("#width").value = sn(w); $("#height").value = sn(h); }
  else $(isW ? "#height" : "#width").value = snap(isW ? h : w);
}
["#width", "#height"].forEach(s => {
  $(s).addEventListener("input", () => { if (S.lockAR) fitOther(s, false); });
  $(s).addEventListener("change", () => {
    if (S.lockAR) fitOther(s, true);
    else {
      $(s).value = snap(+$(s).value);
      if (S.family === "minimax" && +$("#width").value * +$("#height").value > 768 * 1344 * 1.001) { const [w, h] = h3Canvas(+$("#width").value, +$("#height").value); $("#width").value = w; $("#height").value = h; }
      S.ratio = [+$("#width").value, +$("#height").value];
    }
    updateMp();
  });
});
$("#fromImage").onclick = () => {
  const im = S.images[0]; if (!im || !im.w) return;
  const mp = +$("#mpSel").value * 1048576, k = Math.sqrt(mp / (im.w * im.h));
  S.ratio = [im.w, im.h]; $("#width").value = snap(im.w * k); $("#height").value = snap(im.h * k); updateMp();
};

/* ---------------- sampling ---------------- */
const PRESETS = {
  minimax: [["⚡ Турбо: 8 шагов", { steps: 8, turbo: true }], ["Качество: 20 шагов", { steps: 20, turbo: false }], ["Детали: 30 шагов", { steps: 30, turbo: false }]],
  ltx: [["Быстро: без апскейла", { ltxTwo: false }], ["Качество: апскейл ×2", { ltxTwo: true }], ["Свободнее от кадра", { ltxSrc: 0.55, ltxAnchor: 0.6 }], ["Точно как исходник", { ltxSrc: 1, ltxAnchor: 1 }]],
  flux3: [["Distilled: 16 шагов", { steps: 16, cfg: 1, sampler: "er_sde", scheduler: "simple" }], ["Быстро: 8 шагов", { steps: 8, cfg: 1, sampler: "er_sde", scheduler: "simple" }], ["Base: 50 шагов, CFG 3.5", { steps: 50, cfg: 3.5, sampler: "er_sde", scheduler: "simple" }]],
  krea2: [["Turbo: 8 шагов", { steps: 8, cfg: 1, scheduler: "simple", sampler: "euler" }], ["Raw: 52 шага, CFG 3.5", { steps: 52, cfg: 3.5, scheduler: "simple", sampler: "euler" }], ["Raw быстро: 28 шагов", { steps: 28, cfg: 3.5, scheduler: "simple", sampler: "euler" }], ["⚡ Raw + турбо‑LoRA", "kreaTurbo"]],
  qwen: [["⚡ Турбо: 8 шагов", "qwenTurbo"], ["Официально: 40 шагов", { steps: 40, cfg: 1, scheduler: "simple", sampler: "euler" }], ["Быстро: 25 шагов", { steps: 25, cfg: 1, scheduler: "simple", sampler: "euler" }]],
};
function renderPresets() {
  $("#presets").innerHTML = PRESETS[S.family].map((p, i) => `<button class="small" data-preset="${i}">${p[0]}</button>`).join("");
  $$("#presets [data-preset]").forEach(b => b.onclick = () => {
    const v = PRESETS[S.family][+b.dataset.preset][1];
    if (v === "qwenTurbo") return qwenTurboPreset();
    if (v === "kreaTurbo") return kreaTurboPreset();
    for (const k in v) { const el = $("#" + k); if (el.type === "checkbox") el.checked = !!v[k]; else el.value = v[k]; }
    if (S.family === "qwen") S.loras = S.loras.filter(l => !isQwenTurboLora(l.name));
    if (S.family === "krea2" && +v.steps > 14) S.loras = S.loras.filter(l => !isKreaTurboLora(l.name));
    ltxShow(); syncTurbo(); renderLoras(); checkPair(); saveUI();
  });
}
const isKreaTurboLora = n => /turbo/i.test(n || "") && /krea/i.test(n || "");
const kreaTurboLora = () => S.family === "krea2" && S.loras.some(l => l.enabled && isKreaTurboLora(l.name));
function kreaTurboPreset() {
  $("#steps").value = 8; $("#cfg").value = 1; $("#sampler").value = "euler"; $("#scheduler").value = "simple";
  const tl = (S.models.loras || []).find(x => isKreaTurboLora(x.name));
  if (tl) { if (!S.loras.some(l => l.name === tl.name)) S.loras.unshift({ name: tl.name, strength: 1, enabled: true }); toast("Турбо‑LoRA Krea добавлена в стек: 8 шагов, CFG 1", 3500); }
  else { toast("Турбо‑LoRA Krea не скачана — открываю LoRA → ★ Официальные", 5000); openLoraDlg("off"); }
  renderLoras(); checkPair(); saveUI();
}
function qwenTurboPreset() {
  $("#steps").value = 8; $("#cfg").value = 1; $("#sampler").value = "euler"; $("#scheduler").value = "simple";
  if (!/turbo/i.test($("#unet").value || "")) {
    const tm = (S.models.diffusion_models || []).find(x => /turbo/i.test(x.name) && fileFamily("diffusion_models", x.name) === "qwen");
    const tl = (S.models.loras || []).find(x => isQwenTurboLora(x.name));
    if (tl) { if (!S.loras.some(l => l.name === tl.name)) S.loras.unshift({ name: tl.name, strength: 1, enabled: true }); toast("Турбо‑LoRA Qwen добавлена в стек: 8 шагов, CFG 1", 3500); }
    else if (tm) { $("#unet").value = tm.name; toast("Выбрана модель Qwen‑Image 2.1 Turbo: 8 шагов, CFG 1", 3500); }
    else { toast("Нет ни модели Qwen Turbo, ни турбо‑LoRA — открываю LoRA → ★ Официальные (или скачайте турбо‑модель в «Модели»)", 6000); openLoraDlg("off"); }
  }
  renderLoras(); checkPair(); saveUI();
}
$("#seedRand").onclick = () => { $("#seed").value = -1; saveUI(); };
$("#seedLast").onclick = () => { if (S.lastSeed != null) { $("#seed").value = S.lastSeed; saveUI(); } else toast("Ещё не было генераций"); };
FIELDS.forEach(f => { const el = $("#" + f); el && el.addEventListener("change", saveUI); });
$("#prompt").addEventListener("input", saveUI);

/* ---------------- LoRA: стек карточек ---------------- */
S.lib = {}; S.libList = null;
const blurOn = () => $("#civBlur").checked;
async function loadLoraLib() {
  try { const r = await api("/api/loras"); S.libList = r.items; S.lib = Object.fromEntries(r.items.map(x => [x.file, x])); } catch { }
  renderLoras();
  if ($("#loraDlg").open && lTab === "mine") renderMine();
  if ($("#loraDlg").open && lTab === "off") renderOfficial();
}
const stripExt = f => f.replace(/^.*\//, "").replace(/\.[^.]+$/, "");
const loraTitle = (info, file) => (info && info.title) || stripExt(file);
function fullTitle(info, file) {
  const p = [loraTitle(info, file)];
  if (info?.civitai_name && info.title !== info.civitai_name) p.push("Название на Civitai: " + info.civitai_name);
  if (info?.version_name) p.push("Версия: " + info.version_name);
  if (info?.base_model) p.push("База: " + info.base_model);
  if (info?.trained_words?.length) p.push("Триггеры: " + info.trained_words.join(", "));
  if (info?.note) p.push("Заметка: " + info.note);
  if (info?.digest?.summary) p.push("Кратко: " + info.digest.summary);
  if (recStr(info)) p.push("Рекомендуемая сила: " + fmtRng(recStr(info)));
  p.push("Файл: " + file);
  return p.join("\n");
}
function picHtml(src, type, fallbackName) {
  if (src) return type === "video" ? `<video src="${esc(src)}" muted loop autoplay playsinline></video>` : `<img src="${esc(src)}" loading="lazy" alt="">`;
  return `<div class="ph"><div><b>${esc(fallbackName.slice(0, 2).toUpperCase())}</b>${esc(fallbackName)}<br><br>нет описания — ℹ → «Найти на Civitai»</div></div>`;
}
const siteName = () => S.civDomain || "civitai.red";
/* рекомендуемая сила из выжимки: {min,max,rec} или null */
function recStr(info) { const st = info?.digest?.strength; return st && st.min !== undefined && st.min !== null ? st : null; }
const fmtRng = st => st.min === st.max ? `${+st.min}` : `${+st.min}–${+st.max}`;
const outOfRng = (v, st) => st && (v < st.min - 0.05 || v > st.max + 0.05);
function archBad(info) {
  if (!info || !info.arch || info.arch === "unknown") return false;
  return info.arch !== ARCH_OF[S.family];
}

/* турбо‑LoRA MiniMax H3: галочка «⚡ Турбо» сама кладёт нужную LoRA в стек (кадры — fl2v, референсы — ref2v) */
const isTurboLora = n => /turbo/i.test(n || "") && /h3|minimax/i.test(n || "");
const isRefTurbo = n => /ref2?v/i.test(n || "");
function bestTurbo() {
  const ref = S.mm === "ref";
  const names = (S.models.loras || []).map(x => x.name).filter(n => isTurboLora(n) && isRefTurbo(n) === ref);
  names.sort((a, b) => (/8step/i.test(a) ? 0 : 1) - (/8step/i.test(b) ? 0 : 1) || a.localeCompare(b));
  return names[0] || null;
}
function syncTurbo(render = true) {
  if (S.family !== "minimax") return;
  const on = $("#turbo").checked, ref = S.mm === "ref";
  const have = S.loras.filter(l => isTurboLora(l.name));
  const ok = have.find(l => isRefTurbo(l.name) === ref && (S.models.loras || []).some(x => x.name === l.name));
  const want = on ? (ok ? ok.name : bestTurbo()) : null;
  S.loras = S.loras.filter(l => !isTurboLora(l.name) || l.name === want);
  if (want && !S.loras.some(l => l.name === want)) S.loras.unshift({ name: want, strength: 1, enabled: true });
  const t = S.loras.find(l => l.name === want); if (t) t.enabled = true;
  S.turboMissing = on && !want;
  if (want) { const n = /(\d+)\s*step/i.exec(want); if (n) $("#steps").value = n[1]; }
  if (render) { renderLoras(); saveUI(); }
}
function renderLoras() {
  const list = $("#loraList");
  list.innerHTML = S.loras.map((l, i) => {
    const info = S.lib[l.name], missing = S.libList && !info;
    return `<div class="lc ${l.enabled ? "" : "off"} ${info?.nsfw && blurOn() ? "nsfw blur" : ""}" draggable="true" data-i="${i}">
      <div class="ttl" title="${esc(fullTitle(info, l.name))}">${S.family === "minimax" && isTurboLora(l.name) ? `<span class="turboB" title="Турбо‑LoRA: стоит в стеке, пока включена галочка «⚡ Турбо». Шаги ставятся сами (${/(\d+)\s*step/i.exec(l.name)?.[1] || 8})">⚡ турбо</span> ` : ""}${esc(loraTitle(info, l.name))}</div>
      <div class="pic">${picHtml(info?.preview, info?.preview_type, stripExt(l.name))}
        <div class="ov tl"><span class="num" title="Порядок применения">#${i + 1}</span><button data-on="${i}" title="${l.enabled ? "Выключить (не удаляя)" : "Включить"}">${l.enabled ? "вкл" : "выкл"}</button></div>
        <div class="ov tr"><button data-del="${i}" title="Убрать из стека">✕</button></div>
        <div class="ov bl">${info?.trained_words?.length ? `<button data-tw="${i}" title="Вставить триггер-слова в промпт:\n${esc(info.trained_words.join(", "))}">T+</button>` : ""}<button data-inf="${esc(l.name)}" title="Описание и настройки">ℹ</button></div>
        <div class="ov br">${info?.page ? `<a href="${esc(info.page)}" target="_blank" rel="noopener" title="Открыть страницу LoRA на ${esc(siteName())}">Civitai ↗</a>` : ""}</div>
        ${missing ? `<div class="missing">файл не найден в models/loras</div>` : info?.arch === "checkpoint" ? `<div class="missing ckpt" data-tomodel="${esc(l.name)}" title="Это целая дообученная модель, а не LoRA: как LoRA она не работает. Нажмите — перенести в модели и выбрать её">⚠ целая модель, не LoRA — <u>перенести в модели</u></div>` : archBad(info) ? `<div class="missing" title="Слои этой LoRA не подойдут к выбранной нейросети — она не подействует">⚠ LoRA для ${esc(info.arch_label)}</div>` : ""}
      </div>
      <div class="str" title="Сила влияния LoRA"><input type="range" min="-2" max="2" step="0.05" value="${l.strength}" data-sr="${i}"><input type="number" min="-5" max="5" step="0.05" value="${l.strength}" data-sn="${i}"></div>
      ${recStr(info) ? `<div class="rec ${outOfRng(+l.strength, recStr(info)) ? "bad" : ""}" data-rec="${i}" title="Рекомендуемая сила (${esc(recStr(info).source || "")}). Нажмите — поставить ${+recStr(info).rec}">реком. ${fmtRng(recStr(info))}</div>` : ""}
    </div>`;
  }).join("") + (S.family === "minimax" && S.turboMissing ? `<div class="lc turboMiss" id="turboMiss" title="Открыть LoRA → ★ Официальные и скачать">⚡ Турбо включено, но турбо‑LoRA ${S.mm === "ref" ? "для референсов (ref2v, 4 шага)" : "для кадров (fl2v, 8 шагов)"} не скачана — <u>скачать в LoRA → ★ Официальные</u></div>` : "") + `<div class="lc add" id="loraAddTile" title="Выбрать из своих или скачать с Civitai"><b>+</b>Добавить LoRA<span class="tiny muted">мои · Civitai</span></div>`;
  $("#loraCount").textContent = S.loras.length ? `${S.loras.filter(l => l.enabled).length} активн. из ${S.loras.length}` : "";
  $("#loraAddTile").onclick = () => openLoraDlg();
  const tm = $("#turboMiss"); if (tm) tm.onclick = () => openLoraDlg("off");
  const turboOff = l => { if (S.family === "minimax" && isTurboLora(l.name) && $("#turbo").checked) { $("#turbo").checked = false; S.loras = S.loras.filter(x => x !== l); toast("Турбо выключено — без турбо‑LoRA нужно ~20 шагов (поставил 20)", 3500); $("#steps").value = 20; syncTurbo(); checkPair(); return true; } return false; };
  $$("[data-on]", list).forEach(e => e.onclick = () => { const l = S.loras[+e.dataset.on]; if (turboOff(l)) return; l.enabled = !l.enabled; renderLoras(); saveUI(); });
  $$("[data-del]", list).forEach(e => e.onclick = () => { if (turboOff(S.loras[+e.dataset.del])) return; S.loras.splice(+e.dataset.del, 1); renderLoras(); saveUI(); });
  $$("[data-tw]", list).forEach(e => e.onclick = () => insertWords(S.lib[S.loras[+e.dataset.tw].name].trained_words));
  $$("[data-inf]", list).forEach(e => e.onclick = () => openLocalInfo(e.dataset.inf));
  $$("[data-tomodel]", list).forEach(e => e.onclick = () => loraToModel(e.dataset.tomodel));
  const recMark = i => { const r = $(`[data-rec="${i}"]`, list), st = recStr(S.lib[S.loras[i].name]); if (r && st) r.classList.toggle("bad", outOfRng(+S.loras[i].strength, st)); };
  $$("[data-sr]", list).forEach(e => e.oninput = () => { const i = +e.dataset.sr; S.loras[i].strength = +e.value; $(`[data-sn="${i}"]`).value = e.value; recMark(i); saveUI(); });
  $$("[data-sn]", list).forEach(e => e.onchange = () => { const i = +e.dataset.sn; S.loras[i].strength = +e.value; $(`[data-sr="${i}"]`).value = e.value; recMark(i); saveUI(); renderTips(); });
  $$("[data-sr]", list).forEach(e => e.onchange = () => renderTips());
  $$("[data-rec]", list).forEach(e => e.onclick = () => { const i = +e.dataset.rec, st = recStr(S.lib[S.loras[i].name]); S.loras[i].strength = +st.rec; renderLoras(); saveUI(); toast("Сила: " + st.rec); });
  let from = null;
  $$(".lc[data-i]", list).forEach(el => {
    el.ondragstart = e => { from = +el.dataset.i; el.classList.add("drag"); e.dataTransfer.effectAllowed = "move"; };
    el.ondragend = () => { el.classList.remove("drag"); $$(".dropto", list).forEach(x => x.classList.remove("dropto")); };
    el.ondragover = e => { e.preventDefault(); el.classList.add("dropto"); };
    el.ondragleave = () => el.classList.remove("dropto");
    el.ondrop = e => { e.preventDefault(); const to = +el.dataset.i; if (from === null || from === to) return; const [x] = S.loras.splice(from, 1); S.loras.splice(to, 0, x); renderLoras(); saveUI(); };
  });
}
function insertWords(words) {
  const p = $("#prompt"); let t = p.value.trim();
  const add = (words || []).map(w => w.trim()).filter(w => w && !t.toLowerCase().includes(w.toLowerCase()));
  if (!add.length) return toast("Триггер-слова уже есть в промпте");
  p.value = (t ? t + ", " : "") + add.join(", "); saveUI(); toast("Добавлено в промпт: " + add.join(", "));
}
function addToStack(file) {
  if (S.loras.some(l => l.name === file)) return toast("Эта LoRA уже в стеке");
  const d = S.lib[file]?.default_strength, st = recStr(S.lib[file]);
  S.loras.push({ name: file, strength: d !== undefined && d !== null && d !== "" ? +d : st ? +st.rec : 1, enabled: true });
  renderLoras(); saveUI(); toast("Добавлено: " + loraTitle(S.lib[file], file));
  if ($("#loraDlg").open && lTab === "mine") renderMine();
}

/* ---------------- LoRA: библиотека и Civitai ---------------- */
let lTab = "mine", civCursor = null, identTimer = null;
function openLoraDlg(tab) {
  if (tab) lTab = tab;
  openDlg("loraDlg"); showLTab(); pollTransfers();
  api("/api/settings").then(s => { S.civDomain = s.civitai_domain; $("#civKeyNote").textContent = s.civitai_token ? "" : "⚠ API‑ключ Civitai не указан — большинство LoRA без него не скачаются (Настройки)."; }).catch(() => { });
}
$$("#lTabs button").forEach(b => b.onclick = () => { lTab = b.dataset.ltab; showLTab(); });
function showLTab() {
  $$("#lTabs button").forEach(b => b.classList.toggle("on", b.dataset.ltab === lTab));
  $("#lMine").classList.toggle("hidden", lTab !== "mine"); $("#lCiv").classList.toggle("hidden", lTab !== "civ"); $("#lOff").classList.toggle("hidden", lTab !== "off");
  if (lTab === "mine") loadLoraLib();
  else if (lTab === "off") renderOfficial();
  else if (!$("#civGrid").children.length) civSearch();
}
/* официальные LoRA (из каталога, папка loras) — по нейросетям, текущая первой */
const FAM_ORDER = ["qwen", "krea2", "flux3", "minimax", "ltx"];
async function renderOfficial() {
  let c; try { c = await api("/api/catalog"); } catch (e) { $("#offList").innerHTML = `<p class="error">${esc(e.message)}</p>`; return; }
  const items = c.items.filter(x => x.folder === "loras");
  const fams = [S.family, ...FAM_ORDER.filter(f => f !== S.family)].filter(f => items.some(x => x.family === f));
  let html = "";
  for (const f of fams) {
    html += `<div class="catGroup">${esc(FAM[f]?.name || f)}${f === S.family ? " · текущая" : ""}</div>`;
    let g = "";
    for (const it of items.filter(x => x.family === f)) {
      const sub = it.group.replace(/^.*?·\s*/, "");
      if (sub !== g && items.filter(x => x.family === f).some(x => x.group !== it.group)) { g = sub; html += `<div class="muted tiny" style="margin:8px 0 4px">${esc(sub)}</div>`; }
      const inStack = S.loras.some(l => l.name === it.filename);
      html += `<div class="cat"><div class="info"><div class="t">${esc(it.title)}${it.recommended ? `<span class="badge rec">Рекомендуемое</span>` : ""}${it.installed ? '<span class="badge ok">✓ есть</span>' : ""}</div>
        <div class="d">${esc(it.desc)} · <b>${esc(it.size)}</b> · <a href="${esc(it.page)}" target="_blank">страница</a></div></div>
        ${it.installed ? (f === S.family ? `<button class="small ${inStack ? "" : "primary"}" data-offadd="${esc(it.filename)}" ${inStack ? "disabled" : ""}>${inStack ? "В стеке" : "В стек"}</button>` : "")
          : `<button class="small ${it.recommended ? "primary" : ""}" data-dl="${it.id}">${it.partial ? "Докачать" : "Скачать"}</button>`}</div>`;
    }
  }
  $("#offList").innerHTML = html || `<p class="muted">Официальных LoRA в каталоге нет.</p>`;
  $$("#offList [data-dl]").forEach(b => b.onclick = async () => {
    try { await api("/api/download", { id: b.dataset.dl }); b.disabled = true; b.textContent = "В очереди"; pollTransfers(); } catch (e) { toast(e.message, 4000); }
  });
  $$("#offList [data-offadd]").forEach(b => b.onclick = () => {
    const n = b.dataset.offadd;
    if (!S.loras.some(l => l.name === n)) S.loras.push({ name: n, strength: 1, enabled: true });
    if (S.family === "qwen") qwenTurboSteps();
    if (S.family === "krea2" && isKreaTurboLora(n)) { $("#steps").value = 8; $("#cfg").value = 1; }
    syncTurbo(false); renderLoras(); checkPair(); saveUI(); renderOfficial(); toast("Добавлено в стек");
  });
}
/* LoRA этой нейросети: по слоям в самом файле (неопределённые показываем везде) */
const loraFits = x => !x.arch || x.arch === "unknown" || x.arch === "checkpoint" || x.arch === ARCH_OF[S.family];
function renderMine() {
  const q = $("#mineFilter").value.trim().toLowerCase();
  const all = $("#mineAll").checked;
  const found = (S.libList || []).filter(x => !q || [x.title, x.civitai_name, x.file, x.note, (x.trained_words || []).join(" ")].join(" ").toLowerCase().includes(q));
  const items = all ? found : found.filter(loraFits);
  const hidden = found.length - items.length;
  $("#mineAllN").textContent = hidden || all ? `(${all ? found.filter(x => !loraFits(x)).length : hidden})` : "";
  $("#mineAll").closest("label").classList.toggle("hidden", !hidden && !all);
  const inStack = new Set(S.loras.map(l => l.name));
  $("#mineGrid").innerHTML = items.length ? items.map(x => `
    <div class="lc pick ${inStack.has(x.file) ? "instack" : ""} ${x.nsfw && blurOn() ? "nsfw blur" : ""}" data-file="${esc(x.file)}">
      <div class="ttl" title="${esc(fullTitle(x, x.file))}">${esc(loraTitle(x, x.file))}</div>
      <div class="pic">${picHtml(x.preview, x.preview_type, stripExt(x.file))}
        <div class="ov tl">${inStack.has(x.file) ? `<span class="num">в стеке</span>` : ""}</div>
        <div class="ov bl"><button data-inf="${esc(x.file)}" title="Описание и настройки">ℹ</button></div>
        <div class="ov br">${x.page ? `<a href="${esc(x.page)}" target="_blank" rel="noopener">Civitai ↗</a>` : ""}</div>
      </div>
      <div class="info2"><span class="bm" title="${esc(x.file)}">${esc(x.arch === "checkpoint" ? "целая модель, не LoRA" : x.arch_label && x.arch !== "unknown" ? "для " + x.arch_label : (x.base_model || "база неизвестна"))} · ${esc(x.size_h)}</span>${x.arch === "checkpoint" ? `<button class="small" data-tomodel="${esc(x.file)}">Перенести в модели</button>` : archBad(x) ? `<span class="bad">⚠ не для ${esc(FAM[S.family].name)}</span>` : ""}</div>
    </div>`).join("") : `<p class="muted">${S.libList && S.libList.length ? (hidden ? `LoRA для ${esc(FAM[S.family].name)} пока нет (скрыто LoRA других моделей: ${hidden}). Скачайте на вкладке Civitai.` : "Ничего не найдено") : "В папке models/loras пока нет LoRA. Скачайте на вкладке Civitai или добавьте свои файлы."}</p>`;
  $$("#mineGrid .lc").forEach(c => c.onclick = e => {
    if (e.target.closest("a")) return;
    const inf = e.target.closest("[data-inf]"); if (inf) return openLocalInfo(inf.dataset.inf);
    const tm = e.target.closest("[data-tomodel]"); if (tm) return loraToModel(tm.dataset.tomodel);
    addToStack(c.dataset.file);
  });
}
$("#mineFilter").oninput = renderMine;
$("#mineAll").onchange = renderMine;
$("#loraUploadBtn").onclick = () => $("#loraUpload").click();
$("#loraUpload").onchange = async e => { await uploadModels(e.target.files, "loras"); e.target.value = ""; loadLoraLib(); };
$("#identifyAll").onclick = () => startIdentify({});
$("#digestAll").onclick = () => {
  const files = (S.libList || []).filter(x => loraFits(x) && x.needs_digest).map(x => x.file);
  if (!files.length) return toast("У всех LoRA с описанием уже есть выжимка");
  digestRun(files, null);
};
async function startIdentify(body) {
  try { const r = await api("/api/loras/identify", body); if (!r.total) return toast("У всех LoRA уже есть описания"); }
  catch (e) { return toast(e.message, 4000); }
  const box = $("#identProg"); box.classList.remove("hidden");
  const tick = async () => {
    const st = await api("/api/loras/identify").catch(() => null); if (!st) return;
    box.textContent = st.running ? `Считаю хэши и ищу на Civitai: ${st.done}/${st.total} ${st.current ? "· " + st.current : ""}`
      : `Готово: найдено ${st.found} из ${st.total}.` + (st.errors.length ? "\n" + st.errors.join("\n") : "");
    if (st.running) identTimer = setTimeout(tick, 1000); else { loadLoraLib(); if ($("#loraInfoDlg").open && S.infoFile) openLocalInfo(S.infoFile); }
  };
  clearTimeout(identTimer); tick();
}

/* ---- Civitai поиск ---- */
function civCard(c) {
  const img = c.image;
  const nsfw = c.nsfw || (img && img.nsfw_level >= 4);
  const vers = c.versions && c.versions.length > 1 ? `<select data-ver>${c.versions.map(v => `<option value="${v.id}">${esc(v.name)} · ${esc(v.base_model)}${v.installed ? " ✓" : ""}</option>`).join("")}</select>` : "";
  return `<div class="lc ${nsfw && blurOn() ? "nsfw blur" : ""}" data-mid="${c.model_id}" data-vid="${c.version_id}">
    <div class="ttl" title="${esc(c.name + (c.version_name ? "\nВерсия: " + c.version_name : "") + "\nАвтор: " + (c.creator || "—"))}">${esc(c.name)}</div>
    <div class="pic">${img ? picHtml(img.type === "video" ? img.url : img.thumb, img.type, "") : picHtml("", "", c.name)}
      <div class="ov bl"><button data-rinf title="Описание">ℹ</button></div>
      <div class="ov br"><a href="${esc(c.page)}" target="_blank" rel="noopener" title="Открыть на ${esc(siteName())}">Civitai ↗</a></div>
    </div>
    <div class="info2"><span class="bm" title="${esc(c.base_model)}">${esc(c.base_model)} · ⬇ ${fmtNum(c.downloads)} · 👍 ${fmtNum(c.likes)}</span>${vers}
      <button class="small ${c.installed ? "" : "primary"}" data-get>${c.installed ? "✓ Есть · в стек" : "⬇ Скачать"}</button></div>
  </div>`;
}
const fmtNum = n => (n || 0) >= 1000 ? ((n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k") : String(n || 0);
function bindCivCards(root, cards) {
  $$(".lc[data-mid]", root).forEach((el, i) => {
    const c = cards[i]; el._card = c;
    const sel = $("[data-ver]", el);
    const syncBtn = () => {
      const v = sel ? c.versions.find(v => v.id == sel.value) : { installed: c.installed };
      const b = $("[data-get]", el); b.textContent = v.installed ? "✓ Есть · в стек" : "⬇ Скачать"; b.classList.toggle("primary", !v.installed); b.disabled = false;
    };
    if (sel) sel.onchange = syncBtn;
    $("[data-rinf]", el).onclick = () => openRemoteInfo(c.page);
    $("[data-get]", el).onclick = async () => {
      const b = $("[data-get]", el), vid = sel ? +sel.value : c.version_id;
      if (b.textContent.startsWith("✓")) {
        const file = b.dataset.file || Object.values(S.lib).find(x => x.page && x.page.includes("modelVersionId=" + vid))?.file;
        if (file) addToStack(file); else { await loadLoraLib(); toast("Файл есть — найдите его во вкладке «Мои LoRA»"); }
        return;
      }
      b.disabled = true; b.textContent = "Скачивается…";
      try {
        const t = await api("/api/civitai/download", { model_id: c.model_id, version_id: vid });
        b.dataset.file = t.filename; S.pendingCards[t.id] = { el, file: t.filename, vid };
        if (sel) c.versions.find(v => v.id == vid).pending = true;
        pollTransfers();
      } catch (e) { b.disabled = false; b.textContent = "⬇ Скачать"; toast(e.message, 5000); }
    };
  });
}
S.pendingCards = {};
async function civSearch(more) {
  const grid = $("#civGrid");
  if (!more) { civCursor = null; grid.innerHTML = `<p class="muted">Загрузка…</p>`; }
  const q = new URLSearchParams({ q: $("#civQ").value.trim(), base: $("#civBase").value, sort: $("#civSort").value, period: $("#civPeriod").value, nsfw: $("#civNsfw").checked ? "1" : "0" });
  if (more && civCursor) q.set("cursor", civCursor);
  let r;
  try { r = await api("/api/civitai/search?" + q); }
  catch (e) { grid.innerHTML = `<div class="error">${esc(e.message)}</div>`; return; }
  S.civDomain = r.domain || S.civDomain;
  if (!more) grid.innerHTML = "";
  const wrap = document.createElement("div"); wrap.innerHTML = r.items.map(civCard).join("");
  const nodes = [...wrap.children]; nodes.forEach(n => grid.appendChild(n));
  bindCivCards({ querySelectorAll: s => nodes.filter(n => n.matches(s)) }, r.items);
  if (!more && !r.items.length) grid.innerHTML = `<p class="muted">Ничего не найдено. Попробуйте «Любая база».</p>`;
  civCursor = r.cursor; $("#civMore").classList.toggle("hidden", !civCursor);
}
$("#civSearch").onclick = () => civSearch();
$("#civQ").addEventListener("keydown", e => { if (e.key === "Enter") civSearch(); });
$("#civMore").onclick = () => civSearch(true);
["#civBase", "#civSort", "#civPeriod", "#civNsfw"].forEach(s => $(s).addEventListener("change", () => civSearch()));
$("#civBlur").addEventListener("change", () => { $$("#civGrid .lc").forEach(el => { const c = el._card; if (c) el.classList.toggle("blur", blurOn()), el.classList.toggle("nsfw", !!(c.nsfw || (c.image && c.image.nsfw_level >= 4))); }); renderLoras(); if (lTab === "mine") renderMine(); saveUI(); });
$("#civLinkBtn").onclick = async () => {
  const url = $("#civLink").value.trim(); if (!url) return;
  const grid = $("#civGrid"); grid.innerHTML = `<p class="muted">Загрузка…</p>`;
  try {
    const c = await api("/api/civitai/resolve?url=" + encodeURIComponent(url));
    grid.innerHTML = civCard(c); bindCivCards(grid, [c]); $("#civMore").classList.add("hidden");
    const sel = $("[data-ver]", grid); if (sel) { sel.value = c.version_id; sel.onchange(); }
  } catch (e) { grid.innerHTML = `<div class="error">${esc(e.message)}</div>`; }
};
$("#civLink").addEventListener("keydown", e => { if (e.key === "Enter") $("#civLinkBtn").click(); });

/* ---- окно описания ---- */
function sanitize(html) {
  const doc = new DOMParser().parseFromString(html || "", "text/html");
  const ok = new Set(["P", "BR", "B", "STRONG", "I", "EM", "U", "S", "UL", "OL", "LI", "A", "H1", "H2", "H3", "H4", "BLOCKQUOTE", "CODE", "PRE", "IMG", "SPAN", "DIV", "HR"]);
  [...doc.body.querySelectorAll("*")].reverse().forEach(el => {
    if (!ok.has(el.tagName)) { if (/^(SCRIPT|STYLE|IFRAME|OBJECT|EMBED|FORM|INPUT|BUTTON)$/.test(el.tagName)) el.remove(); else el.replaceWith(...el.childNodes); return; }
    [...el.attributes].forEach(a => { const k = a.name.toLowerCase(); if (!((el.tagName === "A" && k === "href") || (el.tagName === "IMG" && k === "src"))) el.removeAttribute(a.name); });
    if (el.tagName === "A") { if (!/^https?:/i.test(el.getAttribute("href") || "")) el.removeAttribute("href"); el.target = "_blank"; el.rel = "noopener noreferrer"; }
    if (el.tagName === "IMG" && !/^https:/i.test(el.getAttribute("src") || "")) el.remove();
  });
  return doc.body.innerHTML.trim();
}
function fillInfo({ title, sub, img, imgType, links, words, desc, local }) {
  $("#liTitle").textContent = title;
  $("#liSub").innerHTML = sub;
  $("#liImg").innerHTML = img ? (imgType === "video" ? `<video src="${esc(img)}" controls muted loop autoplay></video>` : `<img src="${esc(img)}" alt="">`) : `<div class="ph">нет картинки</div>`;
  $("#liLinks").innerHTML = links.map(([u, t]) => `<a class="linkbtn" href="${esc(u)}" target="_blank" rel="noopener">${esc(t)} ↗</a>`).join("");
  $("#liWordsBox").classList.toggle("hidden", !words.length);
  $("#liWords").innerHTML = words.map(w => `<button class="chip w">${esc(w)}</button>`).join("");
  $$("#liWords .w").forEach(b => b.onclick = () => insertWords([b.textContent]));
  $("#liDesc").innerHTML = sanitize(desc) || `<span class="muted">Описания нет</span>`;
  $$("#liCustomTitle,#liNote,#liStrength,#liSave,#liIdentify,#liPreviewFrom,#liDelete,#liTriggers,#liTrigSrc,#liFileTags").forEach(e => (e.closest("label") || e).classList.toggle("hidden", !local));
  $("#liDigestBox").classList.toggle("hidden", !local);
  $$("#loraInfoDlg h4").forEach(h => { if (/^(Своё|Триггер-слова для)/.test(h.textContent)) h.classList.toggle("hidden", !local); });
  openDlg("loraInfoDlg");
}
async function openLocalInfo(file) {
  let m; try { m = await api("/api/loras/info?file=" + encodeURIComponent(file)); } catch (e) { return toast(e.message); }
  S.infoFile = file;
  const x = S.lib[file] || {};
  const links = [];
  if (m.page) links.push([m.page, "civitai.red"]);
  if (m.page_com) links.push([m.page_com, "civitai.com"]);
  fillInfo({
    title: m.title || m.name || stripExt(file),
    sub: [m.title && m.name && m.title !== m.name ? "На Civitai: <b>" + esc(m.name) + "</b>" : "", m.version_name ? "Версия: " + esc(m.version_name) : "", m.base_model ? "База: " + esc(m.base_model) : "", m.creator ? "Автор: " + esc(m.creator) : "", "Файл: " + esc(file), m.source ? "" : "<br><b>Описание не найдено.</b> Нажмите «Найти на Civitai по хэшу» — если файл оттуда, подтянется название, картинка и описание."].filter(Boolean).join(" · "),
    img: x.preview, imgType: x.preview_type, links, words: m.trained_words || [],
    desc: (m.description || "") + (m.version_description ? "<hr><h3>О версии</h3>" + m.version_description : ""), local: true,
  });
  $("#liCustomTitle").value = m.title || ""; $("#liNote").value = m.note || ""; $("#liStrength").value = m.strength ?? "";
  $("#liTriggers").value = (m.triggers || []).join(", ");
  $("#liTrigSrc").innerHTML = { manual: "Заданы вами. <button class='link' id='liTrigReset'>вернуть исходные</button>", civitai: "Со страницы Civitai.", digest: "Помощник нашёл их в тексте описания — проверьте.", file: "Угаданы по тегам обучения в самом файле — проверьте.", "": "Неизвестны: впишите сами или нажмите «Найти на Civitai»." }[m.triggers_source || ""];
  fillDigest(m.digest || {});
  const rs = $("#liTrigReset"); if (rs) rs.onclick = async () => { await api("/api/loras/edit", { file: file, trained_words: null }); await loadLoraLib(); openLocalInfo(file); };
  $("#liFileTags").innerHTML = (m.file_tags || []).length ? `<span class="muted tiny">Частые теги из обучения (клик — добавить): </span>` + m.file_tags.map(t => `<button class="chip w" data-ft>${esc(t)}</button>`).join("") : "";
  $$("[data-ft]", $("#liFileTags")).forEach(b => b.onclick = () => { const cur = $("#liTriggers").value.split(",").map(x => x.trim()).filter(Boolean); if (!cur.includes(b.textContent)) cur.push(b.textContent); $("#liTriggers").value = cur.join(", "); });
}
async function openRemoteInfo(url) {
  let c; try { c = await api("/api/civitai/resolve?url=" + encodeURIComponent(url)); } catch (e) { return toast(e.message, 4000); }
  S.infoFile = null;
  fillInfo({
    title: c.name, sub: [c.version_name && "Версия: " + esc(c.version_name), c.base_model && "База: " + esc(c.base_model), c.creator && "Автор: " + esc(c.creator), c.file && c.file.size_kb ? "Размер: " + fmtBytes(c.file.size_kb * 1024) : ""].filter(Boolean).join(" · "),
    img: c.image && (c.image.type === "video" ? c.image.url : c.image.thumb), imgType: c.image && c.image.type,
    links: [[c.page, siteName()]], words: c.trained_words || [], desc: c.description, local: false,
  });
}
function fillDigest(d) {
  const st = d.strength || {};
  $("#liDigestSrc").textContent = { llm: `· разобрано помощником${d.time ? " " + d.time : ""}`, user: "· исправлено вами", auto: "· найдено без помощника (только сила и триггеры)" }[d.made_by] || "";
  $("#liDgSummary").value = d.summary || ""; $("#liDgRules").value = (d.rules || []).join("\n"); $("#liDgSettings").value = d.settings || "";
  $("#liDgMin").value = st.min ?? ""; $("#liDgMax").value = st.max ?? ""; $("#liDgRec").value = st.rec ?? "";
}
$("#liDgRun").onclick = async () => {
  const b = $("#liDgRun"); b.disabled = true; b.textContent = "Помощник читает описание…";
  try { const r = await api("/api/loras/digest", { file: S.infoFile }); fillDigest(r.digest); if (r.note) toast(r.note, 6000); await loadLoraLib(); }
  catch (e) { toast(e.message, 6000); }
  finally { b.disabled = false; b.textContent = "🧠 Разобрать описание"; }
};
$("#liDgSave").onclick = async () => {
  const n = id => $(id).value === "" ? null : +$(id).value;
  let mn = n("#liDgMin"), mx = n("#liDgMax"), rc = n("#liDgRec");
  if (mn === null && rc !== null) mn = rc; if (mx === null) mx = mn; if (rc === null && mn !== null) rc = +((mn + mx) / 2).toFixed(2);
  const digest = { summary: $("#liDgSummary").value.trim(), settings: $("#liDgSettings").value.trim(),
    rules: $("#liDgRules").value.split("\n").map(x => x.trim()).filter(Boolean), strength: mn === null ? {} : { min: mn, max: mx, rec: rc, source: "задано вами" } };
  await api("/api/loras/edit", { file: S.infoFile, digest }); toast("Выжимка сохранена"); await loadLoraLib(); openLocalInfo(S.infoFile);
};
$("#liDgReset").onclick = async () => { await api("/api/loras/edit", { file: S.infoFile, digest: null }); await loadLoraLib(); openLocalInfo(S.infoFile); toast("Выжимка сброшена"); };
$("#liSave").onclick = async () => {
  const st = $("#liStrength").value;
  const info = await api("/api/loras/info?file=" + encodeURIComponent(S.infoFile));
  const tw = $("#liTriggers").value.split(",").map(x => x.trim()).filter(Boolean);
  const body = { file: S.infoFile, title: $("#liCustomTitle").value.trim(), note: $("#liNote").value.trim(), strength: st === "" ? null : +st };
  if (tw.join(",") !== (info.triggers || []).join(",")) body.trained_words = tw;
  await api("/api/loras/edit", body);
  toast("Сохранено"); await loadLoraLib();
};
$("#liIdentify").onclick = () => { openLoraDlg("mine"); startIdentify({ file: S.infoFile }); };
$("#liPreviewFrom").onclick = async () => {
  const src = $("#mainImg").getAttribute("src") || "";
  if (!src.startsWith("/output/")) return toast("Сначала откройте готовую картинку в центре экрана");
  try { await api("/api/loras/preview_from", { file: S.infoFile, url: src.split("?")[0] }); toast("Превью обновлено"); await loadLoraLib(); openLocalInfo(S.infoFile); } catch (e) { toast(e.message); }
};
$("#liDelete").onclick = async () => {
  if (!confirm("Удалить файл LoRA, её описание и картинку с диска?\n" + S.infoFile)) return;
  try { await api("/api/loras/delete", { file: S.infoFile }); S.loras = S.loras.filter(l => l.name !== S.infoFile); $("#loraInfoDlg").close(); saveUI(); await loadLoraLib(); } catch (e) { toast(e.message); }
};
$("#clearTransfers2").onclick = async () => { await api("/api/transfers/clear", {}); pollTransfers(); };

/* ---------------- generation ---------------- */
function params() {
  return {
    unet: $("#unet").value, clip: $("#clip").value, vae: $("#vae").value, weight_dtype: $("#weight_dtype").value,
    use_text: $("#useText").checked, use_image: $("#useImage").checked, image_mode: S.imageMode,
    images: S.images.map(i => i.name), prompt: $("#prompt").value, negative: $("#negative").value,
    width: +$("#width").value, height: +$("#height").value, steps: +$("#steps").value, cfg: +$("#cfg").value,
    sampler: $("#sampler").value, scheduler: $("#scheduler").value, batch: +$("#batch").value, seed: +$("#seed").value,
    denoise: +$("#denoise").value, ref_megapixels: +$("#refMp").value,
    loras: S.loras.map(l => ({ name: l.name, strength: l.strength, enabled: l.enabled })),
    character: S.character || null, char_phrase: $("#charPhrase").checked, family: S.family, prompt_src: S.promptSrc || "user",
    ...(S.family === "minimax" ? {
      mm_mode: S.mm, refs: S.mm === "ref" ? S.refs.map(x => ({ name: x.name, label: x.label || "" })) : [], ref_max: $("#refMax").checked,
      unet: S.mm === "ref" ? $("#unetRef").value : $("#unet").value,
      avae: $("#avae").value, duration: +$("#duration").value, turbo: $("#turbo").checked, character: null, use_image: false, images: [],
      first_frame: S.frames.first ? S.frames.first.name : null, last_frame: S.frames.last ? S.frames.last.name : null,
    } : {}),
    ...(S.family === "ltx" ? {
      avae: $("#avae").value, duration: +$("#duration").value, character: null, use_image: false, images: [],
      two_stage: $("#ltxTwo").checked, upscaler_name: $("#upscaler").value || null,
      src_strength: +$("#ltxSrc").value, anchor: +$("#ltxAnchor").value, cfg: 1,
      first_frame: S.frames.first ? S.frames.first.name : null, last_frame: S.frames.last ? S.frames.last.name : null,
    } : {}),
    ...(S.family === "krea2" ? { character: null } : {}),
  };
}
function showErr(msg) { $("#err").textContent = msg; $("#err").classList.toggle("hidden", !msg); }
function showWarn(list) {
  const w = [...new Set(list || [])];
  $("#warnBox").innerHTML = w.map(x => "⚠ " + esc(x)).join("<br>");
  $("#warnBox").classList.toggle("hidden", !w.length);
}
async function generate() {
  if (S.job) return;
  S.job = "pending";  // блокируем сразу, до ответа сервера — иначе повторные нажатия ставят копии в очередь
  $("#genBtn").disabled = true;
  showErr("");
  $("#progress").classList.remove("hidden"); $("#pbar").style.width = "0"; $("#ptxt").textContent = "Отправка… (выгружаю помощника из видеопамяти, если он загружен)";
  let r;
  try { r = await api("/api/generate", params()); }
  catch (e) { showErr(e.message); S.job = null; $("#genBtn").disabled = false; $("#progress").classList.add("hidden"); return; }
  S.job = r.id; S.lastSeed = r.seed; saveUI();
  showWarn(r.warnings);
  $("#stopBtn").classList.remove("hidden");
  $("#ptxt").textContent = "В очереди…";
  const t0 = Date.now(); let lastPrev = 0;
  while (S.job === r.id) {
    await new Promise(res => setTimeout(res, 400));
    let j; try { j = await api("/api/job/" + r.id); } catch { continue; }
    const pct = j.max ? Math.round(100 * j.value / j.max) : 0;
    $("#pbar").style.width = (j.status === "done" ? 100 : pct) + "%";
    $("#ptxt").textContent = `${j.stage || ""}${j.max ? ` ${j.value}/${j.max}` : ""} · ${((Date.now() - t0) / 1000).toFixed(0)} с`;
    if (j.preview_n && j.preview_n !== lastPrev) {
      lastPrev = j.preview_n;
      const img = $("#mainImg"); img.src = `/api/job/${r.id}/preview?n=${j.preview_n}`; img.classList.remove("hidden"); $("#emptyMsg").classList.add("hidden");
      $("#mainVideo").classList.add("hidden");
    }
    if (j.status === "done") { showResults(j.images, r.seed, (Date.now() - t0) / 1000); showWarn(j.warnings); break; }
    if (j.status === "error" || j.status === "cancelled") { showErr(j.error || "Ошибка"); break; }
  }
  S.job = null;
  $("#genBtn").disabled = false; $("#stopBtn").classList.add("hidden");
  setTimeout(() => $("#progress").classList.add("hidden"), 800);
}
const isVideo = u => /\.(mp4|webm|mkv|mov)(\?|$)/i.test(u || "");
function showResults(imgs, seed, secs) {
  S.results = imgs;
  if (!imgs.length) { showErr("Движок не вернул результат"); return; }
  selectResult(0);
  $("#strip").innerHTML = imgs.length > 1 ? imgs.map((u, i) => isVideo(u) ? `<video src="${u}" muted preload="metadata" data-i="${i}"></video>` : `<img src="${u}" data-i="${i}">`).join("") : "";
  $$("#strip [data-i]").forEach(im => im.onclick = () => selectResult(+im.dataset.i));
  $("#meta").textContent = `seed ${seed}${secs ? ` · ${secs.toFixed(1)} с` : ""} · сохранено в ${isVideo(imgs[0]) ? "output/video" : "output/images"}`;
}
function selectResult(i) {
  const u = S.results[i], vid = isVideo(u);
  ["#toEditor", "#saveFace", "#toFace"].forEach(s => $(s).classList.toggle("hidden", vid));
  $("#toVideo").classList.toggle("hidden", vid); $("#vidTools").classList.toggle("hidden", !vid);
  const img = $("#mainImg"), v = $("#mainVideo");
  $("#emptyMsg").classList.add("hidden");
  if (vid) {
    img.classList.add("hidden"); v.src = u; v.classList.remove("hidden"); $("#rateBar").classList.add("hidden");
  } else {
    v.pause(); v.removeAttribute("src"); v.classList.add("hidden");
    img.src = u; img.classList.remove("hidden"); rateShow(u);
  }
  $$("#strip [data-i]").forEach(x => x.classList.toggle("on", +x.dataset.i === i));
}

/* ---------------- Qwen ⇄ MiniMax: перенос картинок и кадров ---------------- */
const imgDims = src => new Promise(res => { const im = new Image(); im.onload = () => res([im.naturalWidth, im.naturalHeight]); im.onerror = () => res([0, 0]); im.src = src; });
async function uploadBlob(blob, name) {
  const fd = new FormData(); fd.append("image", blob, name || "frame.png");
  const r = await fetch("/api/input_upload", { method: "POST", body: fd }).then(r => r.json());
  if (!r.names || !r.names[0]) throw new Error("Не удалось сохранить картинку");
  return r.names[0];
}
/* картинка из папки input становится первым/последним кадром видео MiniMax */
async function setFrame(slot, name, quiet) {
  if (slot === "ref") return addRef(name, quiet);
  const [w, h] = await imgDims("/input/" + encodeURIComponent(name));
  if (S.mm === "ref") setMM("frames");
  if (!isVidFam()) setFamily(S.lastVid || "minimax", true);
  if (S.mode !== "gen") setMode("gen");
  S.frames[slot] = { name, w, h };
  if (w && h) setRatio(w, h);
  renderFrames(); saveUI();
  if (!quiet) toast(`${FAM[S.family].name}: картинка стала ${slot === "first" ? "первым" : "последним"} кадром. Опишите движение и нажмите «Сгенерировать».`, 4500);
}
async function outputToInput(url) {
  const r = await api("/api/edit/source", { url: url.split("?")[0] });
  return r.name;
}
$$("#toVideo [data-slot]").forEach(b => b.onclick = async () => {
  const src = $("#mainImg").getAttribute("src") || "";
  if (!src.startsWith("/output/")) return toast("Сначала выберите готовую картинку");
  try { await setFrame(b.dataset.slot, await outputToInput(src)); } catch (e) { toast(e.message, 4000); }
});
$$("#edToVideo [data-slot]").forEach(b => b.onclick = async () => {
  if (!ED.src) return toast("В Редакторе нет картинки");
  try { await setFrame(b.dataset.slot, ED.src.name); } catch (e) { toast(e.message, 4000); }
});
/* кадр, на котором стоит видео, — в PNG */
async function grabFrame() {
  const v = $("#mainVideo");
  if (!v.videoWidth) throw new Error("Видео ещё не загрузилось");
  v.pause();
  const c = document.createElement("canvas"); c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
  const blob = await new Promise(r => c.toBlob(r, "image/png"));
  return await uploadBlob(blob, `frame_${v.currentTime.toFixed(2)}s.png`);
}
$("#grabEdit").onclick = async () => {
  try {
    const name = await grabFrame();
    setMode("edit"); ED.history = []; await ED.load(name);
    toast("Кадр открыт в Редакторе (Qwen). Исправьте его и верните кнопкой «🎬 В видео».", 5000);
  } catch (e) { toast(e.message, 4000); }
};
$("#grabFirst").onclick = async () => { try { await setFrame("first", await grabFrame()); } catch (e) { toast(e.message, 4000); } };
$("#grabLast").onclick = async () => { try { await setFrame("last", await grabFrame()); } catch (e) { toast(e.message, 4000); } };
$("#grabRef").onclick = async () => { try { await addRef(await grabFrame()); } catch (e) { toast(e.message, 4000); } };

/* слоты «Первый кадр» / «Последний кадр» */
function renderFrames() {
  for (const [slot, id] of [["first", "#frFirst"], ["last", "#frLast"]]) {
    const f = S.frames[slot], box = $(id);
    box.innerHTML = f ? `<img src="/input/${encodeURIComponent(f.name)}" alt=""><button class="icon fx" data-frx="${slot}" title="Убрать кадр">✕</button><span class="fs">${f.w}×${f.h}</span>`
      : `<div>＋ перетащите, выберите<br>или вставьте Ctrl+V</div>`;
  }
  $$("[data-frx]").forEach(b => b.onclick = e => { e.stopPropagation(); S.frames[b.dataset.frx] = null; renderFrames(); saveUI(); });
  const f = S.frames.first, l = S.frames.last;
  $("#framesHint").textContent = f && l ? "Режим «первый + последний кадр»: видео плавно переходит от первой картинки к последней. Опишите, что происходит между ними."
    : f ? "Режим «картинка → видео»: видео начнётся с этого кадра. Опишите движение, камеру и звук."
    : l ? "Режим «последний кадр»: видео придёт к этой картинке в конце. Опишите, что происходит до неё."
    : "Без кадров — видео по тексту. С первым кадром — видео начинается с картинки. С обоими — переход от первого к последнему. Размер видео подстраивается под пропорции кадра.";
  if (typeof updateMode === "function" && isVidFam()) { updateMode(); checkPair(); }
}
async function frameUpload(slot, files) {
  const f = [...files].find(x => x.type.startsWith("image/")); if (!f) return;
  try { await setFrame(slot, await uploadBlob(f, f.name || "paste.png"), true); } catch (e) { toast(e.message, 4000); }
}
for (const id of ["#frFirst", "#frLast"]) {
  const box = $(id);
  box.addEventListener("click", e => { if (e.target.closest("button")) return; S.frSlot = box.dataset.slot; $("#frFile").click(); });
  box.addEventListener("dragover", e => { e.preventDefault(); box.classList.add("over"); });
  box.addEventListener("dragleave", () => box.classList.remove("over"));
  box.addEventListener("drop", e => { e.preventDefault(); box.classList.remove("over"); frameUpload(box.dataset.slot, e.dataTransfer.files); });
}
$("#frFile").onchange = e => { frameUpload(S.frSlot || "first", e.target.files); e.target.value = ""; };
renderFrames();

/* ---------------- длительность (ползунок) ---------------- */
function durShow() {
  const s = +$("#duration").value || 5; let n;
  if (S.family === "ltx") { n = Math.max(9, Math.round(s * 24) + 1); while (n % 8 !== 1) n++; }
  else { n = Math.max(5, Math.round(s * 24)); while (n % 17 !== 5) n++; }
  $("#durV").innerHTML = `${s} с <small>· ${n} кадров</small>`;
  $("#duration").title = s < 5 ? "Короче 5 с модель работает хуже (обучена на 5–15 с)" : "";
}
$("#duration").addEventListener("input", () => { durShow(); saveUI(); });
function ltxShow() {
  $("#ltxSrcV").textContent = (+$("#ltxSrc").value).toFixed(2); $("#ltxAnchorV").textContent = (+$("#ltxAnchor").value).toFixed(2);
  const two = $("#ltxTwo").checked;
  $("#ltxAnchor").disabled = !two; $("#upscaler").disabled = !two;
}
["#ltxSrc", "#ltxAnchor"].forEach(s => $(s).addEventListener("input", () => { ltxShow(); saveUI(); }));
$("#ltxTwo").addEventListener("change", () => { ltxShow(); if (S.family === "ltx") { const [w, h] = ltxCanvas(+$("#width").value || 16, +$("#height").value || 9); $("#width").value = w; $("#height").value = h; updateMp(); updateModelSelects(); } saveUI(); });

/* ---------------- MiniMax: кадры ⇄ референсы ---------------- */
function setMM(m) {
  S.mm = m === "ref" ? "ref" : "frames"; document.body.dataset.mm = S.mm;
  $$("#mmMode button").forEach(b => b.classList.toggle("on", b.dataset.v === S.mm));
  syncTurbo(false); updateModelSelects(); renderRefs(); renderLoras(); updateMode(); checkPair(); saveUI();
}
$$("#mmMode button").forEach(b => b.onclick = () => setMM(b.dataset.v));
const REF_MAX = 9;
async function addRef(name, quiet) {
  if (S.family !== "minimax") { if (S.family === "ltx") toast("Референсы (@img) есть только у MiniMax H3 — переключаю", 3500); setFamily("minimax", true); }
  if (S.mode !== "gen") setMode("gen");
  if (S.mm !== "ref") setMM("ref");
  if (S.refs.length >= REF_MAX) { toast(`Референсов уже ${REF_MAX} — это максимум. Уберите лишний.`, 4000); return; }
  const [w, h] = await imgDims("/input/" + encodeURIComponent(name));
  S.refs.push({ name, w, h, label: "" });
  renderRefs(); updateMode(); saveUI();
  if (!quiet) toast(`MiniMax H3: картинка добавлена как @img${S.refs.length}. Подпишите её («локация», «Маша») и упомяните в промпте.`, 5000);
}
async function refUpload(files) {
  const list = [...files].filter(x => x.type.startsWith("image/"));
  for (const f of list) { if (S.refs.length >= REF_MAX) break; try { await addRef(await uploadBlob(f, f.name || "paste.png"), true); } catch (e) { toast(e.message, 4000); } }
}
/* после удаления или перестановки референса номера @img в промпте сдвигаются вместе с картинками */
function retag(map) {
  const ta = $("#prompt");
  ta.value = ta.value.replace(/@img\s*(\d+)/gi, (m, n) => { const k = map(+n); return k === null ? "" : k ? "@img" + k : m; }).replace(/ {2,}/g, " ");
}
function renderRefs() {
  const g = $("#refGrid"); if (!g) return;
  g.innerHTML = S.refs.map((r, i) => `<div class="ref"><img src="/input/${encodeURIComponent(r.name)}" alt=""><span class="tag" data-ins="${i}" title="Вставить в промпт">@img${i + 1}</span>`
    + `<button class="icon rx" data-rx="${i}" title="Убрать">✕</button>`
    + `<span class="mv">${i ? `<button data-mvl="${i}" title="Раньше">◀</button>` : ""}${i < S.refs.length - 1 ? `<button data-mvr="${i}" title="Позже">▶</button>` : ""}</span>`
    + `<input data-lbl="${i}" value="${esc(r.label || "")}" placeholder="что это? (локация, Маша…)"></div>`).join("")
    + (S.refs.length < REF_MAX ? `<div class="ref add" id="refAdd">＋ картинка<br><span class="tiny">перетащите, выберите<br>или Ctrl+V · до ${REF_MAX}</span></div>` : "");
  $$("[data-rx]", g).forEach(b => b.onclick = () => {
    const i = +b.dataset.rx; S.refs.splice(i, 1);
    retag(n => n === i + 1 ? null : n > i + 1 ? n - 1 : 0); renderRefs(); updateMode(); saveUI();
  });
  const swap = (i, j) => { [S.refs[i], S.refs[j]] = [S.refs[j], S.refs[i]]; retag(n => n === i + 1 ? j + 1 : n === j + 1 ? i + 1 : 0); renderRefs(); saveUI(); };
  $$("[data-mvl]", g).forEach(b => b.onclick = () => swap(+b.dataset.mvl, +b.dataset.mvl - 1));
  $$("[data-mvr]", g).forEach(b => b.onclick = () => swap(+b.dataset.mvr, +b.dataset.mvr + 1));
  $$("[data-lbl]", g).forEach(inp => inp.oninput = () => { S.refs[+inp.dataset.lbl].label = inp.value; renderChips(); saveUI(); });
  $$("[data-ins]", g).forEach(b => b.onclick = () => insertTag(+b.dataset.ins + 1));
  const add = $("#refAdd");
  if (add) {
    add.onclick = () => $("#refFile").click();
    add.addEventListener("dragover", e => { e.preventDefault(); add.classList.add("over"); });
    add.addEventListener("dragleave", () => add.classList.remove("over"));
    add.addEventListener("drop", e => { e.preventDefault(); add.classList.remove("over"); refUpload(e.dataTransfer.files); });
  }
  $$("#mmMode button").forEach(b => b.classList.toggle("on", b.dataset.v === S.mm));
  renderChips();
}
$("#refFile").onchange = e => { refUpload(e.target.files); e.target.value = ""; };
function renderChips() {
  const c = $("#refChips"); if (!c) return;
  c.innerHTML = S.refs.length ? `<span class="muted tiny" style="align-self:center">Вставить:</span>` + S.refs.map((r, i) => `<button class="ghost" data-chip="${i + 1}" title="Вставить @img${i + 1} в промпт"><img src="/input/${encodeURIComponent(r.name)}" alt="">@img${i + 1}${r.label ? " · " + esc(r.label) : ""}</button>`).join("") : "";
  $$("[data-chip]", c).forEach(b => b.onclick = () => insertTag(+b.dataset.chip));
}
/* вставить @imgN в промпт: на место набранного «@…» или в позицию курсора */
function insertTag(n, from) {
  const ta = $("#prompt"), v = ta.value;
  let a = from ?? ta.selectionStart ?? v.length, b = ta.selectionEnd ?? a;
  if (from === undefined && a === 0 && b === 0 && document.activeElement !== ta) a = b = v.length;
  const pre = v.slice(0, a), post = v.slice(b);
  const ins = (pre && !/\s$/.test(pre) ? " " : "") + "@img" + n + (/^\s/.test(post) ? "" : " ");
  ta.value = pre + ins + post;
  const pos = (pre + ins).length; ta.focus(); ta.setSelectionRange(pos, pos);
  atClose(); saveUI();
}
/* всплывающий список по «@» */
const AT = { open: false, sel: 0, items: [], start: 0 };
function atClose() { AT.open = false; $("#atPop").classList.add("hidden"); }
function atUpdate() {
  const ta = $("#prompt");
  if (S.family !== "minimax" || S.mm !== "ref") return atClose();
  const pos = ta.selectionStart, before = ta.value.slice(0, pos), m = /(^|\s)@([^\s@]*)$/.exec(before);
  if (!m) return atClose();
  const q = m[2].toLowerCase();
  AT.start = pos - m[2].length - 1;
  AT.items = S.refs.map((r, i) => ({ n: i + 1, r })).filter(({ n, r }) => !q || ("img" + n).startsWith(q) || String(n).startsWith(q) || (r.label || "").toLowerCase().includes(q));
  const pop = $("#atPop");
  pop.style.top = (ta.offsetTop + ta.offsetHeight + 4) + "px";
  pop.innerHTML = AT.items.length ? AT.items.map((x, k) => `<div class="it${k === AT.sel ? " on" : ""}" data-k="${k}"><img src="/input/${encodeURIComponent(x.r.name)}" alt=""><b>@img${x.n}</b><span>${esc(x.r.label || "")}</span></div>`).join("")
    : `<div class="none">${S.refs.length ? "Нет такой картинки" : "Сначала добавьте картинки в «🧩 Референсы» ниже"}</div>`;
  AT.sel = Math.min(AT.sel, Math.max(0, AT.items.length - 1));
  $$(".it", pop).forEach(el => el.onmousedown = e => { e.preventDefault(); insertTag(AT.items[+el.dataset.k].n, AT.start); });
  pop.classList.remove("hidden"); AT.open = true;
}
function atReplace(n) {
  const ta = $("#prompt"), pos = ta.selectionStart;
  ta.setSelectionRange(AT.start, pos); insertTag(n, AT.start);
}
$("#prompt").addEventListener("input", () => { AT.sel = 0; atUpdate(); });
$("#prompt").addEventListener("click", atUpdate);
$("#prompt").addEventListener("blur", () => setTimeout(atClose, 150));
$("#prompt").addEventListener("keydown", e => {
  if (!AT.open || !AT.items.length) { if (e.key === "Escape") atClose(); return; }
  if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); AT.sel = (AT.sel + (e.key === "ArrowDown" ? 1 : AT.items.length - 1)) % AT.items.length; atUpdate(); }
  else if ((e.key === "Enter" && !e.ctrlKey) || e.key === "Tab") { e.preventDefault(); e.stopPropagation(); atReplace(AT.items[AT.sel].n); }
  else if (e.key === "Escape") { e.preventDefault(); atClose(); }
}, true);
$("#genBtn").onclick = generate;
$("#stopBtn").onclick = async () => { try { await api("/api/interrupt", {}); } catch { } };
$("#freeBtn").onclick = async () => { try { await api("/api/engine/free", {}); toast("Модели выгружены из памяти"); } catch (e) { toast(e.message); } };
document.addEventListener("keydown", e => {
  if (!((e.ctrlKey || e.metaKey) && e.key === "Enter") || e.repeat) return;
  if (e.target.closest && e.target.closest("dialog")) return;  // в окнах (чат и т.п.) Ctrl+Enter не запускает генерацию
  e.preventDefault(); if (S.mode === "edit") edRun(); else if (S.mode === "face") fsRun(); else generate();
});
$("#mainImg").onclick = () => $("#viewer").classList.toggle("zoom");

/* ---------------- dialogs ---------------- */
$("#aboutBtn").onclick = () => openDlg("aboutDlg");
$$("[data-open]").forEach(b => b.onclick = () => b.dataset.open === "modelsDlg" ? openModels() : openDlg(b.dataset.open));
$$("[data-close]").forEach(b => b.onclick = () => b.closest("dialog").close());
$$("dialog").forEach(d => d.addEventListener("click", e => { if (e.target === d) d.close(); }));
function openDlg(id) { const d = $("#" + id); if (!d.open) d.showModal(); if (id === "settingsDlg") loadSettings(); }
$$("[data-openfolder]").forEach(b => b.onclick = () => openFolder(b.dataset.openfolder));
async function openFolder(f) { try { await api("/api/open_folder", { folder: f }); } catch (e) { toast(e.message); } }

/* ---------------- models dialog ---------------- */
let mTab = "diffusion_models", trTimer = null;
function openModels(tab) {
  if (tab) mTab = tab;
  openDlg("modelsDlg"); renderModelsTab(); pollTransfers();
}
$$("#mTabs button").forEach(b => b.onclick = () => { mTab = b.dataset.tab; renderModelsTab(); });
const FOLDER_LABEL = { diffusion_models: "models/diffusion_models", text_encoders: "models/text_encoders", vae: "models/vae", loras: "models/loras", latent_upscale_models: "models/latent_upscale_models (апскейлеры LTX)", checkpoints: "models/checkpoints (3D‑модели Hunyuan3D)", llm: "models/llm (помощник промптов, файлы .gguf)" };
async function renderModelsTab() {
  $$("#mTabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === mTab));
  const cat = mTab === "catalog";
  $("#folderView").classList.toggle("hidden", cat); $("#catalogView").classList.toggle("hidden", !cat);
  if (cat) return renderCatalog();
  await refreshState();
  $("#folderPath").textContent = `Папка: ${FOLDER_LABEL[mTab]} — файлы, положенные туда вручную, появятся после нажатия «Обновить» или автоматически через пару секунд.`;
  const files = S.models[mTab] || [];
  $("#fileList").innerHTML = files.length ? files.map(f => `<div class="file"><span class="nm">${esc(f.name)}</span><span class="sz">${f.size_h}</span><button class="icon" data-delm="${esc(f.name)}" title="Удалить файл">🗑</button></div>`).join("")
    : `<div class="file muted">Пусто. Добавьте свой файл ниже или скачайте рекомендованный на вкладке «⬇ Скачать рекомендованные».</div>`;
  $$("[data-delm]").forEach(b => b.onclick = async () => {
    if (!confirm("Удалить файл " + b.dataset.delm + " с диска?")) return;
    try { await api("/api/delete_model", { folder: mTab, name: b.dataset.delm }); renderModelsTab(); } catch (e) { toast(e.message); }
  });
}
$("#openFolderBtn").onclick = () => openFolder(mTab);
dropzone($("#modelDrop"), $("#modelFile"), files => uploadModels(files, mTab));
function uploadModels(files, folder) {
  files = [...files]; if (!files.length) return Promise.resolve();
  return new Promise(res => {
    const fd = new FormData(); files.forEach(f => fd.append("file", f, f.name));
    const xhr = new XMLHttpRequest(); xhr.open("POST", "/api/upload_model?folder=" + folder);
    const box = $("#upProg"); box.classList.remove("hidden");
    xhr.upload.onprogress = e => { if (e.lengthComputable) { $("i", box).style.width = (100 * e.loaded / e.total) + "%"; $("span", box).textContent = `${fmtBytes(e.loaded)} / ${fmtBytes(e.total)}`; } };
    xhr.onload = async () => { box.classList.add("hidden"); toast(xhr.status === 200 ? "Файл добавлен" : "Ошибка загрузки"); await refreshState(); if ($("#modelsDlg").open) renderModelsTab(); res(); };
    xhr.onerror = () => { box.classList.add("hidden"); toast("Ошибка загрузки"); res(); };
    if (!$("#modelsDlg").open) toast("Копирую файл…");
    xhr.send(fd);
  });
}
$("#impBtn").onclick = async () => {
  try { const r = await api("/api/import_path", { path: $("#impPath").value, folder: mTab, mode: $("#impMode").value }); toast(`Добавляю файлов: ${r.items.length}`); $("#impPath").value = ""; pollTransfers(); }
  catch (e) { toast(e.message, 4000); }
};
$("#dlBtn").onclick = async () => {
  try { await api("/api/download", { url: $("#dlUrl").value, filename: $("#dlName").value, folder: mTab }); $("#dlUrl").value = ""; $("#dlName").value = ""; pollTransfers(); }
  catch (e) { toast(e.message, 4000); }
};
/* каталог: отдельные разделы для картинок (Qwen), видео (MiniMax), помощника (LLM) и голоса */
S.catFam = null;
async function renderCatalog() {
  const c = await api("/api/catalog");
  const fam = S.catFam || (document.body.dataset.view === "voice" ? "voice" : document.body.dataset.view === "3d" ? "3d" : document.body.dataset.view === "music" ? "music" : S.family);
  S.catFam = fam;
  $$("#catTabs [data-cf]").forEach(b => {
    const f = b.dataset.cf, items = c.items.filter(x => (x.family || "llm") === f && x.folder !== "loras");
    const have = items.filter(x => x.installed).length;
    b.classList.toggle("on", f === fam);
    b.dataset.n = f === "voice" ? "" : `${have}/${items.length}`;
    b.onclick = () => { S.catFam = f; renderCatalog(); };
  });
  const notes = {
    qwen: `<b>Картинки — Qwen‑Image 2.1.</b> <b>Турбо</b> (раздел ниже): отдельная модель на 8 шагов — в ~5 раз быстрее; турбо‑LoRA к обычной модели — в окне LoRA → ★ Официальные. Нужны три файла: модель + текстовый энкодер + VAE (★ — рекомендуемое). Рекомендуемый набор: модель int8 + энкодер Qwen3‑VL 8B int8 + VAE (~17 ГБ, без токена). Мало видеопамяти — модель GGUF Q4–Q6 и энкодер w4a8. Для 18+ — энкодер Heretic. «Uncensored GGUF» из интернета — те же исходные веса Qwen.`,
    minimax: `<b>Видео со звуком — MiniMax H3.</b> Нужны: модель + текстовый энкодер + видео‑VAE + аудио‑VAE, турбо‑LoRA — в окне LoRA → ★ Официальные (★ — рекомендуемое; на плашке — для какого железа). <b>RTX 50xx</b>: модель NVFP4 + энкодер NVFP4 AWQ + видео‑VAE INT8 + аудио‑VAE + турбо 8 шагов (~33 ГБ). <b>RTX 30/40</b>: модель mixed INT4/INT8 + тот же энкодер + VAE + турбо, лучше с 64 ГБ ОЗУ. Для режима «🧩 Референсы» — отдельная модель Ref2VA (раздел ниже), её турбо‑LoRA — в окне LoRA. При 32 ГБ ОЗУ: Настройки → Память → «Минимум памяти». GGUF для H3 не предлагается.`,
    flux3: `<b>Картинки — FLUX.3 Image Klein</b> (Black Forest Labs, 5.6B): генерация и правка по картинкам в одной модели. <b>Klein</b> (distilled, ★) — 16 шагов, CFG 1; <b>Base</b> — 50 шагов, CFG 3.5, для своих LoRA. Нужны: модель + энкодер Qwen3‑VL 4B (тот же, что у Krea 2 — подойдёт и он) + VAE FLUX.2 (~11 ГБ с int8). Репозиторий <b>закрыт</b>: откройте страницу, примите условия и впишите HF‑токен в Настройках. Нужен движок новой версии — один раз запустите update_engine.bat. Лицензия FLUX Kommunity: бесплатно, коммерция — при доходе до $5 млн.`,
    krea2: `<b>Картинки — Krea 2</b> (фотореализм). <b>Turbo</b> — 8 шагов, CFG 1 (int8 / fp8 / NVFP4 / bf16). <b>Raw</b> — базовая модель: 28–52 шага, CFG ≈3.5, работает негативный промпт, лучше для своих LoRA; с турбо‑LoRA — тоже 8 шагов. Нужны: модель + энкодер Qwen3‑VL 4B + VAE (~18 ГБ с int8, без токена). LoRA (турбо, перенос стиля, 9 официальных стилей) — в окне LoRA → ★ Официальные. LoRA <b>Style Reference</b> для переноса стиля (до 3 картинок в «Изображение → Референс») скачайте в окне LoRA → ★ Официальные — программа подключит её сама. Хватает 16 ГБ видеопамяти. LoRA с Civitai — база «Krea 2». Лицензия Krea: бесплатно при доходе до $1 млн.`,
    ltx: `<b>Видео со звуком — LTX‑2.5</b> (22B, дистиллированная, 8 шагов). Нужны: модель + энкодер Gemma 4 12B + видео‑VAE + аудио‑VAE + апскейлер ×2 (★). Файлы Lightricks <b>закрыты</b>: откройте страницу, примите условия и впишите HF‑токен в Настройках. <b>32 ГБ ОЗУ</b>: модель GGUF Q4_K_M (без токена) + энкодер int8, профиль памяти «Минимум памяти». <b>64 ГБ ОЗУ</b>: модель int8. Большинство LoRA от LTX‑2.3 подходят. Лицензия: бесплатно при доходе до $10 млн.`,
    music: `<b>Музыка — четыре нейросети.</b> <b>ACE‑Step 1.5</b> (★, MIT) — быстрая, поёт по‑русски и ещё на 50 языках, до 10 минут: модель + два энкодера + VAE (~14 ГБ). <b>MiniMax Music 3</b> — чистый сведённый звук, до 5 минут, медленнее (~12 ГБ). <b>YuE2</b> — сильный вокал на английском и китайском, только некоммерческое использование (один файл). <b>Stable Audio 3</b> — инструментал, фоны и звуки без вокала. Нужен движок новой версии (update_engine.bat).`,
    "3d": `<b>3D — Hunyuan3D 2 от Tencent.</b> Один файл модели (в нём всё нужное). <b>Hunyuan3D‑2mv</b> (★) строит модель по 1–4 ракурсам — спереди, слева, сзади, справа; <b>Turbo</b> — то же быстрее; <b>2.1</b> — по одной картинке, детальнее. Результат — серая сетка .glb без текстуры. Хватает 8–12 ГБ видеопамяти. Для автоматического удаления фона нужна нейросеть ISNet (171 МБ, кнопка во вкладке «3D»).`,
    llm: `<b>Помощник промптов (локальная LLM).</b> Нужен один файл модели .gguf; для «Описать картинку» и чтобы помощник видел кадры — модель со зрением (Qwen3‑VL) и её файл mmproj. Движок llama.cpp ставится кнопкой в ⚙ у помощника.`,
    voice: `<b>Голос — переозвучка и озвучка текста.</b> Модели голоса (OmniVoice, Whisper, MelBand‑Roformer, ~5 ГБ) ставятся одной кнопкой «Установить голосовой модуль» во вкладке «🎙 Голос» вверху.`,
  };
  $("#tokenNote").innerHTML = notes[fam];
  let html = "", g = "";
  if (fam === "voice") {
    html = `<p><button class="primary" id="catToVoice">🎙 Открыть вкладку «Голос»</button></p>`;
  } else for (const it of c.items.filter(x => (x.family || "llm") === fam && x.folder !== "loras")) {
    const gname = it.group.replace(/^.*?·\s*/, "");
    if (it.group !== g) { g = it.group; html += `<div class="catGroup">${esc(gname || g)}</div>`; }
    html += `<div class="cat"><div class="info"><div class="t">${esc(it.title)}${it.recommended ? `<span class="badge rec">Рекомендуемое</span>` : ""}${it.needs === "gguf" ? `<span class="badge" title="Нужна поддержка GGUF в движке — ставится кнопкой, ~1 МБ">GGUF</span>` : ""}${it.gated ? '<span class="badge lock">🔒 токен</span>' : ""}${it.hw ? `<span class="badge" title="Для какого компьютера">${esc(it.hw)}</span>` : ""}${it.installed ? '<span class="badge ok">✓ есть</span>' : ""}</div>
      <div class="d">${esc(it.desc)} · <b>${esc(it.size)}</b> · <a href="${esc(it.page)}" target="_blank">страница</a>${it.pairs ? ` · энкодер: ${esc(it.pairs)}` : ""}</div></div>
      ${it.installed ? "" : `<button class="small ${it.recommended ? "primary" : ""}" data-dl="${it.id}">${it.partial ? "Докачать" : "Скачать"}</button>`}</div>`;
  }
  $("#catalogList").innerHTML = html;
  const tv = $("#catToVoice"); if (tv) tv.onclick = () => { $("#modelsDlg").close(); setView("voice"); };
  $$("[data-dl]").forEach(b => b.onclick = async () => {
    try {
      await api("/api/download", { id: b.dataset.dl }); b.disabled = true; b.textContent = "В очереди"; pollTransfers();
      const it = c.items.find(x => x.id === b.dataset.dl);
      if (it && it.needs === "gguf" && !(S.gguf || {}).installed) toast("Для .gguf нужна поддержка GGUF — кнопка «Установить поддержку GGUF» появится под выбором модели", 6000);
    } catch (e) { toast(e.message, 4000); }
  });
}
async function pollTransfers() {
  clearTimeout(trTimer);
  let r; try { r = await api("/api/transfers"); } catch { return; }
  const items = r.items;
  const html = items.length ? items.map(t => {
    const pct = t.total ? (100 * t.done / t.total) : 0;
    const st = { queued: "в очереди", downloading: "скачивание", copying: "копирование", done: "готово ✓", error: "ошибка", cancelled: "отменено" }[t.status];
    const active = ["queued", "downloading", "copying"].includes(t.status);
    return `<div class="tr"><div class="h"><span>${esc(t.title || t.filename || t.source)} <span class="muted">→ ${t.folder}${t.title && t.filename ? " · " + esc(t.filename) : ""}</span></span>
      <span class="muted">${st}${t.total ? ` · ${fmtBytes(t.done)} / ${fmtBytes(t.total)}` : t.done ? " · " + fmtBytes(t.done) : ""}${active && t.speed ? ` · ${fmtBytes(t.speed)}/с` : ""}
      ${active ? `<button class="link" data-cancel="${t.id}">отменить</button>` : ""}</span></div>
      <div class="bar"><i style="width:${t.status === "done" ? 100 : pct}%"></i></div>${t.error ? `<div class="e">${esc(t.error)}</div>` : ""}</div>`;
  }).join("") : `<p class="muted">Нет активных загрузок</p>`;
  $("#transfers").innerHTML = html; $("#transfers2").innerHTML = html;
  $$("[data-cancel]").forEach(b => b.onclick = () => api("/api/transfers/cancel", { id: b.dataset.cancel }));
  // карточки Civitai, которые ждали скачивания
  for (const t of items) {
    const pc = S.pendingCards[t.id]; if (!pc) continue;
    const b = $("[data-get]", pc.el);
    if (t.status === "done") { b.textContent = "✓ Есть · в стек"; b.classList.remove("primary"); b.disabled = false; b.dataset.file = t.filename; const c = pc.el._card; (c.versions || []).forEach(v => { if (v.id == pc.vid) v.installed = true; }); if (!c.versions) c.installed = true; delete S.pendingCards[t.id]; }
    else if (["error", "cancelled"].includes(t.status)) { b.textContent = "⬇ Скачать"; b.disabled = false; delete S.pendingCards[t.id]; }
    else if (t.total) b.textContent = `Скачивается ${Math.round(100 * t.done / t.total)}%`;
  }
  const active = items.some(t => ["queued", "downloading", "copying"].includes(t.status));
  if (pollTransfers._wasActive && !active) { refreshState(); loadLoraLib(); if ($("#modelsDlg").open) (mTab === "catalog" ? renderCatalog() : renderModelsTab()); }
  pollTransfers._wasActive = active;
  if (active || $("#modelsDlg").open || $("#loraDlg").open) trTimer = setTimeout(pollTransfers, active ? 1000 : 3000);
}
$("#clearTransfers").onclick = async () => { await api("/api/transfers/clear", {}); pollTransfers(); };

/* ---------------- settings ---------------- */
async function loadSettings() {
  const s = await api("/api/settings");
  $("#hfToken").value = s.hf_token || ""; $("#civToken").value = s.civitai_token || ""; $("#engineArgs").value = s.engine_args || ""; $("#civDomain").value = s.civitai_domain || "civitai.red";
  $("#memProfile").value = s.mem_profile || "auto";
  api("/api/state").then(st => { const n = { default: "обычный", ram_safe: "бережный к ОЗУ", ram_min: "минимум памяти" }[st.mem_profile_effective]; $("#memProfileNow").textContent = `Сейчас работает: ${n}` + (st.device && st.device.ram_total ? ` (ОЗУ этого компьютера: ${st.device.ram_total})` : ""); }).catch(() => { }); $("#memGuard").checked = s.mem_guard !== false; $("#memGuardGb").value = s.mem_guard_gb ?? 1.5; $("#vramHeadroom").value = s.vram_headroom ?? 2; $("#llmUnloadAlways").checked = s.llm_unload_always !== false;
  $("#rootPath").textContent = "Папка программы: " + s.root;
}
$$("#argChips [data-arg]").forEach(b => b.onclick = () => {
  const cur = $("#engineArgs").value.trim(), a = b.dataset.arg;
  $("#engineArgs").value = cur.includes(a) ? cur.replace(a, "").replace(/\s+/g, " ").trim() : (cur + " " + a).trim();
});
$("#saveSettings").onclick = async () => {
  const r = await api("/api/settings", { hf_token: $("#hfToken").value.trim(), civitai_token: $("#civToken").value.trim(), engine_args: $("#engineArgs").value.trim(), civitai_domain: $("#civDomain").value,
    mem_profile: $("#memProfile").value, mem_guard: $("#memGuard").checked, mem_guard_gb: +$("#memGuardGb").value || 1.5, vram_headroom: Math.max(0, Math.min(6, +$("#vramHeadroom").value || 0)), llm_unload_always: $("#llmUnloadAlways").checked });
  S.civDomain = $("#civDomain").value; loadLoraLib();
  toast(r.restarted ? "Сохранено, движок перезапускается…" : "Сохранено"); $("#settingsDlg").close();
};

/* ---------------- gallery ---------------- */
$("#galleryBtn").onclick = async () => {
  openDlg("galleryDlg");
  const r = await api("/api/gallery");
  const rated = await api("/api/lab/ratings").catch(() => ({}));
  const cells = [];
  r.items.forEach((e, ei) => e.images.forEach(u => { if (!/\.(glb|mp3|flac|wav|opus)(\?|$)/i.test(u)) cells.push({ u, e, ei }); }));
  S._gal = r.items;
  $("#gallery").innerHTML = cells.length ? cells.map(c => `<div class="gi">${rated[c.u] ? `<span class="grate" title="Ваша оценка">★${rated[c.u]}</span>` : ""}${isVideo(c.u) ? `<video src="${c.u}" muted preload="metadata" data-u="${c.u}"></video><span class="gvid">🎬 видео</span>` : `<img src="${c.u}" loading="lazy" data-u="${c.u}">`}<div class="gp">${esc((c.e.params.prompt_user || c.e.params.prompt || "").slice(0, 120)) || "<i>без промпта</i>"}</div>
    <div class="gb">${isVideo(c.u) ? "" : `<button data-toed="${c.u}" title="Открыть в редакторе">✏</button>`}<button data-reuse="${c.ei}">Настройки</button><button data-seed="${c.ei}">+ seed</button><button data-gdel="${c.u}">🗑</button></div></div>`).join("")
    : `<p class="muted">Пока пусто</p>`;
  $$("#gallery [data-u]").forEach(im => im.onclick = () => { S.results = [im.dataset.u]; selectResult(0); $("#strip").innerHTML = ""; $("#galleryDlg").close(); });
  $$("[data-reuse]").forEach(b => b.onclick = () => { applyParams(S._gal[+b.dataset.reuse].params, false); $("#galleryDlg").close(); toast("Настройки восстановлены (seed случайный)"); });
  $$("[data-seed]").forEach(b => b.onclick = () => { applyParams(S._gal[+b.dataset.seed].params, true); $("#galleryDlg").close(); toast("Настройки и seed восстановлены"); });
  $$("[data-toed]").forEach(b => b.onclick = () => { $("#galleryDlg").close(); setMode("edit"); edFromUrl(b.dataset.toed); });
  $$("[data-gdel]").forEach(b => b.onclick = async () => { await api("/api/gallery/delete", { url: b.dataset.gdel }); b.closest(".gi").remove(); });
};
function applyParams(p, withSeed) {
  if (!FAM[p.family || "flux"]) { toast("Это картинка FLUX.2 Klein — эта нейросеть убрана из программы, её настройки не восстановить", 5000); return; }
  setFamily(p.family, true);
  const set = (id, v) => { if (v !== undefined && $("#" + id)) $("#" + id).value = v; };
  const isRefJob = p.family === "minimax" && p.mm_mode === "ref";
  S._pending = isRefJob ? { unetRef: p.unet, clip: p.clip, vae: p.vae, avae: p.avae } : { unet: p.unet, clip: p.clip, vae: p.vae, avae: p.avae };
  if (p.family === "ltx") {
    set("duration", p.duration); durShow(); $("#ltxTwo").checked = p.two_stage !== false;
    set("ltxSrc", p.src_strength); set("ltxAnchor", p.anchor); ltxShow();
    S._pending.upscaler = p.upscaler;
    S.frames = { first: p.first_frame ? { name: p.first_frame, w: 0, h: 0 } : null, last: p.last_frame ? { name: p.last_frame, w: 0, h: 0 } : null };
    renderFrames();
  }
  if (p.family === "minimax") {
    set("duration", p.duration); durShow(); $("#turbo").checked = !!p.turbo; $("#refMax").checked = !!p.ref_max;
    if (isRefJob) S.refs = (p.refs_meta || []).map(r => ({ name: r.name, label: r.label || "", w: 0, h: 0 }));
    else S.frames = { first: p.first_frame ? { name: p.first_frame, w: 0, h: 0 } : null, last: p.last_frame ? { name: p.last_frame, w: 0, h: 0 } : null };
    setMM(isRefJob ? "ref" : "frames"); renderFrames();
  }
  updateModelSelects();
  set("weight_dtype", p.weight_dtype); set("prompt", p.prompt_user ?? p.prompt); set("negative", p.negative); set("width", p.width); set("height", p.height);
  set("steps", p.steps); set("cfg", p.cfg); set("sampler", p.sampler); set("scheduler", p.scheduler); set("batch", p.batch);
  set("denoise", p.denoise); set("refMp", p.ref_megapixels); set("seed", withSeed ? p.seed : -1);
  $("#useText").checked = !!p.use_text; $("#useImage").checked = !!p.use_image; S.imageMode = p.image_mode || "reference";
  S.images = (p.images || []).map(n => ({ name: n, w: 0, h: 0 })); renderThumbs();
  S.loras = (p.loras || []).map(l => ({ ...l })); syncTurbo(false); renderLoras();
  S.character = p.character || null; renderCharSel();
  $("#denoiseV").textContent = (+$("#denoise").value).toFixed(2);
  updateMode(); updateMp(); checkPair(); saveUI();
}

/* ---------------- log ---------------- */
$("#logBtn").onclick = async () => { openDlg("logDlg"); const r = await api("/api/engine/log"); $("#logText").textContent = r.log || "(пусто)"; $("#logText").scrollTop = 1e9; };
$("#restartEngine").onclick = async () => { await api("/api/engine/restart", {}); toast("Движок перезапускается…"); $("#logDlg").close(); };

/* ---------------- проверка триггер-слов ---------------- */
function activeLoras() { return S.loras.filter(l => l.enabled && +l.strength !== 0); }
function checkTriggers() {
  const box = $("#trigWarn");
  const text = $("#prompt").value.toLowerCase();
  const miss = [];
  if ($("#useText").checked) for (const l of activeLoras()) {
    const info = S.lib[l.name]; const words = info?.trained_words || [];
    if (!words.length) continue;
    const lacking = words.filter(w => !text.includes(w.toLowerCase()));
    if (lacking.length === words.length) miss.push({ l, info, words: lacking });
  }
  box.classList.toggle("hidden", !miss.length);
  box.innerHTML = miss.length ? "⚠ В промпте нет триггер-слов: " + miss.map((m, i) =>
    `<b>${esc(loraTitle(m.info, m.l.name))}</b> — ${esc(m.words.join(", "))}${m.info.triggers_source === "file" ? " <span class='muted'>(угадано по тегам файла)</span>" : ""} <button data-addtw="${i}">вставить</button>`).join("; ") : "";
  $$("[data-addtw]", box).forEach(b => b.onclick = () => { insertWords(miss[+b.dataset.addtw].words); checkTriggers(); });
}
$("#prompt").addEventListener("input", () => { clearTimeout(checkTriggers._t); checkTriggers._t = setTimeout(checkTriggers, 300); });
const _renderLoras = renderLoras;
renderLoras = function () { _renderLoras(); checkTriggers(); renderTips(); };
/* подсказки по включённым LoRA: сила, что делает, правила (из выжимки) */
function renderTips() {
  const box = $("#loraTips"); if (!box) return;
  const act = activeLoras().map(l => ({ l, info: S.lib[l.name] })).filter(x => x.info);
  const tips = act.filter(x => recStr(x.info) || x.info.digest?.summary || (x.info.digest?.rules || []).length || x.info.digest?.settings);
  const need = act.filter(x => x.info.needs_digest);
  const bad = act.filter(x => outOfRng(+x.l.strength, recStr(x.info)));
  if (!tips.length && !need.length) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  box.innerHTML = tips.map(({ l, info }) => {
    const d = info.digest || {}, st = recStr(info);
    return `<div class="tip">💡 <b>${esc(loraTitle(info, l.name))}</b>${d.summary ? ` — ${esc(d.summary)}` : ""}${st ? `<br><span class="${outOfRng(+l.strength, st) ? "bad" : "r"}">сила: рекомендуют ${fmtRng(st)}${st.rec !== undefined ? ` (лучше ${+st.rec})` : ""}, сейчас ${+l.strength}</span>` : ""}${(d.rules || []).length ? `<br><span class="r">${d.rules.map(esc).join(" · ")}</span>` : ""}${d.settings ? `<br><span class="r">автор: ${esc(d.settings)}</span>` : ""}</div>`;
  }).join("") + `<div class="row2">${bad.length ? `<button class="small" id="tipApply">Поставить рекомендуемую силу${bad.length > 1 ? ` (${bad.length})` : ""}</button>` : ""}${need.length ? `<button class="small" id="tipDigest" title="Помощник один раз прочитает описания этих LoRA со страницы Civitai и запишет выжимку">🧠 Разобрать описания (${need.length})</button>` : ""}</div>`;
  const ap = $("#tipApply"); if (ap) ap.onclick = () => { bad.forEach(({ l, info }) => { l.strength = +recStr(info).rec; }); renderLoras(); saveUI(); toast("Сила выставлена по рекомендациям"); };
  const dg = $("#tipDigest"); if (dg) dg.onclick = () => digestRun(need.map(x => x.l.name), dg);
}
async function digestRun(files, btn) {
  if (btn) { btn.disabled = true; btn.textContent = "Помощник читает описания…"; }
  try { await api("/api/loras/digest_all", { files }); }
  catch (e) { toast(e.message, 6000); if (btn) { btn.disabled = false; btn.textContent = "🧠 Разобрать описания"; } return; }
  const tick = async () => {
    const st = await api("/api/loras/digest_all").catch(() => null); if (!st) return;
    const txt = st.running ? `Помощник разбирает описания: ${st.done}/${st.total}${st.current ? " · " + st.current : ""}` : `Готово: разобрано ${st.done}.` + (st.errors.length ? " Ошибки: " + st.errors.join("; ") : "");
    if (btn && document.body.contains(btn)) btn.textContent = txt;
    if ($("#loraDlg").open) { const b = $("#identProg"); b.classList.remove("hidden"); b.textContent = txt; }
    if (st.running) setTimeout(tick, 1500); else { await loadLoraLib(); if ($("#loraInfoDlg").open && S.infoFile) openLocalInfo(S.infoFile); }
  };
  tick();
}

/* ---------------- помощник промптов ---------------- */
function aiParams(extra) {
  return {
    prompt: $("#prompt").value, detail: $("#aiDetail").value, censor: +$("#aiCensor").value, family: S.family,
    loras: activeLoras().map(l => ({ name: l.name, strength: l.strength, enabled: true })),
    edit_mode: ["qwen", "flux3"].includes(S.family) && $("#useImage").checked && S.imageMode === "reference" && S.images.length > 0, character: S.family === "qwen" ? (S.character || null) : null,
    ...(S.family === "ltx" ? { duration: +$("#duration").value, first_frame: S.frames.first ? S.frames.first.name : null, last_frame: S.frames.last ? S.frames.last.name : null } : {}),
    ...(S.family === "minimax" ? { mm_mode: S.mm, refs: S.mm === "ref" ? S.refs.map(x => ({ name: x.name, label: x.label || "" })) : [], duration: +$("#duration").value, first_frame: S.frames.first ? S.frames.first.name : null, last_frame: S.frames.last ? S.frames.last.name : null } : {}),
    ...extra,
  };
}
let aiBusy = false;
async function aiRun(url, body, label) {
  if (aiBusy) return null;
  aiBusy = true; $$(".assist button").forEach(b => b.disabled = true);
  const out = $("#aiOut"); out.classList.remove("hidden");
  const t0 = Date.now();
  const tick = setInterval(() => { out.innerHTML = `<span class="spin"></span>${label}… ${Math.round((Date.now() - t0) / 1000)} с <span class="muted tiny">(первый запуск — загрузка модели)</span>`; }, 500);
  out.innerHTML = `<span class="spin"></span>${label}…`;
  try { return await api(url, body); }
  catch (e) {
    out.innerHTML = `<div class="txt" style="color:#ffb3ad">${esc(e.message)}</div><div class="row2">${/не установлен|не выбрана|mmproj|зрени/i.test(e.message) ? `<button class="small primary" data-aiset>Настроить помощника</button>` : ""}<button class="small" data-aiclose>Закрыть</button></div>`;
    bindAiOut(); return null;
  } finally { clearInterval(tick); aiBusy = false; $$(".assist button").forEach(b => b.disabled = false); }
}
function bindAiOut() {
  const out = $("#aiOut");
  $$("[data-aiclose]", out).forEach(b => b.onclick = () => out.classList.add("hidden"));
  $$("[data-aiset]", out).forEach(b => b.onclick = () => openLlm());
}
function setPrompt(text) {
  S.promptUndo = $("#prompt").value;
  $("#prompt").value = text; S.promptSrc = "helper"; saveUI(); checkTriggers();
}
$("#prompt").addEventListener("input", () => { if (S.promptSrc === "helper") S.promptSrc = "helper_edited"; else if (!$("#prompt").value.trim()) S.promptSrc = "user"; });
function showResult(text, note, title, again) {
  const out = $("#aiOut");
  out.innerHTML = `<div class="muted tiny">${esc(title)}</div><div class="txt">${esc(text)}</div>${note ? `<div class="note">⚠ ${esc(note)}</div>` : ""}
    <div class="row2"><button class="small primary" data-use>Заменить промпт</button><button class="small primary" data-usegen title="Заменить промпт и сразу сгенерировать">▶ Заменить и сгенерировать</button>${again ? `<button class="small" data-again>${again.label}</button>` : ""}<button class="small" data-append>Добавить в конец</button><button class="small" data-copy>Копировать</button><button class="small" data-aiclose>✕</button></div>`;
  $("[data-usegen]", out).onclick = () => { setPrompt(text); out.classList.add("hidden"); generate(); };
  if (again) $("[data-again]", out).onclick = again.fn;
  $("[data-use]", out).onclick = () => { setPrompt(text); out.innerHTML = `<div class="row2"><span class="muted tiny">Промпт заменён.</span><button class="small" data-undo>↩ Вернуть прежний</button><button class="small" data-aiclose>✕</button></div>`; bindUndo(); bindAiOut(); };
  $("[data-append]", out).onclick = () => { setPrompt(($("#prompt").value.trim() + " " + text).trim()); out.classList.add("hidden"); };
  $("[data-copy]", out).onclick = () => { navigator.clipboard?.writeText(text); toast("Скопировано"); };
  bindAiOut();
}
function bindUndo() { $$("[data-undo]", $("#aiOut")).forEach(b => b.onclick = () => { $("#prompt").value = S.promptUndo || ""; saveUI(); checkTriggers(); $("#aiOut").classList.add("hidden"); }); }
$("#aiEnhance").onclick = async () => {
  const r = await aiRun("/api/llm/enhance", aiParams({ n: 1 }), activeLoras().some(l => S.lib[l.name]?.needs_digest) ? "Помощник читает описания LoRA (один раз) и пишет промпт" : "Помощник пишет промпт");
  if (activeLoras().some(l => S.lib[l.name]?.needs_digest)) loadLoraLib();
  if (r) showResult(r.prompt, r.note, "Улучшенный промпт" + (activeLoras().length ? ` · учтены LoRA: ${activeLoras().length}` : ""));
};
const censorNames = ["0 · без цензуры", "1 · ню", "2 · чувственно", "3 · мягко", "4 · строго"];
function showCensor() { const v = +$("#aiCensor").value; $("#aiCensorV").textContent = censorNames[v]; $("#aiCensorV").dataset.v = v; }
$("#aiCensor").addEventListener("input", () => { showCensor(); saveUI(); });
showCensor();
async function aiRandom() {
  const r = await aiRun("/api/llm/enhance", aiParams({ mode: "random", n: 1 }), "Помощник придумывает сцену");
  if (!r) return;
  const insp = r.inspiration ? ` · подсказки: ${r.inspiration.setting}, ${r.inspiration.style}` : "";
  S.randomBase = S.randomBase ?? $("#prompt").value;
  showResult(r.prompt, r.note, "Случайная сцена" + insp, { label: "🎲 Ещё вариант", fn: aiRandom });
}
$("#aiRandom").onclick = () => { S.randomBase = null; aiRandom(); };
$("#aiVariants").onclick = async () => {
  const r = await aiRun("/api/llm/enhance", aiParams({ n: 4 }), "Помощник пишет варианты");
  if (!r) return;
  const out = $("#aiOut");
  out.innerHTML = `<div class="muted tiny">Варианты — выберите один</div>${r.note ? `<div class="note">⚠ ${esc(r.note)}</div>` : ""}` +
    r.variants.map((v, i) => `<div class="var"><div class="txt">${esc(v)}</div><div class="row2"><button class="small primary" data-v="${i}">Использовать</button></div></div>`).join("") +
    `<div class="row2"><button class="small" data-aiclose>✕ Закрыть</button></div>`;
  $$("[data-v]", out).forEach(b => b.onclick = () => { setPrompt(r.variants[+b.dataset.v]); out.innerHTML = `<div class="row2"><span class="muted tiny">Промпт заменён.</span><button class="small" data-undo>↩ Вернуть прежний</button><button class="small" data-aiclose>✕</button></div>`; bindUndo(); bindAiOut(); });
  bindAiOut();
};
$("#aiDescribe").onclick = async () => {
  if (S.family === "minimax" && S.mm === "ref") { toast("В режиме референсов нажмите «✨ Улучшить» — помощник посмотрит на все картинки и напишет промпт с @img", 4500); return; }
  const img = isVidFam() ? (S.frames.first && S.frames.first.name) : (S.images[0] && S.images[0].name);
  if (!img) { toast(isVidFam() ? "Добавьте первый кадр — помощник опишет, как его оживить" : "Сначала включите «Изображение» и загрузите картинку", 3500); return; }
  const r = await aiRun("/api/llm/describe", aiParams({ image: img, detail: $("#aiDetail").value === "medium" ? "long" : $("#aiDetail").value }), "Помощник рассматривает картинку");
  if (r) showResult(r.prompt, r.note, "Описание картинки" + ($("#prompt").value.trim() ? " (с учётом ваших пожеланий из промпта)" : ""));
};

/* ---- чат ---- */
S.chat = [];
$("#aiChat").onclick = () => { openDlg("chatDlg"); renderChat(); $("#chatText").focus(); };
function renderChat() {
  const n = activeLoras().length;
  $("#chatCtx").textContent = `Помощник видит: текущий промпт${n ? ` и включённые LoRA (${n}) с их триггер-словами, описаниями и примерами` : ""}.`;
  const log = $("#chatLog");
  const prompts = [];  // готовые промпты из ответов помощника (строки «PROMPT: …» до пустой строки)
  log.innerHTML = S.chat.length ? S.chat.map(m => {
    if (m.role === "user") return `<div class="msg user">${esc(m.content)}</div>`;
    const out = [], lines = m.content.split("\n");
    let cur = null, text = [];
    const flushText = () => { if (text.length) { out.push(esc(text.join("\n"))); text = []; } };
    const flushPrompt = () => {
      if (cur === null) return;
      const p = cur.join("\n").trim(); cur = null;
      if (!p) return;
      const k = prompts.push(p) - 1;
      out.push(`<div class="pblock"><div class="ptext">${esc(p)}</div><div class="pbtns"><button class="small primary" data-prep="${k}">Заменить промпт</button><button class="small primary" data-pgen="${k}">▶ Заменить и сгенерировать</button></div></div>`);
    };
    for (const ln of lines) {
      const mm = ln.match(/^\s*\**\s*PROMPT\s*:?\s*\**\s*:?\s*(.*)$/i);
      if (mm) { flushText(); flushPrompt(); cur = [mm[1]]; continue; }
      if (cur !== null) { if (!ln.trim()) flushPrompt(); else cur.push(ln); continue; }
      text.push(ln);
    }
    flushPrompt(); flushText();
    return `<div class="msg assistant">${out.join("\n")}</div>`;
  }).join("") : `<p class="muted">Напишите, что хотите изменить или получить. Например: «сделай сцену ночной с неоном», «предложи 3 позы», «почему лицо выходит размытым?»</p>`;
  $$("[data-prep]", log).forEach(b => b.onclick = () => { setPrompt(prompts[+b.dataset.prep]); toast("Промпт заменён"); });
  $$("[data-pgen]", log).forEach(b => b.onclick = () => { setPrompt(prompts[+b.dataset.pgen]); $("#chatDlg").close(); if (S.mode !== "gen") setMode("gen"); generate(); });
  log.scrollTop = log.scrollHeight;
}
async function chatSend() {
  const t = $("#chatText").value.trim(); if (!t || aiBusy) return;
  S.chat.push({ role: "user", content: t }); $("#chatText").value = ""; renderChat();
  aiBusy = true; $("#chatSend").disabled = true;
  const log = $("#chatLog"); const w = document.createElement("div"); w.className = "msg assistant"; w.innerHTML = `<span class="spin"></span>думаю…`; log.appendChild(w); log.scrollTop = log.scrollHeight;
  try { const r = await api("/api/llm/chat", aiParams({ messages: S.chat })); S.chat.push({ role: "assistant", content: r.reply }); }
  catch (e) { S.chat.push({ role: "assistant", content: "⚠ " + e.message }); }
  finally { aiBusy = false; $("#chatSend").disabled = false; renderChat(); }
}
$("#chatSend").onclick = chatSend;
$("#chatText").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); chatSend(); } });
$("#chatClear").onclick = () => { S.chat = []; renderChat(); };

/* ---- настройки помощника ---- */
let llmTimer = null;
async function openLlm() { openDlg("llmDlg"); await renderLlm(); }
$("#aiSettings").onclick = openLlm;
async function renderLlm() {
  clearTimeout(llmTimer);
  let s; try { s = await api("/api/llm/state"); } catch { return; }
  const ins = s.install, eng = $("#llmEngine");
  const variants = [["cuda-13.4", "NVIDIA, драйвер 580+ (CUDA 13)"], ["cuda-12.4", "NVIDIA, драйвер постарше (CUDA 12)"], ["vulkan", "AMD / Intel / любая (Vulkan)"], ["cpu", "Только процессор"]];
  if (ins.running) {
    const pct = ins.total ? Math.round(100 * ins.done / ins.total) : 0;
    eng.innerHTML = `<div class="engBox">Скачиваю ${esc(ins.file)}… ${fmtBytes(ins.done)}${ins.total ? " / " + fmtBytes(ins.total) : ""}<div class="bar"><i style="width:${pct}%"></i></div></div>`;
  } else if (s.engine_installed) {
    eng.innerHTML = `<div class="engBox">✓ Установлен: llama.cpp ${esc(s.engine_variant)} <button class="link" id="llmReinstall">переустановить…</button></div>`;
  } else {
    eng.innerHTML = `<div class="engBox">Движок ещё не установлен (~50–600 МБ, один раз).${ins.error ? `<div style="color:#ffb3ad;margin-top:6px">Ошибка: ${esc(ins.error)}</div>` : ""}</div>`;
  }
  if (!ins.running && (!s.engine_installed || S._reinstall)) {
    eng.innerHTML += `<div class="rowbtns"><select id="llmVariant">${variants.map(([v, t]) => `<option value="${v}" ${v === s.detected_variant ? "selected" : ""}>${t}${v === s.detected_variant ? " — определено автоматически" : ""}</option>`).join("")}</select><button class="small primary" id="llmInstall">Установить</button></div>`;
    $("#llmInstall").onclick = async () => { S._reinstall = false; try { await api("/api/llm/install", { variant: $("#llmVariant").value }); } catch (e) { toast(e.message); } renderLlm(); };
  }
  const re = $("#llmReinstall"); if (re) re.onclick = () => { S._reinstall = true; renderLlm(); };
  fillSelect($("#llmModel"), s.models.length ? s.models : [["", "— нет моделей в models/llm —"]], s.model);
  const mm = [["auto", "авто" + (s.mmproj_auto ? ` (${s.mmproj_auto})` : " (не найден)")], ["none", "не использовать"], ...s.mmprojs.map(m => [m, m])];
  fillSelect($("#llmMmproj"), mm, s.mmproj || "auto");
  $("#llmDevice").value = s.device; $("#llmKeep").checked = s.keep_loaded; $("#llmAutoDigest").checked = s.auto_digest !== false;
  $("#llmLoaded").textContent = s.loaded ? `Сейчас загружена: ${s.loaded}` : "Сейчас не загружена";
  if (ins.running && $("#llmDlg").open) llmTimer = setTimeout(renderLlm, 1000);
}
const saveLlm = async () => { await api("/api/llm/config", { model: $("#llmModel").value, mmproj: $("#llmMmproj").value, device: $("#llmDevice").value, keep_loaded: $("#llmKeep").checked, auto_digest: $("#llmAutoDigest").checked }); renderLlm(); };
["#llmModel", "#llmMmproj", "#llmDevice", "#llmKeep", "#llmAutoDigest"].forEach(s => $(s).addEventListener("change", saveLlm));
$("#llmUnload").onclick = async () => { await api("/api/llm/unload", {}); renderLlm(); toast("Помощник выгружен"); };
$("#llmGetModels").onclick = () => { $("#llmDlg").close(); S.catFam = "llm"; openModels("catalog"); };

/* ---------------- режимы: Генерация / Редактор ---------------- */
function setMode(m) {
  if (m !== "gen" && S.family !== "qwen") { setFamily("qwen", true); toast("Редактор и замена лица работают на Qwen‑Image 2.1", 3000); }
  S.mode = m;
  $$(".modeTabs button").forEach(b => b.classList.toggle("on", b.dataset.mode === m));
  const ed = m === "edit" || m === "face";
  $("#editorView").classList.toggle("hidden", !ed);
  $(".edBottom").classList.toggle("hidden", m !== "edit");
  $("#fsBottom").classList.toggle("hidden", m !== "face");
  $("#fsBoxes").classList.toggle("hidden", m !== "face");
  ["#viewer", "#strip", "#meta", "#genActions"].forEach(s => $(s).classList.toggle("hidden", ed));
  ["#cardInput", "#cardRes"].forEach(s => $(s).classList.toggle("hidden", ed));
  if (ed) ED.fit();
  if (m === "face") fsEnter();
  try { localStorage.setItem("flux2mode", m); } catch { }
}
$$(".modeTabs button").forEach(b => b.onclick = () => setMode(b.dataset.mode));

/* ---------------- редактор ---------------- */
const ED = {
  src: null, W: 0, H: 0, scale: 1, tool: "brush", undo: [], history: [], drawing: false, last: null, busy: false,
  img: $("#edImg"), mask: $("#edMask"), cur: $("#edCursor"), stack: $("#edStack"), wrap: $("#edWrap"),
  async load(name, w, h, keepHistory) {
    const im = new Image();
    await new Promise((res, rej) => { im.onload = res; im.onerror = rej; im.src = "/input/" + encodeURIComponent(name); });
    if (ED.src && keepHistory) ED.history.push(ED.src);
    ED.src = { name, w: im.naturalWidth, h: im.naturalHeight, el: im };
    ED.W = im.naturalWidth; ED.H = im.naturalHeight;
    for (const c of [ED.img, ED.mask]) { c.width = ED.W; c.height = ED.H; }
    ED.img.getContext("2d").drawImage(im, 0, 0);
    ED.undo = []; ED.stack.classList.remove("hidden"); $("#edEmpty").classList.add("hidden");
    ED.fit(); ED.info(`Картинка ${ED.W}×${ED.H}`);
  },
  fit() {
    if (!ED.W) return;
    const r = ED.wrap.getBoundingClientRect();
    ED.scale = Math.min((r.width - 16) / ED.W, (r.height - 16) / ED.H, 4);
    const w = Math.round(ED.W * ED.scale), h = Math.round(ED.H * ED.scale);
    for (const c of [ED.img, ED.mask]) { c.style.width = w + "px"; c.style.height = h + "px"; }
    ED.stack.style.width = w + "px"; ED.stack.style.height = h + "px";
  },
  info(t) { $("#edInfo").textContent = t || ""; },
  hasMask() {
    if (!ED.W) return false;
    const d = ED.mask.getContext("2d").getImageData(0, 0, ED.W, ED.H).data;
    for (let i = 3; i < d.length; i += 16) if (d[i] > 0) return true;
    return false;
  },
  pos(e) { const r = ED.mask.getBoundingClientRect(); return { x: (e.clientX - r.left) / ED.scale, y: (e.clientY - r.top) / ED.scale }; },
  brushPx() { return +$("#edSize").value / ED.scale; },
  stroke(a, b) {
    const ctx = ED.mask.getContext("2d");
    ctx.globalCompositeOperation = ED.tool === "eraser" ? "destination-out" : "source-over";
    ctx.strokeStyle = ctx.fillStyle = "rgb(255,40,40)"; ctx.lineWidth = ED.brushPx(); ctx.lineCap = ctx.lineJoin = "round";
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    ctx.beginPath(); ctx.arc(b.x, b.y, ED.brushPx() / 2, 0, Math.PI * 2); ctx.fill();
  },
  pushUndo() { ED.undo.push(ED.mask.getContext("2d").getImageData(0, 0, ED.W, ED.H)); if (ED.undo.length > 12) ED.undo.shift(); },
  async maskBlob() {
    const c = document.createElement("canvas"); c.width = ED.W; c.height = ED.H;
    const ctx = c.getContext("2d");
    const src = ED.mask.getContext("2d").getImageData(0, 0, ED.W, ED.H), out = ctx.createImageData(ED.W, ED.H);
    for (let i = 0; i < src.data.length; i += 4) { const v = src.data[i + 3] > 20 ? 255 : 0; out.data[i] = out.data[i + 1] = out.data[i + 2] = v; out.data[i + 3] = 255; }
    ctx.putImageData(out, 0, 0);
    return await new Promise(r => c.toBlob(r, "image/png"));
  },
};
window.addEventListener("resize", () => ED.fit());
{ const _load = ED.load; ED.load = async (...a) => { await _load(...a); if (S.mode === "face") fsDetect(); }; }
// рисование
ED.mask.addEventListener("pointerdown", e => { if (!ED.W || e.button !== 0) return; ED.mask.setPointerCapture(e.pointerId); ED.pushUndo(); ED.drawing = true; ED.last = ED.pos(e); ED.stroke(ED.last, ED.last); });
ED.mask.addEventListener("pointermove", e => {
  const r = ED.stack.getBoundingClientRect(), s = +$("#edSize").value;
  Object.assign(ED.cur.style, { display: "block", left: (e.clientX - r.left) + "px", top: (e.clientY - r.top) + "px", width: s + "px", height: s + "px" });
  if (!ED.drawing) return; const p = ED.pos(e); ED.stroke(ED.last, p); ED.last = p;
});
ED.mask.addEventListener("pointerup", () => { ED.drawing = false; });
ED.mask.addEventListener("pointerleave", () => { ED.cur.style.display = "none"; });
ED.wrap.addEventListener("wheel", e => { if (!ED.W || !e.altKey && !e.ctrlKey) return; e.preventDefault(); const s = $("#edSize"); s.value = Math.max(4, Math.min(300, +s.value - Math.sign(e.deltaY) * 6)); }, { passive: false });
const setTool = t => { ED.tool = t; $("#edBrush").classList.toggle("on", t === "brush"); $("#edEraser").classList.toggle("on", t === "eraser"); };
$("#edBrush").onclick = () => setTool("brush"); $("#edEraser").onclick = () => setTool("eraser");
$("#edUndo").onclick = () => { const d = ED.undo.pop(); if (d) ED.mask.getContext("2d").putImageData(d, 0, 0); };
$("#edClear").onclick = () => { if (!ED.W) return; ED.pushUndo(); ED.mask.getContext("2d").clearRect(0, 0, ED.W, ED.H); };
$("#edInvert").onclick = () => {
  if (!ED.W) return; ED.pushUndo();
  const ctx = ED.mask.getContext("2d"), d = ctx.getImageData(0, 0, ED.W, ED.H);
  for (let i = 0; i < d.data.length; i += 4) { d.data[i] = 255; d.data[i + 1] = 40; d.data[i + 2] = 40; d.data[i + 3] = d.data[i + 3] > 20 ? 0 : 255; }
  ctx.putImageData(d, 0, 0);
};
$("#edShowMask").onchange = e => ED.stack.classList.toggle("nomask", !e.target.checked);
document.addEventListener("keydown", e => {
  if (S.mode !== "edit" || e.target.matches("input,textarea,select")) return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); $("#edUndo").click(); }
  else if (e.key === "b" || e.key === "и") setTool("brush");
  else if (e.key === "e" || e.key === "у") setTool("eraser");
  else if (e.key === "[") $("#edSize").value = +$("#edSize").value - 8;
  else if (e.key === "]") $("#edSize").value = +$("#edSize").value + 8;
});
// «До» — удерживать
const showBefore = on => {
  const prev = ED.history[ED.history.length - 1]; if (!prev || !ED.W) return;
  const ctx = ED.img.getContext("2d");
  ctx.drawImage(on ? prev.el : ED.src.el, 0, 0, ED.W, ED.H);
};
["pointerdown"].forEach(ev => $("#edBefore").addEventListener(ev, () => showBefore(true)));
["pointerup", "pointerleave"].forEach(ev => $("#edBefore").addEventListener(ev, () => showBefore(false)));
$("#edBack").onclick = async () => {
  const prev = ED.history.pop(); if (!prev) return toast("Это самая первая версия");
  await ED.load(prev.name, prev.w, prev.h, false); toast("Вернулся к предыдущей версии");
};
// загрузка исходника
async function edUpload(files) {
  const f = [...files].find(f => f.type.startsWith("image/")); if (!f) return;
  const fd = new FormData(); fd.append("image", f, f.name || "paste.png");
  const r = await fetch("/api/input_upload", { method: "POST", body: fd }).then(r => r.json());
  if (r.names && r.names[0]) { ED.history = []; await ED.load(r.names[0]); }
}
$("#edOpen").onclick = () => $("#edFile").click();
$("#edFile").onchange = e => { edUpload(e.target.files); e.target.value = ""; };
ED.wrap.addEventListener("dragover", e => { e.preventDefault(); ED.wrap.classList.add("over"); });
ED.wrap.addEventListener("dragleave", () => ED.wrap.classList.remove("over"));
ED.wrap.addEventListener("drop", e => { e.preventDefault(); ED.wrap.classList.remove("over"); edUpload(e.dataTransfer.files); });
document.addEventListener("paste", e => {
  if ((S.mode !== "edit" && S.mode !== "face") || e.target.matches("input,textarea")) return;
  const f = [...(e.clipboardData?.files || [])]; if (!f.length) return;
  e.stopImmediatePropagation();
  if (S.mode === "face" && ED.src) fsDonorUpload(f); else edUpload(f);  // в замене лица: первая вставка — картинка, дальше — новое лицо
}, true);
async function edFromUrl(url, keep) {
  try { const r = await api("/api/edit/source", { url }); if (!keep) ED.history = []; await ED.load(r.name, r.w, r.h, keep); }
  catch (e) { toast(e.message); }
}
$("#edFromResult").onclick = () => {
  const src = $("#mainImg").getAttribute("src") || "";
  if (!src.startsWith("/output/")) return toast("Во вкладке «Генерация» сейчас нет готовой картинки");
  edFromUrl(src);
};
$("#toEditor").onclick = () => { const src = $("#mainImg").getAttribute("src") || ""; if (src.startsWith("/output/")) { setMode("edit"); edFromUrl(src); } };
$("#toFace").onclick = () => { const src = $("#mainImg").getAttribute("src") || ""; if (src.startsWith("/output/")) { setMode("face"); edFromUrl(src); } };
// параметры правки
function edParams() {
  const p = params();
  return {
    unet: p.unet, clip: p.clip, vae: p.vae, weight_dtype: p.weight_dtype, steps: p.steps, cfg: p.cfg, sampler: p.sampler,
    scheduler: p.scheduler, seed: p.seed, loras: p.loras, negative: p.negative, character: p.character, char_phrase: p.char_phrase, family: p.family,
    image: ED.src && ED.src.name, instruction: $("#edInstr").value.trim(), refine: $("#edRefine").checked,
    grow: +$("#edGrow").value, feather: +$("#edFeather").value, context: +$("#edContext").value,
    megapixels: +$("#edMp").value, denoise: +$("#edDenoise").value,
  };
}
async function edUploadMask() {
  if (!ED.hasMask()) return null;
  const fd = new FormData(); fd.append("image", await ED.maskBlob(), "mask.png");
  const r = await fetch("/api/input_upload", { method: "POST", body: fd }).then(r => r.json());
  return r.names && r.names[0];
}
$("#edRefineOnly").onclick = async () => {
  if (!ED.src) return toast("Сначала откройте картинку");
  if (!$("#edInstr").value.trim()) return toast("Напишите, что изменить");
  $("#edRefineOnly").disabled = true; ED.info("ИИ смотрит на картинку и формулирует инструкцию…");
  try {
    const mask = await edUploadMask();
    const r = await api("/api/edit/refine", { ...edParams(), mask });
    $("#edInstr").value = r.instruction; $("#edRefine").checked = false;
    ED.info((r.vision ? "ИИ посмотрел на картинку" : "ИИ без зрения — только перевёл/уточнил текст") + ". Инструкция подставлена в поле, галочка «Уточнить» снята — можно поправить текст и применить.");
  } catch (e) { ED.info(""); toast(e.message, 5000); }
  finally { $("#edRefineOnly").disabled = false; }
};
async function edRun() {
  if (ED.busy) return;
  if (!ED.src) return toast("Сначала откройте картинку");
  if (!$("#edInstr").value.trim()) return toast("Напишите, что изменить");
  ED.busy = true; S.stopReq = false; showErr("");  // блок сразу, до ответа сервера
  runBtns("#edRun", "#edStop", true);
  const hadMask = ED.hasMask();
  try {
    const mask = await edUploadMask();
    const body = { ...edParams(), mask };
    const msg = body.refine ? "ИИ уточняет инструкцию…" : "Отправляю…";
    ED.info(msg); progressMsg(msg);
    const r = await api("/api/edit", body);
    if (S.stopReq) { await stopNow(); ED.info("Остановлено"); return; }
    S.lastSeed = r.seed; saveUI(); showWarn(r.warnings);
    ED.info((r.refined ? "Инструкция для Qwen: " + r.instruction + "  ·  " : "") + (hadMask ? "правится только выделенная область" : "правится вся картинка"));
    await edPollJob(r, t => ED.info(t));
  } catch (e) { showErr(e.message); }
  finally {
    ED.busy = false; runBtns("#edRun", "#edStop", false);
    setTimeout(() => $("#progress").classList.add("hidden"), 800);
  }
}
/* «Стоп»: пока идёт работа, кнопка запуска превращается в «Стоп» на том же месте; есть и общий «■ Стоп» в строке прогресса */
function runBtns(run, stop, busy) {
  $(run).classList.toggle("hidden", busy); $(run).disabled = busy;
  $(stop).classList.toggle("hidden", !busy);
}
function progressMsg(msg) {
  $("#progress").classList.remove("hidden"); $("#pbar").style.width = "0"; $("#ptxt").textContent = msg;
}
async function stopNow() {
  S.stopReq = true;
  try { await api("/api/interrupt", {}); } catch { }
  $("#ptxt").textContent = "Останавливаю…";
}
$("#edRun").onclick = edRun;
/* ожидание результата правки / замены лица: прогресс, затем результат открывается в редакторе как новая версия */
async function edPollJob(r, info) {
  $("#progress").classList.remove("hidden");
  const t0 = Date.now();
  while (true) {
    await new Promise(res => setTimeout(res, 500));
    let j; try { j = await api("/api/job/" + r.id); } catch { continue; }
    const pct = j.max ? Math.round(100 * j.value / j.max) : 0;
    $("#pbar").style.width = (j.status === "done" ? 100 : pct) + "%";
    $("#ptxt").textContent = `${j.stage || ""}${j.max ? ` ${j.value}/${j.max}` : ""} · ${((Date.now() - t0) / 1000).toFixed(0)} с`;
    if (j.status === "done") {
      const url = j.images[0];
      showWarn(j.warnings);
      if (url) { await edFromUrl(url, true); rateShow(url); info("Готово · сохранено в output/images · удерживайте «👁 До», чтобы сравнить, «↩ Шаг назад» — отменить."); }
      return true;
    }
    if (j.status === "error" || j.status === "cancelled") { showErr(j.error || "Ошибка"); return false; }
  }
}
/* вырезка изменений: «до» и «после» → только изменённое на прозрачном фоне / маска */
const CUT = { method: null, res: null, n: 0, t: null };
function cutPair() {
  if (!ED.src || !ED.history.length) return null;
  const before = $("#cutBase").value === "first" ? ED.history[0] : ED.history[ED.history.length - 1];
  return { before: before.name, after: ED.src.name };
}
$("#edCut").onclick = async () => {
  if (!ED.src) return toast("Сначала откройте картинку");
  if (!ED.history.length) return toast("Сначала сделайте правку — вырезаются изменения между версиями «до» и «после»");
  $("#cutBase").disabled = ED.history.length < 2;
  if (ED.history.length < 2) $("#cutBase").value = "prev";
  openDlg("cutDlg");
  await cutMattUI();
  if (!CUT.picked) CUT.method = S.matting && S.matting.installed ? "ai" : "threshold";
  cutRun();
};
async function cutMattUI() {
  try { S.matting = await api("/api/matting"); } catch { S.matting = {}; }
  const m = S.matting || {}, box = $("#cutMatt"), ai = $("#cutMethod button[data-v=ai]");
  ai.disabled = !m.installed;
  if (m.installed) { box.classList.add("hidden"); return; }
  if (CUT.method === "ai") CUT.method = "threshold";
  box.classList.remove("hidden");
  box.innerHTML = m.installing ? `Скачиваю нейросеть для вырезки… ${m.progress ? m.progress + "%" : ""}` :
    `Способ «ИИ» вырезает изменённые предметы целиком (без него от тёмных предметов на тёмном фоне остаются одни контуры). ` +
    `<button class="small" id="cutMattGet">Скачать нейросеть (${esc(m.size || "171 МБ")})</button>` + (m.error ? `<br>${esc(m.error)}` : "");
  const b = $("#cutMattGet");
  if (b) b.onclick = async () => {
    try { await api("/api/matting/install", {}); } catch (e) { return toast(e.message); }
    for (let i = 0; i < 900; i++) {
      await cutMattUI();
      if (!S.matting.installing) break;
      await new Promise(r => setTimeout(r, 1000));
    }
    if (S.matting.installed) { CUT.method = "ai"; cutUIsync(); cutRun(); toast("Нейросеть для вырезки готова"); }
  };
  cutUIsync();
}
function cutUIsync() { $$("#cutMethod button").forEach(b => b.classList.toggle("on", b.dataset.v === CUT.method)); }
async function cutRun() {
  const pair = cutPair(); if (!pair) return;
  cutUIsync();
  const n = ++CUT.n;
  $("#cutWait").classList.remove("hidden"); $("#cutInfo").textContent = "";
  try {
    const r = await api("/api/edit/cutout", { ...pair, method: CUT.method, sens: +$("#cutSens").value / 100, fill: $("#cutFill").checked });
    if (n !== CUT.n) return;
    CUT.res = r;
    $("#cutImg").src = r.url;
    $("#cutInfo").textContent = `Изменено ${r.area}% площади · способ: ${r.method === "ai" ? "ИИ" : "по разнице"} · ${r.w}×${r.h} · сохранено в output/cutouts`;
  } catch (e) { if (n === CUT.n) { CUT.res = null; $("#cutImg").removeAttribute("src"); $("#cutInfo").textContent = e.message; } }
  finally { if (n === CUT.n) $("#cutWait").classList.add("hidden"); }
}
const cutSoon = () => { clearTimeout(CUT.t); CUT.t = setTimeout(cutRun, 300); };
$$("#cutMethod button").forEach(b => b.onclick = () => { if (b.disabled) return; CUT.method = b.dataset.v; CUT.picked = true; cutRun(); });
$("#cutSens").onchange = cutSoon; $("#cutFill").onchange = cutSoon; $("#cutBase").onchange = cutSoon;
$("#cutSave").onclick = () => {
  if (!CUT.res) return;
  const a = document.createElement("a"); a.href = CUT.res.url; a.download = decodeURIComponent(CUT.res.url.split("/").pop());
  document.body.appendChild(a); a.click(); a.remove();
};
$("#cutToMask").onclick = async () => {
  if (!CUT.res || !ED.W) return;
  const im = new Image(); im.src = CUT.res.mask_url + "?t=" + Date.now();
  await im.decode();
  const c = document.createElement("canvas"); c.width = ED.W; c.height = ED.H;
  const cx = c.getContext("2d"); cx.drawImage(im, 0, 0, ED.W, ED.H);
  const src = cx.getImageData(0, 0, ED.W, ED.H);
  ED.pushUndo();
  const ctx = ED.mask.getContext("2d"), out = ctx.createImageData(ED.W, ED.H);
  for (let i = 0; i < src.data.length; i += 4) if (src.data[i] > 127) { out.data[i] = 255; out.data[i + 1] = 40; out.data[i + 2] = 40; out.data[i + 3] = 255; }
  ctx.putImageData(out, 0, 0);
  $("#edShowMask").checked = true; ED.stack.classList.remove("nomask");
  $("#cutDlg").close();
  toast("Изменённая область — теперь маска. Следующая правка затронет только её (↶ — отменить)");
};
$("#edInstr").addEventListener("keydown", e => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); e.stopPropagation(); edRun(); } });
$("#edStop").onclick = stopNow;

/* ---------------- персонажи ---------------- */
S.chars = [];
async function loadChars() { try { S.chars = (await api("/api/chars")).items; } catch { S.chars = []; } renderCharSel(); if ($("#charDlg").open) renderCharGrid(); if (S.fs && S.mode === "face") fsFillChars(); }
function curChar() { return S.chars.find(c => c.id === S.character) || null; }
function renderCharSel() {
  const c = curChar(), box = $("#charSel");
  if (S.character && !c && S.chars.length) S.character = null;
  box.innerHTML = c
    ? `<div class="charPick">${c.photos[0] ? `<img src="${esc(c.photos[0])}" alt="">` : ""}<span class="nm" title="${esc(c.name)}">${esc(c.name)}<br><span class="muted tiny">${c.photos.length} фото</span></span><button class="small" id="charChange">Сменить</button><button class="icon" id="charOff" title="Без персонажа">✕</button></div>`
    : `<button class="small" id="charChange">🙂 Выбрать персонажа…</button>`;
  $("#charChange").onclick = openChars;
  const off = $("#charOff"); if (off) off.onclick = () => { S.character = null; renderCharSel(); saveUI(); };
  $("#charPhrase").closest("label").classList.toggle("hidden", !c);
}
$("#charManage").onclick = openChars;
function openChars() { openDlg("charDlg"); $("#charEdit").classList.add("hidden"); loadChars(); }
function renderCharGrid() {
  $("#charGrid").innerHTML = S.chars.length ? S.chars.map(c => `
    <div class="lc pick ${c.id === S.character ? "sel" : ""}" data-cid="${c.id}">
      <div class="ttl" title="${esc(c.name)}">${esc(c.name)}</div>
      <div class="pic">${c.photos[0] ? `<img src="${esc(c.photos[0])}" alt="">` : `<div class="ph">нет фото</div>`}
        <div class="ov tl">${c.id === S.character ? `<span class="num">выбран</span>` : ""}</div>
        <div class="ov br"><button data-cedit="${c.id}" title="Фото, имя, заметка">✎</button></div></div>
      <div class="info2"><span class="bm">${c.photos.length} фото${c.note ? " · " + esc(c.note) : ""}</span></div>
    </div>`).join("") : `<p class="muted">Персонажей пока нет. Впишите имя и выберите фото лица.</p>`;
  $$("#charGrid .lc").forEach(el => el.onclick = e => {
    const ed = e.target.closest("[data-cedit]"); if (ed) return openCharEdit(ed.dataset.cedit);
    S.character = el.dataset.cid; saveUI(); renderCharSel(); renderCharGrid(); toast("Персонаж выбран");
  });
}
function openCharEdit(id) {
  const c = S.chars.find(x => x.id === id); if (!c) return;
  const box = $("#charEdit"); box.classList.remove("hidden");
  box.innerHTML = `<h4>${esc(c.name)}</h4>
    <div class="grid2"><label>Имя<input id="ceName" value="${esc(c.name)}"></label><label>Заметка<input id="ceNote" value="${esc(c.note)}"></label></div>
    <div class="photos">${c.photos.map((u, i) => `<div class="ph"><img src="${esc(u)}"><button data-rm="${i}" title="Удалить фото">✕</button></div>`).join("")}</div>
    <div class="rowbtns"><button class="small primary" id="ceSave">Сохранить</button><button class="small" id="ceAdd" ${c.photos.length >= 4 ? "disabled" : ""}>+ Добавить фото</button><button class="small danger" id="ceDel">Удалить персонажа</button></div>`;
  $("#ceSave").onclick = async () => { await api(`/api/chars/${id}/edit`, { name: $("#ceName").value, note: $("#ceNote").value }); await loadChars(); openCharEdit(id); toast("Сохранено"); };
  $$("[data-rm]", box).forEach(b => b.onclick = async () => { await api(`/api/chars/${id}/edit`, { remove_photo: +b.dataset.rm }); await loadChars(); openCharEdit(id); });
  $("#ceAdd").onclick = () => { S.charAddTo = id; $("#charFiles").click(); };
  $("#ceDel").onclick = async () => { if (!confirm("Удалить персонажа «" + c.name + "» и его фото?")) return; await api(`/api/chars/${id}/delete`, {}); if (S.character === id) S.character = null; box.classList.add("hidden"); saveUI(); loadChars(); };
  box.scrollIntoView({ block: "nearest" });
}
$("#newCharBtn").onclick = () => { S.charAddTo = null; $("#charFiles").click(); };
$("#charFiles").onchange = async e => {
  const files = [...e.target.files].slice(0, 4); e.target.value = "";
  if (!files.length) return;
  const crops = [];
  for (const f of files) {
    const url = URL.createObjectURL(f);
    const r = await cropFace(url, `Выделите лицо (${crops.length + 1} из ${files.length})`);
    URL.revokeObjectURL(url);
    if (r === undefined) return; // отмена
    crops.push(r.crop);
  }
  const fd = new FormData();
  if (S.charAddTo) fd.append("id", S.charAddTo); else fd.append("name", $("#newCharName").value.trim() || "Персонаж");
  fd.append("crops", JSON.stringify(crops));
  files.forEach(f => fd.append("file", f, f.name));
  const r = await fetch("/api/chars", { method: "POST", body: fd }); const j = await r.json();
  if (!r.ok) return toast(j.error || "Ошибка", 4000);
  if (!S.charAddTo) { S.character = j.id; $("#newCharName").value = ""; saveUI(); toast("Персонаж создан и выбран"); }
  await loadChars(); if (S.charAddTo) openCharEdit(S.charAddTo);
};
// окно рамки: возвращает {crop:[x0,y0,x1,y1]|null, target} или undefined при отмене
function cropFace(url, title, withTarget) {
  return new Promise(resolve => {
    $("#cropTitle").textContent = title || "Выделите лицо";
    const img = $("#cropImg"), box = $("#cropBox"), wrap = $("#cropWrap");
    let sel = null, start = null, done = false;
    box.classList.add("hidden");
    $("#cropTarget").innerHTML = withTarget ? `<select id="cropTo"><option value="">➕ Новый персонаж</option>${S.chars.map(c => `<option value="${c.id}" ${c.id === S.character ? "selected" : ""} ${c.photos.length >= 4 ? "disabled" : ""}>${esc(c.name)} (${c.photos.length}/4)</option>`).join("")}</select><input id="cropName" placeholder="Имя нового персонажа">` : "";
    const syncName = () => { const n = $("#cropName"); if (n) n.classList.toggle("hidden", !!$("#cropTo").value); };
    if (withTarget) { $("#cropTo").onchange = syncName; syncName(); }
    img.src = url;
    const pt = e => { const r = img.getBoundingClientRect(); return { x: Math.max(0, Math.min(r.width, e.clientX - r.left)), y: Math.max(0, Math.min(r.height, e.clientY - r.top)) }; };
    img.draggable = false; img.ondragstart = e => e.preventDefault();
    wrap.onpointerdown = e => { e.preventDefault(); wrap.setPointerCapture(e.pointerId); start = pt(e); sel = null; };
    wrap.onpointermove = e => {
      if (!start) return; const p = pt(e);
      sel = { x: Math.min(start.x, p.x), y: Math.min(start.y, p.y), w: Math.abs(p.x - start.x), h: Math.abs(p.y - start.y) };
      Object.assign(box.style, { left: sel.x + "px", top: sel.y + "px", width: sel.w + "px", height: sel.h + "px" }); box.classList.remove("hidden");
    };
    wrap.onpointerup = () => { start = null; };
    const finish = v => { if (done) return; done = true; $("#cropDlg").close(); resolve(v); };
    const target = () => withTarget ? { id: $("#cropTo").value, name: ($("#cropName")?.value || "").trim() } : null;
    $("#cropOk").onclick = () => {
      if (!sel || sel.w < 10 || sel.h < 10) return toast("Протяните рамку или нажмите «Всё фото»");
      const k = img.naturalWidth / img.getBoundingClientRect().width;
      finish({ crop: [sel.x * k, sel.y * k, (sel.x + sel.w) * k, (sel.y + sel.h) * k], target: target() });
    };
    $("#cropAll").onclick = () => finish({ crop: null, target: target() });
    $("#cropCancel").onclick = () => finish(undefined);
    $("#cropDlg").onclose = () => finish(undefined);
    openDlg("cropDlg");
  });
}
$("#saveFace").onclick = async () => {
  const url = ($("#mainImg").getAttribute("src") || "").split("?")[0];
  if (!url.startsWith("/output/")) return;
  if (!S.chars.length) await loadChars();
  const r = await cropFace(url, "Сохранить лицо как персонажа", true);
  if (!r) return;
  try {
    const c = await api("/api/chars", { url, crop: r.crop, id: r.target.id || undefined, name: r.target.name || "Персонаж" });
    if (!r.target.id) S.character = c.id;
    saveUI(); await loadChars(); toast(r.target.id ? "Фото добавлено персонажу" : "Персонаж создан и выбран");
  } catch (e) { toast(e.message, 4000); }
};

/* ---------------- переключение нейросети ---------------- */
function famWarn() {
  const bad = (S.family === "qwen" && S.qwenSupported === false) || (S.family === "flux3" && S.flux3Supported === false);
  $("#famWarn").innerHTML = bad ? `Движок устарел и не знает ${FAM[S.family].name}. Закройте программу (чёрное окно) и запустите <b>update_engine.bat</b> в папке программы — модели и настройки не пострадают.` : "";
  $("#famWarn").classList.toggle("hidden", !bad);
}
function captureFam() {
  const o = {}; FAM_KEYS.forEach(k => o[k] = $("#" + k).value);
  o.loras = S.loras; o.character = S.character || null; o.charPhrase = $("#charPhrase").checked;
  o.useText = $("#useText").checked; o.useImage = $("#useImage").checked; o.imageMode = S.imageMode; o.images = S.images;
  o.ltxTwo = $("#ltxTwo").checked; o.turbo = $("#turbo").checked; o.refMax = $("#refMax").checked; o.frames = S.frames; o.mm = S.mm; o.refs = S.refs; o.ratio = S.ratio || null;
  return o;
}
function applyFamUI() {
  const kind = kindOf(S.family);
  if (kind === "video") S.lastVid = S.family; else S.lastImg = S.family;
  $$("#famSwitch button[data-kind]").forEach(b => b.classList.toggle("on", b.dataset.kind === kind));
  document.body.dataset.family = S.family; document.body.dataset.kind = kind;
  $("#subFam").innerHTML = KINDS[kind].map(f => `<button data-sf="${f}" class="${f === S.family ? "on" : ""}">${esc(FAM[f].name)}</button>`).join("");
  $$("#subFam [data-sf]").forEach(b => b.onclick = () => setFamily(b.dataset.sf));
  $("#duration").max = S.family === "ltx" ? 12 : 15; $("#duration").min = S.family === "ltx" ? 2 : 3;
  if (+$("#duration").value > +$("#duration").max) $("#duration").value = $("#duration").max;
  durShow(); ltxShow();
  fillSelect($("#scheduler"), SCHEDULERS.filter(x => x[0] !== "flux2"));
  renderPresets(); famWarn();
  $("#edInstr").placeholder = "Что изменить? Например: замени футболку на чёрную кожаную куртку. Картинка для правки — <image1>, фото персонажа — <image2>. Ctrl+Enter — применить";
  $("#prompt").placeholder = isVidFam()
    ? `Что происходит в видео: действие, камера, звук, реплики (можно по-русски) — «✨ Улучшить» сделает из этого промпт по правилам ${FAM[S.family].name}. Ctrl+Enter — генерировать`
    : "Опишите изображение… (Ctrl+Enter — генерировать)";
  $(".modeTabs [data-mode=gen]").textContent = isVidFam() ? "🎬 Видео" : "🎨 Генерация";
  $("#emptyMsg .big").textContent = isVidFam() ? "Здесь появится видео" : "Здесь появится изображение";
  const cb = $("#civBase");
  if (cb) cb.value = ARCH_OF[S.family] === "qwen21" ? "qwen21" : ARCH_OF[S.family];
}
function setFamily(f, quiet) {
  if (f === S.family) return;
  if (!FAM[f]) return;
  S.fam[S.family] = captureFam();          // запомнить «лист» текущей нейросети
  S.family = f;
  const saved = S.fam[f];
  // своя страница: сохранённая или чистая по умолчанию (пустой промпт, без LoRA и персонажа)
  const st = { ...FAM[f].defaults, loras: [], character: null, charPhrase: true, useText: true, useImage: false,
               imageMode: "reference", images: [], ...(saved || {}) };
  if (S.mode && S.mode !== "gen" && f !== "qwen" && typeof setMode === "function") {   // редактор и замена лица — только Qwen
    S.mode = "gen"; setTimeout(() => { setMode("gen"); toast(`Редактор и замена лица работают на Qwen‑Image 2.1 — ${FAM[f].name} открыта в «Генерации»`, 4500); }, 0);
  }
  applyFamUI();
  FAM_KEYS.forEach(k => { if (st[k] !== undefined && !["unet", "clip", "vae", "avae", "unetRef", "upscaler"].includes(k)) $("#" + k).value = st[k]; });
  S._pending = { unet: st.unet, clip: st.clip, vae: st.vae, avae: st.avae, unetRef: st.unetRef, upscaler: st.upscaler };
  $("#ltxTwo").checked = st.ltxTwo !== false; ltxShow(); durShow();
  $("#turbo").checked = st.turbo !== false; $("#refMax").checked = !!st.refMax; S.frames = st.frames || { first: null, last: null };
  S.ratio = st.ratio || null; S.mm = st.mm === "ref" ? "ref" : "frames"; S.refs = st.refs || []; document.body.dataset.mm = S.mm; renderFrames(); renderRefs();
  S.loras = (st.loras || []).map(l => ({ ...l })); syncTurbo(false);
  S.character = st.character || null; $("#charPhrase").checked = st.charPhrase !== false;
  $("#useText").checked = st.useText !== false; $("#useImage").checked = !!st.useImage;
  S.imageMode = st.imageMode || "reference"; S.images = st.images || [];
  $("#aiOut").classList.add("hidden"); showErr("");
  $("#denoiseV").textContent = (+$("#denoise").value).toFixed(2);
  updateModelSelects(); renderLoras(); renderCharSel(); renderThumbs(); updateMode(); updateMp(); checkPair(); checkTriggers(); saveUI();
  if (!quiet) toast(saved ? `${FAM[f].name}: ваши последние настройки` : `${FAM[f].name}: новый чистый лист`);
}
$$("#famSwitch button[data-kind]").forEach(b => b.onclick = () => {
  if (window.setView) setView("gen");
  const k = b.dataset.kind; if (kindOf(S.family) === k) return;
  setFamily(k === "video" ? (S.lastVid || "minimax") : (S.lastImg || "qwen"));
});
applyFamUI();

/* ---------------- start ---------------- */
document.body.dataset.family = S.family; document.body.dataset.kind = kindOf(S.family);
document.body.dataset.mm = S.mm; renderRefs(); durShow();
renderThumbs(); updateMode(); updateMp(); renderLoras();
refreshState();
setInterval(() => { if (!document.hidden) refreshState(); }, 3000);
$("#useText").addEventListener("change", checkTriggers);
S._restoreMode = (() => { try { return localStorage.getItem("flux2mode"); } catch { return null; } })();
$("#charPhrase").addEventListener("change", saveUI);
loadChars();

/* ---------------- замена лица ---------------- */
S.fs = { faces: null, sel: 0, src: "photo", donor: null, donorFaces: null, donorSel: 0, area: "face", detector: null, char: null };
const FS_SRC_KEY = "flux2fs";
function fsInfo(t) { $("#fsInfo").textContent = t || ""; }
async function fsState() {
  try { const r = await api("/api/face/state"); S.fs.detector = r.installed; S.fs.detErr = r.error; } catch { S.fs.detector = false; }
  return S.fs.detector;
}
async function fsEnter() {
  await fsState(); fsFillChars(); fsRenderDet();
  if (ED.src && (!S.fs.faces || S.fs.facesFor !== ED.src.name)) fsDetect();
  else fsRenderBoxes();
}
function fsRenderDet() {
  const box = $("#fsDet");
  if (S.fs.detector === false) {
    box.innerHTML = `Автопоиск лиц: нужен детектор лиц (0.4 МБ, один раз). <button class="small primary" id="fsGetDet">Скачать</button><br><span class="muted tiny">Можно и без него: обведите лицо кистью.</span>`;
    $("#fsGetDet").onclick = async () => {
      $("#fsGetDet").disabled = true; $("#fsGetDet").textContent = "Скачиваю…";
      try { await api("/api/face/install", {}); toast("Детектор лиц установлен"); await fsState(); fsRenderDet(); fsDetect(); if (S.fs.donor) fsDonorDetect(); }
      catch (e) { toast(e.message, 7000); fsRenderDet(); }
    };
    return;
  }
  const f = S.fs.faces, manual = ED.W && ED.hasMask();
  if (!ED.src) box.innerHTML = "1. Откройте картинку (📂, перетащите, Ctrl+V или «⤵ Из генерации»). 2. Добавьте фото нового лица. 3. «Заменить лицо».";
  else if (manual) box.innerHTML = "✓ Область задана вашей маской — лицо будет заменено внутри неё. <button class='link' id='fsClearMask'>убрать маску</button>";
  else if (!f) box.innerHTML = `<span class="spin"></span> Ищу лица…`;
  else if (!f.length) box.innerHTML = "Лицо не найдено автоматически — обведите его кистью (🖌) прямо на картинке.";
  else box.innerHTML = f.length === 1 ? "✓ Найдено лицо — оно и будет заменено. Можно уточнить область кистью." : `✓ Найдено лиц: ${f.length}. Нажмите на рамку, чтобы выбрать, какое заменить (сейчас №${S.fs.sel + 1}).`;
  const cm = $("#fsClearMask"); if (cm) cm.onclick = () => { $("#edClear").click(); fsRenderDet(); fsRenderBoxes(); };
}
async function fsDetect() {
  if (!ED.src) return fsRenderDet();
  S.fs.faces = null; S.fs.sel = 0; S.fs.facesFor = ED.src.name; fsRenderBoxes();
  if (S.fs.detector === null) await fsState();
  if (!S.fs.detector) { S.fs.faces = null; fsRenderDet(); return; }
  fsRenderDet();
  try { const r = await api("/api/face/detect", { image: ED.src.name }); if (S.fs.facesFor === ED.src.name) S.fs.faces = r.faces; }
  catch (e) { S.fs.faces = []; toast(e.message, 5000); }
  fsRenderBoxes(); fsRenderDet();
}
function fsRenderBoxes() {
  const el = $("#fsBoxes"), f = S.fs.faces || [];
  el.innerHTML = ED.W ? f.map((x, i) => { const [a, b, c, d] = x.box; return `<div class="fbox ${i === S.fs.sel ? "on" : ""}" data-fi="${i}" style="left:${100 * a / ED.W}%;top:${100 * b / ED.H}%;width:${100 * (c - a) / ED.W}%;height:${100 * (d - b) / ED.H}%" title="Заменить это лицо"><span>${i + 1}</span></div>`; }).join("") : "";
  el.classList.toggle("dim", !!(ED.W && ED.hasMask()));
  $$(".fbox", el).forEach(b => b.onclick = e => { e.stopPropagation(); S.fs.sel = +b.dataset.fi; fsRenderBoxes(); fsRenderDet(); });
}
ED.mask.addEventListener("pointerup", () => { if (S.mode === "face") { fsRenderBoxes(); fsRenderDet(); } });
["#edClear", "#edUndo", "#edInvert"].forEach(s => $(s).addEventListener("click", () => { if (S.mode === "face") setTimeout(() => { fsRenderBoxes(); fsRenderDet(); }, 0); }));
// источник нового лица
$$("#fsSrcSeg button").forEach(b => b.onclick = () => { S.fs.src = b.dataset.v; fsSrcUI(); try { localStorage.setItem(FS_SRC_KEY, S.fs.src); } catch { } });
function fsSrcUI() {
  $$("#fsSrcSeg button").forEach(b => b.classList.toggle("on", b.dataset.v === S.fs.src));
  $("#fsPhotoBox").classList.toggle("hidden", S.fs.src !== "photo"); $("#fsCharBox").classList.toggle("hidden", S.fs.src !== "char");
  if (S.fs.src === "char") fsFillChars();
}
try { const v = localStorage.getItem(FS_SRC_KEY); if (v === "char" || v === "photo") S.fs.src = v; } catch { }
fsSrcUI();
function fsFillChars() {
  const sel = $("#fsChar"), cur = S.fs.char || sel.value || S.character || "";
  sel.innerHTML = S.chars.length ? S.chars.map(c => `<option value="${c.id}">${esc(c.name)} (${c.photos.length} фото)</option>`).join("") : `<option value="">— персонажей нет: создайте во «Все персонажи» —</option>`;
  if (cur && S.chars.some(c => c.id === cur)) sel.value = cur;
  S.fs.char = sel.value || null; fsCharPrev();
}
function fsCharPrev() {
  const c = S.chars.find(x => x.id === S.fs.char);
  $("#fsCharPrev").innerHTML = c ? `<div>${c.photos.slice(0, 3).map(u => `<img src="${esc(u)}" style="max-height:120px;margin:2px">`).join("")}</div><div class="dnote">Все фото персонажа подаются модели как образец лица</div>` : `<span class="muted">Нет персонажа</span>`;
}
$("#fsChar").onchange = () => { S.fs.char = $("#fsChar").value || null; fsCharPrev(); };
dropzone($("#fsDrop"), $("#fsFile"), files => fsDonorUpload(files));
async function fsDonorUpload(files) {
  const f = [...files].find(f => f.type.startsWith("image/")); if (!f) return;
  const fd = new FormData(); fd.append("image", f, f.name || "face.png");
  try {
    const r = await fetch("/api/input_upload", { method: "POST", body: fd }).then(r => r.json());
    if (!r.names || !r.names[0]) return;
    S.fs.donor = r.names[0]; S.fs.donorFaces = null; S.fs.donorSel = 0; S.fs.src = "photo"; fsSrcUI();
    fsDonorRender(); fsDonorDetect();
  } catch (e) { toast("Не удалось загрузить фото: " + e.message); }
}
async function fsDonorDetect() {
  if (!S.fs.donor || !S.fs.detector) return fsDonorRender();
  try { const r = await api("/api/face/detect", { image: S.fs.donor }); S.fs.donorFaces = r.faces; S.fs.donorWH = [r.w, r.h]; }
  catch { S.fs.donorFaces = []; }
  fsDonorRender();
}
function fsDonorRender() {
  const v = $("#fsDonorView");
  if (!S.fs.donor) { v.innerHTML = `<span class="muted">Перетащите фото нового лица<br>или <u>выберите файл</u></span>`; return; }
  const f = S.fs.donorFaces, wh = S.fs.donorWH;
  const box = f && f.length && wh ? (() => { const [a, b, c, d] = f[S.fs.donorSel].box; return `<div class="dbox" style="left:${100 * a / wh[0]}%;top:${100 * b / wh[1]}%;width:${100 * (c - a) / wh[0]}%;height:${100 * (d - b) / wh[1]}%"></div>`; })() : "";
  const note = !S.fs.detector ? "Детектор не установлен — модели будет подано всё фото" : f === null ? "ищу лицо…" : !f.length ? "⚠ Лицо не найдено — будет подано всё фото" : f.length > 1 ? `Лиц на фото: ${f.length} · нажмите на фото, чтобы выбрать другое (№${S.fs.donorSel + 1})` : "✓ Лицо найдено — модели пойдёт вырезка головы";
  v.innerHTML = `<div><div class="dwrap"><img src="/input/${encodeURIComponent(S.fs.donor)}" alt="">${box}</div><div class="dnote">${note}</div></div><button class="dx" title="Убрать фото">✕</button>`;
  $(".dx", v).onclick = e => { e.stopPropagation(); S.fs.donor = null; fsDonorRender(); };
  $(".dwrap", v).onclick = e => { e.stopPropagation(); if (f && f.length > 1) { S.fs.donorSel = (S.fs.donorSel + 1) % f.length; fsDonorRender(); } else $("#fsFile").click(); };
}
$$("#fsArea button").forEach(b => b.onclick = () => { S.fs.area = b.dataset.v; $$("#fsArea button").forEach(x => x.classList.toggle("on", x === b)); });
$("#fsStr").oninput = () => { $("#fsStrV").textContent = (+$("#fsStr").value).toFixed(2); };
async function fsRun() {
  if (ED.busy) return;
  if (!ED.src) return toast("Сначала откройте картинку, на которой меняем лицо");
  const useChar = S.fs.src === "char";
  if (useChar && !S.fs.char) return toast("Выберите персонажа или переключитесь на «Фото лица»");
  if (!useChar && !S.fs.donor) return toast("Добавьте фото нового лица (или выберите персонажа)");
  const manual = ED.hasMask();
  if (!manual && !(S.fs.faces && S.fs.faces.length) && !S.fs.detector) return toast("Обведите лицо кистью или скачайте детектор лиц", 4000);
  if (!manual && S.fs.faces && !S.fs.faces.length) return toast("Лицо не найдено автоматически — обведите его кистью", 4000);
  ED.busy = true; S.stopReq = false; showErr(""); runBtns("#fsRun", "#fsStop", true);
  try {
    const p = params();
    const body = {
      unet: p.unet, clip: p.clip, vae: p.vae, weight_dtype: p.weight_dtype, steps: p.steps, cfg: p.cfg, sampler: p.sampler,
      scheduler: p.scheduler, seed: p.seed, loras: p.loras, negative: p.negative, family: p.family,
      image: ED.src.name, mask: manual ? await edUploadMask() : null,
      face: !manual && S.fs.faces && S.fs.faces.length ? S.fs.faces[S.fs.sel].box : null,
      character: useChar ? S.fs.char : null, donor: useChar ? null : S.fs.donor, donor_face: S.fs.donorSel,
      area: S.fs.area, keep_expr: $("#fsExpr").checked, strength: +$("#fsStr").value, extra: $("#fsExtra").value.trim(),
      megapixels: +$("#fsMp").value, context: +$("#fsCtx").value,
    };
    fsInfo("Отправляю…"); progressMsg("Отправляю…");
    const r = await api("/api/faceswap", body);
    if (S.stopReq) { await stopNow(); fsInfo("Остановлено"); return; }
    S.lastSeed = r.seed; saveUI(); showWarn(r.warnings);
    fsInfo("Меняю лицо…");
    await edPollJob(r, fsInfo);
  } catch (e) { showErr(e.message); fsInfo(""); }
  finally { ED.busy = false; runBtns("#fsRun", "#fsStop", false); setTimeout(() => $("#progress").classList.add("hidden"), 800); }
}
$("#fsRun").onclick = fsRun;
$("#fsStop").onclick = stopNow;
$("#fsExtra").addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); fsRun(); } });
if (S._restoreMode === "edit" || S._restoreMode === "face") setMode(S._restoreMode);

/* ---------------- оценка результата ---------------- */
S.rate = { url: null, entry: null, tags: { good: [], bad: [] } };
api("/api/lab/tags").then(t => { S.rate.tags = t; }).catch(() => { });
async function rateShow(url) {
  url = (url || "").split("?")[0];
  if (!url.startsWith("/output/")) return;
  S.rate.url = url; S.rate.entry = null;
  $("#rateBar").classList.remove("hidden");
  autoShow(url);
  $("#rateComment").value = ""; $("#rateState").textContent = "";
  rateRender();
  try { const r = await api("/api/lab/entry?image=" + encodeURIComponent(url)); if (S.rate.url === url) { S.rate.entry = r.entry; S.rate.params = r.params; $("#rateComment").value = r.entry?.comment || ""; rateRender(); } } catch { }
}
function rateRender() {
  const e = S.rate.entry, v = e?.rating || 0, sel = new Set(e?.tags || []);
  $$("#rateStars button").forEach(b => b.classList.toggle("on", +b.dataset.star <= v));
  const chip = (t, k) => `<button class="chip ${k} ${sel.has(t) ? "on" : ""}" data-rt="${esc(t)}">${esc(t)}</button>`;
  $("#rateTags").innerHTML = S.rate.tags.good.map(t => chip(t, "good")).join("") + `<span class="sep"></span>` + S.rate.tags.bad.map(t => chip(t, "bad")).join("");
  $$("#rateTags [data-rt]").forEach(b => b.onclick = () => {
    if (!S.rate.entry) return toast("Сначала поставьте звёзды");
    const t = b.dataset.rt, cur = new Set(S.rate.entry.tags || []); cur.has(t) ? cur.delete(t) : cur.add(t);
    rateSave({ tags: [...cur] });
  });
  $("#rateState").textContent = e ? `✓ в журнале` : "";
}
async function rateSave(body, url) {
  url = url || S.rate.url; if (!url) return;
  try {
    const e = await api("/api/lab/rate", { image: url, family: S.family, params: params(), ...body });
    if (S.rate.url === url) { S.rate.entry = e; rateRender(); $("#rateState").textContent = "✓ сохранено в журнал"; }
  } catch (e) { toast(e.message, 4000); }
}
$$("#rateStars button").forEach(b => b.onclick = () => rateSave({ rating: +b.dataset.star, comment: $("#rateComment").value.trim() }));
$("#rateComment").addEventListener("change", () => { if (S.rate.entry) rateSave({ comment: $("#rateComment").value.trim() }); else if ($("#rateComment").value.trim()) toast("Комментарий сохранится вместе с оценкой — поставьте звёзды"); });
$("#rateComment").addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); e.target.blur(); } });
document.addEventListener("keydown", e => {  // цифры 1–5 — быстрая оценка (когда фокус не в поле ввода)
  if (!S.rate.url || e.ctrlKey || e.metaKey || e.altKey || e.target.closest("input,textarea,select,dialog")) return;
  if (/^[1-5]$/.test(e.key) && !$("#rateBar").classList.contains("hidden")) { rateSave({ rating: +e.key, comment: $("#rateComment").value.trim() }); toast("Оценка " + e.key); }
});
$("#rateLab").onclick = () => openLab("chat");

/* ---------------- мастерская ---------------- */
S.lab = { tab: "chat", msgs: [], url: null };
try { const x = JSON.parse(localStorage.getItem("flux2lab") || "null"); if (x) { S.lab.msgs = x.msgs || []; S.lab.url = x.url || null; } } catch { }
const labSave = () => { try { localStorage.setItem("flux2lab", JSON.stringify({ msgs: S.lab.msgs.slice(-40), url: S.lab.url })); } catch { } };
$("#labBtn").onclick = () => openLab();
$$("#labTabs button").forEach(b => b.onclick = () => labTab(b.dataset.lt));
function labTab(t) {
  S.lab.tab = t;
  $$("#labTabs button").forEach(b => b.classList.toggle("on", b.dataset.lt === t));
  $("#labChat").classList.toggle("hidden", t !== "chat"); $("#labRules").classList.toggle("hidden", t !== "rules"); $("#labStats").classList.toggle("hidden", t !== "stats");
  if (t === "rules") labRules(); if (t === "stats") { labStats(); labDevices(); autoPanel(); }
}
async function openLab(tab) {
  openDlg("labDlg");
  if (S.rate.url && S.rate.url !== S.lab.url) {   // новая картинка — новый разговор о ней
    if (S.lab.msgs.length) S.lab.msgs.push({ role: "sys", content: "— новая картинка —" });
    S.lab.url = S.rate.url; labSave();
  }
  labTab(tab || S.lab.tab); labSide(); labRender(); labRulesCount();
}
async function labSide() {
  const url = S.lab.url;
  if (!url) { $("#labImg").innerHTML = `<span class="muted tiny">Откройте или сгенерируйте картинку — помощник обсудит её</span>`; $("#labParams").textContent = ""; $("#labStars").innerHTML = ""; return; }
  $("#labImg").innerHTML = `<img src="${esc(url)}" alt="">`;
  let r = {}; try { r = await api("/api/lab/entry?image=" + encodeURIComponent(url)); } catch { }
  const e = r.entry, p = r.params || {};
  $("#labStars").innerHTML = [1, 2, 3, 4, 5].map(i => `<button data-ls="${i}" class="${e && e.rating >= i ? "on" : ""}">★</button>`).join("") + (e ? ` <span class="muted tiny">${e.tags.join(", ")}</span>` : ` <span class="muted tiny">не оценена</span>`);
  $$("#labStars [data-ls]").forEach(b => b.onclick = async () => { await rateSave({ rating: +b.dataset.ls }, url); labSide(); });
  $("#labExclBox").classList.toggle("hidden", !e); $("#labExcl").checked = !!(e && e.excluded);
  $("#labExcl").onchange = async () => { await api("/api/lab/exclude", { image: url, value: $("#labExcl").checked }).catch(x => toast(x.message)); toast($("#labExcl").checked ? "Картинка не учитывается в статистике" : "Картинка снова учитывается"); };
  $("#labAutoExcl").textContent = e && e.auto_excluded ? "⚠ Не учитывается автоматически: " + e.auto_excluded : "";
  const loras = (p.loras || []).map(l => `${stripExt(l.name)} ×${l.strength}`).join(", ") || "без LoRA";
  $("#labParams").textContent = p.family ? `${FAM[p.family]?.name || p.family} · ${stripExt(String(p.unet || ""))}\nLoRA: ${loras}\n${p.width}×${p.height} · шаги ${p.steps} · CFG ${p.cfg} · ${p.sampler}/${p.scheduler}\nseed ${p.seed}${p.prompt_src === "helper" ? "\nпромпт от помощника" : ""}` : "параметры не найдены";
  $("#labCtx").textContent = "Помощник автоматически видит: настройки программы, параметры этой картинки, вашу оценку и метки, выжимки LoRA, принятые правила и сводку журнала" + (e ? "" : ". Поставьте оценку слева — она тоже попадёт в журнал.");
}
function labRender() {
  const log = $("#labLog");
  log.innerHTML = S.lab.msgs.length ? S.lab.msgs.map((m, i) => {
    if (m.role === "sys") return `<div class="muted tiny" style="text-align:center">${esc(m.content)}</div>`;
    if (m.role === "user") return `<div class="msg user">${esc(m.content)}</div>`;
    const rules = (m.rules || []).map((r, j) => `<div class="ruleProp">📌 ${esc(r)}${m.done && m.done[j] ? ` <span class="muted tiny">— ${m.done[j]}</span>` : ` <button class="small primary" data-ra="${i}:${j}">Принять</button><button class="small" data-rr="${i}:${j}">Отклонить</button>`}</div>`).join("");
    return `<div class="msg assistant">${esc(m.content.replace(/^\s*[-*]?\s*ПРАВИЛО\s*:.*$/gmi, "").trim())}${rules}</div>`;
  }).join("") : `<p class="muted">Расскажите, что получилось хорошо и что плохо — помощник предложит, что поменять, и оформит выводы в правила («📌 Принять» — правило попадёт в свод).</p>`;
  $$("[data-ra],[data-rr]", log).forEach(b => b.onclick = async () => {
    const [i, j] = (b.dataset.ra || b.dataset.rr).split(":").map(Number), m = S.lab.msgs[i], ok = !!b.dataset.ra;
    await api("/api/lab/rules", { action: "add", text: m.rules[j], family: S.family, status: ok ? "accepted" : "rejected", source: "chat", evidence: [m.entry_id] }).catch(e => toast(e.message));
    m.done = m.done || {}; m.done[j] = ok ? "принято в свод" : "отклонено"; labSave(); labRender(); labRulesCount();
  });
  log.scrollTop = log.scrollHeight;
}
async function labSend() {
  const t = $("#labText").value.trim(); if (!t || S.lab.busy) return;
  S.lab.msgs.push({ role: "user", content: t }); $("#labText").value = ""; labRender();
  S.lab.busy = true; $("#labSend").disabled = true;
  const w = document.createElement("div"); w.className = "msg assistant"; w.innerHTML = `<span class="spin"></span>смотрю и думаю…`; $("#labLog").appendChild(w); $("#labLog").scrollTop = 1e9;
  try {
    const hist = []; for (const m of S.lab.msgs) { if (m.role === "sys") hist.length = 0; else hist.push({ role: m.role, content: m.content }); }
    const r = await api("/api/lab/chat", { messages: hist, image: S.lab.url, family: S.family, current: params() });
    S.lab.msgs.push({ role: "assistant", content: r.reply, rules: r.rules, entry_id: r.entry_id });
  } catch (e) { S.lab.msgs.push({ role: "assistant", content: "⚠ " + e.message, rules: [] }); }
  finally { S.lab.busy = false; $("#labSend").disabled = false; labSave(); labRender(); }
}
$("#labSend").onclick = labSend;
$("#labText").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); labSend(); } });
$("#labNew").onclick = () => { S.lab.msgs = []; labSave(); labRender(); };
// свод правил
const labFam = sel => $(sel).value === "all" ? "all" : S.family;
async function labRulesCount() { try { const r = await api("/api/lab/rules?family=" + S.family); const a = r.items.filter(x => x.status === "accepted").length, p = r.items.filter(x => x.status === "proposed").length; $("#labRulesN").textContent = `(${a}${p ? " + " + p + " на проверке" : ""})`; } catch { } }
async function labRules() {
  let r; try { r = await api("/api/lab/rules?family=" + labFam("#labRulesFam")); } catch (e) { return toast(e.message); }
  $("#labInPrompt").checked = r.in_prompt;
  const order = { proposed: 0, accepted: 1, rejected: 2 };
  const items = r.items.sort((a, b) => order[a.status] - order[b.status] || b.id - a.id);
  $("#labRuleList").innerHTML = items.length ? items.map(x => `<div class="rule ${x.status}" data-rid="${x.id}">
      <span class="st">${{ accepted: "принято", proposed: "на проверке", rejected: "отклонено" }[x.status]}</span>
      <div class="t"><div data-rtext>${esc(x.text)}</div><div class="m">${{ chat: "из разговора", stats: "из статистики", manual: "добавлено вами" }[x.source] || x.source} · ${x.time.slice(0, 10)} · ${x.family === "all" ? "все нейросети" : FAM[x.family]?.name || x.family}${x.evidence.filter(Boolean).length ? ` · картинок-доказательств: ${x.evidence.filter(Boolean).length}` : ""}${x.note ? " · " + esc(x.note) : ""}</div></div>
      ${x.status !== "accepted" ? `<button class="small primary" data-rac="accept">✓</button>` : ""}${x.status !== "rejected" ? `<button class="small" data-rac="reject" title="Отклонить">✕</button>` : ""}<button class="small" data-rac="edit" title="Изменить">✎</button><button class="small" data-rac="delete" title="Удалить">🗑</button></div>`).join("")
    : `<p class="muted">Правил пока нет. Они появятся из разговоров («📌 Принять»), из «Найти закономерности» или добавьте своё.</p>`;
  $$("#labRuleList [data-rac]").forEach(b => b.onclick = async () => {
    const id = +b.closest(".rule").dataset.rid, act = b.dataset.rac;
    if (act === "delete" && !confirm("Удалить правило?")) return;
    let text;
    if (act === "edit") { text = prompt("Правило:", b.closest(".rule").querySelector("[data-rtext]").textContent); if (text === null) return; }
    await api("/api/lab/rules", { action: act, id, text }).catch(e => toast(e.message));
    labRules(); labRulesCount();
  });
}
$("#labRulesFam").onchange = labRules;
$("#labInPrompt").onchange = () => api("/api/lab/rules", { action: "in_prompt", value: $("#labInPrompt").checked });
$("#labRuleAdd").onclick = async () => {
  const t = $("#labRuleText").value.trim(); if (!t) return;
  await api("/api/lab/rules", { action: "add", text: t, family: labFam("#labRulesFam") === "all" ? "all" : S.family, source: "manual" }).catch(e => toast(e.message));
  $("#labRuleText").value = ""; labRules(); labRulesCount();
};
$("#labRuleText").addEventListener("keydown", e => { if (e.key === "Enter") $("#labRuleAdd").click(); });
// статистика и отчёт
async function labStats() {
  let st; try { st = await api("/api/lab/stats?family=" + labFam("#labStatsFam")); } catch (e) { return toast(e.message); }
  const ex = st.excluded || {};
  const exLine = ex.manual || ex.auto ? `<p class="muted tiny">Не учитываются: ${ex.auto ? `автоматически ${ex.auto} (${ex.reasons.map(r => esc(r.reason) + " ×" + r.n).join("; ")})` : ""}${ex.auto && ex.manual ? ", " : ""}${ex.manual ? `вручную ${ex.manual}` : ""}. Эти оценки остаются в журнале, но не влияют на цифры и закономерности.</p>` : "";
  if (!st.n) { $("#labStatsBox").innerHTML = exLine + `<p class="muted">Оценок для статистики пока нет. Ставьте звёзды под картинками (или клавиши 1–5) — всё копится здесь.</p>`; return; }
  const rows = st.groups.slice(0, 40).map(r => `<tr><td>${esc(r.feature)}</td><td>${esc(r.value)}</td><td class="num">${r.n}</td><td class="num">${r.avg}</td><td class="num ${r.delta > 0.15 ? "pos" : r.delta < -0.15 ? "neg" : ""}">${r.delta > 0 ? "+" : ""}${r.delta}</td><td class="num">${r.low}</td><td class="num">${r.high}</td></tr>`).join("");
  $("#labStatsBox").innerHTML = `<div class="statsSum"><span>оценок<br><b>${st.n}</b></span><span>средняя<br><b>${st.mean}</b></span><span>за 7 дней<br><b>${st.last7}</b></span><span>распределение 1…5<br><b>${[1, 2, 3, 4, 5].map(i => st.dist[i] || 0).join(" · ")}</b></span><span>период<br><b style="font-size:13px">${st.first.slice(0, 10)} — ${st.last.slice(0, 10)}</b></span></div>
    ${st.tags.length ? `<div class="muted tiny">Метки: ${st.tags.slice(0, 14).map(t => `${esc(t.tag)} ×${t.n} (ср. ${t.avg})`).join(" · ")}</div>` : ""}
    ${exLine}<table class="statsTbl"><tr><th>параметр</th><th>значение</th><th>n</th><th>средняя</th><th>Δ от общей</th><th>≤2</th><th>≥4</th></tr>${rows}</table>
    <p class="muted tiny">Показаны группы от 2 картинок, самые заметные отличия сверху. Выводы надёжны при n от 5–10.</p>`;
}
$("#labStatsFam").onchange = () => { labStats(); autoPanel(); };
/* ---- журналы с разных компьютеров ---- */
async function labDevices() {
  let r; try { r = await api("/api/lab/devices"); } catch { return; }
  if (document.activeElement !== $("#labDevName")) $("#labDevName").value = r.me.name;
  if (document.activeElement !== $("#labSyncDir")) $("#labSyncDir").value = r.sync_dir || "";
  $("#labDevs").textContent = r.devices.length ? "Оценки в журнале: " + r.devices.map(d => `${d.name}${d.me ? " (этот)" : ""} — ${d.n}`).join(" · ") : "";
  const sy = r.sync;
  $("#labSyncState").textContent = !r.sync_dir ? "Общая папка не подключена — можно обмениваться файлами вручную кнопками выше."
    : sy.running ? "Синхронизация…" : sy.error ? "⚠ " + sy.error : sy.last ? `Последняя синхронизация ${sy.last}: ${sy.result}` : "Подключено, синхронизация скоро начнётся";
}
$("#labDevSave").onclick = async () => { try { await api("/api/lab/device", { name: $("#labDevName").value }); toast("Имя сохранено"); labDevices(); } catch (e) { toast(e.message); } };
$("#labPackOut").onclick = () => { location.href = "/api/lab/pack"; };
$("#labPackIn").onclick = () => $("#labPackFile").click();
$("#labPackFile").onchange = async e => {
  const files = [...e.target.files]; e.target.value = ""; if (!files.length) return;
  const fd = new FormData(); files.forEach(f => fd.append("file", f, f.name));
  try {
    const r = await fetch("/api/lab/import", { method: "POST", body: fd }); const j = await r.json();
    if (!r.ok) throw new Error(j.error || "Ошибка");
    toast(`Добавлено: оценок ${j.entries}, разговоров ${j.chats}, правил ${j.rules}${j.rules_merged ? `, объединено правил ${j.rules_merged}` : ""} (${j.devices.join(", ")})`, 7000);
    labStats(); labDevices(); labRulesCount();
  } catch (x) { toast(x.message, 6000); }
};
$("#labSyncSave").onclick = async () => {
  $("#labSyncState").textContent = "Подключаю и синхронизирую…";
  try { await api("/api/lab/device", { sync_dir: $("#labSyncDir").value }); toast($("#labSyncDir").value.trim() ? "Общая папка подключена" : "Общая папка отключена"); } catch (e) { toast(e.message, 6000); }
  labDevices(); labStats(); labRulesCount();
};
$("#labSyncNow").onclick = async () => { $("#labSyncState").textContent = "Синхронизация…"; try { await api("/api/lab/sync", {}); } catch (e) { toast(e.message, 6000); } labDevices(); labStats(); labRulesCount(); };
$("#labPatterns").onclick = async () => {
  const out = $("#labPatOut"), b = $("#labPatterns"); b.disabled = true; out.classList.remove("hidden"); out.innerHTML = `<span class="spin"></span>Считаю статистику, помощник ищет закономерности…`;
  try {
    const r = await api("/api/lab/patterns", { family: labFam("#labStatsFam") });
    out.innerHTML = `<div class="txt">${esc(r.reply.replace(/^\s*[-*]?\s*ПРАВИЛО\s*:.*$/gmi, "").trim())}</div>${r.rules.length ? `<div class="note">Предложено правил: ${r.rules.length} — они во вкладке «Свод правил» со статусом «на проверке». Примите те, что подтверждаются.</div>` : ""}`;
    labRulesCount();
  } catch (e) { out.innerHTML = `<div class="txt" style="color:#ffb3ad">${esc(e.message)}</div>`; }
  finally { b.disabled = false; }
};
$("#labReport").onclick = () => { location.href = "/api/lab/report?fmt=md&family=" + labFam("#labStatsFam"); };
$("#labExport").onclick = () => { location.href = "/api/lab/report?fmt=jsonl&family=" + labFam("#labStatsFam"); };

/* файл в LoRA оказался целой моделью — перенести в models/diffusion_models и выбрать */
async function loraToModel(file) {
  if (!confirm(`«${file}» — это целая модель, а не LoRA.\nПеренести файл в папку моделей (models\\diffusion_models) и выбрать его в поле «Модель»?`)) return;
  try {
    const r = await api("/api/loras/to_model", { file });
    S.loras = S.loras.filter(l => l.name !== file); saveUI();
    await refreshState(); await loadLoraLib();
    S._pending = { unet: r.model, clip: $("#clip").value, vae: $("#vae").value }; updateModelSelects(); saveUI();
    toast("Перенесено и выбрано как модель: " + r.model, 5000);
  } catch (e) { toast(e.message, 6000); }
}

$("#labResetBtn").onclick = async () => {
  const fam = labFam("#labStatsFam"), rules = document.querySelector("input[name=labResetRules]:checked").value;
  const what = fam === "all" ? "всех нейросетей" : FAM[fam].name;
  if (!confirm(`Сбросить статистику ${what}?\nОценки и разговоры будут удалены из журнала (копия базы сохранится в journal\\archive).`)) return;
  try {
    const r = await api("/api/lab/reset", { family: fam, rules });
    toast(`Сброшено оценок: ${r.removed}. Копия: ${r.archive}`, 6000);
    labStats(); labRulesCount(); if (S.rate.url) rateShow(S.rate.url);
  } catch (e) { toast(e.message, 5000); }
};

/* ---------------- автооценка ---------------- */
const fmtPct = p => p === null || p === undefined ? "" : ` · лучше ${p}% ваших`;
async function autoShow(url, tries = 0) {
  const el = $("#rateAuto");
  if (S.rate.url !== url) return;
  let r; try { r = await api("/api/score?image=" + encodeURIComponent(url)); } catch { el.classList.add("hidden"); return; }
  if (S.rate.url !== url) return;
  const s = r.scores;
  if (!s && !r.pending && tries === 0 && !(S.autoInstalled)) { el.classList.add("hidden"); return; }
  el.classList.remove("hidden");
  if (!s) {
    el.innerHTML = `🤖 <span class="spin"></span>`; el.title = "Автооценка считается…";
    if (tries < 40) setTimeout(() => autoShow(url, tries + 1), 3000); else el.classList.add("hidden");
    return;
  }
  const parts = [];
  if (s.pick !== null && s.pick !== undefined) parts.push(`<b>${s.pick.toFixed(1)}</b>`);
  if (s.aes !== null && s.aes !== undefined) parts.push(`эстетика <b>${s.aes.toFixed(1)}</b>`);
  el.innerHTML = "🤖 " + parts.join(" · ");
  el.title = (s.pick !== null && s.pick !== undefined ? `PickScore ${s.pick.toFixed(2)} — насколько люди выбрали бы эту картинку для её промпта${fmtPct(s.pick_pct)}\n` : "")
    + (s.aes !== null && s.aes !== undefined ? `Эстетика LAION ${s.aes.toFixed(2)} из 10 — красота без учёта промпта${fmtPct(s.aes_pct)}\n` : "")
    + "Это мнение моделей, обученных на вкусе многих людей — не ваш вкус. Ваши звёзды важнее.";
}
api("/api/score/state").then(s => { S.autoInstalled = Object.values(s.models).some(m => m.installed); }).catch(() => { });
let autoTimer = null;
async function autoPanel() {
  clearTimeout(autoTimer);
  let s; try { s = await api("/api/score/state?family=" + labFam("#labStatsFam")); } catch { return; }
  S.autoInstalled = Object.values(s.models).some(m => m.installed);
  $("#autoModels").innerHTML = Object.entries(s.models).map(([k, m]) => {
    const dl = m.downloading;
    const status = m.installed ? `<label class="inl"><input type="checkbox" data-aon="${k}" ${m.enabled ? "checked" : ""}> включена</label>`
      : dl ? `скачивается ${dl.total ? Math.round(100 * dl.done / dl.total) + "%" : "…"}`
      : `<button class="small primary" data-aget="${k}">Скачать (${esc(m.size)})</button>`;
    return `<div class="autoRow"><span><b>${esc(m.title)}</b> <span class="muted tiny">${esc(m.size)}${m.installed ? " · ✓ установлена" : ""}</span></span><span>${status}</span></div>`;
  }).join("");
  $$("[data-aget]").forEach(b => b.onclick = async () => { b.disabled = true; try { await api("/api/score/install", { model: b.dataset.aget }); toast("Скачивание началось — прогресс в «Модели» → загрузки"); } catch (e) { toast(e.message); } autoPanel(); pollTransfers(); });
  $$("[data-aon]").forEach(c => c.onchange = async () => { await api("/api/score/config", { ["score_" + c.dataset.aon]: c.checked }); autoPanel(); });
  $("#autoOn").checked = s.auto; $("#autoDev").value = s.device;
  $("#autoState").textContent = s.running || s.queue ? `Оцениваю… в очереди ${s.queue}, готово ${s.done}` : s.error ? "⚠ " + s.error : S.autoInstalled ? (s.loaded.length ? "Модели загружены в память (выгрузятся сами через 10 минут без работы)" : "") : "";
  const a = s.agreement || {}, verdict = rho => rho === null || rho === undefined ? "мало данных (нужно от 5 оценённых картинок)" : rho >= 0.5 ? "сильно совпадает с вами" : rho >= 0.3 ? "заметно совпадает" : rho >= 0.1 ? "слабо совпадает" : rho > -0.1 ? "не совпадает" : "оценивает наоборот";
  $("#autoAgree").innerHTML = S.autoInstalled ? Object.entries({ pick: "PickScore", aes: "Эстетика" }).filter(([k]) => s.models[k].installed && a[k]).map(([k, n]) => {
    const x = a[k];
    return `<div class="agree"><b>${n}</b>: ${verdict(x.rho)}${x.rho !== null && x.rho !== undefined ? ` (ρ = ${x.rho}, по ${x.n} картинкам)` : ""}${Object.keys(x.by_rating).length ? `<span class="muted tiny"> · среднее по вашим оценкам: ${Object.entries(x.by_rating).map(([r, v]) => `★${r} → ${v}`).join(", ")}</span>` : ""}</div>`;
  }).join("") : "";
  if (s.running || s.queue || Object.values(s.models).some(m => m.downloading)) autoTimer = setTimeout(() => { if ($("#labDlg").open && S.lab.tab === "stats") autoPanel(); }, 3000);
}
$("#autoOn").onchange = async () => { await api("/api/score/config", { score_auto: $("#autoOn").checked }); };
$("#autoDev").onchange = async () => { await api("/api/score/config", { score_device: $("#autoDev").value }); autoPanel(); };
$("#autoBackfill").onclick = async () => { try { const r = await api("/api/score/backfill", {}); toast(r.queued ? `В очереди на оценку: ${r.queued}` : "Все картинки журнала уже оценены"); } catch (e) { toast(e.message, 5000); } autoPanel(); };
