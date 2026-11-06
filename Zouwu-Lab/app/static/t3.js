"use strict";
/* ================= Вкладка «3D»: Hunyuan3D — модель объекта по ракурсам =================
   ракурсы (спереди обязательно, слева / сзади / справа по желанию) → «Построить» → .glb в просмотрщике */
const T3 = { views: { front: null, left: null, back: null, right: null }, q: "fast", results: [], cur: -1, job: null, slot: null };
const T3_VIEWS = [["front", "Спереди"], ["left", "Слева"], ["back", "Сзади"], ["right", "Справа"]];
const T3KEY = "zl_3d";
const t3Multi = n => /(^|[-_.])mv([-_.]|$)/i.test((n || "").split(/[\\/]/).pop());

function t3Save() {
  try { localStorage.setItem(T3KEY, JSON.stringify({ views: T3.views, q: T3.q, model: $("#t3Model").value, bg: $("#t3Bg").checked, th: +$("#t3Th").value })); } catch { }
}
(function t3Load() {
  let o = null; try { o = JSON.parse(localStorage.getItem(T3KEY) || "null"); } catch { }
  if (!o) return;
  T3.views = { ...T3.views, ...(o.views || {}) }; T3.q = o.q || "fast"; T3._model = o.model;
  if (o.bg === false) $("#t3Bg").checked = false;
  if (o.th) $("#t3Th").value = o.th;
})();

/* ---------- модели ---------- */
function t3Models() {
  const list = ((S.models || {}).checkpoints || []).map(x => x.name);
  const sel = $("#t3Model"), cur = sel.value || T3._model;
  const hy = list.filter(n => /hunyuan|3d/i.test(n));
  const use = hy.length ? hy : list;
  sel.innerHTML = use.length ? use.map(n => `<option value="${esc(n)}">${esc(n)}</option>`).join("") : `<option value="">— нет модели: скачайте в «Модели» → 3D —</option>`;
  if (cur && use.includes(cur)) sel.value = cur;
  else { const mv = use.find(n => t3Multi(n) && !/turbo/i.test(n)) || use[0]; if (mv) sel.value = mv; }
  t3Hint();
}
function t3Hint() {
  const n = $("#t3Model").value, multi = t3Multi(n);
  $("#t3ModelHint").innerHTML = !n ? `Нужна модель Hunyuan3D. <a href="#" id="t3Get">Скачать</a> — рекомендуется Hunyuan3D‑2mv (≈5 ГБ).`
    : multi ? (/turbo|fast/i.test(n) ? "Быстрая модель по ракурсам: ~10 шагов. Для черновиков." : "Модель по ракурсам: 1–4 картинки. 30 шагов.")
      : "Эта модель строит 3D по одной картинке — используется только «Спереди».";
  const g = $("#t3Get"); if (g) g.onclick = e => { e.preventDefault(); S.catFam = "3d"; openModels("catalog"); };
  T3_VIEWS.forEach(([k]) => { const el = $(`#t3Grid [data-v="${k}"]`); if (el) el.classList.toggle("off", !multi && k !== "front" && !!n); });
}
$("#t3Model").onchange = () => { t3Hint(); t3Save(); };
$("#t3Models").onclick = () => { S.catFam = "3d"; openModels("catalog"); };

/* ---------- ракурсы ---------- */
function t3RenderGrid() {
  $("#t3Grid").innerHTML = T3_VIEWS.map(([k, lab]) => {
    const v = T3.views[k];
    return `<div class="t3slot ${k === "front" ? "req" : ""}" data-v="${k}" title="${lab}: перетащите картинку или нажмите">
      <span class="lab">${lab}</span>
      ${v ? `<img src="/input/${encodeURIComponent(v)}" alt=""><button class="icon x" data-x="${k}" title="Убрать">✕</button>` : `<span>＋ ${lab.toLowerCase()}</span>`}</div>`;
  }).join("");
  $$("#t3Grid .t3slot").forEach(el => {
    const k = el.dataset.v;
    el.onclick = e => { if (e.target.closest("button")) return; T3.slot = k; $("#t3File").click(); };
    el.ondragover = e => { e.preventDefault(); el.classList.add("over"); };
    el.ondragleave = () => el.classList.remove("over");
    el.ondrop = e => { e.preventDefault(); el.classList.remove("over"); t3Upload(k, e.dataTransfer.files); };
  });
  $$("#t3Grid [data-x]").forEach(b => b.onclick = e => { e.stopPropagation(); T3.views[b.dataset.x] = null; t3RenderGrid(); t3Save(); });
  t3Hint();
}
async function t3Upload(k, files) {
  const f = [...(files || [])].find(x => x.type.startsWith("image/"));
  if (!f) return;
  try { T3.views[k] = await uploadBlob(f, f.name || "view.png"); t3RenderGrid(); t3Save(); }
  catch (e) { toast(e.message, 4000); }
}
$("#t3File").onchange = e => { t3Upload(T3.slot || "front", e.target.files); e.target.value = ""; };
document.addEventListener("paste", e => {
  if (document.body.dataset.view !== "3d" || e.target.matches("input,textarea")) return;
  const files = [...(e.clipboardData?.files || [])]; if (!files.length) return;
  const k = (T3_VIEWS.find(([v]) => !T3.views[v]) || ["front"])[0];
  t3Upload(k, files);
}, true);

/* ---------- удаление фона (нейросеть ISNet, общая с Редактором) ---------- */
async function t3BgState() {
  let m = {}; try { m = await api("/api/matting"); } catch { }
  const box = $("#t3BgSt");
  if (m.installed || !$("#t3Bg").checked) { box.innerHTML = m.installed ? `<span class="muted">Фон вырезается нейросетью ISNet; картинки с прозрачным фоном берутся как есть.</span>` : ""; return; }
  box.innerHTML = m.installing ? `Скачиваю нейросеть вырезки… ${m.progress || 0}%` :
    `Для удаления фона нужна нейросеть ISNet (${esc(m.size || "171 МБ")}, один раз). Без неё картинки берутся как есть — лучше на белом или прозрачном фоне. <button class="small" id="t3BgGet">Скачать</button>${m.error ? `<div class="e">${esc(m.error)}</div>` : ""}`;
  const b = $("#t3BgGet"); if (b) b.onclick = async () => { b.disabled = true; try { await api("/api/matting/install", {}); } catch (e) { toast(e.message, 4000); } t3BgPoll(); };
  if (m.installing) t3BgPoll();
}
function t3BgPoll() { clearTimeout(T3._bgT); T3._bgT = setTimeout(t3BgState, 1500); }
$("#t3Bg").onchange = () => { t3BgState(); t3Save(); };

/* ---------- качество и настройки ---------- */
$$("#t3Q [data-q]").forEach(b => b.onclick = () => { T3.q = b.dataset.q; $$("#t3Q [data-q]").forEach(x => x.classList.toggle("on", x === b)); t3Save(); });
$("#t3Th").oninput = () => { $("#t3ThV").textContent = (+$("#t3Th").value).toFixed(2); t3Save(); };
$("#t3SeedR").onclick = () => { $("#t3Seed").value = -1; };

/* ---------- генерация ---------- */
function t3Err(m) { $("#t3Err").textContent = m || ""; $("#t3Err").classList.toggle("hidden", !m); }
async function t3Go() {
  if (T3.job) return;
  t3Err("");
  if (!T3.views.front) return t3Err("Добавьте картинку «Спереди» — она обязательна.");
  const body = { model: $("#t3Model").value, views: T3.views, remove_bg: $("#t3Bg").checked, quality: T3.q,
    steps: +$("#t3Steps").value || 0, seed: +$("#t3Seed").value, threshold: +$("#t3Th").value };
  T3.job = "pending"; $("#t3Go").disabled = true;
  $("#t3Prog").classList.remove("hidden"); $("#t3Bar").style.width = "0"; $("#t3Txt").textContent = $("#t3Bg").checked ? "Подготовка ракурсов (вырезка фона)…" : "Отправка…";
  let r;
  try { r = await api("/api/3d/generate", body); }
  catch (e) { t3Err(e.message); T3.job = null; $("#t3Go").disabled = false; $("#t3Prog").classList.add("hidden"); return; }
  T3.job = r.id; $("#t3Seed").dataset.last = r.seed;
  const t0 = Date.now();
  while (T3.job === r.id) {
    await new Promise(res => setTimeout(res, 500));
    let j; try { j = await api("/api/job/" + r.id); } catch { continue; }
    const pct = j.max ? Math.round(100 * j.value / j.max) : 0;
    $("#t3Bar").style.width = (j.status === "done" ? 100 : pct) + "%";
    $("#t3Txt").textContent = `${j.stage || ""}${j.max ? ` ${j.value}/${j.max}` : ""} · ${((Date.now() - t0) / 1000).toFixed(0)} с`;
    if (j.status === "done") {
      const glb = (j.images || []).find(u => /\.glb(\?|$)/i.test(u));
      if (glb) { await t3LoadHistory(); t3Open(T3.results.findIndex(x => x.url === glb)); toast(`3D‑модель готова за ${((Date.now() - t0) / 1000).toFixed(0)} с`); }
      else t3Err("Движок не вернул модель");
      break;
    }
    if (j.status === "error" || j.status === "cancelled") { t3Err(j.error || "Ошибка"); break; }
  }
  T3.job = null; $("#t3Go").disabled = false;
  setTimeout(() => $("#t3Prog").classList.add("hidden"), 800);
}
$("#t3Go").onclick = t3Go;
$("#t3Stop").onclick = async () => { try { await api("/api/interrupt", {}); } catch { } };
document.addEventListener("keydown", e => { if (document.body.dataset.view === "3d" && e.ctrlKey && e.key === "Enter") { e.preventDefault(); t3Go(); } });

/* ---------- просмотр и история ---------- */
async function t3LoadHistory() {
  let r; try { r = await api("/api/gallery"); } catch { return; }
  T3.results = [];
  for (const e of r.items) for (const u of e.images) if (/\.glb(\?|$)/i.test(u)) {
    const pv = e.params.views || {};
    T3.results.push({ url: u, thumb: pv.front ? "/input/" + encodeURIComponent(pv.front) : "", time: e.time, secs: e.seconds, p: e.params });
  }
  t3Strip();
}
function t3Strip() {
  $("#t3Strip").innerHTML = T3.results.slice(0, 40).map((x, i) => `<div class="ti ${i === T3.cur ? "on" : ""}" data-i="${i}" title="${esc(x.time)} · ${esc(String(x.p.model || ""))}">${x.thumb ? `<img src="${x.thumb}" alt="">` : "🧊"}<span>${esc(x.time.slice(11, 16))}</span></div>`).join("");
  $$("#t3Strip [data-i]").forEach(el => el.onclick = () => t3Open(+el.dataset.i));
}
function t3Open(i) {
  const x = T3.results[i]; if (!x) return;
  T3.cur = i;
  const mv = $("#t3MV"); mv.src = x.url; T3.wire = false;
  $("#t3Empty").classList.add("hidden");
  const dl = $("#t3Dl"); dl.href = x.url; dl.setAttribute("download", decodeURIComponent(x.url.split("/").pop())); dl.classList.remove("hidden");
  const nv = Object.keys(x.p.views || {}).length;
  $("#t3Info").textContent = `${x.time} · ${x.p.model || ""} · ракурсов: ${nv} · ${{ fast: "черновик", mid: "средне", high: "высокое" }[x.p.quality] || ""}${x.secs ? ` · ${x.secs} с` : ""} · seed ${x.p.seed}`;
  t3Strip();
}
/* каркас: перекрашиваем материалы модели в режим wireframe */
$("#t3Wire").onclick = () => {
  const mv = $("#t3MV"); if (!mv.model) return;
  T3.wire = !T3.wire;
  try {
    const scene = mv[Object.getOwnPropertySymbols(mv).find(s => String(s) === "Symbol(scene)")];
    scene && scene.traverse(o => { if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => { m.wireframe = T3.wire; m.needsUpdate = true; }); });
    scene && scene.queueRender && scene.queueRender();
  } catch { }
  $("#t3Wire").classList.toggle("on", T3.wire);
};

/* ---------- вход во вкладку ---------- */
window.t3Show = async () => {
  t3Models(); t3RenderGrid(); t3BgState();
  $$("#t3Q [data-q]").forEach(x => x.classList.toggle("on", x.dataset.q === T3.q));
  $("#t3ThV").textContent = (+$("#t3Th").value).toFixed(2);
  if (!T3.results.length) { await t3LoadHistory(); if (T3.results.length && T3.cur < 0) t3Open(0); }
};
$("#t3Tab").onclick = () => setView("3d");
setInterval(() => { if (document.body.dataset.view === "3d") t3Models(); }, 5000);
