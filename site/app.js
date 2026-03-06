function fmtPct(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "-";
  return `${(Number(v) * 100).toFixed(1)}%`;
}

function fmtNum(v, fixed = 2) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "-";
  return Number(v).toFixed(fixed);
}

function badge(label) {
  const text = label || "观望";
  if (text.includes("强推")) return `<span class="badge b-hot">${text}</span>`;
  if (text.includes("主推")) return `<span class="badge b-star">${text}</span>`;
  if (text.includes("可博")) return `<span class="badge b-ok">${text}</span>`;
  return `<span class="badge b-no">${text}</span>`;
}

function safeText(v) {
  return String(v || "").replace(/"/g, "'");
}

function modelProbCell(x) {
  const one = (name, arr) => {
    if (!arr) return "";
    return `<div><strong>${name}</strong> ${fmtPct(arr[0])} / ${fmtPct(arr[1])} / ${fmtPct(arr[2])}</div>`;
  };
  return `${one("PE", x.pe_p)}${one("ML", x.ml_p)}${one("BM", x.bm_p)}` || "-";
}

function reasonBlock(x) {
  const rs = x.reasons || {};
  const base = rs.base || x.why || "-";
  const gpt = rs.gpt ? `<div><strong>GPT:</strong> ${rs.gpt}</div>` : "";
  const gem = rs.gemini ? `<div><strong>Gemini:</strong> ${rs.gemini}</div>` : "";
  const fb = rs.fallback ? `<div><strong>Fallback:</strong> ${rs.fallback}</div>` : "";
  return `<div class="reason-block"><div><strong>模型:</strong> ${base}</div>${gpt}${gem}${fb}</div>`;
}

function initTheme() {
  const btn = document.getElementById("theme");
  const saved = localStorage.getItem("theme") || "night";
  if (saved === "day") document.documentElement.classList.add("day");
  btn.addEventListener("click", () => {
    document.documentElement.classList.toggle("day");
    localStorage.setItem("theme", document.documentElement.classList.contains("day") ? "day" : "night");
  });
}

function makeSortable(id) {
  const table = document.getElementById(id);
  if (!table || !table.tBodies.length) return;
  const headers = table.querySelectorAll("th");
  headers.forEach((th, idx) => {
    th.addEventListener("click", () => {
      const tbody = table.tBodies[0];
      const rows = [...tbody.rows];
      const dir = th.dataset.dir === "asc" ? -1 : 1;
      rows.sort((a, b) => {
        const va = a.cells[idx]?.textContent?.trim() || "";
        const vb = b.cells[idx]?.textContent?.trim() || "";
        const na = Number(va.replace(/[^0-9.-]/g, ""));
        const nb = Number(vb.replace(/[^0-9.-]/g, ""));
        if (!Number.isNaN(na) && !Number.isNaN(nb)) return (na - nb) * dir;
        return va.localeCompare(vb, "zh-Hans") * dir;
      });
      th.dataset.dir = dir > 0 ? "asc" : "desc";
      rows.forEach((r) => tbody.appendChild(r));
    });
  });
}

async function loadPicks() {
  const res = await fetch("data/picks.json", { cache: "no-store" });
  if (!res.ok) throw new Error(`picks.json ${res.status}`);
  return res.json();
}

async function renderPicks() {
  const data = await loadPicks();
  const meta = data.meta || {};
  const stats = data.stats || {};
  const bt = stats.backtest || {};

  document.getElementById("meta").textContent =
    `UTC ${meta.generated_at_utc || "-"} | 竞彩范围: ${meta.scope || "JCZQ"} | 定时: ${(meta.schedule_bjt || []).join(" / ") || "09:30 / 21:30"}`;

  const llm = meta.llm || {};
  const usage = llm.usage || {};
  const llmLine = `LLM状态 OpenAI:${usage.openai || 0} Gemini:${usage.gemini || 0} 双模型:${usage.both || 0} 回退:${usage.fallback || 0}`;
  const apiUsage = meta.api_usage || {};
  const apiLine = `API状态 Foot:${apiUsage.api_football_enabled ? "ON" : "OFF"} FD:${apiUsage.football_data_enabled ? "ON" : "OFF"} Odds:${apiUsage.odds_api_enabled ? "ON" : "OFF"}`;
  const scheduleLine = document.querySelector(".schedule");
  if (scheduleLine) {
    scheduleLine.textContent = `${scheduleLine.textContent} | ${apiLine} | ${llmLine}`;
  }

  document.getElementById("k_fx").textContent = String(stats.fixtures ?? 0);
  document.getElementById("k_top").textContent = String(stats.top ?? 0);
  document.getElementById("k_roi").textContent = fmtPct(bt.roi);
  document.getElementById("k_hit").textContent = fmtPct(bt.hit_rate);
  document.getElementById("k_ev").textContent = fmtPct(bt.avg_ev);
  document.getElementById("k_log").textContent = fmtNum(bt.logloss, 3);

  document.getElementById("s_matches").textContent = String(bt.matches_used ?? "-");
  document.getElementById("s_bets").textContent = String(bt.bets ?? "-");
  document.getElementById("s_ev").textContent = fmtPct(bt.avg_ev);
  document.getElementById("s_log").textContent = fmtNum(bt.logloss, 3);
  document.getElementById("s_roi").textContent = fmtPct(bt.roi);
  document.getElementById("s_hit").textContent = fmtPct(bt.hit_rate);

  const top = data.top_picks || [];
  const all = data.all || [];

  const cardWrap = document.getElementById("top-cards");
  cardWrap.innerHTML = top.slice(0, 4).map((x, i) => `
    <article class="pick-card" style="animation-delay:${i * 120}ms">
      <div class="pick-head">
        <span>${x.date || ""} ${x.time || ""}</span>
        ${badge(x.label)}
      </div>
      <h3>${x.home || "-"} <small>vs</small> ${x.away || "-"}</h3>
      <p class="pick-main">推荐: <strong>${x.pick || "模型"}</strong> | 预测比分: ${x.most_likely_score || "-"}</p>
      <p class="pick-sub">主/平/客: ${fmtPct(x.p_home)} / ${fmtPct(x.p_draw)} / ${fmtPct(x.p_away)}</p>
      <p class="pick-sub">EV ${fmtNum(x.ev, 4)} | Kelly ${fmtNum(x.kelly, 4)}</p>
      <div class="pick-why" title="${safeText(x.why)}">${reasonBlock(x)}</div>
    </article>
  `).join("");

  document.querySelector("#top tbody").innerHTML = top.map((x) => `
    <tr>
      <td>${x.date || ""} ${x.time || ""}</td>
      <td>${x.league || ""}</td>
      <td>${x.home || ""}</td>
      <td>${x.away || ""}</td>
      <td>${badge(x.label)} ${x.pick ? ` ${x.pick}` : ""}</td>
      <td>${fmtNum(x.score, 0)}</td>
      <td>${fmtNum(x.ev, 4)}</td>
      <td>${fmtNum(x.kelly, 4)}</td>
      <td>${fmtPct(x.p_home)} / ${fmtPct(x.p_draw)} / ${fmtPct(x.p_away)}</td>
      <td class="prob-cell">${modelProbCell(x)}</td>
      <td>${x.odds_win ?? "-"} / ${x.odds_draw ?? "-"} / ${x.odds_lose ?? "-"}</td>
    </tr>
  `).join("");

  document.querySelector("#all tbody").innerHTML = all.map((x) => `
    <tr>
      <td>${x.date || ""} ${x.time || ""}</td>
      <td>${x.league || ""}</td>
      <td>${x.home || ""}</td>
      <td>${x.away || ""}</td>
      <td>${fmtNum(x.xg_home)} / ${fmtNum(x.xg_away)}</td>
      <td>${fmtPct(x.p_home)} / ${fmtPct(x.p_draw)} / ${fmtPct(x.p_away)}</td>
      <td class="prob-cell">${modelProbCell(x)}</td>
      <td>${x.most_likely_score || "-"}</td>
      <td>${badge(x.label)}</td>
      <td title="${safeText(x.why)}">${reasonBlock(x)}</td>
    </tr>
  `).join("");
}

async function renderJCZQ() {
  const tb = document.querySelector("#jczq tbody");
  try {
    const res = await fetch("data/jczq.json", { cache: "no-store" });
    const data = await res.json();
    const rows = data.matches || [];
    if (!rows.length) {
      tb.innerHTML = `<tr><td colspan="8">暂无竞彩网数据: ${data.meta?.error || "请检查抓取日志"}</td></tr>`;
      return;
    }
    tb.innerHTML = rows.slice(0, 120).map((m) => `
      <tr>
        <td>${m.league || ""}</td>
        <td>${m.time || ""}</td>
        <td>${m.home || ""}</td>
        <td>${m.away || ""}</td>
        <td>${m.odds_win ?? "-"}</td>
        <td>${m.odds_draw ?? "-"}</td>
        <td>${m.odds_lose ?? "-"}</td>
        <td>${m.handicap ?? "-"}</td>
      </tr>
    `).join("");
  } catch (err) {
    tb.innerHTML = `<tr><td colspan="8">加载失败: ${err}</td></tr>`;
  }
}

async function bootstrap() {
  initTheme();
  await renderPicks();
  await renderJCZQ();
  makeSortable("top");
  makeSortable("all");
}

bootstrap().catch((err) => {
  const meta = document.getElementById("meta");
  meta.textContent = `页面加载失败: ${err}`;
});
