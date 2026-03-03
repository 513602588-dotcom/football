function badge(label){
  if(label.includes("强推")) return `<span class="badge b-hot">${label}</span>`;
  if(label.includes("主推")) return `<span class="badge b-star">${label}</span>`;
  if(label.includes("可博")) return `<span class="badge b-ok">${label}</span>`;
  return `<span class="badge b-no">${label}</span>`;
}
async function main(){
  const themeBtn = document.getElementById("theme");
  const saved = localStorage.getItem("theme") || "dark";
  if(saved === "light") document.documentElement.classList.add("light");
  themeBtn.onclick = () => {
    document.documentElement.classList.toggle("light");
    localStorage.setItem("theme", document.documentElement.classList.contains("light") ? "light" : "dark");
  };

  const res = await fetch("data/picks.json", {cache:"no-store"});
  const data = await res.json();

  document.getElementById("meta").textContent =
    `更新时间(UTC)：${data.meta.generated_at_utc} ｜ Python ${data.meta.python} ｜ 赛季：${data.meta.seasons_used.join(", ")} ｜ 融合：PE ${data.meta.fusion.W_PE} + ML ${data.meta.fusion.W_ML}（ML=${data.meta.fusion.ml_enabled}）`;

  document.getElementById("k_fx").textContent = String(data.stats.fixtures ?? "-");
  document.getElementById("k_top").textContent = String(data.stats.top ?? "-");
  const bt = data.stats.backtest || {};
  document.getElementById("k_roi").textContent = ((bt.roi ?? 0)*100).toFixed(1) + "%";
  document.getElementById("k_hit").textContent = ((bt.hit_rate ?? 0)*100).toFixed(1) + "%";

  const top = data.top_picks || [];
  const all = data.all || [];

  document.querySelector("#top tbody").innerHTML = top.map(x => `
    <tr>
      <td>${x.date||""}</td>
      <td>${x.league||""}</td>
      <td>${x.home||""}</td>
      <td>${x.away||""}</td>
      <td>${badge(x.label||"-")} ${x.pick?(" · "+x.pick):""}</td>
      <td>${x.score ?? "-"}</td>
      <td>${x.ev ?? "-"}</td>
      <td>${x.kelly ?? "-"}</td>
      <td>${x.p_home?`${(x.p_home*100).toFixed(1)} / ${(x.p_draw*100).toFixed(1)} / ${(x.p_away*100).toFixed(1)}%`:"-"}</td>
      <td>${x.odds_win?`${x.odds_win} / ${x.odds_draw} / ${x.odds_lose} (${x.book||"-"})`:"-"}</td>
    </tr>
  `).join("");

  document.querySelector("#all tbody").innerHTML = all.map(x => `
    <tr>
      <td>${x.date||""}</td>
      <td>${x.league||""}</td>
      <td>${x.home||""}</td>
      <td>${x.away||""}</td>
      <td>${(x.xg_home!=null)?`${x.xg_home.toFixed(2)} / ${x.xg_away.toFixed(2)}`:"-"}</td>
      <td>${x.most_likely_score||""}</td>
      <td>${badge(x.label||"-")}</td>
      <td title="${(x.why||"").replaceAll('"','')}">${x.why||""}</td>
    </tr>
  `).join("");
}
main().catch(e=>{ document.getElementById("meta").textContent="加载失败："+e; });

async function renderJCZQ(){
  try{
    const res = await fetch("data/jczq.json", {cache:"no-store"});
    const data = await res.json();
    const tb = document.querySelector("#jczq tbody");
    if(!tb) return;
    const ms = data.matches || [];
    if(ms.length === 0){
      tb.innerHTML = `<tr><td colspan="8">暂无数据（${data.meta?.error || "可能接口未配置/不可访问"}）</td></tr>`;
      return;
    }
    tb.innerHTML = ms.slice(0,200).map(m=>`
      <tr>
        <td>${m.league||""}</td>
        <td>${m.time||""}</td>
        <td>${m.home||""}</td>
        <td>${m.away||""}</td>
        <td>${m.odds_win ?? "-"}</td>
        <td>${m.odds_draw ?? "-"}</td>
        <td>${m.odds_lose ?? "-"}</td>
        <td>${m.handicap ?? "-"}</td>
      </tr>
    `).join("");
  }catch(e){
    const tb = document.querySelector("#jczq tbody");
    if(tb) tb.innerHTML = `<tr><td colspan="8">加载失败：${e}</td></tr>`;
  }
}
renderJCZQ();
