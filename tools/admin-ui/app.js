(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));

  // 同一份前端支持两种部署：内嵌在 ninfer-serve:8080 的 /console/ 下时直连原生
  // API；独立开发服务器 :8081 下仍走 Python 聚合层的 /api/*。
  const EMBEDDED = location.pathname.indexOf("/console") === 0;
  const API = EMBEDDED
    ? {
        overview: "/console/api/overview",
        log: "/console/api/log",
        chat: "/v1/chat/completions",
        model: (action) => "/ninfer/model/" + action,
      }
    : {
        overview: "/api/overview",
        log: "/api/log",
        chat: "/api/chat",
        model: (action) => "/api/model/" + action,
      };

  const state = {
    view: "chat",
    overview: null,
    chats: [],
    activeId: null,
    streaming: false,
    pendingImages: [],
    poll: null,
    reqShowAll: false,
    lastBubblePaint: 0,
    settings: {
      apiKey: "", temperature: 0.6, topP: 0.8, topK: 20,
      maxTokens: 4096, effort: "none", stream: true,
    },
  };

  // ---------------------------------------------------------------- storage
  function load(key, fallback) {
    try {
      const raw = localStorage.getItem("ninfer." + key);
      return raw ? JSON.parse(raw) : fallback;
    } catch (e) { return fallback; }
  }
  function save(key, value) {
    try { localStorage.setItem("ninfer." + key, JSON.stringify(value)); } catch (e) {}
  }
  function uid() { return Math.random().toString(36).slice(2, 10); }

  // ------------------------------------------------------------------ chat
  function activeChat() {
    let chat = state.chats.find((c) => c.id === state.activeId);
    if (!chat) {
      chat = { id: uid(), title: "新对话", messages: [], ts: Date.now() };
      state.chats.unshift(chat);
      state.activeId = chat.id;
      persistChats();
    }
    return chat;
  }
  function persistChats() { save("chats", state.chats); save("active", state.activeId); }

  function contentText(content) {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      return content.filter((p) => p && p.type === "text").map((p) => p.text || "").join("\n");
    }
    return "";
  }
  function inlineMarkup(text) {
    return esc(text)
      .replace(/`([^`\n]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
      .replace(/\n/g, "<br>");
  }
  function renderText(content) {
    const text = contentText(content);
    const images = Array.isArray(content)
      ? content.filter((p) => p && p.type === "image_url").map((p) => p.image_url?.url || "")
      : [];
    let html = String(text).split("```").map((part, i) =>
      i % 2 ? `<pre><code>${esc(part)}</code></pre>` : inlineMarkup(part)
    ).join("");
    for (const url of images) {
      html += `<div><img src="${esc(url)}" alt="附件" style="max-height:220px;border-radius:6px;margin-top:6px"></div>`;
    }
    return html;
  }

  function renderChatList() {
    const list = $("chatList");
    if (!state.chats.length) {
      list.innerHTML = '<div class="hint" style="padding:10px">还没有会话</div>';
      return;
    }
    list.innerHTML = state.chats.map((c) => `
      <div class="chat-item ${c.id === state.activeId ? "active" : ""}" data-id="${esc(c.id)}">
        <span>${esc(c.title || "新对话")}</span>
        <button class="del-chat" data-id="${esc(c.id)}" title="删除">×</button>
      </div>`).join("");
  }

  function renderMessages() {
    const box = $("messages");
    const chat = state.chats.find((c) => c.id === state.activeId);
    if (!chat || !chat.messages.length) {
      box.innerHTML = `<div class="empty-state">
        <h2>ninfer 控制台</h2>
        <p>本地推理服务的对话与监控面板。<br>数据全部来自 ${esc(state.overview?.machine?.host || "显卡主机")} 上运行的服务。</p>
      </div>`;
      return;
    }
    box.innerHTML = chat.messages.map((m) => {
      const reasoning = m.reasoning
        ? `<details class="reasoning" ${state.streaming && m === chat.messages[chat.messages.length - 1] ? "open" : ""}>
             <summary>思考过程</summary><div>${renderText(m.reasoning)}</div></details>`
        : "";
      return `<div class="msg ${m.role === "user" ? "user" : "assistant"}" id="msg-${esc(m.id || "")}">
        <div class="msg-head"><strong>${m.role === "user" ? "你" : "ninfer"}</strong>
          <span>${m.ts ? new Date(m.ts).toLocaleTimeString("zh-CN", { hour12: false }) : ""}</span></div>
        <div class="msg-body">${reasoning}<div class="msg-content">${renderText(m.content)}</div></div>
      </div>`;
    }).join("");
    box.scrollTop = box.scrollHeight;
  }

  function paintBubble(msg) {
    const node = document.querySelector(`#msg-${CSS.escape(msg.id)} .msg-body`);
    if (!node) return;
    const reasoning = msg.reasoning
      ? `<details class="reasoning" open><summary>思考过程</summary><div>${renderText(msg.reasoning)}</div></details>` : "";
    node.innerHTML = reasoning + `<div class="msg-content">${renderText(msg.content)}</div>`;
    const box = $("messages");
    box.scrollTop = box.scrollHeight;
  }

  function syncSettings() {
    const s = state.settings;
    $("apiKey").value = s.apiKey;
    if ($("aboutApiKey")) $("aboutApiKey").value = s.apiKey;
    $("effort").value = s.effort;
    $("temperature").value = s.temperature;
    $("topP").value = s.topP;
    $("topK").value = s.topK;
    $("maxTokens").value = s.maxTokens;
    $("streamChk").checked = s.stream;
    $("temperatureOut").textContent = Number(s.temperature).toFixed(2);
    $("topPOut").textContent = Number(s.topP).toFixed(2);
    $("topKOut").textContent = s.topK;
  }
  function persistSettings() { save("settings", state.settings); }

  function buildPayload(chat) {
    const s = state.settings;
    const messages = chat.messages.slice(0, -1).map((m) => {
      const item = { role: m.role, content: m.content };
      if (m.role === "assistant" && m.reasoning) item.reasoning_content = m.reasoning;
      return item;
    });
    const payload = {
      messages,
      stream: s.stream,
      temperature: Number(s.temperature),
      top_p: Number(s.topP),
      max_tokens: Number(s.maxTokens),
    };
    if (Number(s.topK) > 0) payload.top_k = Number(s.topK);
    // Qwen3.8 只有 none / low / medium / xhigh 四档；high、max 会被服务端归一为 xhigh，
    // 这里直接发送真实档位，避免界面上出现模型并不支持的选项。
    payload.reasoning_effort = s.effort || "none";
    const model = state.overview?.health?.model;
    if (model) payload.model = model;
    return payload;
  }

  async function sendMessage() {
    if (state.streaming) return;
    const text = $("prompt").value.trim();
    if (!text && !state.pendingImages.length) return;

    const chat = activeChat();
    let content = text;
    if (state.pendingImages.length) {
      content = [];
      if (text) content.push({ type: "text", text });
      for (const url of state.pendingImages) content.push({ type: "image_url", image_url: { url } });
    }
    chat.messages.push({ id: uid(), role: "user", content, ts: Date.now() });
    if (chat.messages.length === 1) chat.title = (text || "图片会话").slice(0, 26);
    const reply = { id: uid(), role: "assistant", content: "", reasoning: "", ts: Date.now() };
    chat.messages.push(reply);

    $("prompt").value = "";
    state.pendingImages = [];
    state.streaming = true;
    $("sendBtn").disabled = true;
    $("sendBtn").textContent = "生成中";
    persistChats();
    renderChatList();
    renderMessages();

    try {
      const headers = { "Content-Type": "application/json" };
      if (EMBEDDED) {
        // 同源直连原生端点：服务端按 Authorization: Bearer 校验。
        if (state.settings.apiKey) {
          headers.Authorization = "Bearer " + state.settings.apiKey;
        }
      } else {
        headers["X-Ninfer-Key"] = state.settings.apiKey;
      }
      const response = await fetch(API.chat, {
        method: "POST",
        headers,
        body: JSON.stringify(buildPayload(chat)),
      });
      if (!response.ok) {
        let detail = "";
        try { detail = (await response.json()).error || ""; } catch (e) {}
        throw new Error(`HTTP ${response.status} ${detail}`.trim());
      }

      if (!state.settings.stream) {
        const data = await response.json();
        const choice = data.choices?.[0]?.message || {};
        reply.content = choice.content || "";
        reply.reasoning = choice.reasoning_content || "";
      } else {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        const handleBlock = (block) => {
          for (const line of block.split("\n")) {
            if (!line.startsWith("data:")) continue;
            const raw = line.slice(5).trim();
            if (!raw || raw === "[DONE]") continue;
            let obj;
            try { obj = JSON.parse(raw); } catch (e) { continue; }
            if (obj.error) throw new Error(obj.error.message || JSON.stringify(obj.error));
            const delta = obj.choices?.[0]?.delta || {};
            if (delta.content) reply.content += delta.content;
            if (delta.reasoning_content) reply.reasoning += delta.reasoning_content;
          }
        };
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let index;
          while ((index = buffer.indexOf("\n\n")) >= 0) {
            handleBlock(buffer.slice(0, index));
            buffer = buffer.slice(index + 2);
          }
          const now = performance.now();
          if (now - state.lastBubblePaint > 60) {
            state.lastBubblePaint = now;
            paintBubble(reply);
          }
        }
        if (buffer.trim()) handleBlock(buffer);
      }
      paintBubble(reply);
      refreshOverview();
    } catch (err) {
      reply.content += `\n\n[请求失败] ${err.message}`;
      paintBubble(reply);
    } finally {
      state.streaming = false;
      $("sendBtn").disabled = false;
      $("sendBtn").textContent = "发送";
      persistChats();
    }
  }

  // --------------------------------------------------------------- monitor
  function lastNonZero(values) {
    if (!Array.isArray(values)) return 0;
    for (let i = values.length - 1; i >= 0; i--) if (values[i] > 0) return values[i];
    return 0;
  }
  const fmtNum = (v, digits = 0) => v == null || !isFinite(v)
    ? "—"
    : Number(v).toLocaleString("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  const kfmt = (v) => {
    if (v == null || !isFinite(v)) return "—";
    const n = Number(v);
    if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}k`;
    return fmtNum(n);
  };
  const gb = (mib, digits = 1) => mib == null ? "—" : fmtNum(mib / 1024, digits);

  function sparkline(id, values, max) {
    const node = $(id);
    if (!node) return;
    const v = (values || []).slice(-60).map((x) => (x == null ? 0 : Number(x) || 0));
    const line = node.querySelector(".line");
    const area = node.querySelector(".area");
    if (!line || !area) return;
    if (v.length < 2) {
      line.setAttribute("d", "");
      area.setAttribute("d", "");
      return;
    }
    const top = Math.max(max || 0, ...v, 1);
    const d = v.map((x, i) => `${i ? "L" : "M"}${(i / (v.length - 1) * 100).toFixed(2)},${(30 - x / top * 26).toFixed(2)}`).join(" ");
    line.setAttribute("d", d);
    area.setAttribute("d", `${d} L100,32 L0,32 Z`);
  }

  function setMetric(key, value, unit, sub, tone = "") {
    const node = $("mv-" + key);
    if (!node) return;
    node.innerHTML = value == null || value === "" ? "—"
      : `${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}`;
    const subNode = $("ms-" + key);
    if (subNode) subNode.textContent = sub || "";
    const card = $("metric-" + key);
    if (card) {
      card.classList.toggle("warn", tone === "warn");
      card.classList.toggle("bad", tone === "bad");
    }
  }

  function metricCard(key, label, extra = "") {
    return `<div class="metric" id="metric-${key}">
      <div class="metric-label">${esc(label)}</div>
      <div class="metric-value" id="mv-${key}">—</div>
      <div class="metric-sub" id="ms-${key}"></div>
      ${extra || `<svg class="metric-spark" id="sp-${key}" viewBox="0 0 100 32" preserveAspectRatio="none">
        <path class="area" opacity=".12"></path><path class="line" fill="none" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"></path></svg>`}
    </div>`;
  }

  function renderMetrics(ov) {
    $("metricGrid").innerHTML = [
      `<div class="metric model" id="metric-model">
        <div class="metric-label">模型</div>
        <div class="metric-value" id="mv-model">—</div>
        <div class="metric-sub" id="ms-model"></div>
        <dl class="model-facts" id="modelFacts"></dl>
      </div>`,
      metricCard("speed", "解码速度"),
      metricCard("gpu", "显卡利用率"),
      metricCard("vram", "显存"),
      metricCard("temp", "显卡温度"),
      metricCard("power", "功耗"),
      metricCard("pcie", "PCIe"),
      metricCard("cpu", "CPU"),
      metricCard("disk", "磁盘读取"),
    ].join("");

    const gpus = ov.machine?.gpus || [];
    const hist = ov.history || {};
    const totals = ov.totals || {};
    const live = ov.live || {};
    const last = ov.requests?.[0] || {};
    const per = (fn) => gpus.map((g) => `GPU ${g.index} ${fn(g)}`).join(" · ");
    const readTokens = Math.max(0, (totals.prompt_tokens || 0) - (totals.reused || 0));
    const avgDecode = totals.decode_ms > 0 && totals.output_tokens > 0
      ? totals.output_tokens / (totals.decode_ms / 1000) : null;
    const avgPrefill = totals.prompt_ms > 0 && readTokens > 0
      ? readTokens / (totals.prompt_ms / 1000) : null;
    const speed = live.state === "generating" ? live.tok_s
      : avgDecode != null ? avgDecode : last.decode_tps;
    const prefill = live.state === "reading" ? live.prefill_tok_s_mean
      : live.state === "generating" ? last.prefill_tps
      : avgPrefill != null ? avgPrefill : last.prefill_tps;

    $("mv-model").textContent = ov.engine?.model || ov.health?.model || "—";
    $("ms-model").textContent = [
      ov.engine?.max_context ? `${kfmt(ov.engine.max_context)} 上下文` : "",
      ov.engine?.tp ? `TP ${ov.engine.tp}` : "",
    ].filter(Boolean).join(" · ");
    // 模型卡只留身份信息；平均值放到「解码速度」和「吞吐量」卡，避免重复。
    const facts = [];
    if (totals.requests) {
      facts.push(["每请求", `${kfmt(totals.prompt_tokens / totals.requests)} in · ${fmtNum(totals.output_tokens / totals.requests, 1)} out`]);
      facts.push(["提示复用", totals.prompt_tokens ? `${fmtNum(totals.reused / totals.prompt_tokens * 100, 1)}%` : "—"]);
      facts.push(["统计起点", `${totals.since ? new Date(totals.since * 1000).toLocaleTimeString("zh-CN", { hour12: false }) : "—"} · ${fmtNum(totals.requests)} 次`]);
    } else {
      facts.push(["提示", "首个请求完成后开始统计"]);
    }
    $("modelFacts").innerHTML = facts.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");

    // 解码速度卡：一个大数字；当前值优先，空闲时回落到全程均值并说明。
    const speedKind = live.state === "generating" ? "当前" : avgDecode != null ? "全程均值" : "最近一次";
    setMetric("speed", speed == null ? null : fmtNum(speed, 1), "tok/s",
      `${speedKind} · 明细见下方吞吐量`);
    sparkline("sp-speed", hist.tok_s || ov.series?.decode);

    // 吞吐量卡顶部的当前值（下方曲线是同一组数据的历史）。
    updateThroughputHead(speed, prefill, avgDecode, avgPrefill, live.state);

    const used = gpus.reduce((a, g) => a + g.vram_used_mib, 0);
    const total = gpus.reduce((a, g) => a + g.vram_total_mib, 0);
    const util = gpus.length ? Math.max(...gpus.map((g) => g.util_pct)) : null;
    setMetric("gpu", util == null ? null : fmtNum(util, 0), "%",
      gpus.length > 1 ? per((g) => `${fmtNum(g.util_pct, 0)}%`) : (ov.hardware_static?.gpu_name || ""),
      util > 95 ? "warn" : "");
    sparkline("sp-gpu", hist.gpu_util, 100);
    setMetric("vram", total ? gb(used) : null, "GB", total ? `/ ${gb(total, 0)} GB · ${per((g) => `${gb(g.vram_used_mib)} GB`)}` : "");
    sparkline("sp-vram", hist.gpu_mem_used, total);
    const temp = gpus.length ? Math.max(...gpus.map((g) => g.temp_c)) : null;
    setMetric("temp", temp == null ? null : fmtNum(temp, 0), "°C",
      gpus.length > 1 ? per((g) => `${fmtNum(g.temp_c, 0)}°`) : "", temp > 80 ? "warn" : "");
    sparkline("sp-temp", hist.gpu_temp, 90);
    const power = gpus.reduce((a, g) => a + g.power_w, 0);
    setMetric("power", gpus.length ? fmtNum(power, 0) : null, "W",
      gpus.length > 1 ? per((g) => `${fmtNum(g.power_w, 0)} W`) : (ov.machine?.gpu_power_limit ? `上限 ${fmtNum(ov.machine.gpu_power_limit, 0)} W` : ""));
    sparkline("sp-power", hist.gpu_power, ov.machine?.gpu_power_limit);
    const gen = ov.machine?.gpu_pcie_gen_max || ov.machine?.gpu_pcie_gen;
    const width = ov.machine?.gpu_pcie_width;
    const rx = ov.machine?.gpu_pcie_rx_mb, tx = ov.machine?.gpu_pcie_tx_mb;
    const bw = (rx || tx) ? ` · 读 ${fmtNum(rx || 0, 1)} / 写 ${fmtNum(tx || 0, 1)} MB/s` : "";
    setMetric("pcie", gen ? `Gen${gen}` : null,
      width ? `x${width}` : "",
      `${gpus.length > 1 ? per((g) => `Gen${g.pcie_gen_max || g.pcie_gen || "—"} x${g.pcie_width || "—"}`) : ""}${bw}`,
      "info");
    sparkline("sp-pcie", hist.pcie_rx_mb || []);
    const cpu = lastNonZero(hist.cpu_pct) || ov.machine?.cpu_pct;
    setMetric("cpu", cpu ? fmtNum(cpu, 0) : null, "%",
      ov.hardware_static?.threads ? `${ov.hardware_static.threads} 线程` : "");
    sparkline("sp-cpu", hist.cpu_pct, 100);
    const read = lastNonZero(hist.disk_read_mb);
    const write = lastNonZero(hist.disk_write_mb);
    setMetric("disk", read ? fmtNum(read, read < 10 ? 1 : 0) : null, "MB/s",
      write ? `写入 ${fmtNum(write, 1)} MB/s` : "", "info");
    sparkline("sp-disk", hist.disk_read_mb);
  }

  function updateThroughputHead(decode, prefill, avgDecode, avgPrefill, state) {
    const show = (id, value, fallback) => {
      const node = $(id);
      if (!node) return;
      node.textContent = value != null ? fmtNum(value, value < 10 ? 1 : 0) : (fallback != null ? fmtNum(fallback, 0) : "—");
    };
    show("throughputDecode", decode, avgDecode);
    show("throughputPrefill", prefill, avgPrefill);
    const note = $("throughputNote");
    if (note) {
      note.textContent = state === "generating" || state === "reading"
        ? "实时采样"
        : "最近值 · 空闲时显示全程均值";
    }
  }

  function renderChart(series) {
    const prefill = (series?.prefill || []).slice(-30);
    const decode = (series?.decode || []).slice(-30);
    // 双 Y 轴：解码与预填充量级差一个数量级（~150 vs ~1300 tok/s），
    // 共用一条轴会把解码压成直线。各自按自己的峰值归一化，轴标签标在两侧。
    const peakDecode = Math.max(1, ...decode) * 1.15;
    const peakPrefill = Math.max(1, ...prefill) * 1.15;
    const X0 = 0, X1 = 600; // 曲线铺满整幅；刻度徽章浮在角落，不占绘图空间
    const points = (arr, peak) => arr.map((v, i) => {
      const x = arr.length === 1 ? X0 : X0 + (i / (arr.length - 1)) * (X1 - X0);
      const y = 145 - Math.min(1, v / peak) * 125;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    const axisLabel = (v) => v >= 1000 ? `${(v / 1000).toFixed(1)}k` : fmtNum(v, 0);
    $("throughputChart").innerHTML = `
      <line class="grid" x1="${X0}" y1="145" x2="${X1}" y2="145" />
      <line class="grid" x1="${X0}" y1="82" x2="${X1}" y2="82" stroke-dasharray="3 5" />
      <line class="grid" x1="${X0}" y1="20" x2="${X1}" y2="20" stroke-dasharray="3 5" />
      <polyline class="series decode" points="${points(decode, peakDecode)}" />
      <polyline class="series prefill" points="${points(prefill, peakPrefill)}" />`;
    $("tpTickDecodeMax").textContent = axisLabel(peakDecode);
    $("tpTickPrefillMax").textContent = axisLabel(peakPrefill);
  }

  const PROTOCOL_NAMES = {
    openai_chat_completions: "OpenAI 对话",
    openai_responses: "OpenAI 响应",
    anthropic_messages: "Anthropic 消息",
  };
  const FINISH_NAMES = {
    stop: "正常结束", stop_token: "正常结束", length: "达到长度上限",
    tool_calls: "调用工具", cancelled: "已取消", error: "出错",
  };
  const protocolName = (p) => PROTOCOL_NAMES[p] || p || "—";
  const finishName = (f) => FINISH_NAMES[f] || f || "—";

  function renderRequestTotals(t) {
    if (!t || !t.requests) { $("requestTotals").textContent = ""; return; }
    const since = t.since ? new Date(t.since * 1000).toLocaleString("zh-CN", { hour12: false }) : "—";
    const read = (t.prompt_tokens || 0) - (t.reused || 0);
    const pSpeed = t.prompt_ms > 0 && read > 0 ? `，${fmtNum(read / (t.prompt_ms / 1000))} tok/s` : "";
    const oSpeed = t.decode_ms > 0 && t.output_tokens > 0 ? `，${fmtNum(t.output_tokens / (t.decode_ms / 1000), 1)} tok/s` : "";
    $("requestTotals").textContent =
      `自 ${since} 起：${fmtNum(t.requests)} 次请求 · 新读提示 ${fmtNum(read)} tokens${pSpeed}（复用 ${fmtNum(t.reused || 0)}）· ` +
      `输出 ${fmtNum(t.output_tokens || 0)} tokens${oSpeed}`;
  }

  function renderRequests(list, totals) {
    const all = list || [];
    const shown = state.reqShowAll ? all : all.slice(0, 12);
    const rows = shown.map((r) => {
      const time = r.at ? new Date(r.at * 1000).toLocaleTimeString("zh-CN", { hour12: false }) : "—";
      const fresh = Math.max(0, (r.prompt_tokens || 0) - (r.reused_tokens || 0));
      const prefill = fresh > 0 && r.prefill_ms > 0 ? fmtNum(fresh / (r.prefill_ms / 1000)) : "—";
      const decode = r.decode_tps != null ? fmtNum(r.decode_tps, 1) : "—";
      const ttft = r.ttft_ms != null && r.ttft_ms > 0 ? `${fmtNum(r.ttft_ms, 0)} ms` : "—";
      const duration = r.total_s != null ? `${fmtNum(r.total_s, 1)} s` : "—";
      return `<tr>
        <td>${esc(time)}</td>
        <td><span class="tag ${r.finish ? "ok" : ""}">${esc(finishName(r.finish))}</span></td>
        <td class="num">${fmtNum(r.prompt_tokens || 0)}</td><td class="num">${fmtNum(r.reused_tokens || 0)}</td>
        <td class="num">${fmtNum(r.output_tokens || 0)}</td><td class="num">${esc(prefill)}</td><td class="num">${esc(decode)}</td>
        <td class="num">${esc(ttft)}</td><td class="num">${esc(duration)}</td>
      </tr>`;
    }).join("");
    $("requestRows").innerHTML = rows || '<tr><td colspan="9" class="hint">日志中暂无已完成的请求</td></tr>';
    const btn = $("reqAllBtn");
    btn.hidden = all.length <= 12;
    btn.textContent = state.reqShowAll ? "收起" : `显示全部 (${all.length})`;
    $("reqWrap").classList.toggle("all", state.reqShowAll);
    renderRequestTotals(totals);
  }

  async function refreshLog() {
    try {
      const data = await (await fetch(API.log)).json();
      const lines = data.lines || [];
      $("logTail").textContent = lines.length
        ? lines.map((line) => {
            try {
              const o = JSON.parse(line);
              const eventNames = {
                throughput: "吞吐", request_done: "请求完成", request_start: "请求开始",
                request_error: "请求出错", request_rejected: "请求被拒", server_start: "服务启动",
              };
              const parts = [eventNames[o.event] || o.event, o.request?.request_id ?? "", o.throughput_tokens_per_second
                ? `预填充=${(o.throughput_tokens_per_second.prefill || 0).toFixed(0)} 解码=${(o.throughput_tokens_per_second.decode || 0).toFixed(0)}`
                : ""];
              return `${new Date((o.timestamp_unix_ms || 0) / 1000).toLocaleTimeString("zh-CN", { hour12: false })} ${parts.filter(Boolean).join(" ")}`;
            } catch (e) { return line.slice(0, 240); }
          }).join("\n")
        : "日志为空";
    } catch (err) { $("logTail").textContent = String(err); }
  }

  function renderAbout(ov) {
    const h = ov.health || {};
    const m = ov.machine || {};
    const pairs = (obj) => Object.entries(obj)
      .filter(([, v]) => v != null && v !== "")
      .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
    $("aboutModel").innerHTML = pairs({
      "模型": h.model || ov.engine?.model || "—",
      "上下文": h.max_context ? `${fmtNum(h.max_context)} tokens` : "—",
      "KV 量化": ov.engine?.kv || "—",
      "推测解码": ov.engine?.spec === "mtp" ? `MTP · 最多 ${Math.max(0, (ov.engine.draft_tokens || 4) - 1)} tokens` : "未启用",
      "图片输入": h.images ? "开启" : "关闭",
      "张量并行": h.tp != null ? `TP ${h.tp}` : "—",
      "最大并发": h.max_concurrency ?? "—",
      "模型驻留": (ov.residency || {}).state || "—",
      "接口密钥": h.api_key ? "已设置" : "未设置",
    });
    $("aboutMachine").innerHTML = pairs({
      "主机": m.host || "—",
      "显卡": (m.gpus || []).map((g) => `${g.name}${g.vram_total_mib ? ` · ${fmtNum(g.vram_total_mib / 1024, 0)} GB` : ""}`).join(" / ") || "—",
      "CPU": String(m.threads || "").trim() ? `${String(m.threads).trim()} 线程` : (ov.hardware_static?.threads ? `${ov.hardware_static.threads} 线程` : "—"),
      "内存": m.ram?.total_bytes ? `${fmtNum(m.ram.total_bytes / 2 ** 30, 0)} GB` : "—",
    });

    const base = location.origin.replace(/:\d+$/, ":8080");
    const modelName = h.model || ov.engine?.model || "—";
    $("openaiEndpoint").textContent = base + "/v1";
    $("anthropicEndpoint").textContent = base;
    $("aboutModelName").textContent = modelName;
  }

  function renderPill(ov) {
    const pill = $("statusPill");
    const residency = ov.residency?.state || "";
    const offloaded = /host|ram|offload|unloaded/i.test(residency);
    const liveState = ov.live?.state || "";
    let mode = "idle", text = "空闲";
    if (!ov.upstream_ok || liveState === "error") { mode = "offline"; text = "服务不可达"; }
    else if (offloaded || liveState === "unloaded") { mode = "offloaded"; text = "模型已卸载"; }
    else if (liveState === "reading") { mode = "reading"; text = "读取提示"; }
    else if (liveState === "generating") { mode = "generating"; text = "生成中"; }
    else if ((ov.live?.queued || 0) > 0) { mode = "queued"; text = "排队中"; }
    pill.dataset.state = mode;
    $("statusText").textContent = text;

    $("unloadBtn").disabled = !ov.upstream_ok;
    $("unloadLabel").textContent = offloaded || liveState === "unloaded" ? "加载模型" : "卸载模型";
  }

  function renderContext(ov) {
    const c = ov.context || {};
    const live = ov.live || {};
    const ctx = c.max_tokens || 0;
    const parts = {
      cached: c.cached || 0,
      reading: c.reading || 0,
      answer: c.answering || 0,
    };
    const used = parts.cached + parts.reading + parts.answer;
    const scale = ctx > 0 && used > ctx ? ctx / used : 1;
    const ARC = 235.6;
    let at = 0;
    for (const [key, id] of [["cached", "ctxSegCached"], ["reading", "ctxSegRead"], ["answer", "ctxSegAnswer"]]) {
      const node = $(id);
      const len = ctx ? ARC * parts[key] * scale / ctx : 0;
      node.setAttribute("stroke-dasharray", `${len.toFixed(1)} 314.2`);
      node.setAttribute("stroke-dashoffset", `${(-at).toFixed(1)}`);
      node.style.opacity = len >= 1 ? "1" : "0";
      at += len;
    }
    // C++ 聚合层返回未取整的浮点百分比，这里统一格式化避免长尾小数。
    $("ctxPct").textContent = c.pct != null ? `${Number(c.pct).toFixed(1)}%` : "—";
    $("ctxSub").textContent = ctx ? `${kfmt(used)} / ${kfmt(ctx)}` : "—";
    const legs = { cached: "ctxLegCached", reading: "ctxLegRead", answer: "ctxLegAnswer", free: "ctxLegFree" };
    for (const [key, id] of Object.entries(legs)) {
      $(id).textContent = ctx || key !== "free" ? kfmt(c.free != null && key === "free" ? c.free : parts[key]) : "—";
    }
    for (const [key, id] of [["reading", "ctxRowRead"], ["answer", "ctxRowAnswer"]]) {
      $(id).classList.toggle("is-now", live.state === key);
    }
    $("ctxNote").textContent = (live.queued || 0) > 0
      ? `排队 ${fmtNum(live.queued)} 个请求；分段显示的是最近一次请求。`
      : "按最近一次请求的提示缓存 / 回答长度与上下文上限计算。";
  }

  function renderResourceBars(ov) {
    const m = ov.machine || {};
    // 前缀缓存预算：ninfer 用 host KV 预算缓存提示前缀，进度条显示累计复用占比。
    const cache = ov.conversation_cache || {};
    const budget = cache.budget_tokens || 0;
    const reusedPct = cache.prompt_tokens
      ? Math.min(100, (cache.reused_tokens || 0) / cache.prompt_tokens * 100) : 0;
    $("slotsText").textContent = budget
      ? `${kfmt(budget)} tokens · 复用 ${fmtNum(reusedPct, 1)}%`
      : "未启用";
    $("slotsBar").style.width = `${reusedPct}%`;
    const ram = m.ram || {};
    const ramPct = ram.total_bytes ? ram.used_bytes / ram.total_bytes * 100 : null;
    $("ramText").textContent = ram.total_bytes
      ? `${fmtNum(ram.used_bytes / 2 ** 30, 1)} / ${fmtNum(ram.total_bytes / 2 ** 30, 1)} GiB`
      : "—";
    $("ramBar").style.width = `${ramPct || 0}%`;
    $("ramBar").parentElement.dataset.tone = ramPct > 92 ? "bad" : "";
    $("tempText").textContent = m.gpu_temp == null ? "—" : `${fmtNum(m.gpu_temp, 0)} °C`;
    $("tempBar").style.width = `${m.gpu_temp == null ? 0 : Math.min(100, m.gpu_temp)}%`;
  }

  function renderConversationCache(ov) {
    const c = ov.conversation_cache || {};
    $("ccCard").hidden = false;
    $("ccSummary").textContent = c.requests
      ? `${fmtNum(c.requests_reused || 0)} / ${fmtNum(c.requests)} 次请求复用了提示`
      : "";
    const rows = [
      ["最近请求", c.last_prompt != null ? `${fmtNum(c.last_reused || 0)} / ${fmtNum(c.last_prompt)} tokens 复用` : null],
      ["累计复用", c.requests ? `${fmtNum(c.reused_tokens || 0)} tokens（${fmtNum(c.prompt_tokens ? c.reused_tokens / c.prompt_tokens * 100 : 0, 1)}%）` : null],
      ["前缀缓存预算", c.budget_tokens ? `${fmtNum(c.budget_tokens)} tokens` : null],
      ["磁盘归档", c.archive ? `${c.archive}${c.archive_max_mib ? ` · ${fmtNum(c.archive_max_mib)} MiB` : ""}` : "未启用"],
      ["统计起点", c.since ? new Date(c.since * 1000).toLocaleString("zh-CN", { hour12: false }) : null],
    ].filter(([, v]) => v != null);
    $("ccFacts").innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")
      || '<dt>提示</dt><dd>首个请求完成后开始统计</dd>';
    $("ccNote").textContent = c.note || "";
  }

  function renderMcp(ov) {
    const servers = ov.mcp?.servers || [];
    $("mcpCard").hidden = !servers.length;
    if (!servers.length) return;
    $("mcpSummary").textContent = `${fmtNum(ov.mcp.tools || 0)} 个工具`;
    $("mcpList").innerHTML = servers.map((s) =>
      `<span class="flag"><b>${esc(s.name || "未命名")}</b> ${esc(s.status || "")}</span>`).join("");
  }

  function renderMonitor(ov) {
    const live = ov.live || {};
    const mode = live.state || $("statusPill").dataset.state || "idle";
    const stateText = {
      idle: "等待请求", unloaded: "模型已卸载", reading: "正在读取提示词",
      generating: "正在生成回答", queued: "请求排队中", error: "服务不可达",
    };
    const active = new Set([mode]);
    if ((live.queued || 0) > 0) active.add("queued");
    if (mode === "generating" && (live.prefilling || 0) > 0) active.add("reading");
    document.querySelectorAll("#stateBadges .badge").forEach((b) =>
      b.classList.toggle("on", active.has(b.dataset.s)));
    $("stateLabel").textContent = stateText[mode] || "—";
    $("stateDetail").textContent = `运行 ${live.running ?? "—"} · 预填充 ${live.prefilling ?? "—"} · 排队 ${live.queued ?? "—"}`;
    const last = ov.requests?.[0] || {};
    $("stateHint").textContent = mode === "generating"
      ? `${fmtNum(last.decode_tps ?? live.tok_s, 1)} tok/s`
      : last.id ? `最近请求 #${esc(last.id)}` : "";
    let pct = mode === "idle" ? 100 : 4;
    if (mode === "reading" && live.prompt_total) pct = Math.round(live.prompt_read / live.prompt_total * 100);
    else if (mode === "generating") pct = ov.context?.pct || 72;
    $("stateBar").style.width = `${Math.max(0, Math.min(100, pct))}%`;

    renderMetrics(ov);
    renderChart(ov.series);
    renderRequests(ov.requests, ov.totals);
    renderContext(ov);
    renderResourceBars(ov);
    renderConversationCache(ov);
    renderMcp(ov);
    renderAbout(ov);
  }

  async function refreshOverview() {
    try {
      const ov = await (await fetch(API.overview)).json();
      state.overview = ov;
      renderPill(ov);
      if (state.view === "monitor") renderMonitor(ov);
      if (state.view === "about") renderAbout(ov);
    } catch (err) {
      $("statusPill").dataset.state = "offline";
      $("statusText").textContent = "面板连接失败";
    }
  }

  function startPolling() {
    if (state.poll) clearInterval(state.poll);
    state.poll = setInterval(refreshOverview, 3000);
  }

  function viewFromHash() {
    const value = location.hash.replace(/^#/, "");
    return ["chat", "monitor", "about"].includes(value) ? value : "chat";
  }

  function switchView(view, updateHash = true) {
    // 与 Strata 一致：视图写进 URL 的 hash，刷新或分享链接都停在同一个标签页。
    if (updateHash && location.hash !== "#" + view) {
      location.hash = view;
      return; // hashchange 事件会再进来，保证前进/后退也能切换
    }
    state.view = view;
    for (const name of ["chat", "monitor", "about"]) {
      const node = $("view-" + name);
      if (node) node.hidden = name !== view;
      const tab = document.querySelector(`.tab[data-view="${name}"]`);
      if (tab) {
        tab.classList.toggle("active", name === view);
        tab.setAttribute("aria-selected", name === view ? "true" : "false");
      }
    }
    if (view === "monitor") {
      refreshOverview();
      refreshLog();
      startPolling();
    } else if (view === "chat") {
      if (state.poll) { clearInterval(state.poll); state.poll = null; }
    } else {
      refreshOverview();
      if (state.poll) { clearInterval(state.poll); state.poll = null; }
    }
  }

  // ------------------------------------------------------------------ init
  function bind() {
    document.querySelectorAll(".tab").forEach((btn) =>
      btn.addEventListener("click", () => switchView(btn.dataset.view)));
    window.addEventListener("hashchange", () => {
      const view = viewFromHash();
      if (location.hash !== "#" + view) { location.hash = view; return; }
      switchView(view, false);
    });

    $("themeBtn").addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      save("theme", next);
    });
    document.documentElement.dataset.theme = load("theme", "dark");

    $("newChatBtn").addEventListener("click", () => {
      state.activeId = null;
      activeChat();
      persistChats();
      renderChatList();
      renderMessages();
      $("prompt").focus();
    });
    $("chatList").addEventListener("click", (e) => {
      const del = e.target.closest(".del-chat");
      if (del) {
        state.chats = state.chats.filter((c) => c.id !== del.dataset.id);
        if (state.activeId === del.dataset.id) state.activeId = null;
        persistChats();
        renderChatList();
        renderMessages();
        return;
      }
      const item = e.target.closest(".chat-item");
      if (item) {
        state.activeId = item.dataset.id;
        persistChats();
        renderChatList();
        renderMessages();
      }
    });
    $("clearChatBtn").addEventListener("click", () => {
      const chat = activeChat();
      chat.messages = [];
      persistChats();
      renderMessages();
    });
    $("closeSettingsBtn").addEventListener("click", () => {
      $("settings").style.display = "none";
    });

    const fields = [
      ["apiKey", "apiKey"], ["effort", "effort"], ["temperature", "temperature"],
      ["topP", "topP"], ["topK", "topK"], ["maxTokens", "maxTokens"], ["streamChk", "stream"],
    ];
    for (const [id, key] of fields) {
      const node = $(id);
      const event = node.type === "checkbox" || node.tagName === "SELECT" || node.type === "range" ? "change" : "input";
      node.addEventListener(event, () => {
        state.settings[key] = node.type === "checkbox" ? node.checked
          : node.type === "range" || node.type === "number" ? Number(node.value) : node.value;
        if (id === "temperature") $("temperatureOut").textContent = Number(node.value).toFixed(2);
        if (id === "topP") $("topPOut").textContent = Number(node.value).toFixed(2);
        if (id === "topK") $("topKOut").textContent = node.value;
        persistSettings();
      });
    }

    $("prompt").addEventListener("input", () => {
      $("prompt").style.height = "auto";
      $("prompt").style.height = Math.min(160, $("prompt").scrollHeight) + "px";
    });
    $("prompt").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("composer").requestSubmit(); }
    });
    $("composer").addEventListener("submit", (e) => { e.preventDefault(); sendMessage(); });

    $("attachBtn").addEventListener("click", () => $("imageInput").click());
    $("imageInput").addEventListener("change", () => {
      for (const file of $("imageInput").files) {
        const reader = new FileReader();
        reader.onload = () => {
          state.pendingImages.push(reader.result);
          $("composerNote").textContent = `已附加 ${state.pendingImages.length} 张图片 · Enter 发送`;
        };
        reader.readAsDataURL(file);
      }
      $("imageInput").value = "";
    });

    $("unloadBtn").addEventListener("click", async () => {
      const label = $("unloadLabel").textContent;
      if (!confirm(`确认${label}？正在进行的请求可能受影响。`)) return;
      const target = label === "加载模型" ? "load" : "offload";
      try {
        const headers = { "Content-Type": "application/json" };
        if (EMBEDDED && state.settings.apiKey) {
          headers.Authorization = "Bearer " + state.settings.apiKey;
        } else if (!EMBEDDED) {
          headers["X-Admin-Token"] = "";
        }
        const res = await fetch(API.model(target), { method: "POST", headers });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || res.status);
      } catch (err) { alert("操作失败：" + err.message); }
      refreshOverview();
    });

    $("reqAllBtn").addEventListener("click", () => {
      state.reqShowAll = !state.reqShowAll;
      if (state.overview) renderRequests(state.overview.requests, state.overview.totals);
    });
    $("aboutApiKey").addEventListener("input", () => {
      state.settings.apiKey = $("aboutApiKey").value;
      $("apiKey").value = state.settings.apiKey;
      persistSettings();
    });
    $("refreshLogBtn").addEventListener("click", refreshLog);
    document.querySelectorAll("[data-copy]").forEach((btn) =>
      btn.addEventListener("click", async () => {
        const text = $(btn.dataset.copy).textContent;
        const isIcon = btn.classList.contains("copy-icon");
        const original = isIcon ? "⧉" : "复制";
        try {
          await navigator.clipboard.writeText(text);
          btn.textContent = isIcon ? "✓" : "已复制";
        } catch (e) {
          btn.textContent = isIcon ? "×" : "复制失败";
        }
        setTimeout(() => { btn.textContent = original; }, 1200);
      }));
  }

  function init() {
    state.settings = Object.assign(state.settings, load("settings", {}));
    // 旧版本保存过 off / high：迁移到 Qwen3.8 真正支持的四档。
    if (state.settings.effort === "off") state.settings.effort = "none";
    if (state.settings.effort === "high" || state.settings.effort === "max") state.settings.effort = "xhigh";
    if (!["none", "low", "medium", "xhigh"].includes(state.settings.effort)) state.settings.effort = "none";
    state.chats = load("chats", []);
    state.activeId = load("active", state.chats[0]?.id || null);
    syncSettings();
    bind();
    renderChatList();
    renderMessages();
    refreshOverview();
    switchView(viewFromHash(), false); // 支持 /#monitor 直接打开监控页，刷新后保持
    setInterval(refreshOverview, 15000);
  }

  init();
})();
