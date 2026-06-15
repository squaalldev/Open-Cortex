const app = document.getElementById("app");
const phase = document.getElementById("phase");
const coreState = document.getElementById("core-state");
const coreDetail = document.getElementById("core-detail");
const kv = document.getElementById("kv");
const contextUsed = document.getElementById("context-used");
const contextUnit = document.getElementById("context-unit");
const contextFill = document.getElementById("context-fill");
const tps = document.getElementById("tps");
const ttft = document.getElementById("ttft");
const memoryState = document.getElementById("memory-state");
const contextState = document.getElementById("context-state");
const tokenState = document.getElementById("token-state");
const healthState = document.getElementById("health-state");
const scenarioLabel = document.getElementById("scenario-label");
const memoryBlocks = [...document.querySelectorAll("#memory-blocks i")];
const messages = document.getElementById("messages");
const runtimeEvent = document.getElementById("runtime-event");
const evidence3Label = document.getElementById("evidence-3-label");
const evidence3Value = document.getElementById("kv-retained");
const promptEval = document.getElementById("prompt-eval");
const prompt = document.getElementById("prompt");
const send = document.getElementById("send");
const experiment = document.getElementById("experiment");
const runExperimentButton = document.getElementById("run-experiment");
const workspace = document.querySelector(".workspace");
const resizer = document.getElementById("resizer");
const drawerToggle = document.getElementById("drawer-toggle");
const memoryOrgan = document.getElementById("memory-organ");
const contextOrgan = document.getElementById("context-organ");
const tokensOrgan = document.getElementById("tokens-organ");
const engineOrgan = document.getElementById("engine-organ");

let chatHistory = [];
let activeAssistant = null;
let previousChatWidth = workspace.getBoundingClientRect().width * 0.4;
let pendingChatWidth = previousChatWidth;
let collapseHoldUntil = 0;
let contextWindowFullPending = false;
const collapseThreshold = 230;

function finishActiveAssistant() {
  if (activeAssistant) {
    activeAssistant.isStreaming = false;
  }
}

function clearHazardState() {
  app.classList.remove("hazard-loop", "hazard-context-full");
  runtimeEvent.classList.remove("hazard", "critical");
}

function setHazardState(kind, critical = false) {
  clearHazardState();
  app.classList.add(kind);
  runtimeEvent.classList.add("hazard");
  runtimeEvent.classList.toggle("critical", critical);
}

function showRuntimeEvent(label, sticky = false) {
  runtimeEvent.textContent = label;
  runtimeEvent.classList.remove("visible", "sticky");
  void runtimeEvent.offsetWidth;
  runtimeEvent.classList.add("visible");
  runtimeEvent.classList.toggle("sticky", sticky);
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderInlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

function renderMarkdown(value) {
  const lines = value.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let paragraph = [];
  let list = [];
  let inCode = false;
  let codeLines = [];
  let codeLang = "";

  function flushParagraph() {
    if (!paragraph.length) return;
    html.push(`<p>${paragraph.map(renderInlineMarkdown).join("<br>")}</p>`);
    paragraph = [];
  }

  function flushList() {
    if (!list.length) return;
    html.push(`<ul>${list.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ul>`);
    list = [];
  }

  for (const line of lines) {
    const codeFence = line.match(/^```(\w+)?\s*$/);
    if (codeFence) {
      if (inCode) {
        html.push(`<pre><code${codeLang ? ` data-lang="${escapeHtml(codeLang)}"` : ""}>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        inCode = false;
        codeLines = [];
        codeLang = "";
      } else {
        flushParagraph();
        flushList();
        inCode = true;
        codeLang = codeFence[1] || "";
      }
      continue;
    }

    if (inCode) {
      codeLines.push(line);
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      flushList();
      continue;
    }

    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length;
      html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    const bullet = line.match(/^\s*[-*]\s+(.+)$/);
    if (bullet) {
      flushParagraph();
      list.push(bullet[1]);
      continue;
    }

    paragraph.push(line);
  }

  if (inCode) {
    html.push(`<pre><code${codeLang ? ` data-lang="${escapeHtml(codeLang)}"` : ""}>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
  }
  flushParagraph();
  flushList();
  return html.join("");
}

function renderMessages() {
  if (!chatHistory.length) {
    messages.classList.remove("context-limited");
    messages.innerHTML = `
      <div class="message">
        <div class="message-meta">Cortex</div>
        <div class="bubble">Send a message to begin observing local inference.</div>
      </div>
    `;
    return;
  }

  const hasCollapsedContext = chatHistory.some((message) => message.outsideContext);
  messages.classList.toggle("context-limited", hasCollapsedContext);
  let boundaryInserted = false;

  messages.innerHTML = chatHistory.map((message, index) => {
    const role = message.role === "user" ? "user" : "";
    const live = message.role === "assistant" && message.isStreaming ? " assistant-live" : "";
    const outside = message.outsideContext ? " historical-context outside-context" : "";
    const ejected = message.contextEjected ? " context-ejected" : "";
    const author = message.role === "user" ? "You" : "Cortex";
    const cursor = live ? '<span class="cursor"></span>' : "";
    const boundary = !boundaryInserted && hasCollapsedContext && !message.outsideContext
      ? '<div class="active-context-boundary"><span>Active context boundary · outside active context above</span></div>'
      : "";
    if (boundary) boundaryInserted = true;
    const body = message.role === "assistant" ? renderMarkdown(message.content) : escapeHtml(message.content);
    return `
      ${boundary}
      <div class="message ${role}${live}${outside}${ejected}">
        <div class="message-meta">${author}</div>
        <div class="bubble markdown-body">${body}${cursor}</div>
        ${message.outsideContext ? '<div class="context-badge">outside active context</div>' : ""}
      </div>
    `;
  }).join("");
  messages.scrollTop = messages.scrollHeight;
}

function setPhase(name, label, core, detail) {
  const preserveCollapse = app.classList.contains("scenario-collapse") && Date.now() < collapseHoldUntil;
  const removableClasses = [
    "phase-idle",
    "phase-prefill",
    "phase-decode",
    "phase-recovery",
    "scenario-loop",
    "scenario-memory",
    "scenario-slow",
    "scenario-collapse",
    "hazard-loop",
    "hazard-context-full"
  ].filter((className) => !(preserveCollapse && className === "scenario-collapse"));

  app.classList.remove(...removableClasses);
  app.classList.add("phase-" + name);
  phase.textContent = label;
  coreState.textContent = core;
  coreDetail.textContent = detail;
}

function resetRuntimeVisuals() {
  app.classList.remove(
    "scenario-loop",
    "scenario-memory",
    "scenario-slow",
    "scenario-collapse",
    "core-slow",
    "hazard-loop",
    "hazard-context-full"
  );
  clearHazardState();
  collapseHoldUntil = 0;
  contextWindowFullPending = false;
  memoryOrgan.className = "organ memory active";
  contextOrgan.className = "organ context active";
  tokensOrgan.className = "organ tokens active";
  engineOrgan.className = "organ health active";
  runtimeEvent.classList.remove("visible", "sticky");
}

function setMetrics({ contextTokens, contextSize, speed, firstToken, promptRate }) {
  const used = contextTokens == null ? "—" : Number(contextTokens).toLocaleString();
  const size = contextSize == null ? "—" : Number(contextSize).toLocaleString();
  contextUsed.textContent = used;
  contextUnit.textContent = `/ ${size}`;
  const fill = contextTokens && contextSize ? Math.min(100, Math.round(contextTokens / contextSize * 100)) : 0;
  contextFill.style.width = fill + "%";
  kv.textContent = contextTokens && contextSize ? String(fill) : "—";
  tps.textContent = speed == null ? "—" : Number(speed).toFixed(1);
  ttft.textContent = firstToken == null ? "—" : Number(firstToken).toFixed(0);
  promptEval.textContent = promptRate == null ? "measuring" : `${Number(promptRate).toFixed(1)} tok/s`;

  const filled = Math.max(1, Math.min(12, Math.round(fill / 100 * memoryBlocks.length)));
  memoryBlocks.forEach((block, index) => block.classList.toggle("filled", index < filled));

  memoryOrgan.classList.toggle("memory-pressure", fill >= 70 && fill < 85);
  memoryOrgan.classList.toggle("memory-overloaded", fill >= 85);
  contextOrgan.classList.toggle("context-pressure", fill >= 70);

  if (fill >= 85) {
    memoryState.textContent = "Memory overloaded";
    contextState.textContent = "Near context limit";
  } else if (fill >= 70) {
    memoryState.textContent = "Memory strained";
    contextState.textContent = "Context pressure";
  } else if (fill >= 45) {
    memoryState.textContent = "Memory busy";
    contextState.textContent = "Context growing";
  }
}

function setRuntimeIdle() {
  resetRuntimeVisuals();
  scenarioLabel.textContent = "Live local chat";
  setPhase("idle", "IDLE", "Ready for input", "ENGINE IDLE · SLOT AVAILABLE");
  runtimeEvent.textContent = "READY";
  memoryState.textContent = "Memory quiet";
  contextState.textContent = "Context resting";
  tokenState.textContent = "Stream dormant";
  healthState.textContent = "Engine healthy";
  kv.textContent = "—";
  setMetrics({ contextTokens: 0, contextSize: null, speed: 0, firstToken: null, promptRate: null });
  evidence3Label.textContent = "KV evidence";
  evidence3Value.textContent = "context proxy";
}

function applyRuntimeEvent(event) {
  if (event.kind === "context_collapse") {
    applyContextCollapse(event);
    return;
  }

  if (event.kind === "request_started") {
    clearHazardState();
    scenarioLabel.textContent = "Live local chat";
    setPhase("prefill", "PREFILL", "Waiting for first token", "PREFILL · PROCESSING PROMPT");
    showRuntimeEvent("PREFILL STARTED");
    memoryState.textContent = "Memory allocating";
    contextState.textContent = "Context pending";
    tokenState.textContent = "Stream waiting";
    healthState.textContent = "Engine charging";
    kv.textContent = "—";
    evidence3Label.textContent = "KV proxy";
    evidence3Value.textContent = "waiting for context";
    return;
  }

  if (event.kind === "first_token") {
    const snapshot = event.snapshot || {};
    const contextTokens = event.context_tokens ?? snapshot.slot_context_tokens?.[0] ?? null;
    const contextSize = event.context_size ?? snapshot.slot_context_size ?? null;
    setPhase("decode", "DECODE", "Generating response", `TTFT ${Number(event.ttft_ms || 0).toFixed(1)} ms · LIVE`);
    showRuntimeEvent("FIRST TOKEN");
    memoryState.textContent = "Memory holding";
    contextState.textContent = "Context active";
    tokenState.textContent = "Decode flowing";
    healthState.textContent = `processing=${snapshot.requests_processing ?? "—"} deferred=${snapshot.requests_deferred ?? "—"}`;
    setMetrics({
      contextTokens,
      contextSize,
      speed: snapshot.decode_tps,
      firstToken: event.ttft_ms,
      promptRate: snapshot.prompt_tps
    });
    evidence3Label.textContent = "KV proxy";
    evidence3Value.textContent = event.working_memory_percent == null ? "context-derived" : `${event.working_memory_percent.toFixed(1)}% context-derived`;
    return;
  }

  if (event.kind === "token") {
    if (event.repetition_detected) {
      app.classList.add("scenario-loop");
      setHazardState("hazard-loop", true);
      showRuntimeEvent("THOUGHT LOOP DETECTED", true);
      coreState.textContent = "Thought loop emerging";
      tokenState.textContent = `Thought loop · ${event.live_tps ? event.live_tps.toFixed(1) : "—"} tok/s`;
      healthState.textContent = "Repetition detected";
      evidence3Label.textContent = "Generation pattern";
      evidence3Value.textContent = "thought loop detected";
    } else {
      runtimeEvent.textContent = "TOKEN STREAM";
      coreState.textContent = "Generating response";
      tokenState.textContent = `Decode flowing · ${event.live_tps ? event.live_tps.toFixed(1) : "—"} tok/s`;
      healthState.textContent = "Engine active";
      evidence3Label.textContent = "Generation pattern";
      evidence3Value.textContent = "no repetition detected";
    }
    phase.textContent = "DECODE";
    coreDetail.textContent = `DECODE TOKEN ${event.generated_tokens || "—"} · LIVE`;
    setMetrics({
      contextTokens: event.context_tokens,
      contextSize: event.context_size,
      speed: event.live_tps,
      firstToken: Number(ttft.textContent) || null,
      promptRate: null
    });
    promptEval.textContent = `${event.generated_tokens || 0} generated tokens`;
    if (event.working_memory_percent != null) {
      evidence3Label.textContent = "KV proxy";
      evidence3Value.textContent = `${event.working_memory_percent.toFixed(1)}% context-derived`;
    }
    return;
  }

  if (event.kind === "request_completed") {
    finishActiveAssistant();
    const contextFull = event.context_size && event.context_tokens >= event.context_size;
    phase.textContent = "DONE";
    showRuntimeEvent(
      contextFull ? "CONTEXT WINDOW FULL" : (event.repetition_detected ? "THOUGHT LOOP DETECTED" : "REQUEST COMPLETE"),
      contextFull || event.repetition_detected
    );
    coreState.textContent = contextFull
      ? "Context full; next turn will evict earliest memory"
      : (event.repetition_detected ? "Generation ended in a loop" : "Generation complete");
    coreDetail.textContent = "FINAL TIMINGS RECEIVED";
    setMetrics({
      contextTokens: event.context_tokens,
      contextSize: event.context_size,
      speed: event.decode_tps,
      firstToken: Number(ttft.textContent) || null,
      promptRate: event.prompt_tps
    });
    tokenState.textContent = `Decode ${event.decode_tps ? event.decode_tps.toFixed(1) : "—"} tok/s`;
    healthState.textContent = "Engine settled";
    promptEval.textContent = `Prefill ${event.prompt_tps ? event.prompt_tps.toFixed(1) : "—"} tok/s`;
    evidence3Label.textContent = "Output tokens";
    evidence3Value.textContent = event.completion_tokens ?? "—";
    if (contextFull) {
      applyContextFullWarning();
    } else {
      contextState.textContent = event.repetition_detected ? "Context saturated by loop" : "Context retained";
    }
    renderMessages();
  }
}

function applyContextFullWarning() {
  contextWindowFullPending = true;
  collapseHoldUntil = Date.now() + 2500;
  scenarioLabel.textContent = "Context full";
  setHazardState("hazard-context-full", true);
  app.classList.add("scenario-collapse");
  contextOrgan.classList.add("context-pressure");
  memoryOrgan.classList.add("memory-overloaded");
  engineOrgan.classList.add("engine-strained");
  runtimeEvent.textContent = "CONTEXT WINDOW FULL";
  runtimeEvent.classList.add("visible", "sticky");
  memoryState.textContent = "Memory overloaded";
  contextState.textContent = "Next turn will evict earliest message";
  tokenState.textContent = "Stream complete";
  healthState.textContent = "Awaiting next turn";
  evidence3Label.textContent = "Next action";
  evidence3Value.textContent = "pop earliest turn";
}

function applyDeferredContextEjection() {
  if (!contextWindowFullPending) return false;

  const target = chatHistory.find((message) => !message.outsideContext);
  if (!target) return false;

  target.outsideContext = true;
  target.contextEjected = true;
  contextWindowFullPending = false;
  clearHazardState();
  collapseHoldUntil = Date.now() + 2500;
  scenarioLabel.textContent = "Context collapse";
  setPhase("decode", "COLLAPSE", "Context boundary shifted", "EARLIEST TURN OUTSIDE ACTIVE MEMORY");
  app.classList.add("scenario-collapse");
  showRuntimeEvent("CONTEXT COLLAPSE", true);
  memoryState.textContent = "Memory rebalanced";
  contextState.textContent = "Earliest message ejected";
  tokenState.textContent = "Sending recent context";
  healthState.textContent = "Engine recovering";
  evidence3Label.textContent = "Dropped turns";
  evidence3Value.textContent = "1";
  return true;
}

function applyContextCollapse(event) {
  const dropped = Math.max(1, Number(event.dropped_messages || 1));
  let marked = 0;
  for (const message of chatHistory) {
    if (message === activeAssistant) continue;
    if (marked >= dropped) break;
    message.outsideContext = true;
    message.contextEjected = marked === 0;
    marked += 1;
  }

  collapseHoldUntil = Date.now() + 2500;
  clearHazardState();
  scenarioLabel.textContent = "Context collapse";
  setPhase("decode", "COLLAPSE", "Context boundary shifted", "EARLIEST TURNS OUTSIDE ACTIVE MEMORY");
  app.classList.add("scenario-collapse");
  showRuntimeEvent("CONTEXT COLLAPSE", true);
  memoryState.textContent = "Memory rebalanced";
  contextState.textContent = "Earliest turns forgotten";
  tokenState.textContent = "Retrying with recent context";
  healthState.textContent = "Recovered locally";
  contextUsed.textContent = "recent";
  contextUnit.textContent = event.context_size ? `/ ${event.context_size}` : "/ active";
  contextFill.style.width = "58%";
  kv.textContent = "58";
  evidence3Label.textContent = "Dropped turns";
  evidence3Value.textContent = String(event.dropped_messages ?? marked);

  if (activeAssistant && !activeAssistant.content) {
    activeAssistant.content = "Earlier turns fell outside the active context.\n\n";
  }
  renderMessages();
}

function seedExperimentMessages({ outside = false } = {}) {
  chatHistory = [
    {
      role: "user",
      content: "Why does a long conversation make the model feel slower?",
      outsideContext: outside
    },
    {
      role: "assistant",
      content: "Each new turn carries more history back into the model, increasing prefill work before decoding begins.",
      outsideContext: outside
    },
    {
      role: "user",
      content: "Can you show me what changes inside the runtime?"
    },
    {
      role: "assistant",
      content: "Watch the runtime organs: context fills, working memory holds KV blocks, and token flow reveals decode health."
    }
  ];
  activeAssistant = chatHistory.at(-1);
  renderMessages();
}

function runLongContextExperiment() {
  resetRuntimeVisuals();
  seedExperimentMessages();
  scenarioLabel.textContent = "Long context stress";
  setPhase("prefill", "PREFILL", "Reading long context", "TTFT RISING · CONTEXT PRESSURE");
  showRuntimeEvent("LONG CONTEXT STRESS", true);
  contextOrgan.classList.add("context-pressure");
  engineOrgan.classList.add("engine-strained");
  memoryState.textContent = "Memory busy";
  contextState.textContent = "Near context limit";
  tokenState.textContent = "Stream waiting";
  healthState.textContent = "Engine strained by prefill";
  setMetrics({ contextTokens: 1840, contextSize: 2048, speed: 0, firstToken: 1380, promptRate: 38.6 });
  evidence3Label.textContent = "Experiment";
  evidence3Value.textContent = "simulated long prompt";
}

function runMemoryPressureExperiment() {
  resetRuntimeVisuals();
  seedExperimentMessages();
  scenarioLabel.textContent = "Memory pressure";
  app.classList.add("scenario-memory");
  setPhase("decode", "STRAINED", "KV blocks reallocating", "WORKING SET PRESSURE · SIMULATED");
  showRuntimeEvent("MEMORY PRESSURE", true);
  memoryOrgan.classList.add("memory-pressure");
  engineOrgan.classList.add("engine-strained");
  memoryState.textContent = "KV blocks fragmenting";
  contextState.textContent = "Context stable";
  tokenState.textContent = "Decode flowing";
  healthState.textContent = "Engine strained";
  setMetrics({ contextTokens: 1240, contextSize: 2048, speed: 24.8, firstToken: 648, promptRate: 72.4 });
  kv.textContent = "91";
  evidence3Label.textContent = "KV proxy";
  evidence3Value.textContent = "simulated pressure";
}

function runSlowDecodeExperiment() {
  resetRuntimeVisuals();
  seedExperimentMessages();
  scenarioLabel.textContent = "Slow decode";
  app.classList.add("scenario-slow", "core-slow");
  setPhase("decode", "DECODE", "Engine decoding slowly", "TOKEN FLOW INTERRUPTED · SIMULATED");
  showRuntimeEvent("SLOW DECODE", true);
  tokensOrgan.classList.add("tokens-slow");
  engineOrgan.classList.add("engine-strained");
  memoryState.textContent = "Memory healthy";
  contextState.textContent = "Context stable";
  tokenState.textContent = "Flow interrupted";
  healthState.textContent = "Decode scheduler strained";
  setMetrics({ contextTokens: 980, contextSize: 2048, speed: 6.8, firstToken: 510, promptRate: 86.2 });
  evidence3Label.textContent = "Experiment";
  evidence3Value.textContent = "simulated decode delay";
}

function runContextCollapseExperiment() {
  resetRuntimeVisuals();
  seedExperimentMessages({ outside: true });
  scenarioLabel.textContent = "Context collapse";
  setPhase("recovery", "COLLAPSE", "Engine recovering", "EARLIEST MEMORY OUTSIDE ACTIVE WINDOW");
  setHazardState("hazard-context-full", true);
  app.classList.add("scenario-collapse");
  showRuntimeEvent("CONTEXT COLLAPSE", true);
  contextOrgan.classList.add("context-pressure", "context-forgetting");
  engineOrgan.classList.add("engine-recovery");
  memoryState.textContent = "Memory rebalanced";
  contextState.textContent = "Earlier memory forgotten";
  tokenState.textContent = "Retrying recent context";
  healthState.textContent = "Engine recovering";
  setMetrics({ contextTokens: 1180, contextSize: 2048, speed: 29.2, firstToken: 1742, promptRate: 34.8 });
  evidence3Label.textContent = "Dropped turns";
  evidence3Value.textContent = "2";
}

function runExperiment() {
  switch (experiment.value) {
    case "long-context":
      runLongContextExperiment();
      break;
    case "memory-pressure":
      runMemoryPressureExperiment();
      break;
    case "slow-decode":
      runSlowDecodeExperiment();
      break;
    case "context-collapse":
      runContextCollapseExperiment();
      break;
    default:
      setRuntimeIdle();
      prompt.focus();
  }
}

async function streamChat() {
  const value = prompt.value.trim();
  if (!value) return;

  prompt.value = "";
  send.disabled = true;
  const ejectedBeforeSend = applyDeferredContextEjection();
  chatHistory.push({ role: "user", content: value });
  activeAssistant = { role: "assistant", content: "", isStreaming: true };
  chatHistory.push(activeAssistant);
  if (ejectedBeforeSend) {
    activeAssistant.content = "Earlier turns fell outside the active context.\n\n";
  }
  renderMessages();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: chatHistory
          .slice(0, -1)
          .filter((message) => !message.outsideContext)
          .map(({ role, content }) => ({ role, content }))
      })
    });

    if (!response.ok || !response.body) {
      finishActiveAssistant();
      activeAssistant.content += `\n[OpenCortex backend error: ${response.status}]`;
      renderMessages();
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value: chunk, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(chunk, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.kind === "error") {
          finishActiveAssistant();
          const isContextOverflow = event.code === "context_overflow";
          showRuntimeEvent(isContextOverflow ? "CONTEXT WINDOW EXCEEDED" : "BACKEND ERROR", true);
          if (isContextOverflow) {
            setPhase("decode", "DEGRADED", "Context window exceeded", "REQUEST REJECTED · CONTEXT FULL");
            memoryState.textContent = "Memory overloaded";
            contextState.textContent = "Context limit reached";
            tokenState.textContent = "Stream blocked";
            healthState.textContent = "Increase -c or start a new run";
            activeAssistant.content += `\n[OpenCortex] ${event.message}`;
          } else {
            activeAssistant.content += `\n[OpenCortex backend error: ${event.message}]`;
          }
          renderMessages();
          continue;
        }
        applyRuntimeEvent(event);
        if (event.text_delta) {
          activeAssistant.content += event.text_delta;
          renderMessages();
        }
      }
    }
  } catch (error) {
    showRuntimeEvent("BACKEND ERROR", true);
    finishActiveAssistant();
    activeAssistant.content += `\n[OpenCortex backend error: ${error.message || error}]`;
    renderMessages();
  } finally {
    finishActiveAssistant();
    renderMessages();
    send.disabled = false;
  }
}

send.addEventListener("click", streamChat);
prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    streamChat();
  }
});
runExperimentButton.addEventListener("click", runExperiment);

function collapseConversation() {
  const current = parseFloat(getComputedStyle(app).getPropertyValue("--chat-width"));
  if (current > collapseThreshold) previousChatWidth = current;
  app.classList.remove("collapse-preview");
  app.classList.add("chat-collapsed");
  app.style.setProperty("--chat-width", "0px");
}

function restoreConversation() {
  app.classList.remove("chat-collapsed", "collapse-preview");
  const width = Math.max(340, previousChatWidth || workspace.clientWidth * 0.4);
  app.style.setProperty("--chat-width", width + "px");
}

resizer.addEventListener("pointerdown", (event) => {
  if (event.target === drawerToggle || app.classList.contains("chat-collapsed")) return;
  event.preventDefault();
  resizer.setPointerCapture(event.pointerId);
  resizer.classList.add("dragging");
  document.body.style.cursor = "col-resize";
});

resizer.addEventListener("pointermove", (event) => {
  if (!resizer.hasPointerCapture(event.pointerId)) return;
  const bounds = workspace.getBoundingClientRect();
  const maximum = bounds.width * 0.62;
  pendingChatWidth = Math.max(72, Math.min(maximum, event.clientX - bounds.left));
  app.style.setProperty("--chat-width", pendingChatWidth + "px");
  app.classList.toggle("collapse-preview", pendingChatWidth < collapseThreshold);
});

resizer.addEventListener("pointerup", (event) => {
  if (!resizer.hasPointerCapture(event.pointerId)) return;
  resizer.releasePointerCapture(event.pointerId);
  resizer.classList.remove("dragging");
  document.body.style.cursor = "";
  if (pendingChatWidth < collapseThreshold) collapseConversation();
  else {
    previousChatWidth = pendingChatWidth;
    app.classList.remove("collapse-preview");
  }
});

resizer.addEventListener("dblclick", () => {
  if (app.classList.contains("chat-collapsed")) restoreConversation();
  else collapseConversation();
});
drawerToggle.addEventListener("click", (event) => {
  event.stopPropagation();
  restoreConversation();
});

setRuntimeIdle();
