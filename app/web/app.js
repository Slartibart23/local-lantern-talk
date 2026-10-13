// Local Lantern Talk — browser client.
// Talks to the backend over one WebSocket: JSON events plus binary PCM audio
// (microphone → server, synthesized speech → browser).

const $ = (id) => document.getElementById(id);

const STATE_LABELS = {
  loading: "Loading model",
  listening: "Listening",
  paused: "Listening paused",
  mic_off: "Mic off",
  thinking: "Thinking",
  speaking: "Speaking",
  error: "Error",
};

const app = {
  ws: null,
  replaced: false,
  state: "loading",
  settings: {},
  voices: { builtin: [], custom: [] },
  models: [],
  audioCtx: null,
  micNode: null,
  micStream: null,
  micError: false,      // the browser refused or has no microphone
  started: false,
  pendingAudio: null,   // metadata of the next binary message
  playback: { scheduled: 0, ended: 0, nextTime: 0, turnDone: false },
  currentAi: null,      // { el, bubble, lastSpeech }
  voiceInfo: null,      // reference text etc. of the selected voice
  modelStatus: null,    // GPU layers of the loaded language model
  gpu: null,            // detected GPU, profile and recommended quantization
  download: null,       // current model download
  components: null,
  catalogFilter: "all",
  chosenQuant: {},      // quantization picked per model in the catalog
  characters: [],
  character: null,      // the active character
  memory: null,         // facts and summary of the active character
  editingId: null,      // character open in the editor (null = new)
};

// ---------------------------------------------------------------- WebSocket

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  app.ws = ws;
  ws.onopen = () => {
    if (app.audioCtx) send({ type: "mic_format", sample_rate: app.audioCtx.sampleRate });
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") handleEvent(JSON.parse(ev.data));
    else handleAudio(ev.data);
  };
  ws.onclose = () => {
    if (app.replaced || app.closed) return;
    setState("error", "Connection to the app lost — reconnecting…");
    setTimeout(connect, 1500);
  };
}

function send(obj) {
  if (app.ws && app.ws.readyState === WebSocket.OPEN) app.ws.send(JSON.stringify(obj));
}

function handleEvent(msg) {
  switch (msg.type) {
    case "hello":
      app.settings = msg.settings;
      app.voices = msg.voices;
      app.models = msg.models;
      app.modelStatus = msg.model_status;
      app.gpu = msg.gpu;
      app.download = msg.download;
      renderVoices();
      renderModels();
      applySettings();
      renderComponents(msg.components);
      renderTranscript(msg.transcript);
      renderVoiceInfo(msg.voice_info);
      applyCharacterState(msg);
      setState(msg.state, msg.detail);
      break;
    case "character":
      applyCharacterState(msg);
      renderTranscript(msg.transcript);
      break;
    case "characters":
      applyCharacterState(msg);
      break;
    case "memory":
      app.memory = msg;
      renderMemory();
      break;
    case "state":
      setState(msg.state, msg.detail);
      break;
    case "user_message":
      addUserMessage(msg);
      break;
    case "assistant_start":
      startAssistantMessage(msg.voice);
      resetPlayback();
      break;
    case "segment":
      appendSegment(msg.kind, msg.text);
      break;
    case "audio":
      app.pendingAudio = msg;
      break;
    case "turn_done":
      finishAssistantMessage(msg);
      app.playback.turnDone = true;
      checkPlaybackEnd();
      break;
    case "settings":
      app.settings = msg.settings;
      applySettings();
      break;
    case "components":
      renderComponents(msg.components);
      break;
    case "models":
      app.models = msg.models;
      renderModels();
      break;
    case "model_status":
      app.modelStatus = msg;
      renderCurrentModel();
      break;
    case "download":
      handleDownloadEvent(msg);
      break;
    case "custom_suggestions":
      showSuggestions(msg);
      break;
    case "preview_start":
      resetPlayback();
      break;
    case "preview_done":
      app.playback.turnDone = true;
      checkPlaybackEnd();
      break;
    case "notice":
      toast(msg.message, "info");
      break;
    case "voice_info":
      renderVoiceInfo(msg);
      break;
    case "voices":
      app.voices = msg.voices;
      renderVoices();
      $("voice-select").value = app.settings.voice;
      break;
    case "cleared":
      renderTranscript([]);
      break;
    case "error":
      toast(msg.message);
      break;
    case "bye":
      appClosed();
      break;
    case "replaced":
      app.replaced = true;
      showOverlay("This conversation was opened in another browser tab.", "Use this tab instead", () => location.reload());
      break;
  }
}

// ---------------------------------------------------------------- state / status

function setState(state, detail) {
  app.state = state;
  app.detail = detail || "";
  const btn = $("status-btn");
  btn.dataset.state = state;
  let label = STATE_LABELS[state] || state;
  let hint = detail || "";
  if (app.micError && (state === "listening" || state === "paused")) {
    label = "Mic blocked";
    hint = "Allow the microphone for this page and reload — you can type meanwhile";
  }
  $("status-label").textContent = label;
  if (!hint) {
    if (state === "listening") hint = "Speak now — click to pause";
    else if (state === "paused") hint = "Click to resume listening";
    else if (state === "mic_off") hint = "Click to turn the microphone on, or type below";
  }
  $("status-detail").textContent = hint;
  const paused = state === "paused";
  $("pause-btn").setAttribute("aria-pressed", String(paused));
  $("pause-btn").querySelector("span").textContent = paused ? "Resume listening" : "Pause listening";
  $("pause-btn").querySelector("use").setAttribute("href", paused ? "#i-play" : "#i-pause");
  $("pause-btn").disabled = !app.settings.mic_enabled;
  if (state !== "listening") $("meter-fill").style.width = "0";
}

function togglePause() {
  if (!app.settings.mic_enabled) {
    if (app.state === "mic_off") changeSetting("mic_enabled", true);
    return;
  }
  if (app.state === "listening") send({ type: "pause", paused: true });
  else if (app.state === "paused") send({ type: "pause", paused: false });
}

// ---------------------------------------------------------------- messages

function scrollToEnd(force = false) {
  const box = $("messages");
  const nearEnd = box.scrollHeight - box.scrollTop - box.clientHeight < 120;
  if (force || nearEnd) box.scrollTop = box.scrollHeight;
}

function hideEmptyHint() {
  const hint = $("empty-hint");
  if (hint) hint.hidden = true;
}

function languageName(code) {
  if (!code) return "";
  try { return new Intl.DisplayNames(["en"], { type: "language" }).of(code); } catch { return code; }
}

function voiceName(id) {
  if (!id) return "text only";
  const all = [...app.voices.builtin, ...app.voices.custom];
  return (all.find((v) => v.id === id) || { name: id.split(":").pop() }).name;
}

function addUserMessage(msg) {
  hideEmptyHint();
  const el = document.createElement("div");
  el.className = "msg user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = msg.text;
  const meta = document.createElement("div");
  meta.className = "meta";
  const parts = [msg.source === "voice" ? "spoken" : "typed"];
  if (msg.source === "voice" && msg.language) parts.push(languageName(msg.language));
  meta.textContent = parts.join(" · ");
  if (msg.stt_ms) {
    const t = document.createElement("span");
    t.className = "timing";
    t.textContent = `STT ${msg.stt_ms} ms`;
    meta.append(t);
  }
  el.append(bubble, meta);
  $("messages").append(el);
  scrollToEnd(true);
}

function startAssistantMessage(voice) {
  hideEmptyHint();
  const el = document.createElement("div");
  el.className = "msg ai streaming";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = voice ? `voice: ${voiceName(voice)}` : "text only";
  el.append(bubble, meta);
  $("messages").append(el);
  app.currentAi = { el, bubble, meta, lastSpeech: null };
  scrollToEnd(true);
}

function appendSegment(kind, text) {
  if (!app.currentAi) startAssistantMessage(null);
  renderSegment(app.currentAi, kind, text);
  scrollToEnd();
}

function renderSegment(target, kind, text) {
  if (kind === "thinking") return;  // model reasoning is neither spoken nor shown
  if (kind === "speech") {
    if (!target.lastSpeech) {
      target.lastSpeech = document.createElement("span");
      target.lastSpeech.className = "speech";
      target.bubble.append(target.lastSpeech);
    }
    target.lastSpeech.textContent += text;
    return;
  }
  target.lastSpeech = null;
  const span = document.createElement("span");
  span.className = kind === "code" ? "action code" : "action";
  const label = document.createElement("span");
  label.className = "ns";
  label.textContent = kind === "ooc" ? "OOC · not spoken" : "not spoken";
  span.append(document.createTextNode(text), label);
  if (kind !== "code") {
    target.bubble.append(" ", span, " ");
  } else {
    target.bubble.append(span);
  }
}

function finishAssistantMessage(msg) {
  const ai = app.currentAi;
  if (!ai) return;
  ai.el.classList.remove("streaming");
  if (!ai.bubble.textContent.trim()) ai.bubble.textContent = "…";
  if (msg && msg.first_audio_ms != null) {
    const t = document.createElement("span");
    t.className = "timing";
    t.title = msg.metrics;
    t.textContent = `first audio ${(msg.first_audio_ms / 1000).toFixed(2)} s`;
    ai.meta.append(t);
  }
  app.currentAi = null;
}

function renderTranscript(items) {
  const box = $("messages");
  box.querySelectorAll(".msg").forEach((m) => m.remove());
  $("empty-hint").hidden = items.length > 0;
  app.currentAi = null;
  for (const item of items) {
    if (item.role === "user") addUserMessage(item);
    else {
      startAssistantMessage(item.voice);
      if (item.greeting) app.currentAi.meta.textContent = "greeting";
      for (const seg of item.segments) renderSegment(app.currentAi, seg.kind, seg.text);
      finishAssistantMessage(null);
    }
  }
  scrollToEnd(true);
}

// ---------------------------------------------------------------- audio playback

function resetPlayback() {
  app.playback = { scheduled: 0, ended: 0, nextTime: 0, turnDone: false };
}

function handleAudio(buffer) {
  const meta = app.pendingAudio;
  app.pendingAudio = null;
  if (!meta) return;
  const ctx = app.audioCtx;
  if (!ctx) { send({ type: "playback_ended" }); return; }
  const pcm = new Int16Array(buffer);
  const samples = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) samples[i] = pcm[i] / 32768;
  const audioBuffer = ctx.createBuffer(1, samples.length, meta.sample_rate);
  audioBuffer.copyToChannel(samples, 0);
  const src = ctx.createBufferSource();
  src.buffer = audioBuffer;
  src.connect(ctx.destination);
  const pb = app.playback;
  const startAt = Math.max(ctx.currentTime + 0.03, pb.nextTime);
  src.start(startAt);
  pb.nextTime = startAt + audioBuffer.duration;
  pb.scheduled++;
  src.onended = () => { pb.ended++; checkPlaybackEnd(); };
}

function checkPlaybackEnd() {
  const pb = app.playback;
  if (pb.turnDone && pb.scheduled > 0 && pb.ended >= pb.scheduled) {
    pb.scheduled = 0;
    send({ type: "playback_ended" });
  }
}

// ---------------------------------------------------------------- microphone

async function startAudio() {
  app.audioCtx = new AudioContext({ latencyHint: "interactive" });
  await app.audioCtx.resume();
  send({ type: "mic_format", sample_rate: app.audioCtx.sampleRate });
  if (app.settings.mic_enabled !== false) await startMic();
}

async function startMic() {
  if (app.micNode || !app.audioCtx) return;
  try {
    app.micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (err) {
    const firstTime = !app.micError;  // settings updates retry; tell the user only once
    app.micError = true;
    setState(app.state, app.detail);
    if (firstTime) toast("Microphone access was not allowed. You can still type, or allow the microphone in the browser and reload.");
    return;
  }
  app.micError = false;
  const ctx = app.audioCtx;
  await ctx.audioWorklet.addModule("/static/mic-worklet.js");
  const source = ctx.createMediaStreamSource(app.micStream);
  const node = new AudioWorkletNode(ctx, "mic-capture");
  node.port.onmessage = (ev) => {
    if (app.state === "listening") {
      if (app.ws && app.ws.readyState === WebSocket.OPEN) app.ws.send(ev.data.pcm);
      $("meter-fill").style.width = `${Math.min(100, Math.sqrt(ev.data.rms) * 260)}%`;
    }
  };
  source.connect(node);
  const sink = ctx.createGain();
  sink.gain.value = 0;
  node.connect(sink).connect(ctx.destination);
  app.micNode = node;
}

function stopMic() {
  if (app.micStream) app.micStream.getTracks().forEach((t) => t.stop());
  if (app.micNode) app.micNode.disconnect();
  app.micStream = null;
  app.micNode = null;
  $("meter-fill").style.width = "0";
}

// ---------------------------------------------------------------- right panel

function renderVoices() {
  const sel = $("voice-select");
  sel.innerHTML = "";
  const groups = [["Built-in voices", app.voices.builtin], ["Your voices (user_data\\voices)", app.voices.custom]];
  for (const [label, list] of groups) {
    if (!list.length) continue;
    const og = document.createElement("optgroup");
    og.label = label;
    for (const v of list) og.append(new Option(v.name, v.id));
    sel.append(og);
  }
}

// ---------------------------------------------------------------- language models (spec §13 / §14.2)

const VARIANT_LABEL = { censored: "censored", uncensored: "uncensored", unknown: "unlabeled" };
const FIT_LABEL = { fits: "fits on the GPU", partial: "partly on the CPU (slower)", too_big: "too big for this GPU" };

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node[k] = v;
    else if (v !== undefined && v !== null) node.setAttribute(k, v);
  }
  for (const c of children) if (c !== null && c !== undefined) node.append(c);
  return node;
}

function renderModels() {
  renderCurrentModel();
  renderCatalog();
}

function renderCurrentModel() {
  const file = app.settings.model;
  const entry = app.models.find((m) => m.downloaded_files.includes(file));
  const quant = entry && entry.quants.find((q) => q.file === file);
  $("model-name").textContent = entry ? entry.name : (file || "No model yet");
  const variant = $("model-variant");
  variant.className = "badge " + (entry ? entry.variant : "");
  variant.textContent = entry ? VARIANT_LABEL[entry.variant] : "—";
  $("model-quant").textContent = quant ? quant.quant : "?";
  $("model-size").textContent = quant ? `${quant.size_gb} GB` : "—";
  const st = app.modelStatus || {};
  const notice = $("model-notice");
  notice.hidden = !(st.partial && st.file === file);
  notice.textContent = st.notice || "";
}

function filteredModels() {
  const f = app.catalogFilter;
  return app.models.filter((m) => f === "all" || (f === "downloaded" ? m.downloaded_files.length > 0 : m.variant === f));
}

function renderCatalog() {
  const g = app.gpu || {};
  $("catalog-gpu").textContent = g.name
    ? `${g.name} · ${(g.vram_mib / 1024).toFixed(1)} GB VRAM · profile ${g.profile} · recommended quantization ${g.recommended_quant}` +
      ` · about ${g.available_gb} GB free for a language model`
    : "GPU not detected";
  const list = $("model-list");
  list.innerHTML = "";
  const models = filteredModels();
  if (!models.length) list.append(el("p", { class: "muted" }, "No models in this view."));
  for (const m of models) list.append(modelRow(m));
  renderDownload();
}

function modelRow(m) {
  const chosen = app.chosenQuant[m.id] || m.recommended.quant;
  const q = m.quants.find((x) => x.quant === chosen) || m.quants[0];
  const loaded = m.state === "loaded" && q.file === app.settings.model;
  const row = el("div", { class: `model-row ${q.fit}${loaded ? " loaded" : ""}` });

  const meta = [m.family, m.languages_note, m.license_note].filter(Boolean).join(" · ");
  row.append(el("div", {},
    el("div", { class: "model-title" }, m.name,
      el("span", { class: `badge ${m.variant}` }, VARIANT_LABEL[m.variant])),
    el("div", { class: "model-purpose" }, m.purpose + (m.note ? ` — ${m.note}` : "")),
    meta ? el("div", { class: "model-meta" }, meta) : null));

  const select = el("select", { "aria-label": `Quantization for ${m.name}` });
  for (const x of m.quants) {
    const opt = new Option(`${x.quant} · ${x.size_gb} GB${x.downloaded ? " · downloaded" : ""}`, x.quant);
    select.append(opt);
  }
  select.value = q.quant;
  select.onchange = () => { app.chosenQuant[m.id] = select.value; renderCatalog(); };
  row.append(el("div", { class: "model-quant" }, select,
    el("div", { class: `fit badge ${q.fit}` }, FIT_LABEL[q.fit])));

  const actions = el("div", { class: "model-actions" });
  const downloading = app.download && app.download.key === m.id && ["queued", "running"].includes(app.download.state);
  if (loaded) {
    actions.append(el("span", { class: "loaded-label" }, "Loaded"));
  } else if (q.downloaded) {
    actions.append(el("button", { type: "button", class: "primary", onclick: () => loadModel(q.file) }, "Load"));
  } else if (downloading) {
    actions.append(el("span", { class: "muted small" }, "Downloading…"));
  } else {
    const btn = el("button", {
      type: "button", class: "primary",
      onclick: () => send({ type: "download_model", id: m.id, quant: q.quant }),
    }, "Download & load");
    if (q.fit === "too_big") {
      btn.disabled = true;
      btn.title = "Needs more GPU memory than is available. Choose a smaller quantization.";
    } else if (app.download && ["queued", "running"].includes(app.download.state)) {
      btn.disabled = true;
      btn.title = "Another download is running.";
    }
    actions.append(btn);
  }
  if (q.downloaded && !loaded) {
    actions.append(el("button", {
      type: "button", class: "secondary", "aria-label": `Delete ${q.file}`,
      onclick: () => { if (confirm(`Delete ${q.file} (${q.size_gb} GB) from models\\llm?`)) send({ type: "delete_model", file: q.file }); },
    }, "Delete"));
  }
  if (m.source !== "catalog") {
    const label = el("select", { "aria-label": `Label for ${m.name}` });
    for (const v of ["unknown", "censored", "uncensored"]) label.append(new Option(VARIANT_LABEL[v], v));
    label.value = m.variant;
    label.onchange = () => send({ type: "set_model_label", file: q.file, variant: label.value });
    actions.append(label);
  }
  if (m.source === "custom" && !q.downloaded) {
    actions.append(el("button", { type: "button", class: "secondary", onclick: () => send({ type: "remove_custom_model", id: m.id }) }, "Remove"));
  }
  row.append(actions);
  return row;
}

function loadModel(file) {
  changeSetting("model", file);
  closeCatalog();
}

function renderDownload() {
  const d = app.download;
  const bar = $("download-bar");
  const active = d && ["queued", "running"].includes(d.state);
  bar.hidden = !active;
  if (!active) return;
  const pct = d.total ? (100 * d.done / d.total) : 0;
  $("download-name").textContent = d.file_name;
  const eta = d.speed > 0 && d.total ? Math.round((d.total - d.done) / d.speed) : null;
  $("download-detail").textContent =
    `${(d.done / 1e9).toFixed(2)} / ${(d.total / 1e9).toFixed(2)} GB · ${(d.speed / 1e6).toFixed(0)} MB/s` +
    (eta !== null ? ` · ${eta >= 60 ? Math.round(eta / 60) + " min" : eta + " s"} left` : "");
  $("download-fill").style.width = `${pct.toFixed(1)}%`;
}

function handleDownloadEvent(d) {
  const wasActive = app.download && ["queued", "running"].includes(app.download.state);
  app.download = d;
  renderDownload();
  if (wasActive && d.state === "done") toast(`${d.file_name} downloaded; loading it now.`, "info");
  if (d.state === "cancelled") toast("Download cancelled. It continues where it stopped if you start it again.", "info");
  if (!["queued", "running"].includes(d.state)) renderCatalog();
}

function showSuggestions(msg) {
  $("toast").hidden = true;  // replaces the "Looking up the model" note
  $("custom-suggestions-text").textContent = msg.message;
  const list = $("custom-suggestions-list");
  list.innerHTML = "";
  for (const repo of msg.suggestions) {
    list.append(el("button", {
      type: "button", class: "secondary",
      onclick: () => {
        $("custom-suggestions").hidden = true;
        send({ type: "add_custom_model", source: repo, quant: msg.quant, variant: msg.variant });
        toast(`Adding ${repo}…`, "info");
      },
    }, `Use ${repo}`));
  }
  $("custom-suggestions").hidden = false;
}

function openCatalog() {
  $("catalog").hidden = false;
  send({ type: "refresh" });
  $("close-catalog").focus();
}

function closeCatalog() {
  $("catalog").hidden = true;
  $("open-catalog").focus();
}

function applySettings() {
  const s = app.settings;
  $("voice-output").checked = !!s.voice_output;
  $("match-language").checked = !!s.match_language;
  $("mic-enabled").checked = !!s.mic_enabled;
  $("show-actions").checked = !!s.show_actions;
  $("voice-select").value = s.voice;
  $("voice-select").disabled = false;
  $("end-silence").value = s.end_silence_ms;
  setSelect($("tts-steps"), String(s.num_step), `${s.num_step} steps`);
  $("tts-precision").value = s.tts_precision;
  $("end-silence-out").textContent = `${s.end_silence_ms} ms`;
  $("messages").classList.toggle("hide-actions", !s.show_actions);
  renderHeader();
  renderCurrentModel();
  if (app.started) {
    if (s.mic_enabled) startMic(); else stopMic();
  }
  setState(app.state, app.detail);
}

function setSelect(sel, value, label) {
  if (![...sel.options].some((o) => o.value === value)) sel.append(new Option(label, value));
  sel.value = value;
}

function renderVoiceInfo(info) {
  app.voiceInfo = info;
  const custom = !!(info && info.custom);
  $("rename-voice").disabled = !custom;
  $("delete-voice").disabled = !custom;
  const warnings = (custom && info.report && info.report.warnings) || [];
  const list = $("voice-warnings");
  list.innerHTML = "";
  for (const w of warnings) list.append(el("li", {}, w));
  list.hidden = warnings.length === 0;
  const box = $("reftext-box");
  box.hidden = !(info && info.custom);
  if (box.hidden) return;
  const area = $("reftext");
  if (document.activeElement !== area) area.value = info.reference_text || "";
  const origin = $("reftext-origin");
  origin.className = "badge" + (info.origin === "user" ? " user" : "");
  origin.textContent = info.origin === "user" ? "your text"
    : info.origin === "whisper" ? "from Whisper — please check" : "not transcribed yet";
  $("reftext-file").textContent = info.text_file || "";
  $("reftext-whisper").disabled = info.origin !== "user";
}

function saveReferenceText(text) {
  if (!app.voiceInfo) return;
  send({ type: "reference_text", voice: app.voiceInfo.voice, text });
  $("reftext").blur();
}

function changeSetting(key, value) {
  send({ type: "settings", values: { [key]: value } });
}

function renderComponents(c) {
  if (!c) return;
  app.components = c;
  const list = $("components");
  list.innerHTML = "";
  const rows = [
    ["Language model", c.llm ? "loaded" : "not loaded", !!c.llm],
    ["Whisper (speech recognition)", c.whisper ? "loaded" : "unloaded", c.whisper],
    ["Silero VAD", c.vad ? "loaded" : "unloaded", c.vad],
    ["OmniVoice (speech synthesis)", c.omnivoice ? "loaded" : "unloaded", c.omnivoice],
  ];
  for (const [name, text, on] of rows) {
    const li = document.createElement("li");
    const a = document.createElement("span");
    a.textContent = name;
    const b = document.createElement("span");
    b.className = on ? "on" : "off";
    b.textContent = text;
    li.append(a, b);
    list.append(li);
  }
}

async function refreshGpu() {
  try {
    const r = await fetch("/api/gpu");
    const g = await r.json();
    if (!g.total_mb) return;
    $("gpu-name").textContent = g.name;
    // Per-component VRAM (spec §5): measured when each component was loaded.
    const v = (app.components && app.components.vram_mb) || {};
    const parts = [
      ["llm", "Language model", v.llm || 0],
      ["whisper", "Whisper + VAD", v.whisper || 0],
      ["omnivoice", "OmniVoice", v.omnivoice || 0],
    ];
    const known = parts.reduce((sum, p) => sum + p[2], 0);
    parts.push(["other", "Other programs / Windows", Math.max(0, g.used_mb - known)]);
    const bar = $("vram-bar");
    const legend = $("vram-legend");
    bar.innerHTML = "";
    legend.innerHTML = "";
    for (const [key, label, mb] of parts) {
      if (mb <= 0) continue;
      bar.append(el("span", { class: `seg-${key}`, style: `width:${(100 * mb / g.total_mb).toFixed(1)}%` }));
      legend.append(el("li", {}, el("i", { class: `seg-${key}` }), `${label}: ${(mb / 1024).toFixed(1)} GB`));
    }
    $("vram-text").textContent = `VRAM ${(g.used_mb / 1024).toFixed(1)} / ${(g.total_mb / 1024).toFixed(1)} GB in use`;
  } catch { /* the app may be restarting */ }
}

async function uploadVoice(input) {
  const file = input.files[0];
  input.value = "";
  if (!file) return;
  const btn = $("add-voice");
  btn.disabled = true;
  btn.textContent = "Uploading…";
  try {
    const r = await fetch(`/api/voices?filename=${encodeURIComponent(file.name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
    const result = await r.json();
    if (!r.ok) { toast(result.error || "Upload failed."); return; }
    if (result.warnings.length) toast(result.warnings.join(" "), "info");
    // Select it right away; the backend transcribes the clip and prepares the voice.
    $("voice-select").value = result.id;
    $("voice-select").disabled = true;
    changeSetting("voice", result.id);
  } catch {
    toast("Upload failed — is the app still running?");
  } finally {
    btn.disabled = false;
    btn.textContent = "Add voice…";
  }
}

// ---------------------------------------------------------------- characters (spec §10)

const KIND_LABEL = { assistant: "Assistant", roleplay: "Roleplay" };

function renderCharacters() {
  const list = $("char-list");
  list.innerHTML = "";
  const activeId = app.character && app.character.id;
  for (const c of app.characters) {
    const active = c.id === activeId;
    const btn = el("button", { type: "button", class: "char" + (active ? " active" : ""),
                               title: c.persona || c.name, onclick: () => selectCharacter(c.id) },
      el("span", { class: "avatar", "aria-hidden": "true" }, c.avatar_initials),
      el("span", { class: "char-text" },
        el("span", { class: "char-name" }, c.name),
        el("span", { class: "char-role" }, KIND_LABEL[c.kind] || c.kind)));
    if (active) btn.setAttribute("aria-current", "true");
    const edit = el("button", { type: "button", class: "char-edit", "aria-label": `Edit ${c.name}`,
                                title: `Edit ${c.name}`, onclick: () => openEditor(c.id) });
    edit.innerHTML = '<svg aria-hidden="true"><use href="#i-edit"/></svg>';
    list.append(el("li", {}, btn, edit));
  }
  renderHeader();
}

function renderHeader() {
  const c = app.character;
  $("header-avatar").textContent = c ? c.avatar_initials : "LA";
  $("header-name").textContent = c ? c.name : "Lantern";
  $("header-role").textContent = c ? (KIND_LABEL[c.kind] || c.kind) : "Assistant";
  $("edit-active-char").hidden = !c;
  const s = app.settings;
  const speaking = s.voice_output && (!c || c.voice_output);
  $("header-voice").textContent = speaking ? `Voice: ${voiceName(s.voice)}` : "Text replies";
  document.title = c ? `${c.name} · Local Lantern Talk` : "Local Lantern Talk";
}

function selectCharacter(id) {
  if (app.character && app.character.id === id) return;
  send({ type: "select_character", id });
  $("app").classList.remove("show-left");
}

function fillVoiceOptions(sel, value) {
  sel.innerHTML = "";
  const groups = [["Built-in voices", app.voices.builtin], ["Your voices", app.voices.custom]];
  for (const [label, list] of groups) {
    if (!list.length) continue;
    const og = document.createElement("optgroup");
    og.label = label;
    for (const v of list) og.append(new Option(v.name, v.id));
    sel.append(og);
  }
  if (value && ![...sel.options].some((o) => o.value === value)) sel.append(new Option(`${value} (missing)`, value));
  sel.value = value;
}

function fillModelOptions(sel, value) {
  sel.innerHTML = "";
  sel.append(new Option("Use the selected model", ""));
  for (const m of app.models) {
    for (const file of m.downloaded_files) {
      const q = m.quants.find((x) => x.file === file);
      sel.append(new Option(q ? `${m.name} · ${q.quant}` : `${m.name} · ${file}`, file));
    }
  }
  if (value && ![...sel.options].some((o) => o.value === value)) sel.append(new Option(`${value} (not downloaded)`, value));
  sel.value = value || "";
}

function openEditor(id) {
  const c = id ? app.characters.find((x) => x.id === id) : null;
  app.editingId = c ? c.id : null;
  const d = c || { name: "", avatar_initials: "", kind: "roleplay", persona: "", system_prompt: "", greeting: "",
                   voice: app.settings.voice, model: "", voice_output: true, show_actions_as_text: true,
                   memory_enabled: true, recording_enabled: true };
  $("char-editor-title").textContent = c ? `Edit ${c.name}` : "New character";
  $("ce-name").value = d.name;
  $("ce-initials").value = c ? d.avatar_initials : "";
  $("ce-kind").value = d.kind;
  $("ce-persona").value = d.persona;
  $("ce-prompt").value = d.system_prompt;
  $("ce-greeting").value = d.greeting;
  fillVoiceOptions($("ce-voice"), d.voice);
  fillModelOptions($("ce-model"), d.model);
  $("ce-voice-output").checked = d.voice_output;
  $("ce-actions").checked = d.show_actions_as_text;
  $("ce-memory").checked = d.memory_enabled;
  $("ce-recording").checked = d.recording_enabled;
  $("ce-delete").hidden = !c;
  $("ce-duplicate").hidden = !c;
  $("app").classList.remove("show-left");
  $("char-editor").hidden = false;
  $("ce-name").focus();
}

function closeEditor() {
  $("char-editor").hidden = true;
}

function saveEditor(ev) {
  ev.preventDefault();
  const name = $("ce-name").value.trim();
  if (!name) { $("ce-name").focus(); return; }
  const character = {
    name,
    avatar_initials: $("ce-initials").value.trim().toUpperCase(),
    kind: $("ce-kind").value,
    persona: $("ce-persona").value.trim(),
    system_prompt: $("ce-prompt").value.trim(),
    greeting: $("ce-greeting").value.trim(),
    voice: $("ce-voice").value,
    model: $("ce-model").value,
    voice_output: $("ce-voice-output").checked,
    show_actions_as_text: $("ce-actions").checked,
    memory_enabled: $("ce-memory").checked,
    recording_enabled: $("ce-recording").checked,
  };
  send({ type: "save_character", id: app.editingId, character });
  closeEditor();
}

function duplicateEdited() {
  if (!app.editingId) return;
  send({ type: "duplicate_character", id: app.editingId });
  closeEditor();
}

function deleteEdited() {
  const c = app.characters.find((x) => x.id === app.editingId);
  if (!c) return;
  if (app.characters.length <= 1) { toast("The last character cannot be deleted."); return; }
  if (confirm(`Delete ${c.name}? The character file and everything ${c.name} remembers are removed.`)) {
    send({ type: "delete_character", id: c.id });
    closeEditor();
  }
}

// ---------------------------------------------------------------- memory (spec §11)

function renderMemory() {
  const m = app.memory || { enabled: false, facts: [], summary: "" };
  $("memory-off").hidden = m.enabled || !m.character;
  $("memory-on").hidden = !m.enabled;
  const list = $("fact-list");
  list.innerHTML = "";
  if (!m.facts.length) list.append(el("li", { class: "empty-facts" }, "No facts yet."));
  for (const f of m.facts) {
    list.append(el("li", { title: f.source === "user" ? "Added by you" : "Learned from the conversation" },
      el("span", {}, f.text),
      el("button", { type: "button", "aria-label": `Edit fact: ${f.text}`, title: "Edit", onclick: () => editFact(f) }, "✎"),
      el("button", { type: "button", "aria-label": `Delete fact: ${f.text}`, title: "Delete",
                     onclick: () => send({ type: "delete_fact", id: f.id }) }, "✕")));
  }
  $("memory-summary").textContent = m.summary || "Nothing summarized yet. Long conversations and finished ones (New conversation) end up here.";
}

function editFact(f) {
  const text = prompt("Edit this fact:", f.text);
  if (text === null) return;
  if (!text.trim()) send({ type: "delete_fact", id: f.id });
  else if (text.trim() !== f.text) send({ type: "update_fact", id: f.id, text: text.trim() });
}

function applyCharacterState(msg) {
  if (msg.characters) app.characters = msg.characters;
  if ("character" in msg) app.character = msg.character;
  if (msg.memory) app.memory = msg.memory;
  renderCharacters();
  renderMemory();
}

// ---------------------------------------------------------------- misc UI

let toastTimer = null;
function toast(text, kind = "error") {
  const el = $("toast");
  el.textContent = text;
  el.classList.toggle("info", kind === "info");
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 7000);
}

function quitApp() {
  if (!confirm("Quit Local Lantern Talk?\n\nThe language model and the voices are unloaded and the GPU memory is freed. " +
               "Your conversations are saved. Start the app again with start.bat.")) return;
  app.closed = true;
  send({ type: "quit" });
  setTimeout(appClosed, 1500);  // in case the goodbye message gets lost
}

function appClosed() {
  if (app.closedShown) return;
  app.closed = true;
  app.closedShown = true;
  stopMic();
  resetPlayback();
  showOverlay("Local Lantern Talk has been closed and the GPU memory is free again. Your conversations are saved. " +
              "You can close this tab now; start the app again with start.bat.", "", null);
  document.querySelector("#start-overlay .overlay-card").classList.add("closed");
}

function showOverlay(text, button, onClick) {
  $("start-text").textContent = text;
  const btn = $("start-btn");
  btn.textContent = button;
  btn.onclick = onClick;
  $("start-overlay").hidden = false;
  btn.focus();
}

function autoGrow(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
  textarea.style.overflowY = textarea.scrollHeight > 160 ? "auto" : "hidden";
}

function togglePanel(side) {
  const appEl = $("app");
  const cls = side === "left" ? "show-left" : "show-right";
  const open = !appEl.classList.contains(cls);
  appEl.classList.remove("show-left", "show-right");
  appEl.classList.toggle(cls, open);
  $(side === "left" ? "toggle-left" : "toggle-right").setAttribute("aria-expanded", String(open));
}

function init() {
  $("start-btn").onclick = async () => {
    $("start-overlay").hidden = true;
    app.started = true;
    await startAudio();
    $("text-input").focus();
  };
  $("start-btn").focus();

  $("status-btn").onclick = togglePause;
  $("pause-btn").onclick = togglePause;

  const input = $("text-input");
  $("text-form").onsubmit = (ev) => {
    ev.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    send({ type: "text", text });
    input.value = "";
    autoGrow(input);
  };
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) {
      ev.preventDefault();
      $("text-form").requestSubmit();
    }
  });
  input.addEventListener("input", () => autoGrow(input));

  $("voice-output").onchange = (e) => changeSetting("voice_output", e.target.checked);
  $("match-language").onchange = (e) => changeSetting("match_language", e.target.checked);
  $("mic-enabled").onchange = (e) => changeSetting("mic_enabled", e.target.checked);
  $("show-actions").onchange = (e) => changeSetting("show_actions", e.target.checked);
  $("voice-select").onchange = (e) => { e.target.disabled = true; changeSetting("voice", e.target.value); };
  $("open-catalog").onclick = openCatalog;
  $("close-catalog").onclick = closeCatalog;
  $("catalog").onclick = (e) => { if (e.target === $("catalog")) closeCatalog(); };
  $("cancel-download").onclick = () => send({ type: "cancel_download" });
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.onclick = () => {
      app.catalogFilter = chip.dataset.filter;
      document.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", String(c === chip)));
      renderCatalog();
    };
  });
  $("custom-form").onsubmit = (ev) => {
    ev.preventDefault();
    send({ type: "add_custom_model", source: $("custom-source").value.trim(),
           quant: $("custom-quant").value, variant: $("custom-variant").value });
    $("custom-suggestions").hidden = true;
    toast("Looking up the model on Hugging Face…", "info");
    $("custom-source").value = "";
  };
  $("end-silence").oninput = (e) => { $("end-silence-out").textContent = `${e.target.value} ms`; };
  $("end-silence").onchange = (e) => changeSetting("end_silence_ms", Number(e.target.value));
  $("new-conv").onclick = () => send({ type: "new_conversation" });
  $("add-voice").onclick = () => $("voice-file").click();
  $("preview-voice").onclick = () => send({ type: "preview_voice", voice: $("voice-select").value });
  $("rename-voice").onclick = () => {
    const current = voiceName($("voice-select").value);
    const name = prompt("New name for this voice:", current);
    if (name && name.trim() && name.trim() !== current) {
      send({ type: "rename_voice", voice: $("voice-select").value, name: name.trim() });
    }
  };
  $("delete-voice").onclick = () => {
    const name = voiceName($("voice-select").value);
    if (confirm(`Delete the voice "${name}"? The audio file in user_data\\voices is removed.`)) {
      send({ type: "delete_voice", voice: $("voice-select").value });
    }
  };
  $("regenerate-builtins").onclick = () => {
    if (confirm("Generate the 8 built-in voices again? Each gets a new, slightly different sound. This takes about a minute.")) {
      send({ type: "regenerate_builtin_voices" });
    }
  };
  $("reftext-save").onclick = () => saveReferenceText($("reftext").value);
  $("reftext-whisper").onclick = () => { $("reftext").value = ""; saveReferenceText(""); };
  $("tts-steps").onchange = (e) => changeSetting("num_step", Number(e.target.value));
  $("tts-precision").onchange = (e) => changeSetting("tts_precision", e.target.value);
  $("voice-file").onchange = (e) => uploadVoice(e.target);
  $("new-char").onclick = () => openEditor(null);
  $("quit-app").onclick = quitApp;
  $("edit-active-char").onclick = () => app.character && openEditor(app.character.id);
  $("char-form").onsubmit = saveEditor;
  $("close-editor").onclick = closeEditor;
  $("ce-cancel").onclick = closeEditor;
  $("ce-duplicate").onclick = duplicateEdited;
  $("ce-delete").onclick = deleteEdited;
  $("char-editor").onclick = (e) => { if (e.target === $("char-editor")) closeEditor(); };
  $("fact-form").onsubmit = (ev) => {
    ev.preventDefault();
    const text = $("fact-input").value.trim();
    if (text) send({ type: "add_fact", text });
    $("fact-input").value = "";
  };
  $("reset-memory").onclick = () => {
    const name = app.character ? app.character.name : "this character";
    if (confirm(`Reset the memory of ${name}? The conversation history, the summary and all facts are deleted.`)) {
      send({ type: "reset_memory" });
    }
  };
  $("toggle-left").onclick = () => togglePanel("left");
  $("toggle-right").onclick = () => togglePanel("right");
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") {
      if (!$("char-editor").hidden) closeEditor();
      else if (!$("catalog").hidden) closeCatalog();
      $("app").classList.remove("show-left", "show-right");
    }
  });

  connect();
  refreshGpu();
  setInterval(refreshGpu, 5000);
}

init();
