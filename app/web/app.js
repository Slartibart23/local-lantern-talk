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
    if (app.replaced) return;
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
      renderVoices();
      renderModels();
      applySettings();
      renderComponents(msg.components);
      renderTranscript(msg.transcript);
      setState(msg.state, msg.detail);
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
    case "cleared":
      renderTranscript([]);
      break;
    case "error":
      toast(msg.message);
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
    else if (state === "mic_off") hint = "Type a message below";
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
  if (!app.settings.mic_enabled) return;
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
    app.micError = true;
    setState(app.state, app.detail);
    toast("Microphone access was not allowed. You can still type, or allow the microphone in the browser and reload.");
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

function renderModels() {
  const sel = $("model-select");
  sel.innerHTML = "";
  if (!app.models.length) sel.append(new Option("No models in models\\llm", ""));
  for (const m of app.models) sel.append(new Option(`${m.file} (${m.size_gb} GB)`, m.file));
}

function applySettings() {
  const s = app.settings;
  $("voice-output").checked = !!s.voice_output;
  $("match-language").checked = !!s.match_language;
  $("mic-enabled").checked = !!s.mic_enabled;
  $("show-actions").checked = !!s.show_actions;
  $("voice-select").value = s.voice;
  $("voice-select").disabled = false;
  $("model-select").value = s.model;
  $("end-silence").value = s.end_silence_ms;
  $("end-silence-out").textContent = `${s.end_silence_ms} ms`;
  $("messages").classList.toggle("hide-actions", !s.show_actions);
  $("header-voice").textContent = s.voice_output ? `Voice: ${voiceName(s.voice)}` : "Text replies";
  const model = app.models.find((m) => m.file === s.model);
  $("model-name").textContent = s.model || "—";
  $("model-quant").textContent = model ? model.quant : "?";
  $("model-size").textContent = model ? `${model.size_gb} GB` : "—";
  if (app.started) {
    if (s.mic_enabled) startMic(); else stopMic();
  }
  setState(app.state, app.detail);
}

function changeSetting(key, value) {
  send({ type: "settings", values: { [key]: value } });
}

function renderComponents(c) {
  if (!c) return;
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
    $("vram-fill").style.width = `${(100 * g.used_mb / g.total_mb).toFixed(1)}%`;
    $("vram-text").textContent = `VRAM ${(g.used_mb / 1024).toFixed(1)} / ${(g.total_mb / 1024).toFixed(1)} GB in use`;
  } catch { /* the app may be restarting */ }
}

// ---------------------------------------------------------------- misc UI

let toastTimer = null;
function toast(text) {
  const el = $("toast");
  el.textContent = text;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 7000);
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
  $("model-select").onchange = (e) => { if (e.target.value) changeSetting("model", e.target.value); };
  $("end-silence").oninput = (e) => { $("end-silence-out").textContent = `${e.target.value} ms`; };
  $("end-silence").onchange = (e) => changeSetting("end_silence_ms", Number(e.target.value));
  $("new-conv").onclick = () => send({ type: "new_conversation" });
  $("toggle-left").onclick = () => togglePanel("left");
  $("toggle-right").onclick = () => togglePanel("right");
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") $("app").classList.remove("show-left", "show-right");
  });

  connect();
  refreshGpu();
  setInterval(refreshGpu, 5000);
}

init();
