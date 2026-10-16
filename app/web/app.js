// Local Lantern Talk — browser client.
// Talks to the backend over one WebSocket: JSON events plus binary PCM audio
// (microphone → server, synthesized speech → browser).

import { SCENES } from "./scenes.js";

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
  playback: { scheduled: 0, ended: 0, nextTime: 0, turnDone: false, sources: [], stopped: false },
  currentAi: null,      // { el, bubble, lastSpeech }
  voiceInfo: null,      // reference text etc. of the selected voice
  modelStatus: null,    // GPU layers of the loaded language model
  gpu: null,            // detected GPU, profile and recommended quantization
  download: null,       // current model download
  components: null,
  catalogFilter: "all",
  chosenQuant: {},      // quantization picked per model in the catalog
  characters: [],
  resume: null,         // unfinished chat loaded from memory: { messages, last }
  sessions: [],         // recorded sessions (newest first)
  recording: null,      // files of the session being recorded
  viewing: null,        // a past session opened read-only
  replay: null,         // the message being replayed
  profiles: [],         // hardware profiles for the settings screen
  contextInfo: null,    // memory need per context of the selected model (context slider)
  version: "",
  speakerOut: null,     // Web Audio node that reaches the chosen speaker (Firefox)
  endingChat: false,
  character: null,      // the active character
  memory: null,         // facts and summary of the active character
  editingId: null,      // character open in the editor (null = new)
  deletableFrom: null,  // id of the oldest message that can still be deleted
  editingLast: false,   // the message box holds the last user message for editing
  collecting: false,    // push-to-talk / "Hold on": the app collects what you say until "Send"
  talkPress: null,      // when the Talk button / Space went down (push-to-talk)
  director: { pending: "", pinned: [] },  // director's notes: for the next reply, and pinned ones
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
      app.helloReceived = true;
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
      app.deletableFrom = msg.deletable_from ?? null;
      if (msg.director) app.director = msg.director;
      renderDirector();
      renderTranscript(msg.transcript);
      renderVoiceInfo(msg.voice_info);
      applyCharacterState(msg);
      app.profiles = msg.profiles || [];
      app.contextInfo = msg.context_info || null;
      app.version = msg.version || "";
      app.sessions = msg.sessions || [];
      app.recording = msg.recording || null;
      renderSessions();
      setState(msg.state, msg.detail);
      break;
    case "sessions":
      app.sessions = msg.sessions;
      app.recording = msg.recording;
      renderSessions();
      break;
    case "recorded":
      markRecorded(msg);
      break;
    case "character":
      applyCharacterState(msg);
      if (msg.director) { app.director = msg.director; renderDirector(); }
      app.deletableFrom = msg.deletable_from ?? null;
      renderTranscript(msg.transcript);
      break;
    case "transcript":
      app.deletableFrom = msg.deletable_from ?? null;
      renderTranscript(msg.transcript);
      break;
    case "director":
      app.director = { pending: msg.pending, pinned: msg.pinned };
      renderDirector();
      break;
    case "talk":
      app.collecting = !!msg.collecting;
      setState(app.state, app.detail);
      break;
    case "editable":
      app.deletableFrom = msg.deletable_from ?? null;
      markEditable();
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
      hideResume();
      addUserMessage(msg);
      break;
    case "assistant_start":
      stopReplay();
      startAssistantMessage(msg.voice, msg.id);
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
    case "context_info":
      app.contextInfo = msg;
      renderContext();
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
      if (app.endingChat) {
        app.endingChat = false;
        const name = app.character ? app.character.name : "the assistant";
        toast(`New chat with ${name}. Pick another character on the left to talk to someone else.`, "info");
      }
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
  const mode = app.settings.talk_mode || "auto";
  if (state === "listening" && !app.micError) {
    if (app.collecting) label = "Listening — take your time";
    else if (mode === "ptt") label = "Ready";
  }
  $("status-label").textContent = label;
  if (!hint) {
    if (state === "listening" && app.collecting) hint = "Click Send when you are done (Esc cancels)";
    else if (state === "listening" && mode === "ptt") hint = "Hold the Talk button or Space to speak";
    else if (state === "listening" && mode === "smart") hint = "Speak now — I answer when you sound finished";
    else if (state === "listening") hint = "Speak now — click to pause";
    else if (state === "paused") hint = "Click to resume listening";
    else if (state === "mic_off") hint = "Click to turn the microphone on, or type below";
  }
  renderTalkControls(state);
  $("status-detail").textContent = hint;
  const paused = state === "paused";
  $("pause-btn").setAttribute("aria-pressed", String(paused));
  $("pause-btn").querySelector("span").textContent = paused ? "Resume listening" : "Pause listening";
  $("pause-btn").querySelector("use").setAttribute("href", paused ? "#i-play" : "#i-pause");
  $("pause-btn").disabled = !app.settings.mic_enabled;
  const busy = state === "thinking" || state === "speaking";
  $("stop-btn").hidden = !busy;
  $("talk-btn").hidden = busy && !app.collecting;  // Stop takes its place while the character answers
  document.body.classList.toggle("busy", busy || state === "loading");
  if (state !== "listening") $("meter-fill").style.width = "0";
  document.querySelectorAll("#messages .msg.speaking").forEach((m) => m.classList.remove("speaking"));
  if (state === "speaking" && app.lastAiEl) app.lastAiEl.classList.add("speaking");
  // While models load, settings are locked so a click cannot switch something off by accident.
  const loading = state === "loading";
  for (const node of [$("right-panel"), document.querySelector("#settings .settings-grid")]) {
    node.inert = loading;
    node.classList.toggle("locked", loading);
  }
  $("locked-note").hidden = !loading || !app.started;
  renderStartOverlay();
}

// The start screen doubles as the loading screen: Start is only offered once everything is loaded.
function renderStartOverlay() {
  if (app.started || app.replaced || app.closed) return;
  const loading = !app.helloReceived || app.state === "loading";
  $("start-loading").hidden = !loading;
  $("start-loading-detail").textContent = app.detail || "Starting…";
  const text = $("start-text");
  text.hidden = loading;
  text.textContent = app.state === "error"
    ? `Loading finished with a problem: ${app.detail} You can still start and type.`
    : "Ready! Everything runs on this computer. Click Start to allow the microphone and audio playback.";
  const btn = $("start-btn");
  const wasDisabled = btn.disabled;
  btn.disabled = loading;
  btn.textContent = loading ? "Please wait…" : "Start";
  if (wasDisabled && !loading) btn.focus();
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
  if (msg.id != null) el.dataset.id = msg.id;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = msg.text;
  const meta = document.createElement("div");
  meta.className = "meta";
  if (msg.note) el.append(directorLine(msg.note));
  if (msg.source === "director") {
    el.classList.add("director-only");
    bubble.hidden = true;
  }
  const parts = [msg.source === "voice" ? "spoken" : msg.source === "director" ? "director's note" : "typed"];
  if (msg.source === "voice" && msg.language) parts.push(languageName(msg.language));
  meta.textContent = parts.join(" · ");
  if (msg.stt_ms) {
    const t = document.createElement("span");
    t.className = "timing";
    t.textContent = `STT ${msg.stt_ms} ms`;
    meta.append(t);
  }
  addReplayButton(meta, msg);
  addMessageActions(meta, "user");
  el.append(bubble, meta);
  $("messages").append(el);
  scrollToEnd(true);
}

function startAssistantMessage(voice, id) {
  hideEmptyHint();
  const el = document.createElement("div");
  el.className = "msg ai streaming";
  if (id != null) el.dataset.id = id;
  app.lastAiEl = el;  // glows while it is spoken
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = voice ? `voice: ${voiceName(voice)}` : "text only";
  addMessageActions(meta, "ai");
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

function finishAssistantMessage(msg, stopped = false) {
  const ai = app.currentAi;
  if (!ai) return;
  ai.el.classList.remove("streaming");
  if (!ai.bubble.textContent.trim()) ai.bubble.textContent = "…";
  if (stopped || (msg && msg.stopped)) {
    ai.meta.insertBefore(el("span", { class: "stopped", title: "You stopped this reply" }, "stopped"), ai.meta.firstChild);
  }
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
      startAssistantMessage(item.voice, item.id);
      if (item.greeting) app.currentAi.meta.querySelector(".msg-actions")?.remove();
      const meta = app.currentAi.meta;
      if (item.greeting) {
        meta.textContent = "greeting";
        app.currentAi.el.classList.add("greeting");
      }
      for (const seg of item.segments) renderSegment(app.currentAi, seg.kind, seg.text);
      finishAssistantMessage(null, item.stopped);
      addReplayButton(meta, item);
    }
  }
  markEditable();
  scrollToEnd(true);
}

// ---------------------------------------------------------------- deleting, regenerating, editing, stopping

function addMessageActions(meta, role) {
  const box = el("span", { class: "msg-actions" });
  const button = (cls, icon, title, onclick) => {
    const b = el("button", { type: "button", class: `msg-act ${cls}`, title, "aria-label": title, onclick });
    b.innerHTML = `<svg aria-hidden="true"><use href="#${icon}"/></svg>`;
    return b;
  };
  if (role === "ai") {
    box.append(button("act-regen", "i-redo", "Write this answer again", () => send({ type: "regenerate" })));
  } else {
    box.append(button("act-edit", "i-edit", "Change your last message and send it again", startEditLast));
  }
  box.append(button("act-delete", "i-trash", "Delete this message and everything after it",
                    (ev) => deleteFrom(ev.currentTarget.closest(".msg"))));
  meta.append(box);
}

// Messages from app.deletableFrom on can be deleted; the last reply can be written again and
// the last own message edited.
function markEditable() {
  const nodes = [...document.querySelectorAll("#messages .msg")];
  const from = app.deletableFrom;
  for (const n of nodes) {
    n.classList.toggle("deletable", from != null && Number(n.dataset.id) >= from && !n.classList.contains("greeting"));
    n.classList.remove("last", "last-user");
  }
  const last = nodes[nodes.length - 1];
  if (last) last.classList.add("last");
  const lastUser = nodes.filter((n) => n.classList.contains("user")).pop();
  if (lastUser) lastUser.classList.add("last-user");
}

function deleteFrom(node) {
  if (!node) return;
  const nodes = [...document.querySelectorAll("#messages .msg")];
  const after = nodes.length - 1 - nodes.indexOf(node);
  const name = app.character ? app.character.name : "the assistant";
  const what = after > 0 ? `this message and the ${after} message${after > 1 ? "s" : ""} after it` : "this message";
  if (!confirm(`Delete ${what}?\n\nThey are removed from the chat, from ${name}'s memory and from the session recording.`)) return;
  send({ type: "delete_from", id: Number(node.dataset.id) });
}

function startEditLast() {
  const node = document.querySelector("#messages .msg.user.last-user");
  if (!node) return;
  const input = $("text-input");
  input.value = node.querySelector(".bubble").textContent;
  app.editingLast = true;
  $("edit-bar").hidden = false;
  autoGrow(input);
  input.focus();
}

function cancelEditLast() {
  if (!app.editingLast) return;
  app.editingLast = false;
  $("edit-bar").hidden = true;
  $("text-input").value = "";
  autoGrow($("text-input"));
}

// ---------------------------------------------------------------- talk modes: Auto, Smart, Push to talk, Hold on

function renderTalkControls(state) {
  const mode = app.settings.talk_mode || "auto";
  document.querySelectorAll(".talk-mode").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.mode === mode)));
  const btn = $("talk-btn");
  const label = btn.querySelector("span");
  const micOk = app.settings.mic_enabled && !app.micError;
  btn.disabled = !micOk || !(state === "listening" || app.collecting);
  btn.classList.toggle("collecting", app.collecting);
  if (app.collecting) {
    label.textContent = "Send";
    btn.title = "Send what you said (Esc cancels)";
  } else if (mode === "ptt") {
    label.textContent = "Talk";
    btn.title = "Hold to talk and release to send — or click once to start and again to send. The Space bar works too.";
  } else {
    label.textContent = "Hold on";
    btn.title = "Hold on: I'm still thinking. Everything you say is collected until you click Send (Esc cancels).";
  }
}

function setTalkMode(mode) {
  if (mode === app.settings.talk_mode) return;
  if (app.collecting) send({ type: "talk_end", send: false });
  changeSetting("talk_mode", mode);
}

// Push to talk: hold (longer than HOLD_MS) = talk while held; a short click = start, a second click = send.
const HOLD_MS = 400;

function talkDown() {
  if ($("talk-btn").disabled) return;
  if (app.collecting) {            // second click (or Space) after a short one: send
    send({ type: "talk_end", send: true });
    app.talkPress = null;
    return;
  }
  send({ type: "talk_start" });
  app.talkPress = performance.now();
}

function talkUp() {
  const pressed = app.talkPress;
  app.talkPress = null;
  const mode = app.settings.talk_mode || "auto";
  if (pressed != null && mode === "ptt" && performance.now() - pressed >= HOLD_MS) {
    send({ type: "talk_end", send: true });  // released after holding: send
  }
}

function cancelTalk() {
  if (app.collecting) send({ type: "talk_end", send: false });
}

function typingTarget(node) {
  return node && (node.isContentEditable || ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(node.tagName));
}

// ---------------------------------------------------------------- director's notes

function directorLine(note) {
  return el("div", { class: "director-line", title: "Director's note: never spoken, only steers the reply" },
            el("span", { "aria-hidden": "true" }, "🎬 "), note);
}

function renderDirector() {
  const box = $("director-chips");
  box.innerHTML = "";
  const d = app.director || { pending: "", pinned: [] };
  const name = app.character ? app.character.name : "the character";
  d.pinned.forEach((note, i) => {
    box.append(el("span", { class: "director-chip pinned", title: `Pinned: ${name} follows this in every reply` },
      el("span", { "aria-hidden": "true" }, "📌 "), note,
      el("button", { type: "button", "aria-label": `Remove the pinned note: ${note}`, title: "Remove",
                     onclick: () => send({ type: "unpin_note", index: i }) }, "✕")));
  });
  if (d.pending) {
    box.append(el("span", { class: "director-chip pending", title: "Goes with your next message or Continue" },
      el("strong", {}, "Next reply: "), d.pending,
      el("button", { type: "button", "aria-label": "Remove the note for the next reply", title: "Remove",
                     onclick: () => send({ type: "director_note", text: "" }) }, "✕")));
  }
  box.hidden = !d.pinned.length && !d.pending;
}

function directorText() {
  const input = $("director-input");
  const text = input.value.trim();
  input.value = "";
  return text;
}

function stopReply() {
  if (!["thinking", "speaking"].includes(app.state)) return;
  for (const src of app.playback.sources) {
    try { src.stop(); } catch { /* already ended */ }
  }
  app.playback.stopped = true;
  app.playback.sources = [];
  send({ type: "stop_reply" });
}

// ---------------------------------------------------------------- audio playback

function resetPlayback() {
  app.playback = { scheduled: 0, ended: 0, nextTime: 0, turnDone: false, sources: [], stopped: false };
}

function handleAudio(buffer) {
  const meta = app.pendingAudio;
  app.pendingAudio = null;
  if (!meta || app.playback.stopped) return;
  const ctx = app.audioCtx;
  if (!ctx) { send({ type: "playback_ended" }); return; }
  const pcm = new Int16Array(buffer);
  const samples = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) samples[i] = pcm[i] / 32768;
  const audioBuffer = ctx.createBuffer(1, samples.length, meta.sample_rate);
  audioBuffer.copyToChannel(samples, 0);
  const src = ctx.createBufferSource();
  src.buffer = audioBuffer;
  src.connect(speakerNode());
  const pb = app.playback;
  const startAt = Math.max(ctx.currentTime + 0.03, pb.nextTime);
  src.start(startAt);
  pb.nextTime = startAt + audioBuffer.duration;
  pb.scheduled++;
  pb.sources.push(src);
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
  await applyOutputDevice();
  send({ type: "mic_format", sample_rate: app.audioCtx.sampleRate });
  if (app.settings.mic_enabled !== false) await startMic();
}

async function startMic() {
  if (app.micNode || !app.audioCtx) return;
  try {
    const audio = { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true };
    const device = storageGet("llt.inputDevice");
    try {
      app.micStream = await navigator.mediaDevices.getUserMedia({ audio: device ? { ...audio, deviceId: { exact: device } } : audio });
    } catch (err) {
      if (!device || err.name === "NotAllowedError") throw err;
      storageSet("llt.inputDevice", "");  // the chosen microphone is gone: use the default one
      app.micStream = await navigator.mediaDevices.getUserMedia({ audio });
    }
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

const VARIANT_LABEL = { censored: "censored", uncensored: "uncensored", explicit: "explicit", unknown: "unlabeled" };
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
  const ci = app.contextInfo;
  $("model-context").textContent = ci && ci.loaded
    ? `Context ${fmtTokens(ci.loaded)} tokens${ci.auto ? " (automatic)" : ""} · about ${fmtTokens(wordsFor(ci.loaded))} words of conversation`
    : "";
}

// ---------------------------------------------------------------- context length (Settings → Language model)

function fmtTokens(n) {
  return Number(n).toLocaleString("en-US").replace(/,/g, "\u202f");  // narrow no-break space: never wraps
}

// Half of the context holds the conversation; ~3.5 characters per token and ~7 per word (incl. space).
function wordsFor(context) {
  return Math.round(context * 0.5 * 3.5 / 7 / 100) * 100;
}

// GPU memory the model needs with ``context`` (piecewise linear between the measured points).
function contextNeed(profile, context) {
  const pts = profile.points;
  if (pts.length === 1) return pts[0][1];
  for (let i = 0; i < pts.length - 1; i++) {
    const [c0, m0] = pts[i];
    const [c1, m1] = pts[i + 1];
    if (context <= c1 || i === pts.length - 2) return m0 + (m1 - m0) * (context - c0) / (c1 - c0);
  }
  return pts[pts.length - 1][1];
}

function contextZone(ci, context) {
  if (!ci || !ci.profile) return null;
  const need = contextNeed(ci.profile, context);
  if (need <= ci.available_mib) return "fits";
  if (need <= ci.available_mib * ci.partial_factor) return "partial";
  return "too_big";
}

function renderContext() {
  const ci = app.contextInfo;
  if (!ci) return;
  const range = $("context-range");
  const max = Math.max(ci.min, Math.min(ci.max, 131072));
  range.min = ci.min;
  range.max = max;
  range.step = ci.step;
  if (document.activeElement !== range) range.value = Math.min(ci.auto ? (ci.loaded || ci.value) : ci.value, max);
  range.disabled = ci.auto;
  $("context-auto").checked = ci.auto;
  $("context-min-label").textContent = `${ci.min / 1024}k`;
  $("context-max-label").textContent = `${Math.round(max / 1024)}k`;
  showContextValue(Number(range.value));
  // The bar under the slider: green fits on the GPU, yellow partly on the CPU, red too big.
  const bar = $("context-zones");
  if (!ci.profile) {
    bar.style.background = "var(--line)";
    $("context-zone-text").textContent = "The coloured zones appear once a model is loaded.";
    return;
  }
  const colour = { fits: "var(--green)", partial: "var(--amber)", too_big: "var(--red)" };
  const steps = Math.max(1, Math.round((max - ci.min) / ci.step));
  const stops = [];
  let lastFit = null;
  let lastPartial = null;
  for (let i = 0; i <= steps; i++) {
    const c = ci.min + i * ci.step;
    const z = contextZone(ci, c);
    if (z === "fits") lastFit = c;
    if (z !== "too_big") lastPartial = c;
    stops.push(`${colour[z]} ${(100 * i / (steps + 1)).toFixed(2)}% ${(100 * (i + 1) / (steps + 1)).toFixed(2)}%`);
  }
  bar.style.background = `linear-gradient(90deg, ${stops.join(", ")})`;
  const parts = [lastFit ? `Fits on the GPU up to ${fmtTokens(lastFit)} tokens`
                         : "Even the shortest context does not fit completely on the GPU"];
  if (lastPartial && lastPartial !== lastFit) parts.push(`partly on the CPU (slower) up to ${fmtTokens(lastPartial)}`);
  if (lastPartial !== max) parts.push("above that too big");
  let text = parts.join(", ") + ".";
  if (ci.auto) text += ` Automatic uses up to ${fmtTokens(Math.min(ci.auto_max, max))} tokens with this hardware profile.`;
  $("context-zone-text").textContent = text;
}

function showContextValue(context) {
  const zone = contextZone(app.contextInfo, context);
  const label = { fits: "fits on the GPU", partial: "partly on the CPU, slower", too_big: "too big for the free GPU memory" };
  const out = $("context-out");
  out.textContent = `${fmtTokens(context)} tokens` + (zone ? ` · ${label[zone]}` : "");
  out.className = zone ? `zone-text ${zone}` : "";
  const words = wordsFor(context);
  $("context-meaning").textContent =
    `About ${fmtTokens(words)} words of conversation (roughly ${Math.max(1, Math.round(words / 150))}–` +
    `${Math.max(2, Math.round(words / 100))} exchanges) stay in the character's memory word for word; ` +
    "older parts live on in the summary and the facts.";
}

function renderProfileTable() {
  const table = $("profile-table");
  table.innerHTML = "";
  table.append(el("tr", {}, el("th", {}, "Profile"), el("th", {}, "GPU memory"), el("th", {}, "Quantization"),
                  el("th", {}, "Whisper"), el("th", {}, "Automatic context")));
  for (const p of app.profiles) {
    table.append(el("tr", {}, el("td", {}, p.name), el("td", {}, p.description.split(":")[0]), el("td", {}, p.quant || ""),
                    el("td", {}, p.whisper || ""), el("td", {}, p.auto_context ? `up to ${fmtTokens(p.auto_context)}` : "")));
  }
}

function toggleInfo(btn) {
  const box = $(btn.dataset.info);
  box.hidden = !box.hidden;
  btn.setAttribute("aria-expanded", String(!box.hidden));
}

function familyMode() {
  return !!(app.settings && app.settings.family_mode);
}

// Family mode hides uncensored and explicit models (a filter, not a lock).
function modelAllowed(m) {
  return !familyMode() || !["uncensored", "explicit"].includes(m.variant);
}

function filteredModels() {
  const f = app.catalogFilter;
  return app.models.filter(modelAllowed)
    .filter((m) => f === "all" || (f === "downloaded" ? m.downloaded_files.length > 0 : m.variant === f));
}

function renderCatalog() {
  const g = app.gpu || {};
  $("catalog-gpu").textContent = g.name
    ? `${g.name} · ${(g.vram_mib / 1024).toFixed(1)} GB VRAM · profile ${g.profile} · recommended quantization ${g.recommended_quant}` +
      ` · about ${g.available_gb} GB free for a language model`
    : "GPU not detected";
  document.querySelectorAll(".chip").forEach((c) => {
    c.hidden = familyMode() && ["uncensored", "explicit"].includes(c.dataset.filter);
    c.setAttribute("aria-pressed", String(c.dataset.filter === app.catalogFilter));
  });
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
  const hf = m.hf_repo ? el("a", { class: "hf-link", href: `https://huggingface.co/${m.hf_repo}`, target: "_blank",
                                   rel: "noopener noreferrer",
                                   title: "Open the model page on Hugging Face: description, license, all files" },
                            `huggingface.co/${m.hf_repo} ↗`) : null;
  row.append(el("div", {},
    el("div", { class: "model-title" }, m.name,
      el("span", { class: `badge ${m.variant}` }, VARIANT_LABEL[m.variant])),
    el("div", { class: "model-purpose" }, m.purpose + (m.note ? ` — ${m.note}` : "")),
    meta ? el("div", { class: "model-meta" }, meta) : null,
    hf));

  const select = el("select", { "aria-label": `Quantization for ${m.name}`,
                                title: "How finely the model is stored: higher numbers are better but bigger" });
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
    actions.append(el("button", { type: "button", class: "primary", title: "Load this model now (takes a few seconds)",
                                  onclick: () => loadModel(q.file) }, "Load"));
  } else if (downloading) {
    actions.append(el("span", { class: "muted small" }, "Downloading…"));
  } else {
    const btn = el("button", {
      type: "button", class: "primary", title: `Download ${q.file} (${q.size_gb} GB) and load it`,
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
      type: "button", class: "secondary", "aria-label": `Delete ${q.file}`, title: "Delete the downloaded file",
      onclick: () => { if (confirm(`Delete ${q.file} (${q.size_gb} GB) from models\\llm?`)) send({ type: "delete_model", file: q.file }); },
    }, "Delete"));
  }
  if (m.source !== "catalog") {
    const label = el("select", { "aria-label": `Label for ${m.name}`, title: "How this model is labeled in the catalog" });
    for (const v of ["unknown", "censored", "uncensored", "explicit"]) label.append(new Option(VARIANT_LABEL[v], v));
    label.value = m.variant;
    label.onchange = () => send({ type: "set_model_label", file: q.file, variant: label.value });
    actions.append(label);
  }
  if (m.source === "custom" && !q.downloaded) {
    actions.append(el("button", { type: "button", class: "secondary", title: "Remove this model from the catalog",
                                  onclick: () => send({ type: "remove_custom_model", id: m.id }) }, "Remove"));
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
  $("language-select").value = s.language || "en";
  $("mic-enabled").checked = !!s.mic_enabled;
  $("show-actions").checked = !!s.show_actions;
  $("record-sessions").checked = !!s.record_sessions;
  $("voice-select").value = s.voice;
  $("voice-select").disabled = false;
  $("end-silence").value = s.end_silence_ms;
  setSelect($("tts-steps"), String(s.num_step), `${s.num_step} steps`);
  $("tts-precision").value = s.tts_precision;
  $("end-silence-out").textContent = `${(s.end_silence_ms / 1000).toFixed(1)} s`;
  $("quick-end-silence").value = s.end_silence_ms;
  $("quick-end-silence-out").textContent = `${(s.end_silence_ms / 1000).toFixed(1)} s`;
  if (!$("settings").hidden) renderSettings();
  if (app.characters.length) renderCharacters();
  if (app.familyShown !== familyMode()) {  // family mode changes what the catalog shows
    app.familyShown = familyMode();
    if (familyMode() && ["uncensored", "explicit"].includes(app.catalogFilter)) app.catalogFilter = "all";
    renderCatalog();
  }
  $("messages").classList.toggle("hide-actions", !s.show_actions);
  renderHeader();
  renderCurrentModel();
  applyLook();
  renderProfile();
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
  $("reftext-lang").value = info.language_set ? info.language : "";
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
  const visible = app.characters.filter((c) => !familyMode() || !c.adult);
  if (familyMode() && app.character && app.character.adult && visible.length) selectCharacter(visible[0].id);
  for (const c of visible) {
    const active = c.id === activeId;
    const btn = el("button", { type: "button", class: "char" + (active ? " active" : ""),
                               title: c.persona || c.name, onclick: () => selectCharacter(c.id) },
      setAvatar(el("span", { class: "avatar", "aria-hidden": "true" }), c.avatar, c.avatar_initials),
      el("span", { class: "char-text" },
        el("span", { class: "char-name" }, c.name),
        el("span", { class: "char-role" }, KIND_LABEL[c.kind] || c.kind)));
    if (active) btn.setAttribute("aria-current", "true");
    btn.style.setProperty("--ring", SCENES[sceneOf(c)].ring);
    const edit = el("button", { type: "button", class: "char-edit", "aria-label": `Edit ${c.name}`,
                                title: `Edit ${c.name}`, onclick: () => openEditor(c.id) });
    edit.innerHTML = '<svg aria-hidden="true"><use href="#i-edit"/></svg>';
    list.append(el("li", {}, btn, edit));
  }
  renderHeader();
}

function renderHeader() {
  const c = app.character;
  setAvatar($("header-avatar"), c && c.avatar, c ? c.avatar_initials : "LA");
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
  for (const m of app.models.filter(modelAllowed)) {
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
                   memory_enabled: true, recording_enabled: true, avatar: "", background: "", background_fade: 80,
                   text_color: "", director_notes: [], adult: false };
  app.editLook = { avatar: d.avatar || "", background: d.background || "", text_color: d.text_color || "",
                   director_notes: d.director_notes || [] };
  $("ce-bg-fade").value = d.background_fade ?? 80;
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
  $("ce-adult").checked = !!d.adult;
  $("ce-scene").value = SCENES[d.scene] ? d.scene : "plain";
  $("ce-delete").hidden = !c;
  $("ce-duplicate").hidden = !c;
  $("ce-export").hidden = !c;
  renderEditorLook();
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
    scene: $("ce-scene").value,
    avatar: app.editLook.avatar,
    background: app.editLook.background,
    background_fade: Number($("ce-bg-fade").value),
    text_color: app.editLook.text_color,
    director_notes: app.editLook.director_notes,
    adult: $("ce-adult").checked,
  };
  send({ type: "save_character", id: app.editingId, character });
  closeEditor();
}

async function exportEdited() {
  if (!app.editingId) return;
  try {
    const r = await fetch(`/api/characters/${encodeURIComponent(app.editingId)}/export`);
    if (!r.ok) throw new Error((await r.json()).error);
    const match = /filename\*=UTF-8''([^;]+)/.exec(r.headers.get("content-disposition") || "");
    const a = el("a", { href: URL.createObjectURL(await r.blob()),
                        download: match ? decodeURIComponent(match[1]) : "character.llt-character.json" });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
    toast("Saved to your downloads folder. Memory and cloned voices are never included.", "info");
  } catch (err) {
    toast(`The character could not be exported. ${err.message || ""}`);
  }
}

async function importCharacter() {
  const file = await new Promise((resolve) => {
    const input = el("input", { type: "file", accept: ".json,application/json" });
    input.onchange = () => resolve(input.files[0] || null);
    input.click();
  });
  if (!file) return;
  try {
    const r = await fetch("/api/characters/import", { method: "POST", body: file,
                                                       headers: { "Content-Type": "application/json" } });
    const result = await r.json();
    if (!r.ok) toast(result.error || "The character could not be imported.");
  } catch {
    toast("The character could not be imported — is the app still running?");
  }
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
  $("memory-summary").textContent = m.summary || "Nothing summarized yet. Long chats and ended chats (End chat) end up here.";
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
  if ("resume" in msg) app.resume = msg.resume;
  renderCharacters();
  renderMemory();
  renderResume();
  renderRecording();
  applyDecorations();
  applyLook();
}

// ---------------------------------------------------------------- end chat / continue last chat

function chatHasUserMessages() {
  return !!document.querySelector("#messages .msg.user");
}

function endChat() {
  const name = app.character ? app.character.name : "the assistant";
  if (!chatHasUserMessages()) {
    toast(`This is already a new chat with ${name}. Pick another character on the left to talk to someone else.`, "info");
    return;
  }
  const keeps = app.character && app.character.memory_enabled
    ? `${name} keeps it in memory (summary and facts).`
    : `Memory is off for ${name}, so this chat is forgotten.`;
  if (!confirm(`End this chat with ${name} and start a new one?\n\n${keeps}`)) return;
  app.endingChat = true;
  send({ type: "new_conversation" });
}

function renderResume() {
  const r = app.resume;
  const banner = $("resume-banner");
  banner.hidden = !r;
  if (!r) return;
  const name = app.character ? app.character.name : "the assistant";
  const when = new Date(r.last * 1000);
  const sameDay = when.toDateString() === new Date().toDateString();
  const time = when.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const date = sameDay ? `today at ${time}` : `on ${when.toLocaleDateString()} at ${time}`;
  $("resume-text").textContent = `Your last chat with ${name} (${r.messages} messages, ${date}) is loaded. Continue it or start a new chat?`;
}

function hideResume() {
  app.resume = null;
  renderResume();
}

// ---------------------------------------------------------------- session recording & replay (spec §12)

function fmtOffset(seconds) {
  const s = Math.max(0, seconds);
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

function addReplayButton(meta, item) {
  if (!meta || !item || !item.session || item.offset == null) return;
  meta.querySelectorAll(".replay, .replay-time").forEach((n) => n.remove());
  const label = fmtOffset(item.offset);
  if (item.duration == null) {
    meta.append(el("span", { class: "replay-time", title: "Position in the session recording (no audio)" }, label));
    return;
  }
  const btn = el("button", { type: "button", class: "replay", title: `Play the recording from ${label}`,
                             "aria-label": `Replay from ${label}` }, `▶ ${label}`);
  btn.onclick = () => replay(btn, item.session, item.offset, item.duration, label);
  meta.append(btn);
}

function markRecorded(msg) {
  const node = document.querySelector(`#messages .msg[data-id="${msg.id}"]`);
  if (node) addReplayButton(node.querySelector(".meta"), msg);
  const s = app.sessions.find((x) => x.name === msg.session);
  if (s && msg.duration != null) {  // keep the length shown in the session list current
    s.duration = Math.max(s.duration, msg.offset + msg.duration);
    renderSessions();
  }
}

function replay(btn, session, offset, duration, label) {
  const playingThis = app.replay && app.replay.btn === btn;
  stopReplay();
  if (playingThis) return;
  const audio = $("replay-audio");
  app.replay = { btn, label, end: duration != null ? offset + duration + 0.2 : null };
  btn.classList.add("playing");
  btn.textContent = `■ ${label}`;
  const msg = btn.closest(".msg");
  if (msg) msg.classList.add("speaking");
  send({ type: "replay", playing: true });  // keeps the microphone muted while it plays
  // A fresh URL each time: the current session's file keeps growing.
  audio.addEventListener("loadedmetadata", () => {
    if (app.replay === null || app.replay.btn !== btn) return;
    audio.currentTime = offset;
    audio.play().catch(() => { toast("The recording could not be played."); stopReplay(); });
  }, { once: true });
  audio.src = `/api/sessions/${encodeURIComponent(session)}.mp3?t=${Date.now()}`;
  audio.load();
}

function stopReplay() {
  const r = app.replay;
  if (!r) return;
  app.replay = null;
  const audio = $("replay-audio");
  audio.pause();
  r.btn.classList.remove("playing");
  const msg = r.btn.closest(".msg");
  if (msg) msg.classList.remove("speaking");
  r.btn.textContent = `▶ ${r.label}`;
  send({ type: "replay", playing: false });
}

function onReplayTime() {
  const r = app.replay;
  if (r && r.end != null && $("replay-audio").currentTime >= r.end) stopReplay();
}

function renderSessions() {
  const list = $("session-list");
  list.innerHTML = "";
  const current = app.recording && app.recording.name;
  $("no-sessions").hidden = app.sessions.length > 0;
  for (const s of app.sessions.slice(0, 15)) {
    const live = s.name === current;
    const sub = [fmtOffset(s.duration)];
    if (live) sub.push("● recording");
    const btn = el("button", { type: "button", class: "session-item" + (app.viewing && app.viewing.name === s.name ? " active" : ""),
                               title: `Open ${s.name}` },
      el("span", { class: "session-name" }, s.title),
      el("span", { class: "session-sub" + (live ? " live" : "") }, sub.join(" · ")));
    btn.onclick = () => openSession(s.name);
    list.append(el("li", {}, btn));
  }
  renderRecording();
}

function renderRecording() {
  const r = app.recording;
  const c = app.character;
  const on = !!app.settings.record_sessions;
  $("rec-indicator").hidden = !r;
  if (r) $("rec-indicator").title = `Recording to user_data\\sessions\\${r.mp3}`;
  let text;
  if (r) text = `Recording to ${r.mp3} and ${r.md} in user_data\\sessions.`;
  else if (!on) text = "Recording is off.";
  else if (c && !c.recording_enabled) text = `${c.name} is not recorded (see the character settings).`;
  else text = "Recording starts with your next message.";
  $("recording-files").textContent = text;
}

async function openSession(name) {
  let data;
  try {
    const r = await fetch(`/api/sessions/${encodeURIComponent(name)}`);
    if (!r.ok) throw new Error(r.statusText);
    data = await r.json();
  } catch {
    toast("This session could not be opened.");
    return;
  }
  stopReplay();
  app.viewing = data;
  $("app").classList.remove("show-left");
  $("session-title").textContent = data.title || name;
  const box = $("session-messages");
  box.innerHTML = "";
  for (const e of data.entries) box.append(sessionMessage(e, data.audio ? name : null));
  if (!data.entries.length) box.append(el("div", { class: "empty" }, "This session has no messages."));
  document.body.classList.add("viewing-session");
  $("session-view").hidden = false;
  box.scrollTop = 0;
  renderSessions();
  $("close-session").focus();
}

function sessionMessage(e, session) {
  const node = el("div", { class: `msg ${e.role === "user" ? "user" : "ai"}` });
  const bubble = el("div", { class: "bubble" });
  const meta = el("div", { class: "meta" });
  if (e.role === "user") {
    bubble.textContent = e.text;
    meta.textContent = e.source === "voice" ? "spoken" : "typed";
  } else {
    const target = { bubble, lastSpeech: null };
    for (const seg of e.segments) renderSegment(target, seg.kind, seg.text);
    if (!bubble.textContent.trim()) bubble.textContent = "…";
    const tag = (e.tags || []).find((t) => !/^\d+(\.\d+)? s$/.test(t) && t !== "no audio");
    meta.textContent = e.greeting ? "greeting" : (tag || "");
  }
  node.append(bubble, meta);
  if (session) addReplayButton(meta, { ...e, session });
  return node;
}

function closeSession() {
  stopReplay();
  app.viewing = null;
  document.body.classList.remove("viewing-session");
  $("session-view").hidden = true;
  renderSessions();
  scrollToEnd(true);
  $("text-input").focus();
}

// ---------------------------------------------------------------- settings (spec §14.3)

function storageGet(key) {
  try { return localStorage.getItem(key) || ""; } catch { return ""; }
}

function storageSet(key, value) {
  try { if (value) localStorage.setItem(key, value); else localStorage.removeItem(key); } catch { /* private mode */ }
}

function openSettings() {
  renderSettings();
  refreshDevices();
  $("app").classList.remove("show-left");
  $("settings").hidden = false;
  $("close-settings").focus();
}

function closeSettings() {
  $("settings").hidden = true;
  $("open-settings").focus();
}

function renderSettings() {
  const s = app.settings;
  $("settings-version").textContent = app.version ? `Local Lantern Talk ${app.version} · settings are saved in user_data\\settings.json` : "";
  $("set-voice-output").checked = !!s.voice_output;
  $("vad-threshold").value = s.vad_threshold;
  $("vad-threshold-out").textContent = Number(s.vad_threshold).toFixed(2);
  renderContext();
  renderProfileTable();
  $("set-record-sessions").checked = !!s.record_sessions;
  $("set-family").checked = !!s.family_mode;
  setSelect($("session-bitrate"), String(s.session_bitrate_kbps), `${s.session_bitrate_kbps} kbit/s`);
  $("session-gap").value = s.session_gap_ms;
  $("session-gap-out").textContent = `${(s.session_gap_ms / 1000).toFixed(1)} s`;
  const sel = $("hardware-profile");
  sel.innerHTML = "";
  const detected = s.hardware_profile || "high";
  sel.append(new Option(`Automatic (detected: ${detected})`, ""));
  for (const p of app.profiles) sel.append(new Option(`${p.name} — ${p.description}`, p.name));
  sel.value = s.hardware_profile_override || "";
}

async function refreshDevices() {
  const inSel = $("input-device");
  const outSel = $("output-device");
  let devices = [];
  try { devices = await navigator.mediaDevices.enumerateDevices(); } catch { /* no permission yet */ }
  const fill = (sel, kind, stored, fallback) => {
    sel.innerHTML = "";
    sel.append(new Option("System default", ""));
    devices.filter((d) => d.kind === kind && d.deviceId && d.deviceId !== "default").forEach((d, i) => {
      sel.append(new Option(d.label || `${fallback} ${i + 1}`, d.deviceId));
    });
    sel.value = [...sel.options].some((o) => o.value === stored) ? stored : "";
  };
  fill(inSel, "audioinput", storageGet("llt.inputDevice"), "Microphone");
  fill(outSel, "audiooutput", storageGet("llt.outputDevice"), "Speaker");
  const canPickOutput = typeof AudioContext.prototype.setSinkId === "function" ||
                        typeof HTMLMediaElement.prototype.setSinkId === "function";
  outSel.disabled = !canPickOutput;
  if (!canPickOutput) $("device-hint").textContent = "This browser always plays on the system's default device.";
}

async function chooseInputDevice(id) {
  storageSet("llt.inputDevice", id);
  if (app.micStream) {
    stopMic();
    await startMic();
  }
}

async function chooseOutputDevice(id) {
  storageSet("llt.outputDevice", id);
  await applyOutputDevice();
}

// Speech is played through Web Audio. Chrome can point the AudioContext at a device;
// Firefox cannot, so the speech is routed through an <audio> element that can.
async function applyOutputDevice() {
  const id = storageGet("llt.outputDevice");
  const replayEl = $("replay-audio");
  try { if (replayEl.setSinkId) await replayEl.setSinkId(id); } catch { /* device gone: default */ }
  const ctx = app.audioCtx;
  if (!ctx) return;
  try {
    if (typeof ctx.setSinkId === "function") {
      await ctx.setSinkId(id);
      app.speakerOut = null;
    } else if (id && typeof HTMLMediaElement.prototype.setSinkId === "function") {
      if (!app.speakerEl) {
        app.speakerDest = ctx.createMediaStreamDestination();
        app.speakerEl = new Audio();
        app.speakerEl.srcObject = app.speakerDest.stream;
      }
      await app.speakerEl.setSinkId(id);
      await app.speakerEl.play();
      app.speakerOut = app.speakerDest;
    } else {
      app.speakerOut = null;
    }
  } catch {
    app.speakerOut = null;
    toast("The chosen speaker is not available; using the default device.");
  }
}

function speakerNode() {
  return app.speakerOut || app.audioCtx.destination;
}

// ---------------------------------------------------------------- decorations (scenes, fairy lights, lanterns)

function decorationsOn() {
  return storageGet("llt.decorations") !== "off";
}

function setDecorations(on) {
  storageSet("llt.decorations", on ? "" : "off");
  applyDecorations();
}

function sceneOf(character) {
  const name = character && character.scene;
  return SCENES[name] ? name : "plain";
}

// The active character's scene sets the background, accent colour, light colours and the
// side-panel cliparts (scenes.js).
function applyDecorations() {
  const on = decorationsOn();
  const name = on ? sceneOf(app.character) : "";
  const body = document.body;
  body.classList.toggle("decorated", on);
  for (const key of Object.keys(SCENES)) body.classList.toggle(`scene-${key}`, key === name);
  body.style.setProperty("--ring", on ? SCENES[name].ring : "var(--amber)");
  $("set-decorations").checked = on;
  $("set-avatar-size").value = avatarSize();
  const [left, right] = on ? SCENES[name].art : ["", ""];
  $("side-art-left").innerHTML = left;
  $("side-art-right").innerHTML = right;
  drawGarland();
}

// A string of lights hanging under the chat header, with a red paper lantern at each end.
function drawGarland() {
  const box = $("garland");
  box.innerHTML = "";
  if (!decorationsOn()) return;
  const scene = SCENES[sceneOf(app.character)];
  const colors = scene.bulbs;
  const w = Math.max(200, box.clientWidth);
  const sag = Math.min(16, w / 60);
  const y = (x) => 2 + sag * (1 - ((x - w / 2) / (w / 2)) ** 2);
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("width", w);
  svg.setAttribute("height", 52);
  svg.setAttribute("viewBox", `0 0 ${w} 52`);
  const add = (tag, attrs, parent = svg) => {
    const node = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    parent.append(node);
    return node;
  };
  add("path", { d: `M 0 2 Q ${w / 2} ${2 + sag * 2} ${w} 2`, class: "garland-wire" });
  const count = Math.max(6, Math.floor((w - 80) / 46));
  for (let i = 0; i < count; i++) {
    const x = 40 + ((w - 80) * i) / (count - 1);
    const by = y(x) + 6;
    const g = add("g", { class: "bulb", style: `--delay:${((i * 0.37) % 2.4).toFixed(2)}s` });
    add("line", { x1: x, y1: y(x), x2: x, y2: by - 2, class: "garland-wire" }, g);
    add("circle", { cx: x, cy: by + 1, r: 7, fill: colors[i % 4], class: "bulb-glow" }, g);
    add("ellipse", { cx: x, cy: by + 1, rx: 2.6, ry: 3.4, fill: colors[i % 4], class: "bulb-core" }, g);
  }
  for (const x of scene.lanterns ? [16, w - 16] : []) {
    const lantern = add("g", { class: "lantern", style: `transform-origin:${x}px 2px` });
    add("line", { x1: x, y1: 0, x2: x, y2: 12, class: "garland-wire" }, lantern);
    add("ellipse", { cx: x, cy: 27, rx: 15, ry: 17, class: "lantern-glow" }, lantern);
    add("rect", { x: x - 5, y: 11, width: 10, height: 3, rx: 1, class: "lantern-cap" }, lantern);
    add("ellipse", { cx: x, cy: 25, rx: 10, ry: 11.5, class: "lantern-body" }, lantern);
    for (const dx of [-5, 0, 5]) {
      add("path", { d: `M ${x + dx} 14 Q ${x + dx * 1.6} 25 ${x + dx} 36`, class: "lantern-rib" }, lantern);
    }
    add("rect", { x: x - 5, y: 36, width: 10, height: 3, rx: 1, class: "lantern-cap" }, lantern);
    add("line", { x1: x, y1: 39, x2: x, y2: 46, class: "lantern-tassel" }, lantern);
  }
  box.append(svg);
}

// ---------------------------------------------------------------- pictures, colours and your profile

// 16 clearly different text colours, all readable on the dark chat (same keys as characters.TEXT_COLORS).
const TEXT_COLORS = {
  white: ["White", "#f4f2ee"], silver: ["Silver", "#c5cad3"], yellow: ["Yellow", "#f7e463"], gold: ["Gold", "#f0c24b"],
  orange: ["Orange", "#f7a14a"], coral: ["Coral", "#ff8f7e"], red: ["Red", "#ff6262"], pink: ["Pink", "#ff92c9"],
  magenta: ["Magenta", "#ec7cf6"], lavender: ["Lavender", "#cdbcff"], violet: ["Violet", "#a98cff"],
  sky: ["Sky blue", "#7fd6fb"], blue: ["Blue", "#8fb0ff"], turquoise: ["Turquoise", "#4fe3cf"], mint: ["Mint", "#8ff0b3"],
  lime: ["Lime", "#b8e64c"],
};
const MAX_PICTURE_FILE = 40 * 2 ** 20;
const AVATAR_PX = 768;

function imageUrl(name) {
  return `/api/images/${encodeURIComponent(name)}`;
}

function setAvatar(node, picture, initials) {
  node.textContent = "";
  node.classList.toggle("has-img", !!picture);
  if (picture) node.append(el("img", { src: imageUrl(picture), alt: "" }));
  else node.textContent = initials || "";
  return node;
}

function avatarSize() {
  const size = storageGet("llt.avatarSize");
  return ["small", "medium", "large"].includes(size) ? size : "medium";
}

function userInitials(name) {
  const words = (name || "").trim().split(/\s+/).filter(Boolean);
  if (!words.length) return "";
  return (words.length === 1 ? words[0].slice(0, 2) : words[0][0] + words[words.length - 1][0]).toUpperCase();
}

function renderSwatches(box, selected, onPick) {
  box.innerHTML = "";
  const options = [["", "Default", ""], ...Object.entries(TEXT_COLORS).map(([k, [label, hex]]) => [k, label, hex])];
  for (const [key, label, hex] of options) {
    const btn = el("button", { type: "button", class: "swatch" + (key ? "" : " default"), role: "radio",
                               "aria-checked": String(key === selected), title: label, "aria-label": label });
    if (hex) btn.style.setProperty("--swatch", hex);
    else btn.textContent = "Aa";
    btn.onclick = () => onPick(key);
    box.append(btn);
  }
}

// The active character's picture, background and colours, and your own picture and colour.
function applyLook() {
  const c = app.character;
  const s = app.settings || {};
  const you = $("user-avatar");
  const initials = userInitials(s.user_name);
  you.hidden = !(s.user_avatar || initials);
  setAvatar(you, s.user_avatar, initials);
  if (s.user_name) you.title = `${s.user_name} — change your name, picture and text colour in Settings → Your profile`;
  // Portraits at the top corners of the chat (Settings → Appearance → Avatar size).
  const size = avatarSize();
  const showChar = size !== "small" && !!(c && c.avatar);
  const showYou = size !== "small" && !!s.user_avatar;
  if (showChar) $("portrait-char").src = imageUrl(c.avatar);
  if (showYou) $("portrait-you").src = imageUrl(s.user_avatar);
  $("portrait-char").hidden = !showChar;
  $("portrait-you").hidden = !showYou;
  if (c) $("portrait-char").title = c.name;
  for (const key of ["small", "medium", "large"]) document.body.classList.toggle(`avatar-${key}`, size === key);
  document.body.classList.toggle("portrait-left", showChar);
  document.body.classList.toggle("portrait-right", showYou);
  const bg = $("chat-bg");
  if (c && c.background) {
    bg.style.backgroundImage = `url("${imageUrl(c.background)}")`;
    bg.style.opacity = String(Math.max(0.05, (100 - (c.background_fade ?? 80)) / 100));
    bg.hidden = false;
  } else {
    bg.style.backgroundImage = "";
    bg.hidden = true;
  }
  const body = document.body;
  const ai = c && TEXT_COLORS[c.text_color];
  const user = TEXT_COLORS[s.user_color];
  body.classList.toggle("colored-ai", !!ai);
  body.classList.toggle("colored-user", !!user);
  body.style.setProperty("--ai-text", ai ? ai[1] : "");
  body.style.setProperty("--user-text", user ? user[1] : "");
}

function renderEditorLook() {
  const look = app.editLook;
  const name = $("ce-name").value.trim();
  setAvatar($("ce-avatar-preview"), look.avatar, $("ce-initials").value.trim().toUpperCase() || userInitials(name) || "?");
  $("ce-avatar-remove").disabled = !look.avatar;
  const thumb = $("ce-bg-preview");
  thumb.style.backgroundImage = look.background ? `url("${imageUrl(look.background)}")` : "";
  thumb.classList.toggle("empty", !look.background);
  $("ce-bg-remove").disabled = !look.background;
  $("ce-bg-fade").disabled = !look.background;
  $("ce-bg-fade-out").textContent = `${$("ce-bg-fade").value} %`;
  $("ce-colors-label").textContent = `Text colour of ${name || "the character"}'s messages`;
  renderSwatches($("ce-colors"), look.text_color, (key) => { look.text_color = key; renderEditorLook(); });
}

function renderProfile() {
  const s = app.settings || {};
  if (document.activeElement !== $("you-name")) $("you-name").value = s.user_name || "";
  if (document.activeElement !== $("you-about")) $("you-about").value = s.user_about || "";
  setAvatar($("you-avatar-preview"), s.user_avatar, userInitials(s.user_name) || "?");
  $("you-avatar-remove").disabled = !s.user_avatar;
  renderSwatches($("you-colors"), s.user_color || "", (key) => changeSetting("user_color", key));
}

function pickFile() {
  return new Promise((resolve) => {
    const input = el("input", { type: "file", accept: "image/*" });
    input.onchange = () => resolve(input.files[0] || null);
    input.click();
  });
}

function canvasBlob(canvas) {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      if (blob && blob.type === "image/webp") resolve(blob);
      else canvas.toBlob(resolve, "image/jpeg", 0.88);  // a browser without WebP encoding
    }, "image/webp", 0.86);
  });
}

// Backgrounds are only shrunk (the chat shows them "cover"); avatars are cropped to a square.
async function chooseImage(kind) {
  const file = await pickFile();
  if (!file) return null;
  if (file.size > MAX_PICTURE_FILE) { toast("This picture is very large. Please choose one under 40 MB."); return null; }
  let bitmap;
  try { bitmap = await createImageBitmap(file); } catch {
    toast("This picture could not be read. Please use a JPG, PNG or WebP picture.");
    return null;
  }
  const blob = kind === "background" ? await shrinkImage(bitmap, 1920) : await cropDialog(bitmap);
  bitmap.close();
  if (!blob) return null;
  try {
    const r = await fetch(`/api/images?kind=${encodeURIComponent(kind)}`, {
      method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: blob,
    });
    const result = await r.json();
    if (!r.ok) { toast(result.error || "The picture could not be saved."); return null; }
    return result.name;
  } catch {
    toast("The picture could not be saved — is the app still running?");
    return null;
  }
}

function shrinkImage(bitmap, maxSide) {
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const canvas = el("canvas", { width: Math.round(bitmap.width * scale), height: Math.round(bitmap.height * scale) });
  canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  return canvasBlob(canvas);
}

// Move and zoom the picture inside a circle; the result is a 768 × 768 square (sharp as a big portrait on 4K).
function cropDialog(bitmap) {
  return new Promise((resolve) => {
    const dialog = $("crop-dialog");
    const canvas = $("crop-canvas");
    const zoomInput = $("crop-zoom");
    const ctx = canvas.getContext("2d");
    const size = canvas.width;
    const base = Math.max(size / bitmap.width, size / bitmap.height);
    let zoom = 1;
    let cx = bitmap.width / 2;   // picture point shown in the centre
    let cy = bitmap.height / 2;
    let drag = null;
    const draw = (target = ctx, out = size) => {
      const scale = base * zoom * out / size;
      const half = out / 2 / scale;
      cx = Math.min(Math.max(cx, half), bitmap.width - half);
      cy = Math.min(Math.max(cy, half), bitmap.height - half);
      target.clearRect(0, 0, out, out);
      target.drawImage(bitmap, out / 2 - cx * scale, out / 2 - cy * scale, bitmap.width * scale, bitmap.height * scale);
    };
    const setZoom = (z) => { zoom = Math.min(4, Math.max(1, z)); zoomInput.value = zoom; draw(); };
    canvas.onpointerdown = (e) => { drag = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); };
    canvas.onpointermove = (e) => {
      if (!drag) return;
      const scale = base * zoom * canvas.clientWidth / size;
      cx -= (e.clientX - drag.x) / scale;
      cy -= (e.clientY - drag.y) / scale;
      drag = { x: e.clientX, y: e.clientY };
      draw();
    };
    canvas.onpointerup = () => { drag = null; };
    canvas.onwheel = (e) => { e.preventDefault(); setZoom(zoom * (e.deltaY < 0 ? 1.08 : 1 / 1.08)); };
    zoomInput.oninput = () => setZoom(Number(zoomInput.value));
    const finish = async (ok) => {
      dialog.hidden = true;
      document.removeEventListener("keydown", onKey, true);
      if (!ok) { resolve(null); return; }
      const out = el("canvas", { width: AVATAR_PX, height: AVATAR_PX });
      draw(out.getContext("2d"), AVATAR_PX);
      resolve(await canvasBlob(out));
    };
    const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); finish(false); } };
    document.addEventListener("keydown", onKey, true);
    $("crop-ok").onclick = () => finish(true);
    $("crop-cancel").onclick = () => finish(false);
    $("crop-close").onclick = () => finish(false);
    setZoom(1);
    dialog.hidden = false;
    $("crop-ok").focus();
  });
}

function openProfileSettings() {
  openSettings();
  $("settings-profile").scrollIntoView({ block: "start" });
  $("you-name").focus();
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
  $("start-loading").hidden = true;
  $("start-text").hidden = false;
  $("start-text").textContent = text;
  const btn = $("start-btn");
  btn.disabled = false;
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
    if ($("start-btn").disabled) return;
    $("start-overlay").hidden = true;
    app.started = true;
    await startAudio();
    $("text-input").focus();
  };
  $("start-btn").focus();

  $("status-btn").onclick = togglePause;
  $("pause-btn").onclick = togglePause;
  $("stop-btn").onclick = stopReply;
  $("director-form").onsubmit = (ev) => {
    ev.preventDefault();
    const text = directorText();
    if (text) send({ type: "director_note", text });
  };
  $("director-pin").onclick = () => {
    const text = directorText();
    if (text) send({ type: "pin_note", text });
    else toast("Type the note first, then click Pin.", "info");
  };
  $("director-continue").onclick = () => {
    const text = directorText();
    if (text) send({ type: "director_note", text });
    send({ type: "continue" });
  };
  document.querySelectorAll(".talk-mode").forEach((b) => { b.onclick = () => setTalkMode(b.dataset.mode); });
  const talk = $("talk-btn");
  talk.onpointerdown = (e) => { if (e.button === 0) { talk.setPointerCapture(e.pointerId); talkDown(); } };
  talk.onpointerup = () => talkUp();
  talk.onkeydown = (e) => { if ((e.key === "Enter" || e.key === " ") && !e.repeat) { e.preventDefault(); talkDown(); } };
  talk.onkeyup = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); talkUp(); } };
  // Space bar = Talk button in push-to-talk, as long as you are not typing.
  const dialogOpen = () => !$("char-editor").hidden || !$("settings").hidden || !$("catalog").hidden || !$("crop-dialog").hidden;
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || e.repeat || typingTarget(document.activeElement) || dialogOpen() || !app.started) return;
    if ((app.settings.talk_mode || "auto") !== "ptt" && !app.collecting) return;
    e.preventDefault();
    talkDown();
  });
  document.addEventListener("keyup", (e) => {
    if (e.code !== "Space" || typingTarget(document.activeElement) || dialogOpen()) return;
    if (app.talkPress != null) { e.preventDefault(); talkUp(); }
  });
  $("edit-cancel").onclick = cancelEditLast;

  const input = $("text-input");
  $("text-form").onsubmit = (ev) => {
    ev.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    if (app.editingLast) {
      send({ type: "edit_last", text });
      app.editingLast = false;
      $("edit-bar").hidden = true;
    } else {
      send({ type: "text", text });
    }
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
  $("language-select").onchange = (e) => changeSetting("language", e.target.value);
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
  for (const id of ["end-silence", "quick-end-silence"]) {  // settings dialog and side panel
    $(id).oninput = (e) => { $(`${id}-out`).textContent = `${(e.target.value / 1000).toFixed(1)} s`; };
    $(id).onchange = (e) => changeSetting("end_silence_ms", Number(e.target.value));
  }
  $("end-chat").onclick = endChat;
  $("resume-continue").onclick = () => { hideResume(); $("text-input").focus(); };
  $("resume-new").onclick = () => { hideResume(); app.endingChat = true; send({ type: "new_conversation" }); };
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
  $("reftext-lang").onchange = (e) => {
    if (app.voiceInfo) send({ type: "voice_language", voice: app.voiceInfo.voice, language: e.target.value });
  };
  $("reftext-whisper").onclick = () => { $("reftext").value = ""; saveReferenceText(""); };
  $("tts-steps").onchange = (e) => changeSetting("num_step", Number(e.target.value));
  $("tts-precision").onchange = (e) => changeSetting("tts_precision", e.target.value);
  $("voice-file").onchange = (e) => uploadVoice(e.target);
  $("new-char").onclick = () => openEditor(null);
  $("ce-avatar-pick").onclick = async () => {
    const name = await chooseImage("avatar");
    if (name) { app.editLook.avatar = name; renderEditorLook(); }
  };
  $("ce-avatar-remove").onclick = () => { app.editLook.avatar = ""; renderEditorLook(); };
  $("ce-bg-pick").onclick = async () => {
    const name = await chooseImage("background");
    if (name) { app.editLook.background = name; renderEditorLook(); }
  };
  $("ce-bg-remove").onclick = () => { app.editLook.background = ""; renderEditorLook(); };
  $("ce-bg-fade").oninput = renderEditorLook;
  $("ce-name").addEventListener("input", () => { if (app.editLook) renderEditorLook(); });
  $("ce-initials").addEventListener("input", () => { if (app.editLook) renderEditorLook(); });
  $("user-avatar").onclick = openProfileSettings;
  $("you-avatar-pick").onclick = async () => {
    const name = await chooseImage("user-avatar");
    if (name) changeSetting("user_avatar", name);
  };
  $("you-avatar-remove").onclick = () => changeSetting("user_avatar", "");
  $("you-name").onchange = (e) => changeSetting("user_name", e.target.value.trim());
  $("you-about").onchange = (e) => changeSetting("user_about", e.target.value.trim());
  $("quit-app").onclick = quitApp;
  $("open-settings").onclick = openSettings;
  $("set-decorations").onchange = (e) => setDecorations(e.target.checked);
  $("set-avatar-size").onchange = (e) => { storageSet("llt.avatarSize", e.target.value); applyLook(); };
  window.addEventListener("resize", () => { if (decorationsOn()) drawGarland(); });
  applyDecorations();
  $("close-settings").onclick = closeSettings;
  $("settings").onclick = (e) => { if (e.target === $("settings")) closeSettings(); };
  $("input-device").onchange = (e) => chooseInputDevice(e.target.value);
  $("output-device").onchange = (e) => chooseOutputDevice(e.target.value);
  $("set-voice-output").onchange = (e) => changeSetting("voice_output", e.target.checked);
  $("set-record-sessions").onchange = (e) => changeSetting("record_sessions", e.target.checked);
  $("vad-threshold").oninput = (e) => { $("vad-threshold-out").textContent = Number(e.target.value).toFixed(2); };
  $("vad-threshold").onchange = (e) => changeSetting("vad_threshold", Number(e.target.value));
  $("context-range").oninput = (e) => showContextValue(Number(e.target.value));
  $("context-range").onchange = (e) => changeSetting("context", Number(e.target.value));
  $("context-auto").onchange = (e) => {
    const values = { context_auto: e.target.checked };
    // Switching to a fixed value starts from what Automatic had chosen.
    if (!e.target.checked && app.contextInfo && app.contextInfo.loaded) values.context = app.contextInfo.loaded;
    send({ type: "settings", values });
  };
  document.querySelectorAll(".info-btn").forEach((btn) => { btn.onclick = () => toggleInfo(btn); });
  $("hardware-profile").onchange = (e) => changeSetting("hardware_profile_override", e.target.value);
  $("session-bitrate").onchange = (e) => changeSetting("session_bitrate_kbps", Number(e.target.value));
  $("session-gap").oninput = (e) => { $("session-gap-out").textContent = `${(e.target.value / 1000).toFixed(1)} s`; };
  $("session-gap").onchange = (e) => changeSetting("session_gap_ms", Number(e.target.value));
  $("close-session").onclick = closeSession;
  $("open-sessions").onclick = () => send({ type: "open_sessions_folder" });
  $("record-sessions").onchange = (e) => changeSetting("record_sessions", e.target.checked);
  $("replay-audio").ontimeupdate = onReplayTime;
  $("replay-audio").onended = stopReplay;
  $("edit-active-char").onclick = () => app.character && openEditor(app.character.id);
  $("char-form").onsubmit = saveEditor;
  $("close-editor").onclick = closeEditor;
  $("ce-cancel").onclick = closeEditor;
  $("ce-duplicate").onclick = duplicateEdited;
  $("ce-export").onclick = exportEdited;
  $("import-char").onclick = importCharacter;
  $("set-family").onchange = (e) => changeSetting("family_mode", e.target.checked);
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
      if (["thinking", "speaking"].includes(app.state) && $("char-editor").hidden && $("settings").hidden) stopReply();
      else if (app.collecting) cancelTalk();
      else if (app.editingLast) cancelEditLast();
      else if (!$("char-editor").hidden) closeEditor();
      else if (!$("settings").hidden) closeSettings();
      else if (app.replay) stopReplay();
      else if (app.viewing) closeSession();
      else if (!$("catalog").hidden) closeCatalog();
      $("app").classList.remove("show-left", "show-right");
    }
  });

  connect();
  refreshGpu();
  setInterval(refreshGpu, 5000);
}

init();
