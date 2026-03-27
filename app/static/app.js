const statusGrid = document.querySelector("#status-grid");
const memoryList = document.querySelector("#memory-list");
const skillList = document.querySelector("#skill-list");
const insightList = document.querySelector("#insight-list");
const chatLog = document.querySelector("#chat-log");
const chatForm = document.querySelector("#chat-form");
const chatInput = document.querySelector("#chat-input");
const runEvolutionButton = document.querySelector("#run-evolution");
const refreshButton = document.querySelector("#refresh-state");
const appNameHeading = document.querySelector("#app-name");
const contextForm = document.querySelector("#context-form");
const contextInput = document.querySelector("#context-input");
const contextList = document.querySelector("#context-list");
const video = document.querySelector("#camera-feed");
const overlayCanvas = document.querySelector("#overlay-canvas");
const canvas = document.querySelector("#snapshot-canvas");
const cameraStatus = document.querySelector("#camera-status");
const toggleVoiceButton = document.querySelector("#toggle-voice");
const voiceStatus = document.querySelector("#voice-status");
const voiceTranscript = document.querySelector("#voice-transcript");
const micBadge = document.querySelector("#mic-badge");
const micLevelBar = document.querySelector("#mic-level");

let cameraStream = null;
let observationTimer = null;
let voiceStream = null;
let speechRecognition = null;
let finalTranscript = "";
let interimTranscript = "";
let lastObservation = null;
let cameraStartPromise = null;
let voiceSubmitTimer = null;
let voiceRecognitionActive = false;
let voiceModeEnabled = false;
let voicePauseReason = null;
let voiceRequestInFlight = false;
let micMeterFrame = null;
let audioContext = null;
let analyserNode = null;
let mediaStreamSource = null;
let observationRequestInFlight = false;
let lastPersistedObservationAt = 0;

const OBSERVATION_INTERVAL_MS = 750;
const PERSISTED_OBSERVATION_INTERVAL_MS = 5000;

const SpeechRecognitionClass = window.SpeechRecognition || window.webkitSpeechRecognition;

const fetchJson = async (url, options = {}) => {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  return response.json();
};

const escapeHtml = (value = "") =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

const renderCards = (container, items, renderer) => {
  container.innerHTML = "";
  if (!items.length) {
    container.innerHTML = `<div class="memory-card"><span class="label">Empty</span><p>No entries yet.</p></div>`;
    return;
  }
  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "memory-card";
    card.innerHTML = renderer(item);
    container.append(card);
  });
};

const roleLabel = (role) => {
  if (role === "assistant") return "Agent";
  if (role === "sensor") return "Sensor";
  return "You";
};

const renderInteractionFeed = (interactions) => {
  chatLog.innerHTML = "";
  const filtered = [];
  let sensorCount = 0;
  interactions.forEach((item) => {
    if (item.role === "sensor") {
      sensorCount += 1;
      if (sensorCount > 2) return;
    }
    filtered.push(item);
  });

  if (!filtered.length) {
    chatLog.innerHTML = `<article class="chat-entry assistant"><div class="meta"><strong>Agent</strong><span>idle</span></div><p>No interactions yet.</p></article>`;
    return;
  }

  filtered.forEach((item) => {
    const entry = document.createElement("article");
    entry.className = `chat-entry ${item.role}`;
    entry.innerHTML = `
      <div class="meta">
        <strong>${roleLabel(item.role)}</strong>
        <span>${escapeHtml(item.modality)} · ${escapeHtml(item.intent)}</span>
      </div>
      <p>${escapeHtml(item.content)}</p>
    `;
    chatLog.append(entry);
  });
};

const renderState = (state) => {
  if (state.app_name) {
    appNameHeading.textContent = state.app_name;
    document.title = state.app_name;
  }

  const cards = [
    { label: "LLM", value: state.llm_enabled ? "Connected" : "Fallback only" },
    { label: "Interactions", value: state.memory_counts.interactions },
    { label: "Memories", value: state.memory_counts.memories },
    { label: "Context", value: state.runtime_context?.length || 0 },
    { label: "Skills", value: state.memory_counts.skills },
    { label: "Insights", value: state.memory_counts.insights },
    {
      label: "Presence",
      value: state.last_observation?.admin_present ? "Face detected" : "No face detected",
    },
  ];

  statusGrid.innerHTML = cards
    .map(
      (card) => `
        <div class="status-card">
          <span class="label">${card.label}</span>
          <span class="value">${card.value}</span>
        </div>
      `
    )
    .join("");

  renderCards(
    contextList,
    state.runtime_context || [],
    (item) => `
      <span class="label">${escapeHtml(item.category)}</span>
      <strong>${escapeHtml(item.title)}</strong>
      <p>${escapeHtml(item.content)}</p>
    `
  );

  renderCards(
    memoryList,
    state.recent_memories || [],
    (item) => `
      <span class="label">${escapeHtml(item.category)}</span>
      <strong>${escapeHtml(item.title)}</strong>
      <p>${escapeHtml(item.content)}</p>
    `
  );

  renderCards(
    skillList,
    state.recent_skills || [],
    (item) => `
      <span class="label">${escapeHtml(item.trigger_hint)}</span>
      <strong>${escapeHtml(item.name)}</strong>
      <p>${escapeHtml(item.description)}</p>
    `
  );

  renderCards(
    insightList,
    state.recent_insights || [],
    (item) => `
      <span class="label">${escapeHtml(item.severity)}</span>
      <strong>${escapeHtml(item.title)}</strong>
      <p>${escapeHtml(item.details)}</p>
      <span class="pill">${escapeHtml(item.source)}</span>
    `
  );

  renderInteractionFeed(state.recent_interactions || []);
};

const refreshState = async () => {
  const state = await fetchJson("/api/state");
  renderState(state);
};

const waitForVideoFrame = () =>
  new Promise((resolve) => {
    if (video.videoWidth && video.videoHeight && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      resolve();
      return;
    }

    video.addEventListener("loadedmetadata", resolve, { once: true });
  });

const clearObservationTimer = () => {
  if (!observationTimer) return;
  window.clearTimeout(observationTimer);
  observationTimer = null;
};

const scheduleNextObservation = (delay = OBSERVATION_INTERVAL_MS) => {
  clearObservationTimer();
  if (!cameraStream) return;
  observationTimer = window.setTimeout(runObservationLoop, delay);
};

const captureObservation = async ({ persist = false, refresh = false } = {}) => {
  if (!cameraStream || observationRequestInFlight) return null;
  const imageDataUrl = captureCurrentFrame();
  if (!imageDataUrl) return null;

  observationRequestInFlight = true;
  try {
    const observation = await fetchJson("/api/observe", {
      method: "POST",
      body: JSON.stringify({ image_data_url: imageDataUrl, persist }),
    });

    if (observation.detail) {
      cameraStatus.textContent = observation.detail;
      return observation;
    }

    lastObservation = observation;
    drawFaceOverlay(observation);
    updateCameraStatus(observation);
    if (refresh) {
      await refreshState();
    }
    return observation;
  } catch (_error) {
    cameraStatus.textContent = "Camera live, but face detection sync failed. Retrying...";
    return null;
  } finally {
    observationRequestInFlight = false;
  }
};

const runObservationLoop = async () => {
  if (!cameraStream) return;

  const startedAt = performance.now();
  const shouldPersist =
    !lastPersistedObservationAt ||
    Date.now() - lastPersistedObservationAt >= PERSISTED_OBSERVATION_INTERVAL_MS;
  const observation = await captureObservation({ persist: shouldPersist, refresh: shouldPersist });
  if (shouldPersist && observation && !observation.detail) {
    lastPersistedObservationAt = Date.now();
  }

  const elapsed = performance.now() - startedAt;
  scheduleNextObservation(Math.max(OBSERVATION_INTERVAL_MS - elapsed, 0));
};

const captureCurrentFrame = () => {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) return;

  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, width, height);
  return canvas.toDataURL("image/jpeg", 0.82);
};

const updateCameraStatus = (observation) => {
  if (!observation.face_count) {
    if (observation.admin_learning_state === "uninitialized") {
      cameraStatus.textContent = "No face detected yet. The first persistent face will become the admin baseline.";
      return;
    }
    cameraStatus.textContent = `No face detected. Brightness: ${observation.brightness}`;
    return;
  }

  const primaryFace = observation.faces?.[0];
  if (!primaryFace) {
    cameraStatus.textContent = `Face detected. Brightness: ${observation.brightness}`;
    return;
  }

  const label = primaryFace.identity === "admin" ? "Admin" : "Unknown";
  cameraStatus.textContent =
    `${label} face. Confidence: ${primaryFace.confidence}, faces: ${observation.face_count}, brightness: ${observation.brightness}, admin model: ${observation.admin_learning_state} (${observation.admin_sample_count} samples)`;
};

const drawFaceOverlay = (observation) => {
  const context = overlayCanvas.getContext("2d");
  const width = video.clientWidth;
  const height = video.clientHeight;
  overlayCanvas.width = width;
  overlayCanvas.height = height;
  context.clearRect(0, 0, width, height);

  if (!observation?.faces?.length) return;

  const scaleX = width / observation.frame_width;
  const scaleY = height / observation.frame_height;

  observation.faces.forEach((face) => {
    const x = face.x * scaleX;
    const y = face.y * scaleY;
    const w = face.w * scaleX;
    const h = face.h * scaleY;
    const color = face.color === "green" ? "#4ade80" : face.color === "yellow" ? "#facc15" : "#f87171";
    const label = `${face.identity} ${Math.round(face.confidence * 100)}%`;

    context.save();
    context.font = "600 14px IBM Plex Sans";
    context.strokeStyle = color;
    context.lineWidth = 3;
    context.setLineDash([10, 6]);
    context.strokeRect(x, y, w, h);
    context.setLineDash([]);
    context.fillStyle = "rgba(4, 16, 24, 0.85)";
    const textWidth = context.measureText(label).width;
    context.fillRect(x, Math.max(0, y - 26), textWidth + 16, 24);
    context.fillStyle = color;
    context.fillText(label, x + 8, Math.max(16, y - 9));
    context.restore();
  });
};

const startCamera = async () => {
  if (cameraStream) return cameraStream;
  if (cameraStartPromise) return cameraStartPromise;

  cameraStartPromise = (async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      cameraStatus.textContent = "Camera access is not available in this browser.";
      return null;
    }

    cameraStatus.textContent = "Requesting camera access...";
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    } catch (error) {
      cameraStatus.textContent =
        "Camera access was blocked. Allow permission and reload the page.";
      return null;
    }

    video.srcObject = cameraStream;
    cameraStatus.textContent =
      "Camera live. Analyzing faces in near realtime and persisting periodic samples while the admin model learns.";
    video.addEventListener(
      "loadedmetadata",
      () => {
        drawFaceOverlay(lastObservation);
      },
      { once: true }
    );
    lastPersistedObservationAt = 0;
    clearObservationTimer();
    await waitForVideoFrame();
    await captureObservation({ persist: true, refresh: true });
    scheduleNextObservation();
    return cameraStream;
  })();

  try {
    return await cameraStartPromise;
  } finally {
    cameraStartPromise = null;
  }
};

const setMicLevel = (level = 0.04) => {
  micLevelBar.style.transform = `scaleX(${Math.max(0.04, Math.min(1, level))})`;
};

const updateMicBadge = (tone, label) => {
  micBadge.className = `pill mic-pill ${tone}`;
  micBadge.textContent = label;
};

const syncVoiceUi = () => {
  toggleVoiceButton.textContent = voiceModeEnabled ? "Mute Mic" : "Enable Mic";
  toggleVoiceButton.classList.toggle("live-active", voiceModeEnabled);

  if (!voiceModeEnabled) {
    updateMicBadge("idle", "Mic off");
    return;
  }

  if (voicePauseReason === "processing") {
    updateMicBadge("thinking", "Thinking");
    return;
  }

  if (voicePauseReason === "speaking") {
    updateMicBadge("speaking", "Speaking");
    return;
  }

  updateMicBadge("live", voiceRecognitionActive ? "Listening" : "Mic ready");
};

const updateTranscriptView = (fallbackText) => {
  const text = `${finalTranscript} ${interimTranscript}`.trim();
  if (text) {
    voiceTranscript.textContent = text;
    return;
  }

  if (fallbackText) {
    voiceTranscript.textContent = fallbackText;
    return;
  }

  if (!voiceModeEnabled) {
    voiceTranscript.textContent = "Enable the mic to talk with Jarvis in real time.";
    return;
  }

  if (voicePauseReason === "processing") {
    voiceTranscript.textContent = "Working on your request...";
    return;
  }

  if (voicePauseReason === "speaking") {
    voiceTranscript.textContent = "Speaking the reply...";
    return;
  }

  voiceTranscript.textContent = "Listening for your next instruction...";
};

const stopMicMeter = async () => {
  if (micMeterFrame) {
    cancelAnimationFrame(micMeterFrame);
    micMeterFrame = null;
  }

  if (mediaStreamSource) {
    mediaStreamSource.disconnect();
    mediaStreamSource = null;
  }

  if (analyserNode) {
    analyserNode.disconnect();
    analyserNode = null;
  }

  if (audioContext) {
    await audioContext.close();
    audioContext = null;
  }

  setMicLevel(0.04);
};

const startMicMeter = async (stream) => {
  await stopMicMeter();

  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass || !stream) {
    setMicLevel(0.18);
    return;
  }

  audioContext = new AudioContextClass();
  analyserNode = audioContext.createAnalyser();
  analyserNode.fftSize = 1024;
  mediaStreamSource = audioContext.createMediaStreamSource(stream);
  mediaStreamSource.connect(analyserNode);
  const samples = new Uint8Array(analyserNode.fftSize);

  const tick = () => {
    if (!analyserNode) return;
    analyserNode.getByteTimeDomainData(samples);
    let sum = 0;
    for (const sample of samples) {
      const normalized = (sample - 128) / 128;
      sum += normalized * normalized;
    }
    const rms = Math.sqrt(sum / samples.length);
    setMicLevel(rms * 4.5);
    micMeterFrame = requestAnimationFrame(tick);
  };

  tick();
};

const createSpeechRecognition = () => {
  const recognition = new SpeechRecognitionClass();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = "en-US";

  recognition.onstart = () => {
    voiceRecognitionActive = true;
    syncVoiceUi();
    updateTranscriptView();
  };

  recognition.onresult = (event) => {
    interimTranscript = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const segment = event.results[i][0].transcript.trim();
      if (!segment) continue;

      if (event.results[i].isFinal) {
        finalTranscript = `${finalTranscript} ${segment}`.trim();
      } else {
        interimTranscript = `${interimTranscript} ${segment}`.trim();
      }
    }

    updateTranscriptView();

    if (finalTranscript) {
      clearTimeout(voiceSubmitTimer);
      voiceSubmitTimer = window.setTimeout(flushVoiceTurn, 850);
    }
  };

  recognition.onerror = (event) => {
    if (event.error === "not-allowed" || event.error === "service-not-allowed") {
      voiceStatus.textContent = "Mic permission was blocked. Allow microphone access and try again.";
      stopVoice();
      return;
    }

    if (event.error === "no-speech" || event.error === "aborted") {
      return;
    }

    voiceStatus.textContent = `Speech recognition error: ${event.error}.`;
    updateMicBadge("error", "Mic error");
  };

  recognition.onend = () => {
    voiceRecognitionActive = false;
    syncVoiceUi();
    if (voiceModeEnabled && !voicePauseReason) {
      window.setTimeout(startSpeechRecognition, 120);
    }
  };

  return recognition;
};

const startSpeechRecognition = () => {
  if (!SpeechRecognitionClass || !voiceModeEnabled || voicePauseReason || voiceRecognitionActive) return;
  if (!speechRecognition) {
    speechRecognition = createSpeechRecognition();
  }

  try {
    speechRecognition.start();
  } catch (error) {
    if (!String(error).includes("already started")) {
      voiceStatus.textContent = "Speech recognition could not start.";
    }
  }
};

const stopSpeechRecognition = () => {
  if (speechRecognition && voiceRecognitionActive) {
    speechRecognition.stop();
  }
};

const speakAssistantReply = async (message) => {
  if (!voiceModeEnabled || !("speechSynthesis" in window) || !message) return;

  voicePauseReason = "speaking";
  syncVoiceUi();
  updateTranscriptView();
  stopSpeechRecognition();
  window.speechSynthesis.cancel();

  await new Promise((resolve) => {
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.rate = 1.02;
    utterance.pitch = 1;
    utterance.onend = resolve;
    utterance.onerror = resolve;
    window.speechSynthesis.speak(utterance);
  });
};

async function flushVoiceTurn() {
  clearTimeout(voiceSubmitTimer);
  voiceSubmitTimer = null;

  const transcript = `${finalTranscript}`.trim();
  if (!transcript || voiceRequestInFlight) return;

  finalTranscript = "";
  interimTranscript = "";
  voicePauseReason = "processing";
  voiceRequestInFlight = true;
  syncVoiceUi();
  updateTranscriptView();
  stopSpeechRecognition();

  try {
    const response = await fetchJson("/api/interactions", {
      method: "POST",
      body: JSON.stringify({
        message: transcript,
        note: transcript,
        modality: "audio",
        metadata: {
          transcript_source: "browser_speech",
          transcript_available: true,
          realtime: true,
          auto_submitted: true,
        },
      }),
    });

    voiceStatus.textContent = response.message || response.detail || "Voice turn processed.";
    await refreshState();

    if (response.message) {
      await speakAssistantReply(response.message);
    }
  } finally {
    voiceRequestInFlight = false;
    voicePauseReason = null;
    syncVoiceUi();
    updateTranscriptView();
    if (voiceModeEnabled) {
      startSpeechRecognition();
    }
  }
}

const startVoice = async () => {
  if (voiceModeEnabled) return;

  if (!navigator.mediaDevices?.getUserMedia) {
    voiceStatus.textContent = "Microphone access is not available in this browser.";
    return;
  }

  if (!SpeechRecognitionClass) {
    voiceStatus.textContent =
      "Live voice needs browser speech recognition support. Use a Chromium-based browser.";
    return;
  }

  try {
    voiceStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch (error) {
    voiceStatus.textContent = "Mic permission was blocked. Allow microphone access and try again.";
    return;
  }

  voiceModeEnabled = true;
  voicePauseReason = null;
  finalTranscript = "";
  interimTranscript = "";
  await startMicMeter(voiceStream);
  syncVoiceUi();
  updateTranscriptView();
  voiceStatus.textContent =
    "Live mic enabled. Speak naturally and each utterance will be sent automatically.";
  startSpeechRecognition();
};

const stopVoice = async () => {
  voiceModeEnabled = false;
  voicePauseReason = null;
  clearTimeout(voiceSubmitTimer);
  voiceSubmitTimer = null;
  finalTranscript = "";
  interimTranscript = "";
  stopSpeechRecognition();

  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }

  if (voiceStream) {
    voiceStream.getTracks().forEach((track) => track.stop());
    voiceStream = null;
  }

  await stopMicMeter();
  syncVoiceUi();
  updateTranscriptView();
  voiceStatus.textContent = "Mic disabled.";
};

const toggleVoiceMode = async () => {
  if (voiceModeEnabled) {
    await stopVoice();
    return;
  }

  await startVoice();
};

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;

  chatInput.value = "";
  await fetchJson("/api/interactions", {
    method: "POST",
    body: JSON.stringify({ message, modality: "text" }),
  });
  await refreshState();
});

runEvolutionButton.addEventListener("click", async () => {
  const response = await fetchJson("/api/evolution/scan", { method: "POST" });
  voiceStatus.textContent = response.summary;
  await refreshState();
});

refreshButton.addEventListener("click", refreshState);
toggleVoiceButton.addEventListener("click", toggleVoiceMode);

contextForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = contextInput.value.trim();
  if (!message) return;

  await fetchJson("/api/interactions", {
    method: "POST",
    body: JSON.stringify({
      message,
      modality: "text",
      metadata: {
        source: "context_form",
        interaction_kind: "operating_context",
      },
    }),
  });
  contextForm.reset();
  await refreshState();
});

syncVoiceUi();
updateTranscriptView();
refreshState();
startCamera();
setInterval(refreshState, 12000);
