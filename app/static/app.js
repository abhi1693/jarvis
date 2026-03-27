const statusGrid = document.querySelector("#status-grid");
const memoryList = document.querySelector("#memory-list");
const skillList = document.querySelector("#skill-list");
const insightList = document.querySelector("#insight-list");
const chatLog = document.querySelector("#chat-log");
const chatForm = document.querySelector("#chat-form");
const chatInput = document.querySelector("#chat-input");
const runEvolutionButton = document.querySelector("#run-evolution");
const refreshButton = document.querySelector("#refresh-state");
const profileForm = document.querySelector("#profile-form");
const video = document.querySelector("#camera-feed");
const overlayCanvas = document.querySelector("#overlay-canvas");
const canvas = document.querySelector("#snapshot-canvas");
const enrollAdminButton = document.querySelector("#enroll-admin");
const cameraStatus = document.querySelector("#camera-status");
const startVoiceButton = document.querySelector("#start-voice");
const stopVoiceButton = document.querySelector("#stop-voice");
const voiceStatus = document.querySelector("#voice-status");
const voiceTranscript = document.querySelector("#voice-transcript");

let cameraStream = null;
let observationTimer = null;
let voiceStream = null;
let mediaRecorder = null;
let voiceChunks = [];
let speechRecognition = null;
let finalTranscript = "";
let interimTranscript = "";
let lastObservation = null;
let cameraStartPromise = null;

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
  const cards = [
    { label: "LLM", value: state.llm_enabled ? "Connected" : "Fallback only" },
    { label: "Interactions", value: state.memory_counts.interactions },
    { label: "Memories", value: state.memory_counts.memories },
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

const blobToDataUrl = (blob) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });

const waitForVideoFrame = () =>
  new Promise((resolve) => {
    if (video.videoWidth && video.videoHeight && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      resolve();
      return;
    }

    video.addEventListener("loadedmetadata", resolve, { once: true });
  });

const captureObservation = async () => {
  if (!cameraStream) return;
  const imageDataUrl = captureCurrentFrame();
  if (!imageDataUrl) return;

  const observation = await fetchJson("/api/observe", {
    method: "POST",
    body: JSON.stringify({ image_data_url: imageDataUrl }),
  });

  if (observation.detail) {
    cameraStatus.textContent = observation.detail;
    return;
  }

  lastObservation = observation;
  drawFaceOverlay(observation);
  updateCameraStatus(observation);
  await refreshState();
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
    cameraStatus.textContent = `No face detected. Brightness: ${observation.brightness}`;
    return;
  }

  const primaryFace = observation.faces?.[0];
  if (!primaryFace) {
    cameraStatus.textContent = `Face detected. Brightness: ${observation.brightness}`;
    return;
  }

  const label = primaryFace.identity === "admin" ? "Admin" : "Unknown";
  cameraStatus.textContent = `${label} face. Confidence: ${primaryFace.confidence}, faces: ${observation.face_count}, brightness: ${observation.brightness}`;
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
    cameraStatus.textContent = "Camera live. Capturing observations every 6 seconds.";
    video.addEventListener(
      "loadedmetadata",
      () => {
        drawFaceOverlay(lastObservation);
      },
      { once: true }
    );
    clearInterval(observationTimer);
    observationTimer = setInterval(captureObservation, 6000);
    await waitForVideoFrame();
    await captureObservation();
    return cameraStream;
  })();

  try {
    return await cameraStartPromise;
  } finally {
    cameraStartPromise = null;
  }
};

const enrollAdmin = async () => {
  if (!cameraStream) {
    await startCamera();
  }
  await waitForVideoFrame();
  const imageDataUrl = captureCurrentFrame();
  if (!imageDataUrl) {
    cameraStatus.textContent = "Waiting for the camera feed before enrolling the admin face.";
    return;
  }

  const response = await fetchJson("/api/vision/enroll", {
    method: "POST",
    body: JSON.stringify({ image_data_url: imageDataUrl }),
  });

  cameraStatus.textContent = response.message || response.detail || "Admin enrollment failed.";
  if (response.face) {
    const frameWidth = video.videoWidth || 1;
    const frameHeight = video.videoHeight || 1;
    drawFaceOverlay({
      faces: [response.face],
      frame_width: frameWidth,
      frame_height: frameHeight,
    });
  }
};

const updateTranscriptView = () => {
  const text = `${finalTranscript} ${interimTranscript}`.trim();
  voiceTranscript.textContent = text || "No speech captured yet.";
};

const startVoice = async () => {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    voiceStatus.textContent = "Voice capture is not available in this browser.";
    return;
  }
  finalTranscript = "";
  interimTranscript = "";
  updateTranscriptView();

  voiceStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  voiceChunks = [];
  mediaRecorder = new MediaRecorder(voiceStream);
  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) voiceChunks.push(event.data);
  };
  mediaRecorder.start();

  if (SpeechRecognitionClass) {
    speechRecognition = new SpeechRecognitionClass();
    speechRecognition.continuous = true;
    speechRecognition.interimResults = true;
    speechRecognition.lang = "en-US";
    speechRecognition.onresult = (event) => {
      interimTranscript = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const segment = event.results[i][0].transcript.trim();
        if (event.results[i].isFinal) {
          finalTranscript = `${finalTranscript} ${segment}`.trim();
        } else {
          interimTranscript = `${interimTranscript} ${segment}`.trim();
        }
      }
      updateTranscriptView();
    };
    speechRecognition.onerror = () => {
      voiceStatus.textContent = "Speech recognition had an error. Audio is still being recorded.";
    };
    speechRecognition.start();
    voiceStatus.textContent = "Listening. Speak naturally.";
  } else {
    voiceStatus.textContent =
      "Recording audio. Browser speech recognition is unavailable, so only the audio file will be stored.";
  }
};

const stopVoice = async () => {
  if (speechRecognition) {
    speechRecognition.stop();
    speechRecognition = null;
  }

  const recorder = mediaRecorder;
  const stream = voiceStream;
  mediaRecorder = null;
  voiceStream = null;

  if (!recorder) return;

  await new Promise((resolve) => {
    recorder.addEventListener("stop", resolve, { once: true });
    recorder.stop();
  });

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }

  const transcript = `${finalTranscript}`.trim();
  const audioBlob = voiceChunks.length ? new Blob(voiceChunks, { type: recorder.mimeType || "audio/webm" }) : null;
  const audioDataUrl = audioBlob ? await blobToDataUrl(audioBlob) : null;

  const response = await fetchJson("/api/interactions", {
    method: "POST",
    body: JSON.stringify({
      message: transcript,
      note: transcript || "audio note captured without transcript",
      modality: "audio",
      audio_data_url: audioDataUrl,
      metadata: {
        transcript_source: SpeechRecognitionClass ? "browser_speech" : "none",
        transcript_available: Boolean(transcript),
      },
    }),
  });

  voiceStatus.textContent = response.message;
  await refreshState();
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
enrollAdminButton.addEventListener("click", enrollAdmin);
startVoiceButton.addEventListener("click", startVoice);
stopVoiceButton.addEventListener("click", stopVoice);

profileForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await fetchJson("/api/profile", {
    method: "POST",
    body: JSON.stringify({
      name: document.querySelector("#profile-name").value.trim(),
      role: document.querySelector("#profile-role").value.trim(),
      goals: document.querySelector("#profile-goals").value.trim(),
      preferences: document.querySelector("#profile-preferences").value.trim(),
    }),
  });
  profileForm.reset();
  await refreshState();
});

refreshState();
startCamera();
setInterval(refreshState, 12000);
