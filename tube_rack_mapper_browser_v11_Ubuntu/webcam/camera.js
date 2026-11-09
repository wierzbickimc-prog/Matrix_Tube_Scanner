"use strict";
(() => {
  const el = id => document.getElementById(id);
  const video = el("video"), photo = el("photo");
  let stream = null, wakeLock = null, timer = null, request = 0, remaining = 0;
  let state = "idle", captureData = null;
  const post = (type, extra = {}) => window.parent.postMessage({isStreamlitMessage: true, type, ...extra}, "*");
  const emit = value => post("streamlit:setComponentValue", {value, dataType: "json"});
  const resize = () => post("streamlit:setFrameHeight", {height: Math.ceil(document.body.scrollHeight) + 2});
  function message(text, error = false) { el("status").textContent = text; el("status").classList.toggle("error", error); resize(); }
  function paint() {
    const live = state === "live" || state === "countdown", still = state === "review" || state === "accepted";
    video.hidden = !live; photo.hidden = !still; el("placeholder").hidden = live || still;
    el("guide").hidden = !live; el("countdown").hidden = state !== "countdown";
    el("enable").hidden = !["idle", "opening"].includes(state); el("enable").disabled = state === "opening";
    el("capture").hidden = !live; el("capture").disabled = state === "countdown";
    el("cancel").hidden = state !== "countdown"; el("use").hidden = state !== "review";
    el("retake").hidden = !still; resize();
  }
  function cancelCountdown() { clearInterval(timer); timer = null; if (state === "countdown") state = "live"; paint(); }
  function stopStream() {
    request++; cancelCountdown();
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null; video.srcObject = null;
    if (wakeLock) { wakeLock.release().catch(() => {}); wakeLock = null; }
  }
  async function enableCamera() {
    if (state === "opening") return;
    stopStream(); state = "opening"; captureData = null; photo.removeAttribute("src");
    emit({event: "clear"}); paint(); message("Waiting for camera permission…");
    const current = request;
    if (!navigator.mediaDevices?.getUserMedia) {
      state = "idle"; paint(); message("Camera requires HTTPS or localhost. Open the secure Built scanner link, or use Upload image.", true); return;
    }
    try {
      const media = await navigator.mediaDevices.getUserMedia({audio: false, video: {facingMode: "user", width: {ideal: 3840}, height: {ideal: 2160}}});
      if (current !== request || document.hidden) { media.getTracks().forEach(track => track.stop()); return; }
      stream = media; video.srcObject = stream;
      await video.play();
      if (current !== request) return;
      state = "live"; paint(); el("stage").focus();
      message(`${video.videoWidth} × ${video.videoHeight} camera · Preview is not mirrored. Space to start the countdown.`);
      stream.getVideoTracks()[0].addEventListener("ended", () => {
        if (state === "live" || state === "countdown") { stopStream(); state = "idle"; paint(); message("Camera disconnected. Reconnect it and choose Enable camera.", true); }
      });
      if (navigator.wakeLock) {
        try {
          const lock = await navigator.wakeLock.request("screen");
          if (current === request) wakeLock = lock; else await lock.release();
        } catch (_) { /* Optional; camera and capture remain usable. */ }
      }
    } catch (error) {
      if (current !== request) return;
      stopStream(); state = "idle"; paint();
      const reasons = {
        NotAllowedError: "Camera access was blocked. Allow camera access for this site in your browser, then try again, or use Upload image.",
        NotFoundError: "No camera was found. Connect a webcam, or use Upload image.",
        NotReadableError: "The camera is unavailable. Close other apps using it and try again.",
        OverconstrainedError: "The camera could not provide a usable image. Try another camera, or upload a photo."
      };
      message(reasons[error.name] || "Could not start the camera. Check browser camera permissions, or use Upload image.", true);
    }
  }
  function takePhoto() {
    if (!stream || !video.videoWidth || video.readyState < 2) { cancelCountdown(); message("Camera is not ready. Wait for a clear preview, then try again.", true); return; }
    try {
      const canvas = document.createElement("canvas"); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0);
      captureData = canvas.toDataURL("image/jpeg", .96); photo.src = captureData;
      stopStream(); state = "review"; paint();
      message(`${canvas.width} × ${canvas.height} photo · Check the barcode squares. Use this photo or retake.`);
    } catch (_) { cancelCountdown(); message("The photo could not be captured. Please try again.", true); }
  }
  function countdown() {
    if (state !== "live") return;
    state = "countdown"; remaining = 3; el("countdown").textContent = remaining; paint(); message("Hold the rack steady…");
    timer = setInterval(() => { remaining--; if (remaining === 0) { clearInterval(timer); timer = null; takePhoto(); } else el("countdown").textContent = remaining; }, 1000);
  }
  async function exitCamera() {
    stopStream(); state = "idle"; captureData = null; paint();
    if (document.fullscreenElement) { try { await document.exitFullscreen(); } catch (_) {} }
    emit({event: "exit"});
  }
  el("enable").onclick = enableCamera; el("retake").onclick = enableCamera;
  el("capture").onclick = countdown; el("cancel").onclick = () => { cancelCountdown(); message("Countdown cancelled. Adjust the rack and try again."); };
  el("exit").onclick = exitCamera;
  el("use").onclick = async () => {
    if (!captureData || state !== "review") return;
    state = "accepted"; paint();
    if (document.fullscreenElement) { try { await document.exitFullscreen(); } catch (_) {} }
    emit({event: "capture", data: captureData});
    message("Photo selected. Choose the rack style below, then Start analysis.");
  };
  el("fullscreen").onclick = async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen();
    } catch (_) { message("Full screen is unavailable in this browser. The white lighting and camera still work."); }
  };
  document.addEventListener("fullscreenchange", () => { el("fullscreen").textContent = document.fullscreenElement ? "Leave full screen" : "Full screen light"; resize(); });
  document.addEventListener("keydown", event => {
    if (event.code === "Space" && ["BODY", "VIDEO", "DIV"].includes(event.target.tagName)) { event.preventDefault(); countdown(); }
    if (event.key === "Escape") exitCamera();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && ["live", "countdown", "opening"].includes(state)) { stopStream(); state = "idle"; paint(); message("Camera paused while this tab was hidden. Choose Enable camera to resume."); }
  });
  window.addEventListener("pagehide", stopStream);
  window.addEventListener("message", event => {
    if (event.source === window.parent && event.data?.type === "streamlit:render") { paint(); }
  });
  new ResizeObserver(resize).observe(document.querySelector("main"));
  post("streamlit:componentReady", {apiVersion: 1}); paint();
})();
