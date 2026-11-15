import { COUNTED, Engine, Frame, Mode } from './engine';
import { BONES, EXERCISES, EXERCISE_LABEL, cue } from './skeleton';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const video = $<HTMLVideoElement>('video');
const overlay = $<HTMLCanvasElement>('overlay');
const ctx = overlay.getContext('2d')!;
const statusEl = $('status');
const engine = new Engine(64);

// Probability bars, built once.
const bars = $('bars');
const barEls = EXERCISES.map((ex) => {
  const row = document.createElement('div');
  row.className = 'b';
  row.innerHTML = `<span>${ex === 'idle' ? 'Not exercising' : EXERCISE_LABEL[ex]}</span><i></i>`;
  bars.appendChild(row);
  return row.querySelector('i') as HTMLElement;
});

// Session tally, one row per exercise.
const tallyEls = Object.fromEntries(COUNTED.map((ex) => {
  const dt = document.createElement('dt');
  dt.textContent = EXERCISE_LABEL[ex];
  const dd = document.createElement('dd');
  dd.textContent = '0';
  $('tally').append(dt, dd);
  return [ex, dd];
})) as Record<(typeof COUNTED)[number], HTMLElement>;

// Exercise switcher.
const modeBtns = [...document.querySelectorAll<HTMLButtonElement>('.modes button')];
function setMode(mode: Mode) {
  engine.setMode(mode);
  modeBtns.forEach((b) => b.setAttribute('aria-checked', String(b.dataset.mode === mode)));
  render();
}
modeBtns.forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode as Mode)));

// Pause / resume for breaks.
const pauseBtns = [$<HTMLButtonElement>('pause'), $<HTMLButtonElement>('hud-pause')];
function togglePause() {
  engine.setPaused(!engine.paused);
  render();
}
pauseBtns.forEach((b) => b.addEventListener('click', togglePause));
// In full screen, tapping the exercise name cycles Auto → Squats → … so you never leave the view.
$('hud-ex').addEventListener('click', () => {
  const order: Mode[] = ['auto', ...COUNTED];
  setMode(order[(order.indexOf(engine.mode) + 1) % order.length]);
});
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !(e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement)) {
    e.preventDefault();
    togglePause();
  }
});

function drawSkeleton(frame: Frame, faulty: boolean) {
  const W = (overlay.width = overlay.clientWidth * devicePixelRatio);
  const H = (overlay.height = overlay.clientHeight * devicePixelRatio);
  ctx.clearRect(0, 0, W, H);
  if (!frame.raw) return;
  // Match the video's object-fit: contain (the whole camera frame is always visible).
  const vw = video.videoWidth, vh = video.videoHeight;
  const s = Math.min(W / vw, H / vh);
  const ox = (W - vw * s) / 2, oy = (H - vh * s) / 2;
  const P = (j: number) => [ox + frame.raw![j].x * vw * s, oy + frame.raw![j].y * vh * s];
  ctx.lineCap = 'round';
  ctx.lineWidth = 6 * devicePixelRatio;
  ctx.strokeStyle = faulty ? '#ff5a4e' : '#f5c518';
  for (const [a, b] of BONES) {
    const [x1, y1] = P(a), [x2, y2] = P(b);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }
  ctx.fillStyle = '#eef2ef';
  for (let j = 0; j < 13; j++) {
    const [x, y] = P(j);
    ctx.beginPath(); ctx.arc(x, y, 5 * devicePixelRatio, 0, Math.PI * 2); ctx.fill();
  }
}

let lastCues = '';
function render() {
  const s = engine.state;
  $('exercise').textContent = s.exercise === 'idle' ? 'Start any exercise' : EXERCISE_LABEL[s.exercise];
  const note = s.paused ? 'Paused. Your counts are kept. Tap Resume when you are ready.'
    : !s.present ? 'Nobody in frame. Counting is on hold and your counts are kept.'
    : s.mode === 'auto'
      ? (s.exercise === 'idle' ? 'Auto mode recognises the exercise once you start moving.'
        : `Auto mode, ${Math.round(s.confidence * 100)}% sure`)
      : `Counting ${EXERCISE_LABEL[s.mode].toLowerCase()} only`;
  $('conf').textContent = note;
  $('hud-state').textContent = s.paused ? 'Paused' : !s.present ? 'Step into frame' : '';
  for (const ex of COUNTED) tallyEls[ex].textContent = String(s.counts[ex]);
  pauseBtns.forEach((b) => (b.textContent = s.paused ? 'Resume' : 'Pause'));
  $('reps').textContent = String(s.reps);
  $('hud-reps').textContent = String(s.reps);
  $('hud-ex').textContent = (s.mode === 'auto' ? 'Auto: ' : '')
    + (s.exercise === 'idle' ? 'start moving' : EXERCISE_LABEL[s.exercise]) + ' ⇄';
  const hudCue = $('hud-cue');
  const firstFault = s.activeFaults[0];
  hudCue.textContent = firstFault ? cue(s.exercise, firstFault) : '';
  hudCue.hidden = !firstFault;
  $('ring').setAttribute('stroke-dashoffset', String(169.6 * (1 - (s.active ? s.repProgress : 0))));
  s.exerciseProbs.forEach((p, i) => (barEls[i].style.transform = `scaleX(${p.toFixed(3)})`));
  const cues = s.activeFaults.map((f) => cue(s.exercise, f));
  const key = cues.join('|') + s.exercise;
  if (key !== lastCues) {
    lastCues = key;
    const list = $('cues');
    list.innerHTML = '';
    const items = cues.length ? cues : [s.exercise === 'idle' ? 'Form cues will show up here while you move.' : 'Form looks good. Keep going.'];
    for (const c of items) {
      const li = document.createElement('li');
      li.textContent = c;
      if (!cues.length) li.className = 'ok';
      list.appendChild(li);
    }
  }
  $('meta').textContent = `${s.fps.toFixed(0)} fps camera, ${s.inferMs.toFixed(1)} ms per analysis, all on this device`;
}

function loop() {
  const now = performance.now();
  if (video.readyState >= 2) {
    const frame = engine.step(video, now);
    drawSkeleton(frame, engine.state.activeFaults.length > 0);
    render();
  }
  if ('requestVideoFrameCallback' in video) video.requestVideoFrameCallback(loop);
  else requestAnimationFrame(loop);
}

// ---------- startup: load models immediately, camera on tap ----------
const startBtn = $<HTMLButtonElement>('start-btn');
const viewport = document.querySelector<HTMLElement>('.viewport')!;
const cameraSel = $<HTMLSelectElement>('camera');
const mirrorBox = $<HTMLInputElement>('mirror');
const isPhone = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
let running = false;
let mirrorTouched = false;

function applyMirror() { viewport.classList.toggle('mirror', mirrorBox.checked); }

const setStatus = (m: string) => (statusEl.textContent = m);
const ready = engine.load('./models/kinetiq.onnx', setStatus).then(
  () => { setStatus('Ready. Tap the button and allow camera access.'); return true; },
  (err) => { console.error(err); setStatus(`Kinetiq could not load its models: ${(err as Error).message}. Check your connection and reload.`); return false; },
);

function inAppBrowser(): string | null {
  const ua = navigator.userAgent;
  if (/Instagram/i.test(ua)) return 'Instagram';
  if (/FBAN|FBAV|FB_IAB/i.test(ua)) return 'Facebook';
  if (/WhatsApp/i.test(ua)) return 'WhatsApp';
  if (/Snapchat|Line\/|LinkedInApp|Twitter/i.test(ua)) return 'this app';
  if (/; wv\)/.test(ua)) return 'this app';
  return null;
}

async function openCamera(): Promise<MediaStream> {
  // Phones reject over-specific constraints, so try progressively simpler ones.
  const v = cameraSel.value;
  const pick: MediaTrackConstraints = v === 'user' || v === 'environment'
    ? { facingMode: { ideal: v } } : { deviceId: { exact: v } };
  const attempts: MediaStreamConstraints[] = [
    { video: { ...pick, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
    { video: pick, audio: false },
    { video: true, audio: false },
  ];
  let last: unknown;
  for (const c of attempts) {
    try { return await navigator.mediaDevices.getUserMedia(c); } catch (e) {
      last = e;
      if ((e as DOMException).name === 'NotAllowedError') throw e; // permission denied: don't retry
    }
  }
  throw last;
}

function cameraError(e: DOMException): string {
  switch (e.name) {
    case 'NotAllowedError': return 'Camera access was blocked. Allow the camera for this site in your browser settings, then reload the page.';
    case 'NotFoundError': case 'OverconstrainedError': return 'No usable camera was found on this device.';
    case 'NotReadableError': return 'Another app is using the camera. Close it, then try again.';
    default: return `The camera could not start: ${e.message || e.name}`;
  }
}

async function start() {
  const app = inAppBrowser();
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    setStatus(app
      ? `${app}'s built-in browser can't use the camera. Open this page in Chrome or Safari instead (tap the ⋯ menu, then "Open in browser").`
      : 'This browser does not allow camera access here. Open the page over https in Chrome, Safari or Firefox.');
    return;
  }
  startBtn.disabled = true;
  try {
    setStatus('Asking for camera access');
    const old = video.srcObject as MediaStream | null;
    old?.getTracks().forEach((t) => t.stop());
    const stream = await openCamera();
    video.srcObject = stream;
    await video.play();
    viewport.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    // Mirror the front camera by default (feels natural), never the back camera.
    const facingNow = stream.getVideoTracks()[0]?.getSettings().facingMode;
    if (!mirrorTouched) mirrorBox.checked = facingNow ? facingNow === 'user' : cameraSel.value === 'user';
    applyMirror();
    await listCameras(stream);
    if (!(await ready)) { startBtn.disabled = false; return; }
    $('camera-slot').appendChild(cameraSel); // camera picker lives in Settings once running
    $('start')?.remove();
    if (!running) { running = true; loop(); }
  } catch (err) {
    console.error(err);
    startBtn.disabled = false;
    setStatus(cameraError(err as DOMException));
  }
}

async function listCameras(stream: MediaStream) {
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
  const current = stream.getVideoTracks()[0]?.getSettings().deviceId;
  const keep = cameraSel.value;
  cameraSel.innerHTML = '';
  const add = (value: string, label: string) => cameraSel.add(new Option(label, value));
  add('user', 'Front camera');
  add('environment', 'Back camera');
  cams.forEach((c, i) => c.deviceId && add(c.deviceId, c.label || `Camera ${i + 1}`));
  cameraSel.value = [...cameraSel.options].some((o) => o.value === keep) ? keep : current ?? 'user';
}

startBtn.addEventListener('click', start);
cameraSel.addEventListener('change', async () => {
  if (video.srcObject) await start();
});
mirrorBox.addEventListener('change', () => { mirrorTouched = true; applyMirror(); });

// Full-screen focus mode: big video with reps and cues on top, readable from across the room.
const fsBtn = $<HTMLButtonElement>('fullscreen');
const exitBtn = $<HTMLButtonElement>('exit-focus');
function setFocus(on: boolean) {
  document.body.classList.toggle('focus', on);
  fsBtn.textContent = on ? 'Exit full screen' : 'Full screen';
  if (on) document.documentElement.requestFullscreen?.().catch(() => { /* iOS: CSS-only mode */ });
  else if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
}
fsBtn.addEventListener('click', () => setFocus(!document.body.classList.contains('focus')));
exitBtn.addEventListener('click', () => setFocus(false));
document.addEventListener('fullscreenchange', () => { if (!document.fullscreenElement) setFocus(false); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') setFocus(false); });

if (isPhone) cameraSel.value = 'environment';

$('reset').addEventListener('click', () => { engine.resetCurrent(); render(); });
$('new-session').addEventListener('click', () => { engine.resetAll(); render(); });

// Record a set of raw landmarks for building a real-world fine-tuning set (see ml/README).
$('record').addEventListener('click', () => {
  const btn = $<HTMLButtonElement>('record');
  if (!engine.recording) {
    engine.recording = [];
    btn.textContent = 'Stop and download';
    return;
  }
  const blob = new Blob([JSON.stringify({
    version: 1,
    aspect: video.videoWidth / video.videoHeight,
    predicted_exercise: engine.state.exercise,
    reps_counted: engine.state.reps,
    frames: engine.recording,
  })], { type: 'application/json' });
  engine.recording = null;
  btn.textContent = 'Record a set';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `kinetiq-set-${new Date().toISOString().slice(0, 19).replace(/:/g, '')}.json`;
  a.click();
});
