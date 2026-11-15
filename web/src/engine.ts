// Camera -> MediaPipe PoseLandmarker -> Kinetiq ONNX model, all on-device.
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import * as ort from 'onnxruntime-web/wasm';
import { RepCounter } from './reps';
import { APPLICABLE, EXERCISES, Exercise, FAULTS, Fault, MEDIAPIPE_INDICES, NUM_JOINTS } from './skeleton';

const MP_VERSION = '0.10.35';
const POSE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

/** Exercises that can be counted (everything except idle). */
export const COUNTED = EXERCISES.filter((e) => e !== 'idle') as Exclude<Exercise, 'idle'>[];
export type Counted = (typeof COUNTED)[number];
export type Mode = 'auto' | Counted;

export interface Frame {
  points: Float32Array; // NUM_JOINTS * 3: x (aspect-scaled), y, visibility
  raw: { x: number; y: number }[] | null; // normalised 0..1 for drawing
}

export interface EngineState {
  /** Exercise being counted right now (manual choice, or the auto-detected one). */
  exercise: Exercise;
  /** What the model thinks is happening this instant. */
  detected: Exercise;
  confidence: number;
  exerciseProbs: number[];
  faults: Record<Fault, number>;
  activeFaults: Fault[];
  counts: Record<Counted, number>;
  reps: number;
  repProgress: number;
  active: boolean;
  present: boolean;
  paused: boolean;
  mode: Mode;
  fps: number;
  inferMs: number;
}

// Auto mode switches exercise only after this many consecutive confident
// inferences (~0.5 s each at 30 fps), so a misread never steals the count.
const SWITCH_AFTER = 5;
const SWITCH_CONF = 0.7;
// Frames without a visible body before counting holds.
const ABSENT_AFTER = 8;

export class Engine {
  readonly window: number;
  readonly stride = 4;
  private pose!: PoseLandmarker;
  private session!: ort.InferenceSession;
  private buffer: Float32Array[] = [];
  private framesSinceInfer = 0;
  private busy = false;
  private probsEma = new Array(EXERCISES.length).fill(0);
  private faultEma = new Array(FAULTS.length).fill(0);
  private counters = Object.fromEntries(COUNTED.map((e) => [e, new RepCounter()])) as Record<Counted, RepCounter>;
  private autoExercise: Counted | null = null;
  private candidate: Counted | null = null;
  private candidateStreak = 0;
  private absentFrames = 0;
  /** True until the first prediction on a fresh buffer (session start, or return to frame). */
  private fresh = true;
  private lastTs = 0;
  private fpsEma = 30;
  mode: Mode = 'auto';
  paused = false;
  state: EngineState;
  recording: { t: number; points: number[] }[] | null = null;

  constructor(window = 64) {
    this.window = window;
    this.state = this.blankState();
  }

  private blankState(): EngineState {
    return {
      exercise: 'idle', detected: 'idle', confidence: 0, exerciseProbs: new Array(EXERCISES.length).fill(0),
      faults: Object.fromEntries(FAULTS.map((f) => [f, 0])) as Record<Fault, number>,
      activeFaults: [], counts: Object.fromEntries(COUNTED.map((e) => [e, 0])) as Record<Counted, number>,
      reps: 0, repProgress: 0, active: false, present: false, paused: false, mode: 'auto', fps: 0, inferMs: 0,
    };
  }

  async load(modelUrl: string, onStatus: (s: string) => void) {
    onStatus('Loading pose tracker');
    const fileset = await FilesetResolver.forVisionTasks(
      `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`,
    );
    // Many phones expose WebGL but fail to run the GPU delegate, so fall back to CPU.
    const make = (delegate: 'GPU' | 'CPU') => PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: POSE_MODEL, delegate },
      runningMode: 'VIDEO',
      numPoses: 1,
    });
    try {
      this.pose = await make('GPU');
    } catch (err) {
      console.warn('GPU pose delegate failed, using CPU', err);
      this.pose = await make('CPU');
    }
    onStatus('Loading form model');
    // The WASM runtime is bundled by Vite and served from our own origin.
    ort.env.wasm.numThreads = 1; // no cross-origin isolation needed on static hosts
    this.session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
  }

  /** The exercise whose counter receives reps. */
  private get target(): Counted | null {
    return this.mode === 'auto' ? this.autoExercise : this.mode;
  }

  setMode(mode: Mode) {
    this.mode = mode;
    this.candidate = null;
    this.candidateStreak = 0;
    for (const c of Object.values(this.counters)) c.update(1, 0, 0); // break any half-tracked rep
    this.refreshCounts();
  }

  setPaused(paused: boolean) {
    this.paused = paused;
    for (const c of Object.values(this.counters)) c.update(1, 0, 0);
    this.state = { ...this.state, paused, active: false };
  }

  /** Reset every exercise's count to zero (a new session). */
  resetAll() {
    for (const c of Object.values(this.counters)) c.reset();
    this.refreshCounts();
  }

  /** Reset only the exercise currently being counted. */
  resetCurrent() {
    const t = this.target;
    if (t) this.counters[t].reset();
    this.refreshCounts();
  }

  private refreshCounts() {
    const counts = Object.fromEntries(COUNTED.map((e) => [e, this.counters[e].count])) as Record<Counted, number>;
    const t = this.target;
    this.state = {
      ...this.state, counts, mode: this.mode, exercise: t ?? 'idle',
      reps: t ? counts[t] : 0, repProgress: t ? this.counters[t].progress : 0,
    };
  }

  /** Process one video frame. Call once per rendered frame. */
  step(video: HTMLVideoElement, now: number): Frame {
    const aspect = video.videoWidth / Math.max(video.videoHeight, 1);
    const res = this.pose.detectForVideo(video, now);
    const lm = res.landmarks[0];
    const points = new Float32Array(NUM_JOINTS * 3);
    let raw: Frame['raw'] = null;
    let visible = false;
    if (lm) {
      raw = MEDIAPIPE_INDICES.map((i) => ({ x: lm[i].x, y: lm[i].y }));
      let vis = 0;
      MEDIAPIPE_INDICES.forEach((i, j) => {
        points[j * 3] = lm[i].x * aspect;
        points[j * 3 + 1] = lm[i].y;
        points[j * 3 + 2] = lm[i].visibility ?? 1;
        vis += lm[i].visibility ?? 1;
      });
      visible = vis / NUM_JOINTS > 0.35;
    } else if (this.buffer.length) {
      // Nobody in frame: repeat the last pose with zero visibility.
      points.set(this.buffer[this.buffer.length - 1]);
      for (let j = 0; j < NUM_JOINTS; j++) points[j * 3 + 2] = 0;
    }
    this.absentFrames = visible ? 0 : this.absentFrames + 1;
    if (this.absentFrames >= ABSENT_AFTER) {
      // Hold every count while nobody is there; tracking restarts cleanly on return.
      for (const c of Object.values(this.counters)) c.update(1, 0, 0);
      this.buffer = [];
      this.framesSinceInfer = 0;
      this.fresh = true;
      this.state = { ...this.state, present: false, active: false, activeFaults: [] };
    }
    if (visible || (this.buffer.length && this.absentFrames < ABSENT_AFTER)) {
      this.buffer.push(points);
      if (this.buffer.length > this.window) this.buffer.shift();
      this.framesSinceInfer++;
      this.recording?.push({ t: now, points: Array.from(points) });
    }
    if (this.lastTs) this.fpsEma = 0.9 * this.fpsEma + 0.1 * (1000 / Math.max(now - this.lastTs, 1));
    this.lastTs = now;
    this.state.fps = this.fpsEma;

    if (this.buffer.length === this.window && this.framesSinceInfer >= this.stride && !this.busy) {
      void this.infer(this.framesSinceInfer);
      this.framesSinceInfer = 0;
    }
    return { points, raw };
  }

  private async infer(newFrames: number) {
    this.busy = true;
    const t0 = performance.now();
    const data = new Float32Array(this.window * NUM_JOINTS * 3);
    this.buffer.forEach((f, i) => data.set(f, i * NUM_JOINTS * 3));
    const input = new ort.Tensor('float32', data, [1, this.window, NUM_JOINTS, 3]);
    let out: ort.InferenceSession.OnnxValueMapType;
    try {
      out = await this.session.run({ landmarks: input });
    } finally {
      this.busy = false;
    }
    if (this.absentFrames >= ABSENT_AFTER) return; // person left while we were thinking
    const exProbs = out.exercise_probs.data as Float32Array;
    const faultProbs = out.fault_probs.data as Float32Array;
    const phase = out.phase.data as Float32Array;
    const active = out.active_probs.data as Float32Array;

    // The first prediction on a fresh buffer replays the whole window into the
    // counter, so reps finished during the 2 s warm-up still count.
    const fresh = this.fresh;
    this.fresh = false;
    if (fresh) newFrames = this.window;
    for (let i = 0; i < EXERCISES.length; i++) {
      this.probsEma[i] = fresh ? exProbs[i] : 0.7 * this.probsEma[i] + 0.3 * exProbs[i];
    }
    for (let i = 0; i < FAULTS.length; i++) this.faultEma[i] = 0.8 * this.faultEma[i] + 0.2 * faultProbs[i];
    const best = this.probsEma.indexOf(Math.max(...this.probsEma));
    const detected = EXERCISES[best];

    // Auto mode: switch only after a sustained, confident detection. Counts are never reset.
    if (detected !== 'idle' && this.probsEma[best] > SWITCH_CONF && detected !== this.autoExercise) {
      this.candidateStreak = detected === this.candidate ? this.candidateStreak + 1 : 1;
      this.candidate = detected;
      if (this.candidateStreak >= SWITCH_AFTER || this.autoExercise === null) {
        if (this.autoExercise) this.counters[this.autoExercise].update(1, 0, 0);
        this.autoExercise = detected;
        this.candidate = null;
        this.candidateStreak = 0;
      }
    } else if (detected === this.autoExercise) {
      this.candidate = null;
      this.candidateStreak = 0;
    }

    const target = this.target;
    // In auto mode, don't count while the model is confident nobody is exercising
    // or is still deciding that you switched to a different exercise.
    const gate = !this.paused && target !== null
      && (this.mode !== 'auto' || (detected === target || this.probsEma[0] < 0.6));
    let isActive = false;
    if (target) {
      const counter = this.counters[target];
      for (let t = this.window - Math.min(newFrames, this.window); t < this.window; t++) {
        const a = gate ? active[t] : 0;
        counter.update(phase[t * 2], phase[t * 2 + 1], a);
        isActive = a > 0.5;
      }
    }

    const faults = Object.fromEntries(FAULTS.map((f, i) => [f, this.faultEma[i]])) as Record<Fault, number>;
    this.state = {
      ...this.state,
      detected,
      present: true,
      confidence: this.probsEma[best],
      exerciseProbs: [...this.probsEma],
      faults,
      activeFaults: target && !this.paused
        ? APPLICABLE[target].filter((f) => faults[f] > 0.6 && (this.mode !== 'auto' || this.probsEma[best] > 0.6))
        : [],
      active: isActive,
      paused: this.paused,
      inferMs: 0.8 * this.state.inferMs + 0.2 * (performance.now() - t0),
    };
    this.refreshCounts();
  }
}
