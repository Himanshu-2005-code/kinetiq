// Camera -> MediaPipe PoseLandmarker -> Kinetiq ONNX model, all on-device.
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import * as ort from 'onnxruntime-web/wasm';
import { RepCounter } from './reps';
import { APPLICABLE, EXERCISES, Exercise, FAULTS, Fault, MEDIAPIPE_INDICES, NUM_JOINTS } from './skeleton';

const MP_VERSION = '0.10.35';
const POSE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

export interface Frame {
  points: Float32Array; // NUM_JOINTS * 3: x (aspect-scaled), y, visibility
  raw: { x: number; y: number }[] | null; // normalised 0..1 for drawing
}

export interface EngineState {
  exercise: Exercise;
  confidence: number;
  exerciseProbs: number[];
  faults: Record<Fault, number>;
  activeFaults: Fault[];
  reps: number;
  repProgress: number;
  active: boolean;
  fps: number;
  inferMs: number;
}

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
  private counter = new RepCounter();
  private lastTs = 0;
  private fpsEma = 30;
  private lastExercise: Exercise = 'idle';
  state: EngineState;
  recording: { t: number; points: number[] }[] | null = null;

  constructor(window = 64) {
    this.window = window;
    this.state = this.blankState();
  }

  private blankState(): EngineState {
    return {
      exercise: 'idle', confidence: 0, exerciseProbs: new Array(EXERCISES.length).fill(0),
      faults: Object.fromEntries(FAULTS.map((f) => [f, 0])) as Record<Fault, number>,
      activeFaults: [], reps: 0, repProgress: 0, active: false, fps: 0, inferMs: 0,
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

  resetCount() {
    this.counter.reset();
    this.state.reps = 0;
  }

  /** Process one video frame. Call once per rendered frame. */
  step(video: HTMLVideoElement, now: number): Frame {
    const aspect = video.videoWidth / Math.max(video.videoHeight, 1);
    const res = this.pose.detectForVideo(video, now);
    const lm = res.landmarks[0];
    const points = new Float32Array(NUM_JOINTS * 3);
    let raw: Frame['raw'] = null;
    if (lm) {
      raw = MEDIAPIPE_INDICES.map((i) => ({ x: lm[i].x, y: lm[i].y }));
      MEDIAPIPE_INDICES.forEach((i, j) => {
        points[j * 3] = lm[i].x * aspect;
        points[j * 3 + 1] = lm[i].y;
        points[j * 3 + 2] = lm[i].visibility ?? 1;
      });
    } else if (this.buffer.length) {
      // Nobody in frame: repeat the last pose with zero visibility.
      points.set(this.buffer[this.buffer.length - 1]);
      for (let j = 0; j < NUM_JOINTS; j++) points[j * 3 + 2] = 0;
    }
    if (lm || this.buffer.length) {
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
    const out = await this.session.run({ landmarks: input });
    const exProbs = out.exercise_probs.data as Float32Array;
    const faultProbs = out.fault_probs.data as Float32Array;
    const phase = out.phase.data as Float32Array;
    const active = out.active_probs.data as Float32Array;

    for (let i = 0; i < EXERCISES.length; i++) this.probsEma[i] = 0.7 * this.probsEma[i] + 0.3 * exProbs[i];
    for (let i = 0; i < FAULTS.length; i++) this.faultEma[i] = 0.8 * this.faultEma[i] + 0.2 * faultProbs[i];
    const best = this.probsEma.indexOf(Math.max(...this.probsEma));
    const exercise = EXERCISES[best];
    if (exercise !== this.lastExercise && exercise !== 'idle') this.counter.reset();
    if (exercise !== 'idle') this.lastExercise = exercise;

    // Feed the counter with the newest frames only (no lookahead beyond the stride).
    let isActive = false;
    for (let t = this.window - Math.min(newFrames, this.window); t < this.window; t++) {
      const a = exercise === 'idle' ? 0 : active[t];
      this.counter.update(phase[t * 2], phase[t * 2 + 1], a);
      isActive = a > 0.5;
    }

    const faults = Object.fromEntries(FAULTS.map((f, i) => [f, this.faultEma[i]])) as Record<Fault, number>;
    this.state = {
      ...this.state,
      exercise,
      confidence: this.probsEma[best],
      exerciseProbs: [...this.probsEma],
      faults,
      activeFaults: APPLICABLE[exercise].filter((f) => faults[f] > 0.6 && this.probsEma[best] > 0.6),
      reps: this.counter.count,
      repProgress: this.counter.progress,
      active: isActive,
      inferMs: 0.8 * this.state.inferMs + 0.2 * (performance.now() - t0),
    };
    this.busy = false;
  }
}
