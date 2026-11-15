# Kinetiq

**A real-time AI form coach that runs entirely in the browser.** Point a laptop camera at yourself and start exercising. Kinetiq recognises the exercise, counts your reps and flags form faults while you move. No video ever leaves the device.

Live site: https://himanshu-2005-code.github.io/kinetiq/

```
webcam ──► MediaPipe PoseLandmarker ──► 13 joints / frame ──► 64-frame window ──► Kinetiq motion transformer (ONNX, WASM)
                                                                                    ├─ exercise         (softmax, 5 classes)
                                                                                    ├─ form faults      (4 sigmoids)
                                                                                    ├─ rep phase        (per-frame unit vector)
                                                                                    └─ activity         (per-frame sigmoid)
                                                                     phase unwrapping ──► rep count
```

## What makes it interesting

**1. Training data with no dataset.** Labelled motion data with per-frame rep phase and form-fault labels doesn't exist publicly. `ml/kinetiq/synth.py` is a procedural motion synthesiser:
- a 3D skeleton with randomised body proportions, with arms and legs solved by two-bone inverse kinematics so bone lengths stay exactly constant (a unit test checks this);
- exercise controllers for squats, push-ups, lunges and jumping jacks, plus "idle" fidgeting. Each one is driven by a per-rep phase signal with tempo drift, pauses at the top and bottom, and rep-to-rep amplitude variation;
- form faults injected as kinematic changes: shallow depth, folding forward, sagging or piking hips;
- a randomised weak-perspective camera (yaw, elevation, framing, aspect ratio, mirrored selfie view);
- pose-estimator corruption: jitter, low-frequency drift, depth-based self-occlusion, transient joint dropouts and frame stalls.

It generates ~2,000 labelled clips per second on a laptop CPU, so training never sees the same clip twice.

**2. Preprocessing inside the graph.** The model takes raw landmarks. Centering, scale normalisation, velocities, mirror-invariant joint angles and trunk orientation are computed in PyTorch inside `forward`. The exported ONNX file is therefore the only contract between training and the browser, and there's no duplicated feature code to drift out of sync. Unit tests check that the features don't change under scaling, translation or mirroring.

**3. Rep counting as phase regression.** Instead of peak-detecting a joint angle, which breaks per exercise and per camera angle, the model regresses each frame's position inside a rep as an angle on the unit circle. A tiny streaming counter unwraps that angle and counts full turns, with hysteresis so jitter never double-counts. The same algorithm exists in `ml/kinetiq/reps.py` and `web/src/reps.ts`, and evaluation runs it on sliding windows exactly the way the browser does.

**4. Modern transformer, small footprint.** Pre-norm encoder with rotary position embeddings, SwiGLU feed-forward layers, RMSNorm, stochastic depth, EMA weights, AdamW with warmup and cosine decay, and multi-task losses. It has 0.83M parameters and a 3.5 MB ONNX file, and one inference takes ~1.4 ms on a single CPU thread.

## Results

Synthetic held-out set (2,048 clips) and streaming rep counting on 60 unseen 12–18 s sessions:

| Metric | Value |
|---|---|
| Exercise accuracy (5 classes) | 100.0% |
| Form-fault macro F1 (4 faults) | 0.998 |
| Rep-phase mean abs. error | 3.1° |
| Activity (in-rep) accuracy | 99.3% |
| Rep count, sessions exactly right | 93% |
| Rep count, mean abs. error | 0.08 reps/session |

These are numbers on synthetic data. Real-world accuracy has to be measured on recorded sets (see the roadmap).

## Repository layout

```
ml/
  kinetiq/synth.py       procedural motion + camera + tracker-noise synthesiser
  kinetiq/model.py       in-graph featuriser + RoPE transformer + heads
  kinetiq/train.py       streaming-data training loop (EMA, cosine LR, checkpointing)
  kinetiq/evaluate.py    clip metrics + browser-faithful streaming rep evaluation
  kinetiq/reps.py        phase-unwrapping rep counter
  kinetiq/export.py      ONNX export + onnxruntime parity check
  configs/default.yaml
  tests/                 13 tests: IK invariants, feature invariances, ONNX parity, counter
web/
  index.html             landing page (hero replays a synthetic set, counted live)
  app.html               live coach
  src/engine.ts          camera → MediaPipe → ONNX loop, smoothing, counter
  public/models/         exported model + metadata
.github/workflows/       CI (pytest + web build) and GitHub Pages deploy
```

## Run it

```bash
# train (CPU is fine: ~15 min for 3k steps)
cd ml
pip install -e '.[dev]'
pytest -q
python -m kinetiq.train --config configs/default.yaml
python -m kinetiq.export runs/default/best.pt ../web/public/models/kinetiq.onnx

# web
cd ../web
npm install
npm run dev        # http://localhost:5173 (camera works on localhost)
npm run build      # static site in web/dist, deployable anywhere
```

Pushing to `main` deploys the site to GitHub Pages through `.github/workflows/deploy.yml`.

## Roadmap

- **Real-data fine-tuning.** The live coach has a *Record a set* button that exports raw landmark sequences. Collect labelled sets, then fine-tune the exercise and fault heads (domain adaptation from synthetic to real).
- More exercises (deadlift, plank hold timer, burpees) and per-side faults such as knee valgus from front views.
- Spoken cues with the Web Speech API, and workout history kept in IndexedDB.
- WebGPU execution provider for the motion model on supported browsers.

Kinetiq is not a medical device.
