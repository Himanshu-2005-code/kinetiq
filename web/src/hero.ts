// Landing-page hero: replays a synthetic squat set and counts it with the real rep counter.
import { RepCounter } from './reps';
import { BONES } from './skeleton';

interface HeroClip {
  aspect: number;
  fps: number;
  frames: number[][][]; // T x 13 x [x, y, vis]
  phase: number[];
  active: number[];
}

const canvas = document.getElementById('hero-canvas') as HTMLCanvasElement;
const repsEl = document.getElementById('hero-reps')!;
const ctx = canvas.getContext('2d')!;
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function draw(clip: HeroClip, t: number) {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const f = clip.frames[t];
  const s = H * 0.95;
  const ox = (W - clip.aspect * s) / 2;
  const P = (j: number) => [ox + f[j][0] * s, f[j][1] * s + H * 0.02];
  ctx.lineCap = 'round';
  ctx.strokeStyle = '#f5c518';
  ctx.lineWidth = 9;
  for (const [a, b] of BONES) {
    const [x1, y1] = P(a), [x2, y2] = P(b);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }
  ctx.fillStyle = '#eef2ef';
  for (let j = 1; j < 13; j++) {
    const [x, y] = P(j);
    ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2); ctx.fill();
  }
  const [nx, ny] = P(0);
  ctx.beginPath(); ctx.arc(nx, ny, 22, 0, Math.PI * 2);
  ctx.lineWidth = 7; ctx.stroke();
}

async function main() {
  const clip: HeroClip = await (await fetch('./hero-clip.json')).json();
  const counter = new RepCounter();
  if (reduced) {
    draw(clip, Math.floor(clip.frames.length * 0.3));
    return;
  }
  let t = 0, last = 0;
  const frameMs = 1000 / clip.fps;
  const tick = (now: number) => {
    if (now - last >= frameMs) {
      last = now;
      if (t === 0) counter.reset();
      counter.update(Math.cos(clip.phase[t]), Math.sin(clip.phase[t]), clip.active[t]);
      repsEl.textContent = String(counter.count);
      draw(clip, t);
      t = (t + 1) % clip.frames.length;
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

main();
