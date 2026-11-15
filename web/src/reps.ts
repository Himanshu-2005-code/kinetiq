// Streaming rep counter. Mirrors ml/kinetiq/reps.py line for line, so the
// browser counts reps exactly the way the offline evaluation does.

const TWO_PI = Math.PI * 2;

export const wrap = (a: number) => ((((a + Math.PI) % TWO_PI) + TWO_PI) % TWO_PI) - Math.PI;

export class RepCounter {
  count = 0;
  private total = 0;
  private last: number | null = null;

  constructor(private activeThreshold = 0.5, private maxStep = Math.PI * 0.9) {}

  reset() {
    this.count = 0;
    this.total = 0;
    this.last = null;
  }

  /** Feed one frame. Returns the running rep count. */
  update(cos: number, sin: number, activeProb: number): number {
    const angle = ((Math.atan2(sin, cos) % TWO_PI) + TWO_PI) % TWO_PI;
    if (activeProb < this.activeThreshold) {
      this.last = null;
      return this.count;
    }
    if (this.last === null) {
      const start = angle > 1.5 * Math.PI ? angle - TWO_PI : angle;
      this.total = this.count * TWO_PI + start;
    } else {
      const step = wrap(angle - this.last);
      if (Math.abs(step) < this.maxStep) this.total += step;
    }
    this.last = angle;
    while (this.total >= (this.count + 1) * TWO_PI) this.count++;
    return this.count;
  }

  /** Progress through the current rep, 0..1, for the progress ring. */
  get progress(): number {
    return Math.min(Math.max(this.total / TWO_PI - this.count, 0), 1);
  }
}
