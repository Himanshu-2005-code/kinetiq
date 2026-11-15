// Contract with ml/kinetiq/skeleton.py. Keep both files in sync.

export const MEDIAPIPE_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
export const NUM_JOINTS = MEDIAPIPE_INDICES.length;

export const BONES: [number, number][] = [
  [1, 2], [1, 3], [3, 5], [2, 4], [4, 6], [1, 7], [2, 8], [7, 8], [7, 9], [9, 11], [8, 10], [10, 12],
];

export const EXERCISES = ['idle', 'squat', 'pushup', 'lunge', 'jumping_jack'] as const;
export const FAULTS = ['partial_range', 'forward_lean', 'hip_sag', 'hip_pike'] as const;
export type Exercise = (typeof EXERCISES)[number];
export type Fault = (typeof FAULTS)[number];

export const EXERCISE_LABEL: Record<Exercise, string> = {
  idle: 'Waiting for you to start',
  squat: 'Squats',
  pushup: 'Push-ups',
  lunge: 'Lunges',
  jumping_jack: 'Jumping jacks',
};

export const APPLICABLE: Record<Exercise, Fault[]> = {
  idle: [],
  squat: ['partial_range', 'forward_lean'],
  pushup: ['partial_range', 'hip_sag', 'hip_pike'],
  lunge: ['partial_range', 'forward_lean'],
  jumping_jack: ['partial_range'],
};

export function cue(ex: Exercise, fault: Fault): string {
  if (fault === 'partial_range') {
    return {
      squat: 'Sit deeper. Aim for thighs parallel to the floor.',
      pushup: 'Lower your chest closer to the floor.',
      lunge: 'Drop your back knee lower.',
      jumping_jack: 'Bring your arms all the way overhead.',
      idle: '',
    }[ex];
  }
  return {
    forward_lean: 'Keep your chest up. You are folding forward.',
    hip_sag: 'Brace your core. Your hips are sagging.',
    hip_pike: 'Lower your hips into one straight line.',
  }[fault];
}
