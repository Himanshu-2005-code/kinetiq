"""Skeleton definition shared by the synthetic generator, the model and the web client.

Kinetiq uses a 13-joint subset of MediaPipe Pose's 33 landmarks. The web client
selects these indices from PoseLandmarker output, so the order here is the
contract between training and inference.
"""

JOINTS = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
NUM_JOINTS = len(JOINTS)

# Index of each Kinetiq joint inside MediaPipe Pose's 33-landmark output.
MEDIAPIPE_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

J = {name: i for i, name in enumerate(JOINTS)}

# Joint angles fed to the model: (a, b, c) means the angle at b between b->a and b->c.
ANGLE_TRIPLETS = [
    (J["left_shoulder"], J["left_elbow"], J["left_wrist"]),
    (J["right_shoulder"], J["right_elbow"], J["right_wrist"]),
    (J["left_hip"], J["left_shoulder"], J["left_elbow"]),
    (J["right_hip"], J["right_shoulder"], J["right_elbow"]),
    (J["left_shoulder"], J["left_hip"], J["left_knee"]),
    (J["right_shoulder"], J["right_hip"], J["right_knee"]),
    (J["left_hip"], J["left_knee"], J["left_ankle"]),
    (J["right_hip"], J["right_knee"], J["right_ankle"]),
]

BONES = [
    (J["left_shoulder"], J["right_shoulder"]),
    (J["left_shoulder"], J["left_elbow"]), (J["left_elbow"], J["left_wrist"]),
    (J["right_shoulder"], J["right_elbow"]), (J["right_elbow"], J["right_wrist"]),
    (J["left_shoulder"], J["left_hip"]), (J["right_shoulder"], J["right_hip"]),
    (J["left_hip"], J["right_hip"]),
    (J["left_hip"], J["left_knee"]), (J["left_knee"], J["left_ankle"]),
    (J["right_hip"], J["right_knee"]), (J["right_knee"], J["right_ankle"]),
]

EXERCISES = ["idle", "squat", "pushup", "lunge", "jumping_jack"]
FAULTS = ["partial_range", "forward_lean", "hip_sag", "hip_pike"]

# Which faults can occur for each exercise (used by the generator and the UI).
APPLICABLE_FAULTS = {
    "idle": [],
    "squat": ["partial_range", "forward_lean"],
    "pushup": ["partial_range", "hip_sag", "hip_pike"],
    "lunge": ["partial_range", "forward_lean"],
    "jumping_jack": ["partial_range"],
}
