#!/usr/bin/env python3
"""
Render 2D skeleton from .sam3dbody file to MP4 on black background.

Usage:
    python render_skeleton.py input.sam3dbody -o output.mp4
    python render_skeleton.py input.sam3dbody -o output.mp4 --width 720 --height 720
"""

import argparse
import os

import cv2
import numpy as np

# MHR 70-keypoint names (index → name)
KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
    # Right hand (21-41)
    "right_thumb_tip", "right_thumb_1st", "right_thumb_2nd", "right_thumb_3rd",
    "right_index_tip", "right_index_1st", "right_index_2nd", "right_index_3rd",
    "right_middle_tip", "right_middle_1st", "right_middle_2nd", "right_middle_3rd",
    "right_ring_tip", "right_ring_1st", "right_ring_2nd", "right_ring_3rd",
    "right_pinky_tip", "right_pinky_1st", "right_pinky_2nd", "right_pinky_3rd",
    "right_wrist",
    # Left hand (42-62)
    "left_thumb_tip", "left_thumb_1st", "left_thumb_2nd", "left_thumb_3rd",
    "left_index_tip", "left_index_1st", "left_index_2nd", "left_index_3rd",
    "left_middle_tip", "left_middle_1st", "left_middle_2nd", "left_middle_3rd",
    "left_ring_tip", "left_ring_1st", "left_ring_2nd", "left_ring_3rd",
    "left_pinky_tip", "left_pinky_1st", "left_pinky_2nd", "left_pinky_3rd",
    "left_wrist",
    # Extra (63-69)
    "left_olecranon", "right_olecranon",
    "left_cubital_fossa", "right_cubital_fossa",
    "left_acromion", "right_acromion", "neck",
]

# Skeleton bones as (idx_a, idx_b) pairs with colors (BGR)
BLUE = (255, 153, 51)
GREEN = (0, 255, 0)
ORANGE = (0, 128, 255)
PINK = (255, 153, 255)
LIGHT_BLUE = (255, 178, 102)
RED = (51, 51, 255)

# Body bones
BODY_BONES = [
    # Left leg (green)
    (13, 11, GREEN), (11, 9, GREEN),
    # Right leg (orange)
    (14, 12, ORANGE), (12, 10, ORANGE),
    # Torso (blue)
    (9, 10, BLUE), (5, 9, BLUE), (6, 10, BLUE), (5, 6, BLUE),
    # Left arm (green)
    (5, 7, GREEN), (7, 62, GREEN),
    # Right arm (orange)
    (6, 8, ORANGE), (8, 41, ORANGE),
    # Face (blue)
    (1, 2, BLUE), (0, 1, BLUE), (0, 2, BLUE), (1, 3, BLUE), (2, 4, BLUE),
    (3, 5, BLUE), (4, 6, BLUE),
    # Left foot (green)
    (13, 15, GREEN), (13, 16, GREEN), (13, 17, GREEN),
    # Right foot (orange)
    (14, 18, ORANGE), (14, 19, ORANGE), (14, 20, ORANGE),
]

# Hand bones: wrist → 3rd joint → 2nd → 1st → tip for each finger
def _hand_bones(wrist, base, color_thumb, color_index, color_middle, color_ring, color_pinky):
    """Generate hand bone list. base is the first finger keypoint index."""
    bones = []
    fingers = [
        (color_thumb, [3, 2, 1, 0]),    # thumb: 3rd, 2nd, 1st, tip
        (color_index, [7, 6, 5, 4]),    # index
        (color_middle, [11, 10, 9, 8]), # middle
        (color_ring, [15, 14, 13, 12]), # ring
        (color_pinky, [19, 18, 17, 16]),# pinky
    ]
    for color, offsets in fingers:
        # wrist → 3rd joint
        bones.append((wrist, base + offsets[0], color))
        # chain: 3rd → 2nd → 1st → tip
        for i in range(len(offsets) - 1):
            bones.append((base + offsets[i], base + offsets[i + 1], color))
    return bones

# Right hand (indices 21-40, wrist=41)
RIGHT_HAND_BONES = _hand_bones(41, 21, ORANGE, PINK, LIGHT_BLUE, RED, GREEN)
# Left hand (indices 42-61, wrist=62)
LEFT_HAND_BONES = _hand_bones(62, 42, ORANGE, PINK, LIGHT_BLUE, RED, GREEN)

ALL_BONES = BODY_BONES + LEFT_HAND_BONES + RIGHT_HAND_BONES

# Keypoint colors
KPT_COLOR = (255, 153, 51)  # blue-ish in BGR


def draw_skeleton(frame, kpts, line_width=2, radius=4):
    """Draw skeleton on frame. kpts: (70, 2) array of x,y coordinates."""
    h, w = frame.shape[:2]

    # Draw bones
    for a, b, color in ALL_BONES:
        x1, y1 = int(kpts[a, 0]), int(kpts[a, 1])
        x2, y2 = int(kpts[b, 0]), int(kpts[b, 1])
        # Skip if either point is out of frame
        if x1 <= 0 or x1 >= w or y1 <= 0 or y1 >= h:
            continue
        if x2 <= 0 or x2 >= w or y2 <= 0 or y2 >= h:
            continue
        cv2.line(frame, (x1, y1), (x2, y2), color, line_width)

    # Draw keypoints
    for i in range(kpts.shape[0]):
        x, y = int(kpts[i, 0]), int(kpts[i, 1])
        if x <= 0 or x >= w or y <= 0 or y >= h:
            continue
        cv2.circle(frame, (x, y), radius, KPT_COLOR, -1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Render 2D skeleton to MP4")
    parser.add_argument("input", help="Path to .sam3dbody file")
    parser.add_argument("-o", "--output", default=None, help="Output MP4 path")
    parser.add_argument("--width", type=int, default=0, help="Output width (0 = from metadata)")
    parser.add_argument("--height", type=int, default=0, help="Output height (0 = from metadata)")
    parser.add_argument("--fps", type=float, default=0, help="FPS (0 = from metadata)")
    parser.add_argument("--line_width", type=int, default=2, help="Skeleton line width")
    parser.add_argument("--radius", type=int, default=4, help="Keypoint circle radius")
    parser.add_argument("--zoom", type=float, default=1.0,
                        help="Zoom factor (< 1.0 zooms out, e.g. 0.7 = 30%% zoom out)")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = base + "_skeleton.mp4"

    # Load data
    data = np.load(args.input, allow_pickle=True)
    keypoints_2d = data["body_keypoints_2d"]  # (frames, 70, 2)
    num_frames = keypoints_2d.shape[0]

    # Metadata
    metadata = {}
    if "metadata" in data:
        meta_arr = data["metadata"]
        metadata = meta_arr[0] if meta_arr.shape == (1,) else meta_arr.item()

    fps = args.fps if args.fps > 0 else metadata.get("fps", 30.0)
    src_w = metadata.get("width", 1920)
    src_h = metadata.get("height", 1080)
    out_w = args.width if args.width > 0 else src_w
    out_h = args.height if args.height > 0 else src_h

    print(f"{num_frames} frames, {fps:.2f} fps, source {src_w}x{src_h}, output {out_w}x{out_h}")

    # Scale factor
    sx = out_w / src_w
    sy = out_h / src_h

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (out_w, out_h))

    for i in range(num_frames):
        frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        kpts = keypoints_2d[i].copy()
        kpts[:, 0] *= sx
        kpts[:, 1] *= sy

        # Apply zoom: scale keypoints around the frame center
        if args.zoom != 1.0:
            cx, cy = out_w / 2.0, out_h / 2.0
            kpts[:, 0] = cx + (kpts[:, 0] - cx) * args.zoom
            kpts[:, 1] = cy + (kpts[:, 1] - cy) * args.zoom

        frame = draw_skeleton(frame, kpts, args.line_width, args.radius)
        writer.write(frame)

    writer.release()
    print(f"Saved to: {args.output}")

    # Re-encode H264
    h264_path = args.output.replace(".mp4", "_h264.mp4")
    ret = os.system(
        f'ffmpeg -y -i "{args.output}" -c:v libx264 -pix_fmt yuv420p "{h264_path}" -loglevel warning'
    )
    if ret == 0:
        print(f"H264 version: {h264_path}")


if __name__ == "__main__":
    main()
