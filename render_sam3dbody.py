#!/usr/bin/env python3
"""
Render a .sam3dbody file (multi-frame NPZ with MHR params) to MP4 video.

Supports two input formats:
  1. Single multi-frame .sam3dbody/.npz file (arrays have shape [N, ...])
  2. Directory of per-frame .npz files (original render_npz_to_mp4.py format)

Usage:
    # Single multi-frame file:
    python render_sam3dbody.py \
        --input ./R20260213_8551.sam3dbody \
        --faces_path ./mhr_faces_lod1.npy \
        --mhr_model_path ./sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
        --output_path ./output.mp4

    # Directory of per-frame NPZ files:
    python render_sam3dbody.py \
        --input ./output_npz/ \
        --faces_path ./mhr_faces_lod1.npy \
        --output_path ./output.mp4
"""

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import glob

import cv2
import numpy as np
import pyrender
import torch
import trimesh
from tqdm import tqdm


def render_mesh_frame(
    vertices,
    faces,
    cam_t,
    focal_length,
    renderer,
    img_width=512,
    img_height=512,
    bg_color=(255, 255, 255, 255),
    mesh_color=(0.6, 0.75, 0.9, 1.0),
):
    """Render a single mesh frame. Reuses the provided OffscreenRenderer."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.7,
        alphaMode="OPAQUE",
        baseColorFactor=mesh_color,
    )
    mesh_pr = pyrender.Mesh.from_trimesh(mesh, material=material)

    scene = pyrender.Scene(
        bg_color=np.array(bg_color) / 255.0,
        ambient_light=[0.3, 0.3, 0.3],
    )
    scene.add(mesh_pr)

    yfov = 2.0 * np.arctan(img_height / (2.0 * focal_length))
    camera = pyrender.PerspectiveCamera(
        yfov=yfov,
        aspectRatio=img_width / img_height,
        znear=0.01,
        zfar=100.0,
    )
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = cam_t
    scene.add(camera, pose=cam_pose)

    # Key light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=cam_pose)

    # Fill light from the side
    fill_pose = cam_pose.copy()
    fill_pose[0, 3] -= 1.0
    fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    scene.add(fill_light, pose=fill_pose)

    color, _ = renderer.render(scene)
    return color


def reconstruct_vertices_batch(mhr_ts, body_pose, hand_pose, shape, expr=None):
    """Reconstruct vertices for a single frame from MHR parameters."""
    identity = torch.tensor(shape, dtype=torch.float32).unsqueeze(0)

    # Build 204-dim model params from body + hand pose
    model_params = np.concatenate([body_pose, hand_pose])
    if model_params.shape[0] < 204:
        model_params = np.pad(model_params, (0, 204 - model_params.shape[0]))
    model_params = torch.tensor(model_params[:204], dtype=torch.float32).unsqueeze(0)

    if expr is not None and expr.shape[0] > 0:
        expr_t = torch.tensor(expr, dtype=torch.float32).unsqueeze(0)
    else:
        expr_t = torch.zeros(1, 72)

    with torch.no_grad():
        vertices, skel_state = mhr_ts(identity, model_params, expr_t)

    return vertices.squeeze(0).cpu().numpy()


def estimate_cam_t_from_bbox(bbox, focal_length, img_width, img_height):
    """
    Estimate a camera translation that places the mesh roughly centered.
    bbox: [x1, y1, x2, y2] in pixel coords of the original image.
    """
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    bbox_h = bbox[3] - bbox[1]

    # Approximate: assume person is ~1.7m tall, use bbox height to estimate depth
    person_height = 1.7
    tz = (focal_length * person_height) / max(bbox_h, 1.0)

    # Convert pixel center offset to 3D translation
    tx = (cx - img_width / 2.0) * tz / focal_length
    ty = (cy - img_height / 2.0) * tz / focal_length

    return np.array([tx, ty, tz])


def load_multiframe_file(path):
    """Load a multi-frame .sam3dbody / .npz file."""
    data = np.load(path, allow_pickle=True)
    result = {k: data[k] for k in data.files}

    # Extract metadata
    metadata = {}
    if "metadata" in result:
        meta_arr = result["metadata"]
        if meta_arr.shape == (1,):
            metadata = meta_arr[0]
        elif meta_arr.shape == ():
            metadata = meta_arr.item()

    return result, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Render SAM 3D Body output to MP4"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to .sam3dbody/.npz file or directory of per-frame NPZ files",
    )
    parser.add_argument(
        "--faces_path",
        required=True,
        help="Path to saved faces array (.npy) from MHR model",
    )
    parser.add_argument(
        "--output_path", default="output.mp4", help="Output MP4 file path"
    )
    parser.add_argument(
        "--mhr_model_path",
        default=None,
        help="Path to TorchScript MHR model (.pt) for vertex reconstruction",
    )
    parser.add_argument("--fps", type=float, default=0, help="FPS (0 = auto from metadata)")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--focal_length",
        type=float,
        default=0,
        help="Default focal length (0 = auto from data)",
    )
    parser.add_argument(
        "--person_idx",
        type=int,
        default=0,
        help="Which person to render (0-indexed)",
    )
    args = parser.parse_args()

    # Load face topology
    faces = np.load(args.faces_path)
    print(f"Loaded faces: {faces.shape}")

    # Load MHR model if provided
    mhr_ts = None
    if args.mhr_model_path:
        mhr_ts = torch.jit.load(args.mhr_model_path, map_location="cpu")
        print(f"Loaded MHR TorchScript model")

    # Determine input type: single file or directory
    is_multiframe = os.path.isfile(args.input)

    if is_multiframe:
        data, metadata = load_multiframe_file(args.input)
        num_frames = metadata.get("num_frames", 0)
        src_fps = metadata.get("fps", 30.0)
        src_width = metadata.get("width", 512)
        src_height = metadata.get("height", 512)

        # Infer num_frames from array shapes if not in metadata
        if num_frames == 0:
            for key in ["body_pose_params", "body_keypoints_3d", "pred_vertices"]:
                if key in data:
                    num_frames = data[key].shape[0]
                    break

        fps = args.fps if args.fps > 0 else src_fps
        print(f"Multi-frame file: {num_frames} frames, {fps:.2f} fps, src {src_width}x{src_height}")

        # Build per-frame data list
        frames = []
        for i in range(num_frames):
            frame = {}
            # Check for pre-computed vertices first
            if "pred_vertices" in data:
                v = data["pred_vertices"]
                frame["vertices"] = v[i] if v.ndim == 3 else v

            # Camera translation
            if "pred_cam_t" in data:
                ct = data["pred_cam_t"]
                frame["cam_t"] = ct[i] if ct.ndim == 2 else ct

            # MHR parameters for reconstruction
            for key in ["body_pose_params", "hand_pose_params", "shape_params",
                        "global_rot", "expr_params", "mhr_model_params"]:
                if key in data:
                    arr = data[key]
                    frame[key] = arr[i] if arr.ndim >= 2 else arr

            # Per-frame focal length and bbox
            if "focal_lengths" in data:
                frame["focal_length"] = float(data["focal_lengths"][i])
            elif "focal_length" in data:
                fl = data["focal_length"]
                frame["focal_length"] = float(fl[i]) if fl.ndim >= 1 and fl.shape[0] > 1 else float(fl)

            if "bboxes" in data:
                frame["bbox"] = data["bboxes"][i]

            frames.append(frame)

    else:
        # Directory of per-frame NPZ files
        npz_files = sorted(glob.glob(os.path.join(args.input, "*.npz")))
        if not npz_files:
            print(f"No .npz files found in {args.input}")
            return

        fps = args.fps if args.fps > 0 else 30.0
        src_width = 512
        src_height = 512
        print(f"Found {len(npz_files)} per-frame NPZ files, {fps:.2f} fps")

        frames = []
        for path in npz_files:
            raw = np.load(path, allow_pickle=True)
            frame = {k: raw[k] for k in raw.files}
            if "pred_vertices" in frame:
                frame["vertices"] = frame["pred_vertices"]
            if "pred_cam_t" in frame:
                frame["cam_t"] = frame["pred_cam_t"]
            frames.append(frame)

    if not frames:
        print("No frames to render")
        return

    # Determine default focal length
    default_focal = args.focal_length
    if default_focal == 0:
        # Try to get from first frame
        if "focal_length" in frames[0]:
            default_focal = frames[0]["focal_length"]
        else:
            default_focal = 500.0

    # Initialize video writer and renderer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output_path, fourcc, fps, (args.width, args.height))
    renderer = pyrender.OffscreenRenderer(args.width, args.height)

    skipped = 0
    for i, frame in enumerate(tqdm(frames, desc="Rendering")):
        # Get or reconstruct vertices
        vertices = frame.get("vertices")

        if vertices is None and mhr_ts is not None:
            body_pose = frame.get("body_pose_params")
            hand_pose = frame.get("hand_pose_params")
            shape = frame.get("shape_params")
            expr = frame.get("expr_params")

            if body_pose is not None and shape is not None:
                if hand_pose is None:
                    hand_pose = np.zeros(108)
                vertices = reconstruct_vertices_batch(
                    mhr_ts, body_pose, hand_pose, shape, expr
                )

        if vertices is None:
            skipped += 1
            continue

        # Handle multi-person
        if vertices.ndim == 3:
            vertices = vertices[args.person_idx]

        # Get camera translation
        cam_t = frame.get("cam_t")
        focal = frame.get("focal_length", default_focal)

        if cam_t is None:
            # Estimate from bbox or use a default centered view
            bbox = frame.get("bbox")
            if bbox is not None:
                cam_t = estimate_cam_t_from_bbox(bbox, focal, src_width, src_height)
            else:
                # Place camera looking at mesh center
                center = vertices.mean(axis=0)
                cam_t = np.array([center[0], center[1], center[2] + 3.0])

        if cam_t.ndim == 2:
            cam_t = cam_t[args.person_idx]

        # Scale focal length to render resolution
        # The focal length is for the original image dimensions, scale to render dims
        focal_scaled = focal * args.height / src_height

        rendered = render_mesh_frame(
            vertices,
            faces,
            cam_t,
            focal_scaled,
            renderer,
            img_width=args.width,
            img_height=args.height,
        )

        frame_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    renderer.delete()
    writer.release()

    rendered_count = len(frames) - skipped
    print(f"Rendered {rendered_count} frames ({skipped} skipped)")
    print(f"Video saved to: {args.output_path}")

    if rendered_count > 0:
        h264_path = args.output_path.replace(".mp4", "_h264.mp4")
        ret = os.system(
            f'ffmpeg -y -i "{args.output_path}" -c:v libx264 -pix_fmt yuv420p "{h264_path}" -loglevel warning'
        )
        if ret == 0:
            print(f"H264 version saved to: {h264_path}")


if __name__ == "__main__":
    main()
