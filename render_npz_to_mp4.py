#!/usr/bin/env python3
"""
Render SAM 3D Body NPZ output files to MP4 video.

Usage:
    python render_npz_to_mp4.py \
        --npz_dir ./output_npz/ \
        --faces_path ./mhr_faces_lod1.npy \
        --output_path ./output.mp4 \
        --fps 30 \
        --width 512 \
        --height 512

    # With overlay on original images:
    python render_npz_to_mp4.py \
        --npz_dir ./output_npz/ \
        --faces_path ./mhr_faces_lod1.npy \
        --image_dir ./input_frames/ \
        --output_path ./output_overlay.mp4 \
        --mode overlay

    # Reconstruct vertices from MHR params (if pred_vertices missing):
    python render_npz_to_mp4.py \
        --npz_dir ./output_npz/ \
        --faces_path ./mhr_faces_lod1.npy \
        --mhr_model_path ./sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
        --output_path ./output.mp4
"""

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import glob

import cv2
import numpy as np
import pyrender
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

    # Convert focal length to vertical FOV for PerspectiveCamera
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

    # Fill light
    fill_pose = cam_pose.copy()
    fill_pose[0, 3] -= 1.0
    fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    scene.add(fill_light, pose=fill_pose)

    color, _ = renderer.render(scene)
    return color


def reconstruct_vertices(mhr_ts, data):
    """Reconstruct vertices from MHR parameters using TorchScript model."""
    import torch

    shape_params = data.get("shape_params")
    mhr_model_params = data.get("mhr_model_params")
    body_pose_params = data.get("body_pose_params")
    hand_pose_params = data.get("hand_pose_params")
    expr_params = data.get("expr_params")

    if shape_params is None:
        return None

    identity = torch.tensor(shape_params, dtype=torch.float32).unsqueeze(0)

    # Build 204-dim model params
    if mhr_model_params is not None:
        model_params = mhr_model_params
    elif body_pose_params is not None:
        if hand_pose_params is not None:
            model_params = np.concatenate([body_pose_params, hand_pose_params])
        else:
            model_params = body_pose_params
    else:
        return None

    if model_params.shape[0] < 204:
        model_params = np.pad(model_params, (0, 204 - model_params.shape[0]))
    model_params = torch.tensor(model_params[:204], dtype=torch.float32).unsqueeze(0)

    if expr_params is not None:
        expr = torch.tensor(expr_params, dtype=torch.float32).unsqueeze(0)
    else:
        expr = torch.zeros(1, 72)

    with torch.no_grad():
        vertices, _ = mhr_ts(identity, model_params, expr)

    return vertices.squeeze(0).cpu().numpy()


def load_npz(path):
    """Load an NPZ and return a dict of arrays."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def main():
    parser = argparse.ArgumentParser(
        description="Render SAM 3D Body NPZ files to MP4"
    )
    parser.add_argument(
        "--npz_dir", required=True, help="Directory containing per-frame .npz files"
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
        "--image_dir",
        default=None,
        help="Directory of original images (for overlay mode)",
    )
    parser.add_argument(
        "--mode",
        choices=["mesh", "overlay", "side_by_side"],
        default="mesh",
        help="Rendering mode",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--focal_length",
        type=float,
        default=500.0,
        help="Default focal length if not in NPZ",
    )
    parser.add_argument(
        "--person_idx",
        type=int,
        default=0,
        help="Which detected person to render (0-indexed)",
    )
    parser.add_argument(
        "--mhr_model_path",
        default=None,
        help="Path to TorchScript MHR model (.pt) for vertex reconstruction",
    )
    args = parser.parse_args()

    # Load face topology
    faces = np.load(args.faces_path)
    print(f"Loaded faces: {faces.shape}")

    # Load MHR model if provided (for vertex reconstruction fallback)
    mhr_ts = None
    if args.mhr_model_path:
        import torch

        mhr_ts = torch.jit.load(args.mhr_model_path, map_location="cpu")
        print(f"Loaded MHR TorchScript model from {args.mhr_model_path}")

    # Gather NPZ files sorted by name
    npz_files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    if not npz_files:
        print(f"No .npz files found in {args.npz_dir}")
        return
    print(f"Found {len(npz_files)} NPZ files")

    # Gather original images if overlay mode
    bg_images = []
    if args.image_dir and args.mode in ("overlay", "side_by_side"):
        img_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for ext in img_exts:
            bg_images.extend(glob.glob(os.path.join(args.image_dir, ext)))
        bg_images = sorted(bg_images)
        assert len(bg_images) == len(npz_files), (
            f"Image count ({len(bg_images)}) != NPZ count ({len(npz_files)})"
        )

    # Determine output dimensions
    if args.mode == "side_by_side":
        out_w = args.width * 2
    else:
        out_w = args.width
    out_h = args.height

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output_path, fourcc, args.fps, (out_w, out_h))

    # Reuse renderer across frames for speed
    renderer = pyrender.OffscreenRenderer(args.width, args.height)

    skipped = 0
    for i, npz_path in enumerate(tqdm(npz_files, desc="Rendering")):
        data = load_npz(npz_path)

        # Extract per-person data
        vertices = data.get("pred_vertices")
        cam_t = data.get("pred_cam_t", np.array([0, 0, 3.0]))
        focal = float(data.get("focal_length", args.focal_length))

        if vertices is None:
            # Try reconstructing from MHR params
            if mhr_ts is not None:
                vertices = reconstruct_vertices(mhr_ts, data)
            if vertices is None:
                print(f"Warning: no vertices for {npz_path}, skipping")
                skipped += 1
                continue

        # Handle multi-person arrays
        if vertices.ndim == 3:
            vertices = vertices[args.person_idx]
        if cam_t.ndim == 2:
            cam_t = cam_t[args.person_idx]

        # Render mesh
        rendered = render_mesh_frame(
            vertices,
            faces,
            cam_t,
            focal,
            renderer,
            img_width=args.width,
            img_height=args.height,
        )

        if args.mode == "mesh":
            frame_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)

        elif args.mode == "overlay":
            bg = cv2.imread(bg_images[i])
            bg = cv2.resize(bg, (args.width, args.height))
            mask = np.any(rendered > 10, axis=-1).astype(float)[..., np.newaxis]
            alpha = 0.6
            bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB).astype(float)
            blended = bg_rgb * (1 - mask * alpha) + rendered.astype(float) * mask * alpha
            frame_bgr = cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_RGB2BGR)

        elif args.mode == "side_by_side":
            bg = cv2.imread(bg_images[i])
            bg = cv2.resize(bg, (args.width, args.height))
            rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            frame_bgr = np.hstack([bg, rendered_bgr])

        writer.write(frame_bgr)

    renderer.delete()
    writer.release()

    rendered_count = len(npz_files) - skipped
    print(f"Rendered {rendered_count} frames ({skipped} skipped)")
    print(f"Video saved to: {args.output_path}")

    if rendered_count > 0:
        # Re-encode with ffmpeg for better compatibility
        h264_path = args.output_path.replace(".mp4", "_h264.mp4")
        ret = os.system(
            f"ffmpeg -y -i {args.output_path} -c:v libx264 -pix_fmt yuv420p {h264_path} -loglevel warning"
        )
        if ret == 0:
            print(f"H264 version saved to: {h264_path}")


if __name__ == "__main__":
    main()
