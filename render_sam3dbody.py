#!/usr/bin/env python3
"""
Render a .sam3dbody file (multi-frame NPZ with MHR params) to MP4 video.

Supports two input formats:
  1. Single multi-frame .sam3dbody / .npz file (arrays have shape [N, ...])
  2. Directory of per-frame .npz files (original render_npz_to_mp4.py format)

Camera framing:
  .sam3dbody files from the SAM 3D Body pipeline do *not* store a camera
  translation (`pred_cam_t`), and their `bboxes` are often just the full frame,
  so there is nothing reliable to position the camera from. By default this
  script therefore **auto-fits** the camera to the reconstructed mesh's own 3D
  bounding box (over all frames), which is unit-agnostic (the MHR mesh is in
  centimetres) and keeps the whole body — arms raised and all — in frame.

Usage:
    # Single multi-frame file (auto-fit camera, resolution from the source video):
    python render_sam3dbody.py \
        --input ./R20260213_8551.sam3dbody \
        --faces_path ./mhr_faces_lod1.npy \
        --mhr_model_path ./assets/mhr_model.pt \
        --output_path ./output.mp4

    # Force a square 720p render, a bit more padding around the body:
    python render_sam3dbody.py --input clip.sam3dbody --faces_path mhr_faces_lod1.npy \
        --mhr_model_path assets/mhr_model.pt -o out.mp4 --width 720 --height 720 --margin 1.4

    # Directory of per-frame NPZ files (these usually carry pred_vertices + pred_cam_t):
    python render_sam3dbody.py --input ./output_npz/ --faces_path ./mhr_faces_lod1.npy -o out.mp4
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


# ── Rendering ──────────────────────────────────────────────────────────────

def render_mesh_frame(
    vertices,
    faces,
    cam_pose,
    yfov,
    aspect,
    renderer,
    znear,
    zfar,
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

    camera = pyrender.PerspectiveCamera(
        yfov=yfov, aspectRatio=aspect, znear=znear, zfar=zfar,
    )
    scene.add(camera, pose=cam_pose)

    # Key light from the camera, fill light from the side.
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=cam_pose)
    fill_pose = cam_pose.copy()
    fill_pose[0, 3] -= 1.0
    fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    scene.add(fill_light, pose=fill_pose)

    color, _ = renderer.render(scene)
    return color


# ── MHR vertex reconstruction ──────────────────────────────────────────────

def reconstruct_vertices(mhr_ts, body_pose, hand_pose, shape, expr=None):
    """Reconstruct vertices for a single frame from MHR parameters.

    Note: this uses a naive [body_pose | hand_pose][:204] assembly, which is
    only approximately right (see export_glb_pymomentum.py for the correct
    MHRHead pipeline). It is good enough for a preview render but the GLB
    exporter is the source of truth for posing.
    """
    import torch
    identity = torch.tensor(shape, dtype=torch.float32).unsqueeze(0)
    model_params = np.concatenate([np.asarray(body_pose), np.asarray(hand_pose)])
    if model_params.shape[0] < 204:
        model_params = np.pad(model_params, (0, 204 - model_params.shape[0]))
    model_params = torch.tensor(model_params[:204], dtype=torch.float32).unsqueeze(0)
    if expr is not None and np.asarray(expr).shape and np.asarray(expr).shape[0] > 0:
        expr_t = torch.tensor(np.asarray(expr), dtype=torch.float32).unsqueeze(0)
    else:
        expr_t = torch.zeros(1, 72)
    with torch.no_grad():
        vertices, _ = mhr_ts(identity, model_params, expr_t)
    return vertices.squeeze(0).cpu().numpy()


# ── Camera helpers ─────────────────────────────────────────────────────────

def fit_camera_to_bounds(vmin, vmax, aspect, yfov, margin):
    """Place a camera on the +Z axis through the bounds' centre, far enough
    that the whole box (× margin) fits the frame. Returns (cam_pose, znear, zfar)."""
    center = (vmin + vmax) / 2.0
    half = (vmax - vmin) / 2.0
    tan_half_v = np.tan(yfov / 2.0)
    # distance needed for the vertical extent, and for the horizontal extent
    d = margin * max(half[1], half[0] / max(aspect, 1e-6)) / max(tan_half_v, 1e-6)
    d += half[2]  # push back past the front of the mesh
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = [center[0], center[1], center[2] + d]
    diag = float(np.linalg.norm(vmax - vmin))
    znear = max(d * 0.01, diag * 0.001, 1e-3)
    zfar = d + diag * 4.0 + 1.0
    return cam_pose, znear, zfar


def estimate_cam_t_from_bbox(bbox, focal_length, img_width, img_height):
    """Legacy: estimate a camera translation from a (tight) person bbox.
    Only meaningful if the mesh is in metres and the bbox tightly frames the
    person — rarely true for .sam3dbody files (kept for --fit bbox)."""
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    bbox_h = bbox[3] - bbox[1]
    tz = (focal_length * 1.7) / max(bbox_h, 1.0)
    tx = (cx - img_width / 2.0) * tz / focal_length
    ty = (cy - img_height / 2.0) * tz / focal_length
    pose = np.eye(4)
    pose[:3, 3] = [tx, ty, tz]
    return pose


# ── Input loading ──────────────────────────────────────────────────────────

def load_multiframe_file(path):
    data = np.load(path, allow_pickle=True)
    result = {k: data[k] for k in data.files}
    metadata = {}
    if "metadata" in result:
        meta_arr = result["metadata"]
        if meta_arr.shape == (1,):
            metadata = meta_arr[0]
        elif meta_arr.shape == ():
            metadata = meta_arr.item()
    return result, metadata


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Render SAM 3D Body output to MP4")
    parser.add_argument("--input", required=True,
                        help="Path to .sam3dbody/.npz file or directory of per-frame NPZ files")
    parser.add_argument("--faces_path", required=True,
                        help="Path to saved faces array (.npy) from the MHR model")
    parser.add_argument("--output_path", "-o", default="output.mp4", help="Output MP4 path")
    parser.add_argument("--mhr_model_path", default=None,
                        help="TorchScript MHR model (.pt) for vertex reconstruction "
                             "(required unless the input already stores pred_vertices)")
    parser.add_argument("--fps", type=float, default=0, help="FPS (0 = auto from metadata)")
    parser.add_argument("--width", type=int, default=0, help="Output width (0 = from source video)")
    parser.add_argument("--height", type=int, default=0, help="Output height (0 = from source video)")
    parser.add_argument("--max-size", type=int, default=720,
                        help="When width/height are auto, longest side is scaled to this (default 720)")
    parser.add_argument("--fit", choices=["auto", "mesh", "bbox", "off"], default="auto",
                        help="Camera framing: auto = pred_cam_t if present else fit to mesh; "
                             "mesh = always fit to the mesh bounds; bbox = legacy bbox estimate; "
                             "off = require pred_cam_t in the data (default: auto)")
    parser.add_argument("--margin", type=float, default=1.25,
                        help="Padding factor around the mesh when --fit fits to the mesh (default 1.25)")
    parser.add_argument("--vfov-deg", type=float, default=0,
                        help="Vertical field of view in degrees (0 = derive from focal length, "
                             "fallback 35deg)")
    parser.add_argument("--every", type=int, default=1, help="Render every Nth frame")
    parser.add_argument("--focal_length", type=float, default=0,
                        help="Override focal length (only used by --fit bbox / vfov derivation)")
    parser.add_argument("--person_idx", type=int, default=0, help="Which person to render")
    args = parser.parse_args()

    faces = np.load(args.faces_path)
    print(f"Loaded faces: {faces.shape}")

    mhr_ts = None
    if args.mhr_model_path:
        import torch
        mhr_ts = torch.jit.load(args.mhr_model_path, map_location="cpu")
        print("Loaded MHR TorchScript model")

    # ── Gather frames ──
    is_multiframe = os.path.isfile(args.input)
    src_width, src_height = 0, 0
    frames = []

    if is_multiframe:
        data, metadata = load_multiframe_file(args.input)
        num_frames = int(metadata.get("num_frames", 0))
        src_fps = float(metadata.get("fps", 30.0))
        src_width = int(metadata.get("width", 0))
        src_height = int(metadata.get("height", 0))
        if num_frames == 0:
            for key in ["body_pose_params", "body_keypoints_3d", "pred_vertices"]:
                if key in data:
                    num_frames = data[key].shape[0]
                    break
        fps = args.fps if args.fps > 0 else src_fps
        print(f"Multi-frame file: {num_frames} frames, {fps:.2f} fps, "
              f"src {src_width or '?'}x{src_height or '?'}")
        for i in range(num_frames):
            frame = {}
            if "pred_vertices" in data:
                v = data["pred_vertices"]; frame["vertices"] = v[i] if v.ndim == 3 else v
            if "pred_cam_t" in data:
                ct = data["pred_cam_t"]; frame["cam_t"] = ct[i] if ct.ndim == 2 else ct
            for key in ["body_pose_params", "hand_pose_params", "shape_params",
                        "global_rot", "expr_params"]:
                if key in data:
                    arr = data[key]; frame[key] = arr[i] if arr.ndim >= 2 else arr
            if "focal_lengths" in data:
                frame["focal_length"] = float(data["focal_lengths"][i])
            elif "focal_length" in data:
                fl = data["focal_length"]
                frame["focal_length"] = float(fl[i]) if (np.ndim(fl) >= 1 and np.shape(fl)[0] > 1) else float(fl)
            if "bboxes" in data:
                frame["bbox"] = data["bboxes"][i]
            frames.append(frame)
    else:
        npz_files = sorted(glob.glob(os.path.join(args.input, "*.npz")))
        if not npz_files:
            print(f"No .npz files found in {args.input}"); return
        fps = args.fps if args.fps > 0 else 30.0
        print(f"Found {len(npz_files)} per-frame NPZ files, {fps:.2f} fps")
        for path in npz_files:
            raw = np.load(path, allow_pickle=True)
            frame = {k: raw[k] for k in raw.files}
            if "pred_vertices" in frame: frame["vertices"] = frame["pred_vertices"]
            if "pred_cam_t" in frame: frame["cam_t"] = frame["pred_cam_t"]
            frames.append(frame)

    if args.every > 1:
        frames = frames[:: args.every]
    if not frames:
        print("No frames to render"); return

    # ── Reconstruct all vertices (so we can fit the camera to the real bounds) ──
    print("Reconstructing meshes...")
    all_vertices = []
    vmin = np.array([np.inf, np.inf, np.inf])
    vmax = np.array([-np.inf, -np.inf, -np.inf])
    for frame in tqdm(frames, desc="Meshing"):
        v = frame.get("vertices")
        if v is None and mhr_ts is not None:
            bp, sh = frame.get("body_pose_params"), frame.get("shape_params")
            if bp is not None and sh is not None:
                hp = frame.get("hand_pose_params")
                if hp is None: hp = np.zeros(108)
                v = reconstruct_vertices(mhr_ts, bp, hp, sh, frame.get("expr_params"))
        if v is None:
            all_vertices.append(None); continue
        v = np.asarray(v)
        if v.ndim == 3:
            v = v[args.person_idx]
        v = v.astype(np.float32)
        all_vertices.append(v)
        vmin = np.minimum(vmin, v.min(axis=0))
        vmax = np.maximum(vmax, v.max(axis=0))

    n_valid = sum(x is not None for x in all_vertices)
    if n_valid == 0:
        print("No meshes could be reconstructed — provide --mhr_model_path or pred_vertices."); return

    # ── Resolve output resolution ──
    if args.width > 0 and args.height > 0:
        out_w, out_h = args.width, args.height
    elif src_width > 0 and src_height > 0:
        scale = args.max_size / float(max(src_width, src_height))
        scale = min(scale, 1.0)
        out_w = max(2, int(round(src_width * scale)) // 2 * 2)
        out_h = max(2, int(round(src_height * scale)) // 2 * 2)
    else:
        out_w = out_h = min(args.max_size, 512)
    aspect = out_w / out_h
    print(f"Output: {out_w}x{out_h}")

    # ── Resolve vertical FOV ──
    if args.vfov_deg > 0:
        yfov = np.radians(args.vfov_deg)
    else:
        focal = args.focal_length or frames[0].get("focal_length", 0)
        ref_h = src_height or 1442
        yfov = 2.0 * np.arctan(ref_h / (2.0 * focal)) if focal > 0 else np.radians(35.0)
    yfov = float(np.clip(yfov, np.radians(10.0), np.radians(90.0)))

    # ── Decide camera mode ──
    have_cam_t = any(("cam_t" in f) for f in frames)
    use_fixed_fit = (args.fit == "mesh") or (args.fit == "auto" and not have_cam_t)
    if args.fit == "off" and not have_cam_t:
        print("--fit off but the data has no pred_cam_t."); return
    if args.fit == "bbox":
        use_fixed_fit = False

    fixed_cam_pose = znear = zfar = None
    if use_fixed_fit:
        fixed_cam_pose, znear, zfar = fit_camera_to_bounds(vmin, vmax, aspect, yfov, args.margin)
        print(f"Camera: auto-fit to mesh bounds (center={((vmin+vmax)/2).round(1)}, "
              f"size={(vmax-vmin).round(1)}, dist={fixed_cam_pose[2,3]-((vmin[2]+vmax[2])/2):.1f}, "
              f"vfov={np.degrees(yfov):.1f}deg)")
    elif args.fit == "bbox":
        print("Camera: legacy bbox estimate")
    else:
        print("Camera: using pred_cam_t from data")
        # znear/zfar in metres-ish; widen to be safe
        znear, zfar = 0.01, 1000.0

    # ── Render ──
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output_path, fourcc, fps, (out_w, out_h))
    renderer = pyrender.OffscreenRenderer(out_w, out_h)

    skipped = 0
    for frame, vertices in tqdm(list(zip(frames, all_vertices)), desc="Rendering"):
        if vertices is None:
            skipped += 1; continue

        if use_fixed_fit:
            cam_pose, zn, zf = fixed_cam_pose, znear, zfar
        elif args.fit == "bbox":
            bbox = frame.get("bbox")
            focal = args.focal_length or frame.get("focal_length", 1000.0)
            if bbox is not None:
                cam_pose = estimate_cam_t_from_bbox(bbox, focal, src_width or 512, src_height or 512)
            else:
                cam_pose, _, _ = fit_camera_to_bounds(vmin, vmax, aspect, yfov, args.margin)
            zn, zf = 1e-3, float(np.linalg.norm(vmax - vmin)) * 10 + 1000
        else:  # use cam_t from data
            ct = np.asarray(frame["cam_t"])
            if ct.ndim == 2:
                ct = ct[args.person_idx]
            cam_pose = np.eye(4)
            cam_pose[:3, 3] = ct
            zn, zf = znear, zfar

        rendered = render_mesh_frame(
            vertices, faces, cam_pose, yfov, aspect, renderer,
            znear=zn if not use_fixed_fit else znear,
            zfar=zf if not use_fixed_fit else zfar,
        )
        writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))

    renderer.delete()
    writer.release()

    rendered_count = len(frames) - skipped
    print(f"Rendered {rendered_count} frames ({skipped} skipped)")
    print(f"Video saved to: {args.output_path}")

    if rendered_count > 0:
        h264_path = args.output_path[:-4] + "_h264.mp4" if args.output_path.endswith(".mp4") else args.output_path + "_h264.mp4"
        ret = os.system(
            f'ffmpeg -y -i "{args.output_path}" -c:v libx264 -pix_fmt yuv420p "{h264_path}" -loglevel warning'
        )
        if ret == 0:
            print(f"H264 version saved to: {h264_path}")


if __name__ == "__main__":
    main()
