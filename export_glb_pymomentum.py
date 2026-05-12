#!/usr/bin/env python3
"""
Export .sam3dbody file to animated GLB using pymomentum's native GltfBuilder.

Uses pymomentum's GltfBuilder which handles all glTF conventions (matrix storage,
skinning, joint hierarchy, inverse bind matrices) correctly in C++.

The parameter assembly replicates the MHRHead.mhr_forward() pipeline:
  1. full_pose_params = [global_trans*10, global_rot, body_pose_params[:130]]  (136)
  2. Decode hand PCA coefficients -> euler angles -> insert at hand joint indices
  3. scales = scale_mean + scale_params @ scale_comps  (68)
  4. model_params = [full_pose_params, scales]  (204)

When --face-expr is provided, the parameter vector is extended to 321:
  5. identity_coeffs (45) — zeros by default (neutral identity)
  6. expr_params (72) — face expression blendshape coefficients from MediaPipe
  The GltfBuilder's add_motion() path handles both skeletal animation AND
  morph target weights, producing face-animated GLB output.

Usage:
    python export_glb_pymomentum.py input.sam3dbody -o output.glb
    python export_glb_pymomentum.py input.sam3dbody -o output.glb --lod 2
    python export_glb_pymomentum.py input.sam3dbody -o output.glb --freeze-legs
    python export_glb_pymomentum.py input.sam3dbody -o output.glb --smooth 3
    python export_glb_pymomentum.py input.sam3dbody --face-expr face.npz -o output.glb
"""

import argparse
import json
import os
import struct

import numpy as np
import pymomentum.geometry as pym_geometry
from pymomentum import skel_state_np


# ── MHR parameter assembly ──
# The hand-PCA / scale-PCA / hand-Euler assembly lives in mhr_params.py so that
# render_sam3dbody.py poses the mesh exactly the same way (without importing
# pymomentum).
from mhr_params import (
    load_head_buffers,
    build_model_params,
    freeze_legs,
    freeze_root,
    smooth_params,
)


def build_anim_only_glb(char, global_ss, fps, output_path):
    """Build a minimal GLB with only skeleton nodes + animation (no mesh)."""
    num_frames = global_ss.shape[0]
    num_joints = global_ss.shape[1]
    names = list(char.skeleton.joint_names)
    parents = list(char.skeleton.joint_parents)

    # Compute local skeleton states from global
    # local[j] = inverse(global[parent[j]]) * global[j]
    local_ss = np.zeros_like(global_ss)
    for f in range(num_frames):
        for j in range(num_joints):
            p = parents[j]
            if p < 0:
                local_ss[f, j] = global_ss[f, j]
            else:
                parent_inv = skel_state_np.inverse(global_ss[f:f+1, p])
                local_ss[f, j] = skel_state_np.multiply(parent_inv, global_ss[f:f+1, j])[0]

    # Rest pose (frame 0 is fine, but use zero params for true rest)
    rest_mp = np.zeros((1, 204), dtype=np.float32)
    rest_global = np.asarray(
        pym_geometry.model_parameters_to_skeleton_state(char, rest_mp),
        dtype=np.float32
    )[0]
    rest_local = np.zeros((num_joints, 8), dtype=np.float32)
    for j in range(num_joints):
        p = parents[j]
        if p < 0:
            rest_local[j] = rest_global[j]
        else:
            parent_inv = skel_state_np.inverse(rest_global[np.newaxis, p])
            rest_local[j] = skel_state_np.multiply(parent_inv, rest_global[np.newaxis, j])[0]

    # Convert cm to meters
    local_t = local_ss[:, :, :3] / 100.0
    local_q = local_ss[:, :, 3:7]
    local_s = local_ss[:, :, 7:8]
    rest_t = rest_local[:, :3] / 100.0
    rest_q = rest_local[:, 3:7]
    rest_s = rest_local[:, 7]

    # ── Build binary buffer ──
    buf = bytearray()
    buffer_views = []
    accessors = []

    def align4(b):
        while len(b) % 4 != 0:
            b.extend(b'\x00')

    def add_accessor(data_arr, comp_type, acc_type, calc_minmax=False):
        bv_idx = len(buffer_views)
        offset = len(buf)
        raw = data_arr.astype(np.float32).tobytes()
        buf.extend(raw)
        align4(buf)
        length = len(raw)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": length})
        acc = {
            "bufferView": bv_idx, "byteOffset": 0,
            "componentType": comp_type, "count": data_arr.shape[0], "type": acc_type,
        }
        if calc_minmax and acc_type == "SCALAR":
            acc["min"] = [float(data_arr.min())]
            acc["max"] = [float(data_arr.max())]
        acc_idx = len(accessors)
        accessors.append(acc)
        return acc_idx

    # Timestamps (shared across all channels)
    timestamps = np.arange(num_frames, dtype=np.float32) / fps
    time_acc = add_accessor(timestamps, 5126, "SCALAR", calc_minmax=True)

    # Per-joint animation channels
    anim_samplers = []
    anim_channels = []
    for j in range(num_joints):
        node_idx = j + 1  # offset by 1 for unnamed root node

        # Translation
        t_acc = add_accessor(local_t[:, j, :], 5126, "VEC3")
        s_idx = len(anim_samplers)
        anim_samplers.append({"input": time_acc, "output": t_acc, "interpolation": "LINEAR"})
        anim_channels.append({"sampler": s_idx, "target": {"node": node_idx, "path": "translation"}})

        # Rotation
        r_acc = add_accessor(local_q[:, j, :], 5126, "VEC4")
        s_idx = len(anim_samplers)
        anim_samplers.append({"input": time_acc, "output": r_acc, "interpolation": "LINEAR"})
        anim_channels.append({"sampler": s_idx, "target": {"node": node_idx, "path": "rotation"}})

        # Scale (only if non-unit)
        if np.any(np.abs(local_s[:, j, 0] - 1.0) > 1e-4):
            sc = np.broadcast_to(local_s[:, j], (num_frames, 3)).copy()
            sc_acc = add_accessor(sc, 5126, "VEC3")
            s_idx = len(anim_samplers)
            anim_samplers.append({"input": time_acc, "output": sc_acc, "interpolation": "LINEAR"})
            anim_channels.append({"sampler": s_idx, "target": {"node": node_idx, "path": "scale"}})

    # ── Build nodes: unnamed root + 127 joints ──
    nodes = [{"name": "root_container", "children": [1]}]  # node 0
    for j in range(num_joints):
        node = {"name": names[j]}
        node["translation"] = rest_t[j].tolist()
        node["rotation"] = rest_q[j].tolist()
        if abs(rest_s[j] - 1.0) > 1e-6:
            node["scale"] = [float(rest_s[j])] * 3
        children = [c + 1 for c in range(num_joints) if parents[c] == j]
        if children:
            node["children"] = children
        nodes.append(node)

    gltf = {
        "asset": {"version": "2.0", "generator": "sam3dbody-anim-export"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "animations": [{
            "name": "body_animation",
            "samplers": anim_samplers,
            "channels": anim_channels,
        }],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(buf)}],
    }

    # ── Write GLB ──
    json_bytes = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    while len(json_bytes) % 4 != 0:
        json_bytes += b' '
    json_chunk = struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk = struct.pack('<II', len(buf), 0x004E4942) + bytes(buf)
    total_length = 12 + len(json_chunk) + len(bin_chunk)
    glb_header = struct.pack('<III', 0x46546C67, 2, total_length)

    with open(output_path, 'wb') as f:
        f.write(glb_header)
        f.write(json_chunk)
        f.write(bin_chunk)


def load_character(assets_dir, lod, load_blendshapes=False):
    """Load pymomentum character from FBX + model files.

    When load_blendshapes=True, loads the 117 blendshapes (45 identity + 72
    expression) and extends the parameter transform to include blendshape
    coefficients (204 -> 321 parameters).
    """
    fbx_path = os.path.join(assets_dir, f"lod{lod}.fbx")
    model_path = os.path.join(assets_dir, "compact_v6_1.model")
    if not os.path.exists(fbx_path):
        raise FileNotFoundError(f"FBX not found: {fbx_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    char = pym_geometry.Character.load_fbx(
        fbx_path, model_path, load_blendshapes=load_blendshapes
    )

    if load_blendshapes:
        # Extend parameter transform to include blendshape coefficients.
        # This adds 117 parameters (45 identity + 72 expression) to the
        # parameter transform, making it 204 + 117 = 321 parameters total.
        char = char.with_blend_shape(char.blend_shape)

    return char


def main():
    parser = argparse.ArgumentParser(
        description="Export .sam3dbody to animated GLB using pymomentum"
    )
    parser.add_argument("input", help="Path to .sam3dbody file")
    parser.add_argument("-o", "--output", default=None, help="Output .glb path")
    parser.add_argument(
        "--assets", default=None,
        help="Path to MHR assets directory (containing lodN.fbx and compact_v6_1.model)"
    )
    parser.add_argument(
        "--lod", type=int, default=0, choices=range(7),
        help="LOD level: 0=highest detail (73K verts), 6=lowest (default: 0)",
        metavar="LOD",
    )
    parser.add_argument("--every", type=int, default=1, help="Use every Nth frame")
    parser.add_argument(
        "--fps", type=float, default=0,
        help="Override FPS (0 = from metadata)"
    )
    parser.add_argument(
        "--freeze-legs", action="store_true",
        help="Freeze leg joints in rest pose (useful for sign language)"
    )
    parser.add_argument(
        "--freeze-root", action="store_true",
        help="Freeze global body rotation (removes forward/backward lean)"
    )
    parser.add_argument(
        "--smooth", type=float, default=0, metavar="SIGMA",
        help="Gaussian smoothing sigma in frames (e.g. 2.0 for light, 5.0 for heavy)"
    )
    parser.add_argument(
        "--anim-only", action="store_true",
        help="Export animation-only GLB (no mesh, for use with a pre-loaded mesh in Babylon.js)"
    )
    parser.add_argument(
        "--face-expr", default=None, metavar="PATH",
        help="Path to .face_expr.npz (from extract_face_blendshapes.py) or "
             ".sam3dbody file containing expr_params. Enables morph target animation."
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + "_skeletal.glb"

    # Find assets directory
    assets_dir = args.assets
    if assets_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for candidate in [
            os.path.join(script_dir, "assets"),
            os.path.join(script_dir, "MHR", "assets"),
            os.path.join(
                script_dir, "sam-3d-body", "checkpoints",
                "sam-3d-body-dinov3", "assets"
            ),
        ]:
            if os.path.isdir(candidate):
                assets_dir = candidate
                break
    if assets_dir is None:
        print("Error: MHR assets directory not found. Use --assets to specify path.")
        return

    # Determine if we need face expression support
    # (anim-only doesn't include mesh so morph targets aren't applicable)
    has_face = args.face_expr is not None and not args.anim_only
    if args.face_expr and args.anim_only:
        print("Warning: --anim-only does not support morph targets; "
              "ignoring --face-expr.")

    print(f"Loading character (LOD {args.lod}, blendshapes={'yes' if has_face else 'no'})...")
    char = load_character(assets_dir, args.lod, load_blendshapes=has_face)
    num_joints = len(char.skeleton.joint_names)
    num_verts = char.mesh.vertices.shape[0]
    num_faces = char.mesh.faces.shape[0]
    pt_size = char.parameter_transform.size
    print(
        f"  {num_joints} joints, "
        f"{num_verts:,} vertices, "
        f"{num_faces:,} faces, "
        f"{pt_size} params"
    )

    # Load MHRHead buffers for parameter assembly
    print("Loading MHRHead buffers...")
    head_buffers = load_head_buffers(assets_dir)

    # Load sam3dbody data
    data = np.load(args.input, allow_pickle=True)
    data_dict = {k: data[k] for k in data.files}

    metadata = {}
    if "metadata" in data_dict:
        meta_arr = data_dict["metadata"]
        metadata = meta_arr[0] if meta_arr.shape == (1,) else meta_arr.item()

    # Load face expression data
    expr_params = None
    if has_face:
        print(f"Loading face expressions from: {args.face_expr}")
        face_data = np.load(args.face_expr, allow_pickle=True)
        if "expr_params" in face_data:
            expr_params = face_data["expr_params"].astype(np.float32)
        else:
            raise ValueError(
                f"No 'expr_params' found in {args.face_expr}. "
                "Run extract_face_blendshapes.py first."
            )
        print(f"  Face expressions: {expr_params.shape}")
    elif not args.anim_only:
        # Auto-detect face expressions in the sam3dbody input file
        if "expr_params" in data_dict:
            candidate = data_dict["expr_params"].astype(np.float32)
            if np.any(np.abs(candidate) > 1e-6):
                print("  Found non-zero expr_params in sam3dbody — enabling face animation")
                expr_params = candidate
                has_face = True
                # Reload character with blendshapes
                char = load_character(assets_dir, args.lod, load_blendshapes=True)
                pt_size = char.parameter_transform.size

    fps = args.fps if args.fps > 0 else metadata.get("fps", 30.0)
    effective_fps = fps / args.every

    # Build body model params (204) with correct assembly
    model_params_all, frame_indices = build_model_params(
        data_dict, head_buffers, every=args.every
    )
    num_frames = len(frame_indices)
    print(f"  {num_frames} frames at {effective_fps:.1f} fps")

    # Post-processing: freeze root rotation
    if args.freeze_root:
        print("Freezing global rotation...")
        model_params_all = freeze_root(model_params_all)

    # Post-processing: freeze legs
    if args.freeze_legs:
        print("Freezing leg joints...")
        model_params_all = freeze_legs(model_params_all)

    # Post-processing: temporal smoothing (body params only)
    if args.smooth > 0:
        print(f"Smoothing (sigma={args.smooth:.1f} frames)...")
        model_params_all = smooth_params(model_params_all, sigma=args.smooth)

    # Build GLB
    if args.anim_only:
        # anim-only path: compute skeleton states manually, no mesh/morph targets
        print("Computing skeleton states...")
        global_ss = pym_geometry.model_parameters_to_skeleton_state(
            char, model_params_all
        )
        global_ss = np.asarray(global_ss, dtype=np.float32)
        print("Building animation-only GLB...")
        build_anim_only_glb(char, global_ss, effective_fps, args.output)
    elif has_face:
        # Face animation path: use add_motion() which handles both skeleton
        # and morph targets. Requires extended parameter vector (321).
        print("Building face+body GLB via add_motion()...")

        # Align face expression frames with body frames
        n_face_frames = expr_params.shape[0]
        n_body_frames = len(frame_indices)
        if frame_indices[-1] >= n_face_frames:
            # Face data has fewer frames than body — pad with last frame
            pad_count = frame_indices[-1] - n_face_frames + 1
            pad = np.tile(expr_params[-1:], (pad_count, 1))
            expr_params = np.concatenate([expr_params, pad], axis=0)
            print(f"  Padded face expressions: {n_face_frames} -> {expr_params.shape[0]} frames")
        face_expr = expr_params[frame_indices]  # subsample to match body frames
        n_expr = face_expr.shape[1]  # 72

        # Build extended parameter vector: [body(204), identity(45), expression(72)]
        n_identity = 45
        identity_coeffs = np.zeros(
            (num_frames, n_identity), dtype=np.float32
        )
        full_params = np.concatenate(
            [model_params_all, identity_coeffs, face_expr], axis=1
        )  # (N, 321)
        assert full_params.shape[1] == pt_size, (
            f"Parameter vector size mismatch: got {full_params.shape[1]}, "
            f"expected {pt_size}"
        )

        param_names = list(char.parameter_transform.names)
        motion = (param_names, full_params)

        builder = pym_geometry.GltfBuilder(fps=effective_fps)
        save_options = pym_geometry.FileSaveOptions(blend_shapes=True)
        builder.add_character(char, options=save_options)
        builder.add_motion(
            char, effective_fps, motion=motion, add_extensions=False
        )
        builder.save(args.output)
    else:
        # Standard body-only path: use add_skeleton_states()
        print("Computing skeleton states...")
        global_ss = pym_geometry.model_parameters_to_skeleton_state(
            char, model_params_all
        )
        global_ss = np.asarray(global_ss, dtype=np.float32)
        print("Building GLB via GltfBuilder...")
        builder = pym_geometry.GltfBuilder(fps=effective_fps)
        builder.add_character(char)
        builder.add_skeleton_states(char, effective_fps, global_ss)
        builder.save(args.output)

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    duration = (num_frames - 1) / effective_fps if num_frames > 1 else 0.0
    print(f"Saved: {args.output} ({file_size:.1f} MB)")
    extras = ""
    if has_face:
        active = int(np.sum(np.abs(face_expr).mean(axis=0) > 0.001))
        extras = f", {active}/72 active morph targets"
    print(f"  {num_frames} keyframes, {duration:.2f}s at {effective_fps:.1f} fps{extras}")


if __name__ == "__main__":
    main()
