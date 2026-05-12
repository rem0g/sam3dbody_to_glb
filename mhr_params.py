#!/usr/bin/env python3
"""
MHR parameter assembly — pure NumPy, no pymomentum / torch dependency.

The `.sam3dbody` file stores *raw* SAM 3D Body estimator outputs (body pose,
hand-pose PCA coefficients, body-scale PCA coefficients, global rotation). To
pose the MHR character you must replicate `MHRHead.mhr_forward()`:

    model_params (204) = [ global_trans*10 (3)   -> always zeros (not stored)
                           global_rot (3)
                           body_pose[:130] (130) with hand Euler angles
                                                   inserted at the hand joints ]   = 136
                       + scales (68) = scale_mean + scale_params @ scale_comps

A naive `concat(body_pose, hand_pose)[:204]` is wrong on ~203/204 entries.

Both `export_glb_pymomentum.py` (GLB via pymomentum) and `render_sam3dbody.py`
(MP4 via the TorchScript MHR model) import `build_model_params` from here so the
posing is identical between the two outputs.
"""

import os

import numpy as np


# ── Hand parameter decoding (NumPy port of sam_3d_body's mhr_utils.py) ──

# 16 hand joints with DOFs: [3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1]
# Total = 27 DOFs; continuous representation = 54 (each DOF doubled).
_HAND_DOFS = [3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1]

_MASK_CONT_3DOF = np.array(
    [b for k in _HAND_DOFS for b in [k == 3] * (2 * k)], dtype=bool
)
_MASK_CONT_1DOF = np.array(
    [b for k in _HAND_DOFS for b in [k in (1, 2)] * (2 * k)], dtype=bool
)
_MASK_MP_3DOF = np.array(
    [b for k in _HAND_DOFS for b in [k == 3] * k], dtype=bool
)
_MASK_MP_1DOF = np.array(
    [b for k in _HAND_DOFS for b in [k in (1, 2)] * k], dtype=bool
)

# Model-parameter indices that drive leg joint rotations (and ankle/foot
# flexibility) — identified via parameter_transform analysis.
LEG_ROTATION_INDICES = [
    50, 51, 52, 53, 54, 55, 56, 57, 58,   # right leg rotations
    59, 60, 61, 62, 63, 64, 65, 66, 67,   # left leg rotations
    122, 123, 124, 125,                    # left foot/ankle flexibility
    126, 127, 128, 129,                    # right foot/ankle flexibility
]
# Backwards-compatible alias
_LEG_ROTATION_INDICES = LEG_ROTATION_INDICES


def _xyz_from_6d(poses):
    """Convert 6D continuous rotation to XYZ Euler angles. (N, 6) -> (N, 3)"""
    x_raw = poses[:, :3]
    y_raw = poses[:, 3:]

    x = x_raw / np.linalg.norm(x_raw, axis=-1, keepdims=True)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x)

    m00 = x[:, 0]; m10 = x[:, 1]; m20 = x[:, 2]
    m01 = y[:, 0]; m11 = y[:, 1]; m21 = y[:, 2]
    m02 = z[:, 0]; m12 = z[:, 1]; m22 = z[:, 2]

    sy = np.sqrt(m00 ** 2 + m10 ** 2)
    singular = sy < 1e-6

    rx = np.where(singular, np.arctan2(-m12, m11), np.arctan2(m21, m22))
    ry = np.arctan2(-m20, sy)
    rz = np.where(singular, np.zeros_like(m00), np.arctan2(m10, m00))

    return np.stack([rx, ry, rz], axis=-1)


def _cont_to_hand_euler(hand_cont):
    """Convert a 54-dim continuous hand representation to 27-dim Euler params."""
    assert hand_cont.shape[-1] == 54

    cont_3dof = hand_cont[_MASK_CONT_3DOF].reshape(-1, 6)
    euler_3dof = _xyz_from_6d(cont_3dof).flatten()

    cont_1dof = hand_cont[_MASK_CONT_1DOF].reshape(-1, 2)
    euler_1dof = np.arctan2(cont_1dof[:, 0], cont_1dof[:, 1])

    result = np.zeros(27, dtype=hand_cont.dtype)
    result[_MASK_MP_3DOF] = euler_3dof
    result[_MASK_MP_1DOF] = euler_1dof
    return result


# ── Head buffers + model-parameter assembly ──

def load_head_buffers(assets_dir):
    """Load MHRHead buffers (scale PCA, hand PCA, hand joint indices) from
    `<assets_dir>/mhr_head_buffers.npz` (extract it with
    tools/extract_mhr_head_buffers.py)."""
    buf_path = os.path.join(assets_dir, "mhr_head_buffers.npz")
    if not os.path.exists(buf_path):
        raise FileNotFoundError(
            f"MHRHead buffers not found: {buf_path}\n"
            "Create it with:  python tools/extract_mhr_head_buffers.py "
            "--ckpt /path/to/model.ckpt --out assets/mhr_head_buffers.npz"
        )
    return dict(np.load(buf_path))


def build_model_params(data, head_buffers, every=1):
    """Assemble (N, 204) MHR model parameters from `.sam3dbody` fields.

    `data` is a dict-like with at least `body_pose_params` (N,133),
    `hand_pose_params` (N,108) and `global_rot` (N,3); `scale_params` (N,28) is
    used if present, otherwise the mean body scale is used.

    Returns `(model_params (M, 204) float32, frame_indices)` where M = N // every.
    """
    body_pose = np.asarray(data["body_pose_params"])   # (N, 133) euler model params
    hand_pose = np.asarray(data["hand_pose_params"])   # (N, 108) PCA coefficients
    global_rot = np.asarray(data["global_rot"])        # (N, 3) euler angles

    num_frames = body_pose.shape[0]

    if "scale_params" in data:
        scale_params = np.asarray(data["scale_params"])   # (N, 28) PCA coefficients
    else:
        scale_params = np.zeros((num_frames, 28), dtype=np.float64)

    scale_mean = head_buffers["scale_mean"]               # (68,)
    scale_comps = head_buffers["scale_comps"]             # (28, 68)
    hand_pose_mean = head_buffers["hand_pose_mean"]       # (54,)
    hand_idxs_l = head_buffers["hand_joint_idxs_left"]    # (27,)
    hand_idxs_r = head_buffers["hand_joint_idxs_right"]   # (27,)

    frame_indices = list(range(0, num_frames, every))
    model_params_list = []

    for i in frame_indices:
        # 1. full_pose_params: [global_trans*10 (zeros), global_rot, body_pose[:130]]
        full_pose = np.zeros(136, dtype=np.float64)
        full_pose[3:6] = global_rot[i]
        full_pose[6:136] = body_pose[i, :130]

        # 2. Decode hand PCA -> continuous -> euler, insert at hand joint indices
        left_hand_cont = hand_pose_mean + hand_pose[i, :54]
        right_hand_cont = hand_pose_mean + hand_pose[i, 54:]
        full_pose[hand_idxs_l] = _cont_to_hand_euler(left_hand_cont)
        full_pose[hand_idxs_r] = _cont_to_hand_euler(right_hand_cont)

        # 3. Per-joint scales via PCA
        scales = scale_mean + scale_params[i] @ scale_comps   # (68,)

        # 4. model_params (204)
        model_params_list.append(np.concatenate([full_pose, scales]))

    return np.stack(model_params_list).astype(np.float32), frame_indices


# ── Optional post-processing ──

def freeze_legs(model_params):
    """Zero leg rotation parameters (legs held in rest pose)."""
    model_params = model_params.copy()
    model_params[:, LEG_ROTATION_INDICES] = 0.0
    return model_params


def freeze_root(model_params):
    """Zero global body rotation (indices 3:6) — keeps the body upright."""
    model_params = model_params.copy()
    model_params[:, 3:6] = 0.0
    return model_params


def smooth_params(model_params, sigma):
    """Gaussian temporal smoothing of model parameters (sigma in frames)."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(model_params, sigma=sigma, axis=0).astype(np.float32)
