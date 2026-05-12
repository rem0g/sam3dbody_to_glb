#!/usr/bin/env python3
"""
Extract `mhr_head_buffers.npz` from the SAM 3D Body checkpoint.

`export_glb_pymomentum.py` needs this file (PCA weights for hand pose / body
scale plus the hand joint indices) to assemble MHR model parameters correctly.

Usage:
    python tools/extract_mhr_head_buffers.py \
        --ckpt /path/to/sam-3d-body-dinov3/model.ckpt \
        --out  assets/mhr_head_buffers.npz
"""
import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Path to SAM 3D Body model.ckpt")
    ap.add_argument("--out", default="assets/mhr_head_buffers.npz",
                    help="Output .npz path (default: assets/mhr_head_buffers.npz)")
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # Some checkpoints wrap the state dict under "state_dict"
    if "state_dict" in sd and "head_pose.scale_mean" not in sd:
        sd = sd["state_dict"]

    keys = [
        "head_pose.scale_mean",
        "head_pose.scale_comps",
        "head_pose.hand_pose_mean",
        "head_pose.hand_pose_comps",
        "head_pose.hand_joint_idxs_left",
        "head_pose.hand_joint_idxs_right",
    ]
    missing = [k for k in keys if k not in sd]
    if missing:
        raise SystemExit(f"Checkpoint is missing expected keys: {missing}")

    np.savez(
        args.out,
        scale_mean=sd["head_pose.scale_mean"].cpu().numpy(),
        scale_comps=sd["head_pose.scale_comps"].cpu().numpy(),
        hand_pose_mean=sd["head_pose.hand_pose_mean"].cpu().numpy(),
        hand_pose_comps=sd["head_pose.hand_pose_comps"].cpu().numpy(),
        hand_joint_idxs_left=sd["head_pose.hand_joint_idxs_left"].cpu().numpy(),
        hand_joint_idxs_right=sd["head_pose.hand_joint_idxs_right"].cpu().numpy(),
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
