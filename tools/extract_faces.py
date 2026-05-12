#!/usr/bin/env python3
"""
Extract the MHR mesh face topology (`mhr_faces_lod1.npy`) from the TorchScript
MHR model that ships with the SAM 3D Body checkpoint.

A pre-extracted copy is already included in this repo as `mhr_faces_lod1.npy`,
so you usually do not need to run this. It is here for reproducibility / if you
use a different MHR build.

Usage:
    python tools/extract_faces.py \
        --mhr-model /path/to/sam-3d-body-dinov3/assets/mhr_model.pt \
        --out mhr_faces_lod1.npy
"""
import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mhr-model", required=True, help="Path to mhr_model.pt (TorchScript)")
    ap.add_argument("--out", default="mhr_faces_lod1.npy", help="Output .npy path")
    args = ap.parse_args()

    mhr = torch.jit.load(args.mhr_model, map_location="cpu")
    faces = mhr.character_torch.mesh.faces.cpu().numpy()
    np.save(args.out, faces)
    print(f"Saved faces: {faces.shape} -> {args.out}")


if __name__ == "__main__":
    main()
