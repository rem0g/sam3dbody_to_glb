#!/usr/bin/env python3
"""
Build static-mesh GLBs (rest pose, no animation) for a given LOD.

Useful for web apps (e.g. Babylon.js) that load the mesh once and then stream
small animation-only clips produced by `export_glb_pymomentum.py --anim-only`.
Node names match between the static mesh and the animation clips, so animations
retarget by name.

Usage:
    python tools/build_static_mesh_glb.py --assets assets --lod 0 --lod 1
    # -> assets/lod0.glb, assets/lod1.glb
"""
import argparse
import os

import pymomentum.geometry as pym_geometry


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", default="assets",
                    help="MHR assets dir (with lodN.fbx + compact_v6_1.model)")
    ap.add_argument("--lod", type=int, action="append", default=None,
                    help="LOD level(s) to build; repeat for multiple (default: 0 and 1)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--outdir", default=None,
                    help="Output directory (default: same as --assets)")
    args = ap.parse_args()

    lods = args.lod if args.lod else [0, 1]
    outdir = args.outdir or args.assets
    os.makedirs(outdir, exist_ok=True)

    model_path = os.path.join(args.assets, "compact_v6_1.model")
    for lod in lods:
        fbx_path = os.path.join(args.assets, f"lod{lod}.fbx")
        if not os.path.exists(fbx_path):
            raise SystemExit(f"Missing {fbx_path}")
        char = pym_geometry.Character.load_fbx(
            fbx_path, model_path, load_blendshapes=False
        )
        builder = pym_geometry.GltfBuilder(fps=args.fps)
        builder.add_character(char)
        out = os.path.join(outdir, f"lod{lod}.glb")
        builder.save(out)
        size_mb = os.path.getsize(out) / (1024 * 1024)
        print(f"Wrote {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
