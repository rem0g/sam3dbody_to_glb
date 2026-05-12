#!/usr/bin/env python3
"""
Batch-convert every .sam3dbody file in ./input/ to an animated GLB in ./output/.

This is the user-friendly entry point: drop your .sam3dbody files into the
`input/` folder, run this script, and collect the .glb files from `output/`.

    python batch_convert.py                       # convert everything in ./input/
    python batch_convert.py --anim-only           # small animation-only GLBs
    python batch_convert.py --lod 1 --smooth 2    # pass options through
    python batch_convert.py --input mydir --output outdir

Any unrecognised flags are forwarded to export_glb_pymomentum.py, so anything
that script accepts (--lod, --every, --fps, --freeze-legs, --freeze-root,
--smooth, --anim-only, --assets, --face-expr ...) works here too.
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT_SCRIPT = os.path.join(HERE, "export_glb_pymomentum.py")


def main():
    parser = argparse.ArgumentParser(
        description="Convert every .sam3dbody in a folder to GLB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", default=os.path.join(HERE, "input"),
                        help="Folder containing .sam3dbody files (default: ./input)")
    parser.add_argument("--output", default=os.path.join(HERE, "output"),
                        help="Folder to write .glb files to (default: ./output)")
    parser.add_argument("--anim-only", action="store_true",
                        help="Export small animation-only GLBs (suffix _anim.glb)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-convert files even if the .glb already exists")
    args, passthrough = parser.parse_known_args()

    if not os.path.isdir(args.input):
        sys.exit(f"Input folder not found: {args.input}")
    os.makedirs(args.output, exist_ok=True)

    sam_files = sorted(
        f for f in os.listdir(args.input) if f.endswith(".sam3dbody")
    )
    if not sam_files:
        sys.exit(f"No .sam3dbody files found in {args.input}")

    suffix = "_anim.glb" if args.anim_only else ".glb"
    if args.anim_only:
        passthrough = passthrough + ["--anim-only"]

    print(f"Found {len(sam_files)} file(s) in {args.input}")
    ok, failed, skipped = 0, 0, 0
    for name in sam_files:
        src = os.path.join(args.input, name)
        base = name[: -len(".sam3dbody")]
        dst = os.path.join(args.output, base + suffix)
        if os.path.exists(dst) and not args.overwrite:
            print(f"  SKIP  {name} -> {os.path.basename(dst)} (already exists)")
            skipped += 1
            continue
        print(f"  ...   {name} -> {os.path.basename(dst)}")
        cmd = [sys.executable, EXPORT_SCRIPT, src, "-o", dst] + passthrough
        result = subprocess.run(cmd)
        if result.returncode == 0 and os.path.exists(dst):
            size_mb = os.path.getsize(dst) / (1024 * 1024)
            print(f"  OK    {os.path.basename(dst)}  ({size_mb:.1f} MB)")
            ok += 1
        else:
            print(f"  FAIL  {name}")
            failed += 1

    print(f"\nDone. {ok} converted, {skipped} skipped, {failed} failed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
