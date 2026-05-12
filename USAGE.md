# USAGE — script reference

A command cheat-sheet for every script in this repo. For background, the
`.sam3dbody` format and troubleshooting, see the [README](README.md).

## Conventions

- Activate the environment first: `conda activate sam3d` (Python 3.12).
- On Linux, before running anything that imports `pymomentum`:
  ```bash
  export LD_LIBRARY_PATH="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):$LD_LIBRARY_PATH"
  ```
- The exporter and renderers auto-detect the MHR model files in `./assets/`
  (download them once — see [`assets/README.md`](assets/README.md)).

---

## `batch_convert.py` — folder in → folder out  *(start here)*

Converts every `.sam3dbody` in a folder to a GLB in another folder. Any flag it
doesn't recognise is forwarded to `export_glb_pymomentum.py`.

```bash
python batch_convert.py                          # ./input/*.sam3dbody  ->  ./output/*.glb
python batch_convert.py --anim-only              # tiny animation-only GLBs (_anim.glb)
python batch_convert.py --lod 1 --smooth 2       # forwarded flags
python batch_convert.py --freeze-legs --freeze-root --smooth 2   # signing / talking-head preset
python batch_convert.py --input some/dir --output other/dir
python batch_convert.py --overwrite              # re-convert even if the .glb exists
```

| flag | default | meaning |
|------|---------|---------|
| `--input DIR` | `./input` | folder of `.sam3dbody` files |
| `--output DIR` | `./output` | folder to write `.glb` files |
| `--anim-only` | off | animation-only GLBs, named `<name>_anim.glb` |
| `--overwrite` | off | re-convert files whose output already exists |
| *(anything else)* | — | passed through to `export_glb_pymomentum.py` |

Exit code is non-zero if any file failed.

---

## `export_glb_pymomentum.py` — one `.sam3dbody` → animated GLB

The core exporter. Correctly assembles MHR parameters (hand-pose PCA → Euler,
per-joint scales), skins the chosen LOD mesh, and writes a GLB via pymomentum's
`GltfBuilder`.

```bash
# basic
python export_glb_pymomentum.py input.sam3dbody -o output.glb

# signing / talking-head: lock legs + global rotation, lightly smooth
python export_glb_pymomentum.py input.sam3dbody -o output.glb --freeze-legs --freeze-root --smooth 2

# lighter mesh (LOD 1 ≈ 18K verts)
python export_glb_pymomentum.py input.sam3dbody -o output.glb --lod 1

# animation only, no mesh (~0.8 MB) — pair with a static mesh GLB in a web app
python export_glb_pymomentum.py input.sam3dbody -o clip_anim.glb --anim-only --freeze-legs --freeze-root --smooth 2

# decimate frames, override fps
python export_glb_pymomentum.py input.sam3dbody -o output.glb --every 2 --fps 30

# point at assets explicitly
python export_glb_pymomentum.py input.sam3dbody -o output.glb --assets /path/to/MHR/assets

# add face/expression morph-target animation from an NPZ
python export_glb_pymomentum.py input.sam3dbody --face-expr face_expr.npz -o output.glb
```

| flag | default | meaning |
|------|---------|---------|
| `input` (positional) | — | `.sam3dbody` file |
| `-o, --output` | `<input>_skeletal.glb` | output path |
| `--assets DIR` | auto (`./assets`, then `./MHR/assets`, …) | MHR assets directory |
| `--lod N` | `0` | mesh detail: `0` = 73K verts … `6` = lowest |
| `--every N` | `1` | keep every Nth frame |
| `--fps F` | from metadata | override animation fps |
| `--freeze-legs` | off | hold leg joints in rest pose |
| `--freeze-root` | off | hold global body rotation fixed |
| `--smooth SIGMA` | `0` | gaussian temporal smoothing (frames) |
| `--anim-only` | off | export animation only, no mesh |
| `--face-expr PATH` | none | add face/expression morph-target animation from an NPZ |

Requires `assets/mhr_head_buffers.npz` — create it with `tools/extract_mhr_head_buffers.py`
if missing.

---

## `render_skeleton.py` — `.sam3dbody` → 2D skeleton MP4

Fastest visual check. Draws the colored 2D skeleton (body + hands) from
`body_keypoints_2d` on a black background. No GPU, no model files needed.

```bash
python render_skeleton.py input.sam3dbody                       # -> input_skeleton.mp4
python render_skeleton.py input.sam3dbody -o out.mp4
python render_skeleton.py input.sam3dbody --zoom 0.7            # zoom out 30%
python render_skeleton.py input.sam3dbody --width 720 --height 720
python render_skeleton.py input.sam3dbody --line_width 3 --radius 5
python render_skeleton.py input.sam3dbody --fps 25
```

| flag | default | meaning |
|------|---------|---------|
| `-o, --output` | `<input>_skeleton.mp4` | output path |
| `--width` / `--height` | from metadata | output size (`0` = auto) |
| `--fps` | from metadata | override fps (`0` = auto) |
| `--zoom` | `1.0` | zoom factor (`< 1.0` zooms out) |
| `--line_width` | `2` | bone line width |
| `--radius` | `4` | keypoint circle radius |

Colors: green = left side, orange = right side, blue = center, pink = index
fingers, light blue = middle fingers, red = ring fingers.

---

## `render_sam3dbody.py` — `.sam3dbody` → 3D mesh MP4

Reconstructs the full 3D MHR mesh from the parameters and renders it with
pyrender (headless EGL by default; needs the TorchScript MHR model and the face
topology).

```bash
python render_sam3dbody.py \
    --input input.sam3dbody \
    --faces_path mhr_faces_lod1.npy \
    --mhr_model_path assets/mhr_model.pt \
    --output_path mesh.mp4

# custom size / fps
python render_sam3dbody.py \
    --input input.sam3dbody \
    --faces_path mhr_faces_lod1.npy \
    --mhr_model_path assets/mhr_model.pt \
    --output_path mesh.mp4 \
    --width 1024 --height 1024 --fps 30
```

| flag | meaning |
|------|---------|
| `--input` | `.sam3dbody` file |
| `--faces_path` | mesh triangle list (`mhr_faces_lod1.npy`, shipped in this repo) |
| `--mhr_model_path` | TorchScript MHR model (`mhr_model.pt`, from the MHR assets / SAM 3D Body checkpoint) |
| `--output_path` | output MP4 |
| `--width` / `--height` / `--fps` | output size / fps |

No GPU? `sudo apt-get install libosmesa6-dev` and `export PYOPENGL_PLATFORM=osmesa`.

---

## `render_npz_to_mp4.py` — per-frame NPZ directory → MP4

For a directory of per-frame `.npz` files (as produced by SAM 3D Body's
`demo.py`). Supports mesh / overlay / side-by-side modes.

```bash
# mesh only, white background
python render_npz_to_mp4.py --npz_dir ./output_npz/ --faces_path mhr_faces_lod1.npy --output_path result.mp4

# side-by-side with the original frames
python render_npz_to_mp4.py --npz_dir ./output_npz/ --faces_path mhr_faces_lod1.npy \
    --image_dir ./input_frames/ --output_path result.mp4 --mode side_by_side

# reconstruct from MHR params when vertices aren't stored in the NPZs
python render_npz_to_mp4.py --npz_dir ./output_npz/ --faces_path mhr_faces_lod1.npy \
    --mhr_model_path assets/mhr_model.pt --output_path result.mp4
```

| flag | meaning |
|------|---------|
| `--npz_dir` | directory of per-frame `.npz` files |
| `--faces_path` | mesh triangle list (`mhr_faces_lod1.npy`) |
| `--output_path` | output MP4 |
| `--image_dir` | original frames (for `overlay` / `side_by_side`) |
| `--mode` | `mesh` (default), `overlay`, or `side_by_side` |
| `--mhr_model_path` | optional — reconstruct vertices from MHR params |

---

## `tools/extract_mhr_head_buffers.py` — make `assets/mhr_head_buffers.npz`

Pulls the PCA weights (hand pose, body scale) and hand-joint indices out of the
SAM 3D Body checkpoint. The exporter needs this file.

```bash
python tools/extract_mhr_head_buffers.py --ckpt /path/to/sam-3d-body-dinov3/model.ckpt --out assets/mhr_head_buffers.npz
```

## `tools/extract_faces.py` — re-make `mhr_faces_lod1.npy`

A copy already ships in the repo root; regenerate only if you use a different
MHR build.

```bash
python tools/extract_faces.py --mhr-model /path/to/mhr_model.pt --out mhr_faces_lod1.npy
```

## `tools/build_static_mesh_glb.py` — rest-pose mesh GLBs for web apps

Builds `lodN.glb` (mesh only, no animation) so a web app can load the mesh once
and stream small `--anim-only` clips onto it (node names match).

```bash
python tools/build_static_mesh_glb.py --assets assets --lod 0 --lod 1   # -> assets/lod0.glb, assets/lod1.glb
```

| flag | default | meaning |
|------|---------|---------|
| `--assets` | `assets` | MHR assets dir (`lodN.fbx` + `compact_v6_1.model`) |
| `--lod N` | `0` and `1` | LOD(s) to build; repeat for several |
| `--fps F` | `30.0` | fps stamped into the GLB |
| `--outdir DIR` | same as `--assets` | where to write the `.glb` files |
