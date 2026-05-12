# sam3dbody → GLB

Convert `.sam3dbody` files (the NPZ output of [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)
inference) into **animated GLB** files — a skinned MHR character with skeletal
animation that opens in any glTF viewer, Blender, Three.js or Babylon.js.

Also includes scripts to render `.sam3dbody` files to MP4 (2D skeleton or 3D mesh).

> Need to *create* `.sam3dbody` files from a video first? See the companion repo
> **[mp4-to-sam3dbody](../mp4-to-sam3dbody)**.

---

## Quick start

```bash
# 1. environment
conda create -n sam3d python=3.12 -y      # Python 3.12 — pymomentum has no 3.11 wheels
conda activate sam3d
pip install -r requirements.txt

# 2. one-time: download the MHR model assets  (see assets/README.md for details)
cd assets
curl -OL https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip
unzip assets.zip && rm assets.zip
cd ..
# ...and extract the PCA buffers from the SAM 3D Body checkpoint:
python tools/extract_mhr_head_buffers.py --ckpt /path/to/model.ckpt --out assets/mhr_head_buffers.npz

# 3. convert
cp my_recording.sam3dbody input/
python batch_convert.py
ls output/                                 # -> my_recording.glb
```

On Linux you may need this before running anything that imports `pymomentum`:

```bash
export LD_LIBRARY_PATH="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):$LD_LIBRARY_PATH"
```

Prefer a guided walkthrough? Open [`notebooks/sam3dbody_to_glb.ipynb`](notebooks/sam3dbody_to_glb.ipynb).
Just want a per-script command cheat-sheet? See [**USAGE.md**](USAGE.md).

---

## Repo layout

```
sam3dbody_to_glb/
├── input/                       # ← drop your .sam3dbody files here
├── output/                      # ← .glb files appear here
├── assets/                      # ← MHR model files (download once, git-ignored)
├── batch_convert.py             # convert everything in input/ → output/
├── export_glb_pymomentum.py     # the GLB exporter (single file, all options)
├── render_skeleton.py           # .sam3dbody → 2D skeleton MP4 (no GPU needed)
├── render_sam3dbody.py          # .sam3dbody → 3D mesh MP4 (pyrender, EGL/OSMesa)
├── render_npz_to_mp4.py         # per-frame NPZ dir → MP4 (SAM 3D Body demo output)
├── mhr_faces_lod1.npy           # MHR mesh triangles (used by the renderers)
├── notebooks/
│   └── sam3dbody_to_glb.ipynb
└── tools/
    ├── extract_mhr_head_buffers.py   # pull mhr_head_buffers.npz from model.ckpt
    ├── extract_faces.py              # re-extract mhr_faces_lod1.npy from mhr_model.pt
    └── build_static_mesh_glb.py      # build rest-pose lodN.glb for web apps
```

---

## Converting to GLB

### Batch (folder in → folder out)

```bash
python batch_convert.py                    # ./input/*.sam3dbody  ->  ./output/*.glb
python batch_convert.py --anim-only        # tiny animation-only GLBs (_anim.glb)
python batch_convert.py --lod 1 --smooth 2 # any export_glb_pymomentum.py flag passes through
python batch_convert.py --input some/dir --output other/dir
```

### Single file

```bash
python export_glb_pymomentum.py input.sam3dbody -o output.glb

# sign-language / talking-head friendly: lock legs + global rotation, lightly smooth
python export_glb_pymomentum.py input.sam3dbody -o output.glb --freeze-legs --freeze-root --smooth 2

# lighter mesh (LOD 1 ≈ 18K verts instead of 73K)
python export_glb_pymomentum.py input.sam3dbody -o output.glb --lod 1

# animation only (no mesh, ~0.8 MB) — pair with a static mesh GLB in a web app
python export_glb_pymomentum.py input.sam3dbody -o clip_anim.glb --anim-only --freeze-legs --freeze-root --smooth 2
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | `<input>_skeletal.glb` | Output path |
| `--assets` | auto (`./assets`) | MHR assets directory |
| `--lod` | `0` | LOD level: 0 = 73K verts … 6 = lowest |
| `--every` | `1` | Use every Nth frame |
| `--fps` | from metadata | Override FPS |
| `--freeze-legs` | off | Hold leg joints in rest pose |
| `--freeze-root` | off | Hold global body rotation fixed |
| `--smooth SIGMA` | `0` | Gaussian temporal smoothing (frames) |
| `--anim-only` | off | Export animation only, no mesh |
| `--face-expr PATH` | none | Add face/expression morph-target animation from an NPZ |

Typical sizes (LOD 0, ~200 frames): full mesh + animation ≈ 10 MB; animation-only ≈ 0.8 MB.

### Web apps: static mesh + streamed animation clips

Build the rest-pose mesh once, then ship small per-clip animations:

```bash
python tools/build_static_mesh_glb.py --assets assets --lod 0 --lod 1   # -> assets/lod0.glb, assets/lod1.glb
python export_glb_pymomentum.py clip1.sam3dbody -o clip1_anim.glb --anim-only --freeze-legs --freeze-root --smooth 2
```

Node names line up between the static mesh and the clips (`body_world`, `root`,
`l_upleg`, `c_spine0`, …), so in Babylon.js you load `lod1.glb` once and then
`ImportAnimationsAsync` each `*_anim.glb` clip.

---

## Rendering to MP4 (optional)

| Script | Input | Output | Needs |
|--------|-------|--------|-------|
| `render_skeleton.py` | `.sam3dbody` | 2D colored skeleton MP4 | nothing (OpenCV only) |
| `render_sam3dbody.py` | `.sam3dbody` | 3D mesh MP4 | `mhr_model.pt`, `mhr_faces_lod1.npy`, EGL/OSMesa |
| `render_npz_to_mp4.py` | dir of per-frame `.npz` | mesh / overlay / side-by-side MP4 | `mhr_faces_lod1.npy` (+ optional `mhr_model.pt`) |

```bash
python render_skeleton.py input.sam3dbody                 # -> input_skeleton.mp4
python render_skeleton.py input.sam3dbody --zoom 0.7 --width 720 --height 720

python render_sam3dbody.py \
    --input input.sam3dbody \
    --faces_path mhr_faces_lod1.npy \
    --mhr_model_path assets/mhr_model.pt \
    -o mesh.mp4                                            # auto-fits camera; size from source aspect
```

`render_sam3dbody.py` is a *preview* renderer (simplified parameter assembly →
approximate pose); for a faithful animation, export a GLB with
`export_glb_pymomentum.py` and view that. Since `.sam3dbody` files carry no
camera pose, it auto-fits the camera to the mesh — see [USAGE.md](USAGE.md) for
`--fit` / `--margin` / `--vfov-deg`. Headless 3D rendering uses EGL by default
(needs NVIDIA drivers). No GPU? Install OSMesa and `export PYOPENGL_PLATFORM=osmesa`.

---

## The `.sam3dbody` file format

An NPZ archive with multi-frame body estimation data. Single-person (the common case):

| Key | Shape | Description |
|-----|-------|-------------|
| `body_keypoints_3d` | `(N, 70, 3)` | 3D keypoints per frame |
| `body_keypoints_2d` | `(N, 70, 2)` | 2D keypoints per frame |
| `body_pose_params` | `(N, 133)` | MHR body pose parameters |
| `hand_pose_params` | `(N, 108)` | MHR hand pose parameters (PCA) |
| `shape_params` | `(N, 45)` | MHR identity / shape coefficients |
| `global_rot` | `(N, 3)` | Global rotation per frame |
| `bboxes` | `(N, 4)` | Person bounding boxes |
| `focal_lengths` | `(N,)` | Camera focal lengths |
| `num_persons_per_frame` | `(N,)` | Always 1 for single-person |
| `lhand_bboxes` / `rhand_bboxes` | `(N, 4)` | Hand bounding boxes |
| `metadata` | `(1,)` object | FPS, resolution, video name, keypoint names, model info |

Multi-person fallback files instead store `frames_data_json` (all frame data as a
JSON string) plus `num_persons_per_frame` and `metadata`.

Peek inside one:

```python
import numpy as np
d = np.load("recording.sam3dbody", allow_pickle=True)
print("Keys:", list(d.keys()))
for k in d: print(f"  {k}: {getattr(d[k],'shape',None)} {getattr(d[k],'dtype','')}")
```

> **Why the dedicated exporter?** The `.sam3dbody` file stores *raw* estimator
> outputs. Correctly posing the MHR skeleton requires replicating
> `MHRHead.mhr_forward()` — decoding hand-pose PCA into Euler angles, inserting
> them at the right joint indices, and reconstructing per-joint scales from
> `scale_mean + scale_params @ scale_comps`. Naively concatenating the pose
> arrays produces a broken pose. `export_glb_pymomentum.py` does it properly.

The 70 keypoints: `0–4` head · `5–8` shoulders/elbows · `9–14` hips/knees/ankles ·
`15–20` feet · `21–41` right hand · `42–62` left hand · `63–69` extras (olecranons,
cubital fossae, acromions, neck).

---

## Troubleshooting

- **`pymomentum-cpu` not found on pip** — use Python 3.12, not 3.11.
- **pymomentum segfaults loading FBX** — known C++ binding issue; the static-mesh
  helper and exporter use the supported code paths. See [MHR#12](https://github.com/facebookresearch/MHR/issues/12).
- **`MHRHead buffers not found`** — run `tools/extract_mhr_head_buffers.py` to
  create `assets/mhr_head_buffers.npz` (see [assets/README.md](assets/README.md)).
- **Blank / black 3D renders** — EGL isn't available; `sudo apt-get install
  libosmesa6-dev` and `export PYOPENGL_PLATFORM=osmesa`.
- **GLB animation looks wrong in old viewers** — most WebGL viewers cap morph
  targets at 8; use the default skeletal export, not `--face-expr`, for those.

---

## Credits & licenses

The code in this repository is released under the [MIT License](LICENSE).

It is glue code around two Meta projects whose **models, weights and assets carry
their own (more restrictive) licenses** — you must obtain and comply with those
separately; the MIT license here does **not** cover them:

- [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) — the inference model that produces `.sam3dbody` files (Meta "SAM License").
- [MHR](https://github.com/facebookresearch/MHR) — the body model whose mesh/skeleton you export (Apache-2.0).

`mhr_faces_lod1.npy` is mesh topology derived from the MHR model; it is included
here for convenience and is governed by the MHR license, not the MIT license above.
