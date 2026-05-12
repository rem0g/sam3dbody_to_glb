# assets/

The MHR body-model files go here. They are **not** checked into git (too large)
— download them once with the steps below. `export_glb_pymomentum.py` and the
`render_*.py` scripts auto-detect this folder.

After setup this directory should contain:

```
assets/
├── compact_v6_1.model              # MHR parameter definition  (~31 KB)
├── lod0.fbx ... lod6.fbx           # MHR mesh LODs              (FBX, ~0.6–30 MB each)
├── mhr_head_buffers.npz            # PCA weights — extracted from the checkpoint
├── mhr_model.pt                    # TorchScript MHR model      (only needed for render_*.py)
└── (corrective_blendshapes_lodN.npz, corrective_activation.npz — only if you use --face-expr)
```

## 1. MHR mesh assets (FBX + model)

```bash
cd assets
curl -OL https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip
unzip assets.zip && rm assets.zip
cd ..
```

This gives you `compact_v6_1.model`, `lod{0..6}.fbx`, the corrective-blendshape
NPZs, and `mhr_model.pt`. (If `unzip` produces an `assets/` subfolder, move its
contents up one level so the files sit directly under `assets/`.)

## 2. `mhr_head_buffers.npz` (from the SAM 3D Body checkpoint)

This file holds the PCA weights used to assemble MHR parameters correctly. It is
extracted from the `model.ckpt` of the SAM 3D Body release (gated on Hugging Face):

```bash
# Request access at https://huggingface.co/facebook/sam-3d-body-dinov3 first, then:
huggingface-cli login
python -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/sam-3d-body-dinov3', local_dir='checkpoint-sam-3d-body')
"

python tools/extract_mhr_head_buffers.py \
    --ckpt checkpoint-sam-3d-body/model.ckpt \
    --out  assets/mhr_head_buffers.npz
```

The `mhr_model.pt` that the `render_*.py` scripts use also lives inside that
checkpoint (`assets/mhr_model.pt`) and inside the MHR `assets.zip` above — either
copy works.

## Notes

- For plain GLB export (`export_glb_pymomentum.py` without `--face-expr`) you only
  need `compact_v6_1.model`, the `lodN.fbx` file for the LOD you want, and
  `mhr_head_buffers.npz`. The big `corrective_blendshapes_*.npz` files are only
  needed for face/expression export.
- `mhr_faces_lod1.npy` (the mesh triangle list used by the renderers) is small,
  so a copy ships in the repo root. Re-extract it with `tools/extract_faces.py`
  if you ever need to.
