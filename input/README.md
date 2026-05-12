# input/

Drop your `.sam3dbody` files in this folder, then run:

```bash
python batch_convert.py
```

Each `recording.sam3dbody` here becomes `recording.glb` in the [`../output/`](../output/) folder.

You can also convert a single file directly without using this folder:

```bash
python export_glb_pymomentum.py path/to/recording.sam3dbody -o path/to/recording.glb
```

`.sam3dbody` files are NPZ archives produced by SAM 3D Body inference — see the
[mp4-to-sam3dbody](../../mp4-to-sam3dbody) repo for how to create them from a video,
and the main [README](../README.md#the-sam3dbody-file-format) for the format spec.

> Files you put here are ignored by git (see `.gitignore`).
