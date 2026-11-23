# Mac demo

A minimal Mac app that shows live depth from the camera, running Depth Pro 2 in unbuffered mode with
Core ML. It needs Apple silicon, macOS 15 or later, and Xcode 16 or later, or its command line tools.

```bash
uv run --extra coreml python -m demo.export stream -m depthpro2-s -w 4
./demo/run.sh depthpro2-s_480x854_w4.mlpackage
```

`run.sh` builds the app (about 20 s the first time) and starts it on the package. macOS asks to let
your terminal use the camera the first time. Press q to quit.

## Export

`python -m demo.export stream` writes `<model>_<H>x<W>_w<window>.mlpackage`, for one input size:

| Option | Default | |
|---|---|---|
| `-m` | | `depthpro2-s`, `depthpro2-b`, `depthpro2-l` or `depthpro2-h` |
| `-s HxW` | 480x854 | Input height and width; the app and the tables below say width×height, so this is 854x480 there. Repeat it for a package with several sizes; the first is the app's default. |
| `-w` | 4 | Frames attended, size of the temporal window. Set to 1 to export to an image model. |
| `-o` | `.` | Output folder. |

Repeat `-s` for one package, `<model>_w<window>.mlpackage`, with a function per size, named
`res_<W>x<H>`, which the app lets you switch between:

```bash
uv run --extra coreml python -m demo.export stream -m depthpro2-s -w 4 \
    -s 480x854 -s 360x640 -s 512x896 -s 640x1138 -s 720x1280
```

The model is exported in float16, with the residual stream and attention logits scaled down so that
they stay in range (`--fp16-peak`). The package's metadata says how.

`python -m demo.export fov` writes `depthpro2-fov_512x896.mlpackage`, the field-of-view estimator,
which takes the 16:9 frames of the demo's cameras. Put it next to the depth package: for a camera
that reports no focal length, the app estimates it from the camera's frames.

## App

Memory grows quickly with the size: S needs about 1.5 GB at 854x480 and about 5 GB at 1280x720, L about
4 GB at 854x480 and about 18 GB at 1280x720. On a Mac with 16 GB, keep to the default size or to S and B.
The app tells you when it holds more than half of the Mac's memory.

```
./demo/run.sh PACKAGE [--resolution WxH] [--compute-units gpu|ane]
                      [--focal-px PIXELS] [--min-depth METERS] [--max-depth METERS]
```

- For a package with several sizes, the window's menu lists them, read from the package itself, and
  switches between them at any time. The app starts at the package's default size, or at the one
  `--resolution` names.
- The app starts on the computer's own camera, not one in a connected display; the others that can
  run at the current size are in the camera menu.
- `--compute-units` is where Core ML runs the model. The default is the GPU. The packages are
  exported for it: on the Neural Engine, the largest sizes may fail to build, so check a size runs
  before relying on `ane`.
- Switching size lets go of the old size before it builds the new one, so the memory of one size is
  enough; the picture freezes for the seconds the new size takes to load, and if it fails to load the
  old size comes back. A size the camera has no format for is refused.
- The built-in camera reports its focal length. For another camera, give `--focal-px`, in pixels of
  the camera's frames at the size the app prints, or export the field-of-view package as above.
- The color bar fits the scene's depth range when a camera starts; press c to fit it again, or
  fix it with `--min-depth` and `--max-depth`.
- Press f to estimate the focal length again, when the field-of-view package is next to the model.

## Use the exported model in your own app

The exported model needs macOS 15 or later.

- **Function.** Load a size with `MLModelConfiguration.functionName = "res_<W>x<H>"`; without a
  name, you get the first size exported.
- **Input.** `frame`, a float16 `MLMultiArray` (1, 3, H, W) of RGB values in [0, 1]. The demo
  scales each camera frame to cover H×W and crops the center, on the GPU, in
  [Preprocess.swift](apple/Sources/demo/Preprocess.swift).
- **Output.** `inverse_depth` (1, H, W): inverse depth normalized by the focal length, 0 where the
  scene is infinitely far. Depth in meters is `f_px / (W * inverse_depth)`, where `f_px` is the
  focal length in pixels of the H×W frame: the camera's, times the scale factor (cropping keeps it).
- **State.** With `-w` above 1, the model keeps the earlier frames in Core ML state. Create it with
  `makeState()`, pass it to every prediction, and create a new one to start over, as for a new
  video or another camera.
- **Compute units.** `MLModelConfiguration.computeUnits` decides where the model runs. The demo
  uses `.cpuAndGPU`.

## Runtime on a MacBook Pro M4 Max

Median time of one prediction in milliseconds per frame, with frames per second (rounded down) in
parentheses, on a **MacBook Pro with an M4 Max** and 64 GB, on the GPU (the app's default), for the
packages exported with `-w 1` and `-w 4` and the five sizes of the table; the released models use
`-w 4`. A first prediction, which compiles for the device, is left out, as are the next `w + 2`.
Preprocessing and rendering are not included. The machine was not otherwise idle, so expect a few
percent of noise.

Single frame (`-w 1`):

| Model, ms (fps) | 640x360 | 854x480 | 896x512 | 1138x640 | 1280x720 |
|---|---|---|---|---|---|
| S | **10 (102)** | **17 (58)** | **19 (53)** | 34 (29) | 45 (22) |
| B | **26 (39)** | 44 (22) | 48 (20) | 85 (11) | 115 (8) |
| L | 80 (12) | 137 (7) | 152 (6) | 272 (3) | 385 (2) |
| H+ | 170 (5) | 303 (3) | 339 (2) | 612 (1) | 826 (1) |

W=4 (`-w 4`):

| Model, ms (fps) | 640x360 | 854x480 | 896x512 | 1138x640 | 1280x720 |
|---|---|---|---|---|---|
| S | **13 (76)** | **26 (39)** | **30 (33)** | 61 (16) | 88 (11) |
| B | **32 (31)** | 60 (16) | 68 (14) | 137 (7) | 211 (4) |
| L | 96 (10) | 188 (5) | 212 (4) | 425 (2) | 648 (1) |
| H+ | 197 (5) | 375 (2) | 424 (2) | 832 (1) | 1233 (0) |

**Bold** is real time, 30 frames per second or more. S is real time up to 896x512 for a single frame
and at W=4, and B at 640x360.
