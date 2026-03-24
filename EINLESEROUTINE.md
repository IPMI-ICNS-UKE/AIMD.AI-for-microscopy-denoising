# EinleseRoutine — Single-File Inference Pipeline

## What was added

This branch introduces `single_file_inference_framework.py`, a self-contained CLI script that runs the denoising model on a **single image file** — without needing the full training notebook setup.

Previously, running inference required the notebook pipeline with its full dataset structure. This script decouples inference from that: point it at one image and a model `.pth` file and it returns a denoised result.

---

## Requirements

```
imageio
numpy
fastai
torch
opencv-python
```

---

## Usage

```bash
python3 single_file_inference_framework.py \
  --input  <path/to/input_image> \
  --output <path/to/output_image> \
  --model-backend hagen_fastai \
  --model-dir <path/to/folder/containing/model/> \
  --model-name Basemodel_Hagen
```

**`--model-dir` must point to the directory that contains the `.pth` file, not the `.pth` file itself.** The script appends `/<model-name>.pth` automatically.

---

## All flags

| Flag | Required | Default | Description |
|---|---|---|---|
| `--input` | yes | — | Path to the input image |
| `--output` | yes | — | Path to write the denoised image |
| `--model-backend` | no | `dummy` | `dummy` or `hagen_fastai` |
| `--model-name` | no | `Basemodel_Hagen` | Stem of the `.pth` file (without extension) |
| `--model-dir` | no | `./DUMMY_models_path` | **Directory** containing `<model-name>.pth` |
| `--device` | no | `auto` | `auto` (GPU if available) or `cpu` |
| `--no-fp16` | no | off | Disables fp16 even when CUDA is available |
| `--dls-bs` | no | `1` | Batch size used to build the FastAI learner skeleton |
| `--tile-size` | no | `256` | Edge length of each tile in pixels |
| `--overlap` | no | `32` | Overlap between adjacent tiles in pixels |
| `--max-side-without-tiling` | no | `256` | Images where max(height, width) ≤ this value are passed directly to the model without tiling |
| `--model-divisible-by` | no | `1` | Pads the image so its dimensions are divisible by this value before inference (e.g. `16` for some architectures) |

### Model backends

- **`dummy`** — passes every tile through unchanged. Useful for testing the pipeline (I/O, tiling, stitching) without loading a model.
- **`hagen_fastai`** — loads `Basemodel_Hagen` (resnet34 U-Net, trained with FastAI) from the given directory and runs actual denoising inference.

---

## Supported file formats

Reading and writing uses `imageio`, so any format it supports works: 
- PNG 
- TIFF 
- BMP 
- JPEG

Internally the `hagen_fastai` backend uses `opencv` and treats tiles as **16-bit grayscale (`uint16`)**. The recommended formats for microscopy images are **16-bit TIFF or 16-bit PNG**. The output dtype always matches the input dtype.

The script also handles **2D+t image stacks** (shape `[T, H, W]`): each frame is processed independently and the result is stacked back to the same shape.

---

## How tiling and stitching works

Large images cannot be fed to the model in one pass (memory constraints, fixed input size). The pipeline splits the image into overlapping tiles, runs inference on each, and stitches the results back together — closely following the approach used by CSBDeep.

### Step 1 — Optional divisibility padding

If `--model-divisible-by N` is set (and N > 1), the image is first padded with reflected values so that both height and width are divisible by N. This padding is removed again after stitching.

### Step 2 — Decide whether to tile

If `max(height, width) ≤ max-side-without-tiling` the image is small enough to be passed directly to the model. No tiling happens.

### Step 3 — Compute tile positions (`_axis_tiling`)

Tiling is computed independently per axis (rows, columns) and then combined into a 2D grid. For a given axis of length `L`:

```
stride = tile_size - 2 * overlap
```

Tiles are placed at regular stride intervals. The last tile is shifted left/right so it always ends exactly at the image boundary — avoiding a partial tile at the edge. This means the last two tiles may overlap more than the configured `overlap`.

Each tile stores four coordinates:
- **`read_start / read_stop`**: which pixels to cut from the image (includes the overlap border on both sides).
- **`write_start / write_stop`**: which region of the output canvas this tile is responsible for (the inner part, without the overlap border).

The first tile writes from pixel 0 up to `tile_size - overlap`.
Interior tiles write only their inner `tile_size - 2*overlap` pixels.
The last tile writes from its `write_start` to the end of the image.

### Step 4 — Run inference

All tiles are collected into a flat list and passed to `model.predict_tiles()` in one call. For the `hagen_fastai` backend this means:

1. Each tile is saved as a temporary 16-bit PNG.
2. A FastAI `test_dl` is built over these files.
3. `learn.get_preds()` runs the model on the batch.
4. Predictions are resized back to the original tile shape (bilinear interpolation), clipped to `[0, 1]`, and scaled to `[0, 65535]`.

### Step 5 — Stitch

For each predicted tile, only the inner `write` region (without the overlap border) is copied into the output canvas:

```
output[write_start:write_stop, ...] = prediction[inner_crop]
```

Because the overlap zones are discarded, boundary artefacts from the model edges never appear in the final image. Adjacent tiles meet exactly at their shared boundary with no blending.

### Step 6 — Remove padding and save

Divisibility padding added in Step 1 is cropped off. The result is cast back to the input dtype (rounding and clipping for integer types) and written to disk.

---

## Example — 2048×2048 image with default settings

```
tile_size=256, overlap=32  →  stride=192
tiles per axis = ceil(2048 / 192) = 11
total tiles = 11 × 11 = 121
```

This is exactly the `[DEBUG]` line printed at runtime:

```
[DEBUG] Tiling frame (2048, 2048) into 11x11 = 121 tiles (tile_size=256, overlap=32)
```

---

## Extending with a custom model

The base class `InferenceModel` defines the interface:

```python
class InferenceModel:
    def predict(self, tile: np.ndarray) -> np.ndarray: ...
    def predict_tiles(self, tiles: List[np.ndarray]) -> List[np.ndarray]: ...
```

Subclass it, implement `predict` (or override `predict_tiles` for batched inference), instantiate your model in `main()`, and the rest of the pipeline works unchanged.
