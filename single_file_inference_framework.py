#!/usr/bin/env python3
"""
Single-file inference scaffold:
- load one image file
- tile if image is larger than a threshold
- run model inference per tile (dummy by default)
- stitch tiles back together
- save output image
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import argparse
import sys
import tempfile

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


@dataclass
class PipelineConfig:
    tile_size: int = 256
    overlap: int = 32
    max_side_without_tiling: int = 256
    model_divisible_by: int = 1


@dataclass(frozen=True)
class AxisTile:
    read_start: int
    read_stop: int
    write_start: int
    write_stop: int


class InferenceModel:
    """Interface for pluggable inference backends."""

    def predict(self, tile: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_tiles(self, tiles: List[np.ndarray]) -> List[np.ndarray]:
        return [self.predict(tile) for tile in tiles]


class DummyIdentityModel(InferenceModel):
    """Dummy model: returns tile unchanged."""

    def predict(self, tile: np.ndarray) -> np.ndarray:
        return tile


class BasemodelHagenFastAI(InferenceModel):
    """FastAI backend for Basemodel_Hagen, inference-only runtime."""

    def __init__(
        self,
        model_name: str = "Basemodel_Hagen",
        model_dir: Path = Path("./DUMMY_models_path"),
        dls_bs: int = 1,
        use_fp16: bool = True,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.dls_bs = dls_bs
        self.use_fp16 = use_fp16
        self.device_pref = device
        self._learn = None
        self._torch = None
        self._tmp_root = Path(tempfile.mkdtemp(prefix="aimd_single_infer_"))
        self._tmp_gt = self._tmp_root / "gt"
        self._tmp_noisy = self._tmp_root / "noisy"
        self._tmp_gt.mkdir(parents=True, exist_ok=True)
        self._tmp_noisy.mkdir(parents=True, exist_ok=True)
        self._cv2 = None

    def _open_img(self, path):
        img_array = self._cv2.imread(str(path), -1)
        if img_array is None:
            raise ValueError(f"Could not read image: {path}")
        img_array = np.stack([img_array, img_array, img_array], 0)
        img_array = img_array / 65535.0
        img_array = img_array.astype("float32")
        img_tensor = self._torch.tensor(img_array, dtype=self._torch.float32)
        tensor_un = img_tensor.unsqueeze(0)
        tensor_res = self._torch.nn.functional.interpolate(
            tensor_un, size=(256, 256), mode="bilinear", align_corners=True
        )
        return tensor_res.squeeze(0)

    def _ensure_dummy_training_files(self):
        dummy = np.zeros((256, 256), dtype=np.uint16)
        dummy_name = "dummy.png"
        self._cv2.imwrite(str(self._tmp_noisy / dummy_name), dummy)
        self._cv2.imwrite(str(self._tmp_gt / dummy_name), dummy)

    def _ensure_model_loaded(self) -> None:
        if self._learn is not None:
            return

        try:
            import cv2
            import torch
            from fastai.vision.all import (
                DataBlock,
                Normalize,
                NormType,
                TransformBlock,
                get_image_files,
                imagenet_stats,
                resnet34,
                unet_learner,
            )
        except ImportError as exc:
            raise ImportError(
                "BasemodelHagenFastAI requires fastai, torch, and opencv-python."
            ) from exc

        self._cv2 = cv2
        self._torch = torch

        model_file = self.model_dir / f"{self.model_name}.pth"
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model weights not found: {model_file} "
                f"(replace dummy path via --model-dir)."
            )

        self._ensure_dummy_training_files()

        # Build learner DataLoaders like notebook pipeline so test_dl batching matches.
        dblock = DataBlock(
            blocks=(TransformBlock(self._open_img), TransformBlock(self._open_img)),
            get_items=get_image_files,
            get_y=lambda x: self._tmp_gt / x.name,
            splitter=lambda items: (list(range(len(items))), []),
            batch_tfms=[Normalize.from_stats(*imagenet_stats)],
        )
        dls_den = dblock.dataloaders(self._tmp_noisy, bs=self.dls_bs, path=self._tmp_root)
        dls_den.c = 3

        loss_func = self._torch.nn.L1Loss()
        learn_den = unet_learner(
            dls_den,
            resnet34,
            loss_func=loss_func,
            blur=True,
            norm_type=NormType.Weight,
            self_attention=True,
            y_range=(-3, 3),
        )
        # Keep an absolute model_dir so fastai does not resolve relative to dls.path.
        learn_den.model_dir = self.model_dir
        learn_den.load(self.model_name)

        if self.device_pref == "cpu":
            learn_den.to_fp32()
        else:
            cuda_ok = torch.cuda.is_available()
            if self.use_fp16 and cuda_ok:
                learn_den.to_fp16()
            else:
                learn_den.to_fp32()

        learn_den.model.eval()
        self._learn = learn_den

    def predict(self, tile: np.ndarray) -> np.ndarray:
        return self.predict_tiles([tile])[0]

    def predict_tiles(self, tiles: List[np.ndarray]) -> List[np.ndarray]:
        self._ensure_model_loaded()
        torch = self._torch
        learn = self._learn

        if not tiles:
            return []
        if any(tile.ndim != 2 for tile in tiles):
            bad = [tile.shape for tile in tiles if tile.ndim != 2][:3]
            raise ValueError(f"BasemodelHagenFastAI expects 2D tiles, got shapes like {bad}")

        original_shapes = [tile.shape for tile in tiles]
        infer_dir = self._tmp_root / "infer_tiles"
        infer_dir.mkdir(parents=True, exist_ok=True)
        for p in infer_dir.glob("*.png"):
            p.unlink()

        tile_paths = []
        for i, tile in enumerate(tiles):
            path = infer_dir / f"tile_{i:06d}.png"
            self._cv2.imwrite(str(path), tile.astype(np.uint16))
            tile_paths.append(path)

        test_dl = learn.dls.test_dl(
            tile_paths,
            bs=min(max(1, self.dls_bs), len(tile_paths)),
            num_workers=0,
        )
        with learn.no_bar(), learn.no_logging():
            preds, _ = learn.get_preds(dl=test_dl, with_decoded=False)

        out_tiles: List[np.ndarray] = []
        for i, pred in enumerate(preds):
            y = pred.detach().float().cpu().unsqueeze(0)
            y = torch.nn.functional.interpolate(y, size=original_shapes[i], mode="bilinear", align_corners=True)
            y = y[0, 0].numpy()
            y = np.clip(y, 0.0, 1.0) * 65535.0
            out_tiles.append(y.astype(np.float32))
        return out_tiles


def load_image(path: Path) -> np.ndarray:
    img = imageio.imread(path)
    if img.ndim not in (2, 3):
        raise ValueError(f"Expected 2D image or 2D+t stack, got shape {img.shape}")
    return img


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 3:
        imageio.mimwrite(path, image)
    else:
        imageio.imwrite(path, image)


def pad_to_divisible(frame: np.ndarray, div_by: int) -> Tuple[np.ndarray, Tuple[Tuple[int, int], Tuple[int, int]]]:
    if div_by <= 1:
        return frame, ((0, 0), (0, 0))
    h, w = frame.shape
    pad_h = (div_by - (h % div_by)) % div_by
    pad_w = (div_by - (w % div_by)) % div_by
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded = np.pad(frame, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect")
    return padded, ((pad_top, pad_bottom), (pad_left, pad_right))


def crop_padding(frame: np.ndarray, pad: Tuple[Tuple[int, int], Tuple[int, int]]) -> np.ndarray:
    (pt, pb), (pl, pr) = pad
    h, w = frame.shape
    y0 = pt
    y1 = h - pb if pb > 0 else h
    x0 = pl
    x1 = w - pr if pr > 0 else w
    return frame[y0:y1, x0:x1]


def _axis_tiling(length: int, tile_size: int, overlap: int) -> List[AxisTile]:
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if tile_size <= 2 * overlap:
        raise ValueError("tile_size must be > 2*overlap")
    if length <= tile_size:
        return [AxisTile(0, length, 0, length)]

    tiles: List[AxisTile] = []
    prev_read_start = 0
    prev_read_stop = tile_size
    prev_write_stop = None

    while True:
        read_start = prev_read_start
        read_stop = prev_read_stop
        at_begin = len(tiles) == 0
        at_end = read_stop == length

        if at_begin:
            write_start = 0
            write_stop = length if at_end else read_stop - overlap
        elif at_end:
            assert prev_write_stop is not None
            write_start = prev_write_stop
            write_stop = length
        else:
            assert prev_write_stop is not None
            write_start = prev_write_stop
            write_stop = write_start + (read_stop - read_start - 2 * overlap)

        tiles.append(AxisTile(read_start, read_stop, write_start, write_stop))

        if at_end:
            break

        next_start = read_stop - 2 * overlap
        next_stop = next_start + tile_size
        if next_stop > length:
            shift = length - next_stop
            next_start += shift
            next_stop += shift

        prev_read_start = next_start
        prev_read_stop = next_stop
        prev_write_stop = write_stop

    return tiles


def predict_tiled_like_csbdeep(frame: np.ndarray, model: InferenceModel, tile_size: int, overlap: int) -> np.ndarray:
    if frame.ndim != 2:
        raise ValueError("predict_tiled_like_csbdeep expects a 2D frame")
    y_tiles = _axis_tiling(frame.shape[0], tile_size, overlap)
    x_tiles = _axis_tiling(frame.shape[1], tile_size, overlap)
    out = np.empty(frame.shape, dtype=np.float32)
    tiles = []
    tile_meta = []
    for yt in y_tiles:
        for xt in x_tiles:
            tile = frame[yt.read_start:yt.read_stop, xt.read_start:xt.read_stop]
            tiles.append(tile)
            tile_meta.append((yt, xt, tile.shape))

    pred_tiles = model.predict_tiles(tiles)
    if len(pred_tiles) != len(tiles):
        raise ValueError("Model returned different number of predicted tiles than input tiles")

    for pred, (yt, xt, tile_shape) in zip(pred_tiles, tile_meta):
        pred = pred.astype(np.float32)
        if pred.shape != tile_shape:
            raise ValueError(f"Model changed tile shape from {tile_shape} to {pred.shape}")

        y_src_start = yt.write_start - yt.read_start
        y_src_stop = y_src_start + (yt.write_stop - yt.write_start)
        x_src_start = xt.write_start - xt.read_start
        x_src_stop = x_src_start + (xt.write_stop - xt.write_start)

        out[yt.write_start:yt.write_stop, xt.write_start:xt.write_stop] = pred[
            y_src_start:y_src_stop, x_src_start:x_src_stop
        ]

    return out


def cast_like_reference(arr: np.ndarray, reference_dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(reference_dtype, np.integer):
        info = np.iinfo(reference_dtype)
        arr = np.rint(arr)
        arr = np.clip(arr, info.min, info.max)
        return arr.astype(reference_dtype)
    return arr.astype(reference_dtype)


def run_frame(frame: np.ndarray, model: InferenceModel, cfg: PipelineConfig) -> np.ndarray:
    padded, pad = pad_to_divisible(frame, cfg.model_divisible_by)
    h, w = padded.shape
    requires_tiling = max(h, w) > cfg.max_side_without_tiling

    if not requires_tiling:
        pred = model.predict(padded).astype(np.float32)
        if pred.shape != padded.shape:
            raise ValueError(f"Model changed frame shape from {padded.shape} to {pred.shape}")
        return crop_padding(pred, pad)

    pred = predict_tiled_like_csbdeep(padded, model, cfg.tile_size, cfg.overlap)
    return crop_padding(pred, pad)


def _tiling_debug_line(frame: np.ndarray, cfg: PipelineConfig) -> None:
    padded, _ = pad_to_divisible(frame, cfg.model_divisible_by)
    y_tiles = _axis_tiling(padded.shape[0], cfg.tile_size, cfg.overlap)
    x_tiles = _axis_tiling(padded.shape[1], cfg.tile_size, cfg.overlap)
    print(
        f"[DEBUG] Tiling frame {frame.shape} into {len(y_tiles)}x{len(x_tiles)} "
        f"= {len(y_tiles) * len(x_tiles)} tiles "
        f"(tile_size={cfg.tile_size}, overlap={cfg.overlap})"
    )


def run_pipeline(input_path: Path, output_path: Path, cfg: PipelineConfig, model: InferenceModel) -> None:
    raw = load_image(input_path)
    in_dtype = raw.dtype

    if raw.ndim == 2:
        _tiling_debug_line(raw, cfg)
        pred = run_frame(raw, model, cfg)
        out = cast_like_reference(pred, in_dtype)
    else:
        _tiling_debug_line(raw[0], cfg)
        frames = [run_frame(raw[t], model, cfg) for t in tqdm(range(raw.shape[0]), desc="Frames", unit="frame", file=sys.stderr, dynamic_ncols=True)]
        stacked = np.stack(frames, axis=0)
        out = cast_like_reference(stacked, in_dtype)

    save_image(output_path, out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-file tiled inference scaffold")
    parser.add_argument("--input", type=Path, required=True, help="Input image path")
    parser.add_argument("--output", type=Path, required=True, help="Output image path")
    parser.add_argument(
        "--model-backend",
        type=str,
        default="dummy",
        choices=["dummy", "hagen_fastai"],
        help="Inference backend",
    )
    parser.add_argument("--model-name", type=str, default="Basemodel_Hagen", help="FastAI model name")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./DUMMY_models_path"),
        help="Directory containing <model-name>.pth",
    )
    parser.add_argument(
        "--dls-bs",
        type=int,
        default=1,
        help="Dummy DataLoader batch size used only to build learner skeleton",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu"],
        help="Inference device preference",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 even when CUDA is available",
    )
    parser.add_argument("--tile-size", type=int, default=256, help="Tile edge length")
    parser.add_argument("--overlap", type=int, default=32, help="Tile overlap in pixels")
    parser.add_argument(
        "--max-side-without-tiling",
        type=int,
        default=256,
        help="Skip tiling if max(height, width) <= this value",
    )
    parser.add_argument(
        "--model-divisible-by",
        type=int,
        default=1,
        help="Optional pad-to-divisibility constraint before inference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        tile_size=args.tile_size,
        overlap=args.overlap,
        max_side_without_tiling=args.max_side_without_tiling,
        model_divisible_by=args.model_divisible_by,
    )

    if args.model_backend == "hagen_fastai":
        model = BasemodelHagenFastAI(
            model_name=args.model_name,
            model_dir=args.model_dir,
            dls_bs=args.dls_bs,
            use_fp16=not args.no_fp16,
            device=args.device,
        )
    else:
        model = DummyIdentityModel()
    run_pipeline(args.input, args.output, cfg, model)
    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()
