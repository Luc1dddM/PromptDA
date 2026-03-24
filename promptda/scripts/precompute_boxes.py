from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import tyro
from PIL import Image
from tqdm.auto import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _find_images(input_dir: Path) -> list[Path]:
    return sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES])


def _to_feature_boxes(
    boxes_xyxy: list[list[float]],
    image_w: int,
    image_h: int,
    feat_w: int,
    feat_h: int,
) -> list[list[float]]:
    if len(boxes_xyxy) == 0:
        return []

    scale_x = feat_w / float(image_w)
    scale_y = feat_h / float(image_h)
    scaled = []
    for x1, y1, x2, y2 in boxes_xyxy:
        sx1 = max(0.0, min(float(feat_w), x1 * scale_x))
        sy1 = max(0.0, min(float(feat_h), y1 * scale_y))
        sx2 = max(0.0, min(float(feat_w), x2 * scale_x))
        sy2 = max(0.0, min(float(feat_h), y2 * scale_y))
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        scaled.append([sx1, sy1, sx2, sy2])
    return scaled


def main(
    input_dir: str,
    output_dir: Optional[str] = None,
    model_name: str = "yolov8n.pt",
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    patch_size: int = 14,
    feature_height: Optional[int] = None,
    feature_width: Optional[int] = None,
    sidecar_suffix: str = ".boxes.json",
    device: str = "cuda",
):
    """Precompute YOLO boxes and save sidecar JSONs in feature-map coordinates (xyxy)."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for offline detection. Install it with: pip install ultralytics"
        ) from exc

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    output_path = Path(output_dir) if output_dir is not None else input_path
    output_path.mkdir(parents=True, exist_ok=True)

    image_paths = _find_images(input_path)
    detector = YOLO(model_name)

    for image_path in tqdm(image_paths, desc="Precomputing boxes"):
        with Image.open(image_path) as im:
            image_w, image_h = im.size

        feat_h = int(feature_height) if feature_height is not None else max(1, image_h // patch_size)
        feat_w = int(feature_width) if feature_width is not None else max(1, image_w // patch_size)

        results = detector.predict(
            source=str(image_path),
            verbose=False,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
        )
        result = results[0]
        raw_boxes = result.boxes.xyxy.detach().cpu().tolist() if result.boxes is not None else []

        feature_boxes = _to_feature_boxes(raw_boxes, image_w, image_h, feat_w, feat_h)

        rel = image_path.relative_to(input_path)
        
        # We want the 'boxes' folder at the same level as the folder directly containing the image.
        if str(rel.parent) == '.':
            # Example: input_path is .../41048190/wide, image is img.jpg
            # target => .../41048190/boxes
            boxes_dir = output_path.parent / "boxes"
        else:
            # Example: input_path is .../41048190, image is wide/img.jpg
            # rel.parent is 'wide', rel.parent.parent is '.'
            # target => .../41048190/boxes
            boxes_dir = output_path / rel.parent.parent / "boxes"
            
        sidecar_path = (boxes_dir / rel.name).with_suffix(sidecar_suffix)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "image_path": str(rel),
            "image_size": [image_h, image_w],
            "feature_size": [feat_h, feat_w],
            "boxes_xyxy_feature": feature_boxes,
        }
        sidecar_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
