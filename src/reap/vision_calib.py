"""
Local addition -- no upstream equivalent.

Multimodal (vision) calibration batches for the layerwise observer. Closes the
text-only-saliency gap documented in el_prune.py's header: REAP's expert
saliency is measured on text-only calibration, so experts that predominantly
serve image tokens look unsalient and get pruned first. Feeding a modest set
of image+text samples through the model during the observer phase makes
vision-token expert usage visible to the saliency statistics.

How it plugs in: each sample becomes ONE processor-built batch dict
(input_ids + pixel_values + image_grid_thw ...). These dicts join the plain
text batches in `data_batches`; `LayerwiseMoEObserver._capture_first_block_
inputs.prepare_model_inputs` already forwards arbitrary tensor-valued dict
entries into `model(**batch)`, so the vision tower runs, image features get
merged into the pre-block-0 hidden states, and everything downstream (block
replay, per-expert stats, MTP replay capture) is position-agnostic and
unchanged. The only extra requirement is a REAL (materialized) vision tower
-- see `materialize_vision_tower`, called from layerwise_prune's model
builder when vision batches are requested.

Image sources supported:
  - a local directory of images (*.jpg/*.jpeg/*.png/*.webp), captioned with a
    generic instruction -- no downloads, drop-your-own-images style;
  - a HuggingFace dataset id whose rows have an 'image' column (PIL) and a
    text column (caption/question/text/sentence -- auto-detected).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
_TEXT_COLUMNS = ("caption", "question", "text", "sentence", "query")
_GENERIC_PROMPT = "Describe this image in detail."


def materialize_vision_tower(model, disk_index, device: str = "cpu") -> None:
    """Materialize model.visual (meta -> real) so image batches can run.
    ~0.8GB bf16 at Ornith scale -- small next to the streamed decoder blocks."""
    from reap.disk_stream_util import materialize_module

    visual = getattr(getattr(model, "model", model), "visual", None)
    if visual is None:
        raise ValueError("Model has no .model.visual submodule -- not a multimodal checkpoint?")
    if not any(str(p.device) == "meta" for p in visual.parameters()):
        return  # already real
    materialize_module(visual, "model.visual", disk_index, device=device)
    logger.info("Materialized vision tower (model.visual) on %s for multimodal calibration", device)


def _iter_local_images(directory: Path, limit: int):
    from PIL import Image

    paths = sorted(p for p in directory.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)
    for path in paths[:limit]:
        try:
            yield Image.open(path).convert("RGB"), _GENERIC_PROMPT
        except Exception as e:
            logger.warning("Skipping unreadable image %s: %s", path, e)


def _iter_hf_images(dataset_name: str, split: str, limit: int):
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=True)
    text_col = None
    count = 0
    for row in ds:
        if count >= limit:
            break
        image = row.get("image")
        if image is None:
            continue
        if text_col is None:
            text_col = next((c for c in _TEXT_COLUMNS if row.get(c)), None)
        text = row.get(text_col) if text_col else None
        if isinstance(text, (list, tuple)):
            text = text[0] if text else None
        yield image.convert("RGB"), (text or _GENERIC_PROMPT)
        count += 1


def load_vision_batches(
    model_dir: str,
    source: str,
    num_samples: int,
    split: str = "train",
    max_pixels: int | None = 451584,  # 672x672: bounds vision-token count per sample
) -> List[Dict[str, Any]]:
    """Build one processor-encoded batch dict per image+text sample."""
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(
            model_dir, trust_remote_code=True, **({"max_pixels": max_pixels} if max_pixels else {})
        )
    except TypeError:
        # Older/newer image-processor signatures without a max_pixels kwarg.
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    source_path = Path(source)
    if source_path.is_dir():
        samples = _iter_local_images(source_path, num_samples)
    else:
        samples = _iter_hf_images(source, split, num_samples)

    batches: List[Dict[str, Any]] = []
    for image, text in samples:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }]
        try:
            encoded = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception as e:
            logger.warning("Skipping sample (processor failed: %s)", e)
            continue
        batches.append({k: v for k, v in encoded.items() if torch.is_tensor(v)})

    if not batches:
        raise ValueError(
            f"No usable vision calibration samples from {source!r} "
            f"(wanted {num_samples})"
        )
    logger.info(
        "Built %d vision calibration batches from %s (median seq len %d)",
        len(batches),
        source,
        sorted(b["input_ids"].shape[-1] for b in batches)[len(batches) // 2],
    )
    return batches
