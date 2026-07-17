from __future__ import annotations

import warnings
import logging
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from ..benchmark.media import resolve_path

from .base import BaseProtocol, ParseResult


class StateSimilarityProtocol(BaseProtocol):
    """Match a generated future/state image to candidate option images."""

    name = "state_similarity"

    _clip_model = None
    _clip_device = None

    def parse(self, generated_path: str, item: Dict[str, Any], benchmark_root: str) -> ParseResult:
        option_paths = self._option_paths(item, benchmark_root)
        if len(option_paths) < 2:
            return ParseResult("", False, {"error": "missing candidate images"})
        missing = [p for p in option_paths if not Path(p).exists()]
        if missing:
            return ParseResult("", False, {"error": f"missing files: {missing[:2]}"})

        try:
            import torch
            from sentence_transformers import util

            clip = self._get_clip()
            query_emb = clip.encode(Image.open(generated_path), convert_to_tensor=True)
            target_embs = clip.encode([Image.open(p) for p in option_paths], convert_to_tensor=True)
            scores = util.cos_sim(query_emb, target_embs)[0]
            best_idx = int(torch.argmax(scores))
            label = self._labels()[best_idx]
            return ParseResult(label, True, {"scores": [float(x) for x in scores]})
        except Exception as exc:
            return ParseResult("", False, {"error": str(exc)})

    def _option_paths(self, item: Dict[str, Any], benchmark_root: str) -> list[str]:
        input_spec = item.get("input") or {}
        media = input_spec.get("media") or input_spec.get("images") or []
        options = []
        for entry in media:
            if isinstance(entry, dict) and entry.get("role") == "option" and entry.get("path"):
                options.append(resolve_path(benchmark_root, entry["path"]))
        if options:
            return options

        choice_options = []
        for choice in item.get("choices") or []:
            if isinstance(choice, dict):
                media_entry = choice.get("media") or {}
                if isinstance(media_entry, dict) and media_entry.get("path"):
                    choice_options.append(resolve_path(benchmark_root, media_entry["path"]))
        if choice_options:
            return choice_options

        names = item.get("metadata", {}).get("file_names", [])
        if len(names) >= 5:
            src_path = Path(resolve_path(benchmark_root, item["image_path"]))
            base = src_path.parent
            return [str((base / name).resolve()) for name in names[1:5]]
        return [resolve_path(benchmark_root, p) for p in item.get("choices", [])]

    def _labels(self) -> list[str]:
        return [str(x) for x in self.config.get("labels", ["A", "B", "C", "D"])]

    def _get_clip(self):
        if self.__class__._clip_model is None:
            import os
            from sentence_transformers import SentenceTransformer

            os.environ["HF_ENDPOINT"] = "https://huggingface.co"
            device = os.getenv("PROVISE_CLIP_DEVICE", self.config.get("device", "cpu")).strip() or "cpu"
            transformers_logger = logging.getLogger("transformers")
            old_level = transformers_logger.level
            transformers_logger.setLevel(logging.ERROR)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Using a slow image processor as `use_fast` is unset.*",
                    )
                    self.__class__._clip_model = SentenceTransformer("clip-ViT-B-32", device=device)
            finally:
                transformers_logger.setLevel(old_level)
            self.__class__._clip_device = device
        return self.__class__._clip_model
