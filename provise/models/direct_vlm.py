from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class LocalQwenDirectVLM:
    """Minimal Qwen-VL runtime for direct text baselines."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_new_tokens: int = 128,
        max_pixels: int = 1_003_520,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels
        self.model: Any = None
        self.processor: Any = None

    def load_model(self) -> None:
        if self.model is not None:
            return
        if not Path(self.model_path).is_dir():
            raise FileNotFoundError(f"Qwen checkpoint does not exist: {self.model_path}")
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Local Qwen evaluation requires torch, transformers, and qwen-vl-utils."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Local Qwen evaluation requires CUDA")

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        ).eval().to("cuda:0")

    def predict_multi(self, image_paths: list[str], question: str) -> str:
        self.load_model()
        import torch
        from qwen_vl_utils import process_vision_info

        content = [
            {
                "type": "image",
                "image": f"file://{os.path.abspath(path)}",
                "max_pixels": self.max_pixels,
            }
            for path in image_paths
        ]
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = process_vision_info(messages)
        inputs = self.processor(
            text=[rendered],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


class LocalInternVLDirectVLM:
    """Minimal InternVL chat runtime for direct text baselines."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_new_tokens: int = 128,
        input_size: int = 448,
        max_tiles_per_image: int = 4,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.max_new_tokens = max_new_tokens
        self.input_size = input_size
        self.max_tiles_per_image = max_tiles_per_image
        self.model: Any = None
        self.tokenizer: Any = None

    def load_model(self) -> None:
        if self.model is not None:
            return
        if not Path(self.model_path).is_dir():
            raise FileNotFoundError(f"InternVL checkpoint does not exist: {self.model_path}")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Local InternVL evaluation requires torch, torchvision, and transformers."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Local InternVL evaluation requires CUDA")

        self.model = AutoModel.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        ).eval().to("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )

    def predict_multi(self, image_paths: list[str], question: str) -> str:
        self.load_model()
        import torch

        tensors = [self._load_image(path) for path in image_paths]
        patch_counts = [tensor.size(0) for tensor in tensors]
        pixel_values = torch.cat(tensors).to(device="cuda:0", dtype=torch.bfloat16)
        image_prefix = "".join(
            f"Image-{index}: <image>\n" for index in range(1, len(image_paths) + 1)
        )
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            image_prefix + question,
            {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
                "pad_token_id": self.tokenizer.eos_token_id,
            },
            num_patches_list=patch_counts,
            history=None,
            return_history=False,
        )
        return str(response).strip()

    def _load_image(self, path: str):
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        from torchvision.transforms.functional import InterpolationMode

        transform = transforms.Compose(
            [
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.Resize(
                    (self.input_size, self.input_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        with Image.open(path) as image:
            tiles = _dynamic_tiles(
                image.convert("RGB"),
                image_size=self.input_size,
                max_tiles=self.max_tiles_per_image,
            )
        return torch.stack([transform(tile) for tile in tiles])


def _dynamic_tiles(image: Any, *, image_size: int, max_tiles: int) -> list[Any]:
    width, height = image.size
    aspect_ratio = width / max(height, 1)
    ratios = sorted(
        {
            (columns, rows)
            for tiles in range(1, max(1, max_tiles) + 1)
            for columns in range(1, tiles + 1)
            for rows in range(1, tiles + 1)
            if 1 <= columns * rows <= max(1, max_tiles)
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    columns, rows = min(
        ratios,
        key=lambda ratio: abs(aspect_ratio - ratio[0] / ratio[1]),
    )
    resized = image.resize((image_size * columns, image_size * rows))
    tiles = []
    for index in range(columns * rows):
        left = (index % columns) * image_size
        top = (index // columns) * image_size
        tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
    if len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles
