"""Image generation model wrappers."""
import os
import base64
import io
import sys
from pathlib import Path
import requests
from PIL import Image
from dotenv import load_dotenv

from ..paths import runtime_root
from ..reporting import detail_print
from .base import BaseGenerativeModel
from .openai_compatible import (
    bearer_headers,
    post_with_proxy_fallback,
    resolve_openai_compatible_config,
)

load_dotenv()

def _env_path(name: str) -> Path | None:
    value = (os.getenv(name) or "").strip()
    if not value:
        return None
    return Path(value).expanduser()


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


PROJECT_ROOT = runtime_root()


def _local_models_root() -> Path:
    return _env_path("PROVISE_LOCAL_MODELS_DIR") or (PROJECT_ROOT / "local_models")


def _resolve_repo_path(repo_name: str, env_name: str) -> str:
    env_path = _env_path(env_name)
    if env_path is not None:
        return str(env_path)
    candidates = [
        PROJECT_ROOT.parent / repo_name,
        Path.home() / "projects" / repo_name,
    ]
    return str(_first_existing_path(candidates) or candidates[0])


def _resolve_local_model_path(
    model_dir_name: str,
    env_name: str,
    extra_candidates: list[Path] | None = None,
) -> str:
    env_path = _env_path(env_name)
    if env_path is not None:
        return str(env_path)
    candidates = [_local_models_root() / model_dir_name]
    if extra_candidates:
        candidates.extend(extra_candidates)
    return str(_first_existing_path(candidates) or candidates[0])

MODELS = {
    "gemini-3.1-flash": {
        "name": "google/gemini-3.1-flash-image-preview",
        "backend": "openai_compatible",
        "modalities": ["image", "text"]
    },
    "nanobanana2": {
        "name": "google/gemini-3.1-flash-image-preview",
        "backend": "openai_compatible",
        "modalities": ["image", "text"],
        "provider": "openrouter",
    },
    "gemini-2.5-flash": {
        "name": "google/gemini-2.5-flash-image",
        "backend": "openai_compatible",
        "modalities": ["image", "text"]
    },
    "gemini-3-pro-image": {
        "name": "google/gemini-3-pro-image-preview",
        "backend": "openai_compatible",
        "modalities": ["image", "text"]
    },
    "seedream": {
        "name": "bytedance-seed/seedream-4.5",
        "backend": "openai_compatible",
        "modalities": ["image"],
        "provider": "openrouter",
    },
    "gpt5-image-mini": {
        "name": "openai/gpt-5-image-mini",
        "backend": "openai_compatible",
        "modalities": ["image", "text"],
        "provider": "openrouter",
    },
    "gpt5-image": {
        "name": "openai/gpt-5-image",
        "backend": "openai_compatible",
        "modalities": ["image", "text"]
    },
    "gpt5.4-image-2": {
        "name": "openai/gpt-5.4-image-2",
        "backend": "openai_compatible",
        "modalities": ["image", "text"]
    },
    "gpt-image-2": {
        "name": "gpt-image-2",
        "backend": "openai_compatible",
        "modalities": ["image", "text"],
    },
    "qwen-image-edit-2511": {
        "name": "Qwen/Qwen-Image-Edit-2511",
        "backend": "local_qwen_image_edit",
        "model_dir_name": "Qwen-Image-Edit-2511",
        "model_path_env": "PROVISE_QWEN_IMAGE_EDIT_MODEL_PATH",
    },
    "joyai-image": {
        "name": "JoyAI-Image-Edit",
        "backend": "local_joyai_image_edit",
        "model_dir_name": "JoyAI-Image-Edit",
        "model_path_env": "PROVISE_JOYAI_MODEL_PATH",
        "repo_name": "JoyAI-Image",
        "repo_path_env": "PROVISE_JOYAI_REPO_PATH",
    },
    "janus-pro-7b": {
        "name": "Janus-Pro-7B",
        "backend": "local_janus_text_to_image",
        "model_dir_name": "Janus-Pro-7B",
        "model_path_env": "PROVISE_JANUS_MODEL_PATH",
        "repo_name": "Janus",
        "repo_path_env": "PROVISE_JANUS_REPO_PATH",
        "repo_model_subdir": "deepseek-ai/Janus-Pro-7B",
    }
}


class OpenAICompatibleImageGenerationModel(BaseGenerativeModel):
    def __init__(
        self,
        model_name: str,
        modalities: list = None,
        timeout: int = 60,
        provider: str | None = None,
        max_tokens: int = 1024,
    ):
        super().__init__(model_name=model_name)
        self.modalities = modalities or ["image", "text"]
        self.timeout = timeout
        self.provider = provider
        self.max_tokens = int(os.getenv("PROVISE_IMAGE_MAX_TOKENS", max_tokens))
        self.api_key = None
        self.api_base = None

    def load_model(self):
        config = resolve_openai_compatible_config(self.provider)
        self.api_key, self.api_base = config.api_key, config.api_base
        if not self.api_key:
            raise ValueError(
                "No OpenAI-compatible API key found. Set PROVISE_API_KEY, "
                "OPENAI_API_KEY, or OPENROUTER_API_KEY."
            )
        if not self.api_base:
            raise ValueError(
                "No OpenAI-compatible API base found. Set PROVISE_API_BASE or "
                "OPENAI_BASE_URL."
            )
        self.api_base = self.api_base.rstrip("/")
        detail_print(f"Model initialized: {self.model_name}")

    def _post_with_proxy_fallback(self, payload: dict):
        return post_with_proxy_fallback(
            f"{self.api_base}/chat/completions",
            headers=bearer_headers(self.api_key),
            json=payload,
            timeout=self.timeout,
            on_proxy_fallback=lambda: detail_print(
                "Proxy unavailable, retrying without proxy..."
            ),
        )

    def generate(self, image_path: str, prompt: str, save_path: str) -> bool:
        return self.generate_multi([image_path], prompt, save_path)

    @staticmethod
    def _mime_type(image_path: str) -> str:
        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".png":
            return "image/png"
        if ext == ".webp":
            return "image/webp"
        return "image/jpeg"

    @staticmethod
    def _closest_aspect_ratio(size: tuple[int, int]) -> str:
        width, height = size
        if width <= 0 or height <= 0:
            return "1:1"
        ratios = {
            "1:1": 1.0,
            "4:3": 4 / 3,
            "3:4": 3 / 4,
            "16:9": 16 / 9,
            "9:16": 9 / 16,
            "3:2": 3 / 2,
            "2:3": 2 / 3,
        }
        actual = width / height
        return min(ratios, key=lambda key: abs(ratios[key] - actual))

    @staticmethod
    def _edit_system_prompt(image_count: int) -> str:
        if image_count == 1:
            return (
                "You are editing the provided source image, not creating a brand-new scene. "
                "Preserve the original composition, viewpoint, framing, lighting, object positions, and aspect ratio. "
                "Make only the minimal visual changes explicitly required by the user prompt. "
                "Return only the edited image."
            )
        return (
            "Use the provided input images as references. "
            "Return only a single output image that follows the user prompt."
        )

    def generate_multi(self, image_paths: list, prompt: str, save_path: str) -> bool:
        if self.api_key is None:
            self.load_model()
        self.clear_last_error()

        if self.model_name == "gpt-image-2":
            return self._generate_with_image_endpoint(image_paths, prompt, save_path)

        content = [{"type": "text", "text": prompt}]
        source_size = None
        for image_path in image_paths:
            if not os.path.exists(image_path):
                detail_print(f"Input image does not exist: {image_path}")
                self.record_last_error("input_missing", f"Input image does not exist: {image_path}")
                return False
            if source_size is None:
                with Image.open(image_path) as src_img:
                    source_size = src_img.size
            with open(image_path, 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode('utf-8')
            mime = self._mime_type(image_path)
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}})

        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self._edit_system_prompt(len(image_paths))},
                    {"role": "user", "content": content},
                ],
                "modalities": self.modalities,
                "max_tokens": self.max_tokens,
            }
            if source_size is not None:
                payload["image_config"] = {
                    "aspect_ratio": self._closest_aspect_ratio(source_size),
                    "image_size": "1K",
                }

            response = self._post_with_proxy_fallback(payload)

            if response.status_code != 200:
                detail_print(f"Generation API error {response.status_code}: {response.text[:200]}")
                self.record_last_error("api_http_error", f"HTTP {response.status_code}: {response.text[:200]}")
                return False

            result = response.json()
            message = result.get("choices", [{}])[0].get("message", {})

            if not message.get("images"):
                detail_print("Generation response did not include an image.")
                if message.get("content"):
                    detail_print(f"   text: {message['content'][:300]}")
                self.record_last_error("empty_image_response", "Generation response did not include an image.")
                return False

            image_url = message["images"][0]["image_url"]["url"]
            b64_data = image_url.split(",")[1] if "," in image_url else image_url
            img = Image.open(io.BytesIO(base64.b64decode(b64_data)))
            if source_size and len(image_paths) == 1 and img.size != source_size:
                img = img.resize(source_size, Image.Resampling.LANCZOS)

            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            ext = os.path.splitext(save_path)[1].lower()
            if ext == ".png":
                img.save(save_path, format="PNG")
            elif ext == ".webp":
                img.save(save_path, format="WEBP", lossless=True)
            else:
                img.save(save_path, quality=100, subsampling=0)
            self.clear_last_error()
            return True

        except requests.exceptions.Timeout:
            detail_print("Generation request timed out.")
            self.record_last_error("timeout", "Generation request timed out.")
            return False
        except Exception as e:
            detail_print(f"Generation failed: {e}")
            self.record_last_error(type(e).__name__, str(e))
            return False


    def _generate_with_image_endpoint(self, image_paths: list, prompt: str, save_path: str) -> bool:
        valid_paths = []
        source_size = None
        for image_path in image_paths:
            if not os.path.exists(image_path):
                detail_print(f"Input image does not exist: {image_path}")
                self.record_last_error("input_missing", f"Input image does not exist: {image_path}")
                return False
            valid_paths.append(image_path)
            if source_size is None:
                with Image.open(image_path) as src_img:
                    source_size = src_img.size

        try:
            session = requests.Session()
            session.trust_env = False
            size = os.getenv("PROVISE_IMAGE_SIZE", "1024x1024")
            timeout = int(os.getenv("PROVISE_IMAGE_TIMEOUT", str(self.timeout)))
            if valid_paths:
                files = []
                handles = []
                try:
                    for path in valid_paths:
                        handle = open(path, "rb")
                        handles.append(handle)
                        files.append(("image", (os.path.basename(path), handle, self._mime_type(path))))
                    response = session.post(
                        f"{self.api_base}/images/edits",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        data={"model": self.model_name, "prompt": prompt, "size": size, "n": "1"},
                        files=files,
                        timeout=timeout,
                    )
                finally:
                    for handle in handles:
                        handle.close()
            else:
                response = session.post(
                    f"{self.api_base}/images/generations",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={"model": self.model_name, "prompt": prompt, "size": size, "n": 1},
                    timeout=timeout,
                )

            if response.status_code != 200:
                detail_print(f"Image API error {response.status_code}: {response.text[:300]}")
                self.record_last_error("api_http_error", f"HTTP {response.status_code}: {response.text[:300]}")
                return False

            result = response.json()
            data = result.get("data") or []
            b64_data = data[0].get("b64_json") if data and isinstance(data[0], dict) else None
            if not b64_data:
                detail_print(
                    f"Image API response did not include b64_json: {str(result)[:300]}"
                )
                self.record_last_error("empty_image_response", "Image API response did not include b64_json.")
                return False

            img = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
            if source_size and len(valid_paths) == 1 and img.size != source_size:
                img = img.resize(source_size, Image.Resampling.LANCZOS)
                detail_print(f"Aligned output image back to source size: {source_size}")

            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            ext = os.path.splitext(save_path)[1].lower()
            if ext == ".png":
                img.save(save_path, format="PNG")
            elif ext == ".webp":
                img.save(save_path, format="WEBP", lossless=True)
            else:
                img.save(save_path, quality=100, subsampling=0)
            detail_print(
                f"Saved image: {save_path} {img.size}; returned_model={result.get('model')}"
            )
            self.clear_last_error()
            return True
        except requests.exceptions.Timeout:
            detail_print("Image API request timed out")
            self.record_last_error("timeout", "Image API request timed out")
            return False
        except Exception as exc:
            detail_print(f"Image API generation failed: {exc}")
            self.record_last_error(type(exc).__name__, str(exc))
            return False


class LocalQwenImageEditModel(BaseGenerativeModel):
    def __init__(self, model_path: str, timeout: int = 60):
        super().__init__(model_name=model_path)
        self.model_path = model_path
        self.timeout = timeout
        self.pipeline = None
        self.torch = None

    def load_model(self):
        if self.pipeline is not None:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "Qwen-Image-Edit model directory does not exist: "
                f"{self.model_path}. Set PROVISE_QWEN_IMAGE_EDIT_MODEL_PATH "
                "or PROVISE_LOCAL_MODELS_DIR to override."
            )
        try:
            import torch
            from modelscope import QwenImageEditPlusPipeline
        except ImportError as e:
            raise ImportError(
                "Qwen-Image-Edit-2511 dependencies are missing. Use an environment with modelscope installed."
            ) from e

        self.torch = torch
        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline.to(device)
        if hasattr(self.pipeline, "set_progress_bar_config"):
            self.pipeline.set_progress_bar_config(disable=True)
        detail_print(f"Model initialized: {self.model_path}")

    def _load_images(self, image_paths: list):
        pil_images = []
        first_size = None
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Input image does not exist: {path}")
            img = Image.open(path).convert("RGB")
            if first_size is None:
                first_size = img.size
            pil_images.append(img)
        return pil_images, first_size

    def _save_output(self, output_image: Image.Image, save_path: str, target_size=None):
        if target_size and output_image.size != target_size:
            output_image = output_image.resize(target_size, Image.Resampling.LANCZOS)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        if ext == ".png":
            output_image.save(save_path, format="PNG")
        elif ext == ".webp":
            output_image.save(save_path, format="WEBP", lossless=True)
        else:
            output_image.save(save_path, quality=100, subsampling=0)

    def generate(self, image_path: str, prompt: str, save_path: str) -> bool:
        return self.generate_multi([image_path], prompt, save_path)

    def generate_multi(self, image_paths: list, prompt: str, save_path: str) -> bool:
        if self.pipeline is None:
            self.load_model()
        self.clear_last_error()
        try:
            pil_images, first_size = self._load_images(image_paths)
            inputs = {
                "image": pil_images,
                "prompt": prompt,
                "generator": self.torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 40,
                "guidance_scale": 1.0,
                "num_images_per_prompt": 1,
            }
            if first_size:
                width, height = first_size
                inputs["width"] = width
                inputs["height"] = height
            with self.torch.inference_mode():
                output = self.pipeline(**inputs)
            output_image = output.images[0]
            target_size = first_size if len(image_paths) == 1 else None
            self._save_output(output_image, save_path, target_size=target_size)
            self.clear_last_error()
            return True
        except FileNotFoundError as e:
            detail_print(f"Generation failed: {e}")
            self.record_last_error("input_missing", str(e))
            return False
        except Exception as e:
            detail_print(f"Generation failed: {e}")
            self.record_last_error(type(e).__name__, str(e))
            return False


class LocalJoyAIImageEditModel(BaseGenerativeModel):
    def __init__(
        self,
        model_path: str,
        repo_path: str = "",
        timeout: int = 60,
    ):
        super().__init__(model_name=model_path)
        self.model_path = model_path
        self.repo_path = repo_path or str(PROJECT_ROOT.parent / "JoyAI-Image")
        self.timeout = timeout
        self.model = None
        self.InferenceParams = None
        self.torch = None
        self.device = None

    def load_model(self):
        if self.model is not None:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "JoyAI-Image checkpoint directory does not exist: "
                f"{self.model_path}. Set PROVISE_JOYAI_MODEL_PATH "
                "or PROVISE_LOCAL_MODELS_DIR to override."
            )

        src_dir = Path(self.repo_path) / "src"
        if not src_dir.exists():
            raise FileNotFoundError(
                "JoyAI-Image source directory does not exist: "
                f"{src_dir}. Set PROVISE_JOYAI_REPO_PATH to override."
            )
        deps_dir = PROJECT_ROOT / ".deps" / "joyai"
        if deps_dir.exists() and str(deps_dir) not in sys.path:
            sys.path.insert(0, str(deps_dir))
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        try:
            import torch
            from infer_runtime.model import InferenceParams, build_model
            from infer_runtime.settings import load_settings
            from modules.models.attention import (
                describe_attention_backend,
                is_flash_attn_available,
            )
        except ImportError as e:
            raise ImportError(
                "JoyAI-Image dependencies are missing. Expected transformers>=4.57,<4.58, "
                "diffusers==0.36.0, einops, sentencepiece, and related packages."
            ) from e

        if torch.cuda.is_available():
            cuda_device = int(os.environ.get("JOYAI_CUDA_DEVICE", os.environ.get("LOCAL_RANK", "0")))
            torch.cuda.set_device(cuda_device)
            device = torch.device(f"cuda:{cuda_device}")
        else:
            device = torch.device("cpu")

        settings = load_settings(
            ckpt_root=self.model_path,
            config_path=os.getenv("JOYAI_CONFIG"),
            rewrite_model=os.getenv("JOYAI_REWRITE_MODEL", "gpt-5"),
            default_seed=int(os.getenv("JOYAI_SEED", "42")),
        )
        if not is_flash_attn_available() and not os.getenv("JOYAI_ATTN_BACKEND"):
            os.environ["JOYAI_ATTN_BACKEND"] = "torch_sdpa"
        self.model = build_model(
            settings,
            device=device,
            hsdp_shard_dim_override=(
                int(os.environ["JOYAI_HSDP_SHARD_DIM"])
                if os.getenv("JOYAI_HSDP_SHARD_DIM")
                else None
            ),
        )
        self.InferenceParams = InferenceParams
        self.torch = torch
        self.device = device
        detail_print(f"Model initialized: {self.model_path}")
        detail_print(f"   device={device}, attention={describe_attention_backend()}")

    def _load_image(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image does not exist: {image_path}")
        return Image.open(image_path).convert("RGB")

    def _compose_multi_image_grid(self, image_paths: list[str]) -> Image.Image:
        from PIL import ImageDraw, ImageFont
        import math

        images = [self._load_image(path) for path in image_paths]
        if len(images) == 1:
            return images[0]

        thumb_w = int(os.getenv("JOYAI_MULTI_THUMB_WIDTH", "512"))
        label_h = 34
        resized = []
        for idx, img in enumerate(images, 1):
            scale = thumb_w / max(1, img.width)
            thumb_h = max(1, int(round(img.height * scale)))
            tile = Image.new("RGB", (thumb_w, thumb_h + label_h), "white")
            tile.paste(img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (0, label_h))
            draw = ImageDraw.Draw(tile)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            draw.text((10, 9), f"Image {idx}", fill=(0, 0, 0), font=font)
            resized.append(tile)

        cols = min(2, len(resized))
        rows = math.ceil(len(resized) / cols)
        cell_h = max(tile.height for tile in resized)
        canvas = Image.new("RGB", (cols * thumb_w, rows * cell_h), "white")
        for idx, tile in enumerate(resized):
            row, col = divmod(idx, cols)
            canvas.paste(tile, (col * thumb_w, row * cell_h))
        return canvas

    def _save_output(self, output_image: Image.Image, save_path: str, target_size=None):
        if target_size and output_image.size != target_size:
            output_image = output_image.resize(target_size, Image.Resampling.LANCZOS)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        if ext == ".png":
            output_image.save(save_path, format="PNG")
        elif ext == ".webp":
            output_image.save(save_path, format="WEBP", lossless=True)
        else:
            output_image.save(save_path, quality=100, subsampling=0)

    def generate(self, image_path: str, prompt: str, save_path: str) -> bool:
        return self.generate_multi([image_path], prompt, save_path)

    def generate_multi(self, image_paths: list, prompt: str, save_path: str) -> bool:
        if self.model is None:
            self.load_model()
        self.clear_last_error()
        try:
            if not image_paths:
                raise ValueError("JoyAI-Image requires at least one input image for editing.")
            input_image = (
                self._load_image(image_paths[0])
                if len(image_paths) == 1
                else self._compose_multi_image_grid(image_paths)
            )
            source_size = input_image.size

            steps = int(os.getenv("JOYAI_STEPS", "50"))
            guidance_scale = float(os.getenv("JOYAI_GUIDANCE_SCALE", "5.0"))
            seed = int(os.getenv("JOYAI_SEED", "42"))
            basesize = int(os.getenv("JOYAI_BASESIZE", "1024"))
            neg_prompt = os.getenv("JOYAI_NEG_PROMPT", "")

            effective_prompt = self.model.maybe_rewrite_prompt(
                prompt,
                input_image,
                os.getenv("JOYAI_REWRITE_PROMPT", "").strip().lower()
                in {"1", "true", "yes", "on"},
            )
            output_image = self.model.infer(
                self.InferenceParams(
                    prompt=effective_prompt,
                    image=input_image,
                    height=source_size[1],
                    width=source_size[0],
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    neg_prompt=neg_prompt,
                    basesize=basesize,
                )
            )
            target_size = source_size if len(image_paths) == 1 else None
            self._save_output(output_image, save_path, target_size=target_size)
            self.clear_last_error()
            return True
        except FileNotFoundError as e:
            detail_print(f"Generation failed: {e}")
            self.record_last_error("input_missing", str(e))
            return False
        except Exception as e:
            detail_print(f"Generation failed: {e}")
            self.record_last_error(type(e).__name__, str(e))
            return False


class LocalJanusProTextToImageModel(BaseGenerativeModel):
    """Janus-Pro text-to-image backend.

    Janus-Pro-7B supports multimodal understanding and text-to-image generation,
    but not image-conditioned editing. For image-conditioned benchmark tasks this
    backend therefore ignores input images and uses only the prompt to generate a
    visual answer image. It is useful as a runnable generative baseline, but it is
    not directly comparable to image-edit models such as JoyAI.
    """

    def __init__(
        self,
        model_path: str,
        repo_path: str = "",
        timeout: int = 60,
    ):
        super().__init__(model_name=model_path)
        self.model_path = model_path
        self.repo_path = repo_path or str(PROJECT_ROOT.parent / "Janus")
        self.timeout = timeout
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.torch = None
        self.np = None
        self.device = None

    def load_model(self):
        if self.model is not None:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "Janus-Pro-7B model directory does not exist: "
                f"{self.model_path}. Set PROVISE_JANUS_MODEL_PATH "
                "or PROVISE_LOCAL_MODELS_DIR to override."
            )
        if not os.path.isdir(self.repo_path):
            raise FileNotFoundError(
                "Janus project directory does not exist: "
                f"{self.repo_path}. Set PROVISE_JANUS_REPO_PATH to override."
            )
        compat_dir = PROJECT_ROOT / "third_party" / "janus_compat"
        if compat_dir.is_dir() and str(compat_dir) not in sys.path:
            sys.path.insert(0, str(compat_dir))
        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)

        import torch
        import numpy as np
        from transformers import AutoConfig, AutoModelForCausalLM
        from janus.models import VLChatProcessor

        if torch.cuda.is_available():
            cuda_device = int(os.environ.get("JANUS_CUDA_DEVICE", os.environ.get("LOCAL_RANK", "0")))
            torch.cuda.set_device(cuda_device)
            device = torch.device(f"cuda:{cuda_device}")
        else:
            device = torch.device("cpu")

        config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        language_config = config.language_config
        language_config._attn_implementation = os.getenv("JANUS_ATTN_IMPL", "eager")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            language_config=language_config,
            trust_remote_code=True,
            local_files_only=True,
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        self.model = self.model.to(dtype).to(device).eval()
        self.processor = VLChatProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.tokenizer = self.processor.tokenizer
        self.torch = torch
        self.np = np
        self.device = device
        detail_print(f"Model initialized: {self.model_path}")
        detail_print(f"   device={device}, mode=text-to-image, edit_input=ignored")

    def _source_size(self, image_paths: list):
        if not image_paths:
            return None
        try:
            with Image.open(image_paths[0]) as img:
                return img.size
        except Exception:
            return None

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _build_prompt_ids(self, prompt: str):
        messages = [
            {"role": "<|User|>", "content": prompt},
            {"role": "<|Assistant|>", "content": ""},
        ]
        text = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=messages,
            sft_format=self.processor.sft_format,
            system_prompt="",
        )
        text = text + self.processor.image_start_tag
        return self.torch.LongTensor(self.tokenizer.encode(text))

    @staticmethod
    def _save_output(output_image: Image.Image, save_path: str, target_size=None):
        if target_size and output_image.size != target_size:
            output_image = output_image.resize(target_size, Image.Resampling.LANCZOS)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        if ext == ".png":
            output_image.save(save_path, format="PNG")
        elif ext == ".webp":
            output_image.save(save_path, format="WEBP", lossless=True)
        else:
            output_image.save(save_path, quality=100, subsampling=0)

    def _generate_image(self, prompt: str) -> Image.Image:
        torch = self.torch
        np = self.np
        seed = int(os.getenv("JANUS_SEED", "42"))
        guidance = float(os.getenv("JANUS_GUIDANCE", "5.0"))
        temperature = float(os.getenv("JANUS_TEMPERATURE", "1.0"))
        parallel_size = int(os.getenv("JANUS_PARALLEL_SIZE", "1"))
        img_size = int(os.getenv("JANUS_IMAGE_SIZE", "384"))
        patch_size = int(os.getenv("JANUS_PATCH_SIZE", "16"))
        image_token_num = int(
            os.getenv("JANUS_IMAGE_TOKENS", str((img_size // patch_size) ** 2))
        )

        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)

        input_ids = self._build_prompt_ids(prompt).to(self.device)
        tokens = torch.zeros(
            (parallel_size * 2, len(input_ids)),
            dtype=torch.int,
            device=self.device,
        )
        for idx in range(parallel_size * 2):
            tokens[idx, :] = input_ids
            if idx % 2 != 0:
                tokens[idx, 1:-1] = self.processor.pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros(
            (parallel_size, image_token_num),
            dtype=torch.int,
            device=self.device,
        )

        past_key_values = None
        for idx in range(image_token_num):
            outputs = self.model.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state

            logits = self.model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + guidance * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, idx] = next_token.squeeze(dim=-1)

            next_token = torch.cat(
                [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)],
                dim=1,
            ).view(-1)
            img_embeds = self.model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        decoded = self.model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
        )
        decoded = decoded.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        decoded = np.clip((decoded + 1) / 2 * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(decoded[0])

    def generate(self, image_path: str, prompt: str, save_path: str) -> bool:
        return self.generate_multi([image_path], prompt, save_path)

    def generate_multi(self, image_paths: list, prompt: str, save_path: str) -> bool:
        if self.model is None:
            self.load_model()
        self.clear_last_error()
        try:
            target_size = (
                self._source_size(image_paths)
                if self._env_bool("JANUS_RESIZE_TO_SOURCE", True)
                else None
            )
            with self.torch.inference_mode():
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
                output_image = self._generate_image(prompt)
            self._save_output(output_image, save_path, target_size=target_size)
            self.clear_last_error()
            return True
        except Exception as e:
            detail_print(f"Generation failed: {e}")
            self.record_last_error(type(e).__name__, str(e))
            return False
        finally:
            if self.torch is not None and self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()


def create_model(model_key: str, timeout: int = 60) -> BaseGenerativeModel:
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
    cfg = MODELS[model_key]
    backend = cfg.get("backend", "openai_compatible")
    if backend == "openai_compatible":
        return OpenAICompatibleImageGenerationModel(
            model_name=cfg["name"],
            modalities=cfg["modalities"],
            timeout=timeout,
            provider=cfg.get("provider"),
        )
    if backend == "local_qwen_image_edit":
        model_path = _resolve_local_model_path(
            cfg["model_dir_name"],
            cfg["model_path_env"],
        )
        return LocalQwenImageEditModel(model_path=model_path, timeout=timeout)
    if backend == "local_joyai_image_edit":
        repo_path = _resolve_repo_path(cfg["repo_name"], cfg["repo_path_env"])
        model_path = _resolve_local_model_path(
            cfg["model_dir_name"],
            cfg["model_path_env"],
        )
        return LocalJoyAIImageEditModel(
            model_path=model_path,
            repo_path=repo_path,
            timeout=timeout,
        )
    if backend == "local_janus_text_to_image":
        repo_path = _resolve_repo_path(cfg["repo_name"], cfg["repo_path_env"])
        extra_candidates = []
        repo_model_subdir = cfg.get("repo_model_subdir")
        if repo_model_subdir:
            extra_candidates.append(Path(repo_path) / repo_model_subdir)
        model_path = _resolve_local_model_path(
            cfg["model_dir_name"],
            cfg["model_path_env"],
            extra_candidates=extra_candidates,
        )
        return LocalJanusProTextToImageModel(
            model_path=model_path,
            repo_path=repo_path,
            timeout=timeout,
        )
    raise ValueError(f"Unknown model backend: {backend}")
