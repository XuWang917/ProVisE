from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseModel(ABC):
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}
        self.last_error_type: str = ""
        self.last_error_message: str = ""

    def clear_last_error(self) -> None:
        self.last_error_type = ""
        self.last_error_message = ""

    def record_last_error(self, error_type: str, message: str = "") -> None:
        self.last_error_type = str(error_type or "").strip()
        self.last_error_message = str(message or "").strip()

    @abstractmethod
    def load_model(self):
        pass


class BaseGenerativeModel(BaseModel):
    """Image generation interface: image plus prompt to image."""

    @abstractmethod
    def generate(self, image_path: str, prompt: str, save_path: str) -> bool:
        pass


class BaseVLM(BaseModel):
    """Vision-language interface: image plus question to text."""

    @abstractmethod
    def predict(self, image_path: str, question: str) -> str:
        pass
