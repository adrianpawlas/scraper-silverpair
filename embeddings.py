"""
Embedding module for generating image and text embeddings using
google/siglip-base-patch16-384 from HuggingFace (768-dim).
"""

import io
import logging
from typing import Optional

import requests
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

import config

logger = logging.getLogger(__name__)


class SigLIPEmbedder:
    """
    Handles loading the SigLIP model and generating embeddings.
    
    - Image embeddings: 768-dim from the vision encoder
    - Text embeddings: 768-dim from the text encoder
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = self._resolve_device()

    def _resolve_device(self) -> str:
        """Determine the best available device."""
        if config.DEVICE == "mps" and torch.backends.mps.is_available():
            logger.info("Using MPS (Apple Silicon GPU)")
            return "mps"
        elif torch.cuda.is_available():
            logger.info("Using CUDA GPU")
            return "cuda"
        else:
            logger.info("Using CPU")
            return "cpu"

    def load_model(self):
        """Load the SigLIP model and processor."""
        if self.model is not None:
            return

        logger.info(f"Loading model: {config.EMBEDDING_MODEL}")
        logger.info(f"This may take a moment on first run (downloading ~600MB)")

        self.processor = AutoProcessor.from_pretrained(
            config.EMBEDDING_MODEL,
            use_fast=True,
        )
        self.model = AutoModel.from_pretrained(
            config.EMBEDDING_MODEL,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded on {self.device}")

    def download_image(self, url: str) -> Optional[Image.Image]:
        """Download an image from URL and return as PIL Image."""
        try:
            resp = requests.get(
                url,
                timeout=config.REQUEST_TIMEOUT,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to download image {url}: {e}")
            return None

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> list[float]:
        """
        Generate 768-dim embedding for a single image.
        Returns a list of floats suitable for vector DB insertion.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.get_image_features(**inputs)
        # SigLIP's get_image_features returns a BaseModelOutputWithPooling,
        # we need the pooler_output which is the actual embedding
        if hasattr(outputs, 'pooler_output'):
            embedding = outputs.pooler_output
        else:
            embedding = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        # Normalize the embedding
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten().tolist()

    @torch.no_grad()
    def embed_text(self, text: str) -> list[float]:
        """
        Generate 768-dim embedding for a text string.
        Returns a list of floats suitable for vector DB insertion.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self.processor(
            text=[text],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.get_text_features(**inputs)
        # SigLIP's get_text_features returns a BaseModelOutputWithPooling,
        # we need the pooler_output which is the actual embedding
        if hasattr(outputs, 'pooler_output'):
            embedding = outputs.pooler_output
        else:
            # If it's already a tensor, use it directly
            embedding = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        # Normalize the embedding
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten().tolist()

    def embed_image_from_url(self, url: str) -> Optional[list[float]]:
        """Download image from URL and generate its embedding."""
        image = self.download_image(url)
        if image is None:
            return None
        return self.embed_image(image)

    def cleanup(self):
        """Free up GPU/MPS memory."""
        if self.model is not None:
            self.model = None
            self.processor = None
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Model unloaded and memory cleaned")
