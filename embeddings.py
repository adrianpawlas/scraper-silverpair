"""
Embedding module for generating image and text embeddings using
google/siglip-base-patch16-384 from HuggingFace (768-dim).

Robust embedding pipeline:
  1. Forward pass through the model
  2. Parse any returned shape to a flat 768-d vector
     (handles nested / batch formats via averaging)
  3. L2-normalise: v = v / ||v||
  4. Validate: length == 768, all values finite
"""

import io
import logging
from typing import Optional

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding post-processing utilities
# ---------------------------------------------------------------------------


def _validate_embedding(arr: np.ndarray, dim: int = 768) -> np.ndarray:
    """
    Validate and normalise a raw embedding array from the model.

    1. If the array is nested (batch dimension > 1), average across
       the batch to produce a single vector.
    2. L2-normalise: v = v / ||v||.
    3. Verify length == *dim* and all values are finite.

    Returns a 1-D float64 NumPy array of length *dim*.
    Raises ``ValueError`` if validation fails.
    """
    arr = np.asarray(arr, dtype=np.float64)

    # --- Flatten to at most 2-D ---
    if arr.ndim == 0:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        # E.g. shape (1, 1, 768) → (1, 768)
        arr = arr.reshape(-1, arr.shape[-1])

    # --- Handle batch / nested shapes ---
    n_vectors, n_dims = arr.shape
    if n_vectors > 1:
        logger.debug(
            "Averaging %d embedding vectors into a single %d-d vector",
            n_vectors, n_dims,
        )
        arr = arr.mean(axis=0, keepdims=True)  # (1, dim)
    elif n_vectors == 0:
        raise ValueError("Empty embedding array — no vectors found")

    # --- L2 normalise ---
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("Zero-norm embedding — all values are zero")
    arr = arr / norm

    # --- Squeeze to 1-D ---
    arr = arr.flatten()

    # --- Validate ---
    if len(arr) != dim:
        raise ValueError(
            f"Expected {dim}-dim embedding but got {len(arr)} dims"
        )
    if not np.all(np.isfinite(arr)):
        n_nonfinite = int(np.sum(~np.isfinite(arr)))
        raise ValueError(
            f"Embedding contains {n_nonfinite} non-finite value(s) "
            f"(NaN / Inf)"
        )

    return arr


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
        Validates the embedding via ``_validate_embedding`` before returning.
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

        embedding_np = embedding.cpu().numpy()
        # Use the robust validation / normalisation pipeline
        embedding_np = _validate_embedding(embedding_np, dim=config.EMBEDDING_DIM)
        return embedding_np.tolist()

    @torch.no_grad()
    def embed_text(self, text: str) -> list[float]:
        """
        Generate 768-dim embedding for a text string.

        Returns a list of floats suitable for vector DB insertion.
        Validates the embedding via ``_validate_embedding`` before returning.
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

        embedding_np = embedding.cpu().numpy()
        # Use the robust validation / normalisation pipeline
        embedding_np = _validate_embedding(embedding_np, dim=config.EMBEDDING_DIM)
        return embedding_np.tolist()

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
