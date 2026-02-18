"""Image Generation Service for Z-Image UI."""

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import torch
from loguru import logger

from config.inference import (
    DEFAULT_CFG_TRUNCATION,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_WIDTH,
)
from utils import AttentionBackend, ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate


@dataclass
class GenerationRequest:
    """Request for image generation."""

    prompt: str
    negative_prompt: str = ""
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    num_inference_steps: int = DEFAULT_INFERENCE_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    seed: int = 42
    num_images: int = 1
    cfg_normalization: bool = False
    cfg_truncation: float = DEFAULT_CFG_TRUNCATION
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
    output_type: str = "pil"


@dataclass
class GenerationResult:
    """Result of image generation."""

    images: list
    seed: int
    generation_time: float
    output_paths: list[Path]
    metadata: dict[str, Any]


class ImageGenerationService:
    """Service for managing Z-Image model and generating images."""

    def __init__(self, config):
        self.config = config
        self.model_path = None
        self.components = None
        self.device = None
        self.dtype = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the model and components."""
        if self._initialized:
            return

        logger.info("Initializing Z-Image service")

        self.model_path = ensure_model_weights(self.config.model_path, verify=False)

        if self.config.dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif self.config.dtype == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.device = self._select_device()

        logger.info(f"Loading models to {self.device} with dtype {self.dtype}")
        self.components = load_from_local_dir(
            self.model_path,
            device=self.device,
            dtype=self.dtype,
            compile=self.config.compile,
        )

        AttentionBackend.print_available_backends()
        set_attention_backend(self.config.attention_backend)
        logger.info(f"Attention backend set to: {self.config.attention_backend}")

        self._initialized = True
        logger.success("Z-Image service initialized successfully")

    def _select_device(self) -> str | torch.device:
        """Select the best available device."""
        if torch.cuda.is_available():
            logger.info("Using CUDA device")
            return "cuda"

        try:
            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
            logger.info(f"Using TPU device: {device}")
            return device
        except (ImportError, RuntimeError):
            pass

        if torch.backends.mps.is_available():
            logger.info("Using MPS device")
            return "mps"

        logger.info("Using CPU device")
        return "cpu"

    def validate_request(self, request: GenerationRequest) -> list[str]:
        """Validate generation request and return list of errors."""
        errors = []

        if not request.prompt or not request.prompt.strip():
            errors.append("Prompt cannot be empty")

        if request.height < self.config.min_height or request.height > self.config.max_height:
            errors.append(f"Height must be between {self.config.min_height} and {self.config.max_height}")

        if request.width < self.config.min_width or request.width > self.config.max_width:
            errors.append(f"Width must be between {self.config.min_width} and {self.config.max_width}")

        if request.height % 8 != 0:
            errors.append("Height must be divisible by 8")

        if request.width % 8 != 0:
            errors.append("Width must be divisible by 8")

        if request.num_inference_steps < self.config.min_steps or request.num_inference_steps > self.config.max_steps:
            errors.append(f"Steps must be between {self.config.min_steps} and {self.config.max_steps}")

        if request.guidance_scale < 0:
            errors.append("Guidance scale cannot be negative")

        if request.num_images < 1 or request.num_images > 4:
            errors.append("Number of images must be between 1 and 4")

        return errors

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate images from prompt."""
        if not self._initialized:
            self.initialize()

        errors = self.validate_request(request)
        if errors:
            raise ValueError(f"Invalid request: {'; '.join(errors)}")

        output_dir = self.config.get_output_dir()
        generation_id = uuid.uuid4().hex[:8]
        output_paths = []

        logger.info(f"Starting generation: {request.prompt[:50]}...")
        start_time = time.time()

        generator = torch.Generator(self.device).manual_seed(request.seed)

        images = generate(
            prompt=request.prompt,
            **self.components,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            negative_prompt=request.negative_prompt if request.guidance_scale > 1.0 else None,
            num_images_per_prompt=request.num_images,
            generator=generator,
            cfg_normalization=request.cfg_normalization,
            cfg_truncation=request.cfg_truncation,
            max_sequence_length=request.max_sequence_length,
            output_type=request.output_type,
        )

        for idx, image in enumerate(images):
            timestamp = int(time.time() * 1000)
            filename = f"zimage_{generation_id}_{idx}_{timestamp}.png"
            output_path = output_dir / filename
            image.save(output_path)
            output_paths.append(output_path)

        generation_time = time.time() - start_time

        metadata = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "height": request.height,
            "width": request.width,
            "steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "seed": request.seed,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "attention_backend": self.config.attention_backend,
        }

        logger.success(f"Generated {len(images)} image(s) in {generation_time:.2f}s")

        return GenerationResult(
            images=images,
            seed=request.seed,
            generation_time=generation_time,
            output_paths=output_paths,
            metadata=metadata,
        )

    def generate_with_progress(self, request: GenerationRequest) -> Generator[tuple[str, Any], None, GenerationResult]:
        """Generate images with progress updates."""
        yield "status", "Initializing model..."

        if not self._initialized:
            self.initialize()

        errors = self.validate_request(request)
        if errors:
            raise ValueError(f"Invalid request: {'; '.join(errors)}")

        yield "status", "Starting generation..."

        output_dir = self.config.get_output_dir()
        generation_id = uuid.uuid4().hex[:8]
        output_paths = []

        start_time = time.time()
        generator = torch.Generator(self.device).manual_seed(request.seed)

        logger.info(f"Starting generation: {request.prompt[:50]}...")

        images = generate(
            prompt=request.prompt,
            **self.components,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            negative_prompt=request.negative_prompt if request.guidance_scale > 1.0 else None,
            num_images_per_prompt=request.num_images,
            generator=generator,
            cfg_normalization=request.cfg_normalization,
            cfg_truncation=request.cfg_truncation,
            max_sequence_length=request.max_sequence_length,
            output_type=request.output_type,
        )

        yield "status", "Saving images..."

        for idx, image in enumerate(images):
            timestamp = int(time.time() * 1000)
            filename = f"zimage_{generation_id}_{idx}_{timestamp}.png"
            output_path = output_dir / filename
            image.save(output_path)
            output_paths.append(output_path)
            yield "progress", (idx + 1) / len(images)

        generation_time = time.time() - start_time

        metadata = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "height": request.height,
            "width": request.width,
            "steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "seed": request.seed,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "attention_backend": self.config.attention_backend,
        }

        result = GenerationResult(
            images=images,
            seed=request.seed,
            generation_time=generation_time,
            output_paths=output_paths,
            metadata=metadata,
        )

        yield "status", f"Completed in {generation_time:.2f}s"
        yield "result", result
