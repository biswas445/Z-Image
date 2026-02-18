"""Tests for Z-Image UI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from ui.config import UIConfig
from ui.service import GenerationRequest, ImageGenerationService


def test_ui_config_default():
    """Test default UI config."""
    config = UIConfig()
    assert config.server_host == "0.0.0.0"
    assert config.server_port == 7860
    assert config.default_height == 1024
    assert config.default_width == 1024
    assert config.default_steps == 8
    print("[PASS] test_ui_config_default")


def test_ui_config_from_env():
    """Test UI config from environment."""
    import os

    os.environ["ZIMAGE_UI_PORT"] = "8080"
    os.environ["ZIMAGE_DEFAULT_HEIGHT"] = "512"

    config = UIConfig.from_env()
    assert config.server_port == 8080
    assert config.default_height == 512

    del os.environ["ZIMAGE_UI_PORT"]
    del os.environ["ZIMAGE_DEFAULT_HEIGHT"]
    print("[PASS] test_ui_config_from_env")


def test_ui_config_output_dir():
    """Test output directory creation."""
    config = UIConfig(output_dir="test_outputs_ui")
    output_dir = config.get_output_dir()
    assert output_dir.exists()
    output_dir.rmdir()
    print("[PASS] test_ui_config_output_dir")


def test_generation_request_default():
    """Test default generation request."""
    request = GenerationRequest(prompt="test prompt")
    assert request.prompt == "test prompt"
    assert request.height == 1024
    assert request.width == 1024
    assert request.num_inference_steps == 8
    assert request.guidance_scale == 0.0
    assert request.seed == 42
    print("[PASS] test_generation_request_default")


def test_generation_request_custom():
    """Test custom generation request."""
    request = GenerationRequest(
        prompt="custom prompt",
        negative_prompt="bad quality",
        height=512,
        width=768,
        num_inference_steps=20,
        guidance_scale=7.5,
        seed=123,
        num_images=2,
    )
    assert request.prompt == "custom prompt"
    assert request.negative_prompt == "bad quality"
    assert request.height == 512
    assert request.width == 768
    assert request.num_inference_steps == 20
    assert request.guidance_scale == 7.5
    assert request.seed == 123
    assert request.num_images == 2
    print("[PASS] test_generation_request_custom")


def test_service_initialization():
    """Test service initialization without loading models."""
    config = UIConfig(model_path="test_model")
    service = ImageGenerationService(config)
    assert service._initialized is False
    assert service.config == config
    print("[PASS] test_service_initialization")


def test_service_validate_request_empty_prompt():
    """Test request validation with empty prompt."""
    config = UIConfig()
    service = ImageGenerationService(config)
    request = GenerationRequest(prompt="")
    errors = service.validate_request(request)
    assert len(errors) > 0
    assert any("empty" in err.lower() for err in errors)
    print("[PASS] test_service_validate_request_empty_prompt")


def test_service_validate_request_invalid_dimensions():
    """Test request validation with invalid dimensions."""
    config = UIConfig(min_height=256, max_height=2048, min_width=256, max_width=2048)
    service = ImageGenerationService(config)
    request = GenerationRequest(prompt="test", height=100, width=100)
    errors = service.validate_request(request)
    assert len(errors) > 0
    print("[PASS] test_service_validate_request_invalid_dimensions")


def test_service_validate_request_invalid_steps():
    """Test request validation with invalid steps."""
    config = UIConfig(min_steps=1, max_steps=50)
    service = ImageGenerationService(config)
    request = GenerationRequest(prompt="test", num_inference_steps=100)
    errors = service.validate_request(request)
    assert len(errors) > 0
    print("[PASS] test_service_validate_request_invalid_steps")


def test_service_validate_request_valid():
    """Test request validation with valid request."""
    config = UIConfig()
    service = ImageGenerationService(config)
    request = GenerationRequest(
        prompt="test prompt",
        height=1024,
        width=1024,
        num_inference_steps=8,
        guidance_scale=0.0,
    )
    errors = service.validate_request(request)
    assert len(errors) == 0
    print("[PASS] test_service_validate_request_valid")


def test_height_divisible_by_8():
    """Test height must be divisible by 8."""
    config = UIConfig()
    service = ImageGenerationService(config)
    request = GenerationRequest(prompt="test", height=1001, width=1024)
    errors = service.validate_request(request)
    assert any("divisible by 8" in err.lower() for err in errors)
    print("[PASS] test_height_divisible_by_8")


def test_width_divisible_by_8():
    """Test width must be divisible by 8."""
    config = UIConfig()
    service = ImageGenerationService(config)
    request = GenerationRequest(prompt="test", height=1024, width=1001)
    errors = service.validate_request(request)
    assert any("divisible by 8" in err.lower() for err in errors)
    print("[PASS] test_width_divisible_by_8")


def test_guidance_scale_negative():
    """Test negative guidance scale validation."""
    config = UIConfig()
    service = ImageGenerationService(config)
    request = GenerationRequest(prompt="test", guidance_scale=-1.0)
    errors = service.validate_request(request)
    assert any("negative" in err.lower() for err in errors)
    print("[PASS] test_guidance_scale_negative")


def test_num_images_range():
    """Test number of images range validation."""
    config = UIConfig()
    service = ImageGenerationService(config)

    request = GenerationRequest(prompt="test", num_images=0)
    errors = service.validate_request(request)
    assert any("number of images" in err.lower() for err in errors)

    request = GenerationRequest(prompt="test", num_images=5)
    errors = service.validate_request(request)
    assert any("number of images" in err.lower() for err in errors)
    print("[PASS] test_num_images_range")


def run_all_tests():
    """Run all tests."""
    print("Running Z-Image UI tests...\n")

    test_ui_config_default()
    test_ui_config_from_env()
    test_ui_config_output_dir()
    test_generation_request_default()
    test_generation_request_custom()
    test_service_initialization()
    test_service_validate_request_empty_prompt()
    test_service_validate_request_invalid_dimensions()
    test_service_validate_request_invalid_steps()
    test_service_validate_request_valid()
    test_height_divisible_by_8()
    test_width_divisible_by_8()
    test_guidance_scale_negative()
    test_num_images_range()

    print("\n[ALL TESTS PASSED]")


if __name__ == "__main__":
    run_all_tests()
