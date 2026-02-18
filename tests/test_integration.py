"""Integration tests for Z-Image UI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ui.config import UIConfig
from ui.service import GenerationRequest


def test_ui_config_integration():
    """Test UI config integration."""
    config = UIConfig.from_env()
    assert config.server_port > 0
    assert config.model_path
    print("[PASS] test_ui_config_integration")


def test_generation_request_integration():
    """Test generation request integration."""
    request = GenerationRequest(
        prompt="A beautiful sunset over mountains",
        negative_prompt="blurry, low quality",
        height=512,
        width=512,
        num_inference_steps=8,
        guidance_scale=0.0,
        seed=42,
        num_images=1,
    )
    assert request.prompt
    assert request.height % 8 == 0
    assert request.width % 8 == 0
    print("[PASS] test_generation_request_integration")


def test_service_validation_integration():
    """Test service validation integration."""
    from ui.service import ImageGenerationService

    config = UIConfig()
    service = ImageGenerationService(config)

    valid_request = GenerationRequest(
        prompt="test prompt",
        height=512,
        width=512,
        num_inference_steps=8,
        guidance_scale=0.0,
        seed=42,
    )

    errors = service.validate_request(valid_request)
    assert len(errors) == 0
    print("[PASS] test_service_validation_integration")


def run_integration_tests():
    """Run all integration tests."""
    print("Running Z-Image integration tests...\n")

    test_ui_config_integration()
    test_generation_request_integration()
    test_service_validation_integration()

    print("\n[ALL INTEGRATION TESTS PASSED]")


if __name__ == "__main__":
    run_integration_tests()
