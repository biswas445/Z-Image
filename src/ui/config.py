"""UI Configuration for Z-Image."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class UIConfig:
    """Configuration for Z-Image UI."""

    server_host: str = "0.0.0.0"
    server_port: int = 7860
    model_path: str = "ckpts/Z-Image-Turbo"
    output_dir: str = "outputs/ui"
    default_height: int = 1024
    default_width: int = 1024
    default_steps: int = 8
    default_guidance_scale: float = 0.0
    default_seed: int = 42
    max_height: int = 2048
    max_width: int = 2048
    min_height: int = 256
    min_width: int = 256
    max_steps: int = 50
    min_steps: int = 1
    attention_backend: str = "_native_flash"
    dtype: str = "bfloat16"
    compile: bool = False
    enable_progress: bool = True
    enable_queue: bool = True
    max_concurrent: int = 1
    ssl_keyfile: str | None = None
    ssl_certfile: str | None = None
    auth_username: str | None = None
    auth_password: str | None = None
    share: bool = False

    @classmethod
    def from_env(cls) -> "UIConfig":
        """Create config from environment variables."""
        return cls(
            server_host=os.environ.get("ZIMAGE_UI_HOST", "0.0.0.0"),
            server_port=int(os.environ.get("ZIMAGE_UI_PORT", "7860")),
            model_path=os.environ.get("ZIMAGE_MODEL_PATH", "ckpts/Z-Image-Turbo"),
            output_dir=os.environ.get("ZIMAGE_OUTPUT_DIR", "outputs/ui"),
            default_height=int(os.environ.get("ZIMAGE_DEFAULT_HEIGHT", "1024")),
            default_width=int(os.environ.get("ZIMAGE_DEFAULT_WIDTH", "1024")),
            default_steps=int(os.environ.get("ZIMAGE_DEFAULT_STEPS", "8")),
            default_guidance_scale=float(os.environ.get("ZIMAGE_DEFAULT_GUIDANCE_SCALE", "0.0")),
            default_seed=int(os.environ.get("ZIMAGE_DEFAULT_SEED", "42")),
            max_height=int(os.environ.get("ZIMAGE_MAX_HEIGHT", "2048")),
            max_width=int(os.environ.get("ZIMAGE_MAX_WIDTH", "2048")),
            min_height=int(os.environ.get("ZIMAGE_MIN_HEIGHT", "256")),
            min_width=int(os.environ.get("ZIMAGE_MIN_WIDTH", "256")),
            max_steps=int(os.environ.get("ZIMAGE_MAX_STEPS", "50")),
            min_steps=int(os.environ.get("ZIMAGE_MIN_STEPS", "1")),
            attention_backend=os.environ.get("ZIMAGE_ATTENTION", "_native_flash"),
            dtype=os.environ.get("ZIMAGE_DTYPE", "bfloat16"),
            compile=os.environ.get("ZIMAGE_COMPILE", "false").lower() == "true",
            enable_progress=os.environ.get("ZIMAGE_ENABLE_PROGRESS", "true").lower() == "true",
            enable_queue=os.environ.get("ZIMAGE_ENABLE_QUEUE", "true").lower() == "true",
            max_concurrent=int(os.environ.get("ZIMAGE_MAX_CONCURRENT", "1")),
            ssl_keyfile=os.environ.get("ZIMAGE_SSL_KEYFILE"),
            ssl_certfile=os.environ.get("ZIMAGE_SSL_CERTFILE"),
            auth_username=os.environ.get("ZIMAGE_AUTH_USERNAME"),
            auth_password=os.environ.get("ZIMAGE_AUTH_PASSWORD"),
            share=os.environ.get("ZIMAGE_SHARE", "false").lower() == "true",
        )

    def get_output_dir(self) -> Path:
        """Get output directory, creating if necessary."""
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def get_auth(self) -> tuple[str, str] | None:
        """Get authentication tuple if configured."""
        if self.auth_username and self.auth_password:
            return (self.auth_username, self.auth_password)
        return None
