"""Z-Image Web UI Application."""

import os
import sys
from pathlib import Path

import gradio as gr
from loguru import logger

from ui.config import UIConfig
from ui.service import GenerationRequest, ImageGenerationService

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

CSS = """
.gradio-container {
    max-width: 1400px !important;
}
#output-gallery {
    min-height: 500px;
}
#prompt-input {
    font-size: 16px;
}
.generation-info {
    font-size: 12px;
    color: #666;
}
"""


def create_ui(config: UIConfig) -> gr.Blocks:
    """Create the Gradio UI."""

    service = ImageGenerationService(config)

    default_prompt = (
        "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. "
        "Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. "
        "Neon lightning-bolt lamp, bright yellow glow, above extended left palm. Soft-lit outdoor night background, "
        "silhouetted tiered pagoda, blurred colorful distant lights."
    )

    with gr.Blocks(
        title="Z-Image",
        fill_height=True,
    ) as demo:

        gr.Markdown(
            """
            # Z-Image Text-to-Image Generator
            
            Generate high-quality images from text descriptions using Z-Image diffusion model.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Prompt")

                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=5,
                    value=default_prompt,
                    elem_id="prompt-input",
                )

                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt (optional)",
                    placeholder="What to avoid in the image...",
                    lines=3,
                    value="",
                )

                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        height_slider = gr.Slider(
                            label="Height",
                            minimum=config.min_height,
                            maximum=config.max_height,
                            value=config.default_height,
                            step=64,
                        )
                        width_slider = gr.Slider(
                            label="Width",
                            minimum=config.min_width,
                            maximum=config.max_width,
                            value=config.default_width,
                            step=64,
                        )

                    with gr.Row():
                        steps_slider = gr.Slider(
                            label="Inference Steps",
                            minimum=config.min_steps,
                            maximum=config.max_steps,
                            value=config.default_steps,
                            step=1,
                        )
                        guidance_slider = gr.Slider(
                            label="Guidance Scale",
                            minimum=0.0,
                            maximum=20.0,
                            value=config.default_guidance_scale,
                            step=0.5,
                        )

                    with gr.Row():
                        seed_input = gr.Number(
                            label="Seed (-1 for random)",
                            value=config.default_seed,
                            precision=0,
                        )
                        num_images_slider = gr.Slider(
                            label="Number of Images",
                            minimum=1,
                            maximum=4,
                            value=1,
                            step=1,
                        )

                    cfg_norm_checkbox = gr.Checkbox(
                        label="CFG Normalization",
                        value=False,
                    )

                generate_btn = gr.Button(
                    "Generate",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=2):
                gr.Markdown("### Output")

                output_gallery = gr.Gallery(
                    label="Generated Images",
                    columns=2,
                    rows=2,
                    height=600,
                    object_fit="contain",
                    elem_id="output-gallery",
                )

                with gr.Accordion("Generation Info", open=False):
                    generation_info = gr.JSON(
                        label="Metadata",
                    )

                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                )

        gr.Markdown(
            """
            ---
            **Tips:**
            - Higher steps = better quality but slower
            - Guidance scale controls how closely the image follows the prompt
            - Use negative prompts to avoid unwanted elements
            - Seed -1 generates a random seed each time
            """
        )

        def generate_images(
            prompt,
            negative_prompt,
            height,
            width,
            steps,
            guidance,
            seed,
            num_images,
            cfg_norm,
        ):
            """Generate images from prompt."""
            try:
                if seed < 0:
                    seed = int(torch.Generator().seed())

                request = GenerationRequest(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=int(height),
                    width=int(width),
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance),
                    seed=int(seed),
                    num_images=int(num_images),
                    cfg_normalization=cfg_norm,
                )

                result = service.generate(request)

                images = [str(path) for path in result.output_paths]

                info = {
                    "generation_time": f"{result.generation_time:.2f}s",
                    "seed": result.seed,
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "dimensions": f"{request.height}x{request.width}",
                    "steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "device": str(result.metadata["device"]),
                }

                return images, info, f"✓ Generated {len(images)} image(s) in {result.generation_time:.2f}s"

            except Exception as e:
                logger.error(f"Generation failed: {e}")
                return [], {"error": str(e)}, f"✗ Error: {e}"

        import torch

        generate_btn.click(
            fn=generate_images,
            inputs=[
                prompt_input,
                negative_prompt_input,
                height_slider,
                width_slider,
                steps_slider,
                guidance_slider,
                seed_input,
                num_images_slider,
                cfg_norm_checkbox,
            ],
            outputs=[output_gallery, generation_info, status_text],
        )

        prompt_input.submit(
            fn=generate_images,
            inputs=[
                prompt_input,
                negative_prompt_input,
                height_slider,
                width_slider,
                steps_slider,
                guidance_slider,
                seed_input,
                num_images_slider,
                cfg_norm_checkbox,
            ],
            outputs=[output_gallery, generation_info, status_text],
        )

    return demo


def main():
    """Main entry point for Z-Image UI."""
    config = UIConfig.from_env()

    logger.info("Starting Z-Image UI")
    logger.info(f"Model path: {config.model_path}")
    logger.info(f"Server: {config.server_host}:{config.server_port}")

    demo = create_ui(config)

    auth = config.get_auth()

    demo.queue(
        default_concurrency_limit=config.max_concurrent,
        api_open=True,
    )

    demo.launch(
        server_name=config.server_host,
        server_port=config.server_port,
        auth=auth,
        share=config.share,
        ssl_keyfile=config.ssl_keyfile,
        ssl_certfile=config.ssl_certfile,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        css=CSS,
    )


if __name__ == "__main__":
    main()
