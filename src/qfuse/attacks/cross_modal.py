from __future__ import annotations

from collections.abc import Sequence

from PIL import Image, ImageDraw, ImageFont


def compose_sensor_panel(
    images: Sequence[Image.Image],
    labels: Sequence[str],
    tile_size: tuple[int, int] = (512, 512),
    gutter: int = 16,
) -> Image.Image:
    """Compose aligned sensor views into one labelled panel for a generic VLM."""
    if len(images) != len(labels) or not images:
        raise ValueError("images and labels must be non-empty and have equal length")
    tw, th = tile_size
    label_h = 36
    width = len(images) * tw + (len(images) - 1) * gutter
    canvas = Image.new("RGB", (width, th + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (img, label) in enumerate(zip(images, labels)):
        x = i * (tw + gutter)
        tile = img.convert("RGB").resize((tw, th))
        canvas.paste(tile, (x, label_h))
        draw.text((x + 8, 8), str(label), fill="black")
    return canvas


def render_instruction_overlay(
    image: Image.Image,
    instruction: str,
    opacity: float = 0.65,
    position: str = "bottom",
) -> Image.Image:
    """Overlay a benign benchmark instruction to test image-text conflict."""
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    h = max(48, int(base.height * 0.14))
    y0 = base.height - h if position == "bottom" else 0
    alpha = int(255 * max(0.0, min(1.0, opacity)))
    draw.rectangle((0, y0, base.width, y0 + h), fill=(255, 255, 255, alpha))
    draw.text((12, y0 + 12), instruction, fill=(0, 0, 0, 255))
    return Image.alpha_composite(base, overlay).convert("RGB")
