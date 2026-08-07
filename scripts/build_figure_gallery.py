from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


def figure_number(path: Path) -> int:
    match = re.match(r"fig(\d+)_", path.name)
    return int(match.group(1)) if match else 10_000


def collect_figures(root: Path) -> list[Path]:
    paths = []
    for folder in ["advanced_demo", "extended_quantitative", "qualitative_demo"]:
        paths.extend((root / folder).glob("fig*.png"))
    return sorted(paths, key=figure_number)


def parse_captions(path: Path) -> dict[int, tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"\*\*Figure (\d+)\. (.*?)\.\*\*\s*(.*)")
    captions: dict[int, tuple[str, str]] = {}
    for match in pattern.finditer(text):
        captions[int(match.group(1))] = (match.group(2).strip(), match.group(3).strip())
    return captions


def wrap_text(text: str, font_name: str, font_size: float, max_width: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def build_pdf(figures: list[Path], captions: dict[int, tuple[str, str]], output: Path) -> None:
    page_w, page_h = landscape(A4)
    c = canvas.Canvas(str(output), pagesize=(page_w, page_h), pageCompression=1)
    c.setTitle("AdvQFuse-RS 36-Figure Results Gallery")

    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(page_w / 2, page_h - 100, "AdvQFuse-RS")
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(page_w / 2, page_h - 135, "36-Figure Quantitative and Qualitative Results Gallery")
    c.setFont("Helvetica", 11)
    cover_lines = [
        "Figures 1-24: quantitative robustness, uncertainty, calibration, policy, efficiency, and statistics.",
        "Figures 25-36: qualitative remote-sensing attacks, fusion, recovery, uncertainty maps, and answer audits.",
        "All current images are synthetic software/layout demonstrations and are not scientific evidence.",
        "Replace them with logged Bonsai results and real test samples before manuscript submission.",
    ]
    y = page_h - 200
    for line in cover_lines:
        c.drawCentredString(page_w / 2, y, line)
        y -= 22
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(page_w / 2, 55, "Generated from the reproducible visualization code included in the package")
    c.showPage()

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for path in figures:
            num = figure_number(path)
            title, desc = captions.get(num, (path.stem.replace("_", " ").title(), ""))
            im = Image.open(path).convert("RGB")
            im.thumbnail((1900, 1350), Image.Resampling.LANCZOS)
            jpg = tmpdir / f"fig{num:02d}.jpg"
            im.save(jpg, quality=88, optimize=True, progressive=True)

            margin = 34
            title_y = page_h - 28
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin, title_y, f"Figure {num}. {title}")

            max_img_w = page_w - 2 * margin
            max_img_h = page_h - 120
            iw, ih = im.size
            scale = min(max_img_w / iw, max_img_h / ih)
            draw_w, draw_h = iw * scale, ih * scale
            x = (page_w - draw_w) / 2
            y_img = 62 + max(0, (max_img_h - draw_h) / 2)
            c.drawImage(ImageReader(str(jpg)), x, y_img, width=draw_w, height=draw_h, preserveAspectRatio=True, mask="auto")

            c.setFont("Helvetica", 8.5)
            lines = wrap_text(desc, "Helvetica", 8.5, page_w - 2 * margin)
            y_text = 45
            for line in lines[:2]:
                c.drawString(margin, y_text, line)
                y_text -= 11
            c.setFont("Helvetica-Oblique", 7.5)
            c.drawRightString(page_w - margin, 20, "SYNTHETIC DEMO - replace with real experiment outputs")
            c.showPage()
    c.save()


def build_contact_sheet(figures: list[Path], captions: dict[int, tuple[str, str]], output: Path) -> None:
    cols = 4
    cell_w, cell_h = 430, 320
    rows = (len(figures) + cols - 1) // cols
    canvas_img = Image.new("RGB", (cols * cell_w, rows * cell_h + 80), "white")
    draw = ImageDraw.Draw(canvas_img)
    draw.text((20, 18), "AdvQFuse-RS - 36-Figure Results Suite", fill="black")
    draw.text((20, 45), "Synthetic demonstration: replace with logged real results and real test samples", fill="black")

    for idx, path in enumerate(figures):
        row, col = divmod(idx, cols)
        x0, y0 = col * cell_w, 80 + row * cell_h
        num = figure_number(path)
        title = captions.get(num, (path.stem, ""))[0]
        im = Image.open(path).convert("RGB")
        im.thumbnail((cell_w - 18, cell_h - 58), Image.Resampling.LANCZOS)
        x = x0 + (cell_w - im.width) // 2
        y = y0 + 34 + (cell_h - 58 - im.height) // 2
        canvas_img.paste(im, (x, y))
        draw.rectangle((x0 + 4, y0 + 4, x0 + cell_w - 4, y0 + cell_h - 4), outline=(175, 175, 175), width=1)
        label = f"Fig. {num}: {title}"
        if len(label) > 56:
            label = label[:53] + "..."
        draw.text((x0 + 10, y0 + 11), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas_img.save(output, optimize=True, compress_level=7)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a combined gallery PDF and contact sheet for all AdvQFuse-RS figures.")
    parser.add_argument("--figures-root", default="figures")
    parser.add_argument("--captions", default="paper/RESULTS_VISUALIZATION_PLAN.md")
    parser.add_argument("--pdf", default="figures/AdvQFuse_RS_36_Figure_Gallery.pdf")
    parser.add_argument("--contact-sheet", default="figures/AdvQFuse_RS_36_Figure_Contact_Sheet.png")
    args = parser.parse_args()
    figures = collect_figures(Path(args.figures_root))
    if len(figures) != 36:
        raise RuntimeError(f"Expected 36 numbered figures, found {len(figures)}")
    captions = parse_captions(Path(args.captions))
    build_pdf(figures, captions, Path(args.pdf))
    build_contact_sheet(figures, captions, Path(args.contact_sheet))
    print(f"Built gallery for {len(figures)} figures.")


if __name__ == "__main__":
    main()
