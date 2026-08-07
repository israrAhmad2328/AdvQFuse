from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

FOLDERS = ["advanced_demo", "extended_quantitative", "qualitative_demo", "tpami_demo", "v4_extended"]


def number(path: Path) -> int:
    m = re.match(r"fig(\d+)_", path.name)
    return int(m.group(1)) if m else 9999


def collect(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for folder in FOLDERS:
        candidates.extend((root / folder).glob("fig*.png"))
    by_number: dict[int, Path] = {}
    for path in candidates:
        by_number.setdefault(number(path), path)
    missing = [i for i in range(1, 56) if i not in by_number]
    if missing:
        raise RuntimeError(f"Missing figure numbers: {missing}")
    return [by_number[i] for i in range(1, 56)]


def captions(path: Path) -> dict[int, tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    result: dict[int, tuple[str, str]] = {}
    # Main bold format: **Figure 44. Title.** Description
    for m in re.finditer(r"\*\*Figure\s+(\d+)\.\s+(.*?)\.\*\*\s*(.*)", text):
        result[int(m.group(1))] = (m.group(2).strip(), m.group(3).strip())
    # TPAMI addendum format: - **Figure 37 - Title.** Description
    for m in re.finditer(r"\*\*Figure\s+(\d+)\s+-\s+(.*?)\.\*\*\s*(.*)", text):
        result[int(m.group(1))] = (m.group(2).strip(), m.group(3).strip())
    return result


def wrap(text: str, font: str, size: float, width: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        if stringWidth(candidate, font, size) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def build_pdf(figures: list[Path], caps: dict[int, tuple[str, str]], output: Path) -> None:
    page_w, page_h = landscape(A4)
    c = canvas.Canvas(str(output), pagesize=(page_w, page_h), pageCompression=1)
    c.setTitle("AdvQFuse TPAMI v4 - 55 Figure Planning Gallery")
    c.setFont("Helvetica-Bold", 25)
    c.drawCentredString(page_w / 2, page_h - 95, "AdvQFuse TPAMI v4")
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(page_w / 2, page_h - 132, "55-Figure Quantitative and Qualitative Planning Gallery")
    c.setFont("Helvetica", 11)
    lines = [
        "Figures 1-24: quantitative robustness, calibration, statistics, policy, efficiency, and distribution analysis.",
        "Figures 25-36: qualitative remote-sensing attacks, fusion, recovery, uncertainty maps, and answer auditing.",
        "Figures 37-43: TPAMI-wide precision-path, held-out generalization, risk-cost, statistics, and architecture layouts.",
        "Figures 44-55: expanded multi-panel qualitative, subgroup, compute, multimodal, and residual-failure audits.",
        "Every current panel is a synthetic layout demonstration, not scientific evidence.",
    ]
    y = page_h - 195
    for line in lines:
        c.drawCentredString(page_w / 2, y, line)
        y -= 23
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(page_w / 2, 55, "Replace all synthetic panels with immutable locked-test outputs before submission")
    c.showPage()

    with tempfile.TemporaryDirectory() as temp:
        tmp = Path(temp)
        for fig_path in figures:
            n = number(fig_path)
            title, desc = caps.get(n, (fig_path.stem.replace("_", " ").title(), ""))
            im = Image.open(fig_path).convert("RGB")
            im.thumbnail((2000, 1420), Image.Resampling.LANCZOS)
            jpg = tmp / f"fig{n:02d}.jpg"
            im.save(jpg, quality=89, optimize=True)
            margin = 32
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin, page_h - 26, f"Figure {n}. {title}")
            max_w = page_w - 2 * margin
            max_h = page_h - 115
            scale = min(max_w / im.width, max_h / im.height)
            dw, dh = im.width * scale, im.height * scale
            c.drawImage(ImageReader(str(jpg)), (page_w - dw) / 2, 58 + (max_h - dh) / 2, width=dw, height=dh, preserveAspectRatio=True)
            c.setFont("Helvetica", 8.2)
            for j, line in enumerate(wrap(desc, "Helvetica", 8.2, max_w)[:2]):
                c.drawString(margin, 43 - 10 * j, line)
            c.setFont("Helvetica-Oblique", 7.5)
            c.drawRightString(page_w - margin, 18, "SYNTHETIC DEMO - NOT MEASURED RESULTS")
            c.showPage()
    c.save()


def build_sheet(figures: list[Path], caps: dict[int, tuple[str, str]], output: Path) -> None:
    cols = 5
    cell_w, cell_h = 360, 275
    rows = (len(figures) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + 90), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 18), "AdvQFuse TPAMI v4 - 55-Figure Planning Suite", fill="black")
    draw.text((20, 47), "Synthetic layout demonstrations only - quantitative and qualitative evidence must be replaced", fill="black")
    for idx, path in enumerate(figures):
        row, col = divmod(idx, cols)
        x0, y0 = col * cell_w, 90 + row * cell_h
        n = number(path)
        title = caps.get(n, (path.stem, ""))[0]
        im = Image.open(path).convert("RGB")
        im.thumbnail((cell_w - 18, cell_h - 54), Image.Resampling.LANCZOS)
        x = x0 + (cell_w - im.width) // 2
        y = y0 + 30 + (cell_h - 50 - im.height) // 2
        sheet.paste(im, (x, y))
        draw.rectangle((x0 + 4, y0 + 4, x0 + cell_w - 4, y0 + cell_h - 4), outline=(170, 170, 170), width=1)
        label = f"Fig. {n}: {title}"
        if len(label) > 48:
            label = label[:45] + "..."
        draw.text((x0 + 9, y0 + 9), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="figures")
    parser.add_argument("--captions", default="paper/RESULTS_VISUALIZATION_PLAN.md")
    parser.add_argument("--pdf", default="figures/AdvQFuse_TPAMI_v4_55_Figure_Gallery.pdf")
    parser.add_argument("--sheet", default="figures/AdvQFuse_TPAMI_v4_55_Figure_Contact_Sheet.png")
    args = parser.parse_args()
    figs = collect(Path(args.root))
    caps = captions(Path(args.captions))
    build_pdf(figs, caps, Path(args.pdf))
    build_sheet(figs, caps, Path(args.sheet))
    print(f"Built complete gallery with {len(figs)} numbered figures")


if __name__ == "__main__":
    main()
