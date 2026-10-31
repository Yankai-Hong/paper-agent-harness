"""Make local contact sheets for visual review of Docling page crops."""
import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from paper_agent.visual_evidence import validate_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True, help="New directory for local contact sheets")
    args = parser.parse_args()
    entries = [json.loads(line) for line in Path(args.evidence).read_text(encoding="utf-8").splitlines() if line.strip()]
    visuals = [e for e in entries if e.get("evidence_type") in {"figure", "table"}]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    columns, per_sheet, width, height = 3, 12, 400, 300
    for start in range(0, len(visuals), per_sheet):
        part = visuals[start:start + per_sheet]
        canvas = Image.new("RGB", (columns * width, math.ceil(len(part) / columns) * height), "white")
        draw = ImageDraw.Draw(canvas)
        for i, evidence in enumerate(part):
            path = validate_image(evidence, args.image_root)
            x, y = i % columns * width, i // columns * height
            with Image.open(path) as image:
                thumbnail = ImageOps.contain(image.convert("RGB"), (width - 16, height - 55))
            canvas.paste(thumbnail, (x + 8, y + 8))
            title = f"p{evidence['page']} {evidence['evidence_type']} {evidence.get('figure_number') or 'number unknown'}"
            draw.text((x + 8, y + height - 40), title, fill="black")
            draw.text((x + 8, y + height - 22), evidence['chunk_id'], fill="black")
        canvas.save(output / f"sheet-{start // per_sheet + 1:03}.png")
    print(json.dumps({"visual_items": len(visuals), "sheets": math.ceil(len(visuals) / per_sheet)}))


if __name__ == "__main__":
    main()
