from pathlib import Path

from qfuse.data import generate_qa_for_image, parse_dota_label


def test_uav_obb_generator(tmp_path: Path):
    image = tmp_path / "tile.png"
    image.write_bytes(b"placeholder")
    label = tmp_path / "tile.txt"
    label.write_text(
        "0 0 10 0 10 5 0 5 car 0\n20 20 25 20 25 25 20 25 truck 0\n",
        encoding="utf-8",
    )
    objects = parse_dota_label(label)
    assert len(objects) == 2
    records = generate_qa_for_image(image, label)
    assert any(r.question_type == "count" and r.answer == "2" for r in records)
    assert any(r.question_type == "orientation" for r in records)
