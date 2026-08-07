from pathlib import Path

from qfuse.data.uav_obb_qa import generate_qa_for_image, parse_obb_label


def test_parse_yolo_corner_format(tmp_path: Path) -> None:
    label = tmp_path / "x.txt"
    label.write_text("2 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")
    objects = parse_obb_label(label)
    assert len(objects) == 1
    assert objects[0].class_name == "car"


def test_parse_yolo_center_angle_format(tmp_path: Path) -> None:
    label = tmp_path / "x.txt"
    label.write_text("1 0.5 0.5 0.2 0.1 0.4\n")
    objects = parse_obb_label(label)
    assert len(objects) == 1
    assert objects[0].points.shape == (4, 2)


def test_parse_dota_format(tmp_path: Path) -> None:
    label = tmp_path / "x.txt"
    label.write_text("0 0 10 0 10 5 0 5 bus 0\n")
    objects = parse_obb_label(label)
    assert len(objects) == 1
    assert objects[0].class_name == "bus"


def test_uav_qa_has_negative_existence(tmp_path: Path) -> None:
    image = tmp_path / "x.jpg"
    image.write_bytes(b"not-an-image")
    label = tmp_path / "x.txt"
    label.write_text("2 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")
    records = generate_qa_for_image(image, label)
    answers = {(r.metadata.get("class_name"), r.question_type): r.answer for r in records}
    assert answers[("car", "existence")] == "yes"
    assert answers[("bus", "existence")] == "no"
