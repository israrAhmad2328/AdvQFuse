from pathlib import Path

from qfuse.data.dataset_layout import locate_uav_obb_root, validate_sen12ms, validate_uav_obb


def test_locate_nested_uav_root(tmp_path: Path) -> None:
    root = tmp_path / "download" / "nested" / "UAV-OBB"
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text("names: [bike, bus, car, other_vehicle, taxi, truck]")
    assert locate_uav_obb_root(tmp_path) == root
    assert validate_uav_obb(tmp_path).state == "ready"


def test_sen12ms_without_labels_is_partial(tmp_path: Path) -> None:
    (tmp_path / "optical").mkdir()
    (tmp_path / "sar").mkdir()
    (tmp_path / "optical/a.tif").write_bytes(b"x")
    (tmp_path / "sar/a.tif").write_bytes(b"x")
    status = validate_sen12ms(tmp_path)
    assert status.state == "partial"
    assert any("ground-truth" in item for item in status.warnings)
