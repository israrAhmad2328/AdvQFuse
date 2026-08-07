import json
from pathlib import Path

from qfuse.data.vqa_adapters import build_earthvqa_manifest, build_floodnet_manifest, build_rsvqa_manifest


def test_earthvqa_adapter(tmp_path: Path) -> None:
    for folder in ("Train", "Val", "Test"):
        (tmp_path / folder / "images_png").mkdir(parents=True)
        (tmp_path / folder / "images_png/1.png").write_bytes(b"x")
        (tmp_path / f"{folder}_QA.json").write_text(json.dumps({"1.png": [{"Type": "Basic Judging", "Question": "Road?", "Answer": "Yes"}]}))
    records = build_earthvqa_manifest(tmp_path)
    assert len(records) == 3


def test_floodnet_datasetninja_adapter(tmp_path: Path) -> None:
    for folder in ("train_image", "valid_image", "test_image"):
        (tmp_path / folder / "img").mkdir(parents=True)
        (tmp_path / folder / "ann").mkdir(parents=True)
        (tmp_path / folder / "img/1.JPG").write_bytes(b"x")
        payload = {"tags": [{"value": "{'Question_ID': '1', 'Question': 'Flooded?', 'Ground_Truth': 'Yes', 'Question_Type': 'Yes_No'}"}]}
        (tmp_path / folder / "ann/1.json").write_text(json.dumps(payload))
    records = build_floodnet_manifest(tmp_path)
    assert len(records) == 3
    assert records[0].answer == "Yes"


def test_rsvqa_adapter(tmp_path: Path) -> None:
    (tmp_path / "annotations").mkdir()
    (tmp_path / "images").mkdir()
    (tmp_path / "images/7.tif").write_bytes(b"x")
    questions = {"questions": [{"question_id": 11, "img_id": 7, "question": "How many roads?", "type": "count"}]}
    answers = {"answers": [{"question_id": 11, "answer": "2"}]}
    (tmp_path / "annotations/USGS_split_train_questions.json").write_text(json.dumps(questions))
    (tmp_path / "annotations/USGS_split_train_answers.json").write_text(json.dumps(answers))
    records = build_rsvqa_manifest(tmp_path, high_resolution=True)
    assert len(records) == 1
    assert records[0].answer == "2"


def test_floodnet_separate_question_answer_tags(tmp_path: Path) -> None:
    folder = "train_image"
    (tmp_path / folder / "img").mkdir(parents=True)
    (tmp_path / folder / "ann").mkdir(parents=True)
    (tmp_path / folder / "img/6562.JPG").write_bytes(b"x")
    payload = {
        "tags": [
            {"name": "question", "value": "{'Question_ID': '154', 'Question': 'How many buildings?', 'Question_Type': 'Simple_Counting'}"},
            {"name": "answer", "value": "{'Question_ID': '154', 'Ground_Truth': '3'}"},
        ]
    }
    (tmp_path / folder / "ann/6562.JPG.json").write_text(json.dumps(payload))
    records = build_floodnet_manifest(tmp_path)
    assert len(records) == 1
    assert records[0].question == "How many buildings?"
    assert records[0].answer == "3"
