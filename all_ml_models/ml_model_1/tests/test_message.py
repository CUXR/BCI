import json


def test_classification_message_format():
    msg = json.dumps({"label": "LEFT", "confidence": 0.82, "ts": 1713200000.0}) + "\n"
    parsed = json.loads(msg.strip())
    assert parsed["label"] == "LEFT"
    assert isinstance(parsed["confidence"], float)
    assert isinstance(parsed["ts"], float)
    assert msg.endswith("\n")


def test_message_is_single_line():
    msg = json.dumps({"label": "FORWARD", "confidence": 0.91, "ts": 1713200001.0}) + "\n"
    lines = msg.strip().split("\n")
    assert len(lines) == 1
