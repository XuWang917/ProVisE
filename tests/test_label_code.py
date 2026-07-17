from provise.protocols.base import ParseResult
from provise.protocols.label_code import LabelCodeProtocol


def test_label_code_scores_choice_text_against_dynamic_label():
    protocol = LabelCodeProtocol({"labels": "from_choices", "layout": "hstrip"})
    item = {
        "answer": "Option B",
        "choices": [
            {"label": "A", "text": "Option A"},
            {"label": "B is closer", "text": "Option B"},
        ],
    }

    scored = protocol.score(ParseResult("B", True), item, ".")

    assert scored.score == 1.0
    assert scored.extra["prediction"] == "B"
    assert scored.extra["ground_truth"] == "B"


def test_label_code_formats_dynamic_slots():
    protocol = LabelCodeProtocol({"labels": "from_choices", "layout": "hstrip"})
    item = {
        "choices": [
            {"label": "A", "text": "left"},
            {"label": "B", "text": "right"},
        ]
    }

    assert protocol.labels(item) == ["A", "B"]
    assert protocol._format_label_slots(item, protocol.labels(item)) == "slot 1=A (left); slot 2=B (right)"
