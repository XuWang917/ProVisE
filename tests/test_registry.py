from provise.protocols import create_protocol, list_protocols


def test_protocol_registry_contains_core_protocols():
    names = set(list_protocols())

    assert "label_code" in names
    assert "dense_depth_ab" in names
    assert "state_similarity" in names


def test_create_protocol():
    protocol = create_protocol("label_code", {"labels": "A-D"})

    assert protocol.name == "label_code"
