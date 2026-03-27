from app.perception import PerceptionService


def test_confidence_color_thresholds(tmp_path):
    service = PerceptionService(tmp_path / "snapshots", tmp_path / "admin_face.npy")

    assert service._confidence_color("admin", 0.85) == "green"
    assert service._confidence_color("unknown", 0.5) == "yellow"
    assert service._confidence_color("unknown", 0.1) == "red"
    assert service._similarity_to_confidence(0.9) == 0.0
