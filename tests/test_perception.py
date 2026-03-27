import numpy as np

from app.perception import PerceptionService


def test_confidence_color_thresholds(tmp_path):
    service = PerceptionService(tmp_path / "snapshots", tmp_path / "admin_face.npy")

    assert service._confidence_color("admin", 0.85) == "green"
    assert service._confidence_color("unknown", 0.5) == "yellow"
    assert service._confidence_color("unknown", 0.1) == "red"
    assert service._similarity_to_confidence(0.9) == 0.0


class FakeDetector:
    def __init__(self, faces):
        self._faces = np.array(faces, dtype=np.int32)

    def detectMultiScale(self, *_args, **_kwargs):
        return self._faces


def test_first_detected_face_bootstraps_admin_profile(tmp_path):
    service = PerceptionService(tmp_path / "snapshots", tmp_path / "admin_face.npy")
    service._detector = FakeDetector([(10, 20, 60, 60)])
    service._decode_data_url = lambda _data: np.zeros((160, 160, 3), dtype=np.uint8)
    service._save_snapshot = lambda _image: "snapshot.jpg"
    service._face_embedding = lambda _gray, _face: np.array([1.0, 0.0], dtype=np.float32)

    observation = service.analyze_snapshot("data:image/jpeg;base64,ignored")

    assert observation["admin_enrolled"] is True
    assert observation["admin_sample_count"] == 1
    assert observation["admin_learning_state"] == "bootstrapping"
    assert observation["faces"][0]["identity"] == "admin"
    assert observation["faces"][0]["color"] == "yellow"


def test_confident_admin_match_updates_profile_over_time(tmp_path):
    service = PerceptionService(tmp_path / "snapshots", tmp_path / "admin_face.npy")
    service._detector = FakeDetector([(10, 20, 60, 60)])
    service._decode_data_url = lambda _data: np.zeros((160, 160, 3), dtype=np.uint8)
    service._save_snapshot = lambda _image: "snapshot.jpg"
    service._admin_embedding = np.array([1.0, 0.0], dtype=np.float32)
    service._admin_sample_count = 2
    service._face_embedding = lambda _gray, _face: np.array([1.0, 0.0], dtype=np.float32)

    observation = service.analyze_snapshot("data:image/jpeg;base64,ignored")

    assert observation["admin_detected"] is True
    assert observation["admin_sample_count"] == 3
    assert observation["admin_learning_state"] == "learning"
    assert observation["faces"][0]["identity"] == "admin"


def test_live_observation_can_skip_snapshot_persistence(tmp_path):
    service = PerceptionService(tmp_path / "snapshots", tmp_path / "admin_face.npy")
    service._detector = FakeDetector([(10, 20, 60, 60)])
    service._decode_data_url = lambda _data: np.zeros((160, 160, 3), dtype=np.uint8)
    service._face_embedding = lambda _gray, _face: np.array([1.0, 0.0], dtype=np.float32)

    calls = {"count": 0}

    def fake_save_snapshot(_image):
        calls["count"] += 1
        return "snapshot.jpg"

    service._save_snapshot = fake_save_snapshot

    observation = service.analyze_snapshot(
        "data:image/jpeg;base64,ignored",
        persist_snapshot=False,
    )

    assert observation["image_path"] is None
    assert calls["count"] == 0


def test_admin_visibility_requires_fresh_admin_detection(tmp_path):
    service = PerceptionService(tmp_path / "snapshots", tmp_path / "admin_face.npy")

    assert service.admin_visible_recently() is False

    service._remember_latest_observation({"admin_detected": True})
    assert service.admin_visible_recently(max_age_seconds=1.0) is True

    service._latest_observation_monotonic -= 5.0
    assert service.admin_visible_recently(max_age_seconds=1.0) is False
