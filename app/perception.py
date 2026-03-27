from __future__ import annotations

import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class PerceptionService:
    def __init__(self, snapshot_dir: Path, admin_face_path: Path):
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(str(cascade_path))
        self._snapshot_dir = snapshot_dir
        self._admin_face_path = admin_face_path
        self._admin_metadata_path = admin_face_path.with_suffix(".json")
        self._admin_embedding, self._admin_sample_count = self._load_admin_profile()
        self._latest_observation: dict[str, Any] | None = None
        self._latest_observation_monotonic: float | None = None

    def analyze_snapshot(
        self,
        image_data_url: str | None,
        note: str | None = None,
        persist_snapshot: bool = True,
    ) -> dict[str, Any]:
        if not image_data_url:
            return self._empty_observation(note)

        image = self._decode_data_url(image_data_url)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self._detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        faces = [tuple(map(int, face)) for face in detections]
        brightness = float(np.mean(gray))
        image_path = self._save_snapshot(image) if persist_snapshot else None

        bootstrapped = False
        if self._admin_embedding is None and faces:
            primary_face = max(faces, key=lambda face: face[2] * face[3])
            self._bootstrap_admin(gray, primary_face)
            bootstrapped = True

        classified_faces = [self._classify_face(gray, face) for face in faces]
        admin_faces = [face for face in classified_faces if face["identity"] == "admin"]

        if self._admin_embedding is not None and not bootstrapped and admin_faces:
            best_admin_face = max(admin_faces, key=lambda face: face["similarity"])
            self._update_admin_profile(best_admin_face["embedding"])
            classified_faces = [self._classify_face(gray, face) for face in faces]
            admin_faces = [face for face in classified_faces if face["identity"] == "admin"]

        admin_detected = bool(admin_faces)
        rendered_faces = [self._render_face(face) for face in classified_faces]

        observation = {
            "admin_present": len(faces) > 0,
            "admin_detected": admin_detected,
            "face_count": int(len(faces)),
            "brightness": round(brightness, 2),
            "note": note,
            "image_path": image_path,
            "faces": rendered_faces,
            "frame_width": int(image.shape[1]),
            "frame_height": int(image.shape[0]),
            "admin_enrolled": self._admin_embedding is not None,
            "admin_sample_count": self._admin_sample_count,
            "admin_learning_state": self._admin_learning_state(),
        }
        self._remember_latest_observation(observation)
        return observation

    def _empty_observation(self, note: str | None) -> dict[str, Any]:
        observation = {
            "admin_present": False,
            "admin_detected": False,
            "face_count": 0,
            "brightness": 0.0,
            "note": note,
            "image_path": None,
            "faces": [],
            "frame_width": 0,
            "frame_height": 0,
            "admin_enrolled": self._admin_embedding is not None,
            "admin_sample_count": self._admin_sample_count,
            "admin_learning_state": self._admin_learning_state(),
        }
        self._remember_latest_observation(observation)
        return observation

    def admin_visible_recently(self, max_age_seconds: float = 2.0) -> bool:
        if not self._latest_observation or self._latest_observation_monotonic is None:
            return False
        if time.monotonic() - self._latest_observation_monotonic > max_age_seconds:
            return False
        return bool(self._latest_observation.get("admin_detected"))

    def _remember_latest_observation(self, observation: dict[str, Any]) -> None:
        self._latest_observation = observation.copy()
        self._latest_observation_monotonic = time.monotonic()

    def _decode_data_url(self, image_data_url: str) -> np.ndarray:
        encoded = image_data_url.split(",", maxsplit=1)[1]
        raw = base64.b64decode(encoded)
        array = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image data")
        return image

    def _save_snapshot(self, image: np.ndarray) -> str:
        filename = f"snapshot-{datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')}.jpg"
        target = self._snapshot_dir / filename
        cv2.imwrite(str(target), image)
        return str(target)

    def _load_admin_profile(self) -> tuple[np.ndarray | None, int]:
        if not self._admin_face_path.exists():
            return None, 0

        embedding = np.load(self._admin_face_path)
        if embedding.ndim != 1:
            return None, 0

        sample_count = self._load_sample_count_fallback()
        return embedding.astype(np.float32), sample_count

    def _load_sample_count_fallback(self) -> int:
        if self._admin_metadata_path.exists():
            try:
                metadata = json.loads(self._admin_metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
            sample_count = int(metadata.get("sample_count", 0))
            if sample_count > 0:
                return sample_count

        return 6

    def _save_admin_profile(self) -> None:
        if self._admin_embedding is None:
            return

        self._admin_face_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._admin_face_path, self._admin_embedding)
        self._admin_metadata_path.write_text(
            json.dumps(
                {
                    "sample_count": self._admin_sample_count,
                    "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _bootstrap_admin(self, gray: np.ndarray, face: tuple[int, int, int, int]) -> None:
        self._admin_embedding = self._face_embedding(gray, face)
        self._admin_sample_count = 1
        self._save_admin_profile()

    def _update_admin_profile(self, embedding: np.ndarray) -> None:
        if self._admin_embedding is None:
            self._admin_embedding = embedding
            self._admin_sample_count = 1
            self._save_admin_profile()
            return

        alpha = 0.32 if self._admin_sample_count < 4 else 0.18
        merged = ((1.0 - alpha) * self._admin_embedding) + (alpha * embedding)
        merged /= np.linalg.norm(merged) + 1e-6
        self._admin_embedding = merged.astype(np.float32)
        self._admin_sample_count += 1
        self._save_admin_profile()

    def _classify_face(self, gray: np.ndarray, face: tuple[int, int, int, int]) -> dict[str, Any]:
        x, y, w, h = map(int, face)
        embedding = self._face_embedding(gray, face)

        similarity = 0.0
        if self._admin_embedding is not None:
            similarity = float(np.clip(np.dot(embedding, self._admin_embedding), -1.0, 1.0))

        confidence = self._similarity_to_confidence(similarity)
        threshold = self._admin_similarity_threshold()
        identity = "admin" if self._admin_embedding is not None and similarity >= threshold else "unknown"
        color = self._confidence_color(identity, confidence)

        return {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "identity": identity,
            "confidence": round(confidence, 2),
            "color": color,
            "similarity": similarity,
            "embedding": embedding,
        }

    def _render_face(self, face: dict[str, Any]) -> dict[str, Any]:
        return {
            "x": face["x"],
            "y": face["y"],
            "w": face["w"],
            "h": face["h"],
            "identity": face["identity"],
            "confidence": face["confidence"],
            "color": face["color"],
        }

    def _face_embedding(self, gray: np.ndarray, face: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = map(int, face)
        face_crop = gray[y : y + h, x : x + w]
        face_crop = cv2.resize(face_crop, (96, 96), interpolation=cv2.INTER_AREA)
        face_crop = cv2.equalizeHist(face_crop).astype(np.float32)
        face_crop = (face_crop - np.mean(face_crop)) / (np.std(face_crop) + 1e-6)
        embedding = face_crop.flatten()
        embedding /= np.linalg.norm(embedding) + 1e-6
        return embedding.astype(np.float32)

    def _similarity_to_confidence(self, similarity: float) -> float:
        if self._admin_embedding is None:
            return 0.0

        base_confidence = float(np.clip((similarity - 0.35) / 0.5, 0.0, 1.0))
        maturity = min(1.0, 0.45 + (0.15 * self._admin_sample_count))
        return float(np.clip(base_confidence * maturity, 0.0, 1.0))

    def _admin_similarity_threshold(self) -> float:
        if self._admin_embedding is None:
            return 1.0
        if self._admin_sample_count < 3:
            return 0.66
        if self._admin_sample_count < 6:
            return 0.7
        return 0.72

    def _admin_learning_state(self) -> str:
        if self._admin_embedding is None:
            return "uninitialized"
        if self._admin_sample_count < 3:
            return "bootstrapping"
        if self._admin_sample_count < 6:
            return "learning"
        return "stable"

    def _confidence_color(self, identity: str, confidence: float) -> str:
        if identity == "admin" and confidence >= 0.7:
            return "green"
        if confidence >= 0.4:
            return "yellow"
        return "red"
