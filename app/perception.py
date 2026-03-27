from __future__ import annotations

import base64
from datetime import datetime
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
        self._admin_embedding = self._load_admin_embedding()

    def analyze_snapshot(self, image_data_url: str | None, note: str | None = None) -> dict[str, Any]:
        if not image_data_url:
            return {
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
            }

        image = self._decode_data_url(image_data_url)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        brightness = float(np.mean(gray))
        image_path = self._save_snapshot(image)
        face_matches = [self._classify_face(gray, face) for face in faces]
        admin_detected = any(face["identity"] == "admin" for face in face_matches)

        return {
            "admin_present": len(faces) > 0,
            "admin_detected": admin_detected,
            "face_count": int(len(faces)),
            "brightness": round(brightness, 2),
            "note": note,
            "image_path": image_path,
            "faces": face_matches,
            "frame_width": int(image.shape[1]),
            "frame_height": int(image.shape[0]),
            "admin_enrolled": self._admin_embedding is not None,
        }

    def enroll_admin(self, image_data_url: str) -> dict[str, Any]:
        image = self._decode_data_url(image_data_url)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            raise ValueError("No face detected for admin enrollment.")

        largest_face = max(faces, key=lambda face: face[2] * face[3])
        embedding = self._face_embedding(gray, largest_face)
        self._admin_face_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._admin_face_path, embedding)
        self._admin_embedding = embedding

        x, y, w, h = map(int, largest_face)
        return {
            "status": "ok",
            "message": "Admin face enrolled.",
            "face": {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "identity": "admin",
                "confidence": 1.0,
                "color": "green",
            },
        }

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

    def _load_admin_embedding(self) -> np.ndarray | None:
        if not self._admin_face_path.exists():
            return None
        embedding = np.load(self._admin_face_path)
        if embedding.ndim != 1:
            return None
        return embedding.astype(np.float32)

    def _classify_face(self, gray: np.ndarray, face: tuple[int, int, int, int]) -> dict[str, Any]:
        x, y, w, h = map(int, face)
        embedding = self._face_embedding(gray, face)

        similarity = 0.0
        if self._admin_embedding is not None:
            similarity = float(np.clip(np.dot(embedding, self._admin_embedding), -1.0, 1.0))

        confidence = self._similarity_to_confidence(similarity)
        identity = "admin" if self._admin_embedding is not None and similarity >= 0.72 else "unknown"
        color = self._confidence_color(identity, confidence)

        return {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "identity": identity,
            "confidence": round(confidence, 2),
            "color": color,
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
        return float(np.clip((similarity - 0.35) / 0.5, 0.0, 1.0))

    def _confidence_color(self, identity: str, confidence: float) -> str:
        if identity == "admin" and confidence >= 0.7:
            return "green"
        if confidence >= 0.4:
            return "yellow"
        return "red"
