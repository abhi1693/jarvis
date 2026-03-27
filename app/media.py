from __future__ import annotations

import base64
import re
from datetime import datetime
from pathlib import Path


class MediaService:
    _MIME_EXTENSIONS = {
        "audio/webm": ".webm",
        "audio/wav": ".wav",
        "audio/wave": ".wav",
        "audio/x-wav": ".wav",
        "audio/ogg": ".ogg",
        "audio/mp4": ".m4a",
        "audio/mpeg": ".mp3",
    }

    def __init__(self, media_dir: Path):
        self._media_dir = media_dir

    def save_audio_data_url(self, audio_data_url: str) -> str:
        mime_type, payload = self._split_data_url(audio_data_url)
        extension = self._MIME_EXTENSIONS.get(mime_type, ".bin")
        filename = f"audio-{datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')}{extension}"
        target = self._media_dir / filename
        target.write_bytes(base64.b64decode(payload))
        return str(target)

    def _split_data_url(self, data_url: str) -> tuple[str, str]:
        match = re.match(r"^data:([^;]+);base64,(.+)$", data_url, flags=re.DOTALL)
        if not match:
            raise ValueError("Invalid data URL")
        return match.group(1), match.group(2)
