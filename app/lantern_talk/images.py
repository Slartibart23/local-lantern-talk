"""Pictures the user adds: character avatars, chat backgrounds and the user's own avatar.

The browser crops and shrinks a picture before uploading it (avatars 512 × 512,
backgrounds at most 1920 px wide), so the files here are small WebP, PNG or JPEG
images. They live in ``user_data\\images`` under names the app chooses; a new
picture always gets a new name, so a browser never shows an old cached one.
"""

from __future__ import annotations

import logging
import re
import secrets
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

KINDS = ("avatar", "background", "user-avatar")
MAX_IMAGE_BYTES = 8 * 2**20
_NAME_RE = re.compile(r"^(avatar|background|user-avatar)-[0-9a-f]{12}\.(webp|png|jpg)$")


class ImageError(ValueError):
    """An unusable picture, with a message meant for the user."""


def image_format(data: bytes) -> str | None:
    """File extension for WebP, PNG or JPEG data, else None."""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    return None


def valid_name(name: str) -> bool:
    return bool(_NAME_RE.match(name or ""))


class ImageStore:
    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def path(self, name: str) -> Path | None:
        if not valid_name(name):
            return None
        path = self.folder / name
        return path if path.is_file() else None

    def save(self, kind: str, data: bytes) -> str:
        if kind not in KINDS:
            raise ImageError(f"Unknown kind of picture: {kind}.")
        if not data:
            raise ImageError("The picture is empty.")
        if len(data) > MAX_IMAGE_BYTES:
            raise ImageError("The picture is too large.")
        ext = image_format(data)
        if ext is None:
            raise ImageError("Please use a JPG, PNG or WebP picture.")
        self.folder.mkdir(parents=True, exist_ok=True)
        name = f"{kind}-{secrets.token_hex(6)}.{ext}"
        (self.folder / name).write_bytes(data)
        return name

    def copy(self, name: str) -> str:
        """A copy under a new name (a duplicated character gets its own pictures)."""
        source = self.path(name)
        if source is None:
            return ""
        kind = name.rsplit("-", 1)[0]
        target = f"{kind}-{secrets.token_hex(6)}{source.suffix}"
        shutil.copyfile(source, self.folder / target)
        return target

    def delete(self, name: str) -> None:
        path = self.path(name)
        if path is not None:
            path.unlink(missing_ok=True)

    def cleanup(self, keep: set[str]) -> None:
        """Remove pictures nobody uses any more (replaced or removed avatars and backgrounds)."""
        if not self.folder.is_dir():
            return
        for f in self.folder.iterdir():
            if f.is_file() and valid_name(f.name) and f.name not in keep:
                f.unlink(missing_ok=True)
                log.info("Removed unused picture %s", f.name)
