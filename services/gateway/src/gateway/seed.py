"""Seed the built-in voice catalogue.

Reads the reference pack in ``assets/voices`` — carried over from v1, where it lived in
``voices/`` and was addressed by filesystem path — uploads each sample to object storage,
and registers it as a built-in voice with no owner.

Idempotent: a voice already present by name is left alone, so this can run on every
deploy rather than needing to be remembered once.

Run with::

    uv run --package story2audio-gateway python -m gateway.seed
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from sqlalchemy import select

from gateway.audio import validate_voice_upload
from gateway.db import create_engine, create_session_factory
from story2audio_shared.config import limit_settings, storage_settings
from story2audio_shared.errors import AppError
from story2audio_shared.ids import uuid7
from story2audio_shared.logging import configure_logging, get_logger
from story2audio_shared.models import Voice
from story2audio_shared.storage import ObjectStorage, voice_key

log = get_logger(__name__)

#: Repository root, four levels up from this file
#: (src/gateway/seed.py -> src/gateway -> src -> gateway -> services -> root).
REPO_ROOT = Path(__file__).resolve().parents[4]
VOICES_DIR = REPO_ROOT / "assets" / "voices"
CATALOGUE_FILE = VOICES_DIR / "speakers.json"


def _discover() -> dict[str, Path]:
    """Map display name to sample path.

    ``speakers.json`` is the v1 catalogue and is authoritative where it exists; any other
    WAV in the directory is picked up by filename. Names are stripped because several
    entries in the v1 file carry trailing spaces (``"Cristiano Ronaldo "``).
    """
    found: dict[str, Path] = {}

    if CATALOGUE_FILE.exists():
        raw = json.loads(CATALOGUE_FILE.read_text(encoding="utf-8"))
        for name, relative in raw.items():
            path = REPO_ROOT / "assets" / Path(relative)
            if not path.exists():
                path = VOICES_DIR / Path(relative).name
            if path.exists():
                found[name.strip()] = path

    for path in sorted(VOICES_DIR.glob("*.wav")):
        found.setdefault(path.stem.strip(), path)

    return found


async def seed_builtin_voices() -> int:
    """Register every built-in voice that is not already present.

    Returns the number of voices added.
    """
    limits = limit_settings()
    storage = ObjectStorage(storage_settings())
    storage.ensure_bucket()

    engine = create_engine()
    session_factory = create_session_factory(engine)
    added = 0

    try:
        async with session_factory() as session:
            for name, path in sorted(_discover().items()):
                existing = await session.scalar(
                    select(Voice.id).where(Voice.name == name, Voice.is_builtin.is_(True))
                )
                if existing is not None:
                    continue

                try:
                    validated = validate_voice_upload(
                        path.read_bytes(),
                        # Built-ins bypass the user-facing minimum: several reference
                        # samples in the v1 pack are shorter than the upload floor but
                        # are known-good, curated clones.
                        min_duration_seconds=0.5,
                        max_duration_seconds=max(limits.max_voice_duration_seconds, 600.0),
                        max_bytes=100 * 1024 * 1024,
                        # Clipped like any upload. The v1 reference pack is 30-second
                        # 48 kHz files, which are far larger than cloning needs.
                        clip_seconds=limits.reference_clip_seconds,
                    )
                except AppError as exc:
                    log.warning("builtin_voice_rejected", name=name, reason=exc.code.value)
                    continue

                voice_id = uuid7()
                key = voice_key(voice_id)
                storage.put_bytes(key, validated.wav_bytes, content_type="audio/wav")

                session.add(
                    Voice(
                        id=voice_id,
                        owner_id=None,
                        name=name,
                        storage_key=key,
                        duration_seconds=validated.duration_seconds,
                        sample_rate=validated.sample_rate,
                        is_builtin=True,
                    )
                )
                added += 1
                log.info(
                    "builtin_voice_seeded",
                    name=name,
                    duration_seconds=round(validated.duration_seconds, 2),
                )

            await session.commit()
    finally:
        await engine.dispose()

    return added


def main() -> None:
    configure_logging()
    added = asyncio.run(seed_builtin_voices())
    log.info("seed_complete", added=added)


if __name__ == "__main__":
    main()
