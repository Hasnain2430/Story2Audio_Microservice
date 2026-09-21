"""S3-compatible object storage.

Audio and reference voices live here, never in the request path. v1 returned rendered
audio as protobuf ``bytes`` — which is why both ends had to raise their gRPC message limit
to 100 MB — and wrote every upload to a fixed filename in the process working directory,
so two concurrent requests silently overwrote each other.

Backed by MinIO locally and Cloudflare R2 in production. R2 is the production choice
specifically because it charges nothing for egress, which for an audio app is the line
item that matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import IO, TYPE_CHECKING, Final
from uuid import UUID

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from story2audio_shared.config import StorageSettings, storage_settings
from story2audio_shared.enums import AudioFormat
from story2audio_shared.errors import AppError, ErrorCode
from story2audio_shared.logging import get_logger

if TYPE_CHECKING:
    from types_boto3_s3.client import S3Client

log = get_logger(__name__)

VOICE_PREFIX: Final = "voices"
AUDIO_PREFIX: Final = "audio"
#: Per-segment audio, published during synthesis so playback can start early. Kept
#: under its own top-level prefix rather than beside the finished track, so the
#: expiry rule can target it with no chance of matching the real output.
SEGMENT_PREFIX: Final = "segments"
#: How long a streamed segment survives. One day is the smallest an S3 lifecycle rule
#: can express, and far longer than the minutes a client needs it for.
SEGMENT_RETENTION_DAYS: Final = 1

_CONTENT_TYPES: Final[dict[AudioFormat, str]] = {
    AudioFormat.MP3: "audio/mpeg",
    AudioFormat.WAV: "audio/wav",
}


def voice_key(voice_id: UUID) -> str:
    """Storage key for a reference voice sample.

    Derived from the voice id, so two uploads can never collide. v1 derived output paths
    from the prompt text, which meant two users submitting the same prompt wrote to — and
    then read back from — the same file.
    """
    return f"{VOICE_PREFIX}/{voice_id}.wav"


def audio_key(job_id: UUID, audio_format: AudioFormat) -> str:
    """Storage key for a job's rendered audio."""
    return f"{AUDIO_PREFIX}/{job_id}.{audio_format.value}"


def segment_audio_key(job_id: UUID, index: int) -> str:
    """Storage key for one rendered segment, published before the job finishes.

    These exist so playback can start while synthesis is still running, and are
    superseded by the assembled track the moment it is uploaded. They are expired by the
    lifecycle rule in :meth:`ObjectStorage.ensure_bucket` rather than deleted inline: a
    client is usually still playing the early segments when the last one finishes, and a
    job that failed half-way should leave what it did render addressable.
    """
    return f"{SEGMENT_PREFIX}/{job_id}/segment-{index:04d}.wav"


@dataclass(frozen=True, slots=True)
class PresignedUrl:
    """A time-limited URL and the moment it stops working."""

    url: str
    expires_at: datetime


class ObjectStorage:
    """Thin, typed wrapper over the S3 API.

    Deliberately small: put, get, presign, delete, exists. Anything richer belongs in the
    caller, and keeping the surface narrow is what makes swapping MinIO for R2 a config
    change.

    Every botocore failure is translated into :class:`AppError` with
    ``STORAGE_UNAVAILABLE``, so storage problems reach the client as one classified,
    retryable error rather than as a leaked vendor exception.
    """

    def __init__(self, settings: StorageSettings | None = None) -> None:
        self._settings = settings or storage_settings()
        self._client: S3Client = boto3.client(
            "s3",
            endpoint_url=self._settings.s3_endpoint_url,
            region_name=self._settings.s3_region,
            aws_access_key_id=self._settings.s3_access_key_id,
            aws_secret_access_key=self._settings.s3_secret_access_key.get_secret_value(),
            config=Config(
                signature_version="s3v4",
                s3={
                    "addressing_style": (
                        "path" if self._settings.s3_force_path_style else "virtual"
                    )
                },
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

        # A second client bound to the browser-reachable address, used only for signing.
        # The signature covers the host, so a URL signed against the internal endpoint is
        # not merely unreachable from a browser -- rewriting its host afterwards would
        # invalidate it.
        public_endpoint = self._settings.s3_public_endpoint_url
        self._signing_client: S3Client = (
            self._client
            if not public_endpoint or public_endpoint == self._settings.s3_endpoint_url
            else boto3.client(
                "s3",
                endpoint_url=public_endpoint,
                region_name=self._settings.s3_region,
                aws_access_key_id=self._settings.s3_access_key_id,
                aws_secret_access_key=self._settings.s3_secret_access_key.get_secret_value(),
                config=Config(
                    signature_version="s3v4",
                    s3={
                        "addressing_style": (
                            "path" if self._settings.s3_force_path_style else "virtual"
                        )
                    },
                ),
            )
        )

    @property
    def bucket(self) -> str:
        return self._settings.s3_bucket

    def put_bytes(self, key: str, data: bytes, *, content_type: str) -> None:
        """Upload an in-memory object."""
        try:
            self._client.put_object(
                Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
            )
        except (BotoCoreError, ClientError) as exc:
            raise AppError(
                ErrorCode.STORAGE_UNAVAILABLE, detail=f"put_object failed for {key}: {exc}"
            ) from exc

    def put_stream(self, key: str, stream: IO[bytes], *, content_type: str) -> None:
        """Upload from a file-like object without reading it fully into memory."""
        try:
            self._client.upload_fileobj(
                stream, self.bucket, key, ExtraArgs={"ContentType": content_type}
            )
        except (BotoCoreError, ClientError) as exc:
            raise AppError(
                ErrorCode.STORAGE_UNAVAILABLE, detail=f"upload_fileobj failed for {key}: {exc}"
            ) from exc

    def put_audio(self, key: str, data: bytes, audio_format: AudioFormat) -> None:
        """Upload rendered audio with the right content type for in-browser playback."""
        self.put_bytes(key, data, content_type=_CONTENT_TYPES[audio_format])

    def get_bytes(self, key: str) -> bytes:
        """Download an object into memory."""
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except (BotoCoreError, ClientError) as exc:
            raise AppError(
                ErrorCode.STORAGE_UNAVAILABLE, detail=f"get_object failed for {key}: {exc}"
            ) from exc

    def presign_get(self, key: str, *, ttl_seconds: int | None = None) -> PresignedUrl:
        """Return a short-lived download URL.

        This is how audio reaches the browser: the API response carries a URL, not bytes.
        The TTL is deliberately short — a leaked URL should stop working quickly.
        """
        ttl = ttl_seconds if ttl_seconds is not None else self._settings.presigned_url_ttl_seconds
        try:
            url = self._signing_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=ttl,
            )
        except (BotoCoreError, ClientError) as exc:
            raise AppError(
                ErrorCode.STORAGE_UNAVAILABLE, detail=f"presign failed for {key}: {exc}"
            ) from exc
        return PresignedUrl(url=url, expires_at=datetime.now(UTC) + timedelta(seconds=ttl))

    def delete(self, key: str) -> None:
        """Delete an object. Succeeds whether or not the key existed."""
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
        except (BotoCoreError, ClientError) as exc:
            raise AppError(
                ErrorCode.STORAGE_UNAVAILABLE, detail=f"delete_object failed for {key}: {exc}"
            ) from exc

    def exists(self, key: str) -> bool:
        """Return whether an object exists."""
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise AppError(
                ErrorCode.STORAGE_UNAVAILABLE, detail=f"head_object failed for {key}: {exc}"
            ) from exc
        except BotoCoreError as exc:
            raise AppError(
                ErrorCode.STORAGE_UNAVAILABLE, detail=f"head_object failed for {key}: {exc}"
            ) from exc
        return True

    def ensure_bucket(self) -> None:
        """Create the bucket if it does not exist.

        For local MinIO and CI only. In a deployed environment the bucket is provisioned
        out of band and the application's credentials should not be able to create one.
        """
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except ClientError:
            try:
                self._client.create_bucket(Bucket=self.bucket)
            except (BotoCoreError, ClientError) as exc:
                raise AppError(
                    ErrorCode.STORAGE_UNAVAILABLE,
                    detail=f"create_bucket failed for {self.bucket}: {exc}",
                ) from exc

        self._expire_segments()

    def _expire_segments(self) -> None:
        """Expire streamed segments, so early playback does not cost storage forever.

        Segments are WAV, and a story's worth of them runs about three times the size of
        its finished MP3. Without this they accumulate for every job ever run — a real
        bill rather than a rounding error, and one that grows fastest exactly when the
        product is being used.

        Best effort. A deployment where the application's credentials cannot write bucket
        policy is the *expected* one — there the bucket is provisioned out of band and
        this rule is applied alongside it — so a refusal is logged rather than raised.
        """
        try:
            self._client.put_bucket_lifecycle_configuration(
                Bucket=self.bucket,
                LifecycleConfiguration={
                    "Rules": [
                        {
                            "ID": "expire-streamed-segments",
                            "Status": "Enabled",
                            "Filter": {"Prefix": f"{SEGMENT_PREFIX}/"},
                            "Expiration": {"Days": SEGMENT_RETENTION_DAYS},
                        }
                    ]
                },
            )
        except (BotoCoreError, ClientError) as exc:
            log.warning("segment_lifecycle_not_applied", bucket=self.bucket, error=str(exc))


@lru_cache(maxsize=1)
def object_storage() -> ObjectStorage:
    """Process-wide storage client.

    botocore clients are thread-safe and expensive to construct, so one per process is
    both correct and the cheap option.
    """
    return ObjectStorage()
