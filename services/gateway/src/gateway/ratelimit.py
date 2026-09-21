"""Rate limits and spend guardrails.

An open endpoint that spends GPU money per request needs a ceiling it cannot exceed, not a
billing alert after it already has. v1 had neither: anything that could reach `:50051` or
`:8000` could queue unbounded GPU work.

Three different problems get three different mechanisms, because one shape does not fit
all of them:

============================  ==========================================================
Hourly per-session limits     Sliding-window log in a Redis sorted set. A fixed window
                              would let a caller spend their whole allowance at 10:59 and
                              again at 11:00.
Concurrent jobs per session   A count of non-terminal rows, taken in the same transaction
                              as the insert. A Redis counter drifts permanently the first
                              time a worker dies without decrementing it.
Global daily spend cap        ``INCR`` with a TTL. This one genuinely *is* a fixed window,
                              so the simple mechanism is also the correct one.
============================  ==========================================================

Only the first two live here; the concurrency check needs the database session and belongs
with the job creation logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from redis.asyncio import Redis

# Check-and-add in one round trip, so two concurrent requests cannot both observe a count
# below the limit and both be admitted.
#
# KEYS[1] = window key
# ARGV    = now_ms, window_ms, limit, member
_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]

redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
local used = redis.call('ZCARD', key)
if used >= limit then
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry_after = window
    if oldest[2] then
        retry_after = (tonumber(oldest[2]) + window) - now
    end
    return {0, used, retry_after}
end

redis.call('ZADD', key, now, member)
redis.call('PEXPIRE', key, window)
return {1, used + 1, 0}
"""


@dataclass(frozen=True, slots=True)
class LimitDecision:
    """Outcome of one limit check."""

    allowed: bool
    used: int
    limit: int
    #: Seconds until the caller could succeed. Surfaced as the ``Retry-After`` header.
    retry_after_seconds: int

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)


class RateLimiter:
    """Redis-backed limits.

    Every method is fail-closed on a limit breach and fail-open on a Redis outage: if the
    limiter itself cannot answer, the caller decides. That choice is deliberate — a demo
    that stops serving because Redis blipped is worse than one that briefly over-serves,
    and the global daily cap is the backstop that actually protects spend.
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self._script = redis.register_script(_SLIDING_WINDOW_LUA)

    async def check_sliding_window(
        self,
        *,
        bucket: str,
        subject: UUID | str,
        limit: int,
        window_seconds: int,
        token: str,
    ) -> LimitDecision:
        """Consume one unit from a per-subject sliding window.

        ``token`` must be unique per attempt — a UUID works. Sorted-set members are
        deduplicated, so a repeated token would silently overwrite an earlier entry and
        hand the caller a free request.
        """
        key = f"rl:{bucket}:{subject}"
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        window_ms = window_seconds * 1000

        raw = await self._script(keys=[key], args=[now_ms, window_ms, limit, token])
        allowed, used, retry_after_ms = (int(value) for value in raw)

        return LimitDecision(
            allowed=bool(allowed),
            used=used,
            limit=limit,
            retry_after_seconds=max(1, -(-retry_after_ms // 1000)) if not allowed else 0,
        )

    async def check_daily_cap(self, *, cap: int) -> LimitDecision:
        """Consume one unit from the global daily job cap.

        This is the hard spend ceiling. It is global rather than per-session on purpose:
        anonymous sessions are free to create, so a per-session limit alone bounds nothing.
        """
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        key = f"cap:jobs:{today}"

        used = int(await self._redis.incr(key))
        if used == 1:
            # First increment of the day creates the key; expire it a day later so the
            # counter cannot outlive its window if the process restarts.
            await self._redis.expire(key, 60 * 60 * 25)

        if used > cap:
            seconds_left = _seconds_until_utc_midnight()
            return LimitDecision(
                allowed=False, used=used - 1, limit=cap, retry_after_seconds=seconds_left
            )
        return LimitDecision(allowed=True, used=used, limit=cap, retry_after_seconds=0)

    async def release_daily_cap(self) -> None:
        """Give back one unit of the daily cap.

        Called when a job is rejected *after* the cap was consumed — by a later validation
        failure, say. Without this, a burst of invalid requests would burn the day's
        budget without generating anything.
        """
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        await self._redis.decr(f"cap:jobs:{today}")


def _seconds_until_utc_midnight() -> int:
    now = datetime.now(UTC)
    tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() + 86_400
    return max(1, int(tomorrow - now.timestamp()))
