"""Anonymous session identity.

A signed, HttpOnly cookie carrying a user id. No password, no email, no login screen —
enough to scope voices, job history and quotas to one person, and the `users` table
already has an ``email`` column so that adding real accounts later is a migration rather
than a redesign.

Why a cookie and not a bearer token in ``localStorage``: the token is readable by any
script that gets injected into the page, the cookie is not. The cost is that cookies are
subject to same-site rules, which is why :class:`GatewaySettings` exposes the domain and
SameSite policy — see the deployment note below.

**Cross-origin note.** The browser sends this cookie on the WebSocket handshake exactly as
it does on an HTTP request, so the same identity covers both. With the frontend on one
registrable domain and the API on another, ``SameSite=Lax`` drops the cookie entirely.
Deploy both under one registrable domain (``app.example.com`` / ``api.example.com``) with
``session_cookie_domain=".example.com"``, or fall back to ``SameSite=None; Secure`` with a
strict CORS allowlist.
"""

from __future__ import annotations

from uuid import UUID

from itsdangerous import BadSignature, URLSafeSerializer
from starlette.requests import HTTPConnection
from starlette.responses import Response

from gateway.settings import GatewaySettings

_SALT = "story2audio.session.v1"


class SessionCodec:
    """Signs and verifies the session cookie payload.

    The cookie is signed, not encrypted: its contents are a user id, which the API returns
    in responses anyway. What matters is that a client cannot forge one and adopt another
    user's voices or job history.
    """

    def __init__(self, settings: GatewaySettings) -> None:
        #: Public: the codec owns the cookie policy, and `write_session` reads it.
        self.settings = settings
        self._serializer = URLSafeSerializer(settings.session_secret.get_secret_value(), salt=_SALT)

    def dumps(self, user_id: UUID) -> str:
        return self._serializer.dumps({"uid": str(user_id)})

    def loads(self, raw: str) -> UUID | None:
        """Return the user id, or ``None`` if the cookie is forged, corrupt or stale.

        A bad cookie is never an error the caller sees: the gateway issues a fresh session
        instead. Rejecting the request would strand anyone whose cookie predates a secret
        rotation behind a failure they cannot clear themselves.
        """
        try:
            payload = self._serializer.loads(raw)
        except BadSignature:
            return None
        if not isinstance(payload, dict):
            return None
        raw_uid = payload.get("uid")
        if not isinstance(raw_uid, str):
            return None
        try:
            return UUID(raw_uid)
        except ValueError:
            return None


def read_session(connection: HTTPConnection, codec: SessionCodec, cookie_name: str) -> UUID | None:
    """Extract the user id from the request's cookies, if it carries a valid one.

    Takes an :class:`HTTPConnection` rather than a ``Request`` so the same function serves
    both HTTP routes and the WebSocket handshake.
    """
    raw = connection.cookies.get(cookie_name)
    if raw is None:
        return None
    return codec.loads(raw)


def write_session(response: Response, user_id: UUID, codec: SessionCodec) -> None:
    """Attach the session cookie to a response."""
    settings = codec.settings
    response.set_cookie(
        key=settings.session_cookie_name,
        value=codec.dumps(user_id),
        max_age=settings.session_max_age_seconds,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite=settings.samesite_value,  # type: ignore[arg-type]
        domain=settings.session_cookie_domain,
        path="/",
    )
