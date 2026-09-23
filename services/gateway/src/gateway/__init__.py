"""Public REST + WebSocket API.

Accepts a job, validates it, enqueues it, and returns a job id. Nothing in this service
is allowed to block on generation work.
"""

__version__ = "2.0.0"
