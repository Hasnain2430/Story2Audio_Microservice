"""Shared domain for Story2Audio.

Everything defined here is imported by the gateway and by both workers, so that job
statuses, request schemas and prompt templates have exactly one definition and cannot
drift between services.
"""

__version__ = "2.0.0"
