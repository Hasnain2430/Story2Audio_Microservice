"""Write the gateway's OpenAPI schema to `openapi.json`.

The frontend generates its TypeScript types from this file, so the two sides cannot
drift: a route or field that changes shape here becomes a type error in the client.

Dumped to a committed file rather than fetched from a running server, so type generation
works in CI and in an editor with nothing running. Regenerating in CI and diffing is what
catches a schema change that was never propagated.

Run with::

    uv run python infra/scripts/dump_openapi.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "openapi.json"


def main() -> None:
    from gateway.main import create_app

    schema = create_app().openapi()

    # Stable key order and a trailing newline, so the file only changes when the schema
    # does -- otherwise every regeneration would look like a diff.
    OUTPUT.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(f"wrote {OUTPUT.relative_to(REPO_ROOT)} ({len(schema['paths'])} paths)\n")


if __name__ == "__main__":
    main()
