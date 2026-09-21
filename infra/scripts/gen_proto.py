"""Generate the gRPC stubs.

Stubs are generated rather than committed (`*_pb2.py` is gitignored), so the checked-in
`.proto` is unambiguously the source of truth and a stale stub cannot drift from it. CI
runs this before typechecking.

Each consumer gets its own copy under its own package. `tts-engine` deliberately shares
no Python with the rest of the workspace -- only this contract -- so a shared generated
package would reintroduce the coupling the split was meant to avoid.

Run with::

    uv run python infra/scripts/gen_proto.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTO_DIR = REPO_ROOT / "proto" / "tts" / "v1"
PROTO_FILE = PROTO_DIR / "tts.proto"

#: Where generated stubs land. One package per consumer.
TARGETS = (
    REPO_ROOT / "services" / "tts_engine" / "src" / "tts_engine" / "pb",
    REPO_ROOT / "services" / "tts_worker" / "src" / "tts_worker" / "pb",
)

PACKAGE_DOCSTRING = '''"""Generated gRPC stubs for `proto/tts/v1/tts.proto`.

Do not edit. Regenerate with `uv run python infra/scripts/gen_proto.py` (or `make proto`).
"""
'''


def generate(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)

    # Compiled with the proto's own directory on the include path, so the generated
    # module names are flat (`tts_pb2`) rather than carrying a `tts.v1` package that
    # would have to exist as a top-level import.
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"--proto_path={PROTO_DIR}",
            f"--python_out={target}",
            f"--pyi_out={target}",
            f"--grpc_python_out={target}",
            str(PROTO_FILE),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(f"protoc failed for {target}")

    _fix_grpc_import(target / "tts_pb2_grpc.py")
    (target / "__init__.py").write_text(PACKAGE_DOCSTRING, encoding="utf-8")


def _fix_grpc_import(path: Path) -> None:
    """Rewrite protoc's absolute sibling import into a relative one.

    protoc emits ``import tts_pb2``, which only resolves if the output directory happens
    to be on `sys.path`. Inside a package it has to be ``from . import tts_pb2``. This is
    a long-standing protoc limitation and the one-line rewrite is the standard fix.
    """
    source = path.read_text(encoding="utf-8")
    patched = re.sub(
        r"^import (tts_pb2)( as \w+)?$",
        r"from . import \1\2",
        source,
        flags=re.MULTILINE,
    )
    if patched == source:
        raise SystemExit(f"expected to rewrite a sibling import in {path}, found none")
    path.write_text(patched, encoding="utf-8")


def main() -> None:
    if not PROTO_FILE.exists():
        raise SystemExit(f"missing {PROTO_FILE}")
    for target in TARGETS:
        generate(target)
        sys.stdout.write(f"generated {target.relative_to(REPO_ROOT)}\n")


if __name__ == "__main__":
    main()
