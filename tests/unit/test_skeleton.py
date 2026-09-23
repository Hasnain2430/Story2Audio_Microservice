"""Phase 0 guard: the workspace installs and every service package imports.

This is deliberately trivial. It exists so that CI fails loudly if the workspace layout,
the editable installs or the shared-package wiring break, rather than failing later
inside a service that is hard to bisect.
"""

import importlib

import pytest

SERVICE_PACKAGES = [
    "story2audio_shared",
    "gateway",
    "story_worker",
    "tts_worker",
]


@pytest.mark.parametrize("module_name", SERVICE_PACKAGES)
def test_service_package_imports_and_declares_a_version(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.__version__ == "2.0.0"
