"""The job state machine.

Celery delivers at least once, so a duplicate task must not be able to move a job
backwards or resurrect a terminal one. These tests pin that property.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from story2audio_shared.enums import (
    ALLOWED_TRANSITIONS,
    CANCELLABLE_STATUSES,
    TERMINAL_STATUSES,
    JobStatus,
    can_transition,
)


def test_every_status_has_a_transition_entry() -> None:
    assert set(ALLOWED_TRANSITIONS) == set(JobStatus)


@pytest.mark.parametrize("status", sorted(TERMINAL_STATUSES))
def test_terminal_statuses_have_no_outgoing_transitions(status: JobStatus) -> None:
    assert ALLOWED_TRANSITIONS[status] == frozenset()


def test_happy_path_is_reachable() -> None:
    path = [
        JobStatus.QUEUED,
        JobStatus.WRITING,
        JobStatus.WRITTEN,
        JobStatus.SYNTHESIZING,
        JobStatus.DONE,
    ]
    for current, target in pairwise(path):
        assert can_transition(current, target)


def test_no_transition_ever_moves_backwards() -> None:
    order = {
        JobStatus.QUEUED: 0,
        JobStatus.WRITING: 1,
        JobStatus.WRITTEN: 2,
        JobStatus.SYNTHESIZING: 3,
    }
    for current, targets in ALLOWED_TRANSITIONS.items():
        for target in targets:
            if current in order and target in order:
                assert order[target] > order[current], f"{current} -> {target} moves backwards"


def test_terminal_states_cannot_be_re_entered_or_left() -> None:
    for terminal in TERMINAL_STATUSES:
        for target in JobStatus:
            assert not can_transition(terminal, target)


def test_cancellable_statuses_are_exactly_the_non_terminal_ones() -> None:
    assert frozenset(JobStatus) - TERMINAL_STATUSES == CANCELLABLE_STATUSES


def test_every_non_terminal_status_can_reach_failed_and_cancelled() -> None:
    for status in CANCELLABLE_STATUSES:
        assert can_transition(status, JobStatus.FAILED)
        assert can_transition(status, JobStatus.CANCELLED)
