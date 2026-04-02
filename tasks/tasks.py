"""Task definitions for the Email Triage RL environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple


VALID_LABELS: Tuple[str, ...] = ("spam", "important", "urgent", "normal")


def _normalize_label(label: str) -> str:
    return label.strip().lower()


def _default_grader(predicted: str, expected: str) -> float:
    """Return score in [0.0, 1.0].

    - 1.0 for exact match
    - 0.5 for close operational overlap (important <-> urgent)
    - 0.3 for weak overlap (spam <-> important)
    - 0.2 for mild operational overlap involving `normal`
    - 0.0 otherwise
    """
    predicted_norm = _normalize_label(predicted)
    expected_norm = _normalize_label(expected)

    if predicted_norm not in VALID_LABELS:
        return 0.0
    if predicted_norm == expected_norm:
        return 1.0
    if (predicted_norm, expected_norm) in {("important", "urgent"), ("urgent", "important")}:
        return 0.5
    if (predicted_norm, expected_norm) in {("important", "spam"), ("spam", "important")}:
        return 0.3
    if (predicted_norm, expected_norm) in {
        ("normal", "important"),
        ("important", "normal"),
        ("normal", "urgent"),
        ("urgent", "normal"),
    }:
        return 0.2
    return 0.0


@dataclass(frozen=True)
class EmailTask:
    """Single evaluation task with email text and grading function."""

    name: str
    difficulty: str
    input_email: str
    expected_output: str
    grader: Callable[[str, str], float]


TASKS: List[EmailTask] = [
    EmailTask(
        name="task_easy_clear_spam",
        difficulty="easy",
        input_email=(
            "Subject: Congrats, you won a free gift card\n\n"
            "Claim your prize now by clicking this link and confirming your bank details."
        ),
        expected_output="spam",
        grader=_default_grader,
    ),
    EmailTask(
        name="task_medium_ops_followup",
        difficulty="medium",
        input_email=(
            "Subject: Incident notes + vendor invoice due today\n\n"
            "Yesterday's checkout incident has been resolved, no customer impact now.\n"
            "Please approve vendor invoice #4821 before end of day to avoid late fees."
        ),
        expected_output="important",
        grader=_default_grader,
    ),
    EmailTask(
        name="task_hard_bec_finance_phish",
        difficulty="hard",
        input_email=(
            "Subject: RE: invoice #9912 settlement required today\n\n"
            "From: cfo-finance@outlook-mail-support.com\n"
            "I am in a meeting and cannot talk. Process the attached invoice urgently and\n"
            "wire payment today, then confirm only by reply email."
        ),
        expected_output="spam",
        grader=_default_grader,
    ),
]


def get_tasks() -> List[EmailTask]:
    """Return all tasks in evaluation order (easy -> medium -> hard)."""
    return TASKS
