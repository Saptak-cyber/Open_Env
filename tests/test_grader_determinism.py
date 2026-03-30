import json
import os
import socket
import subprocess
from pathlib import Path

import pytest

from pr_review_env import Action, InlineComment, ReviewDecision
from pr_review_env.models import GroundTruth
from pr_review_env.server.grader import ReviewGrader


def _load_task(task_id: str) -> dict:
    root = Path(__file__).resolve().parent.parent
    with open(root / "tasks" / f"{task_id}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _task2_action() -> Action:
    return Action(
        inline_comments=[
            InlineComment(
                file_path="services/payment.py",
                line_number=47,
                comment="Race condition in check-then-update; not atomic.",
                severity="error",
                category="security",
            ),
            InlineComment(
                file_path="models/user.py",
                line_number=18,
                comment="Password stored in plaintext; must hash with bcrypt.",
                severity="critical",
                category="security",
            ),
            InlineComment(
                file_path="utils/validator.py",
                line_number=90,
                comment="Regex '.*@.*' is too permissive and accepts invalid emails.",
                severity="warning",
                category="bug",
            ),
        ],
        general_comments=[],
        decision=ReviewDecision(
            decision="request_changes",
            summary="Found security and validation issues."
        ),
        submit=True,
    )


def test_grader_is_deterministic_for_same_input():
    task = _load_task("task2_quality_logic")
    grader = ReviewGrader()
    action = _task2_action()
    gt = GroundTruth(**task["pr_scenario"]["ground_truth"])

    first = grader.grade_review(action, gt, task)
    second = grader.grade_review(action, gt, task)

    assert first.model_dump() == second.model_dump()


def test_grader_edge_cases_score_range():
    task = _load_task("task2_quality_logic")
    grader = ReviewGrader()
    gt = GroundTruth(**task["pr_scenario"]["ground_truth"])

    # empty action
    empty = Action(
        inline_comments=[],
        general_comments=[],
        decision=ReviewDecision(decision="comment", summary="No findings."),
        submit=True,
    )
    empty_feedback = grader.grade_review(empty, gt, task)
    assert 0.0 <= empty_feedback.score <= 1.0

    # duplicate comments
    dup = Action(
        inline_comments=[
            InlineComment(
                file_path="models/user.py",
                line_number=18,
                comment="Plaintext password storage vulnerability.",
                severity="critical",
                category="security",
            ),
            InlineComment(
                file_path="models/user.py",
                line_number=18,
                comment="Password is not hashed before storage.",
                severity="critical",
                category="security",
            ),
        ],
        general_comments=[],
        decision=ReviewDecision(decision="request_changes", summary="Security issue."),
        submit=True,
    )
    dup_feedback = grader.grade_review(dup, gt, task)
    assert 0.0 <= dup_feedback.score <= 1.0

    # wrong file right line should not get full credit
    wrong_file = Action(
        inline_comments=[
            InlineComment(
                file_path="wrong/path.py",
                line_number=18,
                comment="Plaintext password vulnerability.",
                severity="critical",
                category="security",
            ),
        ],
        general_comments=[],
        decision=ReviewDecision(decision="request_changes", summary="Issue found."),
        submit=True,
    )
    wrong_feedback = grader.grade_review(wrong_file, gt, task)
    assert wrong_feedback.true_positives <= dup_feedback.true_positives
    assert 0.0 <= wrong_feedback.score <= 1.0


def test_grader_is_not_constant():
    task = _load_task("task2_quality_logic")
    grader = ReviewGrader()
    gt = GroundTruth(**task["pr_scenario"]["ground_truth"])

    empty = Action(
        inline_comments=[],
        general_comments=[],
        decision=ReviewDecision(decision="comment", summary="No findings."),
        submit=True,
    )
    good = _task2_action()

    s_empty = grader.grade_review(empty, gt, task).score
    s_good = grader.grade_review(good, gt, task).score

    assert s_good != s_empty
    assert s_good > s_empty


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


@pytest.mark.skipif(
    not os.getenv("RUN_BASELINE_REPRO_TEST"),
    reason="Set RUN_BASELINE_REPRO_TEST=1 to run networked reproducibility check.",
)
def test_baseline_reproducibility_is_bounded(tmp_path: Path):
    """
    Optional integration test.
    Requires local env server on :8000 and HF_TOKEN/MODEL_NAME/API_BASE_URL configured.
    """
    assert _is_port_open("127.0.0.1", 8000), "Environment server must be running on port 8000."
    assert os.getenv("HF_TOKEN"), "HF_TOKEN required for reproducibility check."
    assert os.getenv("MODEL_NAME"), "MODEL_NAME required for reproducibility check."
    assert os.getenv("API_BASE_URL"), "API_BASE_URL required for reproducibility check."

    out1 = tmp_path / "run1.json"
    out2 = tmp_path / "run2.json"
    cmd = [
        "python3",
        "inference.py",
        "--env-url", "http://localhost:8000",
    ]

    subprocess.run(cmd + ["--output", str(out1)], check=True)
    subprocess.run(cmd + ["--output", str(out2)], check=True)

    r1 = json.loads(out1.read_text(encoding="utf-8"))
    r2 = json.loads(out2.read_text(encoding="utf-8"))
    assert abs(r1["average_score"] - r2["average_score"]) <= 0.08
