from fastapi.testclient import TestClient

from pr_review_env.server.app import app


def test_required_endpoints():
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200

    tasks = client.get("/tasks")
    assert tasks.status_code == 200
    body = tasks.json()
    assert body["total"] >= 3
    assert "action_schema" in body

    reset = client.post("/reset", json={"task_id": "task1_security_basic"})
    assert reset.status_code == 200

    step = client.post("/step", json={
        "action": {
            "inline_comments": [
                {
                    "file_path": "app/auth.py",
                    "line_number": 11,
                    "comment": "SQL injection risk in query construction.",
                    "severity": "critical",
                    "category": "security"
                }
            ],
            "general_comments": [],
            "decision": {"decision": "request_changes", "summary": "Fix security issues."},
            "submit": True
        }
    })
    assert step.status_code == 200

    grader = client.post("/grader", json={
        "task_id": "task1_security_basic",
        "action": {
            "inline_comments": [
                {
                    "file_path": "app/auth.py",
                    "line_number": 11,
                    "comment": "SQL injection vulnerability.",
                    "severity": "critical",
                    "category": "security"
                }
            ],
            "general_comments": [],
            "decision": {"decision": "request_changes", "summary": "Fix critical issue."},
            "submit": True
        }
    })
    assert grader.status_code == 200
    grader_body = grader.json()
    assert 0.0 <= grader_body["feedback"]["score"] <= 1.0

    baseline = client.get("/baseline")
    # Baseline can return cache (200) or service unavailable when cache is absent and
    # no API key is configured for a live refresh run.
    assert baseline.status_code in (200, 503)
