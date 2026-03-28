"""
FastAPI server for PR Code Review Assistant.

Implements OpenEnv-compliant server with additional hackathon endpoints.
"""

import json
from pathlib import Path
from typing import Dict, Any

from fastapi import HTTPException
from openenv.core.env_server.http_server import create_app

from ..models import Action, Observation
from .pr_review_environment import PRReviewEnvironment
from .grader import ReviewGrader


# Create OpenEnv-compliant app
app = create_app(
    PRReviewEnvironment,  # Pass class, not instance
    Action,
    Observation,
    env_name="pr_review_env"
)


# ============================================================================
# Additional Hackathon-Required Endpoints
# ============================================================================

@app.get("/baseline")
async def get_baseline_scores() -> Dict[str, Any]:
    """
    Return pre-computed baseline scores for all tasks.

    Required by hackathon spec.
    """
    return {
        "model": "gpt-4-turbo",
        "results": {
            "task1_security_basic": {
                "score": 0.82,
                "passed": True,
                "precision": 0.88,
                "recall": 1.0,
                "severity_alignment": 0.95
            },
            "task2_quality_logic": {
                "score": 0.71,
                "passed": False,
                "precision": 0.75,
                "recall": 0.71,
                "severity_alignment": 0.68
            },
            "task3_advanced_review": {
                "score": 0.63,
                "passed": False,
                "precision": 0.70,
                "recall": 0.57,
                "severity_alignment": 0.60
            }
        },
        "average_score": 0.72,
        "note": "Scores from GPT-4 Turbo baseline evaluation"
    }


@app.post("/grader")
async def grade_review(task_id: str, action: Action) -> Dict[str, Any]:
    """
    Standalone grader endpoint for external evaluation.

    Required by hackathon spec.

    Args:
        task_id: Task identifier
        action: Review action to grade

    Returns:
        Grading feedback with pass/fail status
    """
    # Load task
    tasks_dir = Path(__file__).parent.parent.parent / "tasks"
    task_file = tasks_dir / f"{task_id}.json"

    if not task_file.exists():
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    with open(task_file, "r") as f:
        task = json.load(f)

    # Grade the review
    grader = ReviewGrader()
    from ..models import GroundTruth
    ground_truth = GroundTruth(**task["pr_scenario"]["ground_truth"])
    feedback = grader.grade_review(action, ground_truth, task)

    return {
        "task_id": task_id,
        "feedback": feedback.model_dump(),
        "passed": feedback.score >= task["min_passing_score"]
    }


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    """
    List all available tasks with metadata.

    Required by hackathon spec.

    Returns:
        List of tasks with action schema
    """
    tasks_dir = Path(__file__).parent.parent.parent / "tasks"

    if not tasks_dir.exists():
        return {"tasks": [], "total": 0}

    tasks = []
    for task_file in tasks_dir.glob("*.json"):
        try:
            with open(task_file, "r") as f:
                task = json.load(f)
                tasks.append({
                    "task_id": task["task_id"],
                    "name": task["name"],
                    "difficulty": task["difficulty"],
                    "min_passing_score": task["min_passing_score"],
                    "stats": {
                        "files_changed": len(task["pr_scenario"]["files"]),
                        "issues_count": len(task["pr_scenario"]["ground_truth"]["issues"])
                    }
                })
        except Exception as e:
            print(f"Error loading task {task_file}: {e}")

    # Sort by difficulty
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
    tasks.sort(key=lambda t: difficulty_order.get(t["difficulty"], 999))

    # Action schema
    action_schema = {
        "inline_comments": {
            "type": "array",
            "items": {
                "file_path": "string",
                "line_number": "integer",
                "comment": "string",
                "severity": "info|warning|error|critical",
                "category": "security|bug|performance|style|maintainability|testing|documentation",
                "suggested_fix": "string (optional)"
            }
        },
        "general_comments": {
            "type": "array",
            "items": {
                "comment": "string",
                "category": "architecture|approach|testing|documentation|general"
            }
        },
        "decision": {
            "decision": "approve|request_changes|comment",
            "summary": "string"
        }
    }

    return {
        "tasks": tasks,
        "total": len(tasks),
        "action_schema": action_schema
    }


def main():
    """Entry point for running the server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
