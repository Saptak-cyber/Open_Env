"""
FastAPI server for PR Code Review Assistant.

Implements OpenEnv-compliant server with additional hackathon endpoints.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import HTTPException
from pydantic import BaseModel, Field
from openenv.core.env_server.http_server import create_app

from ..models import Action, Observation
from .pr_review_environment import PRReviewEnvironment
from .grader import ReviewGrader

# Use a shared environment instance so reset/step state persists across HTTP requests.
_shared_env = PRReviewEnvironment()
_project_root = Path(__file__).parent.parent.parent
_baseline_cache_file = _project_root / "baseline_results_latest.json"


def _default_task_ids() -> List[str]:
    tasks_dir = _project_root / "tasks"
    task_ids: List[str] = []
    for task_file in sorted(tasks_dir.glob("*.json")):
        try:
            with open(task_file, "r", encoding="utf-8") as f:
                task = json.load(f)
                if isinstance(task.get("task_id"), str):
                    task_ids.append(task["task_id"])
        except Exception:
            continue
    return task_ids


class GraderRequest(BaseModel):
    """Request contract for deterministic standalone grading."""
    task_id: str = Field(description="Task identifier to use for grading")
    action: Action = Field(description="Final review action payload")

# Create OpenEnv-compliant app
app = create_app(
    lambda: _shared_env,
    Action,
    Observation,
    env_name="pr_review_env"
)


# ============================================================================
# Additional Hackathon-Required Endpoints
# ============================================================================

@app.get("/baseline")
async def get_baseline_scores(
    refresh: bool = False,
    provider: str = "openai",
    model: Optional[str] = None,
    env_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Return baseline scores for all tasks.

    By default this returns cached results. Set refresh=true to trigger inference.
    """
    # Fast path: cached baseline artifact
    if not refresh and _baseline_cache_file.exists():
        with open(_baseline_cache_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["source"] = "cache"
        return payload

    # Refresh path: run baseline inference through the baseline script class.
    try:
        from baseline.baseline_inference import BaselineAgent
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to import baseline agent: {e}"
        ) from e

    provider_env_key = "OPENAI_API_KEY" if provider == "openai" else "GROQ_API_KEY"
    if not os.getenv(provider_env_key):
        if _baseline_cache_file.exists():
            with open(_baseline_cache_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            payload["source"] = "cache_fallback_missing_api_key"
            return payload
        raise HTTPException(
            status_code=503,
            detail=(
                f"Missing {provider_env_key}. Set it to run live baseline, "
                "or call /baseline without refresh after generating cache."
            )
        )

    task_ids = _default_task_ids()
    if not task_ids:
        raise HTTPException(status_code=500, detail="No tasks found for baseline run.")

    baseline = BaselineAgent(provider=provider, model=model, env_base_url=env_url)
    results = baseline.run_evaluation(task_ids)
    results["provider"] = provider
    results["task_ids"] = task_ids

    with open(_baseline_cache_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    results["source"] = "live"
    return results


@app.post("/grader")
async def grade_review(request: GraderRequest) -> Dict[str, Any]:
    """
    Standalone grader endpoint for external evaluation.

    Required by hackathon spec.

    Args:
        request: GraderRequest with task_id and action

    Returns:
        Grading feedback with pass/fail status
    """
    # Load task
    tasks_dir = Path(__file__).parent.parent.parent / "tasks"
    task_file = tasks_dir / f"{request.task_id}.json"

    if not task_file.exists():
        raise HTTPException(status_code=404, detail=f"Task {request.task_id} not found")

    with open(task_file, "r") as f:
        task = json.load(f)

    # Grade the review
    grader = ReviewGrader()
    from ..models import GroundTruth
    ground_truth = GroundTruth(**task["pr_scenario"]["ground_truth"])
    feedback = grader.grade_review(request.action, ground_truth, task)

    return {
        "task_id": request.task_id,
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

    return {
        "tasks": tasks,
        "total": len(tasks),
        "action_schema": Action.model_json_schema()
    }


def main():
    """Entry point for running the server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
