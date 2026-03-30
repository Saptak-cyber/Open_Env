"""
OpenEnv inference runner for PR Review environment.

MANDATORY hackathon variables:
- API_BASE_URL: OpenAI-compatible LLM endpoint
- MODEL_NAME: model identifier
- HF_TOKEN: API key
"""

import argparse
import json
import os
import re
from statistics import mean
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
ENV_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:8000")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1800"))
DEFAULT_OUTPUT = "inference_results.json"

VALID_SEVERITIES = {"info", "warning", "error", "critical"}
VALID_INLINE_CATEGORIES = {
    "security", "bug", "performance", "style", "maintainability", "testing", "documentation"
}
VALID_GENERAL_CATEGORIES = {"architecture", "approach", "testing", "documentation", "general"}
VALID_DECISIONS = {"approve", "request_changes", "comment"}

SYSTEM_PROMPT = """You are an expert software code reviewer.

Review the PR diff and output ONLY valid JSON with this schema:
{
  "inline_comments": [
    {
      "file_path": "string",
      "line_number": 123,
      "comment": "string",
      "severity": "info|warning|error|critical",
      "category": "security|bug|performance|style|maintainability|testing|documentation",
      "suggested_fix": "optional string"
    }
  ],
  "general_comments": [
    {
      "comment": "string",
      "category": "architecture|approach|testing|documentation|general"
    }
  ],
  "decision": {
    "decision": "approve|request_changes|comment",
    "summary": "string"
  }
}

Rules:
- Be concise and evidence-based.
- Focus on high-impact findings.
- Avoid duplicate comments on the same root cause.
"""


def require_env() -> None:
    missing = []
    if not API_KEY:
        missing.append("HF_TOKEN")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")


def extract_observation(payload: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
    if "pr_state" in payload:
        return payload
    observation = payload.get("observation")
    if isinstance(observation, dict) and "pr_state" in observation:
        return observation
    raise ValueError(f"Unexpected {endpoint} response payload keys: {list(payload.keys())}")


def build_prompt(pr_state: Dict[str, Any], task_id: str) -> str:
    md = pr_state.get("metadata", {})
    lines = [
        f"Task ID: {task_id}",
        f"PR: {pr_state.get('pr_id', 'unknown')}",
        f"Title: {md.get('title', '')}",
        f"Description: {md.get('description', '')}",
        f"Author: {md.get('author', '')}",
        "",
        "Changed files:",
    ]

    for file_obj in pr_state.get("files", []):
        path = file_obj.get("path", "unknown")
        lang = file_obj.get("language", "unknown")
        additions = file_obj.get("additions", [])[:80]
        deletions = file_obj.get("deletions", [])[:40]
        context = file_obj.get("context", [])[:30]

        lines.append(f"\n## {path} ({lang})")
        if context:
            lines.append("Context:")
            lines.extend(f"  {c}" for c in context)
        if additions:
            lines.append("Added lines:")
            lines.extend(f"+ {a}" for a in additions)
        if deletions:
            lines.append("Deleted lines:")
            lines.extend(f"- {d}" for d in deletions)

    lines.append("\nReturn JSON only. No markdown.")
    return "\n".join(lines)


def parse_json_response(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, flags=re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))
        raise


def _parse_line_number(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        m = re.search(r"\d+", value)
        if m:
            return int(m.group(0))
    return None


def normalize_action(raw_action: Dict[str, Any], max_inline_comments: int = 8) -> Dict[str, Any]:
    if not isinstance(raw_action, dict):
        raw_action = {}

    inline_comments = []
    seen = set()
    for c in raw_action.get("inline_comments", []) or []:
        if not isinstance(c, dict):
            continue
        file_path = c.get("file_path")
        line_number = _parse_line_number(c.get("line_number"))
        comment = c.get("comment")
        severity = str(c.get("severity", "warning")).lower()
        category = str(c.get("category", "bug")).lower()

        if not isinstance(file_path, str) or not file_path.strip():
            continue
        if line_number is None:
            continue
        if not isinstance(comment, str) or not comment.strip():
            continue

        if severity not in VALID_SEVERITIES:
            severity = "warning"
        if category not in VALID_INLINE_CATEGORIES:
            category = "bug"

        dedupe_key = (file_path.strip(), line_number, category)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        item = {
            "file_path": file_path.strip(),
            "line_number": line_number,
            "comment": comment.strip(),
            "severity": severity,
            "category": category,
        }
        suggested_fix = c.get("suggested_fix")
        if isinstance(suggested_fix, str) and suggested_fix.strip():
            item["suggested_fix"] = suggested_fix.strip()
        inline_comments.append(item)

    general_comments = []
    for c in raw_action.get("general_comments", []) or []:
        if not isinstance(c, dict):
            continue
        comment = c.get("comment")
        category = str(c.get("category", "general")).lower()
        if not isinstance(comment, str) or not comment.strip():
            continue
        if category not in VALID_GENERAL_CATEGORIES:
            category = "general"
        general_comments.append({"comment": comment.strip(), "category": category})

    decision_obj = raw_action.get("decision")
    if not isinstance(decision_obj, dict):
        decision_obj = {}
    decision = str(decision_obj.get("decision", "comment")).lower()
    if decision not in VALID_DECISIONS:
        decision = "comment"
    summary = decision_obj.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = "Automated review generated by inference script."

    return {
        "inline_comments": inline_comments[:max_inline_comments],
        "general_comments": general_comments,
        "decision": {"decision": decision, "summary": summary.strip()},
        "submit": True,
    }


class InferenceRunner:
    def __init__(
        self,
        env_url: str,
        output: str,
        task_ids: Optional[List[str]] = None,
        temperature: float = TEMPERATURE,
    ) -> None:
        self.env_url = env_url.rstrip("/")
        self.output = output
        self.task_ids = task_ids
        self.temperature = temperature
        self.http = requests.Session()
        self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    def _discover_tasks(self) -> List[str]:
        if self.task_ids:
            return self.task_ids
        resp = self.http.get(f"{self.env_url}/tasks", timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        tasks = payload.get("tasks", [])
        return [t["task_id"] for t in tasks if isinstance(t, dict) and t.get("task_id")]

    def _review_with_model(self, pr_state: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        prompt = build_prompt(pr_state, task_id)
        completion = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
        return normalize_action(parse_json_response(content))

    def run(self) -> Dict[str, Any]:
        task_ids = self._discover_tasks()
        if not task_ids:
            raise RuntimeError("No tasks discovered. Ensure /tasks endpoint is available.")

        print(f"Found {len(task_ids)} task(s): {', '.join(task_ids)}")

        results: Dict[str, Any] = {}
        for task_id in task_ids:
            print(f"\n{'=' * 60}\nEvaluating {task_id}\n{'=' * 60}")
            reset_resp = self.http.post(
                f"{self.env_url}/reset",
                json={"task_id": task_id},
                timeout=30,
            )
            reset_resp.raise_for_status()
            reset_obs = extract_observation(reset_resp.json(), "/reset")

            action = self._review_with_model(reset_obs["pr_state"], task_id)
            print(f"Generated {len(action['inline_comments'])} inline comments")
            print(f"Decision: {action['decision']['decision']}")

            step_resp = self.http.post(
                f"{self.env_url}/step",
                json={"action": action},
                timeout=30,
            )
            step_resp.raise_for_status()
            step_obs = extract_observation(step_resp.json(), "/step")
            feedback = step_obs.get("feedback") or {}

            score = float(feedback.get("score", 0.0))
            passed = bool(step_obs.get("metadata", {}).get("passed", False))
            results[task_id] = {
                "score": score,
                "passed": passed,
                "precision": feedback.get("precision", 0.0),
                "recall": feedback.get("recall", 0.0),
                "severity_alignment": feedback.get("severity_alignment", 0.0),
                "true_positives": feedback.get("true_positives", 0),
                "false_positives": feedback.get("false_positives", 0),
                "false_negatives": feedback.get("false_negatives", 0),
            }
            print(
                f"Score: {score:.2f} | Passed: {'yes' if passed else 'no'} | "
                f"Precision: {results[task_id]['precision']:.2f} | "
                f"Recall: {results[task_id]['recall']:.2f}"
            )

        avg = mean([r["score"] for r in results.values()]) if results else 0.0
        payload = {
            "provider": "openai_compatible",
            "api_base_url": API_BASE_URL,
            "model": MODEL_NAME,
            "temperature": self.temperature,
            "env_url": self.env_url,
            "task_ids": task_ids,
            "results": results,
            "average_score": avg,
        }
        with open(self.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nAverage score: {avg:.2f}")
        print(f"Saved results to {self.output}")
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenEnv PR review inference.")
    parser.add_argument(
        "--env-url",
        default=ENV_URL,
        help="OpenEnv server base URL (default: OPENENV_BASE_URL or http://localhost:8000)",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
        help="Optional task IDs to evaluate (default: discover from /tasks)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Path to output JSON file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Sampling temperature for model calls (default from TEMPERATURE env or 0.0)",
    )
    return parser.parse_args()


def main() -> None:
    require_env()
    args = parse_args()
    runner = InferenceRunner(
        env_url=args.env_url,
        output=args.output,
        task_ids=args.task_ids,
        temperature=args.temperature,
    )
    runner.run()


if __name__ == "__main__":
    main()
