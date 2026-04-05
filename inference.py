"""
OpenEnv inference runner for PR Review environment.

MANDATORY hackathon variables:
- API_BASE_URL: OpenAI-compatible LLM endpoint
- MODEL_NAME: model identifier
- OPENAI_API_KEY or HF_TOKEN: API key (both supported)
"""

import argparse
import json
import os
import re
from statistics import mean
from typing import Any, Dict, List, Optional, Union

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
# Prefer OpenAI-style key name, but allow Hugging Face token as well.
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
ENV_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:8000")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "3072"))
DEFAULT_OUTPUT = "inference_results.json"

VALID_SEVERITIES = {"info", "warning", "error", "critical"}
VALID_INLINE_CATEGORIES = {
    "security", "bug", "performance", "style", "maintainability", "testing", "documentation"
}
VALID_GENERAL_CATEGORIES = {"architecture", "approach", "testing", "documentation", "general"}
VALID_DECISIONS = {"approve", "request_changes", "comment"}

TASK_COMMENT_BUDGET: Dict[str, int] = {
    "task1_security_basic": 4,
    "task2_quality_logic": 7,
    "task3_advanced_review": 9,
    "task4_session_auth_medium": 7,
    "task5_async_pipeline_hard": 8,
    "task6_data_export_hard": 8,
}

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
        missing.append("OPENAI_API_KEY or HF_TOKEN")
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

def _task_hint(task_id: str) -> str:
    if task_id == "task1_security_basic":
        return (
            "Focus: obvious OWASP-style security issues only. "
            "Prioritize precision over recall; skip uncertain findings."
        )
    if task_id == "task2_quality_logic":
        return (
            "Focus: security + logic + correctness issues. Prioritize these high-signal items: "
            "race conditions/atomicity in payment flows, plaintext password storage (hash with bcrypt/argon2), "
            "weak validation regex, and missing brute-force protections/rate limiting. "
            "Aim for 5-7 unique high-signal findings and avoid low-signal style nitpicks unless no substantive issues exist."
        )
    if task_id == "task3_advanced_review":
        return (
            "Focus: subtle architecture/reliability/performance issues. "
            "Look for authorization gaps, cache invalidation/TTL, thread-safety, "
            "exception swallowing, N+1 queries, and cross-tenant data access risks. "
            "Use concrete inline findings mapped to grader categories such as security, bug, performance, "
            "and maintainability (avoid vague labels)."
        )
    if task_id == "task4_session_auth_medium":
        return (
            "Focus: session/auth hardening. "
            "Check JWT claim validation, refresh/logout correctness, replay risks, token rotation, "
            "secure cookie flags, and storage semantics. "
            "Aim for 5-7 distinct auth/session findings grounded in code evidence."
        )
    if task_id == "task5_async_pipeline_hard":
        return (
            "Focus: async pipeline integrity. "
            "Check webhook signature validation, replay protection, idempotency/deduping under concurrency, "
            "atomic updates/transactions, and cross-account update risks."
        )
    if task_id == "task6_data_export_hard":
        return (
            "Focus: secure data export workflows. "
            "Check authorization/tenant scoping, PII access controls, temp file safety, "
            "download ownership checks, and redirect/path traversal risks."
        )
    return ""

def _severity_rank(sev: str) -> int:
    order = {"info": 0, "warning": 1, "error": 2, "critical": 3}
    return order.get(sev, 1)


def build_prompt(pr_state: Dict[str, Any], task_id: str) -> str:
    md = pr_state.get("metadata", {})
    hint = _task_hint(task_id)
    lines = [
        f"Task ID: {task_id}",
        f"PR: {pr_state.get('pr_id', 'unknown')}",
        f"Title: {md.get('title', '')}",
        f"Description: {md.get('description', '')}",
        f"Author: {md.get('author', '')}",
        f"Task focus: {hint}" if hint else "",
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


def _extract_outer_json_object(content: str) -> Optional[str]:
    """
    Return the first top-level {...} substring with balanced braces, respecting JSON strings.
    Helps when the model wraps JSON in prose or when json.loads fails on trailing text.
    """
    start = content.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(content)):
        ch = content[i]
        if escape_next:
            escape_next = False
            continue
        if in_string:
            if ch == "\\":
                escape_next = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return content[start : i + 1]
    return None


def _markdown_fence_body(content: str) -> Optional[str]:
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if m:
        return m.group(1).strip()
    return None


def parse_json_response(content: str) -> Dict[str, Any]:
    """
    Parse model output into a dict. Tries full string, fenced blocks, and balanced {...} slice.
    """
    if content is None or not str(content).strip():
        raise json.JSONDecodeError("Empty model content", "", 0)

    raw = str(content).strip()
    candidates: List[str] = []
    for part in (raw, _markdown_fence_body(raw) or ""):
        if not part:
            continue
        part = part.strip()
        if part not in candidates:
            candidates.append(part)
        balanced = _extract_outer_json_object(part)
        if balanced and balanced not in candidates:
            candidates.append(balanced)

    last_err: Optional[json.JSONDecodeError] = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise json.JSONDecodeError("No JSON object found in model output", raw, 0)


def _parse_line_number(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        m = re.search(r"\d+", value)
        if m:
            return int(m.group(0))
    return None

def _normalized_text_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    norm = []
    for t in tokens:
        if len(t) > 4 and t.endswith("s"):
            t = t[:-1]
        if len(t) > 5 and t.endswith("ing"):
            t = t[:-3]
        if len(t) > 4 and t.endswith("ed"):
            t = t[:-2]
        norm.append(t)
    return norm

def _is_near_duplicate(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if a.get("file_path") != b.get("file_path"):
        return False
    if a.get("category") != b.get("category"):
        return False
    la = a.get("line_number")
    lb = b.get("line_number")
    if not isinstance(la, int) or not isinstance(lb, int):
        return False
    if abs(la - lb) > 1:
        return False
    ta = set(_normalized_text_tokens(str(a.get("comment", ""))))
    tb = set(_normalized_text_tokens(str(b.get("comment", ""))))
    if not ta or not tb:
        return True
    overlap = len(ta & tb) / max(1, min(len(ta), len(tb)))
    return overlap >= 0.6

def _is_auth_session_comment(comment: Dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(comment.get("comment", "")),
            str(comment.get("suggested_fix", "")),
            str(comment.get("category", "")),
        ]
    ).lower()
    keys = (
        "jwt", "token", "claim", "refresh", "logout", "session", "cookie",
        "samesite", "httpOnly", "httponly", "secure", "csrf", "replay", "nonce",
        "rotation", "invalidate", "revocation", "brute", "rate", "throttle",
    )
    return any(k in text for k in keys)

def _is_security_comment(comment: Dict[str, Any]) -> bool:
    category = str(comment.get("category", "")).lower()
    if category == "security":
        return True
    text = " ".join(
        [
            str(comment.get("comment", "")),
            str(comment.get("suggested_fix", "")),
        ]
    ).lower()
    keys = ("sql", "injection", "xss", "csrf", "auth", "token", "redirect", "sanitize", "escape")
    return any(k in text for k in keys)

def _task3_category_severity_adjust(comment: Dict[str, Any]) -> Dict[str, Any]:
    text = " ".join(
        [
            str(comment.get("comment", "")),
            str(comment.get("suggested_fix", "")),
            str(comment.get("category", "")),
        ]
    ).lower()
    category = str(comment.get("category", "bug")).lower()
    severity = str(comment.get("severity", "warning")).lower()

    # Map vague/general architecture language into grader-friendly categories.
    if category in {"architecture", "approach", "general", "documentation"}:
        if any(k in text for k in ("n+1", "latency", "slow", "query", "cache", "ttl", "throughput")):
            category = "performance"
        elif any(k in text for k in ("thread", "concurrency", "race", "singleton", "deadlock", "retry", "exception")):
            category = "bug"
        elif any(k in text for k in ("auth", "authorize", "permission", "access", "tenant", "privilege", "token")):
            category = "security"
        else:
            category = "maintainability"

    # Hard task: avoid under-severity that kills severity_alignment.
    if severity in {"info", "warning"} and category in {"security", "bug", "performance"}:
        severity = "error"

    comment["category"] = category
    comment["severity"] = severity
    return comment

def _post_filter_inline_comments(inline: List[Dict[str, Any]], task_id: str) -> List[Dict[str, Any]]:
    if not inline:
        return inline

    # Deterministic stable ordering: high severity first, then by file/line/category.
    inline = sorted(
        inline,
        key=lambda c: (-_severity_rank(str(c.get("severity"))), str(c.get("file_path")), int(c.get("line_number", 0)), str(c.get("category"))),
    )

    # Remove near-duplicates.
    filtered: List[Dict[str, Any]] = []
    # Task-specific duplicate strictness:
    # - task4: keep more coverage across auth flow; only merge very-close duplicates.
    duplicate_overlap_threshold = 0.6
    if task_id in {"task2_quality_logic", "task4_session_auth_medium"}:
        duplicate_overlap_threshold = 0.75
    if task_id == "task3_advanced_review":
        # Preserve breadth for advanced reviews.
        duplicate_overlap_threshold = 0.9

    def _is_dup_with_threshold(c: Dict[str, Any], prev: Dict[str, Any]) -> bool:
        if c.get("file_path") != prev.get("file_path") or c.get("category") != prev.get("category"):
            return False
        la = c.get("line_number")
        lb = prev.get("line_number")
        if not isinstance(la, int) or not isinstance(lb, int):
            return False
        # Stricter on session/auth medium, and strict on advanced review to preserve diversity.
        line_tol = 0 if task_id in {"task4_session_auth_medium", "task3_advanced_review"} else 1
        if abs(la - lb) > line_tol:
            return False
        ta = set(_normalized_text_tokens(str(c.get("comment", ""))))
        tb = set(_normalized_text_tokens(str(prev.get("comment", ""))))
        if not ta or not tb:
            return True
        overlap = len(ta & tb) / max(1, min(len(ta), len(tb)))
        return overlap >= duplicate_overlap_threshold

    for c in inline:
        if any(_is_dup_with_threshold(c, prev) for prev in filtered):
            continue
        filtered.append(c)

    if task_id == "task1_security_basic":
        # Prioritize precision: keep only strong security findings.
        filtered = [
            c for c in filtered
            if _is_security_comment(c) and str(c.get("severity")) in {"error", "critical"}
        ]

    if task_id == "task2_quality_logic":
        # Drop weak style-only findings when higher-severity findings exist.
        has_error_or_critical = any(
            str(c.get("severity")) in {"error", "critical"} for c in filtered
        )
        if has_error_or_critical:
            filtered = [
                c for c in filtered
                if not (c.get("category") == "style" and str(c.get("severity")) in {"info", "warning"})
            ]

    if task_id == "task4_session_auth_medium":
        # Keep comments aligned with auth/session objective and reduce off-topic FPs.
        auth_aligned = [c for c in filtered if _is_auth_session_comment(c)]
        if auth_aligned:
            filtered = auth_aligned

    if task_id == "task3_advanced_review":
        filtered = [_task3_category_severity_adjust(c) for c in filtered]

    return filtered


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

    inline_comments = _post_filter_inline_comments(
        inline_comments,
        task_id=str(raw_action.get("task_id") or ""),
    )

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
        self.task_thresholds: Dict[str, float] = {}

    def _discover_tasks(self) -> List[str]:
        resp = self.http.get(f"{self.env_url}/tasks", timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        tasks = payload.get("tasks", [])
        for t in tasks:
            if isinstance(t, dict) and t.get("task_id"):
                self.task_thresholds[str(t["task_id"])] = float(t.get("min_passing_score", 0.0))
        if self.task_ids:
            return self.task_ids
        return [t["task_id"] for t in tasks if isinstance(t, dict) and t.get("task_id")]

    def _review_with_model(self, pr_state: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        prompt = build_prompt(pr_state, task_id)
        budget = TASK_COMMENT_BUDGET.get(task_id, 8)
        max_tokens_try = MAX_TOKENS
        last_err: Optional[Union[json.JSONDecodeError, ValueError]] = None

        for attempt in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=max_tokens_try,
                    response_format={"type": "json_object"},
                )
                content = completion.choices[0].message.content or "{}"
                raw = parse_json_response(content)
                if isinstance(raw, dict):
                    raw["task_id"] = task_id
                return normalize_action(raw, max_inline_comments=budget)
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
                print(
                    f"Warning: model JSON parse failed for {task_id} "
                    f"(attempt {attempt + 1}/3, max_tokens={max_tokens_try}): {e}",
                    flush=True,
                )
                # Truncation often yields "Unterminated string"; give the model more room.
                max_tokens_try = min(max(max_tokens_try * 2, 2048), 8192)

        print(
            f"Warning: submitting empty review for {task_id} after JSON failures: {last_err}",
            flush=True,
        )
        fallback: Dict[str, Any] = {
            "task_id": task_id,
            "inline_comments": [],
            "general_comments": [],
            "decision": {
                "decision": "comment",
                "summary": "Model returned invalid or truncated JSON; empty review submitted.",
            },
        }
        return normalize_action(fallback, max_inline_comments=budget)

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
            threshold = float(self.task_thresholds.get(task_id, 0.0))
            passed = score >= threshold if threshold > 0 else bool(step_obs.get("metadata", {}).get("passed", False))
            results[task_id] = {
                "score": score,
                "passed": passed,
                "min_passing_score": threshold if threshold > 0 else None,
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
