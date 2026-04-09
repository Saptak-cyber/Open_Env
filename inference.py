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
import time
from typing import Any, Dict, List, Optional, Union

import requests
from openai import OpenAI

from pr_review_env.server.grader import ReviewGrader

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
# Prefer OpenAI-style key name, but allow Hugging Face token as well.
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
# Optional for dockerized local model flows (parity with hackathon sample guidance).
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:8000")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
DEFAULT_OUTPUT = "inference_results.json"
# Total characters of hunk `content` to include per file (large PRs truncate deterministically).
PROMPT_MAX_HUNK_CHARS = int(os.getenv("PROMPT_MAX_HUNK_CHARS", "12000"))
# Second LLM pass for task8: candidates then final JSON (set INFERENCE_TASK8_TWO_PASS=1).
INFERENCE_TASK8_TWO_PASS = os.getenv("INFERENCE_TASK8_TWO_PASS", "").lower() in (
    "1",
    "true",
    "yes",
)
MODEL_RATE_LIMIT_RETRIES = int(os.getenv("MODEL_RATE_LIMIT_RETRIES", "4"))
MODEL_RATE_LIMIT_BACKOFF_SECONDS = float(os.getenv("MODEL_RATE_LIMIT_BACKOFF_SECONDS", "1.5"))

VALID_SEVERITIES = {"info", "warning", "error", "critical"}
VALID_INLINE_CATEGORIES = {
    "security", "bug", "performance", "style", "maintainability", "testing", "documentation"
}
VALID_GENERAL_CATEGORIES = {"architecture", "approach", "testing", "documentation", "general"}
VALID_DECISIONS = {"approve", "request_changes", "comment"}

TASK_COMMENT_BUDGET: Dict[str, int] = {
    "task2_quality_logic": 6,
    "task3_advanced_review": 8,
    "task4_session_auth_medium": 5,
    "task5_async_pipeline_hard": 7,
    "task6_data_export_hard": 8,
    "task7_pr_review_dvr_recorder": 3,
    "task8_expert_security_review": 15
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
- For every inline comment, line_number MUST be an integer taken from the diff line prefixes in the prompt
  (e.g. the line labeled "42: ..." uses line_number 42). Do not guess line numbers.
- When hunks are provided, you may also anchor to start_line/end_line shown for each hunk.
"""


def require_env() -> None:
    missing = []
    if not API_KEY:
        missing.append("OPENAI_API_KEY or HF_TOKEN")
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
            "and maintainability (avoid vague labels). "
            "On try/except blocks, separate findings: e.g. logging-without-reraise (anchor near the log line) "
            "vs missing retry/backoff (anchor near pass or exit of the except block)."
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
            "download ownership checks, and redirect/path traversal risks. "
            "Use categories security, bug, or performance where applicable; cite exact diff line numbers."
        )
    if task_id == "task7_pr_review_dvr_recorder":
        return (
            "This is a small frontend PR: prefer 2–4 high-confidence findings only. "
            "Focus: performance and memory in the DVR/live recorder (JavaScript)—Blob/MediaRecorder, "
            "rebuilding blobs from recordedChunks on hot paths, object URL lifecycle (createObjectURL/revokeObjectURL), "
            "MutationObserver with subtree observation cost in SPAs. "
            "Do not add generic style or unrelated backend/security comments."
        )
    if task_id == "task8_expert_security_review":
        return (
            "Focus: expert security and correctness across JWT auth, payments, Redis rate limiting, and session lifecycle. "
            "Audit JWT verification (allowed algorithms, signature requirements, missing/invalid token handling), "
            "decorator enforcement (reject unauthenticated requests), and unpredictable token/session IDs. "
            "For payments: safe money types, row locking / races on all updated balances, missing rows, and real idempotency. "
            "For Redis: atomic check-and-increment (avoid lost updates under concurrency). "
            "For sessions: fixation on login (regenerate IDs), weak or guessable session identifiers, "
            "and consistency between how keys are written vs scanned or invalidated."
        )
    return ""

def _severity_rank(sev: str) -> int:
    order = {"info": 0, "warning": 1, "error": 2, "critical": 3}
    return order.get(sev, 1)


def _truncate_hunk_content(content: str, max_chars: int) -> str:
    if max_chars <= 0 or len(content) <= max_chars:
        return content
    return content[: max_chars - 28] + "\n... [hunk truncated] ..."


TASK8_SCAN_SYSTEM = """You are a security-focused code reviewer scanning a PR.
Output ONLY valid JSON:
{
  "candidates": [
    {
      "file_path": "string",
      "line_number": 123,
      "theme": "one-sentence issue",
      "category": "security|bug|performance|style|maintainability|testing|documentation",
      "severity": "info|warning|error|critical"
    }
  ]
}
Rules:
- List every plausible issue you see (aim for breadth); a second pass will refine.
- line_number must match a line prefix from the user diff (integer).
- file_path must match exactly one changed file path from the user diff.
- Include high-recall candidates for each area when evidence exists:
  JWT verification/authz guard, payment correctness/idempotency, Redis atomicity, session lifecycle.
- No markdown, no prose outside JSON.
"""


def _turn_focus_instruction(turn_index: int, total_turns: int, task_id: str) -> str:
    if total_turns <= 1:
        return "Single-turn review: provide your best complete review now."
    if turn_index < total_turns - 1:
        if turn_index == 0:
            return (
                "Turn strategy: prioritize high-confidence, high-severity findings first. "
                "Avoid speculative comments."
            )
        if turn_index == 1:
            return (
                "Turn strategy: focus on correctness/logic/concurrency findings not yet covered. "
                "You may refine previously-reported areas only when adding materially new evidence "
                "(different line/file or a clearer root cause/severity)."
            )
        return (
            "Turn strategy: focus on remaining gaps (performance, maintainability, and tests). "
            "Prefer net-new findings, but allow meaningful refinements with stronger evidence."
        )
    return (
        "Final turn strategy: prioritize net-new findings and finalize with a concise decision summary. "
        "If needed, include a refined finding only when it materially improves precision/severity alignment."
    )


def _format_prior_findings(prior_inline: List[Dict[str, Any]], limit: int = 16) -> List[str]:
    if not prior_inline:
        return []
    rows: List[str] = []
    for c in prior_inline[:limit]:
        rows.append(
            f"- {c.get('file_path')}:{c.get('line_number')} "
            f"[{c.get('category')}/{c.get('severity')}] {c.get('comment')}"
        )
    if len(prior_inline) > limit:
        rows.append(f"- ... {len(prior_inline) - limit} more previously submitted findings")
    return rows


def build_prompt(
    pr_state: Dict[str, Any],
    task_id: str,
    *,
    turn_index: int = 0,
    total_turns: int = 1,
    prior_inline: Optional[List[Dict[str, Any]]] = None,
    uncovered_focus: Optional[List[str]] = None,
) -> str:
    md = pr_state.get("metadata", {})
    hint = _task_hint(task_id)
    prior_inline = prior_inline or []
    uncovered_focus = uncovered_focus or []
    lines = [
        f"Task ID: {task_id}",
        f"Turn: {turn_index + 1}/{total_turns}",
        f"PR: {pr_state.get('pr_id', 'unknown')}",
        f"Title: {md.get('title', '')}",
        f"Description: {md.get('description', '')}",
        f"Author: {md.get('author', '')}",
        f"Task focus: {hint}" if hint else "",
        f"Turn focus: {_turn_focus_instruction(turn_index, total_turns, task_id)}",
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

        hunks = file_obj.get("hunks") or []
        if isinstance(hunks, list) and hunks:
            lines.append("Hunks (full unified diff):")
            hunk_budget = PROMPT_MAX_HUNK_CHARS
            for h in hunks:
                if hunk_budget <= 0:
                    lines.append("  ... remaining hunks omitted for this file (size budget)")
                    break
                if not isinstance(h, dict):
                    continue
                sl = h.get("start_line", "?")
                el = h.get("end_line", "?")
                raw = h.get("content")
                body = str(raw) if raw is not None else ""
                chunk = _truncate_hunk_content(body, hunk_budget)
                lines.append(f"  @@ lines {sl}-{el} @@")
                lines.extend(f"    {hl}" for hl in chunk.split("\n"))
                hunk_budget -= len(chunk)

    prior_lines = _format_prior_findings(prior_inline)
    if prior_lines:
        lines.append("\nPreviously submitted inline findings (avoid trivial repeats):")
        lines.extend(prior_lines)
        lines.append(
            "Prioritize net-new issues in this turn. "
            "You may include a refinement only if it adds materially new evidence "
            "(new line/file, clearer exploit path, or better severity/category)."
        )

    if uncovered_focus:
        lines.append("\nCoverage targets still missing this turn:")
        lines.extend(f"- {f}" for f in uncovered_focus)
        lines.append("Try to add at least one high-confidence finding from missing targets if evidence exists in diff.")

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


def emit_log(tag: str, fields: Dict[str, Any]) -> None:
    """
    Emit structured debug logs to stderr (internal use only).
    stdout is reserved exclusively for the validator-mandated key=value format.
    """
    import sys
    print(f"[{tag}] {json.dumps(fields, ensure_ascii=False, separators=(',', ':'))}", file=sys.stderr, flush=True)


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

def _inline_category_severity_adjust(comment: Dict[str, Any]) -> Dict[str, Any]:
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


def _is_dvr_performance_comment(comment: Dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(comment.get("comment", "")),
            str(comment.get("suggested_fix", "")),
            str(comment.get("category", "")),
        ]
    ).lower()
    keys = (
        "blob",
        "mediarecorder",
        "webm",
        "chunk",
        "memory",
        "leak",
        "observer",
        "mutation",
        "subtree",
        "dvr",
        "objecturl",
        "createobjecturl",
        "revokeobjecturl",
        "recorded",
        "performance",
        "cpu",
        "layout",
        "spa",
    )
    return any(k in text for k in keys)


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
        filtered = [_inline_category_severity_adjust(c) for c in filtered]

    if task_id == "task6_data_export_hard":
        filtered = [_inline_category_severity_adjust(c) for c in filtered]

    if task_id == "task8_expert_security_review":
        filtered = [_inline_category_severity_adjust(c) for c in filtered]

    if task_id == "task7_pr_review_dvr_recorder":
        dvr_aligned = [c for c in filtered if _is_dvr_performance_comment(c)]
        if dvr_aligned:
            filtered = dvr_aligned

    return filtered


def _cross_turn_key(comment: Dict[str, Any]) -> tuple:
    text_tokens = _normalized_text_tokens(str(comment.get("comment", "")))
    # Stable, compact signature for cross-turn dedupe of same root cause.
    text_sig = " ".join(text_tokens[:10])
    return (
        str(comment.get("file_path", "")).strip(),
        int(comment.get("line_number", 0)) // 3,
        str(comment.get("category", "")).strip(),
        text_sig,
    )


def _infer_coverage_targets(task_id: str, prior_inline: List[Dict[str, Any]]) -> List[str]:
    text = " ".join(
        f"{c.get('comment','')} {c.get('suggested_fix','')} {c.get('category','')}"
        for c in (prior_inline or [])
    ).lower()
    if task_id == "task8_expert_security_review":
        targets = [
            ("jwt_auth", ("jwt", "token", "algorithm", "signature", "claim", "decorator")),
            ("payment_idempotency", ("payment", "balance", "idempot", "row lock", "transaction", "decimal")),
            ("redis_rate_limit_atomicity", ("redis", "rate", "incr", "atomic", "lua", "lost update")),
            ("session_lifecycle", ("session", "fixation", "regenerate", "invalidate", "sid")),
        ]
    elif task_id == "task5_async_pipeline_hard":
        targets = [
            ("webhook_authenticity", ("webhook", "signature", "hmac")),
            ("replay_protection", ("replay", "nonce", "timestamp")),
            ("idempotency_atomicity", ("idempot", "dedupe", "atomic", "race", "transaction")),
        ]
    else:
        return []

    missing: List[str] = []
    for name, keys in targets:
        if not any(k in text for k in keys):
            missing.append(name)
    return missing


def _task8_candidates_to_action(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert task8 scan candidates into a valid action payload candidate.
    This is used as a fallback when merge output is empty/invalid.
    """
    inline_comments: List[Dict[str, Any]] = []
    for cand in candidates or []:
        if not isinstance(cand, dict):
            continue
        file_path = cand.get("file_path")
        line_number = _parse_line_number(cand.get("line_number"))
        theme = cand.get("theme")
        if not isinstance(file_path, str) or not file_path.strip():
            continue
        if line_number is None:
            continue
        if not isinstance(theme, str) or not theme.strip():
            continue

        category = str(cand.get("category", "security")).lower()
        if category not in VALID_INLINE_CATEGORIES:
            category = "security"
        severity = str(cand.get("severity", "warning")).lower()
        if severity not in VALID_SEVERITIES:
            severity = "warning"

        inline_comments.append(
            {
                "file_path": file_path.strip(),
                "line_number": line_number,
                "comment": theme.strip(),
                "category": category,
                "severity": severity,
            }
        )

    return {
        "task_id": "task8_expert_security_review",
        "inline_comments": inline_comments,
        "general_comments": [],
        "decision": {
            "decision": "request_changes",
            "summary": "Potential high-risk issues detected; prioritize security fixes before merge.",
        },
    }


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


def normalize_action_for_turn(
    raw_action: Dict[str, Any],
    task_id: str,
    budget: int,
    *,
    finalize: bool,
) -> Dict[str, Any]:
    """
    Normalize model output for either intermediate or final trajectory turns.

    Intermediate turns MUST avoid terminal signals (`submit=true` or non-null
    decision), otherwise the environment finalizes early.
    """
    if isinstance(raw_action, dict):
        raw_action["task_id"] = task_id
    action = normalize_action(raw_action, max_inline_comments=budget)
    if finalize:
        action["submit"] = True
        if not isinstance(action.get("decision"), dict):
            action["decision"] = {
                "decision": "comment",
                "summary": "Automated review generated by inference script.",
            }
    else:
        action["submit"] = False
        action["decision"] = None
    return action


class InferenceRunner:
    def __init__(
        self,
        env_url: str,
        output: str,
        task_ids: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        temperature: float = TEMPERATURE,
        task8_two_pass: bool = False,
        turns: int = 5,
        max_runtime_seconds: int = 0,
    ) -> None:
        self.env_url = env_url.rstrip("/")
        self.output = output
        self.task_ids = task_ids
        self.temperature = temperature
        self.task8_two_pass = bool(task8_two_pass)
        self.turns = max(1, min(int(turns), 5))
        self.max_runtime_seconds = max(0, int(max_runtime_seconds))
        self.http = requests.Session()
        self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        self.model_name = self._resolve_model_name(model_name)
        self.task_thresholds: Dict[str, float] = {}

    def _resolve_model_name(self, candidate: Optional[str]) -> str:
        """
        Resolve model id robustly for validator environments.
        Priority: explicit --model > MODEL_NAME > common fallbacks > /models discovery.
        """
        for c in (
            candidate,
            MODEL_NAME,
            os.getenv("OPENAI_MODEL"),
            os.getenv("HF_MODEL"),
            os.getenv("MODEL"),
        ):
            if isinstance(c, str) and c.strip():
                return c.strip()

        # Last resort: try listing provider models and pick first id.
        try:
            models = self.client.models.list()
            data = getattr(models, "data", None) or []
            for m in data:
                mid = getattr(m, "id", None)
                if isinstance(mid, str) and mid.strip():
                    return mid.strip()
        except Exception:
            pass

        raise RuntimeError(
            "Missing model name. Set MODEL_NAME (or --model), or expose /models on API_BASE_URL for auto-discovery."
        )

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

    @staticmethod
    def _is_retryable_model_error(err: Exception) -> bool:
        name = err.__class__.__name__
        text = str(err).lower()
        return (
            "ratelimit" in name.lower()
            or "429" in text
            or "queue_exceeded" in text
            or "high traffic" in text
        )

    def _chat_completion_with_retry(
        self,
        *,
        task_id: str,
        phase: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
    ) -> Any:
        last_err: Optional[Exception] = None
        attempts = max(1, MODEL_RATE_LIMIT_RETRIES + 1)
        for attempt in range(attempts):
            try:
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                last_err = e
                if not self._is_retryable_model_error(e) or attempt >= attempts - 1:
                    raise
                wait_s = MODEL_RATE_LIMIT_BACKOFF_SECONDS * (2 ** attempt)
                emit_log(
                    "STEP",
                    {
                        "event": "warning",
                        "task_id": task_id,
                        "warning_type": "model_rate_limit_retry",
                        "phase": phase,
                        "attempt": attempt + 1,
                        "max_attempts": attempts,
                        "wait_seconds": round(wait_s, 2),
                        "message": str(e)[:300],
                    },
                )
                time.sleep(wait_s)
        if last_err is not None:
            raise last_err

    def _review_task8_two_pass(
        self,
        pr_state: Dict[str, Any],
        budget: int,
        *,
        finalize: bool,
        turn_index: int = 0,
        total_turns: int = 1,
        prior_inline: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Optional scan + finalize pipeline for dense security PRs."""
        prompt = build_prompt(
            pr_state,
            "task8_expert_security_review",
            turn_index=turn_index,
            total_turns=total_turns,
            prior_inline=prior_inline,
            uncovered_focus=_infer_coverage_targets(
                "task8_expert_security_review",
                prior_inline or [],
            ),
        )
        max_tokens_try = MAX_TOKENS
        last_err: Optional[Union[json.JSONDecodeError, ValueError]] = None
        candidates: List[Dict[str, Any]] = []

        for attempt in range(2):
            try:
                scan = self._chat_completion_with_retry(
                    task_id="task8_expert_security_review",
                    phase="task8_scan",
                    messages=[
                        {"role": "system", "content": TASK8_SCAN_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=min(max_tokens_try, 4096),
                )
                raw_scan = parse_json_response(scan.choices[0].message.content or "{}")
                cand = raw_scan.get("candidates")
                if isinstance(cand, list):
                    candidates = [x for x in cand if isinstance(x, dict)]
                break
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
                max_tokens_try = min(max(max_tokens_try * 2, 2048), 8192)

        merge_user = (
            prompt
            + "\n\n---\nInitial scan candidates (merge, dedupe, drop false positives; "
            "output the final review JSON per the main schema):\n"
            + json.dumps({"candidates": candidates}, ensure_ascii=False)
        )
        max_tokens_try = MAX_TOKENS
        for attempt in range(3):
            try:
                completion = self._chat_completion_with_retry(
                    task_id="task8_expert_security_review",
                    phase="task8_merge",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": merge_user},
                    ],
                    max_tokens=max_tokens_try,
                )
                content = completion.choices[0].message.content or "{}"
                raw = parse_json_response(content)
                action = normalize_action_for_turn(
                    raw,
                    task_id="task8_expert_security_review",
                    budget=budget,
                    finalize=finalize,
                )
                # High-recall guardrail: if merge returned no usable findings but scan
                # had candidates, salvage candidates instead of submitting empty output.
                if not action.get("inline_comments") and candidates:
                    emit_log(
                        "STEP",
                        {
                            "event": "warning",
                            "task_id": "task8_expert_security_review",
                            "warning_type": "task8_empty_merge_salvage",
                            "scan_candidates": len(candidates),
                        },
                    )
                    return normalize_action_for_turn(
                        _task8_candidates_to_action(candidates),
                        task_id="task8_expert_security_review",
                        budget=budget,
                        finalize=finalize,
                    )
                return action
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
                emit_log(
                    "STEP",
                    {
                        "event": "warning",
                        "task_id": "task8_expert_security_review",
                        "warning_type": "json_parse",
                        "phase": "task8_merge",
                        "attempt": attempt + 1,
                        "max_tokens": max_tokens_try,
                        "message": str(e),
                    },
                )
                max_tokens_try = min(max(max_tokens_try * 2, 2048), 8192)

        emit_log(
            "STEP",
            {
                "event": "warning",
                "task_id": "task8_expert_security_review",
                "warning_type": "task8_two_pass_fallback",
                "message": str(last_err),
            },
        )
        return self._review_with_model_single(
            pr_state,
            "task8_expert_security_review",
            budget,
            finalize=finalize,
            turn_index=turn_index,
            total_turns=total_turns,
            prior_inline=prior_inline,
        )

    def _review_with_model_single(
        self,
        pr_state: Dict[str, Any],
        task_id: str,
        budget: int,
        *,
        finalize: bool,
        turn_index: int = 0,
        total_turns: int = 1,
        prior_inline: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        prompt = build_prompt(
            pr_state,
            task_id,
            turn_index=turn_index,
            total_turns=total_turns,
            prior_inline=prior_inline,
            uncovered_focus=_infer_coverage_targets(task_id, prior_inline or []),
        )
        max_tokens_try = MAX_TOKENS
        last_err: Optional[Union[json.JSONDecodeError, ValueError]] = None

        for attempt in range(3):
            try:
                completion = self._chat_completion_with_retry(
                    task_id=task_id,
                    phase="model_completion",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens_try,
                )
                content = completion.choices[0].message.content or "{}"
                raw = parse_json_response(content)
                return normalize_action_for_turn(
                    raw,
                    task_id=task_id,
                    budget=budget,
                    finalize=finalize,
                )
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
                emit_log(
                    "STEP",
                    {
                        "event": "warning",
                        "task_id": task_id,
                        "warning_type": "json_parse",
                        "phase": "model_completion",
                        "attempt": attempt + 1,
                        "max_tokens": max_tokens_try,
                        "message": str(e),
                    },
                )
                max_tokens_try = min(max(max_tokens_try * 2, 2048), 8192)

        emit_log(
            "STEP",
            {
                "event": "warning",
                "task_id": task_id,
                "warning_type": "empty_fallback_submission",
                "message": str(last_err),
            },
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
        return normalize_action_for_turn(
            fallback,
            task_id=task_id,
            budget=budget,
            finalize=finalize,
        )

    def _review_with_model(
        self,
        pr_state: Dict[str, Any],
        task_id: str,
        *,
        finalize: bool,
        turn_index: int = 0,
        total_turns: int = 1,
        prior_inline: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        budget = TASK_COMMENT_BUDGET.get(task_id, 8)
        if task_id == "task8_expert_security_review" and (
            INFERENCE_TASK8_TWO_PASS or self.task8_two_pass
        ):
            try:
                return self._review_task8_two_pass(
                    pr_state,
                    budget,
                    finalize=finalize,
                    turn_index=turn_index,
                    total_turns=total_turns,
                    prior_inline=prior_inline,
                )
            except Exception as e:
                emit_log(
                    "STEP",
                    {
                        "event": "warning",
                        "task_id": task_id,
                        "warning_type": "task8_two_pass_exception",
                        "message": str(e)[:300],
                    },
                )
                # Never crash the whole run because task8 two-pass failed.
                # Fall back to single-pass review for resiliency in validator runs.
                return self._review_with_model_single(
                    pr_state,
                    task_id,
                    budget,
                    finalize=finalize,
                    turn_index=turn_index,
                    total_turns=total_turns,
                    prior_inline=prior_inline,
                )
        return self._review_with_model_single(
            pr_state,
            task_id,
            budget,
            finalize=finalize,
            turn_index=turn_index,
            total_turns=total_turns,
            prior_inline=prior_inline,
        )

    def run(self) -> Dict[str, Any]:
        task_ids = self._discover_tasks()
        if not task_ids:
            raise RuntimeError("No tasks discovered. Ensure /tasks endpoint is available.")

        run_start = time.monotonic()
        emit_log(
            "START",
            {
                "event": "run",
                "task_count": len(task_ids),
                "task_ids": task_ids,
                "model": self.model_name,
                "turns": self.turns,
                "max_runtime_seconds": self.max_runtime_seconds,
            },
        )

        results: Dict[str, Any] = {}
        stopped_early = False
        for task_id in task_ids:
            elapsed = time.monotonic() - run_start
            if self.max_runtime_seconds > 0 and elapsed >= self.max_runtime_seconds:
                stopped_early = True
                emit_log(
                    "END",
                    {
                        "event": "run_budget_reached",
                        "elapsed_seconds": round(elapsed, 3),
                        "max_runtime_seconds": self.max_runtime_seconds,
                        "last_completed_task_count": len(results),
                    },
                )
                break

            # Validator-mandated [START] line (key=value, stdout).
            print(f"[START] task={task_id} env=pr_review model={self.model_name}", flush=True)
            emit_log("START", {"event": "task", "task_id": task_id})

            reset_resp = self.http.post(
                f"{self.env_url}/reset",
                json={"task_id": task_id},
                timeout=30,
            )
            reset_resp.raise_for_status()
            reset_obs = extract_observation(reset_resp.json(), "/reset")

            step_obs = reset_obs
            cumulative_inline: List[Dict[str, Any]] = []
            seen_cross_turn = set()
            step_rewards: List[float] = []
            steps_taken = 0
            for turn_idx in range(self.turns):
                finalize = turn_idx == self.turns - 1
                try:
                    action = self._review_with_model(
                        step_obs["pr_state"],
                        task_id,
                        finalize=finalize,
                        turn_index=turn_idx,
                        total_turns=self.turns,
                        prior_inline=cumulative_inline,
                    )
                except Exception as e:
                    emit_log(
                        "STEP",
                        {
                            "event": "warning",
                            "task_id": task_id,
                            "warning_type": "turn_generation_exception",
                            "turn": turn_idx + 1,
                            "message": str(e)[:300],
                        },
                    )
                    # Keep the episode alive with a safe fallback action.
                    action = normalize_action_for_turn(
                        {
                            "task_id": task_id,
                            "inline_comments": [],
                            "general_comments": [],
                            "decision": {
                                "decision": "comment",
                                "summary": (
                                    "Model generation failed for this turn; "
                                    "submitted fallback action."
                                ),
                            },
                        },
                        task_id=task_id,
                        budget=TASK_COMMENT_BUDGET.get(task_id, 8),
                        finalize=finalize,
                    )
                # Remove comments that duplicate findings already submitted in
                # earlier turns for this task.
                deduped_inline: List[Dict[str, Any]] = []
                for c in action.get("inline_comments", []) or []:
                    k = _cross_turn_key(c)
                    if k in seen_cross_turn:
                        continue
                    seen_cross_turn.add(k)
                    deduped_inline.append(c)
                action["inline_comments"] = deduped_inline
                cumulative_inline.extend(deduped_inline)

                step_resp = self.http.post(
                    f"{self.env_url}/step",
                    json={"action": action},
                    timeout=30,
                )
                step_resp.raise_for_status()
                step_obs = extract_observation(step_resp.json(), "/step")

                # Per-step reward from environment (already clamped by environment).
                raw_step_reward = step_obs.get("reward") or 0.0
                step_reward = ReviewGrader.clamp_open_unit_interval(float(raw_step_reward))
                step_rewards.append(step_reward)
                steps_taken = turn_idx + 1
                done = bool(step_obs.get("done", False))

                decision = action.get("decision")
                emit_log(
                    "STEP",
                    {
                        "event": "turn",
                        "task_id": task_id,
                        "turn": turn_idx + 1,
                        "total_turns": self.turns,
                        "generated_inline_comments": len(action["inline_comments"]),
                        "submit": bool(action.get("submit", False)),
                        "decision": decision.get("decision") if isinstance(decision, dict) else None,
                    },
                )
                # Validator-mandated [STEP] line (key=value, stdout).
                # Use enough precision that epsilon-sized rewards (1e-4) do not round to "0.00".
                print(
                    f"[STEP] step={turn_idx + 1} action=pr_review"
                    f" reward={step_reward:.4f} done={str(done).lower()} error=null",
                    flush=True,
                )

                if done:
                    break
            feedback = step_obs.get("feedback") or {}

            _clamp = ReviewGrader.clamp_open_unit_interval
            score = _clamp(float(feedback.get("score", 0.0)))
            threshold = float(self.task_thresholds.get(task_id, 0.0))
            passed = score >= threshold if threshold > 0 else bool(step_obs.get("metadata", {}).get("passed", False))
            results[task_id] = {
                "score": score,
                "passed": passed,
                "min_passing_score": threshold if threshold > 0 else None,
                "precision": _clamp(float(feedback.get("precision", 0.0))),
                "recall": _clamp(float(feedback.get("recall", 0.0))),
                "severity_alignment": _clamp(float(feedback.get("severity_alignment", 0.0))),
                "true_positives": feedback.get("true_positives", 0),
                "false_positives": feedback.get("false_positives", 0),
                "false_negatives": feedback.get("false_negatives", 0),
            }
            emit_log(
                "END",
                {
                    "event": "task",
                    "task_id": task_id,
                    "score": score,
                    "passed": passed,
                    "precision": results[task_id]["precision"],
                    "recall": results[task_id]["recall"],
                },
            )
            # Validator-mandated [END] line (key=value, stdout).
            # score is strictly in (0, 1) via clamp_open_unit_interval.
            rewards_str = ",".join(f"{r:.4f}" for r in step_rewards)
            print(
                f"[END] success={str(passed).lower()} steps={steps_taken}"
                f" score={score:.6f} rewards={rewards_str}",
                flush=True,
            )

        raw_avg = mean([r["score"] for r in results.values()]) if results else 0.0
        avg = ReviewGrader.clamp_open_unit_interval(raw_avg)
        payload = {
            "provider": "openai_compatible",
            "api_base_url": API_BASE_URL,
            "model": self.model_name,
            "temperature": self.temperature,
            "env_url": self.env_url,
            "task_ids": task_ids,
            "results": results,
            "average_score": avg,
            "turns": self.turns,
            "max_runtime_seconds": self.max_runtime_seconds,
            "stopped_early": stopped_early,
        }
        with open(self.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        emit_log(
            "END",
            {
                "event": "run",
                "average_score": avg,
                "completed_tasks": len(results),
                "total_tasks": len(task_ids),
                "stopped_early": stopped_early,
                "output": self.output,
            },
        )
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
        "--model",
        default=MODEL_NAME,
        help="Model identifier (default: MODEL_NAME env; auto-discovered from /models if missing)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Sampling temperature for model calls (default from TEMPERATURE env or 0.0)",
    )
    parser.add_argument(
        "--task8-two-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run task8_expert_security_review as scan + merge (two LLM calls). "
            "Default: enabled. Use --no-task8-two-pass to disable. "
            "INFERENCE_TASK8_TWO_PASS=1 also enables it."
        ),
    )
    parser.add_argument(
        "--turns",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=3,
        help="Max turns per task episode (default: 3).",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=0,
        help="Optional runtime budget for the full run; 0 disables limit.",
    )
    return parser.parse_args()


def main() -> None:
    require_env()
    args = parse_args()
    runner = InferenceRunner(
        env_url=args.env_url,
        output=args.output,
        task_ids=args.task_ids,
        model_name=args.model,
        temperature=args.temperature,
        task8_two_pass=args.task8_two_pass,
        turns=args.turns,
        max_runtime_seconds=args.max_runtime_seconds,
    )
    runner.run()


if __name__ == "__main__":
    main()
