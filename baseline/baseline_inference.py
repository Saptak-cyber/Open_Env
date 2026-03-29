"""
Baseline inference for PR Code Review Assistant.

This script runs baseline evaluation using various LLM providers.
Supports: OpenAI GPT-4, Groq (GPT OSS 120B), and more.
"""

import os
import json
import re
import requests
import argparse
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Any, Optional
from openai import OpenAI

# System prompt for code review
SYSTEM_PROMPT = """You are an expert code reviewer specializing in security, code quality, and best practices.

Your task is to review pull request changes and identify:
1. Security vulnerabilities (SQL injection, XSS, authentication issues, etc.)
2. Logic bugs and edge cases
3. Code quality issues (style, maintainability, performance)

For each issue you find, provide:
- Exact file path and line number
- Clear description of the problem
- Severity level (info, warning, error, or critical)
- Category (security, bug, performance, style, maintainability, testing, or documentation)
- Suggested fix if possible

Your response must be valid JSON matching this schema:
{
  "inline_comments": [
    {
      "file_path": "string",
      "line_number": integer,
      "comment": "string",
      "severity": "info|warning|error|critical",
      "category": "security|bug|performance|style|maintainability|testing|documentation",
      "suggested_fix": "string (optional)"
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

Be thorough but precise. Focus on real issues, not nitpicks.
Avoid duplicate comments about the same root cause.

IMPORTANT OUTPUT LIMITS:
- Return at most 8 inline comments total (highest impact only).
- Keep each comment under 220 characters.
- Return strictly valid JSON with double quotes and no trailing text."""


class BaselineAgent:
    """Baseline agent supporting multiple LLM providers."""

    # Provider configurations
    PROVIDERS = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "env_key": "OPENAI_API_KEY",
            "default_model": "gpt-4-turbo-preview"
        },
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "env_key": "GROQ_API_KEY",
            "default_model": "openai/gpt-oss-120b"
        }
    }
    VALID_SEVERITIES = {"info", "warning", "error", "critical"}
    VALID_INLINE_CATEGORIES = {"security", "bug", "performance", "style", "maintainability", "testing", "documentation"}
    VALID_GENERAL_CATEGORIES = {"architecture", "approach", "testing", "documentation", "general"}
    VALID_DECISIONS = {"approve", "request_changes", "comment"}
    TASK_COMMENT_BUDGET = {
        "task1_security_basic": 5,
        "task2_quality_logic": 7,
        "task3_advanced_review": 8,
    }

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        env_base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        seed: Optional[int] = 42,
    ):
        """
        Initialize baseline agent.

        Args:
            provider: LLM provider ("openai" or "groq")
            model: Model name (uses provider default if None)
            env_base_url: Base URL for the environment server
            api_key: API key (uses env variable if None)
        """
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Choose from: {list(self.PROVIDERS.keys())}")

        provider_config = self.PROVIDERS[provider]

        # Get API key
        self.api_key = api_key or os.getenv(provider_config["env_key"])
        if not self.api_key:
            raise ValueError(
                f"{provider_config['env_key']} environment variable not set. "
                f"Please set it with: export {provider_config['env_key']}='your-key-here'"
            )

        # Initialize OpenAI client with provider's base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=provider_config["base_url"]
        )

        self.provider = provider
        self.model = model or provider_config["default_model"]
        self.env_base_url = env_base_url
        self.temperature = temperature
        self.seed = seed
        self.http = requests.Session()
        self.project_root = Path(__file__).resolve().parent.parent

        print(f"✓ Initialized {provider.upper()} baseline")
        print(f"  Model: {self.model}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Environment: {env_base_url}")
        print()

    def _build_prompt(self, pr_state: Dict[str, Any], task_id: Optional[str] = None) -> str:
        """Build prompt from PR state."""
        comment_budget = self.TASK_COMMENT_BUDGET.get(task_id or "", 8)
        task_hint = ""
        if task_id == "task1_security_basic":
            task_hint = (
                "Task focus: obvious security vulnerabilities only. "
                "Prioritize precision over recall; skip uncertain/non-security findings."
            )
        elif task_id == "task2_quality_logic":
            task_hint = (
                "Task focus: security + logic + quality issues. "
                "Balance precision and recall; include concrete findings with direct code evidence."
            )
        elif task_id == "task3_advanced_review":
            task_hint = (
                "Task focus: architectural/subtle issues. "
                "Prioritize broad but relevant coverage of high-impact architecture/performance/reliability issues."
            )

        prompt = f"""Review this pull request:

PR #{pr_state['pr_id']}: {pr_state['metadata']['title']}
Description: {pr_state['metadata']['description']}
Author: {pr_state['metadata']['author']}
{f"Task ID: {task_id}" if task_id else ""}
{task_hint}

Output constraints:
- Return at most {comment_budget} inline comments.
- Prefer one comment per root cause; do not duplicate.

Changes:
"""

        for file in pr_state['files']:
            prompt += f"\n\n## File: {file['path']} ({file['language']})\n"
            prompt += f"Additions: {len(file['additions'])}, Deletions: {len(file['deletions'])}\n\n"

            if file['context']:
                prompt += "Context:\n"
                for line in file['context']:
                    prompt += f"  {line}\n"

            if file['additions']:
                prompt += "\nAdded lines:\n"
                for line in file['additions']:
                    prompt += f"+ {line}\n"

            if file['deletions']:
                prompt += "\nDeleted lines:\n"
                for line in file['deletions']:
                    prompt += f"- {line}\n"

        return prompt

    def review_pr(self, pr_state: Dict[str, Any], task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate review using LLM.

        Args:
            pr_state: PR state from environment

        Returns:
            Action dictionary
        """
        prompt = self._build_prompt(pr_state, task_id=task_id)

        try:
            # Build API call parameters
            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": 4096
            }
            if self.seed is not None:
                api_params["seed"] = self.seed

            # OpenAI-compatible providers support json_object mode.
            if self.provider in {"openai", "groq"}:
                api_params["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**api_params)

            # Parse response
            content = response.choices[0].message.content

            # Try to extract JSON from response
            action = self._parse_json_response(content)
            return action

        except Exception as e:
            print(f"Error calling {self.provider.upper()}: {e}")
            import traceback
            traceback.print_exc()
            # Return minimal valid action
            return {
                "inline_comments": [],
                "general_comments": [],
                "decision": {
                    "decision": "comment",
                    "summary": f"Error during review: {e}"
                }
            }

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        try:
            # Try direct parsing first
            return json.loads(content)
        except json.JSONDecodeError:
            # Try extracting from markdown code block
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            else:
                raise ValueError(f"Could not parse JSON from response: {content[:200]}...")

    def _extract_observation(self, payload: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        """
        Extract observation object from API payload.

        OpenEnv servers may return either:
        1) raw observation: {"pr_state": ..., "feedback": ..., "metadata": ...}
        2) wrapped response: {"observation": {"pr_state": ...}, ...}
        """
        if isinstance(payload, dict):
            if "pr_state" in payload:
                return payload
            if isinstance(payload.get("observation"), dict):
                return payload["observation"]

        raise ValueError(
            f"Unexpected {endpoint} response format. "
            f"Top-level keys: {list(payload.keys()) if isinstance(payload, dict) else type(payload)}"
        )

    def _parse_line_number(self, value: Any) -> Optional[int]:
        """Convert LLM-provided line number to int if possible."""
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                return int(match.group(0))
        return None

    def _normalize_action(self, action: Dict[str, Any], max_inline_comments: int = 8) -> Dict[str, Any]:
        """Normalize LLM output into schema-valid Action payload."""
        if not isinstance(action, dict):
            action = {}

        normalized_inline = []
        for comment in action.get("inline_comments", []) or []:
            if not isinstance(comment, dict):
                continue

            file_path = comment.get("file_path")
            text = comment.get("comment")
            line_number = self._parse_line_number(comment.get("line_number"))
            severity = str(comment.get("severity", "warning")).lower()
            category = str(comment.get("category", "bug")).lower()

            if not isinstance(file_path, str) or not file_path.strip():
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            if line_number is None:
                continue
            if severity not in self.VALID_SEVERITIES:
                severity = "warning"
            if category not in self.VALID_INLINE_CATEGORIES:
                category = "bug"

            item = {
                "file_path": file_path.strip(),
                "line_number": line_number,
                "comment": text.strip(),
                "severity": severity,
                "category": category,
            }
            suggested_fix = comment.get("suggested_fix")
            if isinstance(suggested_fix, str) and suggested_fix.strip():
                item["suggested_fix"] = suggested_fix.strip()
            normalized_inline.append(item)

        normalized_general = []
        for comment in action.get("general_comments", []) or []:
            if not isinstance(comment, dict):
                continue
            text = comment.get("comment")
            category = str(comment.get("category", "general")).lower()
            if not isinstance(text, str) or not text.strip():
                continue
            if category not in self.VALID_GENERAL_CATEGORIES:
                category = "general"
            normalized_general.append({
                "comment": text.strip(),
                "category": category,
            })

        decision_obj = action.get("decision", {})
        if not isinstance(decision_obj, dict):
            decision_obj = {}
        decision = str(decision_obj.get("decision", "comment")).lower()
        if decision not in self.VALID_DECISIONS:
            decision = "comment"
        summary = decision_obj.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = "Automated review generated by baseline agent."

        return {
            "inline_comments": normalized_inline[:max_inline_comments],
            "general_comments": normalized_general,
            "decision": {
                "decision": decision,
                "summary": summary.strip(),
            },
            "submit": True,
        }

    def _load_task_threshold(self, task_id: str) -> float:
        """Load task min passing score from task config."""
        task_file = self.project_root / "tasks" / f"{task_id}.json"
        if not task_file.exists():
            return 0.7
        with open(task_file, "r", encoding="utf-8") as f:
            task = json.load(f)
        return float(task.get("min_passing_score", 0.7))

    def _task_comment_budget(self, task_id: str) -> int:
        return self.TASK_COMMENT_BUDGET.get(task_id, 8)

    def run_evaluation(self, task_ids: List[str]) -> Dict[str, Any]:
        """
        Run baseline on all tasks.

        Args:
            task_ids: List of task IDs to evaluate

        Returns:
            Results dictionary with scores
        """
        results = {}

        for task_id in task_ids:
            print(f"\n{'='*60}")
            print(f"Evaluating task: {task_id}")
            print(f"{'='*60}")

            try:
                # Reset environment to this task
                reset_response = self.http.post(
                    f"{self.env_base_url}/reset",
                    json={"task_id": task_id}
                )
                reset_response.raise_for_status()
                reset_payload = reset_response.json()
                obs = self._extract_observation(reset_payload, "/reset")

                # Generate review
                print(f"Generating review with {self.provider.upper()} ({self.model})...")
                action = self._normalize_action(
                    self.review_pr(obs["pr_state"], task_id=task_id),
                    max_inline_comments=self._task_comment_budget(task_id),
                )

                print(f"Found {len(action['inline_comments'])} inline comments")
                print(f"Decision: {action['decision']['decision']}")

                # Execute step
                step_response = self.http.post(
                    f"{self.env_base_url}/step",
                    json={"action": action}
                )
                step_response.raise_for_status()
                step_payload = step_response.json()
                result = self._extract_observation(step_payload, "/step")

                # Extract results
                feedback = result.get("feedback", {})
                score = float(feedback.get("score", 0.0))
                threshold = self._load_task_threshold(task_id)
                passed = score >= threshold

                results[task_id] = {
                    "score": score,
                    "passed": passed,
                    "min_passing_score": threshold,
                    "precision": feedback.get("precision", 0.0),
                    "recall": feedback.get("recall", 0.0),
                    "severity_alignment": feedback.get("severity_alignment", 0.0),
                    "true_positives": feedback.get("true_positives", 0),
                    "false_positives": feedback.get("false_positives", 0),
                    "false_negatives": feedback.get("false_negatives", 0)
                }

                print(f"\nResults:")
                print(f"  Score: {results[task_id]['score']:.2f}")
                print(f"  Passed: {'✓' if passed else '✗'}")
                print(f"  Precision: {results[task_id]['precision']:.2f}")
                print(f"  Recall: {results[task_id]['recall']:.2f}")

            except Exception as e:
                print(f"Error evaluating {task_id}: {e}")
                results[task_id] = {
                    "error": str(e),
                    "score": 0.0,
                    "passed": False
                }

        # Calculate average
        valid_scores = [r["score"] for r in results.values() if "error" not in r]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        print(f"\n{'='*60}")
        print(f"AVERAGE SCORE: {avg_score:.2f}")
        print(f"{'='*60}\n")

        return {
            "results": results,
            "average_score": avg_score,
            "model": self.model,
            "provider": self.provider,
            "temperature": self.temperature,
            "seed": self.seed,
            "task_ids": task_ids,
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run baseline inference for PR code review")
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "groq"],
        help="LLM provider to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (overrides provider default)"
    )
    parser.add_argument(
        "--env-url",
        type=str,
        default="http://localhost:8000",
        help="Environment server base URL"
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        nargs="+",
        default=["task1_security_basic", "task2_quality_logic", "task3_advanced_review"],
        help="Task IDs to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="baseline_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for model calls"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for API calls"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of repeated baseline runs for reproducibility stats"
    )

    args = parser.parse_args()

    # Initialize baseline agent
    baseline = BaselineAgent(
        provider=args.provider,
        model=args.model,
        env_base_url=args.env_url,
        temperature=args.temperature,
        seed=args.seed,
    )

    # Run evaluation (single or repeated trials)
    if args.trials <= 1:
        results = baseline.run_evaluation(args.task_ids)
    else:
        trial_results = []
        for trial_idx in range(args.trials):
            print(f"\nRunning trial {trial_idx + 1}/{args.trials}")
            trial_seed = args.seed + trial_idx if args.seed is not None else None
            baseline.seed = trial_seed
            trial_results.append(baseline.run_evaluation(args.task_ids))

        averages = [r["average_score"] for r in trial_results]
        results = {
            "provider": args.provider,
            "model": args.model or baseline.model,
            "temperature": args.temperature,
            "task_ids": args.task_ids,
            "trials": args.trials,
            "trial_average_scores": averages,
            "mean_average_score": mean(averages),
            "std_average_score": pstdev(averages) if len(averages) > 1 else 0.0,
            "trial_results": trial_results,
        }

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
