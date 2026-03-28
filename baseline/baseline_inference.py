"""
OpenAI baseline for PR Code Review Assistant.

This script runs a baseline evaluation using OpenAI's GPT-4 model.
"""

import os
import json
import requests
from typing import Dict, List, Any
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

Be thorough but precise. Focus on real issues, not nitpicks."""


class OpenAIBaseline:
    """Baseline agent using OpenAI API."""

    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        base_url: str = "http://localhost:8000"
    ):
        """
        Initialize baseline.

        Args:
            model: OpenAI model to use
            base_url: Base URL for the environment server
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.base_url = base_url

    def _build_prompt(self, pr_state: Dict[str, Any]) -> str:
        """Build prompt from PR state."""
        prompt = f"""Review this pull request:

PR #{pr_state['pr_id']}: {pr_state['metadata']['title']}
Description: {pr_state['metadata']['description']}
Author: {pr_state['metadata']['author']}

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

    def review_pr(self, pr_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate review using OpenAI.

        Args:
            pr_state: PR state from environment

        Returns:
            Action dictionary
        """
        prompt = self._build_prompt(pr_state)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2048
            )

            action = json.loads(response.choices[0].message.content)
            return action

        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            # Return minimal valid action
            return {
                "inline_comments": [],
                "general_comments": [],
                "decision": {
                    "decision": "comment",
                    "summary": f"Error during review: {e}"
                }
            }

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
                reset_response = requests.post(
                    f"{self.base_url}/reset",
                    json={"task_id": task_id}
                )
                reset_response.raise_for_status()
                obs = reset_response.json()

                # Generate review
                print("Generating review with OpenAI...")
                action = self.review_pr(obs["pr_state"])

                print(f"Found {len(action['inline_comments'])} inline comments")
                print(f"Decision: {action['decision']['decision']}")

                # Execute step
                step_response = requests.post(
                    f"{self.base_url}/step",
                    json={"action": action}
                )
                step_response.raise_for_status()
                result = step_response.json()

                # Extract results
                feedback = result.get("feedback", {})
                passed = result.get("metadata", {}).get("passed", False)

                results[task_id] = {
                    "score": feedback.get("score", 0.0),
                    "passed": passed,
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
            "model": self.model
        }


def main():
    """Main entry point."""
    baseline = OpenAIBaseline()

    task_ids = [
        "task1_security_basic",
        "task2_quality_logic",
        "task3_advanced_review"
    ]

    results = baseline.run_evaluation(task_ids)

    # Save results
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
