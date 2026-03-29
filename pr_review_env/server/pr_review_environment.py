"""
PR Code Review Assistant Environment.

Implements the OpenEnv interface for code review training.
"""

import json
import random
from pathlib import Path
from typing import Optional, Any, Dict
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..models import Action, Observation, PRState, PRStateForAgent, GroundTruth, ReviewDecision
from .grader import ReviewGrader


class PRReviewEnvironment(Environment):
    """OpenEnv environment for PR code review."""

    def __init__(self):
        """Initialize the environment."""
        self.grader = ReviewGrader()
        self._state = State(episode_id=None, step_count=0)
        self.current_task: Optional[Dict[str, Any]] = None
        self.current_pr: Optional[PRState] = None
        self.episode_done = False
        self.max_steps = 5
        self.submitted_inline_comments = []
        self.submitted_general_comments = []
        self.last_feedback = None

        # Load all available tasks
        self.tasks = self._load_all_tasks()

    def _load_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Load all task definitions from tasks/ directory."""
        tasks = {}
        tasks_dir = Path(__file__).parent.parent.parent / "tasks"

        if not tasks_dir.exists():
            return tasks

        for task_file in tasks_dir.glob("*.json"):
            try:
                with open(task_file, "r") as f:
                    task_data = json.load(f)
                    task_id = task_data["task_id"]
                    tasks[task_id] = task_data
            except Exception as e:
                print(f"Error loading task {task_file}: {e}")

        return tasks

    def _load_task(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a specific task or random task.

        Args:
            task_id: Specific task ID, or None for random

        Returns:
            Task configuration dictionary
        """
        if task_id is None:
            # Pick random task
            if not self.tasks:
                raise ValueError("No tasks available")
            task_id = random.choice(list(self.tasks.keys()))

        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        return self.tasks[task_id]

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any
    ) -> Observation:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed (for future stochasticity)
            episode_id: Optional episode ID
            task_id: Specific task to load, or None for random

        Returns:
            Initial observation with PR state
        """
        if seed is not None:
            random.seed(seed)

        # Initialize state
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0
        )

        # Load task
        self.current_task = self._load_task(task_id)

        # Extract PR state from task
        pr_data = self.current_task["pr_scenario"]
        self.current_pr = PRState(**pr_data)

        # Reset episode flag
        self.episode_done = False
        self.submitted_inline_comments = []
        self.submitted_general_comments = []
        self.last_feedback = None

        # Create observation (without ground truth for agent)
        pr_state_for_agent = PRStateForAgent(
            pr_id=self.current_pr.pr_id,
            metadata=self.current_pr.metadata,
            files=self.current_pr.files
        )

        return Observation(
            pr_state=pr_state_for_agent,
            feedback=None,
            done=False,
            reward=0.0,
            metadata={
                "task_id": self.current_task["task_id"],
                "difficulty": self.current_task["difficulty"],
                "max_steps": self.max_steps,
            }
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any
    ) -> Observation:
        """
        Execute one step of the environment.

        Args:
            action: Review action to execute
            timeout_s: Optional timeout (not used currently)

        Returns:
            Observation with feedback and reward
        """
        if self.episode_done:
            raise RuntimeError("Episode already done. Call reset() first.")

        if self.current_pr is None or self.current_task is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Increment step count
        self._state.step_count += 1

        # Keep cumulative review state across the trajectory.
        self.submitted_inline_comments.extend(action.inline_comments)
        self.submitted_general_comments.extend(action.general_comments)

        cumulative_action = Action(
            inline_comments=self.submitted_inline_comments,
            general_comments=self.submitted_general_comments,
            decision=action.decision,
            submit=action.submit
        )

        should_finalize = (
            action.submit
            or action.decision is not None
            or self._state.step_count >= self.max_steps
        )

        feedback = None
        if should_finalize:
            if cumulative_action.decision is None:
                cumulative_action.decision = ReviewDecision(
                    decision="request_changes",
                    summary="Auto-finalized at max steps without explicit final decision."
                )
            feedback = self.grader.grade_review(
                cumulative_action,
                self.current_pr.ground_truth,
                self.current_task
            )
            reward = self._compute_reward(feedback, cumulative_action)
            self.episode_done = True
        else:
            # Partial-trajectory reward: encourage new signal over previous step.
            partial_feedback = self.grader.grade_review(
                Action(
                    inline_comments=self.submitted_inline_comments,
                    general_comments=self.submitted_general_comments,
                    decision=None,
                    submit=False
                ),
                self.current_pr.ground_truth,
                self.current_task
            )
            reward = self._compute_intermediate_reward(partial_feedback, action)
            self.last_feedback = partial_feedback

        # Create PR state for agent (without ground truth)
        pr_state_for_agent = PRStateForAgent(
            pr_id=self.current_pr.pr_id,
            metadata=self.current_pr.metadata,
            files=self.current_pr.files
        )

        return Observation(
            pr_state=pr_state_for_agent,
            feedback=feedback,
            done=self.episode_done,
            reward=reward,
            metadata={
                "task_id": self.current_task["task_id"],
                "step_count": self._state.step_count,
                "max_steps": self.max_steps,
                "passed": (
                    feedback.score >= self.current_task["min_passing_score"]
                    if feedback is not None else None
                ),
                "finalized": self.episode_done,
            }
        )

    @property
    def state(self) -> State:
        """Return current environment state."""
        return self._state

    def _compute_reward(self, feedback, action: Action) -> float:
        """
        Compute reward with multiple signals for partial progress.

        Components:
        1. Base score (0.0-1.0 from grader) - 90% weight
        2. Early detection bonus - find issues in first files
        3. Actionability bonus - provide suggested fixes
        4. Coverage reward - incentivize thorough review
        5. False positive penalty - scaled by severity
        """
        # Base reward from grader (primary signal)
        base_reward = feedback.score

        # Early detection bonus (0-0.05)
        # Bonus for finding critical issues early in review
        early_bonus = self._compute_early_detection_bonus(action)

        # Actionability bonus (0-0.03)
        # Bonus for providing suggested fixes
        actionability_bonus = self._compute_actionability_bonus(action, feedback)

        # Coverage reward (0-0.02)
        coverage_reward = 0.02 * feedback.coverage

        # False positive penalty (0-0.1)
        fp_penalty = self._compute_fp_penalty(feedback)

        # Combine signals
        total_reward = (
            base_reward +
            early_bonus +
            actionability_bonus +
            coverage_reward -
            fp_penalty
        )

        # Bound to reasonable range
        return max(-0.1, min(1.1, total_reward))

    def _compute_intermediate_reward(self, partial_feedback, action: Action) -> float:
        """
        Intermediate reward for multi-step trajectories.
        Rewards newly found issues and coverage gains while discouraging spam/duplicates.
        """
        prev_tp = self.last_feedback.true_positives if self.last_feedback else 0
        prev_cov = self.last_feedback.coverage if self.last_feedback else 0.0

        new_tp = max(0, partial_feedback.true_positives - prev_tp)
        coverage_gain = max(0.0, partial_feedback.coverage - prev_cov)

        existing_keys = {
            (c.file_path, c.line_number, c.category)
            for c in self.submitted_inline_comments[:-len(action.inline_comments or [])]
        }
        duplicate_count = sum(
            1 for c in action.inline_comments
            if (c.file_path, c.line_number, c.category) in existing_keys
        )

        over_comment_penalty = max(0, len(action.inline_comments) - 4) * 0.01
        reward = (0.12 * new_tp) + (0.08 * coverage_gain) - (0.03 * duplicate_count) - over_comment_penalty
        return max(-0.1, min(0.35, reward))

    def _compute_early_detection_bonus(self, action: Action) -> float:
        """Bonus for finding critical issues in early files."""
        if not action.inline_comments:
            return 0.0

        # Check if first few comments include critical issues
        first_comments = action.inline_comments[:3]
        critical_count = sum(
            1 for c in first_comments if c.severity == "critical"
        )

        return min(0.05, critical_count * 0.02)

    def _compute_actionability_bonus(self, action: Action, feedback) -> float:
        """Bonus for providing suggested fixes."""
        if feedback.true_positives == 0:
            return 0.0

        # Count comments with suggested fixes
        with_fixes = sum(
            1 for c in action.inline_comments if c.suggested_fix
        )

        # Ratio of TP comments with fixes
        ratio = with_fixes / max(1, len(action.inline_comments))

        return min(0.03, ratio * 0.03)

    def _compute_fp_penalty(self, feedback) -> float:
        """Penalty for false positives."""
        if feedback.false_positives == 0:
            return 0.0

        # Scale penalty by number of false positives
        return min(0.1, feedback.false_positives * 0.02)
