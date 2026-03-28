"""
Pydantic models for PR Code Review Assistant environment.

These models define the action and observation spaces for the OpenEnv interface.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from openenv.core.env_server.types import (
    Action as BaseAction,
    Observation as BaseObservation,
)


# ============================================================================
# PR State Representation
# ============================================================================

class FileDiff(BaseModel):
    """Represents changes to a single file in the PR."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="File path relative to repo root")
    language: str = Field(description="Programming language (python, javascript, etc.)")
    additions: List[str] = Field(description="Lines added with line numbers")
    deletions: List[str] = Field(description="Lines removed with line numbers")
    context: List[str] = Field(description="Surrounding context lines")
    hunks: List[Dict[str, Any]] = Field(
        description="Structured diff hunks with start_line, end_line, content"
    )


class PRMetadata(BaseModel):
    """Metadata about the pull request."""

    model_config = ConfigDict(extra="forbid")

    title: str
    description: str
    author: str
    target_branch: str = "main"
    files_changed: int
    additions_total: int
    deletions_total: int


class GroundTruthIssue(BaseModel):
    """A single issue in the ground truth."""

    model_config = ConfigDict(extra="forbid")

    file: str = Field(description="File path where issue is located")
    line: int = Field(description="Line number of the issue")
    severity: Literal["info", "warning", "error", "critical"]
    category: Literal[
        "security", "bug", "performance", "style",
        "maintainability", "testing", "documentation"
    ]
    description: str = Field(description="Description of the issue")
    cwe: Optional[str] = Field(None, description="CWE identifier if applicable")


class GroundTruth(BaseModel):
    """Ground truth for grading (hidden from agent)."""

    model_config = ConfigDict(extra="forbid")

    issues: List[GroundTruthIssue] = Field(
        description="All known issues in this PR"
    )


class PRState(BaseModel):
    """Complete state of a pull request for review."""

    model_config = ConfigDict(extra="forbid")

    pr_id: str = Field(description="Unique identifier for this PR")
    metadata: PRMetadata
    files: List[FileDiff] = Field(description="All changed files")
    ground_truth: GroundTruth = Field(
        description="Hidden ground truth for grading (not visible to agent in observation)"
    )


# ============================================================================
# Action Space
# ============================================================================

class InlineComment(BaseModel):
    """Comment on a specific line of code."""

    model_config = ConfigDict(extra="forbid")

    file_path: str
    line_number: int
    comment: str = Field(description="The review comment text")
    severity: Literal["info", "warning", "error", "critical"] = Field(
        description="Severity level of the issue"
    )
    category: Literal[
        "security", "bug", "performance", "style",
        "maintainability", "testing", "documentation"
    ]
    suggested_fix: Optional[str] = Field(
        None, description="Optional code suggestion"
    )


class GeneralComment(BaseModel):
    """General comment not tied to a specific line."""

    model_config = ConfigDict(extra="forbid")

    comment: str
    category: Literal[
        "architecture", "approach", "testing", "documentation", "general"
    ]


class ReviewDecision(BaseModel):
    """Final review decision."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "request_changes", "comment"]
    summary: str = Field(description="Overall summary of the review")


class Action(BaseAction):
    """Agent action in the PR review environment."""

    model_config = ConfigDict(extra="forbid")

    inline_comments: List[InlineComment] = Field(
        default_factory=list,
        description="Comments on specific lines"
    )
    general_comments: List[GeneralComment] = Field(
        default_factory=list,
        description="General review comments"
    )
    decision: ReviewDecision = Field(
        description="Final review decision"
    )


# ============================================================================
# Observation Space
# ============================================================================

class ReviewFeedback(BaseModel):
    """Feedback on the review quality (post-grading)."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0, description="Overall review score")
    true_positives: int = Field(description="Correctly identified issues")
    false_positives: int = Field(description="Incorrectly flagged issues")
    false_negatives: int = Field(description="Missed issues")
    coverage: float = Field(ge=0.0, le=1.0, description="% of critical lines reviewed")
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    severity_alignment: float = Field(
        ge=0.0, le=1.0,
        description="How well severity levels match ground truth"
    )


class PRStateForAgent(BaseModel):
    """PR state without ground truth (what the agent sees)."""

    model_config = ConfigDict(extra="forbid")

    pr_id: str
    metadata: PRMetadata
    files: List[FileDiff]


class Observation(BaseObservation):
    """Observation returned after each step."""

    model_config = ConfigDict(extra="forbid")

    pr_state: PRStateForAgent = Field(description="Current PR state")
    feedback: Optional[ReviewFeedback] = Field(
        None,
        description="Grading feedback (only after terminal action)"
    )
    # Required by BaseObservation
    done: bool = Field(default=False, description="Episode terminated")
    reward: Optional[float] = Field(default=None, description="Reward from last action")
    # metadata is inherited from BaseObservation
