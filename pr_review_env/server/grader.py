"""
Grader for PR Code Review Assistant.

Implements deterministic grading with comment-to-issue matching.
"""

from typing import List, Dict, Any, Set, Tuple
from ..models import (
    Action,
    InlineComment,
    GroundTruth,
    GroundTruthIssue,
    ReviewFeedback,
)


class ReviewGrader:
    """Grades code review actions against ground truth."""

    def __init__(self, line_tolerance: int = 2):
        """
        Initialize grader.

        Args:
            line_tolerance: Number of lines tolerance for matching (±2 default)
        """
        self.line_tolerance = line_tolerance

    def grade_review(
        self,
        action: Action,
        ground_truth: GroundTruth,
        task_config: Dict[str, Any]
    ) -> ReviewFeedback:
        """
        Grade a review action against ground truth.

        Args:
            action: The review action to grade
            ground_truth: Ground truth issues
            task_config: Task configuration with grading weights

        Returns:
            ReviewFeedback with metrics
        """
        # Match comments to ground truth issues
        matches, unmatched_comments, unmatched_issues = self._match_comments_to_issues(
            action.inline_comments,
            ground_truth.issues
        )

        # Calculate metrics
        true_positives = len(matches)
        false_positives = len(unmatched_comments)
        false_negatives = len(unmatched_issues)

        # Precision and recall
        precision = self._calculate_precision(true_positives, false_positives)
        recall = self._calculate_recall(true_positives, false_negatives)

        # Coverage calculation
        coverage = self._calculate_coverage(action, ground_truth)

        # Severity alignment
        severity_alignment = self._calculate_severity_alignment(matches)

        # Calculate weighted score
        weights = task_config.get("grading_weights", {})
        score = self._calculate_weighted_score(
            precision, recall, coverage, severity_alignment, weights
        )

        # Apply decision penalties
        score = self._apply_decision_penalties(
            score, action, ground_truth
        )

        # Ensure score is in valid range
        score = max(0.0, min(1.0, score))

        return ReviewFeedback(
            score=score,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            coverage=coverage,
            precision=precision,
            recall=recall,
            severity_alignment=severity_alignment
        )

    def _match_comments_to_issues(
        self,
        comments: List[InlineComment],
        issues: List[GroundTruthIssue]
    ) -> Tuple[List[Dict[str, Any]], List[InlineComment], List[GroundTruthIssue]]:
        """
        Match comments to ground truth issues.

        Returns:
            (matches, unmatched_comments, unmatched_issues)
            matches: List of {comment, issue, severity_match_score}
        """
        matched_issues: Set[int] = set()
        matched_comments: Set[int] = set()
        matches: List[Dict[str, Any]] = []

        # For each comment, find best matching issue
        for comment_idx, comment in enumerate(comments):
            best_match_idx = None
            best_match_score = 0.0

            for issue_idx, issue in enumerate(issues):
                if issue_idx in matched_issues:
                    continue

                # Check if this comment matches this issue
                match_score = self._compute_match_score(comment, issue)

                if match_score > best_match_score and match_score > 0.3:  # Threshold
                    best_match_score = match_score
                    best_match_idx = issue_idx

            # If found a match, record it
            if best_match_idx is not None:
                matched_comments.add(comment_idx)
                matched_issues.add(best_match_idx)
                matches.append({
                    "comment": comment,
                    "issue": issues[best_match_idx],
                    "match_score": best_match_score,
                    "severity_match_score": self._severity_match_score(
                        comment.severity, issues[best_match_idx].severity
                    )
                })

        # Collect unmatched
        unmatched_comments = [
            c for idx, c in enumerate(comments)
            if idx not in matched_comments
        ]
        unmatched_issues = [
            i for idx, i in enumerate(issues)
            if idx not in matched_issues
        ]

        return matches, unmatched_comments, unmatched_issues

    def _compute_match_score(
        self,
        comment: InlineComment,
        issue: GroundTruthIssue
    ) -> float:
        """
        Compute how well a comment matches an issue.

        Factors:
        - File path match (0/1)
        - Line proximity (decays with distance)
        - Category overlap (0/1)
        - Keyword overlap (0-1)
        """
        score = 0.0

        # File path must match
        if comment.file_path != issue.file:
            return 0.0

        score += 0.3  # File match

        # Line proximity (±tolerance)
        line_distance = abs(comment.line_number - issue.line)
        if line_distance <= self.line_tolerance:
            proximity_score = 1.0 - (line_distance / (self.line_tolerance + 1))
            score += 0.4 * proximity_score
        else:
            return 0.0  # Too far, no match

        # Category alignment
        if comment.category == issue.category:
            score += 0.2

        # Keyword overlap (simple check)
        issue_keywords = set(issue.description.lower().split())
        comment_keywords = set(comment.comment.lower().split())
        overlap = len(issue_keywords & comment_keywords)
        if overlap > 0:
            score += 0.1

        return score

    def _severity_match_score(self, comment_severity: str, issue_severity: str) -> float:
        """
        Score how well comment severity matches issue severity.

        Returns:
            1.0 for exact match
            0.5 for off-by-one
            0.0 for off-by-two or more
        """
        severity_levels = ["info", "warning", "error", "critical"]
        try:
            comment_idx = severity_levels.index(comment_severity)
            issue_idx = severity_levels.index(issue_severity)
            diff = abs(comment_idx - issue_idx)

            if diff == 0:
                return 1.0
            elif diff == 1:
                return 0.5
            else:
                return 0.0
        except ValueError:
            return 0.0

    def _calculate_precision(self, true_positives: int, false_positives: int) -> float:
        """Calculate precision."""
        total = true_positives + false_positives
        if total == 0:
            return 1.0  # No comments = no false positives
        return true_positives / total

    def _calculate_recall(self, true_positives: int, false_negatives: int) -> float:
        """Calculate recall."""
        total = true_positives + false_negatives
        if total == 0:
            return 1.0  # No issues = perfect recall
        return true_positives / total

    def _calculate_coverage(self, action: Action, ground_truth: GroundTruth) -> float:
        """
        Calculate coverage - % of files with issues that were reviewed.

        Simplification: If any comment on a file with issues, count as covered.
        """
        if len(ground_truth.issues) == 0:
            return 1.0

        # Files with issues
        files_with_issues = {issue.file for issue in ground_truth.issues}

        # Files that were reviewed
        reviewed_files = {comment.file_path for comment in action.inline_comments}

        # Count covered files
        covered_files = files_with_issues & reviewed_files

        return len(covered_files) / len(files_with_issues)

    def _calculate_severity_alignment(self, matches: List[Dict[str, Any]]) -> float:
        """Calculate average severity alignment across all matches."""
        if len(matches) == 0:
            return 0.0

        total_alignment = sum(m["severity_match_score"] for m in matches)
        return total_alignment / len(matches)

    def _calculate_weighted_score(
        self,
        precision: float,
        recall: float,
        coverage: float,
        severity_alignment: float,
        weights: Dict[str, float]
    ) -> float:
        """Calculate weighted score from components."""
        score = (
            weights.get("precision", 0.3) * precision +
            weights.get("recall", 0.5) * recall +
            weights.get("severity", 0.2) * severity_alignment +
            weights.get("coverage", 0.0) * coverage
        )
        return score

    def _apply_decision_penalties(
        self,
        score: float,
        action: Action,
        ground_truth: GroundTruth
    ) -> float:
        """Apply penalties based on review decision."""
        # Check if there are critical issues
        has_critical = any(
            issue.severity == "critical" for issue in ground_truth.issues
        )

        # Penalty for approving with critical issues
        if action.decision.decision == "approve" and has_critical:
            score *= 0.8  # 20% penalty

        # Smaller penalty for requesting changes when there are no critical issues
        if action.decision.decision == "request_changes" and not has_critical:
            score *= 0.95  # 5% penalty

        return score
