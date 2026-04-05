"""
Grader for PR Code Review Assistant.

Implements deterministic grading with comment-to-issue matching.
"""

import re
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
        self.keyword_synonyms: Dict[str, Set[str]] = {
            "sql": {"sql", "query", "database"},
            "injection": {"injection", "inject", "unsanitized"},
            "xss": {"xss", "script", "javascript", "crosssite", "cross-site"},
            "redirect": {"redirect", "openredirect", "open-redirect"},
            "race": {"race", "concurrency", "concurrent", "atomic"},
            "password": {"password", "plaintext", "hash", "hashed", "bcrypt"},
            "cache": {"cache", "ttl", "eviction", "memory"},
            "limit": {"limit", "pagination", "slice"},
            "exception": {"exception", "error", "swallow", "retry"},
            # auth/session
            "jwt": {"jwt", "token", "bearer", "claim", "claims"},
            "refresh": {"refresh", "rotation", "rotate", "reissue"},
            "logout": {"logout", "invalidate", "revocation", "revoke"},
            "csrf": {"csrf", "xsrf", "sameSite", "samesite"},
            "cookie": {"cookie", "httponly", "secure", "samesite"},
            "rate": {"rate", "ratelimit", "throttle", "bruteforce", "brute-force"},
            "session": {"session", "sid", "stateful"},
            # async pipeline / webhooks
            "webhook": {"webhook", "signature", "hmac"},
            "replay": {"replay", "nonce", "timestamp", "freshness"},
            "idempotency": {"idempotency", "idempotent", "dedupe", "dedup", "duplicate"},
            "transaction": {"transaction", "atomic", "commit", "rollback"},
            # data export / access control / multitenancy
            "tenant": {"tenant", "tenancy", "org", "workspace", "account"},
            "authorize": {"authorize", "authz", "authorization", "permission", "rbac", "acl"},
            "pii": {"pii", "personal", "sensitive", "ssn", "email", "phone"},
            "path": {"path", "filepath", "directory", "tmp", "tempfile", "symlink"},
        }

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
        difficulty = str(task_config.get("difficulty", "medium")).lower()

        # Match comments to ground truth issues
        matches, unmatched_comments, unmatched_issues = self._match_comments_to_issues(
            action.inline_comments,
            ground_truth.issues,
            difficulty=difficulty
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

        # Difficulty-sensitive penalties: hard tasks punish loose findings more.
        score = self._apply_difficulty_penalties(
            score=score,
            false_positives=false_positives,
            severity_alignment=severity_alignment,
            difficulty=difficulty,
            matches_count=len(matches)
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

    def _normalize_file_path(self, path: str) -> str:
        """Normalize paths so model/repo style differences still match ground truth."""
        if not path:
            return ""
        p = path.strip().replace("\\", "/")
        while p.startswith("./"):
            p = p[2:]
        return p.lower()

    def _match_comments_to_issues(
        self,
        comments: List[InlineComment],
        issues: List[GroundTruthIssue],
        difficulty: str = "medium"
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

        threshold = self._difficulty_match_threshold(difficulty)

        # Process comments with fewest viable issues first to reduce greedy order bias.
        flex_order: List[Tuple[int, int]] = []
        for comment_idx, comment in enumerate(comments):
            viable = 0
            for issue in issues:
                if self._compute_match_score(comment, issue) > threshold:
                    viable += 1
            flex_order.append((viable, comment_idx))
        flex_order.sort(key=lambda x: (x[0], x[1]))

        for _, comment_idx in flex_order:
            comment = comments[comment_idx]
            best_match_idx = None
            best_match_score = 0.0

            for issue_idx, issue in enumerate(issues):
                if issue_idx in matched_issues:
                    continue

                match_score = self._compute_match_score(comment, issue)

                if match_score > best_match_score and match_score > threshold:
                    best_match_score = match_score
                    best_match_idx = issue_idx

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

        if self._normalize_file_path(comment.file_path) != self._normalize_file_path(issue.file):
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

        # Keyword overlap: comment text + optional suggested_fix (models often put evidence there)
        issue_keywords = self._normalized_tokens(issue.description)
        comment_blob = comment.comment
        if comment.suggested_fix:
            comment_blob = f"{comment_blob} {comment.suggested_fix}"
        comment_keywords = self._normalized_tokens(comment_blob)
        issue_expanded = self._expand_with_synonyms(issue_keywords)
        comment_expanded = self._expand_with_synonyms(comment_keywords)
        overlap = len(issue_expanded & comment_expanded)
        if overlap > 0:
            score += 0.1
        if overlap >= 2:
            score += 0.05

        return min(1.0, score)

    def _difficulty_match_threshold(self, difficulty: str) -> float:
        if difficulty == "easy":
            return 0.28
        if difficulty == "hard":
            return 0.45
        return 0.35

    def _apply_difficulty_penalties(
        self,
        score: float,
        false_positives: int,
        severity_alignment: float,
        difficulty: str,
        matches_count: int
    ) -> float:
        """Apply deterministic strictness based on task difficulty."""
        if difficulty == "hard":
            score -= min(0.28, false_positives * 0.07)
            if matches_count > 0:
                score -= (1.0 - severity_alignment) * 0.14
        elif difficulty == "medium":
            score -= min(0.12, false_positives * 0.025)
            if matches_count > 0:
                score -= (1.0 - severity_alignment) * 0.04
        else:
            # Keep easy tasks permissive.
            score -= min(0.05, false_positives * 0.01)
        return score

    def _normalized_tokens(self, text: str) -> Set[str]:
        """Normalize tokens for deterministic matching."""
        tokens = re.findall(r"[a-z0-9\-]+", text.lower())
        normalized: Set[str] = set()
        for token in tokens:
            t = token.strip("-")
            if not t:
                continue
            # very small deterministic singularization heuristic
            if t.endswith("s") and len(t) > 4:
                t = t[:-1]
            normalized.add(t)
        return normalized

    def _expand_with_synonyms(self, tokens: Set[str]) -> Set[str]:
        """Expand token set with small deterministic synonym groups."""
        expanded = set(tokens)
        for token in list(tokens):
            for key, syns in self.keyword_synonyms.items():
                if token == key or token in syns:
                    expanded.add(key)
                    expanded.update(syns)
        return expanded

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

        files_with_issues = {
            self._normalize_file_path(issue.file) for issue in ground_truth.issues
        }

        reviewed_files = {
            self._normalize_file_path(comment.file_path)
            for comment in action.inline_comments
        }

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
        if action.decision is None:
            return score

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
