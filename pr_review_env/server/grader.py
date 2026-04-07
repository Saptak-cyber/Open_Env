"""
Grader for PR Code Review Assistant.

Implements deterministic grading with comment-to-issue matching.
"""

import re
from typing import List, Dict, Any, Set, Tuple, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..models import (
    Action,
    InlineComment,
    GroundTruth,
    GroundTruthIssue,
    ReviewFeedback,
)


class ReviewGrader:
    """Grades code review actions against ground truth."""
    SCORE_EPSILON = 1e-4

    @staticmethod
    def clamp_open_unit_interval(value: float, epsilon: Optional[float] = None) -> float:
        """Map a value into (0, 1) for hackathon Phase-2 validators (strict open interval)."""
        eps = ReviewGrader.SCORE_EPSILON if epsilon is None else float(epsilon)
        return max(eps, min(1.0 - eps, float(value)))

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
            # frontend / DVR / browser
            "leak": {"leak", "leaks", "oom", "gc", "footprint"},
            "blob": {"blob", "chunks", "chunk", "webm", "mediarecorder", "concatenate"},
            "observer": {"observer", "mutationobserver", "mutation", "childlist"},
            "subtree": {"subtree", "descendant", "layout", "reflow", "spa"},
            # JWT / crypto
            "algorithm": {"algorithm", "algorithms", "hs256", "none", "jwk", "signature"},
            "jti": {"jti", "jwe", "kid", "nonce"},
            # money / types
            "decimal": {"decimal", "currency", "money"},
            "float": {"float", "rounding", "precision"},
            # Redis / concurrency
            "incr": {"incr", "increment", "lost", "race"},
            "atomic": {"atomic", "lua", "transaction", "compare"},
            # sessions
            "md5": {"md5", "weak", "guessable", "predictable"},
            "fixation": {"fixation", "regenerate", "rotation", "existing"},
            "scan": {"scan", "scan_iter", "keys", "pattern"},
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
        line_tol = int(task_config.get("grader_line_tolerance", self.line_tolerance))
        thr_override = task_config.get("match_threshold_override")
        thr_override_f = float(thr_override) if thr_override is not None else None

        # Match comments to ground truth issues
        matches, unmatched_comments, unmatched_issues = self._match_comments_to_issues(
            action.inline_comments,
            ground_truth.issues,
            difficulty=difficulty,
            line_tolerance=line_tol,
            match_threshold_override=thr_override_f,
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
            difficulty=difficulty,
        )

        # Apply decision penalties
        score = self._apply_decision_penalties(
            score, action, ground_truth
        )

        # Validator requires strict open interval: 0 < score < 1.
        score = self.clamp_open_unit_interval(score)
        precision = self.clamp_open_unit_interval(precision)
        recall = self.clamp_open_unit_interval(recall)
        coverage = self.clamp_open_unit_interval(coverage)
        severity_alignment = self.clamp_open_unit_interval(severity_alignment)

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
        difficulty: str = "medium",
        line_tolerance: Optional[int] = None,
        match_threshold_override: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], List[InlineComment], List[GroundTruthIssue]]:
        """
        Match comments to ground truth issues via max-weight bipartite assignment.

        Returns:
            (matches, unmatched_comments, unmatched_issues)
            matches: List of {comment, issue, match_score, severity_match_score}
        """
        tol = self.line_tolerance if line_tolerance is None else int(line_tolerance)
        threshold = (
            float(match_threshold_override)
            if match_threshold_override is not None
            else self._difficulty_match_threshold(difficulty)
        )

        c_count = len(comments)
        i_count = len(issues)
        if c_count == 0:
            return [], [], list(issues)
        if i_count == 0:
            return [], list(comments), []

        # cost[c, j] minimized; real issue cols minimize -score; dummy cols cost 0 (unmatched comment)
        inf = 1e6
        dummy_cols = c_count
        cols = i_count + dummy_cols
        cost = np.zeros((c_count, cols), dtype=np.float64)
        score_matrix = np.zeros((c_count, i_count), dtype=np.float64)

        for c in range(c_count):
            for i in range(i_count):
                s = self._compute_match_score(comments[c], issues[i], line_tolerance=tol)
                score_matrix[c, i] = s
                if s > threshold:
                    cost[c, i] = -float(s)
                else:
                    cost[c, i] = inf
            for k in range(dummy_cols):
                cost[c, i_count + k] = 0.0

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_comment_idx: Set[int] = set()
        matched_issue_idx: Set[int] = set()
        matches: List[Dict[str, Any]] = []

        for r, j in zip(row_ind, col_ind):
            if j < i_count and cost[r, j] < inf * 0.5:
                match_score = float(score_matrix[r, j])
                iss = issues[j]
                com = comments[r]
                matches.append({
                    "comment": com,
                    "issue": iss,
                    "match_score": match_score,
                    "severity_match_score": self._severity_match_score(
                        com.severity, iss.severity
                    ),
                })
                matched_comment_idx.add(r)
                matched_issue_idx.add(j)

        unmatched_comments = [comments[idx] for idx in range(c_count) if idx not in matched_comment_idx]
        unmatched_issues = [issues[idx] for idx in range(i_count) if idx not in matched_issue_idx]

        return matches, unmatched_comments, unmatched_issues

    def _compute_match_score(
        self,
        comment: InlineComment,
        issue: GroundTruthIssue,
        line_tolerance: Optional[int] = None,
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

        tol = self.line_tolerance if line_tolerance is None else int(line_tolerance)
        # Line proximity (±tolerance)
        line_distance = abs(comment.line_number - issue.line)
        if line_distance <= tol:
            proximity_score = 1.0 - (line_distance / (tol + 1))
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
        difficulty: str,
    ) -> float:
        """Apply deterministic strictness based on task difficulty."""
        if difficulty == "hard":
            score -= min(0.28, false_positives * 0.07)
        elif difficulty == "medium":
            score -= min(0.12, false_positives * 0.025)
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
        weights: Dict[str, float],
    ) -> float:
        """Weighted score from precision, recall, coverage, and severity alignment.

        Weights are taken from task JSON (``grading_weights``) and renormalized
        so a perfect match on all included components yields 1.0.
        """
        wp = float(weights.get("precision", 0.3))
        wr = float(weights.get("recall", 0.5))
        wc = float(weights.get("coverage", 0.0))
        ws = float(weights.get("severity", 0.0))
        denom = wp + wr + wc + ws
        if denom <= 0:
            return 0.0
        return (
            wp * precision
            + wr * recall
            + wc * coverage
            + ws * severity_alignment
        ) / denom

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
