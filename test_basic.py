"""
Simple test to verify the environment works correctly.
"""

import sys
sys.path.insert(0, '.')

from pr_review_env import (
    Action,
    InlineComment,
    ReviewDecision,
    PRReviewEnvironment
)

def test_basic_environment():
    """Test basic environment functionality."""
    print("Testing PR Review Environment...")

    # Create environment
    env = PRReviewEnvironment()
    print("✓ Environment created")

    # Reset to task 1
    obs = env.reset(task_id="task1_security_basic")
    print(f"✓ Environment reset (task: {obs.metadata['task_id']})")
    print(f"  PR ID: {obs.pr_state.pr_id}")
    print(f"  Files changed: {len(obs.pr_state.files)}")
    print(f"  Done: {obs.done}")

    # Create a simple action
    action = Action(
        inline_comments=[
            InlineComment(
                file_path="app/auth.py",
                line_number=11,
                comment="SQL Injection vulnerability detected",
                severity="critical",
                category="security",
                suggested_fix="Use parameterized queries"
            )
        ],
        general_comments=[],
        decision=ReviewDecision(
            decision="request_changes",
            summary="Critical security issues found"
        ),
        submit=True
    )
    print("✓ Action created")

    # Take step
    obs = env.step(action)
    print(f"✓ Step executed")
    print(f"  Score: {obs.reward:.2f}")
    print(f"  Done: {obs.done}")
    print(f"  Passed: {obs.metadata.get('passed', False)}")

    if obs.feedback:
        print(f"\nFeedback:")
        print(f"  True Positives: {obs.feedback.true_positives}")
        print(f"  False Positives: {obs.feedback.false_positives}")
        print(f"  False Negatives: {obs.feedback.false_negatives}")
        print(f"  Precision: {obs.feedback.precision:.2f}")
        print(f"  Recall: {obs.feedback.recall:.2f}")

    print("\n✅ All tests passed!")

if __name__ == "__main__":
    try:
        test_basic_environment()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
