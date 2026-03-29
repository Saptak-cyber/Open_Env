from pr_review_env import Action, InlineComment, PRReviewEnvironment, ReviewDecision


def test_multistep_episode_progression():
    env = PRReviewEnvironment()
    obs = env.reset(task_id="task1_security_basic")
    assert obs.done is False
    assert obs.metadata["max_steps"] == 5

    partial_action = Action(
        inline_comments=[
            InlineComment(
                file_path="app/auth.py",
                line_number=11,
                comment="Potential SQL injection in f-string query.",
                severity="critical",
                category="security",
                suggested_fix="Use parameterized statements."
            )
        ],
        general_comments=[],
        decision=None,
        submit=False,
    )
    mid = env.step(partial_action)
    assert mid.done is False
    assert mid.feedback is None
    assert mid.metadata["finalized"] is False

    final_action = Action(
        inline_comments=[
            InlineComment(
                file_path="app/profile.py",
                line_number=27,
                comment="Unsanitized user-controlled input rendered in HTML.",
                severity="error",
                category="security",
            )
        ],
        general_comments=[],
        decision=ReviewDecision(
            decision="request_changes",
            summary="Security issues should be fixed before merge."
        ),
        submit=True,
    )
    done = env.step(final_action)
    assert done.done is True
    assert done.feedback is not None
    assert isinstance(done.feedback.score, float)
    assert done.metadata["finalized"] is True
