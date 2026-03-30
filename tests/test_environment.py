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
    assert isinstance(mid.reward, float)

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


def test_intermediate_reward_penalizes_duplicates():
    env = PRReviewEnvironment()
    env.reset(task_id="task1_security_basic")

    action1 = Action(
        inline_comments=[
            InlineComment(
                file_path="app/auth.py",
                line_number=11,
                comment="SQL injection risk in query construction.",
                severity="critical",
                category="security",
            )
        ],
        general_comments=[],
        decision=None,
        submit=False,
    )
    r1 = env.step(action1).reward

    action2 = Action(
        inline_comments=[
            InlineComment(
                file_path="app/auth.py",
                line_number=12,  # line-jitter duplicate
                comment="Possible SQL injection due to string concatenation.",
                severity="critical",
                category="security",
            )
        ],
        general_comments=[],
        decision=None,
        submit=False,
    )
    r2 = env.step(action2).reward

    assert isinstance(r1, float) and isinstance(r2, float)
    assert r2 <= r1
