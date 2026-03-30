---
title: PR Code Review Assistant
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
tags: [openenv, code-review, security, ai-agents]
---

# PR Code Review Assistant 🔍

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment for training AI agents to perform high-quality code reviews. Agents learn to identify security vulnerabilities, logic bugs, and code quality issues in pull requests.

## 🎯 Why This Environment?

Code review is a critical bottleneck in software development:
- Human reviewers miss **30-50% of bugs** (Microsoft Research)
- Security vulnerabilities cost **$4M average per breach** (IBM 2023)
- PR review delays slow down development velocity

This environment provides a realistic training ground for AI agents to learn code review skills with:
- **Progressive difficulty** (easy → medium → hard)
- **Deterministic grading** (precision, recall, severity alignment)
- **Multi-signal rewards** for partial progress
- **Real-world patterns** from OWASP, CWE, and production codebases

---

## 📋 Environment Description

The PR Code Review Assistant simulates realistic pull request review scenarios where agents must:
1. Analyze code changes across multiple files
2. Identify security vulnerabilities, bugs, and quality issues
3. Provide actionable feedback with severity levels
4. Make final review decisions (approve/request changes/comment)

Each episode presents a PR with known ground truth issues. The agent is graded on:
- **Precision**: Avoiding false positives
- **Recall**: Finding all real issues
- **Coverage**: Reviewing critical code sections
- **Severity Alignment**: Correctly prioritizing issues

---

## 🎮 Action Space

Agents submit structured reviews with four components:

### 1. Inline Comments
Comment on specific lines of code:
```python
{
  "file_path": "app/auth.py",
  "line_number": 11,
  "comment": "SQL Injection vulnerability - use parameterized queries",
  "severity": "critical",  # info | warning | error | critical
  "category": "security",  # security | bug | performance | style | ...
  "suggested_fix": "cursor.execute('SELECT * FROM users WHERE username=%s', (username,))"
}
```

### 2. General Comments
High-level feedback not tied to specific lines:
```python
{
  "comment": "Consider adding integration tests for the authentication flow",
  "category": "testing"  # architecture | approach | testing | documentation | general
}
```

### 3. Review Decision
Final verdict:
```python
{
  "decision": "request_changes",  # approve | request_changes | comment
  "summary": "Found 3 critical security issues that must be addressed before merging"
}
```

### 4. Submit Flag (Multi-Step Episodes)
Set `submit=true` when finalizing the episode:
```python
{
  "submit": true
}
```

---

## 👁️ Observation Space

Agents observe the PR state without seeing ground truth:

```python
{
  "pr_state": {
    "pr_id": "PR-001",
    "metadata": {
      "title": "Add user authentication",
      "description": "...",
      "author": "junior-dev",
      "files_changed": 2,
      "additions_total": 15,
      "deletions_total": 0
    },
    "files": [
      {
        "path": "app/auth.py",
        "language": "python",
        "additions": ["10: def login(username, password):", ...],
        "deletions": [],
        "context": [...],
        "hunks": [...]
      }
    ]
  },
  "feedback": {  # Only after terminal step
    "score": 0.82,
    "true_positives": 3,
    "false_positives": 1,
    "false_negatives": 0,
    "precision": 0.75,
    "recall": 1.0,
    "severity_alignment": 0.95
  },
  "done": false,
  "reward": 0.0
}
```

---

## 📊 Tasks & Difficulty

### Task 1: Security Basics (Easy)
**Focus**: Obvious OWASP vulnerabilities
- SQL Injection with string concatenation
- Cross-Site Scripting (XSS) in HTML output
- Open Redirect vulnerability
- **Min passing score**: 0.7
- **Expected baseline (GPT-4)**: 0.82 ✓

### Task 2: Quality + Logic (Medium)
**Focus**: Code quality and logic bugs
- Race conditions in payment processing
- Weak input validation
- Plaintext password storage
- Missing brute force protection
- Style issues (imports in functions)
- **Min passing score**: 0.75
- **Expected baseline (GPT-4)**: 0.71 ✗

### Task 3: Advanced Review (Hard)
**Focus**: Architectural and subtle issues
- Non-thread-safe singleton pattern
- Cache without access control (privilege escalation)
- Memory leaks (no TTL)
- N+1 query problems
- Silent exception swallowing
- **Min passing score**: 0.8

### Task 4: Session and Auth Hardening (Medium)
**Focus**: JWT/session lifecycle vulnerabilities
- Missing token claim validation
- Refresh flow type confusion
- Incomplete logout invalidation
- Weak refresh token storage semantics
- **Min passing score**: 0.76

### Task 5: Async Pipeline Security (Hard)
**Focus**: Webhook authenticity + idempotency under concurrency
- Replay-prone signature validation
- Non-atomic dedupe in worker
- Transaction safety in ledger updates
- Cross-account update risk
- **Min passing score**: 0.8

### Task 6: Secure Data Export (Hard)
**Focus**: Multi-step PII export and access control
- Cross-tenant export creation
- Unscoped PII query exposure
- Unsafe temporary file handling
- Download redirect without ownership checks
- **Min passing score**: 0.82

---

## 🏆 Grading System

### Deterministic Matching Algorithm

Comments are matched to ground truth issues using:
1. **File path match** (required)
2. **Line proximity** (±2 lines tolerance)
3. **Category alignment** (exact or overlapping)
4. **Keyword overlap** (issue keywords in comment)

### Metrics

- **Precision**: TP / (TP + FP) - penalizes false positives
- **Recall**: TP / (TP + FN) - penalizes missed issues
- **Coverage**: % of files with issues that were reviewed
- **Severity Alignment**:
  - Exact match: 1.0
  - Off by one level: 0.5
  - Off by 2+ levels: 0.0

### Weighted Score

```python
score = (
    0.3 * precision +
    0.5 * recall +
    0.2 * severity_alignment +
    0.0 * coverage  # Task-dependent
)
```

### Decision Penalties

- Approve with critical issues: **-20%**
- Request changes with no critical issues: **-5%**

---

## 💰 Reward Function

The environment is now multi-step (`max_steps=5`) with dense trajectory rewards:

- **Intermediate reward** (`done=false`): positive signal for newly discovered true positives and coverage gains, penalties for duplicate/spam comments.
- **Terminal reward** (`done=true`): weighted grader score + quality shaping (early critical detection, actionability, coverage, false-positive penalties).

This provides a meaningful learning signal throughout the episode, not only at the terminal step.

Reward decomposition (current implementation):
```python
# Intermediate step reward (bounded to [-0.1, 0.35])
intermediate_reward = (
    0.12 * new_true_positives
    + 0.08 * coverage_gain
    + 0.05 * severity_alignment_gain
    + 0.03 * newly_reviewed_files
    - 0.06 * duplicate_comments
    - over_comment_penalty
)

# Terminal reward (bounded to [-0.1, 1.1])
terminal_reward = (
    grader_score
    + early_detection_bonus
    + actionability_bonus
    + coverage_bonus
    - false_positive_penalty
)
```

---

## 🚀 Setup & Usage

### Local Installation

```bash
# Clone repository
git clone https://github.com/yourusername/pr-review-env
cd pr-review-env

# Install dependencies
pip install -e .

# Run server
python -m uvicorn pr_review_env.server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
#Build image
docker build -t pr-review-env .

# Run container
docker run -p 8000:8000 pr-review-env
```

### Using the Environment

```python
import requests

# Reset to start new episode
response = requests.post("http://localhost:8000/reset", json={
    "task_id": "task1_security_basic"  # or None for random
})
observation = response.json()

# Submit review action (can be partial in early steps)
action = {
    "inline_comments": [
        {
            "file_path": "app/auth.py",
            "line_number": 11,
            "comment": "SQL Injection vulnerability",
            "severity": "critical",
            "category": "security",
            "suggested_fix": "Use parameterized queries"
        }
    ],
    "general_comments": [],
    "decision": {
        "decision": "request_changes",
        "summary": "Critical security issues found"
    },
    "submit": True
}

response = requests.post("http://localhost:8000/step", json={"action": action})
result = response.json()

print(f"Score: {result['feedback']['score']:.2f}")
print(f"Passed: {result['metadata']['passed']}")
```

---

## 📈 Baseline Results

### Deterministic Baseline (Latest)

```bash
# Set required variables (OpenAI-compatible client via HF router)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="openai/gpt-oss-120b"
export HF_TOKEN="your-hf-token"

# Run inference baseline against a running env server
python3 inference.py \
  --env-url http://localhost:8000 \
  --output inference_results.json
```

**Results (3-task sample before expansion)**:
| Task | Score | Passed | Precision | Recall |
|------|-------|--------|-----------|--------|
| Task 1 (Easy) | 0.85 | ✓ | 0.60 | 1.00 |
| Task 2 (Medium) | 0.58 | ✗ | 0.57 | 0.57 |
| Task 3 (Hard) | 0.77 | ✗ | 0.83 | 0.71 |
| **Average** | **0.73** | - | - | - |

For full 6-task runs, use the same command (defaults include Tasks 1-6) and regenerate the artifact.
Canonical artifact served by `/baseline`: `inference_results.json`

---

## 🔌 API Endpoints

### Core OpenEnv Endpoints

- `POST /reset` - Reset environment to new episode
- `POST /step` - Execute action and get observation
- `GET /state` - Get current episode state
- `GET /health` - Health check
- `GET /schema` - Get action/observation schemas
- `GET /metadata` - Get environment metadata

### Hackathon-Required Endpoints

- `GET /baseline` - Returns cached baseline results; use `?refresh=true` to trigger live baseline run
- `POST /grader` - Standalone deterministic grading endpoint with request body:
  - `{ "task_id": "...", "action": { ... } }`
- `GET /tasks` - Lists tasks and returns machine-readable `Action` JSON schema

---

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Check OpenEnv compliance
openenv validate

# Run full pre-submission checklist
./scripts/pre_submission_check.sh
```

---

## 🌐 Deployment to Hugging Face Spaces

This environment is deployed as a Docker-based Hugging Face Space.

### Automatic Deployment

The repository is configured for automatic deployment. Simply push to the `main` branch.

### Manual Deployment

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create Space
huggingface-cli repo create --type space --space_sdk docker pr-review-env

# Push code
git remote add hf https://huggingface.co/spaces/your-username/pr-review-env
git push hf main
```

Access at: `https://your-username-pr-review-env.hf.space`

---

## 📚 Project Structure

```
pr-review-env/
├── pr_review_env/
│   ├── __init__.py
│   ├── models.py                 # Typed Action/Observation models
│   └── server/
│       ├── app.py                # FastAPI server
│       ├── pr_review_environment.py  # Core environment logic
│       └── grader.py             # Deterministic grading algorithm
├── server/
│   └── app.py                    # Compatibility entrypoint for validators
├── tasks/
│   ├── task1_security_basic.json
│   ├── task2_quality_logic.json
│   └── task3_advanced_review.json
├── baseline/
│   └── baseline_inference.py     # OpenAI-compatible baseline
├── tests/
│   ├── test_environment.py
│   ├── test_api.py
│   └── test_grader_determinism.py
├── openenv.yaml                  # OpenEnv specification
├── pyproject.toml                # Dependencies
├── uv.lock                       # Resolver lock file for validation
├── Dockerfile                    # Container definition
└── README.md
```

---

## 🎯 Real-World Applications

This environment trains agents for:

1. **Automated Security Scanning** - Deploy as security bots in CI/CD
2. **Review Augmentation** - Suggest issues to human reviewers
3. **Custom Bots** - Train on company-specific patterns
4. **Developer Education** - Teach juniors what to look for

Known limitations:
- Diff-centric simulation (not full repository execution context).
- Keyword-driven deterministic grading is robust but not semantic-equivalence complete.
- Current tasks focus on Python PRs; multilingual task packs are future work.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional tasks (more languages, domains)
- More sophisticated grading (semantic similarity)
- Multi-turn review conversations
- Integration with real GitHub PRs

---

## 📄 License

MIT License - see LICENSE file

---

## 🙏 Acknowledgments

- Built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework
- Security patterns from [OWASP](https://owasp.org/) and [CWE](https://cwe.mitre.org/)
- Inspired by real code review processes at Meta, Google, and other tech companies

---

## 📞 Contact

For questions or issues:
- Open an issue on GitHub
- Contact: [Your contact info]

---

**🚀 Ready to train an AI code reviewer? Get started above!**
