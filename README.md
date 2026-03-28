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

Agents submit structured reviews with three components:

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
- **Expected baseline (GPT-4)**: 0.63 ✗

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

Multi-signal reward with partial progress:

```python
reward = (
    base_score              # Grader score (0.0-1.0)
    + early_detection       # +0.05 for finding critical issues early
    + actionability         # +0.03 for providing suggested fixes
    + coverage              # +0.02 for thorough review
    - false_positive_penalty # -0.1 for excessive false positives
)
# Bounded: [-0.1, 1.1]
```

This encourages:
- **Accuracy** (primary signal)
- **Actionable feedback** (suggested fixes bonus)
- **Thoroughness** (coverage bonus)
- **Precision** (false positive penalty)

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

# Submit review action
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
    }
}

response = requests.post("http://localhost:8000/step", json={"action": action})
result = response.json()

print(f"Score: {result['feedback']['score']:.2f}")
print(f"Passed: {result['metadata']['passed']}")
```

---

## 📈 Baseline Results

### OpenAI GPT-4 Turbo Baseline

```bash
# Set API key
export OPENAI_API_KEY="your-key-here"

# Run baseline
python baseline/baseline_inference.py
```

**Results**:
| Task | Score | Passed | Precision | Recall |
|------|-------|--------|-----------|--------|
| Task 1 (Easy) | 0.82 | ✓ | 0.88 | 1.00 |
| Task 2 (Medium) | 0.71 | ✗ | 0.75 | 0.71 |
| Task 3 (Hard) | 0.63 | ✗ | 0.70 | 0.57 |
| **Average** | **0.72** | - | - | - |

The hard task genuinely challenges frontier models!

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

- `GET /baseline` - Pre-computed baseline scores
- `POST /grader` - Standalone grading endpoint
- `GET /tasks` - List all tasks with action schema

---

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Check OpenEnv compliance
openenv validate
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
│   ├── models.py                 # Pydantic models (Action, Observation)
│   ├── client.py                 # Client for testing
│   └── server/
│       ├── app.py                # FastAPI server
│       ├── pr_review_environment.py  # Core environment logic
│       ├── grader.py             # Grading algorithm
│       └── scenarios.py          # PR generation
├── tasks/
│   ├── task1_security_basic.json
│   ├── task2_quality_logic.json
│   └── task3_advanced_review.json
├── baseline/
│   └── baseline_inference.py     # OpenAI baseline
├── tests/
│   ├── test_environment.py
│   └── test_grader.py
├── openenv.yaml                  # OpenEnv specification
├── pyproject.toml                # Dependencies
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
