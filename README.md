# Email Triage RL Environment

OpenEnv-compliant, backend-only Python environment that models a real email triage workflow for AI agents.

## Why this project matters

In real operations (IT support, finance, security, incident response), teams receive mixed-quality inbox traffic. Missing a critical message is costly, while over-escalating noise slows response time. This environment evaluates an agent's ability to classify incoming email into practical operational labels:

- `spam`
- `important`
- `urgent`
- `normal`

The setup is intentionally lightweight and deterministic, so judges can run and inspect behavior quickly.

## Environment design

### State
- Current email text (subject + body) as plain text.

### Action
- One classification label from the discrete action space:
  - `spam`, `important`, `urgent`, `normal`

### Reward (nuanced + grader-aligned)
- Exact classification: `+1.0`
- Partial correctness: meaningful positive reward based on overlap, e.g.:
  - `0.5` for close operational overlap (`important` <-> `urgent`)
  - `0.3` for weaker overlap (`important` <-> `spam`)
  - `0.2` for mild overlap (`normal` <-> `important` / `urgent`)
- Fully wrong classification: `-1.0`

### Episode flow
- One-step episode (single classification).
- Episode ends immediately after `step(action)`.

### OpenEnv-style API
- `reset()` -> initial observation
- `step(action)` -> `(observation, reward, done, info)`
- `state()` -> serializable state dictionary

## Task design (increasing difficulty)

Defined in `tasks/tasks.py`:

1. **Easy - clear spam**
   - Obvious phishing language (free prize + bank details request).
   - Expected label: `spam`.

2. **Medium - slightly confusing business context**
   - Contains "incident" wording but primary action is invoice approval.
   - Expected label: `important`.

3. **Hard - tricky, realistic BEC-style phishing**
   - Looks like a finance escalation with urgency cues.
   - Expected label: `spam`.

Each task includes:
- input email text,
- expected output label,
- grader function returning score in `[0.0, 1.0]`.

## Project structure

- `env/email_env.py` - environment implementation (`reset`, `step`, `state`)
- `tasks/tasks.py` - task dataset + grading logic
- `inference.py` - baseline inference runner across all tasks
- `openenv.yaml` - OpenEnv metadata/config
- `Dockerfile` - container runtime definition
- `requirements.txt` - dependency manifest (standard library only)

## How to run

### Local

```bash
python inference.py
```

Optional environment variable (not required for baseline):

```bash
# Linux/macOS
export OPENENV_API_KEY="your_key"

# Windows PowerShell
$env:OPENENV_API_KEY="your_key"
```

### Docker

```bash
docker build -t email-triage-env .
docker run --rm email-triage-env
```

## Example output (`inference.py`)

```text
=== Email Triage RL Inference ===

Task: task_easy_clear_spam (easy)
Predicted: spam
Expected: spam
Grader score: 1.00
Environment reward: 1.00
Done: True
Info: {'grade': 'correct', 'score': 1.0, 'expected_label': 'spam', 'predicted_label': 'spam'}

Task: task_medium_ops_followup (medium)
Predicted: urgent
Expected: important
Grader score: 0.50
Environment reward: 0.50
Done: True
Info: {'grade': 'partially_correct', 'score': 0.5, 'expected_label': 'important', 'predicted_label': 'urgent'}

Task: task_hard_bec_finance_phish (hard)
Predicted: important
Expected: spam
Grader score: 0.30
Environment reward: 0.30
Done: True
Info: {'grade': 'partially_correct', 'score': 0.3, 'expected_label': 'spam', 'predicted_label': 'important'}

=== Summary ===
Tasks run: 3
Average grader score: 0.60
Average environment reward: 0.60
```

## Judge-facing highlights

- Real-world scenario (email triage for operations/security workflows)
- OpenEnv API compliance (`reset`, `step`, `state`)
- Clear difficulty progression (easy -> medium -> hard)
- Meaningful reward shaping with explicit wrong-action penalty
- Fast, low-resource execution (CPU-friendly, no heavy libraries)

