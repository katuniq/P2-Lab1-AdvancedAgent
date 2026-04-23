# OpenRouter Reflexion Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the scaffold's mock-only TODO path with a real OpenRouter-backed Reflexion runtime that reads `.env` configuration, records real token/latency metrics, and preserves the benchmark report format required by the lab.

**Architecture:** Keep the current `agents.py` flow as the orchestration layer, but introduce a provider-aware runtime module that can either use the existing mock behavior or call OpenRouter's chat completions API. Use structured JSON parsing for evaluator and reflector outputs so report fields remain stable and the Reflexion loop can reliably update memory between attempts.

**Tech Stack:** Python, Pydantic, Typer, `python-dotenv`, OpenRouter chat completions over `urllib`

---

### Task 1: Add failing tests for schemas and Reflexion control flow

**Files:**
- Create: `C:/Users/Kat/Documents/AICB_lab/phase1-track3-lab1-advanced-agent/tests/test_agents.py`
- Modify: `C:/Users/Kat/Documents/AICB_lab/phase1-track3-lab1-advanced-agent/tests/test_utils.py`
- Test: `C:/Users/Kat/Documents/AICB_lab/phase1-track3-lab1-advanced-agent/tests/test_agents.py`

- [ ] **Step 1: Write the failing test**

```python
from src.reflexion_lab.agents import ReflexionAgent
from src.reflexion_lab.schemas import JudgeResult, QAExample, ReflectionEntry


def test_reflexion_agent_records_reflection_before_retry(monkeypatch):
    example = QAExample.model_validate(
        {
            "qid": "q1",
            "difficulty": "medium",
            "question": "Where is the river?",
            "gold_answer": "Thames",
            "context": [{"title": "A", "text": "x"}],
        }
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agents.py -v`
Expected: FAIL because the current schemas and agent loop do not yet provide the required fields/behavior.

- [ ] **Step 3: Write minimal implementation**

```python
class JudgeResult(BaseModel):
    score: int
    reason: str
    missing_evidence: list[str] = Field(default_factory=list)
    spurious_claims: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agents.py -v`
Expected: PASS for the new behavior-focused test.

- [ ] **Step 5: Commit**

```bash
git add tests/test_agents.py tests/test_utils.py src/reflexion_lab/schemas.py src/reflexion_lab/agents.py
git commit -m "test: cover reflexion loop contracts"
```

### Task 2: Add failing tests for OpenRouter configuration and response parsing

**Files:**
- Create: `C:/Users/Kat/Documents/AICB_lab/phase1-track3-lab1-advanced-agent/tests/test_runtime.py`
- Modify: `C:/Users/Kat/Documents/AICB_lab/phase1-track3-lab1-advanced-agent/src/reflexion_lab/mock_runtime.py`
- Test: `C:/Users/Kat/Documents/AICB_lab/phase1-track3-lab1-advanced-agent/tests/test_runtime.py`

- [ ] **Step 1: Write the failing test**

```python
def test_parse_openrouter_usage_counts_tokens():
    payload = {
        "choices": [{"message": {"content": "hello"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runtime.py -v`
Expected: FAIL because no real provider parsing helpers exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def parse_chat_response(payload: dict) -> RuntimeResult:
    usage = payload.get("usage", {})
    return RuntimeResult(
        content=payload["choices"][0]["message"]["content"],
        token_count=int(usage.get("total_tokens") or 0),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_runtime.py -v`
Expected: PASS with correct token extraction and content parsing.

- [ ] **Step 5: Commit**

```bash
git add tests/test_runtime.py src/reflexion_lab/mock_runtime.py
git commit -m "feat: add openrouter runtime parsing"
```

### Task 3: Implement OpenRouter-backed actor, evaluator, and reflector runtime

**Files:**
- Modify: `C:/Users/Kat/Documents/AICB_lab/phase1-track3-lab1-advanced-agent/src/reflexion_lab/mock_runtime.py`
- Modify: `C:/Users/Kat/Documents/AICB_lab/phase1-track3-lab1-advanced-agent/src/reflexion_lab/prompts.py`
- Modify: `C:/Users/Kat/Documents/AICB_lab/phase1-track3-lab1-advanced-agent/src/reflexion_lab/schemas.py`
- Test: `C:/Users/Kat/Documents/AICB_lab/phase1-track3-lab1-advanced-agent/tests/test_runtime.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_runtime_from_env_reads_openrouter_settings(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runtime.py::test_build_runtime_from_env_reads_openrouter_settings -v`
Expected: FAIL because the runtime is still mock-only.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass
class OpenRouterConfig:
    api_key: str
    model: str
    base_url: str = "https://openrouter.ai/api/v1"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_runtime.py -v`
Expected: PASS with provider config, JSON parsing, and prompt wiring in place.

- [ ] **Step 5: Commit**

```bash
git add src/reflexion_lab/mock_runtime.py src/reflexion_lab/prompts.py src/reflexion_lab/schemas.py tests/test_runtime.py
git commit -m "feat: connect reflexion scaffold to openrouter"
```

### Task 4: Wire benchmark mode selection and report-safe metrics

**Files:**
- Modify: `C:/Users/Kat/Documents/AICB_lab/phase1-track3-lab1-advanced-agent/src/reflexion_lab/agents.py`
- Modify: `C:/Users/Kat/Documents/AICB_lab/phase1-track3-lab1-advanced-agent/run_benchmark.py`
- Test: `C:/Users/Kat/Documents/AICB_lab/phase1-track3-lab1-advanced-agent/tests/test_agents.py`

- [ ] **Step 1: Write the failing test**

```python
def test_reflexion_agent_aggregates_real_metrics():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agents.py::test_reflexion_agent_aggregates_real_metrics -v`
Expected: FAIL because the agent still uses hard-coded token and latency estimates.

- [ ] **Step 3: Write minimal implementation**

```python
trace = AttemptTrace(
    attempt_id=attempt_id,
    answer=answer,
    score=judge.score,
    reason=judge.reason,
    token_estimate=actor_result.token_count + evaluator_result.token_count + reflector_tokens,
    latency_ms=actor_result.latency_ms + evaluator_result.latency_ms + reflector_latency,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agents.py -v`
Expected: PASS with per-attempt metrics and reflections preserved.

- [ ] **Step 5: Commit**

```bash
git add src/reflexion_lab/agents.py run_benchmark.py tests/test_agents.py
git commit -m "feat: wire benchmark runtime selection"
```

### Task 5: Verify benchmark output and grader compatibility

**Files:**
- Modify: `C:/Users/Kat/Documents/AICB_lab/phase1-track3-lab1-advanced-agent/requirements.txt`
- Test: `C:/Users/Kat/Documents/AICB_lab/phase1-track3-lab1-advanced-agent/outputs/sample_run/report.json`

- [ ] **Step 1: Write the failing test**

```python
def test_report_payload_keys_remain_stable():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -v`
Expected: FAIL until the runtime changes still preserve the report contract.

- [ ] **Step 3: Write minimal implementation**

```python
report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -v`
Expected: PASS, followed by a sample benchmark run and `python autograde.py --report-path outputs/sample_run/report.json`.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt outputs/sample_run/report.json outputs/sample_run/report.md
git commit -m "chore: verify openrouter benchmark output"
```
