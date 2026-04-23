from src.reflexion_lab.agents import BaseAgent
from src.reflexion_lab.schemas import JudgeResult, QAExample, ReflectionEntry


def make_example() -> QAExample:
    return QAExample.model_validate(
        {
            "qid": "hp-test",
            "difficulty": "medium",
            "question": "Which river flows through the birthplace city?",
            "gold_answer": "Thames",
            "context": [{"title": "Doc", "text": "Context text"}],
        }
    )


def test_reflexion_agent_records_reflection_before_retry(monkeypatch):
    attempts = []

    def fake_actor_answer(example, attempt_id, agent_type, reflection_memory):
        attempts.append((attempt_id, list(reflection_memory)))
        return "London" if attempt_id == 1 else example.gold_answer

    def fake_evaluator(example, answer):
        if answer == example.gold_answer:
            return JudgeResult(score=1, reason="correct")
        return JudgeResult(
            score=0,
            reason="need second hop",
            missing_evidence=["river through city"],
            spurious_claims=[answer],
        )

    def fake_reflector(example, attempt_id, judge):
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Need the second hop",
            next_strategy="Look up the river through the city",
        )

    monkeypatch.setattr("src.reflexion_lab.agents.actor_answer", fake_actor_answer)
    monkeypatch.setattr("src.reflexion_lab.agents.evaluator", fake_evaluator)
    monkeypatch.setattr("src.reflexion_lab.agents.reflector", fake_reflector)

    agent = BaseAgent(agent_type="reflexion", max_attempts=2)
    record = agent.run(make_example())

    assert record.is_correct is True
    assert record.attempts == 2
    assert len(record.reflections) == 1
    assert record.traces[0].reflection is not None
    assert attempts == [(1, []), (2, ["Need the second hop\nStrategy: Look up the river through the city"])]


def test_reflexion_agent_aggregates_real_metrics(monkeypatch):
    def fake_actor_answer(example, attempt_id, agent_type, reflection_memory):
        return example.gold_answer

    def fake_evaluator(example, answer):
        return JudgeResult(score=1, reason="correct")

    monkeypatch.setattr("src.reflexion_lab.agents.actor_answer", fake_actor_answer)
    monkeypatch.setattr("src.reflexion_lab.agents.evaluator", fake_evaluator)

    agent = BaseAgent(agent_type="react", max_attempts=1)
    record = agent.run(make_example())

    assert record.token_estimate >= 0
    assert record.latency_ms >= 0
    assert record.attempts == len(record.traces) == 1
