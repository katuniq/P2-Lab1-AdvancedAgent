from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .mock_runtime import (
    FAILURE_MODE_BY_QID,
    EvaluatorRuntimeResult,
    ReflectorRuntimeResult,
    RuntimeResult,
    actor_answer,
    evaluator,
    reflector,
)
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, self.max_attempts + 1):
            actor_result = self._coerce_actor_result(actor_answer(example, attempt_id, self.agent_type, reflection_memory))
            judge_result = self._coerce_judge_result(evaluator(example, actor_result.content))
            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=actor_result.content,
                score=judge_result.judge.score,
                reason=judge_result.judge.reason,
                token_estimate=actor_result.token_count + judge_result.token_count,
                latency_ms=actor_result.latency_ms + judge_result.latency_ms,
            )
            final_answer = actor_result.content
            final_score = judge_result.judge.score
            if judge_result.judge.score == 1:
                traces.append(trace)
                break

            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflector_result = self._coerce_reflection_result(reflector(example, attempt_id, judge_result.judge))
                trace.reflection = reflector_result.reflection
                reflections.append(reflector_result.reflection)
                reflection_memory.append(
                    f"{reflector_result.reflection.lesson}\nStrategy: {reflector_result.reflection.next_strategy}"
                )
                trace.token_estimate += reflector_result.token_count
                trace.latency_ms += reflector_result.latency_ms
            traces.append(trace)
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

    @staticmethod
    def _coerce_actor_result(result: str | RuntimeResult) -> RuntimeResult:
        if isinstance(result, RuntimeResult):
            return result
        return RuntimeResult(content=result)

    @staticmethod
    def _coerce_judge_result(result) -> EvaluatorRuntimeResult:
        if isinstance(result, EvaluatorRuntimeResult):
            return result
        return EvaluatorRuntimeResult(judge=result)

    @staticmethod
    def _coerce_reflection_result(result) -> ReflectorRuntimeResult:
        if isinstance(result, ReflectorRuntimeResult):
            return result
        return ReflectorRuntimeResult(reflection=result)

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
