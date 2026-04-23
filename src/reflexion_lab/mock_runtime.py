from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from urllib import error, request

from dotenv import load_dotenv

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

load_dotenv()

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    model_name: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 180
    max_retries: int = 2


@dataclass
class RuntimeResult:
    content: str
    token_count: int = 0
    latency_ms: int = 0


@dataclass
class EvaluatorRuntimeResult:
    judge: JudgeResult
    token_count: int = 0
    latency_ms: int = 0


@dataclass
class ReflectorRuntimeResult:
    reflection: ReflectionEntry
    token_count: int = 0
    latency_ms: int = 0


def runtime_config_from_env() -> OpenAIConfig | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = os.getenv("OPENAI_MODEL_NAME", "").strip()
    if not api_key or not model_name:
        return None
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    timeout_seconds = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "180").strip())
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2").strip())
    return OpenAIConfig(
        api_key=api_key,
        model_name=model_name,
        base_url=base_url.rstrip("/"),
        timeout_seconds=max(1, timeout_seconds),
        max_retries=max(0, max_retries),
    )


def parse_json_object(text: str) -> dict:
    fenced = text.strip()
    if "```json" in fenced:
        fenced = fenced.split("```json", 1)[1]
        fenced = fenced.split("```", 1)[0]
    elif "```" in fenced:
        fenced = fenced.split("```", 1)[1]
        fenced = fenced.split("```", 1)[0]
    fenced = fenced.strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def parse_openai_response(payload: dict, latency_ms: int) -> RuntimeResult:
    choices = payload.get("choices", [])
    if not choices:
        raise ValueError("OpenAI response missing choices")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    usage = payload.get("usage", {})
    token_count = int(usage.get("total_tokens") or 0)
    return RuntimeResult(content=content, token_count=token_count, latency_ms=latency_ms)


def _openai_request_body(model_name: str, messages: list[dict[str, str]], temperature: float) -> bytes:
    body = {
        "model": model_name,
        "messages": messages,
    }
    return json.dumps(body).encode("utf-8")


def _generate_content(messages: list[dict[str, str]], temperature: float = 0.0) -> RuntimeResult | None:
    config = runtime_config_from_env()
    if config is None:
        return None
    body = _openai_request_body(config.model_name, messages, temperature)
    endpoint = f"{config.base_url}/chat/completions"
    req = request.Request(
        url=endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
    )
    last_error: Exception | None = None
    for attempt in range(config.max_retries + 1):
        start = time.perf_counter()
        try:
            with request.urlopen(req, timeout=config.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            latency_ms = int((time.perf_counter() - start) * 1000)
            return parse_openai_response(payload, latency_ms)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI request failed with status {exc.code}: {detail}") from exc
        except (TimeoutError, socket.timeout) as exc:
            last_error = exc
        except error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                last_error = exc
            else:
                raise RuntimeError(f"OpenAI request failed: {exc.reason}") from exc

        if attempt < config.max_retries:
            time.sleep(min(2 ** attempt, 4))

    raise RuntimeError(
        f"OpenAI request timed out after {config.max_retries + 1} attempt(s). "
        f"Increase OPENAI_TIMEOUT_SECONDS or lower the workload."
    ) from last_error


def _format_context(example: QAExample) -> str:
    return "\n\n".join(f"[{chunk.title}]\n{chunk.text}" for chunk in example.context)


def _mock_actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if example.qid not in FIRST_ATTEMPT_WRONG:
        return example.gold_answer
    if agent_type == "react":
        return FIRST_ATTEMPT_WRONG[example.qid]
    if attempt_id == 1 and not reflection_memory:
        return FIRST_ATTEMPT_WRONG[example.qid]
    return example.gold_answer


def _mock_evaluator(example: QAExample, answer: str) -> JudgeResult:
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
    if normalize_answer(answer) == "london":
        return JudgeResult(
            score=0,
            reason="The answer stopped at the birthplace city and never completed the second hop to the river.",
            missing_evidence=["Need to identify the river that flows through London."],
            spurious_claims=[],
        )
    return JudgeResult(
        score=0,
        reason="The final answer selected the wrong second-hop entity.",
        missing_evidence=["Need to ground the answer in the second paragraph."],
        spurious_claims=[answer],
    )


def _mock_reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    strategy = (
        "Do the second hop explicitly: birthplace city -> river through that city."
        if example.qid == "hp2"
        else "Verify the final entity against the second paragraph before answering."
    )
    return ReflectionEntry(
        attempt_id=attempt_id,
        failure_reason=judge.reason,
        lesson="A partial first-hop answer is not enough; the final answer must complete all hops.",
        next_strategy=strategy,
    )


def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    return actor_answer_with_metrics(example, attempt_id, agent_type, reflection_memory).content


def evaluator(example: QAExample, answer: str) -> JudgeResult:
    return evaluator_with_metrics(example, answer).judge


def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    return reflector_with_metrics(example, attempt_id, judge).reflection


def actor_answer_with_metrics(
    example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]
) -> RuntimeResult:
    memory_block = "\n".join(f"- {item}" for item in reflection_memory) if reflection_memory else "- none"
    user_prompt = (
        f"Question:\n{example.question}\n\n"
        f"Context:\n{_format_context(example)}\n\n"
        f"Attempt: {attempt_id}\n"
        f"Agent type: {agent_type}\n"
        f"Reflection memory:\n{memory_block}"
    )
    runtime = _generate_content(
        [{"role": "system", "content": ACTOR_SYSTEM}, {"role": "user", "content": user_prompt}],
        temperature=0.2,
    )
    if runtime is not None:
        return runtime
    return RuntimeResult(content=_mock_actor_answer(example, attempt_id, agent_type, reflection_memory))


def evaluator_with_metrics(example: QAExample, answer: str) -> EvaluatorRuntimeResult:
    if runtime_config_from_env() is None:
        return EvaluatorRuntimeResult(judge=_mock_evaluator(example, answer))
    user_prompt = (
        f"Question:\n{example.question}\n\n"
        f"Gold answer:\n{example.gold_answer}\n"
        f"Predicted answer:\n{answer}\n\n"
        f"Normalized gold: {normalize_answer(example.gold_answer)}\n"
        f"Normalized prediction: {normalize_answer(answer)}\n"
    )
    runtime = _generate_content(
        [{"role": "system", "content": EVALUATOR_SYSTEM}, {"role": "user", "content": user_prompt}],
        temperature=0.0,
    )
    if runtime is None:
        return EvaluatorRuntimeResult(judge=_mock_evaluator(example, answer))
    judge = JudgeResult.model_validate(parse_json_object(runtime.content))
    return EvaluatorRuntimeResult(judge=judge, token_count=runtime.token_count, latency_ms=runtime.latency_ms)


def reflector_with_metrics(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectorRuntimeResult:
    if runtime_config_from_env() is None:
        return ReflectorRuntimeResult(reflection=_mock_reflector(example, attempt_id, judge))
    user_prompt = (
        f"Question:\n{example.question}\n\n"
        f"Context:\n{_format_context(example)}\n\n"
        f"Attempt id: {attempt_id}\n"
        f"Evaluator result:\n{judge.model_dump_json(indent=2)}"
    )
    runtime = _generate_content(
        [{"role": "system", "content": REFLECTOR_SYSTEM}, {"role": "user", "content": user_prompt}],
        temperature=0.2,
    )
    if runtime is None:
        return ReflectorRuntimeResult(reflection=_mock_reflector(example, attempt_id, judge))
    payload = parse_json_object(runtime.content)
    payload.setdefault("attempt_id", attempt_id)
    reflection = ReflectionEntry.model_validate(payload)
    return ReflectorRuntimeResult(
        reflection=reflection,
        token_count=runtime.token_count,
        latency_ms=runtime.latency_ms,
    )
