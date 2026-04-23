# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_qa.json
- Mode: openai
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.96 | 1.0 | 0.04 |
| Avg attempts | 1 | 1.03 | 0.03 |
| Avg token estimate | 0 | 0 | 0 |
| Avg latency (ms) | 0 | 0 | 0 |

## Failure modes
```json
{
  "react": {
    "none": 96,
    "wrong_final_answer": 4
  },
  "reflexion": {
    "none": 100
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. In a real report, students should explain when the reflection memory was useful, which failure modes remained, and whether evaluator quality limited gains.
