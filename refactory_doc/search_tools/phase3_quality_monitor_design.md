# Phase 3: 搜索质量旁路监控 (LLM-as-Judge) 设计

> **模块**: `search_service/quality_monitor.py` (新建)  
> **依赖**: LLM API (DeepSeek/GPT)  
> **日期**: 2026-05-05

---

## 1. 设计目标

建立搜索结果质量的自动化监控:
- 使用 LLM 评判搜索结果与 query 的相关性
- 检测搜索退化 (如 CAPTCHA 导致的低质量结果)
- 生成质量报告和告警

---

## 2. 架构

```
搜索请求流 (正常路径)
    │
    SearchRouter.search()
    │    │
    │    └─ 返回 SearchResponse
    │
    ▼ (旁路: 异步采样)
QualityMonitor
    │
    ├─ 采样率: 10% (可配)
    ├─ 异步评估: 不阻塞主流程
    ├─ LLM 评分: relevance 0-10
    │
    ├─ 输出: Prometheus metric
    │         search_quality_score (histogram)
    │
    └─ 告警: 评分 < 阈值时 warning 日志
```

---

## 3. 核心类

### 3.1 QualityMonitor

```python
class QualityMonitor:
    """搜索质量旁路监控 — LLM-as-Judge。

    功能:
    - 按采样率随机选取搜索结果进行质量评估
    - 使用 LLM 评判 query-result 相关性 (0-10)
    - 异步执行，不阻塞主搜索路径
    - 输出 Prometheus metrics + 结构化日志

    Args:
        model: LLM model name for evaluation.
        sample_rate: Fraction of queries to evaluate (0.0-1.0).
        min_acceptable_score: Scores below this trigger warning.
    """

    EVALUATION_PROMPT: str = """You are a search quality evaluator.

Given a search query and a list of search results, rate the overall
relevance on a scale of 0-10.

Scoring guide:
- 9-10: All results highly relevant, comprehensive coverage
- 7-8: Most results relevant, good coverage
- 5-6: Mixed — some relevant, some noise
- 3-4: Mostly irrelevant results
- 0-2: Completely irrelevant or empty results

Query: {query}

Results:
{results_text}

Respond with ONLY a JSON object:
{{"score": <0-10>, "reason": "<brief explanation>"}}
"""

    def __init__(
        self,
        model: str = "deepseek-chat",
        sample_rate: float = 0.1,
        min_acceptable_score: float = 5.0,
    ) -> None: ...

    async def maybe_evaluate(
        self,
        query: str,
        response: SearchResponse,
        tool_name: str,
    ) -> None:
        """按采样率决定是否评估。异步执行。"""
        if random.random() > self._sample_rate:
            return
        # Fire-and-forget: 不阻塞主流程
        asyncio.create_task(
            self._evaluate(query, response, tool_name),
        )

    async def _evaluate(
        self,
        query: str,
        response: SearchResponse,
        tool_name: str,
    ) -> None:
        """执行 LLM 质量评估。"""
        results_text = self._format_results(response)
        prompt = self.EVALUATION_PROMPT.format(
            query=query, results_text=results_text,
        )

        try:
            score, reason = await self._call_llm(prompt)

            # Prometheus metric
            self._metrics.search_quality_score.labels(
                tool=tool_name,
            ).observe(score)

            # 告警
            if score < self._min_acceptable_score:
                self._logger.warning(
                    "search_quality_degraded",
                    extra={
                        "query": query,
                        "tool": tool_name,
                        "score": score,
                        "reason": reason,
                        "result_count": response.result_count,
                    },
                )
            else:
                self._logger.info(
                    "search_quality_ok",
                    extra={"query": query, "tool": tool_name, "score": score},
                )
        except Exception as exc:
            self._logger.debug(
                "quality_evaluation_failed",
                extra={"error": str(exc)},
            )
```

### 3.2 集成点: SearchRouter

```python
class SearchRouter:
    def __init__(self, ..., quality_monitor: Optional[QualityMonitor] = None):
        self._quality_monitor = quality_monitor

    async def search(self, query: str, **kwargs) -> SearchResponse:
        # ... existing search logic ...
        result = await self._backends[name].search(query, **kwargs)

        # 旁路质量监控 (异步, 不阻塞)
        if self._quality_monitor:
            await self._quality_monitor.maybe_evaluate(query, result, "web_search")

        return result
```

---

## 4. 质量评分指标

```python
from prometheus_client import Histogram

search_quality_score = Histogram(
    "search_quality_score",
    "LLM-judged search result quality (0-10)",
    labelnames=["tool"],
    buckets=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
)
```

Grafana 告警规则:
```
WHEN avg(search_quality_score{tool="web_search"}) < 5 FOR 10min
THEN send alert
```

---

## 5. 成本控制

| 配置 | 默认值 | 说明 |
|:---|:---|:---|
| `sample_rate` | 0.1 (10%) | 只评估 10% 的查询 |
| `model` | deepseek-chat | 最便宜的评估模型 |
| `max_results_to_judge` | 5 | 只取前 5 条结果评估 |
| 每次评估 tokens | ~200 | 约 ¥0.001/次 |
| 日均成本 (100 查询) | ~¥0.01 | 极低成本 |

---

## 6. 行业价值分析与主流实现方式

### 6.1 功能实际价值评估

| 维度 | 评估 | 说明 |
|:---|:---|:---|
| **质量保障** | ⭐⭐⭐⭐⭐ | 搜索质量退化（如 CAPTCHA 导致空结果）是无声故障，LLM 监控是唯一自动化检测手段 |
| **前瞻性** | ⭐⭐⭐⭐⭐ | LLM-as-Judge 是 2025-2026 年搜索/RAG 质量评估的行业标准方法 |
| **成本效益** | ⭐⭐⭐⭐⭐ | 10% 采样率 + DeepSeek 模型，日均成本 ~¥0.01，极高 ROI |

### 6.2 业界主流实现方式

**LLM-as-Judge 已成为搜索和 RAG 系统质量监控的行业标准**。核心实践：

1. **评估模式**: Pointwise（单条评分，我们采用）、Pairwise（A/B 对比）、Listwise（整体排序评估）。行业标准是生产监控用 Pointwise，模型迭代用 Pairwise
2. **Rubric 评分**: 行业最佳实践是使用明确的评分标准（rubric），而非模糊指令。我们的 0-10 评分指南设计合理
3. **模型选择**: 不一定需要最贵的模型。行业趋势是用 GPT-4o-mini 或 Gemini Flash 做高频监控，复杂评估保留高端模型
4. **人工校准**: LLM judge 可能有偏见。行业实践是定期抽样与人工标注对比，校准评估准确性
5. **多维评估**: RAG 系统标准评估维度——Faithfulness（是否基于检索内容）、Context Relevance（检索是否准确）、Answer Relevancy（回答是否相关）
6. **框架工具**: 行业常用 Ragas、DeepEval、Agenta、Langfuse、Microsoft RELEVANCE 等框架

> [!TIP]
> 我们的设计（旁路采样、async fire-and-forget、Prometheus 指标输出）与行业最佳实践高度一致。建议补充：1) 人工校准机制 2) 考虑多维评估（相关性+时效性）。

---

## 7. 待确认事项

1. **采样率**: 10% 是否合适？开发阶段可以提高到 50%。
2. **评估模型**: 使用 `deepseek-chat` 还是更便宜的模型？
3. **评估维度**: 是否需要多维评估（相关性 + 时效性 + 多样性），还是单一 relevance 分数够用？
4. **告警渠道**: 质量降级时仅 warning 日志，还是需要企业微信通知？
