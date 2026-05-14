# Worker 输出设计 — Structured Findings with Provenance

> **状态**：设计阶段 | **父文档**：[citation_system_design.md](./citation_system_design.md) | **更新**：2026-05-02

---

## 一、设计目标

改造 Worker 输出格式：从自由文本叙述 → **claim-source-evidence 三元组**，在做深度总结的同时保留 fact-to-source 追溯链。

### 约束条件

1. Worker 返回值通过 `ToolMessage(content=str)` 传递给 Supervisor
2. `deepagents` 库的 `SubAgent` 支持 `response_format` 参数——当指定 Pydantic `BaseModel` 时，框架自动将 `structured_response` 序列化为 JSON 字符串作为 `ToolMessage.content`
3. Worker **应该做最大化的信息提取和过滤**，不是透传原始搜索结果
4. Worker 可能执行多轮搜索（多跳），输出应是所有搜索轮次的综合结果
5. 用户 query 和搜索结果可能是中文或英文，系统需要跨语言兼容

### 设计原则

| 原则 | 说明 |
|------|------|
| **Provenance 不丢失** | 每个事实性声明必须绑定到其来源 URL(s) |
| **Evidence 可验证** | 保留原始 source 中的关键引文片段 |
| **格式 100% 可解析** | 通过 Pydantic structured output 保证结构正确性 |
| **Token 高效** | 不输出原始网页全文，只输出提炼后的 findings |
| **降级优雅** | 即使 structured output 失败，纯文本输出仍有用 |
| **1:N Source 映射** | 一个 Finding 可绑定多个 Source URL |

---

## 二、输出格式设计

### 2.1 为什么选 Pydantic Structured Output

| 维度 | Pydantic `response_format` | Markdown 三元组 | 原始 JSON |
|------|---------------------------|-----------------|----------|
| 输出结构可靠性 | ✅ 框架级保证 | 🟡 依赖 LLM 遵守模板 | ❌ 括号/引号容易出错 |
| 下游解析 | ✅ `model_validate_json()` | 🟡 正则 + 降级 | ❌ 一个逗号错误 = 整体失败 |
| Supervisor 可读性 | ✅ LLM 可直接理解 JSON | ✅ LLM 直接理解 Markdown | 🟡 不如 Markdown 自然 |
| 降级表现 | ✅ 退回纯文本 | ✅ 内容仍有用 | ❌ 无法降级 |
| 1:N Source 映射 | ✅ `list[str]` 原生支持 | 🟡 需要扩展模板 | ✅ 支持 |

**决策**：使用 Pydantic `response_format` 作为主路径。

**`deepagents` 库支持确认**：`SubAgent` TypedDict 包含 `response_format: NotRequired[ResponseFormat[Any] | type | dict[str, Any]]` 字段。当 subagent 产出 `structured_response` 时，框架自动调用 `model_dump_json()` 序列化为 `ToolMessage.content`。

### 2.2 Pydantic Schema 定义

```python
from urllib.parse import urlparse, urlunparse

from pydantic import BaseModel, Field, field_validator
import re


class Finding(BaseModel):
    """A single claim-source-evidence triple preserving the provenance chain.

    Represents one factual finding extracted by a research Worker.
    """

    claim: str = Field(
        description=(
            "A specific factual statement extracted from research. "
            "Must not contain numbered citations like [1], [2]."
        ),
    )
    source_urls: list[str] = Field(
        description=(
            "URLs that directly support this claim. "
            "Each must be a valid HTTP/HTTPS URL. "
            "Use the most authoritative source(s) available. "
            "For paywalled content, use the accessible secondary source URL."
        ),
        min_length=1,
    )
    source_titles: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable titles for each source URL, in the same "
            "order as source_urls. Extract from the search result's "
            "title field. If a title is unavailable, use the URL's "
            "domain as fallback (e.g., 'langchain.com'). Length should "
            "match source_urls when populated."
        ),
    )
    evidence: str = Field(
        description=(
            "A brief quote or close paraphrase from the source(s) "
            "that directly supports the claim. Use quotation marks "
            'for direct quotes (e.g., "exact words from the source"). '
            "Keep concise but sufficient for downstream verification."
        ),
    )

    @field_validator("claim")
    @classmethod
    def claim_must_not_contain_numbered_citations(cls, v: str) -> str:
        if re.search(r"\[\d+\]", v):
            raise ValueError(
                "Claim must not contain numbered citations like [1], [2]. "
                "Citation numbering is handled downstream by CitationAgent."
            )
        return v

    @field_validator("source_urls")
    @classmethod
    def urls_must_be_valid_and_normalized(cls, v: list[str]) -> list[str]:
        """Validate URL scheme and normalize to canonical form.

        Normalization rules:
        1. Lowercase scheme and hostname (RFC 3986 §3.1, §3.2.2)
        2. Strip trailing slash from path (unless path is exactly '/')
        3. Remove default ports (:80 for http, :443 for https)
        4. Remove empty fragment (#)

        This ensures semantically identical URLs like
        ``https://A.com/path/`` and ``https://a.com/path`` produce
        the same canonical string, enabling correct dedup in
        CitationAgent's [N] assignment.
        """
        normalized: list[str] = []
        for url in v:
            if not url.startswith(("http://", "https://")):
                raise ValueError(
                    f"Source URL must start with http:// or https://, got: {url}"
                )
            normalized.append(Finding.normalize_url(url))
        return normalized

    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize a URL to its canonical form.

        Exposed as a static method so downstream consumers (e.g.
        CitationAgent dedup, L1-04 duplicate URL detection) can
        reuse the same normalization logic.

        Args:
            url: A valid HTTP/HTTPS URL string.

        Returns:
            The canonicalized URL string.
        """
        parsed = urlparse(url)

        # Lowercase scheme and host
        scheme = parsed.scheme.lower()
        host = parsed.hostname or ""
        port = parsed.port

        # Remove default ports
        if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            port = None

        netloc = host
        if port:
            netloc = f"{host}:{port}"
        if parsed.username:
            userinfo = parsed.username
            if parsed.password:
                userinfo += f":{parsed.password}"
            netloc = f"{userinfo}@{netloc}"

        # Strip trailing slash (unless path is root '/')
        path = parsed.path
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")

        # Remove empty fragment
        fragment = parsed.fragment if parsed.fragment else ""

        return urlunparse((scheme, netloc, path, parsed.params, parsed.query, fragment))


class WorkerOutput(BaseModel):
    """Structured output schema for a research Worker subagent.

    This model is used as ``response_format`` in the Worker SubAgent spec,
    ensuring the LLM produces well-structured, machine-parseable output
    while preserving full fact-to-source provenance.
    """

    summary: str = Field(
        description="2-3 sentence overview of the core findings.",
    )
    findings: list[Finding] = Field(
        description=(
            "Key factual findings extracted from research. "
            "Each finding binds a claim to its source(s) and evidence. "
            "Aim for 3-10 findings for typical research tasks."
        ),
        min_length=1,
    )
    sources_consulted: list[str] = Field(
        default_factory=list,
        description=(
            "All URLs searched during research, including those "
            "not directly cited in findings. Each entry is a URL "
            "optionally followed by ' — ' and a brief description."
        ),
    )
    caveats: str = Field(
        default="",
        description=(
            "Information gaps, uncertainties, or search limitations "
            "encountered during research. Empty string if none."
        ),
    )
```

### 2.3 关键决策：Worker 不使用编号引用

Worker 输出中**禁止使用 `[1]`, `[2]` 编号引用**。原因：
- 多个 Worker 各自独立编号会导致冲突
- 编号分配应留给下游（Supervisor 或 CitationAgent）统一处理
- 通过 `Finding.claim` 的 Pydantic `field_validator` 在 schema 级别强制执行

### 2.4 1:N Source 映射

一个 Finding 可以绑定多个 Source URL（`source_urls: list[str]`）。

**适用场景**：
- 一个 claim 需要多个来源佐证（例如 "LangGraph 和 CrewAI 都支持并行执行" 来自两个不同文档）
- 同一事实在多个来源中被独立验证

**下游处理**：
- CitationAgent 为每个 distinct URL 分配独立编号：`"LangGraph 和 CrewAI 都支持并行执行 [1][2]"`
- 如果多个 findings 共享同一 URL，CitationAgent 复用编号

---

## 三、多跳搜索与 Output Token Limit

### 3.1 多跳搜索的 Provenance 处理

Worker 经常执行多轮搜索：Search 1 发现线索 → Search 2 深入追踪。

**规则**：每个 finding 的 `source_urls` 应标注**最直接支撑该 claim 的来源**，而非中间跳转的 URL。即使通过 Search 1 发现 PostgresSaver 的存在，如果池化细节来自 Search 2，finding 的 source 应指向 Search 2 的结果。中间跳转的 URL 记录在 `sources_consulted` 中。

### 3.2 Output Token Limit 风险与缓解

Worker 在执行多轮搜索后，累积的 findings 可能导致最终 structured output 接近或超过 output token limit。

**缓解策略**：Worker prompt 中包含 "Extract Immediately" 指令——要求 Worker 在**每轮搜索后立即提取和记录关键 findings**，而非在所有搜索完成后一次性回忆。这确保：
- 即使 output 被截断，早期 findings 不会丢失（因为 LLM 的内部推理已经处理过）
- Worker 的认知负载分散到每轮搜索后，提高提取质量

**SummarizationMiddleware 注**：Worker subagent 的中间件栈包含 `SummarizationMiddleware`，但 Worker 任务已是拆解后的子任务，搜索轮次通常较少（2-5 轮），不应触发 summarization threshold。暂不做特殊处理。

---

## 四、Prompt 变更

### 4.1 修改后的 Worker Prompt

```python
WORKER: str = """\
## Role
You are an expert research worker. Think like a human researcher with 
limited time. Your goal is to answer the user's objective as efficiently 
as possible with full source attribution.

## Research Strategy
1. **Analyze the Request**: Read the objective carefully. What *specific* 
   information is needed?
2. **Broad First**: Start with broad searches (`max_results=10`) for 
   comprehensive coverage.
3. **Parallel Execution**: You should make **multiple tool calls in 
   parallel** for independent sub-topics.
   - Example: If you need info on Companies A, B, and C, call searches 
     for all three in the *same* turn.
4. **Extract Immediately**: After EACH search round, mentally extract 
   and record the key findings, source URLs, and supporting evidence. 
   Do NOT defer extraction to the end — your context window is finite.
5. **Reflect & Assess**: After every step:
   - "Do I have enough to answer the objective?"
   - "Am I stuck in a loop or repeating searches?"
   - "Have I recorded findings from all completed searches?"
6. **Stop Early**: Quality > Quantity. Stop if sufficient info is found
   or last 2 searches yielded same results.
7. **Extract Quantitative Data**: Prioritize specific numbers 
   (percentages, scores, dollar amounts, dates).

## Output Rules
- Your output MUST conform to the structured schema provided.
- Each Finding must have at least one source URL.
- A Finding may have multiple source URLs if the claim is supported 
  by multiple sources.
- For each source URL, include its title from the search results in 
  the source_titles field. If the title is unavailable, use the URL's 
  domain name as fallback.
- Evidence should be a brief quote or close paraphrase — concise but
  sufficient for downstream verification. Use quotation marks for 
  direct quotes.
- DO NOT use numbered citations like [1], [2] anywhere in your output.
- For paywalled sources, use the accessible secondary source URL 
  directly — do not reference the paywalled URL.

## Protocol
- Search tool ALWAYS returns URLs. Never claim URLs are "not provided".
- If primary source is paywalled, secondary source is ACCEPTABLE.
- If you hit a limit, output what you have with a caveat.
"""
```

### 4.2 变更对照

| 部分 | 之前 | 之后 | 原因 |
|------|------|------|------|
| Output Format | 无强制格式 | Pydantic `response_format=WorkerOutput` | 100% 解析可靠性 |
| Output Template | 无 | 不需要（由 Pydantic schema Field descriptions 控制） | 结构由 schema 保证 |
| 编号引用 | 隐含允许 | 明确禁止 + Pydantic validator | 避免跨 Worker 冲突 |
| Evidence 字段 | 不存在 | Pydantic `Finding.evidence` | 可验证性 |
| Source 映射 | 隐含 1:1 | 明确 1:N (`list[str]`) | 灵活性 |
| Source Title | 不存在 | `Finding.source_titles` | 下游 Sources 列表可读性 |
| Extract Immediately | 不存在 | 新增 Research Strategy §4 | 防止 output token limit 截断 |
| Paywalled 内容 | "使用二级来源" | "Source URL 直接指向二级来源" | 简化逻辑 |

### 4.3 Subagent 注册变更

```python
research_subagent: dict[str, Any] = {
    "name": "research-worker",
    "description": (
        "Conducts focused web research on a specific sub-topic. "
        "Delegate narrow, well-scoped objectives to this agent."
    ),
    "system_prompt": DeepAgentPrompts.WORKER,
    "tools": [search_tool],
    "model": worker_model,
    "response_format": WorkerOutput,  # ← 新增：Pydantic structured output
}
```

---

## 五、消费策略

### 5.1 Supervisor 直接消费（主路径）

Supervisor 收到的 `ToolMessage.content` 是 `WorkerOutput` 的 JSON 字符串。LLM 可以直接理解 JSON 内容，综合写 draft report。

```json
{
  "summary": "LangGraph v0.2 引入了 checkpointing 机制...",
  "findings": [
    {
      "claim": "LangGraph 支持 MemorySaver 和 PostgresSaver 两种 checkpointer 后端",
      "source_urls": ["https://langchain.com/docs/langgraph/checkpointing"],
      "evidence": "\"LangGraph provides two built-in checkpointer implementations...\""
    }
  ],
  "sources_consulted": [
    "https://langchain.com/docs/langgraph/checkpointing — LangGraph official docs",
    "https://blog.langchain.com/langgraph-persistence — LangChain blog"
  ],
  "caveats": ""
}
```

### 5.2 程序化解析（CitationAgent / Validation 路径）

```python
from deep_research_agent.models.worker_output import WorkerOutput

def parse_worker_output(tool_message_content: str) -> WorkerOutput:
    """Parse Worker structured output from ToolMessage content.
    
    Args:
        tool_message_content: JSON string from ToolMessage.content.
        
    Returns:
        Parsed WorkerOutput instance.
        
    Raises:
        ValidationError: If JSON is malformed or violates schema constraints.
    """
    return WorkerOutput.model_validate_json(tool_message_content)
```

### 5.3 降级路径

如果 `structured_response` 为 `None`（例如模型不支持 structured output），`ToolMessage.content` 退回为 Worker 最后一条消息的纯文本。

**降级链**：
```
L1 (主路径): Pydantic response_format → structured_response → JSON → model_validate_json()
    ↓ structured_response 为 None
L2 (降级):  ToolMessage.content 是纯文本 → Supervisor LLM 直接阅读（可用但无结构保证）
            CitationAgent 使用 LLM 提取 claim-source pairs（CitationAgent 本身是 LLM）
```

> **注**：L2 降级路径中不引入额外的"正则解析"或独立 LLM 调用。如果 Worker 输出是纯文本，CitationAgent 作为 LLM 会在其自身的 inference 中处理非结构化输入（02_citation_annotation_design.md §5.3 已定义此容错路径）。

### 5.4 跨语言兼容

- **Schema 字段名**（`claim`, `source_urls`, `evidence`, `summary` 等）：始终英文——由 Pydantic schema 固定
- **字段值**（claim 文本、evidence 引文、summary 内容）：跟随 Worker query 和搜索结果的语言自适应
- **示例**：用户中文 query → Worker 搜索中文资料 → `claim` 和 `evidence` 为中文；用户中文 query 但研究更适合英文资料 → Worker 搜索英文 → 字段值为英文

这天然兼容 Pydantic 方案，无需额外设计。

---

## 六、Token 开销与边界场景

### Token 对比

| 格式 | 典型 token 数 | vs 纯叙述增幅 |
|------|-------------|-------------|
| 纯叙述（现状） | 300-500 | baseline |
| Pydantic JSON | 400-750 | +20-50% |
| 原始搜索结果 | 5000-50000 | +10x-100x |

> JSON 格式比 Markdown 三元组略高（字段名、引号、括号等开销），但换来 100% 解析可靠性。核心 trade-off：~20% 额外 token 换取零解析失败。

### 边界场景

| 场景 | 处理方式 |
|------|---------|
| **同一 Claim 多个 Sources** | `source_urls` 列出所有支撑来源（1:N 映射） |
| **Source 被 paywalled** | 使用可访问的二级来源，`source_urls` 直接指向二级来源 URL |
| **搜索无结果** | `findings` 列表为空（`min_length=1` 约束在无结果时放宽），`caveats` 中说明原因 |
| **findings 过多（>10）** | 不硬限制，但 prompt 中 "Aim for 3-10" 提供软指导 |
| **output token 接近上限** | "Extract Immediately" 策略确保早期 findings 已被处理 |

---

## 七、与其他组件的交互

| 下游组件 | 消费 Worker 输出的哪部分 | 如何消费 |
|---------|------------------------|---------| 
| **Supervisor** | 整个 JSON（via ToolMessage） | LLM 直接阅读 JSON，综合写 draft report |
| **CitationAgent** | `findings[*].claim` + `findings[*].source_urls` + `findings[*].source_titles` | `WorkerOutput.model_validate_json()` 或 LLM 直接阅读 |
| **Citation Validation (L2)** | `findings[*].claim` + `findings[*].evidence` | 与报告 claim 做语义匹配 |
| **Citation Validation (L3)** | `findings[*].source_urls` + `findings[*].evidence` | 回溯 URL 验证 evidence 真实性 |
| **Source Retention** | `sources_consulted` | 搜索审计记录 |

---

## 八、输出验证

### 8.1 验证数据结构

```python
from dataclasses import dataclass


@dataclass
class WorkerOutputValidation:
    """Validation result for Worker structured output quality."""

    is_structured: bool         # response_format 是否成功产出 structured_response
    finding_count: int
    findings_with_urls: int     # 有 ≥1 个 source_url 的 Finding 数
    findings_with_evidence: int
    has_numbered_citations: bool  # claim/evidence 中是否（错误地）包含 [N]
    has_summary: bool
    total_source_urls: int      # 所有 findings 中 source_urls 的总 URL 数

    @property
    def is_valid(self) -> bool:
        """最低合格标准。"""
        return (
            self.is_structured
            and self.finding_count >= 1
            and self.findings_with_urls == self.finding_count
            and not self.has_numbered_citations
        )

    @property
    def quality_score(self) -> float:
        """输出质量评分 [0.0, 1.0]，用于 monitoring 和 benchmark。"""
        if self.finding_count == 0:
            return 0.0
        source_ratio = self.findings_with_urls / self.finding_count
        evidence_ratio = self.findings_with_evidence / self.finding_count
        structure_score = 1.0 if self.is_structured else 0.0
        citation_penalty = 0.0 if not self.has_numbered_citations else -0.3
        return max(0.0, min(1.0, (
            structure_score * 0.2
            + source_ratio * 0.4
            + evidence_ratio * 0.3
            + (0.1 if self.has_summary else 0.0)
            + citation_penalty
        )))
```

### 8.2 验证触发时机

| 阶段 | 触发方式 | 失败处理 |
|------|---------|---------|
| 开发/测试 | 每次 Worker 输出后同步执行 | Warning log + 记录到 benchmark 报告 |
| 生产 | 异步验证（不阻塞主流程） | Warning log + observability 指标 |

---

## 九、验收标准

| ID | 验收条件 | 验证方法 |
|----|---------|---------|
| AC-1 | Worker subagent 使用 `response_format=WorkerOutput` 注册 | 代码审查 |
| AC-2 | Worker 输出在 10/10 测试查询中成功产出 `structured_response` | 集成测试 |
| AC-3 | `WorkerOutput.model_validate_json()` 在所有 Worker 输出上成功 | 单元测试 |
| AC-4 | Worker 输出不包含 `[N]` 编号引用 | Pydantic validator 自动验证 |
| AC-5 | 每个 Finding 至少有 1 个 `source_url` | Pydantic `min_length=1` |
| AC-6 | 下游 CitationAgent 能从 JSON 输出中提取 claim-source 映射 | 端到端集成测试 |
| AC-7 | 降级路径：即使 `structured_response=None`，Supervisor 仍能生成报告 | Monkey-test |
| AC-8 | Token 增幅在 +20-50% 范围内（vs 纯叙述 baseline） | Token 计数对比 |
| AC-9 | 跨语言兼容：中文 query → 中文 findings；英文 query → 英文 findings | 手动测试 |
| AC-10 | `source_titles` 与 `source_urls` 长度一致（当 titles 可用时） | 单元测试 |
| AC-11 | URL 标准化：`https://A.com/path/` 和 `https://a.com/path` 产出相同规范 URL | 单元测试 |
| AC-12 | `Finding.normalize_url()` 与 `CitationStructureValidator._extract_url()` 配合使用后，L1-04 去重不产生误判 | 集成测试 |
