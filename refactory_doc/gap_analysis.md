# Deep Research Agent — 业界差异与不足全景分析

> **分析日期**: 2026-05-05
> **对比范围**: 项目现状 + 迭代路线图 vs 10款主流竞品
> **竞品**: OpenAI, Google, Claude, Perplexity, Grok, You.com, 豆包, Kimi, 秘塔, 夸克

---

## 目录

1. [分析方法论](#1-分析方法论)
2. [差异分析 — 我们与业界的不同选择](#2-差异分析--我们与业界的不同选择)
3. [不足分析 — 我们落后于业界的维度](#3-不足分析--我们落后于业界的维度)
4. [路线图覆盖度评估](#4-路线图覆盖度评估)
5. [优先级建议](#5-优先级建议)

---

## 1. 分析方法论

**差异（Differences）** = 技术路线或产品策略的不同选择，不一定是劣势，可能是差异化优势或有意识的取舍。

**不足（Shortcomings）** = 在业界已形成共识的"必备能力"上，我们明确落后或缺失的维度。

对比维度覆盖：架构设计、模型策略、搜索基础设施、Agent 协作模式、产品体验、输出能力、安全合规、商业化就绪度。

---

## 2. 差异分析 — 我们与业界的不同选择

### 2.1 模型策略：性价比路线 vs 顶级推理路线

| 维度 | 我们的选择 | 业界主流 |
|------|-----------|---------|
| **主力模型** | DeepSeek V4 (单一模型) | OpenAI o3/o4, Gemini 2.5 Pro, Claude 4 Opus |
| **模型切换** | ❌ 无 | Perplexity/You.com 支持用户自选模型 |
| **双模型协作** | ❌ 无 | Google Flash+Pro, 秘塔小模型+大模型 |
| **推理模式** | thinking-mode 兼容补丁 | 原生推理模型 (o3, QWQ, K2.6-thinking) |

**分析**：我们选择了 DeepSeek V4 作为唯一模型，走性价比路线。这与 OpenAI ($200/月 Pro) 和 Grok ($300/月 Heavy) 的高端路线形成鲜明对比。这是**有意识的差异化选择**，但也意味着在推理深度上存在天花板。

> [!IMPORTANT]
> 路线图 Phase 3.3 规划了多模型路由，但未涉及"快速模型+深度模型"协作模式（如 Google 的 Flash 做检索 + Pro 做推理），这是一个值得考虑的架构模式。

---

### 2.2 Agent 架构：Supervisor-Worker vs 业界多种范式

| 架构类型 | 代表产品 | 我们 |
|---------|---------|------|
| **MARS 多角色 Agent** | Claude (Lead+Searcher+Reader+Synthesizer) | ❌ |
| **Agent Swarm 集群** | Kimi (100-300 子智能体) | ❌ |
| **Supervisor-Worker** | — | ✅ 当前架构 |
| **Plan-Search-Synthesize** | OpenAI | 部分相似 |
| **Agentic RAG** | Perplexity | ❌ |
| **RL 驱动工具选择** | Grok | ❌ |

**分析**：我们的 Supervisor-Worker 架构参考了 Anthropic MARS 博客，但实现上更接近简化版的 "Plan → Delegate → Synthesize" 流程。与 Claude MARS 的关键区别：

1. **角色分化不足**：MARS 有明确的 Searcher/Reader/Synthesizer 分工，我们的 Worker 是"全能型"——搜索+阅读+提取合一
2. **无交叉验证**：MARS 多 Agent 间会交叉验证信息，减少幻觉；我们的 Worker 之间无交互
3. **规模差距**：Kimi 的 Agent Swarm 可调度 100-300 个子智能体；我们的 `max_parallel_workers=10`

> [!NOTE]
> Supervisor-Worker 模式并非劣势——它更简单、可控、易调试。但在处理 Complex/Deep 级查询时，角色分化和交叉验证是业界已验证的质量提升手段。

---

### 2.3 搜索基础设施：自建 SearXNG vs 商业搜索 API

| 维度 | 我们 | 业界主流 |
|------|------|---------|
| **搜索引擎** | 自建 SearXNG (baidu+sogou+bing) | Google 原始索引 / Bing API / 自有爬虫 |
| **搜索质量** | 受引擎可用性影响 (CAPTCHA 问题) | 商业级 SLA |
| **中文平台** | site: fallback (知乎/微博/微信) | 原生平台 API 或深度爬虫 |
| **私域数据** | ❌ | Google Drive/Gmail, SharePoint, 上传文件 |
| **实时数据** | ❌ | Grok X 平台实时流, Google 实时索引 |

**分析**：自建 SearXNG 是一个**成本敏感的务实选择**，避免了 Tavily/Serper 等商业 API 的费用。但这引入了：
- CAPTCHA 导致引擎频繁挂起的稳定性问题（已有 `captcha_mitigation_strategy.md`）
- 搜索质量不如 Google 原始索引或 Bing 商业 API
- 中文平台内容（知乎/微博/微信）的 `site:` fallback 质量远低于原生 API

---

### 2.4 引用系统：结构化引用流水线 vs 内联引用

| 维度 | 我们 | 业界主流 |
|------|------|---------|
| **引用机制** | Worker→Finding→CitationSpecialist 三阶段流水线 | 模型内联生成 |
| **数据模型** | Pydantic 强类型 (Finding/WorkerOutput) | 通常无结构化中间层 |
| **结构校验** | L1 Validator (7 条规则) | 无公开的结构化校验 |
| **URL 归一化** | RFC 3986 规范化去重 | 不详 |

**分析**：这是我们的**差异化优势**。三阶段引用流水线（Worker 提取 → Middleware 注入 → Specialist 标注）在开源项目中属于较先进的设计。Perplexity 和 Claude 的引用透明度被市场高度认可，我们的结构化方案在工程质量上有竞争力。

---

### 2.5 中间件架构：独特的 Budget + Citation 中间件

业界竞品中，**没有公开的中间件系统**用于动态注入预算约束或引用数据。我们的 `BudgetTrackingMiddleware` 和 `CitationDataMiddleware` 是**架构上的创新**，允许在不修改 Agent 核心逻辑的情况下注入横切关注点。

---

## 3. 不足分析 — 我们落后于业界的维度

### 3.1 🔴 P0 — 产品化基础缺失

#### 3.1.1 无前端（所有竞品均有）

| 竞品 | 前端形态 |
|------|---------|
| OpenAI/Claude/Google | 精美 Web UI + 移动 App |
| Perplexity | Web + App + Pages 社交化 |
| 豆包/Kimi/秘塔/夸克 | Web + App + PC 客户端 |
| **我们** | **❌ 仅 CLI (examples/run_deep_agent.py)** |

> [!CAUTION]
> 路线图 Phase 1.4 规划了 MVP 前端 (Streamlit/FastAPI+React)，但作为 B2C 产品，Streamlit demo 与竞品的精美 UI 差距极大。需要认真评估前端技术选型和投入级别。

#### 3.1.2 无多轮对话（仅 API 骨架）

所有 10 款竞品均支持多轮追问。我们的 `thread_id` + `checkpointer` 参数已就绪，但**从未实际启用**（无 MemorySaver 实例化）。

**路线图覆盖**: Phase 1.1 (1天) — 但这只是最基础的内存持久化，远不足以支撑生产环境。

#### 3.1.3 无状态持久化

| 竞品 | 持久化 |
|------|-------|
| 所有商业产品 | 会话历史持久化，跨设备同步 |
| **我们** | **❌ 无任何持久化** |

**路线图覆盖**: Phase 1.5 (Redis) + Phase 2.6 (PostgreSQL) — 正确的两步走策略，但均未开始。

---

### 3.2 🔴 P0 — 研究质量机制缺失

#### 3.2.1 无 Gap Analysis / 迭代补充

| 竞品 | 信息缺口处理 |
|------|-------------|
| OpenAI | ✅ 自动识别信息缺口，迭代补充搜索 |
| Claude | ✅ 交叉验证，发现不一致时自动补研 |
| 秘塔 | ✅ 信息缺口识别 |
| **我们** | **❌ Worker 搜完即止，无质量自检** |

Supervisor Prompt 中虽有 "If a task is finished, review its findings. Do you need more info? Add new tasks" 的指令，但这完全依赖 LLM 的自主判断，**无结构化的 Gap Analysis 机制**。

**路线图覆盖**: Phase 2.4 (Reflection Loop) — 规划正确但未开始。

#### 3.2.2 无上下文管理

| 竞品 | 上下文策略 |
|------|----------|
| Google | 1M+ Token 窗口，无需截断 |
| Grok | 2M Token |
| Kimi | 256K Token |
| Claude | 200K Token |
| **我们** | **DeepSeek V4 ~128K，无截断/摘要策略** |

当研究内容超过上下文窗口时，我们没有任何策略来管理。业界通过超长上下文（Google/Grok）或摘要压缩来解决。

**路线图覆盖**: Phase 2.1 — 规划了 Token 截断 + 摘要压缩，但未开始。

#### 3.2.3 无事实检查 / 交叉验证

Claude MARS 的核心设计之一是多 Agent 交叉验证提取的信息。我们的 Worker 之间完全隔离，无任何信息交叉验证机制。路线图中**未规划**此能力。

---

### 3.3 🟡 P1 — 交互体验差距

#### 3.3.1 无研究计划预览/修改 (Human-in-the-Loop)

| 竞品 | 计划交互 |
|------|---------|
| OpenAI | ✅ 展示计划，用户可修改方向 |
| Google | ✅ 研究大纲确认 |
| Kimi | ✅ 开始前澄清问题 |
| Perplexity | ✅ 澄清对话 |
| **我们** | **❌ 无** |

**路线图覆盖**: Phase 1.2 (3-4天) — 正确规划但依赖 Phase 1.1。

#### 3.3.2 无异步研究模式

| 竞品 | 异步能力 |
|------|---------|
| Kimi | ✅ 后台运行 10-25 分钟，无需保持页面 |
| **其他 9 款** | ❌ |
| **我们** | **❌** |

Kimi 是唯一支持异步的产品。虽然这不是行业共识，但对于耗时长的深度研究，异步模式是重要的 UX 优化。路线图中**未规划**。

#### 3.3.3 无实时进度可视化

OpenAI、Google、Claude 均提供实时搜索进度可视化（已访问来源、当前思考路径）。我们的 `stream_deep_research` 产出 `tool_start`/`tool_end` 事件，但**缺少前端消费层**。

---

### 3.4 🟡 P1 — 输出能力差距

#### 3.4.1 单一输出格式

| 竞品 | 输出格式 |
|------|---------|
| OpenAI | Markdown + PDF 导出 |
| Google | Markdown + Google Docs + Sheets |
| Perplexity | Markdown + Pages (可分享网页) |
| You.com | PDF + PPT 幻灯片 |
| 豆包 | 报告 + 可视化网页 + **播客** |
| 秘塔 | 报告 + **思维导图** + 表格 + Word/PDF |
| Kimi | 文本 + 动态可视化 |
| **我们** | **仅 Markdown 文本** |

**路线图覆盖**: Phase 3.4 (报告导出与分享) — 规划过于笼统，缺少具体的多格式策略。

#### 3.4.2 无多模态能力

Google Gemini 和 Grok 支持图片/视频/音频理解。豆包支持语音、视频、方言输入。我们仅支持纯文本。路线图中**未规划**多模态。

---

### 3.5 🟡 P1 — 搜索能力差距

#### 3.5.1 无私域数据支持

| 竞品 | 私域数据 |
|------|---------|
| OpenAI | Google Drive, SharePoint |
| Google | Drive, Gmail, Calendar |
| Claude | 上传文件 (PDF/Word/Excel) |
| You.com | 企业数据集成 |
| **我们** | **❌ 完全没有** |

路线图中**未规划**任何私域数据集成（文件上传或企业数据源）。

#### 3.5.2 中文平台搜索质量低

知乎/微博/微信搜索目前使用 `site:` prefix fallback，质量远低于原生 API 或专用爬虫。

**路线图覆盖**: Phase 2.8 — 规划了专用 Scrapers，设计文档已完成（`04b_zhihu_scraper.md` 等），但代码未实现。

#### 3.5.3 无学术搜索模式

秘塔提供专门的学术搜索模式（聚焦期刊和数据库），Perplexity 有学术来源偏好。我们的搜索引擎对学术内容没有特殊处理。路线图中**未规划**。

---

### 3.6 🟢 P2 — 生态与商业化差距

#### 3.6.1 无生态集成

| 竞品 | 生态 |
|------|------|
| Google | Workspace (Docs/Sheets/Drive/Gmail) |
| Grok | X 平台实时数据 |
| 夸克 | 阿里生态 (淘宝/支付宝/高德) |
| **我们** | **❌ 无任何生态** |

**路线图覆盖**: Phase 3.5 规划了 MCP 接入 (GitHub/Notion/ArXiv)，方向正确但范围有限。

#### 3.6.2 无安全合规体系

| 竞品 | 安全 |
|------|------|
| You.com | SOC 2 合规，零数据留存 |
| Claude | Constitutional AI |
| **我们** | **❌ 无任何安全审计** |

**路线图覆盖**: Phase 2.7 (安全审计基础) — 仅一行描述，缺少具体规划。

#### 3.6.3 无 API 产品化

Perplexity Sonar 是唯一公开的 Deep Research API。我们虽有 MCP Server，但它是内部工具，不是面向开发者的 API 产品。路线图中**未规划** API 商业化。

#### 3.6.4 无用户系统与计费

所有 B2C 产品均有用户系统和计费逻辑。

**路线图覆盖**: Phase 3.1 (用户管理) — 仅规划了 API Key → OAuth，未涉及计费和配额。

---

## 4. 路线图覆盖度评估

### 4.1 路线图已覆盖的差距

| 差距 | 路线图位置 | 评估 |
|------|-----------|------|
| 多轮对话 | Phase 1.1 | ✅ 合理，1天可完成基础版 |
| Human-in-the-Loop | Phase 1.2 | ✅ 正确优先级 |
| MVP 前端 | Phase 1.4 | ⚠️ Streamlit 不足以支撑 B2C |
| 状态持久化 | Phase 1.5 + 2.6 | ✅ 两步走策略合理 |
| 上下文管理 | Phase 2.1 | ✅ 必要 |
| 质量自检 | Phase 2.4 | ✅ 必要但描述过简 |
| 中文 Scrapers | Phase 2.8 | ✅ 设计文档已完成 |
| 多模型路由 | Phase 3.3 | ✅ 差异化方向 |
| 报告导出 | Phase 3.4 | ⚠️ 描述过于笼统 |
| 长期记忆 | Phase 3.6 | ✅ 远期规划合理 |

### 4.2 路线图未覆盖的重要差距

| 差距 | 业界参考 | 建议优先级 |
|------|---------|-----------|
| **交叉验证/事实检查** | Claude MARS | **P0** — 直接影响报告质量 |
| **文件上传/私域数据** | OpenAI/Claude/Google | **P1** — B2C 基础能力 |
| **异步研究模式** | Kimi | **P1** — 长时间研究的 UX 必备 |
| **多模态理解** | Google/Grok/豆包 | **P2** — 趋势性能力 |
| **学术搜索模式** | 秘塔/Perplexity | **P2** — 垂直场景差异化 |
| **思维导图/可视化输出** | 秘塔/Kimi | **P2** — 中国用户偏好 |
| **多格式输出 (PDF/PPT/播客)** | You.com/豆包/秘塔 | **P2** |
| **API 商业化** | Perplexity Sonar | **P2** — 开发者生态 |
| **搜索结果 Re-ranking** | Claude/Perplexity | **P1** — 搜索质量提升 |
| **自动来源质量评估** | OpenAI | **P1** — 过滤低质量来源 |

---

## 5. 优先级建议

### 5.1 立即行动（本周）

1. **多轮对话激活** — Phase 1.1，API 已就绪，仅需实例化 MemorySaver
2. **基线质量评测** — 用现有 benchmark 框架量化当前报告质量

### 5.2 短期重点（2 周内）

3. **交叉验证机制** — 在 Supervisor 层增加 "Critic" 节点，对 Worker findings 做一致性检查
4. **Gap Analysis 结构化** — 不依赖 LLM 自主判断，显式设计信息缺口检测逻辑
5. **前端技术选型** — B2C 产品不应使用 Streamlit；建议 Next.js + SSE/WebSocket
6. **搜索质量提升** — 引入 Re-ranking 模型 + 来源质量评分

### 5.3 中期规划（1-2 月）

7. **文件上传支持** — 允许用户上传 PDF/Word 作为研究输入
8. **异步研究模式** — 长时间研究任务后台执行
9. **多格式输出** — 至少支持 PDF 导出
10. **双模型协作** — 快速模型做初步搜索，深度模型做推理分析

### 5.4 总结判断

> [!WARNING]
> **核心差距不在技术深度，而在产品化程度。** 我们的 Agent 核心（Supervisor-Worker + Citation Pipeline + Budget Middleware）在开源项目中属于中上水平。但作为 B2C 产品，缺少前端、持久化、多轮对话、用户系统等**基础产品设施**，与任何一款竞品都有显著差距。路线图的优先级排序是正确的（Phase 1 聚焦核心体验），但执行速度是关键。

> [!TIP]
> **差异化机会**：
> 1. 中英双语深度研究 — 市场调研显示"中英双语深度支持是一个被忽视的蓝海"
> 2. 结构化引用流水线 — 我们的 Finding/WorkerOutput/CitationSpecialist 三阶段设计在引用透明度上有竞争力
> 3. DeepSeek V4 性价比 — 成本优势明显，适合免费/低价策略切入
> 4. MCP 开放架构 — 工具层可扩展性强，未来可接入更多数据源
