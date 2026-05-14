# 第二章：架构设计

## 2.1 问题定义

`deep_research_agent` 是一个 **Supervisor-Worker 架构**的深度研究系统。当前的 prompt 体系（`DeepAgentPrompts`）定义了三种角色：

- **Supervisor**：调度者，负责查询分解、任务分配、结果综合和报告撰写
- **Worker**：搜索研究员，负责执行具体的搜索任务和信息提取
- **Citation Specialist**：引用标注员，负责为最终报告添加精确的内联引用

这三种角色的"人格"是**功能型**的——为了完成研究任务而优化。而名人 Skills 的本质是**认知型**的——模拟特定个体的决策框架、表达风格和知识边界。

**核心问题**：在一个以"客观研究"为目标的系统中，注入"主观人格"，两者如何共存而不冲突？

---

## 2.2 三种候选架构方案

在确定最终方案之前，我们评估了三种不同的集成方式：

### 方案 A：作为独立 Subagent 挂载

**思路**：将名人 Skill 封装为一个独立的 `persona-advisor` subagent，与 `research-worker` 和 `citation-specialist` 平级。Supervisor 在需要时按需调度。

```python
# 概念示意
persona_advisor: dict[str, Any] = {
    "name": "persona-advisor",
    "description": "Provides analysis through the lens of a specific expert persona...",
    "system_prompt": loaded_persona_prompt,  # 动态加载
    "tools": [],  # 纯推理，无需搜索工具
    "model": worker_model,
}
```

| 维度 | 评价 |
|------|------|
| **隔离性** | ⭐⭐⭐⭐⭐ — 人格 prompt 与核心研究 prompt 完全隔离 |
| **可插拔性** | ⭐⭐⭐⭐⭐ — 新增/删除人格不影响主流程 |
| **框架遵循度** | ⭐⭐ — Supervisor 可能不会总是调度该 subagent |
| **全局影响力** | ⭐⭐ — 人格框架只影响 advisor 的输出，不影响任务分解 |
| **实现复杂度** | ⭐⭐⭐⭐ — 只需扩展 subagents 列表 |

**缺点**：用户选择"巴菲特视角"时，期望的是**整个研究流程都以价值投资框架来分析**，而不仅仅是在最后加一段巴菲特的点评。方案 A 无法影响 Supervisor 的任务分解策略。

### 方案 B：Middleware 注入 Supervisor Prompt ✅ 最终选型

**思路**：类似现有的 `BudgetTrackingMiddleware`，创建 `PersonaMiddleware`，在 Supervisor 的每轮 system prompt 尾部注入当前激活的人格框架。

```
Supervisor System Prompt
├── §1-7: Research Operations Handbook（不变）
├── §8: Analysis Framework Integration Hook（新增静态 hook）
├── 🧠 Active Analysis Framework: 巴菲特（PersonaMiddleware 动态注入）
└── ⏱ Budget Status（BudgetTrackingMiddleware 动态注入，始终在最末端）
```

| 维度 | 评价 |
|------|------|
| **隔离性** | ⭐⭐⭐⭐ — 仅影响 Supervisor，Worker/Citation 完全隔离 |
| **可插拔性** | ⭐⭐⭐⭐⭐ — Middleware 可空挂（persona=None 时为 passthrough） |
| **框架遵循度** | ⭐⭐⭐⭐⭐ — 直接影响 Supervisor 的任务分解和报告撰写 |
| **全局影响力** | ⭐⭐⭐⭐ — 从任务分解到搜索指令到最终报告都受框架影响 |
| **实现复杂度** | ⭐⭐⭐⭐ — 复用现有 Middleware 基础设施 |

**为什么选择方案 B？**

用户使用 `--persona buffett` 时的意图很明确：**按照巴菲特的价值投资框架来分析这个问题**。这意味着：
1. Supervisor 应该按照"护城河 → 安全边际 → 管理层 → 财务健康"来**分解子任务**
2. 给 Worker 的任务描述应该体现框架的**关键词和优先级**
3. 最终报告应该以框架的**叙事风格和分析维度**来组织

方案 B 是唯一能同时实现以上三点的架构。

### 方案 C：后处理 Rewriter

**思路**：研究流程完全不变。在 `citation-specialist` 输出最终报告后，新增一个 `persona-rewriter` 环节，用名人视角对报告进行"二次解读"。

| 维度 | 评价 |
|------|------|
| **客观性保护** | ⭐⭐⭐⭐⭐ — 核心研究流程完全不受影响 |
| **实现简单度** | ⭐⭐⭐⭐⭐ — 只需在 stream 末尾追加一步 |
| **框架遵循度** | ⭐⭐ — 仅改变表述风格，不影响研究方向 |
| **用户价值** | ⭐⭐⭐ — 用户拿到两份输出（客观报告 + 人格解读），但增加延迟 |

**缺点**：本质上只是"换一个语气重写"，没有从根本上改变研究的分析维度。

---

## 2.3 最终方案：Middleware 注入 + Supervisor-Only 作用域

### 核心设计决策

#### 决策 1：人格框架仅作用于 Supervisor

**理由**：Worker 的职责是**客观地搜索和提取信息**。如果 Worker 也被注入巴菲特框架，它可能在搜索时倾向于价值投资相关的结果，而忽略技术分析或量化交易的观点，这会导致**信息采集偏见**。

**实现机制**：Middleware 的 `wrap_model_call` 只在 Supervisor 的 LLM 调用链上触发。Worker 和 Citation Specialist 运行在独立的 subagent 图中，有自己的 system prompt，不经过 Supervisor 的 middleware 链。

```
┌─────────────────────────────────────────────┐
│ Supervisor (with PersonaMiddleware)          │
│  ├── System Prompt = SUPERVISOR + §8 Hook   │
│  ├── + 🧠 Active Analysis Framework (注入)  │
│  └── + ⏱ Budget Status (注入)               │
├─────────────────────────────────────────────┤
│ Worker (独立 subagent, 不受影响)             │
│  └── System Prompt = WORKER (原版)           │
├─────────────────────────────────────────────┤
│ Citation Specialist (独立 subagent, 不受影响)│
│  └── System Prompt = CITATION (原版)         │
└─────────────────────────────────────────────┘
```

#### 决策 2：Middleware 排序 — PersonaMiddleware 在 BudgetTrackingMiddleware 之前

**理由**：Middleware 链采用洋葱模型（first = outermost）。最后注入的内容出现在 system prompt 最末端。Budget 警告（如 "⚠️ CRITICAL: 仅剩 2 turns"）必须始终是 Supervisor 看到的**最后一段**，具有最高优先级。

```python
middleware=[
    persona_middleware,       # 第一个注入 → 在 system prompt 中间
    budget_middleware,        # 第二个注入 → 在 system prompt 最末端
    CitationDataMiddleware(max_retries=citation_max_retries),
]
```

**结果的 System Prompt 结构**：

```
[原始 SUPERVISOR prompt: §1-§8]
[PersonaMiddleware 注入: 🧠 Active Analysis Framework]
[BudgetTrackingMiddleware 注入: ⏱ Budget Status]
```

#### 决策 3：使用名人真名作为 persona_id

**理由**：当前为个人研究项目，使用真名更直观。后续如果商业化部署，可以将 `persona_id` 映射为功能性描述（如 `buffett` → `value_investor`）。`registry.yaml` 的声明式结构使得这种映射改动只需修改配置文件，不涉及代码变更。

#### 决策 4：使用上游 .skill 文件原版，不做精简

**理由**：
- 精简 prompt 需要逐个 skill 进行人工审查，工作量大
- 上游 skills 的完整内容包含了详细的思维框架、示例和边界定义，截断可能丢失关键信息
- 对于 128K+ context 的模型（如 DeepSeek V4），额外 2-4K tokens 的开销完全可以接受
- 后续可根据实际效果有选择性地精简

#### 决策 5：注册全部 19 个适合的 Skills

**理由**：原设计仅计划 Phase 1 注册 6 个核心人格，但用户明确要求"所有适合融入的都加入"。由于 `registry.yaml` 是声明式的，且框架文件是按需加载的（仅在 `persona_id` 匹配时才读入内存），注册 19 个 vs 6 个对运行时性能无影响。

#### 决策 6：在 SUPERVISOR prompt 中预留 §8 Hook

**理由**：PersonaMiddleware 的注入块虽然已包含 "MUST decompose tasks using this framework" 指令，但 Supervisor 的原始 prompt 中没有任何关于 persona 的提及。添加 §8 让 Supervisor 在接收到注入块时有一个明确的**行为锚点**——它知道这个块的存在是预期的、该如何处理。

如果实测中 Supervisor 仍然忽略注入框架，可以在 §8 中进一步强化指令语气。
