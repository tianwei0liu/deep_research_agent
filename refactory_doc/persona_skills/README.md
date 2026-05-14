# 名人 Skills（人格蒸馏）集成方案

## 文档概览

本文档完整记录了将**名人蒸馏 Skills（Persona Distillation Skills）**融入 `deep_research_agent` 的技术方案、实现细节与设计决策。

### 文档结构

| 章节 | 文件 | 内容 |
|------|------|------|
| 第一章 | [01_background_and_research.md](01_background_and_research.md) | 背景研究：人格蒸馏概念、上游 GitHub 生态、完整 Skills 清单与来源标注 |
| 第二章 | [02_architecture_design.md](02_architecture_design.md) | 架构设计：三种集成方案对比、最终选型（Middleware 注入）及其核心设计决策 |
| 第三章 | [03_implementation_details.md](03_implementation_details.md) | 实现细节：数据层（PersonaRegistry）、中间件层（PersonaMiddleware）、Supervisor Prompt Hook、API/CLI 贯通 |
| 第四章 | [04_risks_and_mitigations.md](04_risks_and_mitigations.md) | 风险与缓解：版权合规、Prompt 冲突、Context Window、幻觉放大、文化敏感性 |
| 第五章 | [05_verification_and_roadmap.md](05_verification_and_roadmap.md) | 验证计划与路线图：测试策略、已验证结果、后续迭代方向 |

### 核心设计决策摘要

1. **集成方式**：选择 **Middleware 注入模式**（非独立 Subagent），通过 `PersonaMiddleware` 在 `wrap_model_call` 层面拦截 Supervisor LLM 请求，动态注入人格框架
2. **作用范围**：人格框架**仅作用于 Supervisor**（负责任务分解与报告撰写），Worker 和 Citation Specialist 保持独立运行环境
3. **命名策略**：使用名人**真名**作为 `persona_id`（如 `buffett`、`zhangxuefeng`），后续按需评估匿名化
4. **数据管理**：使用 `registry.yaml` 声明式注册表 + `frameworks/*.md` 纯 Markdown 框架文件
5. **框架内容**：直接使用上游 `.skill` 文件原版，不做精简
6. **覆盖范围**：注册全部 19 个适合深度研究分析的 Skills

### 变更文件清单

| 操作 | 文件 | 说明 |
|------|------|------|
| NEW | `personas/__init__.py` | 包初始化 |
| NEW | `personas/registry.py` | PersonaRegistry + PersonaConfig |
| NEW | `personas/registry.yaml` | 声明式注册表（19 个人格） |
| NEW | `personas/frameworks/*.md` | 19 个上游 .skill 框架文件 |
| NEW | `agents/persona_middleware.py` | PersonaMiddleware |
| MODIFY | `agents/prompts.py` | 新增 §8 Analysis Framework Integration hook |
| MODIFY | `agents/agent.py` | 添加 persona_id 参数，注册 middleware |
| MODIFY | `examples/run_deep_agent.py` | --persona CLI 参数 |
