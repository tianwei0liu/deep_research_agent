# 第四章：风险与缓解策略

## 4.1 版权与法律风险

### 4.1.1 开源许可证合规

**现状**：所有上游仓库均未声明开源许可证。GitHub 默认版权规则下，未声明 LICENSE 的仓库保留所有权利（All Rights Reserved）。

| 上游仓库 | LICENSE 文件 | 许可证类型 | 兼容性 |
|-----------|------------|-----------|--------|
| `will2025btc/buffett-perspective` | 存在 | 未确认具体类型 | 需逐一核查 |
| `alchaincyf/steve-jobs-skill` | 存在 | 未确认具体类型 | 需逐一核查 |
| `alchaincyf/zhangxuefeng-skill` | 存在 | 未确认具体类型 | 需逐一核查 |
| 其余 alchaincyf/* 系列 | 需确认 | - | - |
| 各独立作者仓库 | 需确认 | - | - |

> **注**：在 API 探查中，部分仓库（如 `will2025btc/buffett-perspective` 和 `alchaincyf/steve-jobs-skill`）的文件列表中确实存在 `LICENSE` 文件，但未深入确认其具体许可证类型。在 `registry.yaml` 中统一标记为 `NONE` 以保守处理。

### 4.1.2 人格权与肖像权

在中国法律下（《民法典》第994条），自然人死亡后，其姓名权、肖像权、名誉权仍受保护。以名人名义提供商业化的咨询服务（即使是 AI 模拟），家属有权主张侵权。

在美国法律下，公众人物的 "Right of Publicity" 在某些州延续到死后（加州：死后70年）。

### 4.1.3 风险矩阵

| 使用场景 | 风险等级 | 说明 |
|---------|---------|------|
| 个人学习 / 内部研究 | 🟢 低 | 可以自由使用真名，无合规要求 |
| 开源项目（非商业） | 🟡 中 | 确认上游许可证；README 中不以名人名字作为功能卖点 |
| B2C 商业产品 | 🔴 高 | **不要使用真实名人姓名**；改用功能性描述（如"价值投资分析框架"） |

### 4.1.4 当前策略

- **使用真名**：当前为个人学习项目，风险可接受
- **架构预留**：`registry.yaml` 的声明式结构允许在不改代码的情况下将 `display_name` 从 "巴菲特" 改为 "价值投资框架"
- **溯源记录**：每个 persona 条目记录了 `upstream.repo` 和 `upstream.license`，为后续合规审计提供追踪链

---

## 4.2 Prompt 冲突风险

### 问题描述

名人 Skills 本质是**超长 system prompt**，通常包含：
- 角色设定（"你是巴菲特"）
- 行为约束（"你必须用犀利的语气"）
- 知识边界（"你的知识截止于某年"）

当这些被注入到 Supervisor system prompt 后，可能与 DeepAgentPrompts 产生**指令冲突**：

```
Supervisor prompt: "你是一个 Research Supervisor，客观地协调研究团队..."
巴菲特 skill:      "你是沃伦·巴菲特，用老派价值投资者的犀利口吻..."
```

### 缓解措施（已实施）

1. **Supervisor-Only 作用域**：PersonaMiddleware 仅影响 Supervisor，Worker 和 Citation Specialist 完全隔离，避免信息采集偏见

2. **§8 Hook 明确优先级**：SUPERVISOR prompt 的 §8 明确声明 "The framework does NOT override the Research Operations Handbook above — it adds a lens on top of it"，建立了层级关系

3. **注入块包含边界指令**：PersonaMiddleware 的注入块第4条明确要求 "If a sub-topic falls outside this framework's scope, acknowledge it explicitly rather than forcing a fit"

4. **DISCLAIMER 尾缀**：注入块末尾强制追加免责声明，提醒这是 AI 模拟框架

### 潜在的二阶风险

- **角色混淆**：Supervisor 同时收到"你是 Research Supervisor"和"你是巴菲特"两个角色设定，可能导致行为不一致
- **缓解方向**：后续可考虑对上游 `.skill` 文件做预处理，去掉其中的角色扮演指令（如 "你是..."），仅保留分析框架和决策模型

---

## 4.3 Context Window 消耗

### 定量分析

| 组件 | 估算 tokens |
|------|-----------|
| SUPERVISOR 原始 prompt（§1-§8） | ~3,000 |
| PersonaMiddleware 注入块（头尾 + SKILL.md） | ~2,700-4,500 |
| BudgetTrackingMiddleware 注入块 | ~200-400 |
| **Supervisor 总 system prompt** | **~5,900-7,900** |

对比：DeepSeek V4 的 context window 为 128K tokens，Supervisor 的 system prompt 占比 < 6%。

### 多 Persona 场景（未来扩展）

当前设计（Phase 1）仅支持单 persona 模式。如果未来需要支持 "巴菲特 + 芒格" 组合框架：
- 两个框架合计约 6,500 tokens，system prompt 总计约 9,500 tokens
- 仍在可接受范围内，但需要重构 PersonaMiddleware 以支持多框架合并注入

### 缓解措施（预留但未实施）

- `registry.yaml` 中的 `max_tokens` 字段为**自动摘要压缩**预留接口
- 后续可实现：当框架文本超过 `max_tokens` 时，自动使用 LLM 摘要压缩

---

## 4.4 幻觉放大效应

### 问题描述

名人 Skills 鼓励模型"扮演"特定角色。这会**降低模型的不确定性表达能力**：
- 巴菲特不会说"我不确定"——他会给出一个斩钉截铁的判断
- 张雪峰不会说"这个问题需要更多数据"——他会直接给出务实的建议

在 B2C 产品中，用户可能将 AI 的"巴菲特式断言"误读为真实的投资建议。

### 缓解措施（已实施）

1. **DISCLAIMER 强制追加**：PersonaMiddleware 的注入块末尾始终包含：
   > ⚠️ DISCLAIMER: This analysis uses an AI-simulated decision framework. It does not represent the views of any real individual and does not constitute professional advice.

2. **§8 Hook 中的 scope 限制**：明确要求 Supervisor "Acknowledge scope limits: If a sub-topic falls outside the framework's expertise, explicitly state so rather than forcing a fit"

### 缓解措施（待实施）

- 在 UI 层面做视觉区分（如不同颜色的卡片），明确标注"AI 模拟视角"
- 在最终报告的开头/结尾自动插入更醒目的免责声明

---

## 4.5 文化敏感性

### 风险点

- **张雪峰**：在其离世的语境下，将其"蒸馏"为 AI 工具，在中国社会可能存在情感上的抵触。部分用户可能认为这是对逝者的不敬
- **毛选.skill**：在部分市场/语境中，使用政治人物或政治文献的框架可能引发争议
- **特朗普.skill**：使用其 skill 可能在某些市场引发政治争议

### 缓解策略

- 当前为个人研究项目，不涉及公开部署，风险可忽略
- 未来若商业化，建议：
  1. 将所有涉及真实个人的 persona 替换为功能性描述
  2. 添加用户知情同意机制
  3. 在特定地区/语言环境中禁用部分敏感 persona

---

## 4.6 版本更新与维护成本

### 问题描述

上游 `.skill` 文件本质上是**纯文本 prompt 文件**：
- 没有语义化版本号（SemVer）
- 没有 changelog
- 随时可能被作者大幅修改或删除
- 名人的"认知模型"有**时效性**（如巴菲特 2020 年和 2026 年对 AI 股票的看法可能截然不同）

### 当前策略：Fork + Pin

我们选择**本地 fork**上游内容：
- 所有框架文件存放在本项目 Git 仓库的 `personas/frameworks/` 目录下
- `registry.yaml` 中记录上游仓库 URL，但不做 `git submodule` 引用
- 每次修改都通过 Git commit 跟踪变更历史

### 后续建议

1. **定期 diff**：编写脚本对比本地 fork 与上游 `SKILL.md` 的差异，由人工决定是否合并
2. **Pin to commit hash**：在 `registry.yaml` 中增加 `pinned_commit` 字段
3. **本地版本管理**：在 `registry.yaml` 中增加 `local_version` 字段

```yaml
# 未来增强示例
buffett:
    file: frameworks/buffett.md
    upstream:
      repo: "https://github.com/will2025btc/buffett-perspective"
      pinned_commit: "a1b2c3d"      # ← 新增
      last_synced: "2026-05-13"     # ← 新增
      license: "NONE"
    local_version: "1.0.0"          # ← 新增
```
