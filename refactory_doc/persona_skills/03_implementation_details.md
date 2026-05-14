# 第三章：实现细节

## 3.1 数据层：`personas/` 目录结构

### 目录布局

```
personas/
├── __init__.py                # 包初始化，导出 PersonaConfig, PersonaRegistry
├── registry.py                # PersonaRegistry 类 + PersonaConfig 数据类
├── registry.yaml              # 声明式注册表（19 个人格）
└── frameworks/                # 纯 Markdown 人格框架文件
    ├── buffett.md             #  8,125 chars — 巴菲特
    ├── munger.md              # 11,613 chars — 芒格
    ├── jobs.md                # 12,610 chars — 乔布斯
    ├── musk.md                # 10,002 chars — 马斯克
    ├── feynman.md             # 10,243 chars — 费曼
    ├── naval.md               # 11,360 chars — 纳瓦尔
    ├── taleb.md               # 10,955 chars — 塔勒布
    ├── duan_yongping.md       #  8,569 chars — 段永平
    ├── zhang_yiming.md        # 10,286 chars — 张一鸣
    ├── trump.md               # 12,153 chars — 特朗普
    ├── paul_graham.md         # 13,566 chars — Paul Graham
    ├── zhangxuefeng.md        #  7,745 chars — 张雪峰
    ├── zizek.md               #  3,682 chars — 齐泽克
    ├── maoxuan.md             # 10,782 chars — 毛选
    ├── karpathy.md            # 12,967 chars — Karpathy
    ├── ilya.md                # 12,723 chars — Ilya Sutskever
    ├── mrbeast.md             #  9,445 chars — MrBeast
    ├── guodegang.md           # 13,073 chars — 郭德纲
    └── saul_goodman.md        #  6,524 chars — Saul Goodman
```

### 框架文件来源

所有 `frameworks/*.md` 文件均从上游 GitHub 仓库的 `SKILL.md` 文件直接下载。下载脚本尝试 `main` 和 `master` 两个分支：

```bash
# 下载逻辑（已执行，脚本仅供参考）
for branch in main master; do
    curl -fsSL "https://raw.githubusercontent.com/${repo}/${branch}/SKILL.md" \
        -o "personas/frameworks/${name}.md"
done
```

---

## 3.2 PersonaConfig 数据类

位置：`personas/registry.py`

```python
@dataclass(frozen=True)
class PersonaConfig:
    """Validated persona configuration from registry.yaml."""
    persona_id: str                                  # 唯一标识符，如 "buffett"
    display_name: str                                # UI 显示名，如 "巴菲特"
    description: str                                 # 一行描述，如 "价值投资 / 护城河 / 安全边际"
    framework_prompt: str                            # 原始 SKILL.md 内容（完整 markdown）
    max_tokens: int                                  # 软 token 预算（当前未强制）
    applicable_domains: list[str] = field(...)       # 适用领域标签
```

设计要点：
- 使用 `frozen=True` 确保 Config 实例不可变（线程安全）
- `framework_prompt` 存储完整的原始内容，注入时不做截断
- `max_tokens` 目前为软限制（占位符），后续可用于自动摘要压缩
- `applicable_domains` 预留给未来的自动 persona 推荐功能

---

## 3.3 PersonaRegistry 类

位置：`personas/registry.py`

```python
class PersonaRegistry:
    """Discovers and loads persona frameworks from the personas/ directory."""

    def __init__(self, registry_path: Optional[Path] = None) -> None:
        self._registry_path = registry_path or (_PERSONAS_DIR / "registry.yaml")
        self._personas: dict[str, PersonaConfig] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Parse registry.yaml and eagerly load each persona's framework."""
        # 1. 读取 YAML
        # 2. 遍历 personas 映射
        # 3. 校验框架文件存在性
        # 4. 读取框架内容 → 构造 PersonaConfig
        # 5. 日志记录加载结果

    def get(self, persona_id: str) -> Optional[PersonaConfig]:
        """Look up a persona by ID."""

    def list_personas(self) -> list[PersonaConfig]:
        """Return all registered personas."""
```

设计要点：
- **饥饿加载（Eager Loading）**：构造时即读取所有框架文件到内存。由于 19 个文件总共约 200KB，内存开销忽略不计
- **容错设计**：框架文件缺失时仅发出 `warning` 日志并跳过，不抛异常
- **可测试性**：`registry_path` 参数允许测试时注入自定义路径

---

## 3.4 registry.yaml 注册表

位置：`personas/registry.yaml`

```yaml
# 示例条目（完整文件包含 19 个条目）
personas:
  buffett:
    file: frameworks/buffett.md              # 相对于 personas/ 目录
    display_name: "巴菲特"
    description: "价值投资 / 护城河 / 安全边际 / 能力圈"
    max_tokens: 800
    applicable_domains: [finance, investment, business]
    upstream:
      repo: "https://github.com/will2025btc/buffett-perspective"
      license: "NONE"                        # 上游版权状态

  zhangxuefeng:
    file: frameworks/zhangxuefeng.md
    display_name: "张雪峰"
    description: "教育 / 职业规划 / 阶层流动 / 就业倒推法"
    max_tokens: 800
    applicable_domains: [education, career_planning]
    upstream:
      repo: "https://github.com/alchaincyf/zhangxuefeng-skill"
      license: "NONE"

  # ... 其余 17 个条目格式一致
```

设计要点：
- **声明式管理**：新增 persona 只需添加 YAML 条目 + 框架文件，不涉及 Python 代码变更
- **溯源追踪**：`upstream` 字段记录原始仓库 URL 和版权状态，为后续合规审计预留
- **按分类组织**：YAML 中以注释分隔四大类别（商业思维 / 教育社会 / AI 科技 / 内容传播）

---

## 3.5 PersonaMiddleware 中间件

位置：`agents/persona_middleware.py`

### 核心逻辑

```python
class PersonaMiddleware(AgentMiddleware):
    """Injects persona framework into Supervisor system prompt."""

    def __init__(self, *, persona: Optional[PersonaConfig] = None) -> None:
        super().__init__()
        self._persona = persona
        self._persona_block: str = self._build_persona_block()

    @property
    def active(self) -> bool:
        """Whether a persona is currently active."""
        return self._persona is not None
```

### 注入块结构

当 persona 激活时，`_build_persona_block()` 生成以下结构的文本块：

```markdown
## 🧠 Active Analysis Framework: 巴菲特

The user has explicitly requested analysis through the following cognitive
framework. You MUST:
1. Decompose tasks using the mental models and analysis dimensions defined below.
2. Instruct workers with task descriptions that reflect this framework's priorities.
3. Write the final report in the voice and style described below.
4. If a sub-topic falls outside this framework's scope, acknowledge it explicitly
   rather than forcing a fit.

---

[完整的上游 SKILL.md 内容]

---

> ⚠️ DISCLAIMER: This analysis uses an AI-simulated decision framework.
> It does not represent the views of any real individual and does not
> constitute professional advice.
```

### 关键实现细节

1. **注入块在构造时一次性生成**（`_build_persona_block()`），之后每次 LLM 调用时复用相同的块
2. **无 persona 时为 no-op**：`active` 属性返回 `False`，`wrap_model_call` 直接调用 `handler(request)` 透传
3. **使用 `append_to_system_message` 工具函数**：这是 `deepagents` 库提供的标准方法，与 `BudgetTrackingMiddleware` 使用相同的注入机制
4. **同时实现 sync 和 async hook**：`wrap_model_call` 和 `awrap_model_call` 逻辑完全对称

---

## 3.6 Supervisor Prompt Hook（§8）

位置：`agents/prompts.py` — `DeepAgentPrompts.SUPERVISOR`

在 SUPERVISOR prompt 的 §7（Resource Limits）之后，新增了 §8：

```
## 8. Analysis Framework Integration
If a `## 🧠 Active Analysis Framework` block appears at the end of this
system prompt, it means the user has explicitly requested analysis through
a specific cognitive framework (e.g., value investing, first-principles
engineering, career planning).

When an Analysis Framework is active, you MUST:
- **Prioritize the framework's mental models and dimensions** when
  decomposing the query into tasks.
- **Frame worker task descriptions** to reflect the framework's priorities
  and vocabulary.
- **Write the final report** in the voice, tone, and analytical style
  defined by the framework.
- **Acknowledge scope limits**: If a sub-topic falls outside the
  framework's expertise, explicitly state so rather than forcing a fit.

The framework does NOT override the Research Operations Handbook above —
it adds a lens on top of it. Budget rules, citation workflow, and
resource limits still apply unconditionally.
```

**设计原理**：
- §8 是**静态 hook**，无论是否有 persona 激活都存在于 prompt 中
- 它让 Supervisor 知道"如果在后面看到 🧠 Active Analysis Framework 块，那是预期的行为"
- 明确声明框架**不覆盖**操作手册（§1-§7），只是在其上叠加一个分析视角
- 预算规则（§7）和引用流程（§6）无条件生效

---

## 3.7 API 层改造

位置：`agents/agent.py`

### `build_deep_agent` 新增 `persona_id` 参数

```python
async def build_deep_agent(
    *,
    mcp_client: MCPSearchClient,
    checkpointer: Optional[Checkpointer] = None,
    persona_id: Optional[str] = None,        # ← 新增
    **overrides: Any,
) -> CompiledStateGraph:
```

内部逻辑：

```python
# 1. 若提供了 persona_id，从 PersonaRegistry 加载对应配置
persona_config = None
if persona_id:
    registry = PersonaRegistry()
    persona_config = registry.get(persona_id)
    if persona_config is None:
        logger.warning("Persona '%s' not found, running without persona", persona_id)
    else:
        logger.info("Persona activated: %s (%s)", persona_config.display_name, persona_id)

# 2. 创建 PersonaMiddleware（persona_config=None 时为 passthrough）
persona_middleware = PersonaMiddleware(persona=persona_config)

# 3. 注册到 middleware 列表（在 BudgetTrackingMiddleware 之前）
middleware=[
    persona_middleware,                                  # ← 新增
    budget_middleware,
    CitationDataMiddleware(max_retries=citation_max_retries),
]
```

### 调用链透传

`persona_id` 参数通过整个调用链透传：

```
run_deep_agent.py  (--persona buffett)
    ↓ args.persona
stream_deep_research(query, persona_id=args.persona)
    ↓ persona_id
build_deep_agent(mcp_client, persona_id=persona_id)
    ↓ PersonaRegistry().get(persona_id)
PersonaMiddleware(persona=persona_config)
    ↓ wrap_model_call
append_to_system_message(system_msg, persona_block)
```

同时，`run_deep_research`（一次性调用接口）也透传了 `persona_id`。

---

## 3.8 CLI 层改造

位置：`examples/run_deep_agent.py`

新增 `--persona` 参数：

```python
parser.add_argument(
    "--persona",
    type=str,
    default=None,
    help=(
        "Activate a persona analysis framework "
        "(e.g., 'buffett', 'zhangxuefeng', 'feynman')."
    ),
)
```

使用示例：

```bash
# 巴菲特视角分析比亚迪
python examples/run_deep_agent.py --persona buffett "比亚迪2025年财报分析"

# 张雪峰视角分析职业规划
python examples/run_deep_agent.py --persona zhangxuefeng "计算机考研还是就业？"

# 费曼视角解释物理概念
python examples/run_deep_agent.py --persona feynman "量子纠缠的通俗解释"

# 塔勒布视角分析风险
python examples/run_deep_agent.py --persona taleb "AI 行业的黑天鹅风险"

# 毛选视角分析地缘政治
python examples/run_deep_agent.py --persona maoxuan "中美贸易战的矛盾分析"

# 不使用 persona（默认行为，完全不受影响）
python examples/run_deep_agent.py "大模型应用未来的发展方向"
```

---

## 3.9 完整变更文件清单

| 操作 | 文件路径 | 说明 |
|------|----------|------|
| NEW | `personas/__init__.py` | 包初始化，导出 PersonaConfig, PersonaRegistry |
| NEW | `personas/registry.py` | PersonaRegistry 类 + PersonaConfig 数据类（113 行） |
| NEW | `personas/registry.yaml` | 声明式注册表（19 个人格，含上游溯源信息） |
| NEW | `personas/frameworks/*.md` | 19 个上游 SKILL.md 框架文件（7.7KB-34KB） |
| NEW | `agents/persona_middleware.py` | PersonaMiddleware 类（124 行） |
| MODIFY | `agents/prompts.py` | SUPERVISOR prompt 新增 §8 Analysis Framework Integration |
| MODIFY | `agents/agent.py` | `build_deep_agent` / `stream_deep_research` / `run_deep_research` 新增 `persona_id` 参数 |
| MODIFY | `examples/run_deep_agent.py` | 新增 `--persona` CLI 参数 |
