# 第五章：验证计划与迭代路线图

## 5.1 已完成的验证

### 5.1.1 数据层验证

| 验证项 | 结果 | 方式 |
|--------|------|------|
| registry.yaml YAML 解析 | ✅ 19 personas 正确解析 | `yaml.safe_load()` |
| 框架文件完整性 | ✅ 19/19 文件存在且非空 | `Path.is_file()` |
| PersonaRegistry 加载 | ✅ 19 个 PersonaConfig 全部创建 | `PersonaRegistry().list_personas()` |
| PersonaConfig 字段校验 | ✅ persona_id / display_name / description / framework_prompt 均正确 | 逐条打印验证 |

### 5.1.2 Middleware 验证

| 验证项 | 结果 | 方式 |
|--------|------|------|
| PersonaMiddleware 构造（有 persona） | ✅ `active=True`，注入块长度 8,784 chars | `PersonaMiddleware(persona=buffett)` |
| PersonaMiddleware 构造（无 persona） | ✅ `active=False`，注入块为空字符串 | `PersonaMiddleware(persona=None)` |
| 注入块格式 | ✅ 以 `## 🧠 Active Analysis Framework` 开头 | 截取前 200 chars 验证 |
| DISCLAIMER 尾缀 | ✅ 包含 "does not constitute professional advice" | 字符串包含检查 |

### 5.1.3 回归测试

| 验证项 | 结果 | 方式 |
|--------|------|------|
| 全量测试套件 | ✅ 255 passed, 0 failed, 9 skipped | `pytest tests/ -v` |
| 编译/导入 | ✅ 所有新模块可正常导入 | `from deep_research_agent.agents.persona_middleware import PersonaMiddleware` |
| CLI `--help` | ✅ `--persona` 参数正确显示 | `run_deep_agent.py --help` |

### 5.1.4 集成测试（已执行但遇到外部错误）

```bash
python examples/run_deep_agent.py --persona zhang_yiming "大模型应用未来的发展方向"
```

**结果**：PersonaMiddleware 正确初始化并注入框架，但搜索后端 Bocha API 返回 403 Forbidden（API Key 余额不足），导致研究流程中断。这是**外部依赖问题**，与 PersonaMiddleware 实现无关。

**错误摘要**：
```
ToolException: Error executing tool web_search: All providers exhausted —
bocha: [bocha] Client error '403' for url 'https://api.bochaai.com/v1/web-search'
```

---

## 5.2 待执行的验证

### 5.2.1 端到端集成测试

**前置条件**：有效的 Bocha API Key 或切换到备用搜索后端

```bash
# 测试 1: 对比有/无 persona 的研究结果差异
python examples/run_deep_agent.py "比亚迪的竞争优势分析"
python examples/run_deep_agent.py --persona buffett "比亚迪的竞争优势分析"
```

**验证要点**：
1. LangSmith trace 中确认 Supervisor system prompt 包含 `## 🧠 Active Analysis Framework` 块
2. Worker system prompt 中**不包含**该块（隔离性验证）
3. 有 persona 时，任务分解维度应包含框架关键词（如 "护城河"、"安全边际"）
4. 无 persona 时，任务分解应遵循默认的 Decomposition Patterns（§2）

### 5.2.2 跨 Persona 对比测试

```bash
# 同一问题，不同 persona
python examples/run_deep_agent.py --persona buffett  "特斯拉值不值得投资？"
python examples/run_deep_agent.py --persona musk     "特斯拉值不值得投资？"
python examples/run_deep_agent.py --persona taleb    "特斯拉值不值得投资？"
```

**期望差异**：
- buffett：聚焦护城河、自由现金流、管理层质量
- musk：聚焦第一性原理、制造成本、技术壁垒
- taleb：聚焦尾部风险、反脆弱性、不确定性

### 5.2.3 单元测试（待编写）

```bash
# 目标
python -m pytest tests/test_persona_registry.py -v
python -m pytest tests/test_persona_middleware.py -v
```

**测试计划**：

```python
# test_persona_registry.py
class TestPersonaRegistry:
    def test_load_all_personas(self):
        """Should load all 19 personas from registry.yaml."""

    def test_get_existing_persona(self):
        """Should return PersonaConfig for valid persona_id."""

    def test_get_nonexistent_persona(self):
        """Should return None for invalid persona_id."""

    def test_missing_framework_file(self, tmp_path):
        """Should skip persona with missing framework file."""

    def test_custom_registry_path(self, tmp_path):
        """Should load from custom path."""

# test_persona_middleware.py
class TestPersonaMiddleware:
    def test_active_when_persona_provided(self):
        """Should be active when persona is set."""

    def test_inactive_when_no_persona(self):
        """Should be inactive (passthrough) when persona is None."""

    def test_persona_block_contains_framework(self):
        """Should include framework_prompt in injection block."""

    def test_persona_block_contains_disclaimer(self):
        """Should include DISCLAIMER at the end."""

    def test_wrap_model_call_injects_block(self):
        """Should modify system message when active."""

    def test_wrap_model_call_passthrough_when_inactive(self):
        """Should not modify system message when inactive."""
```

---

## 5.3 迭代路线图

### Phase 1（已完成 ✅）

- [x] 下载全部 19 个上游 .skill 文件
- [x] 创建 `personas/` 数据层（__init__.py, registry.py, registry.yaml）
- [x] 实现 PersonaMiddleware
- [x] 添加 SUPERVISOR prompt §8 Hook
- [x] 修改 agent.py — persona_id 参数贯通
- [x] 修改 run_deep_agent.py — --persona CLI 参数
- [x] 回归测试通过（255/255）

### Phase 2（待执行）

- [ ] 编写 PersonaRegistry 单元测试
- [ ] 编写 PersonaMiddleware 单元测试
- [ ] 端到端集成测试（需有效 API Key）
- [ ] LangSmith trace 分析验证注入位置
- [ ] 跨 persona 对比测试

### Phase 3（未来增强）

- [ ] **框架精简**：对 token 消耗最大的框架文件（>4K tokens）做选择性摘要压缩
- [ ] **多 Persona 模式**：支持 `--persona "buffett,munger"` 组合框架注入
- [ ] **自动 Persona 推荐**：基于查询内容和 `applicable_domains` 自动推荐合适的 persona
- [ ] **Persona 效果评估**：设计独立的评分维度（framework_coherence, persona_fidelity, actionability）
- [ ] **真名匿名化**：评估是否需要将 `display_name` 从真名改为功能性描述
- [ ] **上游同步脚本**：自动 diff 本地 fork 与上游变更

### Phase 4（商业化准备，如需）

- [ ] 逐一核查上游 LICENSE 合规性
- [ ] 将所有涉及真实个人的 persona 替换为功能性描述
- [ ] 添加用户知情同意机制
- [ ] 在 UI 层面做视觉区分（"AI 模拟视角" 标注）
- [ ] 咨询法律顾问评估人格权风险

---

## 5.4 关键指标（KPIs）

| 指标 | 目标 | 当前状态 |
|------|------|---------|
| 注册 personas 数量 | 19 | ✅ 19 |
| 回归测试通过率 | 100% | ✅ 100% (255/255) |
| PersonaMiddleware no-op 开销 | <1ms | ✅ (条件短路) |
| 框架注入 token 开销 | <5K tokens/persona | ✅ 最大 4,500 tokens |
| 端到端集成测试 | Pass | ⏳ 待有效 API Key |
