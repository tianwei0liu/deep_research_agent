# 第一章：背景研究与 Skills 清单

## 1.1 人格蒸馏（Persona Distillation）概念

**"人格蒸馏"**是 2025-2026 年 AI 开发者社区中兴起的一种技术范式。其核心思路是：从特定名人（或角色）的演讲、著作、采访、投资记录和决策历史中，利用大语言模型提取其核心"思维模式"（Mental Models）与"认知操作系统"，最终封装为可调用的 `.skill` 文件或提示词配置。

这类项目不仅是让 AI 模仿说话的语气，更重要的是复刻名人在遇到复杂问题时的**决策逻辑**和**分析框架**。例如：
- **巴菲特.skill**：不是教 AI 说"我喜欢买好公司"，而是让 AI 按照"护城河 → 安全边际 → 能力圈 → 管理层质量"的框架来分析任何企业
- **张雪峰.skill**：不是复制语气，而是注入"就业倒推法 → 行业天花板 → 城市选择 → 性价比"的职业规划决策树

### 与本项目的关系

`deep_research_agent` 是一个 Supervisor-Worker 架构的深度研究系统。其 Supervisor 通过 system prompt 中的 `DeepAgentPrompts.SUPERVISOR` 控制任务分解和报告撰写策略。将名人 Skills 注入 Supervisor，本质上是**用特定的认知框架来引导整个研究流程**——从任务分解到搜索策略到最终报告的叙事结构。

---

## 1.2 上游 GitHub 生态

### 三大索引仓库

我们从以下三个 GitHub 仓库中系统性地提取了所有可用 Skills：

| 仓库 | 类型 | Stars | 描述 |
|------|------|-------|------|
| [tmstack/awesome-persona-skills](https://github.com/tmstack/awesome-persona-skills) | 索引合集 | ~2.1k | 社区最大的 persona skills 合集，分类为"关于自己"、"关于身边人"、"关于偶像"、"关于圈子"等板块 |
| [misshiding/human-distillation-skills](https://github.com/misshiding/human-distillation-skills) | 索引合集 | ~159 | "一场关于人类的蒸馏实验"，补充了部分 tmstack 未收录的 skills |
| [alchaincyf](https://github.com/alchaincyf) (系列独立仓库) | 独立 skills | - | 张雪峰.skill 的原作者，产出了 10+ 个高质量名人 skill 仓库 |

### 现象级源头项目

引发这一轮"人格蒸馏"热潮的源头项目是 [alchaincyf/zhangxuefeng-skill](https://github.com/alchaincyf/zhangxuefeng-skill)，由"女娲.skill"生成。该项目试图还原张雪峰在高考志愿填报、考研及职业规划领域的实战思维框架。

---

## 1.3 完整 Skills 清单与来源标注

以下为从三个上游仓库中提取的**全部名人/角色 skills**。按"适合融入 deep_research_agent 做研究分析"和"不适合（纯情感/娱乐/关系类）"两大组分类。

### 1.3.1 适合融入 Deep Research Agent 的 Skills（19 个）

这些 skills 具备明确的**分析框架或决策模型**，能实质性地影响 Supervisor 的任务分解和报告视角。

#### 💼 商业思维类（11 个）

| persona_id | 名称 | 核心框架描述 | 上游仓库 | LICENSE 状态 |
|---|---|---|---|---|
| `buffett` | 巴菲特.skill | 价值投资 / 护城河 / 安全边际 / 能力圈 | [will2025btc/buffett-perspective](https://github.com/will2025btc/buffett-perspective) | ⚠️ 未声明 |
| `munger` | 芒格.skill | 多元思维模型 / 逆向思考 / 跨学科决策 | [alchaincyf/munger-skill](https://github.com/alchaincyf/munger-skill) | ⚠️ 未声明 |
| `jobs` | 乔布斯.skill | 产品直觉 / 极致设计 / 用户体验 / 现实扭曲力场 | [alchaincyf/steve-jobs-skill](https://github.com/alchaincyf/steve-jobs-skill) | ⚠️ 未声明 |
| `musk` | 马斯克.skill | 第一性原理 / 工程思维 / 成本优化 / 激进时间线 | [alchaincyf/elon-musk-skill](https://github.com/alchaincyf/elon-musk-skill) | ⚠️ 未声明 |
| `feynman` | 费曼.skill | 学习方法论 / 教学逻辑 / 科学思维 / 费曼技巧 | [alchaincyf/feynman-skill](https://github.com/alchaincyf/feynman-skill) | ⚠️ 未声明 |
| `naval` | 纳瓦尔.skill | 财富杠杆 / 判断力 / 人生哲学 | [alchaincyf/naval-skill](https://github.com/alchaincyf/naval-skill) | ⚠️ 未声明 |
| `taleb` | 塔勒布.skill | 风险管理 / 反脆弱 / 不确定性 / 黑天鹅理论 | [alchaincyf/taleb-skill](https://github.com/alchaincyf/taleb-skill) | ⚠️ 未声明 |
| `duan_yongping` | 段永平.skill | 买股票就是买公司 / 现金流折现 / 本分哲学 | [derrickgong87/duan-yongping-skill](https://github.com/derrickgong87/duan-yongping-skill) | ⚠️ 未声明 |
| `zhang_yiming` | 张一鸣.skill | 产品方法论 / 组织管理 / 全球化战略 / 人才观 | [alchaincyf/zhang-yiming-skill](https://github.com/alchaincyf/zhang-yiming-skill) | ⚠️ 未声明 |
| `trump` | 特朗普.skill | 谈判策略 / 权力运作 / 传播学 / 行为预判 | [alchaincyf/trump-skill](https://github.com/alchaincyf/trump-skill) | ⚠️ 未声明 |
| `paul_graham` | PG.skill | 创业方法论 / YC 投资哲学 / 技术写作 | [alchaincyf/paul-graham-skill](https://github.com/alchaincyf/paul-graham-skill) | ⚠️ 未声明 |

#### 🎓 教育 / 职业 / 社会分析类（3 个）

| persona_id | 名称 | 核心框架描述 | 上游仓库 | LICENSE 状态 |
|---|---|---|---|---|
| `zhangxuefeng` | 张雪峰.skill | 教育路径规划 / 职业发展 / 阶层流动 / 就业倒推法 | [alchaincyf/zhangxuefeng-skill](https://github.com/alchaincyf/zhangxuefeng-skill) | ⚠️ 未声明 |
| `zizek` | 齐泽克.skill | 意识形态批判 / 辩证法 / 反常识分析 | [JikunR/zizek-skill](https://github.com/JikunR/zizek-skill) | ⚠️ 未声明 |
| `maoxuan` | 毛选.skill | 矛盾分析法 / 调查研究方法 / 群众路线 / 策略制定 | [leezythu/maoxuan-skill](https://github.com/leezythu/maoxuan-skill) | ⚠️ 未声明 |

#### 🧪 AI / 科技思维类（2 个）

| persona_id | 名称 | 核心框架描述 | 上游仓库 | LICENSE 状态 |
|---|---|---|---|---|
| `karpathy` | Karpathy.skill | 深度学习方法论 / 教育传播 / 工程实践 | [alchaincyf/karpathy-skill](https://github.com/alchaincyf/karpathy-skill) | ⚠️ 未声明 |
| `ilya` | Ilya.skill | AGI 研究路径 / AI 安全 / 前沿方向 | [alchaincyf/ilya-sutskever-skill](https://github.com/alchaincyf/ilya-sutskever-skill) | ⚠️ 未声明 |

#### 🎙️ 内容创作 / 传播类（3 个）

| persona_id | 名称 | 核心框架描述 | 上游仓库 | LICENSE 状态 |
|---|---|---|---|---|
| `mrbeast` | MrBeast.skill | YouTube 增长黑客 / 内容创作 / 传播策略 | [alchaincyf/mrbeast-skill](https://github.com/alchaincyf/mrbeast-skill) | ⚠️ 未声明 |
| `guodegang` | 郭德纲.skill | 相声逻辑 / 叙事结构 / 幽默技巧 | [ByteRax/guodegang-skills](https://github.com/ByteRax/guodegang-skills) | ⚠️ 未声明 |
| `saul_goodman` | 风骚律师.skill | 谈判策略 / 说服技巧 / 法律思维角度 | [YeJe-cpu/saul-goodman-skill](https://github.com/YeJe-cpu/saul-goodman-skill) | ⚠️ 未声明 |

### 1.3.2 不融入 Deep Research Agent 的 Skills（记录备查）

以下 skills 偏向情感陪伴、关系模拟或纯工具功能，不适合作为研究分析框架注入 Supervisor。记录于此供后续参考。

- **职场类**（7 个）：同事.skill、老板.skill、导师.skill、大学老师.skill、师兄.skill、HR.skill、反蒸馏.skill
- **人际关系类**（10 个）：暗恋对象.skill、初恋.skill、前任.skill、现任.skill、她.skill、父母.skill、兄弟.skill、相亲.skill、恋爱训练营.skill、MamaSkill
- **自我成长类**（5 个）：自己.skill、数字人生.skill、永生.skill、Relic.skill、韭菜.skill
- **情感陪伴类**（3 个）：内娱.skill、重逢.skill、舔狗.skill
- **传统文化类**（6 个）：赛博算命.skill、月老.skill、佛教大师.skill、金刚经.skill、新青年.skill、永乐大典.skill
- **工具类**（13 个）：饕餮.skill、女娲.skill、X导师.skill、博主.skill、图鉴.skill、SBTI.skill、ContentWriter.skill、多比.skill、大案牍库.skill、万物皆可角色.skill、达尔文.skill、博主蒸馏器.skill、PPT.skill
- **网络名人类**（5 个）：峰哥.skill、童锦程.skill、户晨风.skill、凉兮.skill、卡兹克.skill

**筛选标准**：一个 skill 是否"适合融入"取决于它是否提供了**可迁移的分析框架或决策模型**，而非仅仅是角色扮演或情感交互。

---

## 1.4 版权状态总结

### 关键发现

**所有上游仓库均未声明开源许可证（LICENSE 文件缺失或未标注具体协议）**。

在 GitHub 默认版权规则下，未声明 LICENSE 的仓库保留所有权利（All Rights Reserved）。但鉴于本项目的使用场景：

- ✅ 个人学习和研究项目
- ✅ 未公开商业部署
- ✅ 不以名人姓名作为商业卖点

当前可以直接使用。`registry.yaml` 中保留 `upstream.license` 字段，为后续合规评估预留接口。

### 上游作者分布

大部分 skill 出自 [alchaincyf](https://github.com/alchaincyf) 这一作者（也是女娲.skill 和张雪峰.skill 的作者），其余分散在各独立仓库中：

| 作者/仓库 | 贡献的 Skills |
|-----------|-------------|
| alchaincyf | jobs, musk, feynman, munger, naval, taleb, zhang_yiming, trump, paul_graham, zhangxuefeng, karpathy, ilya, mrbeast (13 个) |
| will2025btc | buffett (1 个) |
| derrickgong87 | duan_yongping (1 个) |
| JikunR | zizek (1 个) |
| leezythu | maoxuan (1 个) |
| ByteRax | guodegang (1 个) |
| YeJe-cpu | saul_goodman (1 个) |

### 实际框架文件大小

所有 19 个框架文件均从上游仓库的 `SKILL.md` 文件直接下载，保留原版内容：

| persona_id | 文件大小 (bytes) | 估算 tokens |
|---|---|---|
| buffett | 8,125 | ~2,700 |
| munger | 11,613 | ~3,900 |
| jobs | 12,610 | ~4,200 |
| musk | 10,002 | ~3,300 |
| feynman | 10,243 | ~3,400 |
| naval | 11,360 | ~3,800 |
| taleb | 10,955 | ~3,600 |
| duan_yongping | 8,569 | ~2,900 |
| zhang_yiming | 10,286 | ~3,400 |
| trump | 12,153 | ~4,000 |
| paul_graham | 13,566 | ~4,500 |
| zhangxuefeng | 7,745 | ~2,600 |
| zizek | 3,682 | ~1,200 |
| maoxuan | 10,782 | ~3,600 |
| karpathy | 12,967 | ~4,300 |
| ilya | 12,723 | ~4,200 |
| mrbeast | 9,445 | ~3,100 |
| guodegang | 13,073 | ~4,400 |
| saul_goodman | 6,524 | ~2,200 |

> **注意**：使用原版意味着 Context Window 消耗较大（2K-4.5K tokens/persona）。Supervisor 原始 prompt 约 2.5K tokens，加上 persona 后总共约 5-7K tokens。对于 128K+ 上下文的模型（如 DeepSeek V4）来说，这个开销可以接受。
