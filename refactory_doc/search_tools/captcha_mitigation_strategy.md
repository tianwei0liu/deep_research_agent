# SearXNG CAPTCHA Suspension 缓解策略

> **文档状态**: Draft  
> **日期**: 2026-05-04  
> **关联问题**: 搜索引擎因 CAPTCHA 触发导致 SearXNG 引擎被 Suspended，MCP 搜索服务退化为 Bing-only 模式  

---

## §1. 问题描述

### 1.1 现象

SearXNG 实例在正常使用一段时间后，搜索引擎（baidu、sogou、startpage）被标记为 `Suspended: CAPTCHA`，停止响应搜索请求。此时仅 Bing 引擎存活，但 Bing 对中文查询的质量极差（P@5 可低至 0%），导致整个搜索服务实质性不可用。

```
$ curl "http://localhost:8080/search?q=test&format=json" | python -c "..."
=== Unresponsive Engines ===
  ['baidu', 'Suspended: CAPTCHA']
  ['sogou', 'Suspended: CAPTCHA']
  ['startpage', 'Suspended: CAPTCHA']
  ['karmasearch', 'Suspended: access denied']
=== Active Engines ===
  {'bing'}     ← 唯一存活引擎，中文查询质量灾难
```

### 1.2 影响范围

| 影响 | 严重性 | 描述 |
|:---|:---:|:---|
| 中文搜索质量 | **P0** | Bing 中文查询返回大量不相关结果（Microsoft 支持页、酒店广告等） |
| Deep Research 报告质量 | **P0** | Agent 基于垃圾搜索结果生成报告，内容完全偏离主题 |
| 搜索 QA 验证 | **P1** | `search_quality_report.py` 因引擎不可用无法运行，阻塞迭代 |
| 用户体验 | **P0** | 在生产环境中，用户发起的任何查询都可能得到低质量结果 |

### 1.3 根因链

```
单一 IP 出口（Docker 容器 → 宿主机 → 公网）
    ↓
SearXNG 向搜索引擎发送自动化请求
    ↓
搜索引擎检测到异常流量模式（无浏览器指纹、请求频率集中）
    ↓
返回 CAPTCHA Challenge
    ↓
SearXNG 捕获 SearxEngineCaptchaException
    ↓
引擎被标记为 Suspended，封禁 suspended_times 秒
    ↓
封禁期间所有该引擎的请求直接跳过（不尝试）
```

---

## §2. SearXNG Suspension 机制

### 2.1 默认配置（极度保守）

SearXNG 的默认 `suspended_times` 配置如下：

```yaml
search:
  suspended_times:
    SearxEngineCaptcha: 86400            # 24 小时
    SearxEngineAccessDenied: 86400       # 24 小时
    SearxEngineTooManyRequests: 3600     # 1 小时
    cf_SearxEngineCaptcha: 1296000       # 15 天
    cf_SearxEngineAccessDenied: 86400    # 24 小时
    recaptcha_SearxEngineCaptcha: 604800 # 7 天
```

> [!CAUTION]
> **一次 CAPTCHA 触发 → 引擎 24 小时不可用**。对于 Cloudflare CAPTCHA 甚至长达 15 天。这是为公共实例（避免 IP 被永久拉黑）设计的，对我们的**私有低流量实例**过于激进。

### 2.2 Suspension 生命周期

```
时间线 →
│
├─ T=0: 引擎返回 CAPTCHA
├─ T=0: SearXNG 记录 suspend_end_time = now + suspended_times
├─ T=0 ~ T=suspend_end_time: 引擎被跳过，不发送任何请求
├─ T=suspend_end_time: 引擎自动恢复，下次查询时重新尝试
│
└─ 如果恢复后再次 CAPTCHA → 重新进入 Suspended 状态
```

**关键发现**：重启 SearXNG 容器会**清除所有 suspension 状态**（因为状态存储在内存中，不持久化）。这就是为什么 `docker compose restart searxng` 能临时恢复 baidu 引擎。

---

## §3. 解决方案

### 方案概览

| 层级 | 方案 | 分类 | 成本 | 实施时间 | 预期效果 |
|:---:|:---|:---:|:---:|:---:|:---|
| **L1** | 降低 `suspended_times` | 短期 | 0 | 5min | 封禁窗口 24h→10min，CAPTCHA 间歇性场景恢复率 ~80% |
| **L1.5** | 应用层请求节流 | 短期 | 0 | 1-2h | 降低 CAPTCHA 触发概率 ~50%，从源头减少引擎压力 |
| **L1.7** | SearXNG Limiter + Valkey | 短期 | 0 | 1-2h | 自动拦截突发流量，降低引擎触发 CAPTCHA 概率 ~30% |
| **L2** | 出口代理轮转 | **长期/生产** | ¥50-200/月 | 1-2h | **根治** IP 封禁，高并发场景可用，CAPTCHA 率→~0% |
| **L3** | 多 SearXNG 实例 | **长期/生产** | +256MB RAM | 2-4h | 实例级 failover，单实例 CAPTCHA 不影响服务，可用性 +40% |
| **L4** | 付费搜索 API fallback | **长期/生产** | ¥200+/月 | 0.5-1d | 终极兜底，SLA 可达 99.9%，SearXNG 全挂仍可服务 |

---

### L1：降低封禁时间（立即可做）

> **分类**：短期应急 | **适用场景**：低流量私有实例（QPS < 1）

#### 变更

文件：`docker/searxng/settings.yml`

```yaml
search:
  # ... existing config ...
  suspended_times:
    # 私有实例：低流量场景，aggressive retry 风险可控
    SearxEngineCaptcha: 600              # 10 分钟（默认 24h）
    SearxEngineAccessDenied: 600         # 10 分钟（默认 24h）
    SearxEngineTooManyRequests: 300      # 5 分钟（默认 1h）
    cf_SearxEngineCaptcha: 3600          # 1 小时（默认 15 天）
    cf_SearxEngineAccessDenied: 600      # 10 分钟（默认 24h）
    recaptcha_SearxEngineCaptcha: 3600   # 1 小时（默认 7 天）
```

#### 预期效果

| 指标 | 变更前 | 变更后 |
|:---|:---|:---|
| CAPTCHA 封禁窗口 | 24 小时 | 10 分钟 |
| Cloudflare 封禁窗口 | 15 天 | 1 小时 |
| 间歇性 CAPTCHA 恢复率 | ~0%（需手动重启） | ~80%（10min 自动恢复） |
| 持续性 CAPTCHA 场景 | 无效 | 无效（需升级 L2） |

#### 执行步骤

```bash
# 1. 编辑 settings.yml，添加上述 suspended_times 配置
vim docker/searxng/settings.yml

# 2. 重启容器使配置生效
cd docker && docker compose restart searxng

# 3. 验证配置已加载（检查日志无 YAML 报错）
docker compose logs searxng --tail 20

# 4. 验证引擎状态
curl -s "http://localhost:8080/search?q=test&format=json" | \
  python -c "import sys,json; d=json.load(sys.stdin); print([e for e in d.get('unresponsive_engines',[])])"
```

#### Trade-off

> [!WARNING]
> **风险**：缩短封禁时间意味着 SearXNG 更频繁地重试被 CAPTCHA 的引擎，可能导致：
> 1. 搜索引擎将该 IP 标记为持续性爬虫
> 2. IP 被永久拉黑（非临时 CAPTCHA，而是 403 Forbidden）
>
> **缓解**：我们是私有实例，日均请求量 < 100 次，这个风险在低流量场景下可以接受。如果被永久封禁，则需要升级到 L2（代理轮转）。

---

### L1.5：应用层请求节流（新增方案）

> **分类**：短期应急 | **适用场景**：所有场景（与 L1 互补）

#### 原理

SearXNG 本身不支持对下游引擎的出站请求限速。我们在 `SearchRouter` 层引入 **Token Bucket / 最小间隔** 限流器，确保对 SearXNG 的请求不会在短时间内密集发出。Deep Research Agent 的 Worker 并发调用搜索时，多个查询可能在 <1s 内全部到达 SearXNG，导致 SearXNG 同时向百度/搜狗发起大量请求，触发 CAPTCHA。

#### 变更

文件：`search_service/backends/base.py` — 在 `SearchRouter.search()` 中添加最小间隔控制。

```python
import asyncio
import time

class SearchRouter:
    MIN_REQUEST_INTERVAL: float = 2.0  # 两次搜索最小间隔(秒)

    def __init__(self, ...) -> None:
        # ... existing init ...
        self._last_request_time: float = 0.0
        self._throttle_lock = asyncio.Lock()

    async def search(self, query: str, **kwargs) -> SearchResponse:
        # 节流：确保请求间隔 >= MIN_REQUEST_INTERVAL
        async with self._throttle_lock:
            now = time.monotonic()
            wait = self.MIN_REQUEST_INTERVAL - (now - self._last_request_time)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_time = time.monotonic()

        # ... existing cache-aside + fallback logic ...
```

#### 预期效果

| 指标 | 效果 |
|:---|:---|
| CAPTCHA 触发概率 | 降低 ~50%（消除突发流量峰值） |
| 搜索延迟增加 | 平均 +1-2s（可接受，Deep Research 本身耗时数分钟） |
| 高并发场景 | 请求排队，不丢弃，保证最终执行 |

#### 执行步骤

1. 在 `SearchRouter.__init__` 中添加 `_last_request_time` 和 `_throttle_lock`
2. 在 `SearchRouter.search()` 方法头部插入节流逻辑
3. 添加单元测试验证并发调用时请求间隔 ≥ 2s
4. 运行 `python -m pytest tests/test_search_service/test_search_router.py -v`

#### Trade-off

- ✅ 零成本，纯代码改动
- ✅ 从源头降低 CAPTCHA 触发率
- ⚠️ 增加搜索延迟（但对 Deep Research 可接受）
- ⚠️ 高并发场景下排队可能导致超时（需配合 L3 多实例解决）

---

### L1.7：SearXNG Limiter + Valkey（新增方案）

> **分类**：短期加固 | **适用场景**：多用户 / API 暴露场景

#### 原理

SearXNG 内置 Limiter 模块，基于 Valkey（Redis 分支）存储请求计数，自动检测并拦截异常流量模式（如同一 IP 短时间内大量请求）。虽然 Limiter 主要设计用于公共实例防 bot，但在私有实例中，它可以防止 Agent 并发调用时产生的突发流量穿透到下游引擎。

#### 变更

文件：`docker/docker-compose.yml`

```yaml
services:
  valkey:
    image: valkey/valkey:8-alpine
    container_name: search-valkey
    mem_limit: 128m
    restart: unless-stopped

  searxng:
    # ... existing config ...
    depends_on:
      - valkey
```

文件：`docker/searxng/settings.yml`

```yaml
server:
  secret_key: "change-this-in-production"
  limiter: true          # 启用 Limiter
  image_proxy: false

redis:
  url: "redis://valkey:6379/0"
```

文件（新建）：`docker/searxng/limiter.toml`

```toml
[botdetection.ip_limit]
link_token = true
# 每 IP 每分钟最多 30 次请求
burst_limit = 15
burst_window = 60
```

#### 预期效果

| 指标 | 效果 |
|:---|:---|
| 突发流量拦截 | 自动拦截超出阈值的请求，降低 CAPTCHA 触发 ~30% |
| 与 L1.5 叠加 | L1.5 在应用层限流 + L1.7 在 SearXNG 层限流，双重保护 |
| 生产就绪度 | 中等 — 适合作为 L2 之前的过渡方案 |

#### 执行步骤

1. `docker-compose.yml` 添加 `valkey` 服务
2. `settings.yml` 添加 `server.limiter: true` 和 `redis.url`
3. 创建 `docker/searxng/limiter.toml`，mount 到容器
4. `docker compose up -d` 重建服务
5. 验证：`docker compose logs searxng | grep limiter` 确认 Limiter 已启动

#### Trade-off

- ✅ SearXNG 原生功能，无代码改动
- ✅ 额外增加 Valkey 仅需 ~50MB 内存
- ⚠️ 配置复杂度略高（需调优 `limiter.toml`）
- ⚠️ Limiter 拒绝的请求返回 429，需确保 SearchRouter 能正确处理

---

### L2：出口代理轮转（推荐方案）

> **分类**：**长期/生产级** | **适用场景**：高并发生产环境，多用户场景

#### 原理

SearXNG 原生支持 `outgoing.proxies` 配置。配置多个代理后，SearXNG 会自动进行 round-robin 轮转，每次请求使用不同的出口 IP，有效避免单一 IP 被搜索引擎标记。

#### 配置

文件：`docker/searxng/settings.yml`

```yaml
outgoing:
  request_timeout: 8.0
  max_request_timeout: 12.0

  # 代理轮转池 — SearXNG 自动 round-robin
  proxies:
    all://:
      - socks5://user:pass@proxy1:1080
      - socks5://user:pass@proxy2:1080
      - socks5://user:pass@proxy3:1080

  # 或者使用 HTTP 代理
  # proxies:
  #   http:
  #     - http://proxy1:8080
  #     - http://proxy2:8080
  #   https:
  #     - http://proxy1:8080
  #     - http://proxy2:8080

  # 可选：如果有多个网卡/IP
  # source_ips:
  #   - 192.168.0.1
  #   - 192.168.0.2
```

#### 代理选型

| 代理类型 | 成本 | 效果 | 适用场景 |
|:---|:---|:---|:---|
| **住宅代理**（如快代理、芝麻代理） | ¥50-200/月 | 最佳 — IP 信誉高 | 推荐 |
| **数据中心代理**（如 VPS 出口） | ¥30-100/月 | 中等 — IP 可能被标记 | 预算有限 |
| **免费代理** | 0 | 差 — 不稳定且 IP 信誉极低 | 不推荐 |

#### 按引擎配置代理

SearXNG 支持为特定引擎配置独立代理，可以为高风险引擎（baidu、sogou）配置代理，低风险引擎（bing）直连：

```yaml
engines:
  - name: baidu
    engine: baidu
    proxies:
      all://:
        - socks5://user:pass@china-proxy:1080
    # ...

  - name: bing
    engine: bing
    # 不配置 proxies → 使用直连
    # ...
```

#### 预期效果

| 指标 | 效果 |
|:---|:---|
| CAPTCHA 触发率 | **→ ~0%**（每次请求不同 IP，无法触发频率检测） |
| 高并发支持 | ✅ 天然支持 — 请求分散到不同 IP |
| IP 永久封禁风险 | 极低 — 住宅 IP 信誉高 |
| 百度/搜狗恢复率 | ~99%（住宅代理）/ ~80%（数据中心代理） |

#### 执行步骤

1. **选择代理服务商**：推荐快代理或芝麻代理（国内住宅 IP），注册并购买最低档套餐
2. 获取 SOCKS5/HTTP 代理地址（至少 3 个出口）
3. 编辑 `docker/searxng/settings.yml`，在 `outgoing` 下添加 `proxies` 配置
4. 高风险引擎（baidu, sogou）配置专用代理，低风险引擎（bing）保持直连
5. `docker compose restart searxng`
6. 验证：连续执行 10 次搜索，确认无 CAPTCHA suspension

#### Trade-off

- ✅ 根治 CAPTCHA 封禁问题
- ✅ SearXNG 原生支持，无需代码改动
- ⚠️ 增加外部依赖（代理服务商的稳定性）
- ⚠️ 有月度成本
- ⚠️ 住宅代理通常有带宽/流量限制

---

### L3：多 SearXNG 实例

> **分类**：**长期/生产级** | **适用场景**：需要 HA SLA 的生产环境

#### 原理

在 Docker Compose 中运行 2+ 个独立的 SearXNG 实例。由于每个实例的 suspension 状态是独立的内存存储，当 Instance A 的 baidu 被 suspended 时，Instance B 的 baidu 可能仍然可用。

#### 配置

文件：`docker/docker-compose.yml`

```yaml
services:
  searxng-primary:
    image: searxng/searxng:latest
    container_name: searxng-primary
    ports:
      - "8080:8080"
    volumes:
      - ./searxng/settings.yml:/etc/searxng/settings.yml:ro
    mem_limit: 512m
    restart: unless-stopped

  searxng-secondary:
    image: searxng/searxng:latest
    container_name: searxng-secondary
    ports:
      - "8081:8080"
    volumes:
      - ./searxng/settings.yml:/etc/searxng/settings.yml:ro
    mem_limit: 512m
    restart: unless-stopped
```

#### 代码变更

需要在 `agents/tools.py` 的 `make_internet_search` 中注册两个 `SearXNGClient`，让 `SearchRouter` 实现真正的 backend-level fallback：

```python
def make_internet_search(config):
    from search_service.backends.searxng_client import SearXNGClient
    from search_service.backends.base import SearchRouter
    from search_service.cache import NullCache

    # 多实例 fallback
    primary = SearXNGClient(config)  # :8080
    secondary = SearXNGClient(config, base_url="http://localhost:8081")  # :8081

    router = SearchRouter(
        backends=[primary, secondary],  # primary 优先，失败时 fallback 到 secondary
        cache=NullCache(),
    )
    # ...
```

#### 预期效果

| 指标 | 效果 |
|:---|:---|
| 服务可用性 | 单实例 CAPTCHA 不影响服务，可用性提升 ~40% |
| CAPTCHA 影响范围 | 从“全局不可用”降级为“单实例降级” |
| 与 L2 组合 | 每实例独立代理 → 真正的 IP 隔离，可用性接近 99.9% |

#### 执行步骤

1. `docker-compose.yml` 添加 `searxng-secondary` 服务（端口 8081）
2. `SearXNGClient.__init__` 添加可选 `base_url` 参数支持
3. `agents/tools.py` 创建两个 `SearXNGClient`，注册到 `SearchRouter.backends`
4. 添加集成测试验证 primary 失败时自动 fallback 到 secondary
5. `docker compose up -d` 启动双实例

#### Trade-off

- ✅ 实例级容错 — 一个实例的 suspension 不影响另一个
- ✅ 无外部依赖
- ⚠️ 增加内存开销（每个实例 ~256-512MB）
- ⚠️ 两个实例共享同一出口 IP → 如果 IP 被永久封禁，两个实例同时失效
- ⚠️ 需要代码改动（SearchRouter 支持多 backend）

> [!IMPORTANT]
> L3 解决的是 **suspension 状态隔离** 问题，不解决根本的 **IP 信誉** 问题。如果搜索引擎对该 IP 持续封禁，多实例无济于事。L3 最适合与 L2（代理轮转）搭配使用。

---

### L4：付费搜索 API Fallback

> **分类**：**长期/生产级** | **适用场景**：需要 100% 可用性的生产环境

#### 原理

当 SearXNG 所有引擎都 suspended 时（`AllProvidersExhaustedError`），自动降级到付费搜索 API。这是最昂贵但最可靠的兜底方案。

#### 候选 API

| API | 中文支持 | 价格 | 特点 |
|:---|:---:|:---|:---|
| **Bocha（博查）** | ✅ 优秀 | ¥0.01/次 | 国内服务，中文搜索质量高 |
| **Serper** | ⚠️ 一般 | $0.001/次 | Google 结果，需翻墙 |
| **SerpAPI** | ⚠️ 一般 | $50/月起 | 多引擎支持 |
| **Tavily**（已弃用） | ⚠️ 一般 | $5/月起 | 我们之前使用的方案 |

#### 架构

```
SearchRouter
  ├─ Backend 1: SearXNGClient (primary)      ← 免费，优先使用
  ├─ Backend 2: SearXNGClient (secondary)    ← 免费，SearXNG fallback
  └─ Backend 3: PaidSearchClient (fallback)  ← 付费，终极兜底
```

#### 预期效果

| 指标 | 效果 |
|:---|:---|
| 服务 SLA | 可达 **99.9%** — SearXNG 全挂仍可服务 |
| 搜索质量 | 博查 AI Reranking 质量优于原始 SERP |
| 成本可控性 | 仅在 SearXNG 失败时消耗付费额度，正常场景成本接近 0 |

#### 执行步骤

1. 注册博查开放平台，获取 API Key
2. 实现 `BochaSearchClient`，实现 `SearchBackend` 协议
3. 配置类添加 `bocha_api_key` 和 `bocha_base_url`
4. `SearchRouter` 的 `backends` 链尾部添加 `BochaSearchClient`
5. 添加单元测试和集成测试
6. 端到端验证：关闭 SearXNG，确认 fallback 到博查正常工作

#### Trade-off

- ✅ 终极可靠性保证 — SearXNG 全挂也不影响服务
- ✅ SearchRouter 已有 fallback 链机制，代码改动小
- ⚠️ 持续成本
- ⚠️ 引入外部 API 依赖
- ⚠️ 每个 API 的结果格式不同，需要适配层

---

## §4. 推荐实施路径

```
短期应急（0 成本）                        长期生产级
  │                                        │
  ├─ L1: 降低 suspended_times（5min）       │
  ├─ L1.5: 应用层请求节流（1-2h）           │
  ├─ L1.7: Limiter + Valkey（1-2h）         │
  │                                        │
  └─ 短期组合足以覆盖低流量场景 ─────────── 评估长期方案
                                           │
                            高并发/多用户 → L2（代理轮转）
                            需要 HA SLA  → L2 + L3（多实例）
                            需要 100% 可用 → L2 + L3 + L4
```

### 分阶段执行

1. **立即（Phase 2.1）**: 应用 L1 + L1.5，零成本缓解
2. **本周（Phase 2.1b）**: 部署 L1.7（Valkey + Limiter），加固防线
3. **短期（Phase 2.2）**: 评估住宅代理服务商，配置 L2 代理轮转
4. **中期（Phase 2.3）**: 部署 L3 多实例 + L4 博查 fallback，达成生产级 SLA

---

## §5. 临时恢复手段

在正式方案落地前，可以用以下方式临时恢复服务：

### 方法 1：重启容器（清除 suspension 状态）

```bash
cd docker && docker compose restart searxng
```

**原理**：suspension 状态存储在内存中，重启后清零。所有引擎恢复到初始可用状态。

### 方法 2：设置 suspended_times 为 0（禁用 suspension）

```yaml
search:
  suspended_times:
    SearxEngineCaptcha: 0        # 禁用 CAPTCHA suspension
    SearxEngineAccessDenied: 0   # 禁用 Access Denied suspension
```

> [!CAUTION]
> 将 `suspended_times` 设为 0 会**完全禁用 suspension 机制**。SearXNG 每次查询都会尝试所有引擎，即使它们持续返回 CAPTCHA。这会增加被永久封禁的风险，仅建议在调试时使用。

### 方法 3：定时重启（crontab）

```bash
# 每 6 小时重启一次 SearXNG，清除 suspension 状态
0 */6 * * * cd /home/tianwei/workspace/deep_research_agent/docker && docker compose restart searxng
```

这是 L1 的补充手段，确保即使封禁时间设为 10 分钟，也有定期的全面重置。

---

## §6. 架构风险评估与改进建议 (Architectural Risk Assessment)

> **⚠️ 架构师审查批注**: 在实施上述策略时，需特别注意以下由于系统级并发和网络特性引入的潜在风险。

### 6.1 核心风险点

#### 🚨 风险点一：L1.5 应用层全局锁导致的并发灾难（P0 级风险）
* **风险描述**: 在 `SearchRouter` 中引入 `asyncio.Lock()` 并使用固定间隔（如 2.0s）。这是一个**全局锁**。当 Agent 多个 Worker 并发发起搜索时，所有请求将被严格串行化。
* **影响**:
  * **长尾延迟骤增**: 第 $N$ 个并发请求将被迫等待 $(N-1) \times 2$ 秒，导致大面积 Timeout。
  * **引擎无差别限速**: 针对 Baidu 的限流会无辜阻塞对 Bing 的查询（False Sharing），导致全局吞吐量暴跌。

#### 🚨 风险点二：L3 多实例容错的“伪命题”（P1 级风险）
* **风险描述**: 方案假设“Instance A 的 baidu 被 suspended 时，Instance B 仍可用”。但搜索引擎封禁是基于 **出口 IP (Client IP)** 的，而非容器实例。
* **影响**: 如果不配合 L2（代理轮转），多个实例共享同一个宿主机出口 IP。一旦该 IP 触发 CAPTCHA，所有实例将同时遭遇封禁（雪崩效应）。这不仅无法提供 HA，还增加了无效的内存和运维成本。

#### 🚨 风险点三：粗暴的 Cron 定时重启（P1 级风险）
* **风险描述**: 方案 §5 推荐使用 `crontab` 定时 `docker compose restart searxng` 来清除状态。这是极端的**反模式（Anti-pattern）**。
* **影响**: Docker 重启并非优雅停机（Graceful Shutdown）。重启瞬间正在处理和排队中的所有 HTTP 请求会被强行阻断（Connection Reset by Peer），直接导致处于搜索阶段的 Agent 异常崩溃。

#### ⚠️ 风险点四：L1.7 与 L2 引入的新故障域（P2 级风险）
* **风险描述**: 部署 Limiter(L1.7) 会产生 HTTP 429 响应；部署代理(L2) 会频繁产生 502/504 或 Timeout 等代理层异常。
* **影响**: 若 `SearchRouter` 未针对这些新增 Error Code 建立带指数退避的重试机制，Agent 将因生硬的失败断言而中断任务。

#### ⚠️ 风险点五：L1 激进重试导致“永久拉黑”（P2 级风险）
* **风险描述**: 将 `suspended_times` 粗暴缩短到 10 分钟。
* **影响**: 在缺乏 IP 轮转时，短时间内反复向已触发 CAPTCHA 的引擎撞墙，极易触发搜索引擎风控升级，导致 IP 被永久 403 封禁。

### 6.2 架构师改进建议 (Actionable Next Steps)

为了确保本架构满足 **Production-Ready** 标准，建议执行以下修正：

1. **重构 L1.5 为单引擎级别限速 (Per-Engine Token Bucket)**
   * 废弃全局 `asyncio.Lock()`。改用专用的异步限流库（如 `aiolimiter`）实现细粒度的单引擎并发控制（例如限制 `baidu` 1 req/2s，但不对 `bing` 限速）。
2. **强制解绑 L3 无代理部署 (Network Isolation)**
   * 明确声明：**L3 (多实例) 必须作为 L2 的子集部署**。若没有多出口 IP/代理池的支撑，坚决不部署多实例。
3. **强化 SearchRouter 容错层 (Exponential Backoff with Jitter)**
   * 引入 `tenacity` 库。在捕获代理异常（ProxyError）和限流响应（429）时，采用带抖动的指数退避策略进行重试，掩盖网络波动。
4. **废除定时重启策略**
   * 从实施计划中彻底删除 Crontab 重启操作。如确需重置状态，应依赖合理的 `suspended_times` 配置或重启代理隧道，坚决保证应用层连接不断。

---

## §7. §6 风险评估的同行审查 (Peer Review of §6 Risk Assessment)

> **审查人**: Staff Engineer / Principal Architect  
> **审查日期**: 2026-05-05  
> **审查结论**: §6 的五个风险点**整体合理且有价值**，但存在部分过度定性、优先级偏差和遗漏关键风险的问题。以下逐条评估。

### 7.1 风险点逐条评估

#### ✅ 风险点一（L1.5 全局锁并发灾难）— 合理，**但 P0 定级过高**

**§6 判断**：`asyncio.Lock()` 会将所有并发搜索请求严格串行化，导致长尾延迟骤增。

**审查意见**：
- **技术描述准确**：全局锁确实会导致 $O(N)$ 排队延迟。当 5 个 Worker 并发搜索时，第 5 个请求等待 ≥ 8s，这在 Deep Research（本身耗时数分钟）场景下**有影响但不致命**。
- **P0 定级偏高**：应降级为 **P1**。原因：
  1. 当前系统 `max_parallel_workers` 默认为 3，最大并发搜索请求数有限
  2. Deep Research 的单次任务耗时在 3-10 分钟量级，额外 4-6s 搜索排队延迟占比 < 3%
  3. 真正的 P0 是"搜索完全不可用"（如 CAPTCHA 导致所有引擎 suspended），而非"搜索略慢"
- **Per-Engine Token Bucket 建议合理**：这是正确的改进方向，但 §6 未考虑实现成本。引入 `aiolimiter` 需要为每个引擎维护独立的令牌桶状态，增加了系统复杂度。**对于当前阶段（低流量私有实例），全局锁 + 适当缩短间隔（如 1.0s）已经是足够的 MVP**。

> [!TIP]
> **建议**：保留全局锁作为 v1 实现，将 `MIN_REQUEST_INTERVAL` 从 2.0s 降至 1.0s。Per-Engine Token Bucket 作为 v2 演进方向，当实际监控到引擎级别的无差别限速问题（False Sharing）时再实施。过早优化是万恶之源。

---

#### ✅ 风险点二（L3 多实例"伪命题"）— **完全合理，最有价值的风险识别**

**§6 判断**：多实例共享出口 IP，CAPTCHA 封禁是 IP 级别的，多实例无法提供真正的 HA。

**审查意见**：
- **这是 §6 中最精准的风险判断**。原文 §3 的 L3 方案确实隐含了一个未经验证的假设——"不同实例的 suspension 状态独立 → HA"。但搜索引擎的封禁决策发生在**它们的服务端**，基于 Client IP 而非 SearXNG 实例 ID。
- **"L3 必须作为 L2 的子集部署" 这一建议完全正确**，应直接写入 §4 的实施路径中作为硬性前置条件。
- **补充**：L3 在没有 L2 支撑时，唯一有价值的场景是**应对 SearXNG 进程级故障**（如 OOM、容器崩溃），而非 CAPTCHA。如果只是为了进程级容错，Docker `restart: unless-stopped` 已经足够，不需要多实例。

---

#### ⚠️ 风险点三（Cron 定时重启反模式）— 合理，但**"废除"的建议过于绝对**

**§6 判断**：`docker compose restart` 是极端反模式，会导致在途请求被强行中断。

**审查意见**：
- **风险描述准确**：非优雅停机确实会中断在途 HTTP 请求，可能导致 Agent Worker 的 `SearchRouter.search()` 调用抛出 `ConnectionResetError`。
- **但"彻底废除"的建议过于激进**：
  1. 如果 `SearchRouter` 已经有 fallback + retry 机制（它应该有），那么 SearXNG 重启期间的短暂不可用（< 5s）可以被容错层吸收
  2. 在低流量场景下（夜间无用户使用时段），定时重启是一个**成本为零的实用手段**
  3. 更合理的建议是：**将 Cron 重启限制在低谷时段（如凌晨 3:00），而非"每 6 小时"，并确保 SearchRouter 的 retry 机制能覆盖短暂中断**
- **替代建议**：如果要彻底消除定时重启，应提供一个 **SearXNG Admin API 调用来清除特定引擎的 suspension 状态**（如果 SearXNG 支持），或实现一个轻量级的 health check + 条件重启脚本。

---

#### ✅ 风险点四（L1.7/L2 新故障域）— 合理，但缺乏可操作细节

**§6 判断**：429/502/504 等新增 Error Code 需要指数退避重试。

**审查意见**：
- **方向完全正确**，但 §6 仅点出了问题，没有给出足够的设计指导。
- **关键缺失**：
  1. **429 和 502 的重试策略应不同**：429（Rate Limit）适合较长退避（基于 `Retry-After` Header），502/504（Proxy Timeout）适合短退避快速重试
  2. **重试预算**：需要设定全局重试上限（如 max 3 retries），否则在所有 backend 都返回 429 时，重试风暴会使延迟雪崩
  3. **`tenacity` 库的建议合理**：但应配合 `CircuitBreaker` 模式（如 `pybreaker`），当某个 backend 连续失败 N 次后自动熔断，避免无效重试
- **建议**：将此条从"建议"升级为具体的 `SearchRouter` 容错层设计规范，包含 Error Code 分类 → 重试策略映射表。

---

#### ⚠️ 风险点五（L1 激进重试导致永久拉黑）— **描述准确但风险被高估**

**§6 判断**：10 分钟的 `suspended_times` 可能导致 IP 被永久 403 封禁。

**审查意见**：
- **理论上正确**，但在实际的低流量场景下，这个风险被**显著高估**了：
  1. 我们的私有实例日均请求 < 100 次，远低于搜索引擎的永久封禁阈值
  2. 10 分钟的 suspension + L1.5 的请求节流，实际对单个引擎的请求频率不会超过 1 req/10min（CAPTCHA 恢复后才重试）
  3. 原文 §3 的 L1 Trade-off 中已经明确提及了这个风险并给出了缓解措施
- **真正需要担心的场景**：不是 L1 本身，而是 **L1 + L1.5 被绕过**的情况——例如开发者在调试时直接调用 SearXNG API（绕过 SearchRouter 的节流），短时间内触发大量请求。
- **建议**：补充一条运维规范——**禁止直接访问 SearXNG 端口（8080）进行调试，所有请求必须经过 SearchRouter**。

### 7.2 §6 遗漏的关键风险

§6 的五个风险点聚焦于"系统层面的并发和网络故障"，但遗漏了以下同等重要的风险：

#### 🚨 遗漏风险一：搜索结果质量退化（Silent Degradation）— P0 级

**描述**：所有 L1-L4 方案都聚焦于"搜索能不能用"（可用性），却**没有一个方案关注"搜索结果质量是否退化"**。例如：
- CAPTCHA 恢复后，百度可能返回更多广告或低质量结果（引擎"惩罚"可疑 IP）
- SearXNG 的引擎适配器可能因百度/搜狗的前端改版而静默失效（返回空结果但不报错）
- 代理 IP 可能导致搜索引擎返回代理所在地区的本地化结果，而非用户期望的结果

**建议**：在 §4 的实施路径中，增加"搜索质量持续监控"作为贯穿所有阶段的横切关注点（Cross-cutting Concern），复用已有的 `search_quality_report.py` 实现定期自动化检测。

#### ⚠️ 遗漏风险二：L2 代理引入的地域偏差（Geo-Bias）— P1 级

**描述**：当使用住宅代理时，搜索引擎会根据代理 IP 的地理位置返回本地化结果。如果代理 IP 位于广东而用户在北京，百度可能返回广东本地化的结果。

**建议**：代理选型时优先选择支持 **地区定向** 的代理服务商，并在 `SearXNGClient` 层增加 `locale/region` 参数透传。

#### ⚠️ 遗漏风险三：缺乏端到端的可观测性（Observability Gap）— P1 级

**描述**：整个 §3 的方案矩阵缺乏统一的指标体系。当 CAPTCHA 事件发生时，没有告警机制，运维人员可能在用户投诉后才发现引擎已被 suspended 数小时。

**建议**：增加一个横切的 Monitoring 层，包含：
- `searxng_engine_suspension_total` (Counter) — 引擎被 suspended 的次数
- `search_router_fallback_total` (Counter) — fallback 触发次数
- `search_quality_score` (Gauge) — 搜索结果质量评分
- 当主引擎被 suspended 时，通过 Webhook/钉钉/飞书推送告警

### 7.3 §6 改进建议的评估总结

| §6 建议 | 合理性 | 优先级调整 | 补充意见 |
|:---|:---:|:---:|:---|
| 重构 L1.5 为 Per-Engine Token Bucket | ✅ 方向正确 | 降为 v2 | v1 用全局锁 + 缩短间隔已足够 |
| 强制解绑 L3 无代理部署 | ✅ **完全正确** | 保持 | 应写入实施路径作为硬性前置条件 |
| 强化 SearchRouter 容错层 | ✅ 方向正确 | 需细化 | 缺少 Error Code 分类和重试策略映射 |
| 废除定时重启策略 | ⚠️ 过于绝对 | 降级 | 改为"限制在低谷时段 + 确保 retry 覆盖" |

---

## §8. 行业方案横向对比：其他 LLM 应用如何解决搜索基础设施问题

> **研究日期**: 2026-05-05  
> **研究范围**: 头部商业 LLM 应用（Perplexity, ChatGPT, Gemini）、开源 Deep Research Agent、AI-Native 搜索 API 生态

### 8.1 核心结论

> [!IMPORTANT]
> **没有任何一家头部 LLM 应用通过"自建 SearXNG + 爬虫"来解决搜索问题。** 它们要么拥有自己的搜索引擎基础设施，要么通过商业合作获得授权的 API 访问，要么两者兼有。自建 SearXNG/MCP 搜索服务是**独立开发者和开源社区的独特路径**，有其合理性，但必须清醒认识到这与头部玩家的基础设施差距是数量级的。

### 8.2 头部商业 LLM 应用的搜索架构

#### 8.2.1 Google Gemini Deep Research — "上帝模式"

| 维度 | 方案 |
|:---|:---|
| **搜索基础设施** | 直接调用 Google Search 内部 API（`google_search` tool） |
| **索引规模** | 全球最大的网页索引（万亿级页面） |
| **CAPTCHA 问题** | **不存在** — 内部系统调用，无反爬检测 |
| **架构** | Agentic Loop：分解查询 → 多次搜索 → 迭代推理 → 综合报告 |
| **数据时效性** | 实时（基于 Google 的持续爬虫和索引更新） |

**启示**：Gemini 的 Deep Research 之所以质量极高，根本原因不是模型更强，而是**搜索基础设施是 Google Search 本身**。这是一个我们无法复制的结构性优势。

#### 8.2.2 OpenAI ChatGPT Search — "商业合作模式"

| 维度 | 方案 |
|:---|:---|
| **搜索基础设施** | Bing Search API（Microsoft 战略合作） |
| **索引规模** | Bing 索引（十亿级页面） |
| **CAPTCHA 问题** | **不存在** — 授权 API 访问，无 CAPTCHA 挑战 |
| **内容抓取** | 通过 `OAI-SearchBot` 爬虫获取候选 URL 的完整内容 |
| **浏览器架构** | Atlas 浏览器（基于 Chromium + OWL 架构）用于深度网页交互 |
| **隐私** | 搜索查询与用户身份解耦（脱敏后发送给 Bing） |

**启示**：ChatGPT 通过 **商业合作** 而非技术对抗来解决 CAPTCHA 问题。Bing API 是授权通道，不触发任何反爬机制。

#### 8.2.3 Perplexity AI — "自建搜索引擎路径"

| 维度 | 方案 |
|:---|:---|
| **搜索基础设施** | 早期依赖 Bing API → 逐步建设 **PerplexityBot** 自有爬虫 + 自建索引 |
| **索引策略** | ML 驱动的选择性爬取（优先高质量权威内容，非全网爬取） |
| **CAPTCHA 问题** | 通过 **高质量代理网络 + 浏览器指纹模拟 + 行为模拟** 规避 |
| **检索架构** | 混合检索：关键词搜索 + 向量语义检索 → 多阶段排序 → 子文档级摘取 |
| **核心壁垒** | 自建索引 + AI 优化的检索管线（非简单的 SERP 转发） |

**启示**：
1. Perplexity 是唯一选择"自建搜索引擎"路径的头部玩家，但其投入规模（数百工程师 + 大量计算资源）远超我们的能力范围
2. 即使 Perplexity，早期也是从 **Bing API** 起步，验证产品后才开始自建索引
3. 它对抗 CAPTCHA 的方式是**住宅代理 + 浏览器指纹 + 行为模拟**，这与我们 L2 方案的方向一致

### 8.3 开源社区和中小型 AI 应用的搜索方案

| 方案 | 典型用户 | CAPTCHA 策略 | 优势 | 劣势 |
|:---|:---|:---|:---|:---|
| **付费 AI-Native 搜索 API**（Tavily, Exa, Firecrawl） | 绝大多数开源 Agent | 不存在（API 授权） | 零运维、高质量、即插即用 | 持续成本、供应商依赖、中国大陆不可达 |
| **SearXNG 自建** | 隐私敏感的开源项目 | 降低 suspension + 代理轮转 | 免费、自控 | CAPTCHA 对抗、维护负担 |
| **SearXNG + MCP 封装** | 我们的项目、部分开源 Agent | 同上 + MCP 协议标准化 | 免费、可扩展 | 所有 SearXNG 的劣势 + MCP 生态尚早期 |
| **传统 SERP API**（Serper, SerpApi） | 预算有限的应用 | 不存在（API 授权） | 低成本、Google 结果 | 只返回元数据需二次抓取、中国不可达 |
| **一次性爬取 + RAG** | 特定领域知识库 | 爬取阶段处理 | 零运行时成本、低延迟 | **数据时效性极差**、不适合实时研究 |

### 8.4 三种根本性架构路径的对比

```
路径 A：付费搜索 API（授权访问）
├─ 代表：ChatGPT (Bing API), Tavily/Exa 用户
├─ CAPTCHA 问题：不存在
├─ 适合：99% 的 AI 应用
└─ 不适合：中国大陆（大部分 API 被墙）

路径 B：自建搜索引擎/索引
├─ 代表：Perplexity (PerplexityBot), Google Gemini
├─ CAPTCHA 问题：通过高级反检测技术规避
├─ 适合：有搜索引擎工程团队和大量资源的公司
└─ 不适合：中小型团队和独立开发者

路径 C：元搜索引擎 + 反爬对抗
├─ 代表：SearXNG 自建（我们的方案）
├─ CAPTCHA 问题：核心挑战，需要多层缓解
├─ 适合：低预算、隐私敏感、中国大陆环境
└─ 不适合：需要高 SLA (>99.9%) 的生产环境
```

### 8.5 "一次性爬取 + RAG" 是否可行？

> **用户问题**：能否一次性爬取互联网信息，通过建立 RAG 解决实时搜索问题？

**结论：不可行，作为 Deep Research 的唯一搜索手段。**

| 维度 | 实时搜索 | 一次性爬取 + RAG |
|:---|:---|:---|
| **数据时效性** | 实时（秒级） | 滞后（天/周/月级） |
| **覆盖范围** | 理论上无限（整个互联网） | 受限于爬取范围和更新频率 |
| **适用场景** | "特朗普最新关税政策" | "Python GIL 的工作原理" |
| **存储成本** | 无（按需检索） | 巨大（向量数据库 + 文档存储） |
| **维护成本** | 搜索 API 费用 | 持续爬取 + 重新索引的计算成本 |

**但 RAG 作为补充层有价值**：
- 对于**稳定的知识类查询**（技术文档、学术论文、产品说明书），预爬取 + RAG 可以提供比实时搜索更高质量的结果（因为经过了 chunking + embedding 优化）
- **推荐架构**：实时搜索（SearXNG/博查）作为主通道，RAG 作为"知识缓存层"加速高频/稳定查询

### 8.6 自建 MCP 搜索服务是否合理？

> **用户问题**：其他大语言模型应用会自建 MCP 搜索服务吗？

**现状**：
- **头部商业应用（ChatGPT, Gemini, Perplexity）**：不使用 MCP。它们有自己的内部工具协议或直接集成搜索 API。
- **开源社区**：MCP 生态在 2025-2026 年爆发式增长（17,000+ MCP Server），部分开源 Agent 开始采用 MCP 作为工具标准化协议。
- **我们的定位**：作为面向 C 端的产品化 Agent，采用 MCP 作为搜索服务的内部通信协议是**合理但非必要的**。

**MCP 的价值不在于搜索本身，而在于标准化**：
- MCP 解决的是"如何让 Agent 动态发现和调用工具"的问题
- 对于搜索场景，直接的 HTTP API 调用（如 `SearchRouter.search()`）已经足够
- MCP 的真正价值在于：**当我们需要扩展到 知乎/微博/代码搜索/学术搜索 等多种异构工具时，MCP 提供了统一的发现和调用协议**

### 8.7 对我们设计方案的战略建议

基于上述行业分析，对现有 CAPTCHA 缓解策略的战略层面建议如下：

#### 建议一：接受"路径 C"的局限性，设定合理期望

我们选择的是"元搜索引擎 + 反爬对抗"路径，这是**中国大陆低预算环境下的合理选择**。但必须清醒认识到：
- 这条路径的 SLA 天花板约为 **95-98%**，无法达到路径 A 的 99.9%
- CAPTCHA 对抗是一场**持续的军备竞赛**，需要长期投入
- **不要期望通过纯技术手段彻底消除 CAPTCHA**

#### 建议二：尽早引入付费 API 作为兜底（L4 前置）

行业经验表明，**所有成功的 AI 搜索产品都在某个层面依赖授权的搜索 API**。建议：
- 将 L4（博查 API Fallback）的优先级从"中期"提升到与 L1/L1.5 同期实施
- 博查 API 的成本（￥0.01/次）远低于因 CAPTCHA 导致的搜索不可用的机会成本
- 形成"SearXNG 优先 + 博查兜底"的双层架构，兼顾成本和可用性

#### 建议三：将搜索质量监控纳入核心架构

参照 Perplexity 的做法，搜索质量不应是事后验证，而是系统的核心指标：
- 复用已有的 `search_quality_report.py` 构建**持续评估管线**
- 当搜索结果质量评分低于阈值时自动触发 fallback 到付费 API
- 这比单纯的"引擎是否可用"判断更有价值——引擎可用但返回垃圾结果是更隐蔽的故障模式

#### 建议四：不要在当前阶段自建搜索索引或全量 RAG

- 自建搜索索引需要的工程投入（爬虫 + 索引 + 排序）远超当前团队规模
- 全量 RAG 的数据时效性问题使其不适合 Deep Research 场景
- **正确的演进路径**：SearXNG + 博查（现在）→ 垂直平台 MCP Scraper（Phase 3）→ 评估是否需要自建索引（远期，视用户规模决定）

---

## §9. 头部 LLM 应用的搜索实现机制深度拆解

> **研究日期**: 2026-05-05  
> **研究目的**: 回答"头部玩家到底如何实现搜索功能"——不是宏观架构分类，而是具体的技术实现细节。

### 9.1 统一的技术范式：Tool Calling / Function Calling

**核心发现：所有头部 LLM 应用都使用相同的技术范式 — Tool Calling。** 模型本身不访问互联网，它只是生成一个"我需要搜索 X"的结构化 JSON 请求，由外部 Orchestrator 执行。

```
用户 Query
    ↓
LLM 分析是否需要搜索（Decision Layer）
    ↓ 需要
LLM 生成结构化 JSON（Tool Call: {"tool": "web_search", "query": "..."}）
    ↓
Orchestrator 拦截 JSON，调用实际搜索后端（Bing API / Google API / Brave API）
    ↓
搜索结果返回给 LLM 作为新 Context
    ↓
LLM 判断是否需要继续搜索（Iterative Loop）
    ↓ 不需要
LLM 基于搜索结果生成最终回答（带 Citation）
```

**关键点**：搜索功能是一个**平台层**的能力，不是模型层的。本地部署任何开源模型（DeepSeek-R1、Llama 等）时都没有搜索功能，需要开发者自己实现 Tool Calling + 搜索后端。

### 9.2 各家搜索后端速览

| 玩家 | 搜索后端 | 获取方式 | CAPTCHA 问题 | 核心特色 |
|:---|:---|:---|:---:|:---|
| **Google Gemini** | Google Search 内部 API | 自有基础设施 | 无 | 句级溯源（`groundingSupports`），万亿级索引 |
| **OpenAI ChatGPT** | Bing API + OAI-SearchBot | Microsoft 战略合作 | 无 | Atlas 浏览器（Chromium + OWL IPC 架构） |
| **Anthropic Claude** | **Brave Search API** | 第三方付费 API | 无 | Dynamic Filtering（代码预处理搜索结果） |
| **DeepSeek** | **博查 API** | 第三方付费 API | 无 | 中文搜索优化，推理链可视化 |
| **Perplexity** | **自建索引** (PerplexityBot) | 自有爬虫 + 自建索引 | 通过代理规避 | 唯一自建搜索引擎的头部玩家 |

### 9.3 各家架构技术深度拆解

#### 9.3.1 Google Gemini — "上帝模式"的句级溯源

Gemini 的 `google_search` tool 是一个 **Managed Service**，开发者无需管理任何搜索基础设施。

**Tool 声明与调用**：

```python
# Gemini API 调用（概念性伪代码）
response = model.generate(
    prompt="2026年中国GDP增速预测",
    tools=[{"google_search": {}}],  # 声明可用工具
)
```

**返回的 `groundingMetadata` 结构**：

```json
{
  "grounding_chunks": [
    {"web": {"uri": "https://...", "title": "国际货币基金组织报告"}}
  ],
  "grounding_supports": [
    {
      "segment": {"text": "2026年中国GDP预计增长4.5%"},
      "grounding_chunk_indices": [0],
      "confidence_scores": [0.95]
    }
  ],
  "search_entry_point": {
    "rendered_content": "<!-- 搜索建议 HTML -->"
  }
}
```

**独特之处**：
- **句级溯源**：`groundingSupports` 将生成文本中的每一个关键段落映射到具体来源 URL，并附带置信度分数。这比其他平台的 URL 级 Citation 精度高得多
- **Managed Pipeline**：开发者无需处理 embedding、chunking、retrieval 逻辑，Google 在后端全部完成
- **解耦执行**：Gemini 模型本身不执行搜索，它只生成"搜索意图"，由 Google GenAI SDK / Vertex AI 的执行层调用 Google Search 基础设施

**根本优势**：Gemini Deep Research 质量极高的根本原因**不是模型更强**，而是搜索基础设施是 Google Search 本身——拥有万亿级页面索引、持续更新的爬虫、成熟的排序算法。这是结构性优势，无法复制。

---

#### 9.3.2 OpenAI ChatGPT — 双层架构 + Atlas 浏览器

ChatGPT 的搜索实际上是**两层管线**：

```
Layer 1: Bing Search API
  → 获取候选 URL 列表 + Snippet（元数据）

Layer 2: OAI-SearchBot 爬虫
  → 对候选 URL 做 Full Page Crawl
  → 提取完整页面内容
  → 喂给模型做深度理解
```

**API 层面的实现**：

```python
# OpenAI Responses API（概念性伪代码）
response = client.responses.create(
    model="gpt-4.1",
    tools=[{"type": "web_search"}],  # 启用搜索
    input="最新的 AI Agent 框架对比"
)
```

**Atlas 浏览器（近期演进）**：

OpenAI 推出了 **Atlas** — 一个基于 Chromium 的独立浏览器架构：
- **OWL (OpenAI's Web Layer)**：Chromium 作为独立进程运行，通过 IPC（进程间通信）与主应用通信
- **能力范围**：不是简单的搜索，而是真正的浏览器自动化 — 能在多个 Tab 中导航、点击、提交表单、摘要当前页面
- **定位**：从"搜索 API 消费者"向"浏览器代理"演进

**隐私保护**：
- 搜索查询发送给 Bing 前，会与用户身份**解耦**（脱敏处理）
- 不传递 Session ID、Device ID 等标识信息
- 仅使用通用位置信息（基于 IP）优化结果相关性

---

#### 9.3.3 Anthropic Claude — 最透明的实现 + Dynamic Filtering

Claude 的搜索是目前公开文档最详尽的实现，提供了最精细的控制粒度。

**API 实现**：

```python
# Claude API 启用搜索（真实 API 签名）
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    tools=[{
        "type": "web_search_20260209",
        "name": "web_search",
        "max_uses": 5,                              # 单次对话最多搜索 5 次
        "allowed_domains": ["arxiv.org", "github.com"],  # 域名白名单
        "blocked_domains": ["spam-site.com"],        # 域名黑名单
    }],
    messages=[{"role": "user", "content": "..."}]
)
```

**独特特性**：

1. **Dynamic Filtering**：Claude 在搜索结果返回后，可以**自动编写并执行代码**来预处理结果。不是将整个 HTML 页面塞入 Context Window，而是先过滤出最相关的内容片段，减少 Token 消耗并提升准确性

2. **搜索后端是 Brave Search**：Claude 选择了 Brave Search 而非 Google 或 Bing，这是因为 Brave 提供了更注重隐私的搜索 API

3. **服务端循环 (Server-Side Loop)**：整个搜索 → 评估 → 再搜索的迭代循环在 Anthropic 的服务端完成，开发者收到的是最终结果。Claude 可以在一次对话轮次中执行多次搜索

4. **安全治理**：搜索结果通过 Anthropic 后端管理，企业管理员可以强制执行域名黑白名单策略

**MCP 的角色**：Claude 支持通过 Playwright MCP Server 扩展浏览器自动化能力，但这是**可选扩展**，不是搜索的核心架构。核心搜索仍然是 Brave Search API。

---

#### 9.3.4 DeepSeek — 国内路径，最值得我们参考

DeepSeek 和我们面临**完全相同的约束**（中国大陆、中文搜索优先），因此它的方案最具参考价值。

**实现架构**：

```
用户 Query
    ↓
DeepSeek-R1 模型分析（推理链可见）
    ↓
生成搜索关键词 + 选择搜索策略
    ↓
调用外部搜索 API（博查 API 等）
    ↓
结果评估与过滤（小模型打分排序）
    ↓
实时爬取高相关性 URL，提取正文
    ↓
结构化内容注入模型 Context
    ↓
推理模型综合生成回答
```

**关键特点**：
- **搜索后端**：中文搜索使用 **博查 (Bocha) API**，这正是我们 §3 L4 方案推荐的 fallback
- **平台层能力**：搜索功能在 DeepSeek 平台上可用，但本地部署 DeepSeek-R1（如通过 Ollama）时**没有搜索功能**，需要开发者自己实现 RAG Pipeline
- **推理链可视化**：用户可以看到 DeepSeek 如何选择搜索词、如何评估结果质量、如何决定是否需要继续搜索
- **解耦架构**：搜索能力不是 Transformer 架构的一部分（MoE、MLA 等与搜索无关），而是模型之上的 Agentic Workflow

**对我们的验证**：DeepSeek 选择博查 API 作为中文搜索后端，验证了我们"尽早引入博查作为兜底"的建议（§8.7 建议二）。

---

#### 9.3.5 Perplexity AI — 唯一的"自建搜索引擎"路径

Perplexity 是头部玩家中唯一选择自建搜索基础设施的公司，但其投入规模远超个人/小团队。

**演进路径**：

```
Phase 1（早期）: Bing API 作为搜索后端
  → 验证产品 PMF（Product-Market Fit）
  → 低成本快速迭代

Phase 2（成长期）: 自建 PerplexityBot 爬虫 + 自建索引
  → ML 驱动的选择性爬取（优先高质量权威内容）
  → 混合检索管线（关键词 + 向量语义）

Phase 3（成熟期）: AI-First 搜索引擎
  → 子文档级精准摘取（非全文档索引）
  → 多阶段排序（启发式 → AI 重排序）
  → 实时索引更新
```

**检索管线（Production Pipeline）**：

```
Query → Query Expansion（改写/扩展）
    ↓
Hybrid Retrieval:
  ├─ Lexical Search（关键词 BM25）
  └─ Semantic Search（向量 Embedding）
    ↓
Multi-Stage Ranking:
  ├─ Stage 1: 启发式过滤（去除过时/低质量内容）
  ├─ Stage 2: ML 排序（相关性 + 可信度 + 时效性）
  └─ Stage 3: 子文档级摘取（精确到段落/章节）
    ↓
Context Assembly → LLM 生成带 Citation 的回答
```

**CAPTCHA 应对**：
- **高质量住宅代理网络**：使用住宅/移动 IP，IP 信誉高，难以被识别为爬虫
- **浏览器指纹模拟**：精心管理 User-Agent、WebGL、Canvas、WebRTC 指纹
- **行为模拟**：模拟鼠标移动、点击节奏、滚动模式等人类行为特征
- **投入规模**：数百名工程师 + 大量计算/网络资源，非个人/小团队可复制

> [!IMPORTANT]
> **Perplexity 的启示**：即使 Perplexity 这样的"自建派"，在早期也是从 Bing API 起步，先验证产品再逐步自建。这强化了 §8.7 建议二 — 我们当前阶段应先用博查 API 保证可用性，自建基础设施（如果需要的话）是远期目标。

### 9.4 三个关键洞察

#### 洞察一：搜索后端 ≠ MCP，也 ≠ 自建爬虫

头部玩家使用的都是各自私有的 **Function Calling / Tool Use** 协议：
- OpenAI：`tools` 参数 + `web_search` type
- Google：`google_search` tool + `groundingMetadata`
- Anthropic：`web_search_20260209` tool type
- 无一使用 MCP

MCP 是开源社区的标准化尝试。它的价值在于**异构工具发现和调用的统一协议**，而非搜索本身。

#### 洞察二：没有一家在"对抗 CAPTCHA"

因为它们通过**授权通道**获取搜索结果。Bing API、Brave Search API、博查 API 都是正规商业接口，无 CAPTCHA。唯一的例外是 Perplexity 的自建爬虫，但它用住宅代理 + 指纹模拟来规避，投入规模不是个人/小团队能复制的。

#### 洞察三：DeepSeek 的路径最值得我们参考

DeepSeek 和我们面临完全相同的约束（中国大陆、中文优先、需要低成本方案），它选择了**博查 API**。这从行业实践角度验证了我们方案的方向。

### 9.5 对我们设计方案的启示

| 我们的现状 | 行业做法 | 差距 | 行动建议 |
|:---|:---|:---|:---|
| SearXNG 自建（反爬对抗） | 付费 API（授权访问） | 路径根本不同 | SearXNG 做免费主通道，**必须有博查兜底** |
| MCP 作为搜索协议 | 私有 Tool Calling 协议 | MCP 非必须 | MCP 价值在多工具扩展，搜索用 HTTP 即可 |
| 单一搜索后端 | 多层 fallback + 迭代搜索 | 缺少容错和迭代 | 参照 Claude 的 `max_uses` + 多轮搜索模式 |
| 无句级溯源 | Gemini `groundingSupports` | Citation 精度差距 | 中期目标：实现段落级 Citation mapping |

**总结**：我们走的"SearXNG + 反爬"路径是中国大陆低预算约束下的独特选择，行业里没有直接先例。最务实的策略是**借鉴 DeepSeek 的方案**（博查 API 做保保底），用 SearXNG 的免费优势降成本，形成"SearXNG 优先 + 博查兜底"的双层架构。

---

## §10. MCP 服务的价值重评估：搜索、Skills 与 GitHub MCP

> **研究日期**: 2026-05-05
> **核心问题**: 自建 MCP 搜索服务遭遇 CAPTCHA 困境后，当前 MCP 架构是否仍有价值？能否转型为 Skills 接入层？GitHub MCP 与自建搜索 MCP 是什么关系？

### 10.1 概念澄清：三种完全不同的 MCP 用途

> [!IMPORTANT]
> **MCP (Model Context Protocol) 是一个通用协议，不是"搜索服务"的同义词。** 必须将以下三种用途严格区分，否则会陷入"MCP 搜索不行 → MCP 都没用"的错误推断。

```
用途 A：MCP 作为搜索服务协议（我们当前的实现）
  ├─ 功能：封装 SearXNG/知乎/微博的搜索能力为 MCP Tools
  ├─ 核心痛点：底层依赖的 SearXNG 有 CAPTCHA 问题
  ├─ 结论：MCP 协议本身无问题，问题出在底层搜索引擎
  └─ 价值：有限 — 对搜索场景，直接 HTTP API 即可

用途 B：MCP 作为 Skills/工具集成标准协议
  ├─ 功能：Agent 通过 MCP 动态发现和调用外部 Skills
  ├─ 与搜索无关：Skills = 代码执行、数据库查询、文件操作等
  ├─ 结论：这是 MCP 的核心价值 — 标准化异构工具集成
  └─ 价值：极高 — 解决 N×M 工具集成问题

用途 C：GitHub 官方 MCP Server
  ├─ 功能：通过 MCP 协议访问 GitHub API（搜索代码、管理 PR/Issue）
  ├─ 与自建搜索 MCP 完全无关：它连的是 GitHub REST/GraphQL API
  ├─ 结论：即插即用，无 CAPTCHA 问题，授权 API 通道
  └─ 价值：极高 — 可直接用于搜索/下载 GitHub 上的 Skills
```

### 10.2 自建 MCP 搜索服务的剩余价值

#### 10.2.1 CAPTCHA 问题的归因分析

| 层级 | 组件 | 是否有问题 | 说明 |
|:---|:---|:---:|:---|
| **协议层** | MCP (Model Context Protocol) | ❌ 无问题 | MCP 只是传输和发现协议，与 CAPTCHA 无关 |
| **中间层** | `MCPSearchClient` / `SearchMCPServer` | ❌ 无问题 | 工具注册、工具发现、session 管理均正常 |
| **底层** | SearXNG → 百度/搜狗/Startpage | ✅ **根因** | 元搜索引擎对下游搜索引擎的非授权爬取触发 CAPTCHA |

**结论：CAPTCHA 是底层搜索引擎的问题，不是 MCP 协议的问题。** 将 MCP 协议与搜索失败挂钩，是一种归因谬误。

#### 10.2.2 自建 MCP 搜索服务的价值分层评估

| 价值维度 | 当前评估 | 未来演进 |
|:---|:---|:---|
| **通用搜索（web_search）** | ⚠️ **低价值** — SearXNG 的 CAPTCHA 问题使其不可靠；行业共识是用付费 API（博查/Brave）替代 | 博查 API 直接 HTTP 调用即可，无需 MCP 封装 |
| **垂直平台搜索（zhihu/weibo/weixin）** | ⭐ **中等价值** — 这些平台无公开 API，Playwright 爬虫是唯一路径；但反爬对抗成本极高 | 如果产品化需要垂直数据源，MCP 封装提供统一接口 |
| **页面抓取（scrape_url）** | ⭐⭐ **有价值** — 页面内容提取不依赖搜索引擎，无 CAPTCHA 问题 | 可独立保留作为 Deep Research 的内容获取工具 |
| **MCP 协议标准化本身** | ⭐⭐⭐ **高价值** — 工具发现、调用、生命周期管理的标准化 | **重点转向：非搜索类工具的 MCP 接入** |

#### 10.2.3 战略结论

> [!WARNING]
> **自建 MCP "搜索"服务作为 Deep Research 的核心搜索基础设施，ROI 不合理。** 行业内没有任何一家成功的 LLM 应用通过自建元搜索引擎来解决搜索问题（详见 §8, §9）。但 MCP 作为协议层的价值仍然存在。

**推荐行动**：
1. **搜索回归正道**：放弃 SearXNG 作为主搜索引擎的路线，转向博查 API 作为主搜索后端（直接 HTTP，无需 MCP 封装）
2. **保留 MCP 架构**：将 `SearchMCPServer` 重新定位为 **"非搜索类工具的 MCP 接入层"**
3. **SearXNG 降级为辅助**：SearXNG 仅作为零成本的降级通道，不再作为主路径

---

### 10.3 MCP 用于 Skills 接入的可行性分析

#### 10.3.1 什么是 MCP 的 Skills 接入？

MCP 的 **核心价值不在搜索，而在标准化工具发现和调用**。将 MCP 用于 Skills 接入，意味着：

```
传统方式（N×M 问题）:
  Agent A → Custom wrapper → Skill 1
  Agent A → Custom wrapper → Skill 2
  Agent B → Different wrapper → Skill 1  ← 重复造轮子
  Agent B → Different wrapper → Skill 2

MCP 方式（N+M 问题）:
  Agent A ─┐
  Agent B ─┤── MCP Client ── MCP Server A ── Skill 1, Skill 2
  Agent C ─┘                 MCP Server B ── Skill 3, Skill 4
```

#### 10.3.2 我们的 MCP 架构能否直接用于 Skills？

**可以，但需要重新定位。** 当前 `MCPSearchClient` 的架构设计已经具备了通用 MCP 客户端的能力：

| 已有能力 | 用于 Skills 的适用性 |
|:---|:---|
| `load_mcp_tools(session)` 动态工具发现 | ✅ **直接可用** — 连接任何 MCP Server 都能自动发现其暴露的工具 |
| `get_tools(names=[...])` 工具名过滤 | ✅ **直接可用** — 可以过滤出 Supervisor/Worker 需要的工具子集 |
| `StdioServerParameters` 子进程管理 | ✅ **直接可用** — stdio 传输层与 MCP Server 的实现语言无关 |
| `async with MCPSearchClient()` 生命周期 | ✅ **直接可用** — 适用于任何 MCP Server 的连接管理 |

**需要的改动**：

1. **类名重构**：`MCPSearchClient` → `MCPToolClient`（语义更通用）
2. **MultiServerMCPClient 升级**：当接入多个异构 MCP Server 时（搜索 + GitHub + 代码执行），需要从单 Server 升级到 Multi-Server 架构
3. **工具注册表**：需要一个 Registry 来管理"哪些 MCP Server 可用、暴露了哪些工具、每个工具的权限边界"

#### 10.3.3 Skills 接入的具体场景

| MCP Server | 暴露的工具/Skills | 用途 | 与搜索的关系 |
|:---|:---|:---|:---:|
| **GitHub MCP** | `search_code`, `get_file_contents`, `list_repos` | 搜索/下载开源 Skills 和代码 | ❌ 完全无关 |
| **Database MCP** | `query_sql`, `list_tables` | 数据查询和分析 | ❌ 完全无关 |
| **Code Executor MCP** | `run_python`, `run_bash` | 代码执行和验证 | ❌ 完全无关 |
| **File System MCP** | `read_file`, `write_file`, `list_dir` | 本地文件操作 | ❌ 完全无关 |
| **搜索 MCP**（当前） | `web_search`, `scrape_url` | 互联网搜索 | ✅ 当前用途 |

> [!TIP]
> **MCP 的价值在于它是一个"USB-C 接口"，能接各种设备。** 当前我们只接了一个"U盘"（搜索），但同一个接口可以接"键盘"（GitHub）、"显示器"（数据库）、"打印机"（代码执行器）。搜索出了问题不代表接口出了问题。

---

### 10.4 GitHub MCP 与自建搜索 MCP 的关系

#### 10.4.1 核心结论

> [!IMPORTANT]
> **GitHub MCP 与自建搜索 MCP 完全无关。它们是不同的 MCP Server，解决不同的问题，有不同的底层数据源。** 唯一的共同点是它们都遵循 MCP 协议。

| 维度 | 自建搜索 MCP | GitHub 官方 MCP |
|:---|:---|:---|
| **数据源** | SearXNG → 百度/搜狗/知乎 | GitHub REST/GraphQL API |
| **CAPTCHA 问题** | ✅ 有（SearXNG 下游引擎封禁） | ❌ **无**（GitHub 授权 API，无反爬） |
| **部署方式** | 自建 Python MCP Server（stdio） | GitHub 官方托管 或 本地 Docker/Go binary |
| **认证方式** | 无认证（内部服务） | GitHub OAuth 或 Personal Access Token |
| **维护责任** | 我们自己维护 | GitHub 官方维护 |
| **目的** | 互联网搜索 | 代码搜索、仓库浏览、PR/Issue 管理 |
| **稳定性** | ⭐⭐ 低（依赖反爬对抗） | ⭐⭐⭐⭐⭐ 极高（授权 API，SLA 保障） |

#### 10.4.2 GitHub MCP 用于搜索和下载 Skills 的架构

如果要接入 GitHub MCP 用于搜索和下载开源 Skills，架构如下：

```
Deep Research Agent
  ├─ MCPToolClient (MultiServerMCPClient)
  │   ├─ Server: "search" → 博查 API (HTTP 直调，或轻量 MCP 封装)
  │   │   └─ Tools: web_search, scrape_url
  │   │
  │   ├─ Server: "github" → GitHub 官方 MCP Server ← 新增
  │   │   └─ Tools: search_code, get_file_contents, list_repos,
  │   │            search_repositories, get_pull_request, ...
  │   │
  │   └─ Server: "skills" → 自建 Skills Registry MCP ← 未来可选
  │       └─ Tools: discover_skills, install_skill, list_installed
  │
  └─ Agent Graph (Supervisor + Workers)
      ├─ Worker: 使用 web_search 做互联网研究
      └─ Worker: 使用 search_code/get_file_contents 搜索 GitHub Skills
```

**GitHub MCP 的接入方式**：

```python
# 使用 MultiServerMCPClient 同时接入多个 MCP Server
from langchain_mcp_adapters.client import MultiServerMCPClient

async with MultiServerMCPClient({
    "github": {
        # 方式 A：远程 GitHub 托管 MCP（推荐，零运维）
        "url": "https://api.githubcopilot.com/mcp/",
        "transport": "sse",
        "headers": {"Authorization": f"Bearer {github_token}"},
    },
    # 方式 B：本地 Docker 运行
    # "github": {
    #     "command": "docker",
    #     "args": ["run", "-i", "--rm",
    #              "-e", f"GITHUB_PERSONAL_ACCESS_TOKEN={pat}",
    #              "ghcr.io/github/github-mcp-server"],
    #     "transport": "stdio",
    # },
}) as client:
    github_tools = client.get_tools("github")
    # search_code, get_file_contents, list_repos, ...
```

#### 10.4.3 GitHub MCP 在 Skills 场景的具体能力

| GitHub MCP 工具 | Skills 搜索/下载的用途 | 可行性 |
|:---|:---|:---:|
| `search_code` | 搜索 GitHub 上的 MCP Server 实现代码 | ✅ 直接可用 |
| `search_repositories` | 搜索包含特定 Skill 的开源仓库 | ✅ 直接可用 |
| `get_file_contents` | 下载 Skill 的配置文件、代码 | ✅ 直接可用 |
| `list_repos` | 浏览特定组织/用户的 Skill 仓库 | ✅ 直接可用 |
| `get_pull_request` | 查看 Skill 的最新更新和变更 | ✅ 直接可用 |

> [!NOTE]
> GitHub MCP 的 `search_code` 和 `get_file_contents` 能力使其成为一个天然的"Skills 搜索引擎"。你可以用它搜索 `topic:mcp-server language:python` 来发现新的 MCP Server Skills，然后用 `get_file_contents` 读取其 README 和配置。

---

### 10.5 综合战略结论

#### 10.5.1 三个独立决策

以下三个决策是**相互独立**的，不应混为一谈：

| # | 决策 | 结论 | 理由 |
|:---|:---|:---|:---|
| **D1** | 是否继续用自建 SearXNG 作为主搜索引擎？ | ❌ **不推荐** | CAPTCHA 问题是结构性的，行业无先例，ROI 不合理 |
| **D2** | 是否保留 MCP 作为工具集成协议？ | ✅ **保留** | MCP 是 2025-2026 行业标准，架构前瞻性极高 |
| **D3** | 是否接入 GitHub MCP 做 Skills 搜索？ | ✅ **推荐** | 授权 API，零 CAPTCHA，即插即用 |

#### 10.5.2 推荐的演进路径

```
当前状态:
  Agent → MCPSearchClient (单 Server) → SearchMCPServer → SearXNG（CAPTCHA 频发）

Phase 1（立即）:
  Agent → 博查 API (直接 HTTP 调用，无需 MCP 封装)
         └─ SearXNG 降级为零成本 fallback

Phase 2（短期）:
  Agent → MCPToolClient (MultiServerMCPClient)
         ├─ 博查 HTTP (或薄 MCP 封装)  ← 主搜索
         └─ GitHub MCP Server           ← Skills 搜索/下载（新增）

Phase 3（中期）:
  Agent → MCPToolClient (MultiServerMCPClient)
         ├─ 博查 HTTP                   ← 主搜索
         ├─ GitHub MCP Server           ← Skills 搜索/下载
         ├─ Code Executor MCP           ← 代码验证
         └─ SearXNG (低优先级 fallback) ← 免费兜底
```

#### 10.5.3 关于 SearXNG 和自建搜索 MCP 的最终定位

| 角色 | 之前的定位 | 新定位 |
|:---|:---|:---|
| SearXNG | 主搜索引擎 | **降级为零成本 fallback**，仅在博查不可用时使用 |
| 自建 SearchMCPServer | 核心搜索基础设施 | **降级为实验性组件**，垂直平台爬虫（知乎/微博）按需保留 |
| MCP 协议 | 搜索服务封装 | **升级为通用工具集成层**，接入 GitHub/代码执行/数据库等多种 MCP Server |
| `MCPSearchClient` | 搜索专用客户端 | **重构为 `MCPToolClient`**，支持 MultiServer 架构 |

> [!CAUTION]
> **不要因为"自建 MCP 搜索"的失败而否定 MCP 协议本身的价值。** 这是两个完全不同层次的问题：
> - ❌ "自建 SearXNG 搜索" 失败了 → 搜索应该用授权 API（博查）
> - ✅ "MCP 作为协议" 仍然有价值 → 用于接入 GitHub MCP、Code Executor 等非搜索工具
>
> 类比：不能因为"用 USB-C 接的 U 盘坏了"就说"USB-C 接口没用了"。

