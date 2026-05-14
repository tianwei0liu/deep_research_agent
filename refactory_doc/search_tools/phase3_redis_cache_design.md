# Phase 3: Redis 缓存层设计

> **模块**: `search_service/cache.py` (扩展)  
> **依赖**: `redis[hiredis]>=5.0`, SearchServiceConfig  
> **日期**: 2026-05-05

---

## 1. 设计目标

替换 `NullCache` 为 Redis 实现:
- 搜索结果缓存 — 减少重复查询对 SearXNG/平台的压力
- 差异化 TTL — 根据内容时效性设置不同过期时间
- 优雅降级 — Redis 不可用时自动退化为 NullCache
- TTL Jitter — 防止 thundering herd

---

## 2. 核心类: RedisCache

```python
class RedisCache:
    """Redis-backed cache implementing CacheLayer protocol.

    Features:
    - redis-py 5.0+ async client with connection pooling
    - Pydantic model JSON serialization
    - Circuit breaker: 30s cooldown on failure
    - TTL jitter: +5-10% to prevent thundering herd
    - Key prefix: namespace isolation ("search:")
    """
    def __init__(self, redis_url, default_ttl=3600, max_connections=20, key_prefix="search:") -> None: ...
    async def get(self, key: str) -> Optional[Any]: ...       # 返回 dict 或 None
    async def set(self, key, value, ttl_seconds) -> None: ... # Pydantic → JSON
    async def delete(self, key: str) -> None: ...
    async def clear(self) -> None: ...                         # SCAN+DELETE (非 KEYS)
    async def health_check(self) -> bool: ...
    async def close(self) -> None: ...
```

**关键设计**:
- **序列化**: `model_dump_json()` for Pydantic, `json.dumps()` for dict
- **反序列化**: 返回 `dict`，由 `SearchRouter` 通过 `SearchResponse(**cached)` 重建
- **降级**: 所有 Redis 错误被 catch，返回 None (get) 或 no-op (set)

---

## 3. Circuit Breaker

```
CLOSED (正常) → Redis 错误 → OPEN (冷却 30s)
                                  ↓ 30s 后
                          HALF-OPEN (尝试重连)
                            ├─ 成功 → CLOSED
                            └─ 失败 → OPEN
```

冷却期内所有 cache 操作直接跳过（退化为 NullCache 行为）。

---

## 4. CacheFactory

```python
class CacheFactory:
    @staticmethod
    def create(config: SearchServiceConfig) -> CacheLayer:
        if config.cache_backend == "null": return NullCache()
        elif config.cache_backend == "redis":
            assert config.redis_url, "redis_url required"
            return RedisCache(redis_url=config.redis_url, ...)
```

---

## 5. Docker 配置

```yaml
services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    mem_limit: 256m
    command: redis-server --maxmemory 200mb --maxmemory-policy allkeys-lru --save "" --appendonly no
```

---

## 6. 差异化 TTL

| 工具 | TTL | 理由 |
|:---|:---|:---|
| `web_search` | 1h | 中等变化 |
| `zhihu_search` | 4h | 稳定内容 |
| `weibo_search` | 15min | 实时性极强 |
| `weixin_search` | 2h | 中等更新 |
| `github_search` | 30min | 活跃 |
| `scrape_url` | 24h | 页面稳定 |

已在 `cache.py` 的 `CACHE_TTL` 中定义。

---

## 7. SearchRouter 适配

当前 `SearchRouter.search()` 返回 `SearchResponse` 对象。需调整:
- `cache.set(key, result, ttl)` — Pydantic model 自动 JSON 序列化
- `cache.get(key)` → dict → `SearchResponse(**cached)` — 反序列化

---

## 8. 行业价值分析与主流实现方式

### 8.1 功能实际价值评估

| 维度 | 评估 | 说明 |
|:---|:---|:---|
| **性能提升** | ⭐⭐⭐⭐ | 减少重复查询对 SearXNG/平台的压力，降低延迟 |
| **稳定性** | ⭐⭐⭐⭐⭐ | Circuit Breaker 确保 Redis 故障不级联到搜索服务 |
| **成本节约** | ⭐⭐⭐ | 减少对外部搜索引擎的请求次数，降低被反爬的风险 |

### 8.2 业界主流实现方式

**Cache-Aside + Circuit Breaker 是 2025-2026 年搜索缓存的行业标准组合**。

1. **Cache-Aside (Lazy Loading)**: 我们采用的模式，也是行业标准——先查缓存，miss 时查源并填充
2. **TTL Jitter**: 在 TTL 上加 5-10% 随机偏移防止 thundering herd（缓存雪崩），我们已设计此功能
3. **Cache Stampede Protection**: 行业实践用 `SETNX` 分布式锁确保 cache miss 时只有一个请求去源查询，其他等待。当前设计未包含此优化
4. **Circuit Breaker**: CLOSED → OPEN → HALF-OPEN 状态机是标准模式，我们的 30s 冷却设计合理
5. **Eviction Policy**: `allkeys-lru` 是搜索缓存的行业推荐策略，我们已配置
6. **差异化 TTL**: 按内容时效性设置不同 TTL 是行业最佳实践，我们的设计（weibo 15min / scrape_url 24h）体现了此理念

> [!TIP]
> 设计与行业标准高度一致。建议补充 Cache Stampede Protection（SETNX 锁），防止高并发下同一 query 同时穿透缓存。

---

## 9. 待确认事项

1. **Redis vs Valkey**: 搜索缓存 Redis 和 SearXNG Limiter 的 Valkey 是否共用实例？
2. **msgpack**: 大结果集（scrape_url）是否需要切换到 msgpack？
3. **InMemoryLRUCache**: 是否需要作为不部署 Redis 时的中间方案？
