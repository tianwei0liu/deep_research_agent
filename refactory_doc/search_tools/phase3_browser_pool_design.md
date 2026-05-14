# Phase 3: BrowserPool 进程回收 + 内存监控设计

> **模块**: `search_service/browser/pool.py` (增强)  
> **依赖**: `psutil`  
> **日期**: 2026-05-05

---

## 1. 设计目标

增强 BrowserPool 的生产健壮性:
- **内存监控**: 实时追踪 Chromium 进程树内存使用
- **智能回收**: 基于内存 / 请求数 / 时间的多维度回收策略
- **生命周期绑定**: BrowserPool 与 MCP Server 进程绑定，优雅清理
- **孤儿进程防护**: 确保异常退出时不留下 zombie Chromium 进程

---

## 2. 当前 BrowserPool 不足

| 问题 | 现状 | 风险 |
|:---|:---|:---|
| 回收策略单一 | 仅按请求数回收 | 内存泄漏时不触发回收 |
| 无内存监控 | 不知道 Chromium 用了多少内存 | 可能 OOM kill |
| 无时间回收 | 长时间运行不回收 | 内存碎片累积 |
| 无进程清理 | `shutdown()` 可能遗留子进程 | zombie 进程 |
| 无 Agent 绑定 | `make_scrape_url` 创建的 pool 无 shutdown hook | 资源泄漏 |

---

## 3. 增强设计

### 3.1 多维度回收策略

```python
class RecyclePolicy:
    """Browser recycling policy — multi-dimensional triggers.

    Any one condition being met triggers a recycle:
    1. max_requests: 请求数超限
    2. max_age_seconds: 运行时间超限
    3. max_memory_mb: 内存使用超限

    Args:
        max_requests: Max requests before recycle (default 100).
        max_age_seconds: Max browser age in seconds (default 1800 = 30min).
        max_memory_mb: Max RSS memory in MB (default 512).
    """
    def __init__(
        self,
        max_requests: int = 100,
        max_age_seconds: int = 1800,
        max_memory_mb: int = 512,
    ) -> None: ...

    def should_recycle(
        self,
        request_count: int,
        age_seconds: float,
        memory_mb: float,
    ) -> tuple[bool, str]:
        """Check if browser should be recycled.

        Returns:
            (should_recycle, reason) — reason is one of
            "requests", "age", "memory", or "".
        """
        if request_count >= self.max_requests:
            return True, "requests"
        if age_seconds >= self.max_age_seconds:
            return True, "age"
        if memory_mb >= self.max_memory_mb:
            return True, "memory"
        return False, ""
```

### 3.2 内存监控

```python
class BrowserMemoryMonitor:
    """Monitor Chromium process tree memory usage via psutil.

    Chromium spawns multiple child processes:
    - Main browser process
    - GPU process
    - Renderer processes (one per tab/context)
    - Network service
    - Utility processes

    This monitor sums RSS across the entire process tree.
    """

    @staticmethod
    async def get_memory_mb(browser_pid: int) -> float:
        """Get total RSS memory of browser process tree.

        Args:
            browser_pid: PID of the main browser process.

        Returns:
            Total RSS in MB across all child processes.
        """
        try:
            import psutil
            parent = psutil.Process(browser_pid)
            children = parent.children(recursive=True)
            total_rss = parent.memory_info().rss
            for child in children:
                try:
                    total_rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return total_rss / (1024 * 1024)  # bytes → MB
        except Exception:
            return 0.0

    @staticmethod
    async def kill_process_tree(pid: int) -> None:
        """Force-kill browser process tree.

        Used as last resort when graceful close fails.
        """
        try:
            import psutil
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                child.kill()
            parent.kill()
        except Exception:
            pass  # Process already dead
```

### 3.3 增强的 BrowserPool

```python
class BrowserPool:
    # ... existing code ...

    async def _maybe_recycle_browser(self) -> None:
        """Recycle browser based on multi-dimensional policy."""
        async with self._lock:
            age = time.monotonic() - self._browser_start_time
            memory = await BrowserMemoryMonitor.get_memory_mb(
                self._browser_pid,
            )

            should_recycle, reason = self._recycle_policy.should_recycle(
                request_count=self._request_count,
                age_seconds=age,
                memory_mb=memory,
            )

            if should_recycle:
                self._logger.info(
                    "browser_recycle",
                    extra={
                        "reason": reason,
                        "requests": self._request_count,
                        "age_seconds": int(age),
                        "memory_mb": f"{memory:.0f}",
                    },
                )
                await self._recycle_browser()

    async def _recycle_browser(self) -> None:
        """Close current browser and launch a new one."""
        old_pid = self._browser_pid
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                # Force kill if graceful close fails
                await BrowserMemoryMonitor.kill_process_tree(old_pid)

        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
                   f"--js-flags=--max-old-space-size={self._memory_limit_mb}"],
        )
        self._browser_pid = self._browser.process.pid  # type: ignore
        self._browser_start_time = time.monotonic()
        self._request_count = 0
```

---

## 4. Agent 进程绑定

### 4.1 问题

当前 `stream_deep_research` 使用 `async with MCPSearchClient()` 管理 MCP subprocess。
MCP subprocess 内部的 `SearchMCPServer.startup()` 启动 BrowserPool。
当 MCP subprocess 被 `__aexit__` 终止时，`SearchMCPServer.shutdown()` 被调用，关闭 BrowserPool。

**但**: 如果 Agent 进程被强制 kill（SIGKILL），MCP subprocess 可能成为孤儿进程，
其内部的 Chromium 进程也不会被清理。

### 4.2 方案: 进程组管理

```python
# mcp_client.py 增强
class MCPSearchClient:
    async def __aenter__(self) -> MCPSearchClient:
        # 注册信号处理器，确保异常退出时清理 subprocess
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._signal_handler)
        # ... existing spawn logic ...

    def _signal_handler(self, signum, frame):
        """Handle SIGTERM: cleanup MCP subprocess."""
        if self._stdio_cm is not None:
            # Force terminate subprocess
            # (无法 await async cleanup in signal handler)
            import os
            if hasattr(self, '_subprocess_pid'):
                os.killpg(os.getpgid(self._subprocess_pid), signal.SIGTERM)
        # Restore original handler and re-raise
        signal.signal(signal.SIGTERM, self._original_sigterm)
        os.kill(os.getpid(), signum)
```

### 4.3 方案: atexit 注册

```python
import atexit

class BrowserPool:
    async def start(self) -> None:
        # ... existing start logic ...
        self._browser_pid = self._browser.process.pid
        # 注册 atexit 清理
        atexit.register(self._force_cleanup)

    def _force_cleanup(self) -> None:
        """Synchronous cleanup for atexit — force-kill Chromium."""
        if self._browser_pid:
            BrowserMemoryMonitor.kill_process_tree(self._browser_pid)
```

---

## 5. Metrics 集成

```python
# 在 BrowserPool 中暴露指标给 MetricsCollector
class BrowserPool:
    def get_stats(self) -> dict:
        """Return pool statistics for metrics collection."""
        return {
            "active_contexts": self._max_concurrency - self._semaphore._value,
            "total_requests": self._request_count,
            "browser_age_seconds": time.monotonic() - self._browser_start_time,
            "browser_pid": self._browser_pid,
        }
```

---

## 6. 验收标准

| 测试用例 | 说明 |
|:---|:---|
| `test_recycle_by_requests` | 请求数 ≥100 → 回收 |
| `test_recycle_by_age` | 运行 >30min → 回收 |
| `test_recycle_by_memory` | 内存 >512MB → 回收 |
| `test_memory_monitor` | psutil 获取进程树 RSS |
| `test_force_kill` | graceful close 失败 → force kill |
| `test_shutdown_cleanup` | shutdown → 无残留进程 |
| `test_stats` | get_stats() 返回正确统计 |

---

## 7. 行业价值分析与主流实现方式

### 7.1 功能实际价值评估

| 维度 | 评估 | 说明 |
|:---|:---|:---|
| **生产必要性** | ⭐⭐⭐⭐⭐ | 无内存回收的 Chromium 长时间运行**必然 OOM**，这是生产环境的刚需 |
| **稳定性价值** | ⭐⭐⭐⭐⭐ | zombie 进程防护避免资源泄漏导致宿主机崩溃 |
| **可观测性** | ⭐⭐⭐⭐ | 内存监控数据为容量规划提供依据 |

### 7.2 业界主流实现方式

| 方案 | 采用度 | 说明 |
|:---|:---|:---|
| **多维度回收策略** | ⭐⭐⭐⭐⭐ | 按请求数+时间+内存三维触发回收，行业标准 |
| **Context 隔离** | ⭐⭐⭐⭐⭐ | 每任务创建新 BrowserContext，用 try/finally 确保关闭 |
| **Browser-as-a-Service** | ⭐⭐⭐⭐ 上升 | Browserbase/Browserless 等托管方案，外包基础设施管理 |
| **Chromium 启动参数优化** | ⭐⭐⭐⭐⭐ | `--disable-dev-shm-usage`（Docker 必需）、`--disable-gpu`、`--disable-extensions` |
| **psutil 进程树监控** | ⭐⭐⭐⭐ | 标准做法——sum(RSS) across Chromium 子进程树 |

> [!TIP]
> 我们的多维度回收策略 (requests/age/memory) 与行业最佳实践完全一致。Playwright 的原生多 Context 隔离设计使其比 Puppeteer 更适合生产环境的并发场景。

行业建议补充：
- **保守起步**: 并发数从 1-2 开始，根据压测结果逐步扩展
- **API 认证优先**: 用 API 而非浏览器 UI 完成登录，减少渲染开销
- **CI/CD 优化**: 仅安装需要的浏览器引擎 (`playwright install chromium`)

---

## 8. 待确认事项

1. **psutil 依赖**: `psutil` 是新增依赖。是否接受？可以用 `/proc` 文件系统替代但更脆弱。
2. **回收策略参数**: `max_age_seconds=1800` (30min) 和 `max_memory_mb=512` 是否合理？
3. **信号处理**: 信号处理器方案较复杂且与 asyncio 交互有隐患。是否接受 atexit 作为简单方案？
4. **Docker OOM**: 是否需要在 docker-compose 中为 MCP server 容器设置 `mem_limit`？
