# S03: Browser Pool (Playwright)

> **Phase**: 2 | **预估工时**: 1-2 天  
> **产出文件**: `search_service/browser/pool.py`, `search_service/browser/stealth.py`, `search_service/resources/stealth.min.js`  
> **依赖**: S01 (config, exceptions)  
> **下游**: S04a-d (所有 Playwright Scrapers)

---

## 1. 目标

- 实现 Playwright Browser 实例池，管理 BrowserContext 生命周期
- 信号量控制并发（最大 `browser_max_concurrency` 个并发 Context）
- 注入 stealth.js 绕过基础反爬检测
- 进程级资源保护（单 Browser 最大请求数、内存上限）
- 异步上下文管理器接口，确保资源自动释放

---

## 2. BrowserPool 核心接口

```python
class BrowserPool:
    """Playwright BrowserContext 池。

    通过 asyncio.Semaphore 控制并发。每个 Context 独立隔离（Cookie/Cache/Session）。
    支持单 Browser 请求数上限 + 内存监控。

    Usage:
        pool = BrowserPool(config)
        await pool.start()

        async with pool.acquire() as context:
            page = await context.new_page()
            await page.goto("https://example.com")
            content = await page.content()

        await pool.shutdown()
    """

    def __init__(self, config: SearchServiceConfig):
        self._max_concurrency = config.browser_max_concurrency
        self._max_requests = config.browser_max_requests_per_instance
        self._memory_limit_mb = config.browser_memory_limit_mb
        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        self._browser: Browser | None = None
        self._playwright: Playwright | None = None
        self._request_count = 0
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)

    async def start(self) -> None:
        """启动 Playwright Browser 进程。"""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                f"--js-flags=--max-old-space-size={self._memory_limit_mb}",
            ],
        )
        self._logger.info("browser_pool_started", extra={
            "max_concurrency": self._max_concurrency,
        })

    @asynccontextmanager
    async def acquire(
        self,
        timeout: float = 10.0,
        cookies: list[dict] | None = None,
    ) -> AsyncGenerator[BrowserContext, None]:
        """获取一个 BrowserContext。

        Args:
            timeout: 等待信号量的超时时间 (秒)。
            cookies: 预设 Cookie（用于知乎/微博等登录态）。

        Yields:
            BrowserContext (已注入 stealth.js)。

        Raises:
            BrowserPoolExhaustedError: 等待超时。
        """
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            raise BrowserPoolExhaustedError(self._max_concurrency, timeout)

        try:
            await self._maybe_recycle_browser()
            context = await self._create_context(cookies)
            try:
                yield context
            finally:
                await context.close()
                async with self._lock:
                    self._request_count += 1
        finally:
            self._semaphore.release()

    async def _create_context(self, cookies: list[dict] | None) -> BrowserContext:
        """创建新 Context 并注入 stealth.js。"""
        assert self._browser is not None
        context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=self._random_user_agent(),
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
        )
        # 注入 stealth.js（每个 context 的每个新页面自动注入）
        stealth_js = StealthInjector.load_script()
        await context.add_init_script(stealth_js)

        if cookies:
            await context.add_cookies(cookies)

        return context

    async def _maybe_recycle_browser(self) -> None:
        """单 Browser 请求数超限时回收重建。"""
        async with self._lock:
            if self._request_count >= self._max_requests:
                self._logger.info("browser_recycle", extra={
                    "requests_served": self._request_count,
                })
                if self._browser:
                    await self._browser.close()
                assert self._playwright is not None
                self._browser = await self._playwright.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage"],
                )
                self._request_count = 0

    async def shutdown(self) -> None:
        """关闭 Browser 和 Playwright。"""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._logger.info("browser_pool_shutdown")

    @staticmethod
    def _random_user_agent() -> str:
        """随机选择 User-Agent (Chrome/Firefox/Edge)。"""
        agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
        ]
        return random.choice(agents)
```

---

## 3. Stealth 注入 (`browser/stealth.py`)

```python
class StealthInjector:
    """管理 stealth.min.js 的加载和注入。

    stealth.js 作用：
    1. 覆盖 navigator.webdriver 属性 (返回 undefined)
    2. 修改 Chrome DevTools detection 指纹
    3. 伪装 WebGL/Canvas 渲染器信息
    4. 拦截 Headless Chrome 特征检测
    """

    _SCRIPT_PATH = Path(__file__).parent.parent / "resources" / "stealth.min.js"
    _cached_script: str | None = None

    @classmethod
    def load_script(cls) -> str:
        """加载 stealth.js 脚本（内存缓存）。"""
        if cls._cached_script is None:
            if not cls._SCRIPT_PATH.exists():
                raise FileNotFoundError(
                    f"stealth.min.js not found at {cls._SCRIPT_PATH}. "
                    "Run 'npx extract-stealth-evasions' to generate it."
                )
            cls._cached_script = cls._SCRIPT_PATH.read_text(encoding="utf-8")
        return cls._cached_script
```

### stealth.min.js 获取方式

```bash
# 在项目根目录运行（需要 Node.js）
npx extract-stealth-evasions
mv stealth.min.js search_service/resources/stealth.min.js
```

---

## 4. 生命周期管理

```
Application Startup
    │
    ▼
BrowserPool.start()
    │  ── 启动 Playwright + Chromium 进程
    │
    ▼
╔═══════════════════════════════════════════╗
║  Semaphore(max_concurrency=3)            ║
║                                           ║
║  Request 1 ─► acquire() ─► Context ─► ✓  ║
║  Request 2 ─► acquire() ─► Context ─► ✓  ║
║  Request 3 ─► acquire() ─► Context ─► ✓  ║
║  Request 4 ─► acquire() ─► wait...       ║
║                         ─► timeout ─► ✗  ║
║                              BrowserPoolExhaustedError
╚═══════════════════════════════════════════╝
    │
    │  每 max_requests_per_instance 次请求
    ▼
_maybe_recycle_browser()
    │  ── 关闭旧 Browser，启动新 Browser
    │
    ▼
Application Shutdown
    │
    ▼
BrowserPool.shutdown()
    │  ── 关闭 Browser + Playwright
```

---

## 5. 验收标准

```bash
python -m pytest tests/test_search_service/test_browser_pool.py -v
```

| 测试用例 | 说明 |
|:---|:---|
| `test_acquire_and_release` | 获取 Context，完成后信号量释放 |
| `test_concurrency_limit` | 超过 max_concurrency 时排队 |
| `test_timeout_raises_exhausted` | 等待超时抛出 BrowserPoolExhaustedError |
| `test_browser_recycle` | 请求数到达上限后 Browser 重建 |
| `test_stealth_injection` | Context 中 navigator.webdriver = undefined |
| `test_cookie_injection` | 预设 Cookie 正确注入 |
| `test_shutdown_cleanup` | shutdown 后所有资源释放 |
