# Phase 2: Cookie 过期检测 + 告警机制设计

> **模块**: `search_service/cookie_manager.py` (新建)  
> **依赖**: SearchServiceConfig  
> **被依赖**: ZhihuScraper, WeiboScraper  
> **日期**: 2026-05-05

---

## 1. 设计目标

构建统一的 Cookie 管理模块，为 ZhihuScraper 和 WeiboScraper 提供:
- Cookie JSON 文件的加载/保存
- 过期检测（`expires` 字段 + 时间戳比较）
- TTL 预警（过期前 N 小时告警）
- 半自动 Cookie 刷新辅助脚本

---

## 2. 类设计

### 2.1 CookieManager

```python
class CookieManager:
    """平台 Cookie 持久化管理器。

    统一管理知乎/微博等平台的 Cookie:
    - 加载: 从 JSON 文件读取 Playwright 兼容格式的 Cookie
    - 校验: 检查关键 Cookie 的 expires 字段
    - 预警: 距过期 < warning_hours 时记录 warning 日志
    - 保存: Cookie 刷新后写入磁盘

    Args:
        platform: 平台标识 ("zhihu", "weibo")。
        storage_dir: Cookie 根目录 (含平台子目录)。
        critical_cookies: 关键 Cookie 名称列表 (用于过期检测)。
        warning_hours: 过期前多少小时触发预警。

    Example::

        manager = CookieManager(
            platform="zhihu",
            storage_dir=Path("./data/cookies"),
            critical_cookies=["z_c0"],
            warning_hours=24,
        )
        cookies = await manager.load()  # 可能抛出 CookieExpiredError
    """

    # 每个平台的关键 Cookie 映射
    _PLATFORM_CRITICAL_COOKIES: dict[str, list[str]] = {
        "zhihu": ["z_c0"],
        "weibo": ["SUB"],
    }

    def __init__(
        self,
        platform: str,
        storage_dir: Path,
        critical_cookies: Optional[list[str]] = None,
        warning_hours: int = 24,
    ) -> None:
        self._platform = platform
        self._cookie_dir = storage_dir / platform
        self._critical_cookies = (
            critical_cookies
            or self._PLATFORM_CRITICAL_COOKIES.get(platform, [])
        )
        self._warning_hours = warning_hours
        self._logger = logging.getLogger(__name__)

    @property
    def cookie_file(self) -> Path:
        """Cookie JSON 文件路径。"""
        return self._cookie_dir / "cookies.json"

    async def load(self) -> list[dict]:
        """加载并校验 Cookie。

        Returns:
            Playwright 兼容的 Cookie dict 列表。

        Raises:
            CookieExpiredError: Cookie 文件不存在或关键 Cookie 已过期。
        """
        if not self.cookie_file.exists():
            self._logger.warning(
                "cookie_file_not_found",
                extra={"platform": self._platform, "path": str(self.cookie_file)},
            )
            raise CookieExpiredError(self._platform)

        try:
            cookies = json.loads(self.cookie_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._logger.error(
                "cookie_file_corrupt",
                extra={"platform": self._platform, "error": str(exc)},
            )
            raise CookieExpiredError(self._platform)

        if not isinstance(cookies, list) or not cookies:
            raise CookieExpiredError(self._platform)

        # 检查关键 Cookie 过期
        if self._is_expired(cookies):
            self._logger.warning(
                "cookie_expired",
                extra={"platform": self._platform},
            )
            raise CookieExpiredError(self._platform)

        # TTL 预警
        self._check_ttl_warning(cookies)

        return cookies

    async def save(self, cookies: list[dict]) -> None:
        """保存 Cookie 到磁盘。

        创建目录结构（如不存在）并以 pretty-print JSON 写入。

        Args:
            cookies: Playwright 兼容的 Cookie dict 列表。
        """
        self._cookie_dir.mkdir(parents=True, exist_ok=True)
        self.cookie_file.write_text(
            json.dumps(cookies, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._logger.info(
            "cookie_saved",
            extra={
                "platform": self._platform,
                "count": len(cookies),
                "path": str(self.cookie_file),
            },
        )

    def _is_expired(self, cookies: list[dict]) -> bool:
        """检查关键 Cookie 是否已过期。

        逻辑:
        1. 找到所有关键 Cookie (按名称匹配)
        2. 如果任一关键 Cookie 不存在 → 视为过期
        3. 如果 expires 字段存在且 < 当前时间戳 → 过期
        4. 如果 expires 字段不存在 → 视为会话 Cookie，不判定过期
        """
        now = time.time()

        for critical_name in self._critical_cookies:
            found = False
            for cookie in cookies:
                if cookie.get("name") == critical_name:
                    found = True
                    expires = cookie.get("expires")
                    if expires is not None and expires > 0:
                        if expires < now:
                            return True
                    break

            if not found:
                # 关键 Cookie 不存在
                return True

        return False

    def _check_ttl_warning(self, cookies: list[dict]) -> None:
        """检查关键 Cookie TTL 并输出预警日志。

        距过期 < warning_hours 时输出 WARNING 级别日志。
        """
        now = time.time()
        warning_threshold = now + (self._warning_hours * 3600)

        for critical_name in self._critical_cookies:
            for cookie in cookies:
                if cookie.get("name") == critical_name:
                    expires = cookie.get("expires")
                    if expires is not None and expires > 0:
                        if expires < warning_threshold:
                            hours_left = max(0, (expires - now) / 3600)
                            self._logger.warning(
                                "cookie_expiring_soon",
                                extra={
                                    "platform": self._platform,
                                    "cookie": critical_name,
                                    "hours_remaining": f"{hours_left:.1f}",
                                },
                            )
                    break

    async def health_check(self) -> bool:
        """检查 Cookie 是否存在且未过期。"""
        try:
            await self.load()
            return True
        except CookieExpiredError:
            return False
```

---

## 3. Cookie 半自动刷新脚本

### 3.1 设计

创建 `search_service/scripts/cookie_helper.py`:

```python
"""Cookie 半自动刷新脚本。

用法:
    python -m search_service.scripts.cookie_helper --platform zhihu
    python -m search_service.scripts.cookie_helper --platform weibo

工作流:
    1. 启动有头 Playwright 浏览器
    2. 自动导航到平台登录页
    3. 等待用户手动登录
    4. 用户确认登录完成后，自动提取并保存 Cookie
"""

class CookieHelper:
    """半自动 Cookie 刷新工具。

    打开有头浏览器 → 用户手动登录 → 自动保存 Cookie。

    Args:
        platform: 平台标识 ("zhihu", "weibo")。
        storage_dir: Cookie 存储目录。
    """

    _LOGIN_URLS: dict[str, str] = {
        "zhihu": "https://www.zhihu.com/signin",
        "weibo": "https://passport.weibo.com/sso/signin",
    }

    _LOGIN_SUCCESS_INDICATORS: dict[str, str] = {
        "zhihu": "zhihu.com/people",  # URL 含 /people/ 表示已登录
        "weibo": "weibo.com",          # 重定向回微博主站
    }

    async def run(self) -> None:
        """执行 Cookie 刷新流程。"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()

            # 导航到登录页
            await page.goto(self._LOGIN_URLS[self._platform])

            # 等待用户手动登录
            self._logger.info("请在浏览器中手动登录，完成后按回车继续...")
            input("按回车确认登录完成 > ")

            # 提取并保存 Cookie
            cookies = await context.cookies()
            manager = CookieManager(
                platform=self._platform,
                storage_dir=self._storage_dir,
            )
            await manager.save(cookies)

            self._logger.info(
                "cookie_refresh_complete",
                extra={"count": len(cookies)},
            )

            await browser.close()
```

### 3.2 使用方式

```bash
# 刷新知乎 Cookie
python -m search_service.scripts.cookie_helper --platform zhihu

# 刷新微博 Cookie
python -m search_service.scripts.cookie_helper --platform weibo
```

---

## 4. 告警机制

### 4.1 日志级别告警

```
WARNING  cookie_expiring_soon platform=zhihu cookie=z_c0 hours_remaining=12.5
WARNING  cookie_expired platform=weibo
ERROR    cookie_file_corrupt platform=zhihu error=JSONDecodeError(...)
```

### 4.2 结构化日志字段

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `platform` | str | 平台标识 |
| `cookie` | str | Cookie 名称 |
| `hours_remaining` | str | 剩余有效时间 |
| `path` | str | Cookie 文件路径 |

### 4.3 未来扩展: Prometheus Metrics

Phase 3 可新增:
```python
from prometheus_client import Gauge

cookie_ttl_hours = Gauge(
    "search_cookie_ttl_hours",
    "Hours until platform cookie expires",
    ["platform"],
)
```

### 4.4 未来扩展: Webhook 通知

如果需要即时通知（如企业微信/Slack），可在 `_check_ttl_warning` 中添加 Webhook:

```python
async def _send_expiry_alert(self, platform: str, hours_left: float) -> None:
    """发送 Cookie 过期预警到外部渠道。"""
    # Phase 3: 集成企业微信/Slack Webhook
    pass
```

---

## 5. 与 SearchMCPServer 的集成

### 5.1 Server 启动时初始化 CookieManager

```python
class SearchMCPServer:
    async def startup(self) -> None:
        # ... existing SearXNG/BrowserPool init ...

        # Cookie managers for platform scrapers
        self._zhihu_cookie_manager = CookieManager(
            platform="zhihu",
            storage_dir=self._config.cookie_storage_dir,
        )
        self._weibo_cookie_manager = CookieManager(
            platform="weibo",
            storage_dir=self._config.cookie_storage_dir,
        )

        # Log cookie status at startup
        for mgr in [self._zhihu_cookie_manager, self._weibo_cookie_manager]:
            healthy = await mgr.health_check()
            self._logger.info(
                "cookie_status",
                extra={"platform": mgr._platform, "available": healthy},
            )
```

### 5.2 Scraper 注入 CookieManager

```python
def _get_zhihu_scraper(self) -> ZhihuScraper:
    assert self._browser_pool is not None
    return ZhihuScraper(
        browser_pool=self._browser_pool,
        config=self._config,
        cookie_manager=self._zhihu_cookie_manager,
    )
```

---

## 6. 验收标准

| 测试用例 | 说明 |
|:---|:---|
| `test_load_valid_cookies` | 有效 Cookie 文件 → 返回列表 |
| `test_load_no_file` | 文件不存在 → CookieExpiredError |
| `test_load_corrupt_json` | JSON 格式错误 → CookieExpiredError |
| `test_load_empty_list` | 空列表 → CookieExpiredError |
| `test_expired_by_timestamp` | expires < now → CookieExpiredError |
| `test_session_cookie_no_expire` | expires 不存在 → 不视为过期 |
| `test_critical_cookie_missing` | z_c0 不在列表中 → CookieExpiredError |
| `test_ttl_warning` | 距过期 < 24h → 输出 WARNING 日志 |
| `test_ttl_no_warning` | 距过期 > 24h → 无 WARNING |
| `test_save_creates_dir` | 目录不存在 → 自动创建 |
| `test_save_overwrites` | 文件已存在 → 覆盖写入 |
| `test_health_check_true` | Cookie 有效 → True |
| `test_health_check_false` | Cookie 过期 → False |

---

## 7. 行业价值分析与主流实现方式

### 7.1 功能实际价值评估

| 维度 | 评估 | 说明 |
|:---|:---|:---|
| **基础设施必要性** | ⭐⭐⭐⭐⭐ | Cookie 管理是所有需要登录态的垂直平台 scraper 的前置依赖 |
| **运维价值** | ⭐⭐⭐⭐ | TTL 预警机制可提前发现 Cookie 过期，避免无声降级 |
| **复用性** | ⭐⭐⭐⭐⭐ | 统一 CookieManager 供 Zhihu/Weibo 共用，减少重复代码 |

### 7.2 业界主流实现方式

| 方案 | 采用度 | 说明 |
|:---|:---|:---|
| **加密存储 + Redis/Vault** | ⭐⭐⭐⭐⭐ | 生产环境标准——Cookie 加密后存入 Redis 或 HashiCorp Vault |
| **事件驱动重认证** | ⭐⭐⭐⭐ | 遇到 401/403 自动触发重新登录流程 |
| **Context 一致性** | ⭐⭐⭐⭐ | Cookie 必须与 User-Agent、IP、指纹保持一致 |
| **Session 心跳** | ⭐⭐⭐ | 周期性低强度导航延长 session 有效期 |
| **明文 JSON 文件** | ⭐⭐ 仅 dev | 仅适用于开发/原型阶段 |

> [!TIP]
> 行业标准：生产环境使用集中式 key-value 存储（Redis）+ 加密，支持分布式 worker 共享 session。我们当前明文 JSON 方案适合 Phase 2 MVP，Phase 3 应迁移到 Redis + AES 加密。

### 7.3 与行业最佳实践的差距

| 项 | 当前设计 | 行业实践 | 差距 | 建议 |
|:---|:---|:---|:---|:---|
| 存储安全 | 明文 JSON | 加密 + Vault | 高 | Phase 3 加 AES 加密层 |
| 自动刷新 | 手动半自动 | 事件驱动重认证 | 中 | Phase 3 实现 401/403 自动触发 |
| 集中管理 | 本地文件 | Redis + 分布式共享 | 中 | 与 Redis Cache 同步迁移 |
| Context 一致性 | 未校验 | Cookie-UA-IP 绑定校验 | 中 | 加验证逻辑 |

---

## 8. 待确认事项

> [!NOTE]

1. **Cookie 文件目录**: `data/cookies/zhihu/cookies.json` 这个路径是否需要加入 `.gitignore`？Cookie 包含登录凭证，不应提交到代码仓库。

2. **Cookie 加密存储**: 当前设计为明文 JSON。是否需要对 Cookie value 进行加密存储（如 AES）？这会增加复杂度但提高安全性。

3. **Cookie Pool**: 是否需要支持多个 Cookie 文件（如 `cookies_1.json`, `cookies_2.json`）实现账号轮换？当前设计为单文件，如果需要扩展，CookieManager 的 `load()` 可以随机选择一个有效的 Cookie 文件。

4. **企业微信通知**: 是否在 Phase 2 就集成 Cookie 过期的企业微信/Slack 通知？还是 Phase 3 再做？
