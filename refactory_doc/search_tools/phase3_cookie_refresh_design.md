# Phase 3: Cookie 半自动刷新流程设计

> **模块**: `search_service/scripts/cookie_helper.py` (新建)  
> **依赖**: Playwright (有头模式), CookieManager  
> **日期**: 2026-05-05

---

## 1. 设计目标

提供 Cookie 的半自动刷新流程:
- 打开有头浏览器 → 用户手动登录 → 自动保存 Cookie
- 定时检测 Cookie TTL → 预警提醒刷新
- 可选: 通知渠道 (企业微信/Slack)

---

## 2. Cookie 刷新脚本

```python
"""Cookie 半自动刷新工具。

用法:
    python -m search_service.scripts.cookie_helper --platform zhihu
    python -m search_service.scripts.cookie_helper --platform weibo
    python -m search_service.scripts.cookie_helper --check-all

--check-all: 检查所有平台 Cookie 状态，不启动浏览器。
"""

class CookieHelper:
    """半自动 Cookie 刷新工具。

    流程:
    1. 启动有头 Playwright 浏览器 (用户可见)
    2. 自动导航到平台登录页
    3. 用户手动完成登录
    4. 用户按回车确认后，自动提取 Cookie
    5. 保存到 data/cookies/{platform}/cookies.json

    Args:
        platform: "zhihu" or "weibo"
        storage_dir: Cookie 根目录
    """

    _LOGIN_URLS: dict[str, str] = {
        "zhihu": "https://www.zhihu.com/signin",
        "weibo": "https://passport.weibo.com/sso/signin",
    }

    _SUCCESS_INDICATORS: dict[str, list[str]] = {
        "zhihu": ["zhihu.com/people", "zhihu.com/"],
        "weibo": ["weibo.com/", "m.weibo.cn/"],
    }

    _CRITICAL_COOKIES: dict[str, list[str]] = {
        "zhihu": ["z_c0", "d_c0"],
        "weibo": ["SUB", "SUBP"],
    }

    async def refresh(self) -> None:
        """执行 Cookie 刷新。"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(
                viewport={"width": 1280, "height": 800},
                locale="zh-CN",
            )
            page = await context.new_page()

            # 1. 导航到登录页
            login_url = self._LOGIN_URLS[self._platform]
            await page.goto(login_url)
            self._logger.info(f"请在浏览器中登录 {self._platform}...")

            # 2. 等待用户手动登录
            print(f"\n{'='*50}")
            print(f"请在浏览器中完成 {self._platform} 登录")
            print(f"登录成功后按回车键继续...")
            print(f"{'='*50}\n")
            input()

            # 3. 验证登录状态
            current_url = page.url
            is_logged_in = any(
                indicator in current_url
                for indicator in self._SUCCESS_INDICATORS[self._platform]
            )
            if not is_logged_in:
                self._logger.warning(
                    "login_verification_uncertain",
                    extra={"url": current_url},
                )

            # 4. 提取 Cookie
            cookies = await context.cookies()

            # 5. 验证关键 Cookie 存在
            cookie_names = {c["name"] for c in cookies}
            critical = self._CRITICAL_COOKIES[self._platform]
            missing = [c for c in critical if c not in cookie_names]
            if missing:
                self._logger.error(
                    "critical_cookies_missing",
                    extra={"missing": missing, "platform": self._platform},
                )
                print(f"⚠️  缺少关键 Cookie: {missing}")
                print("请确认已正确登录后重试。")
                await browser.close()
                return

            # 6. 保存
            manager = CookieManager(
                platform=self._platform,
                storage_dir=self._storage_dir,
            )
            await manager.save(cookies)

            print(f"\n✅ {self._platform} Cookie 已保存 ({len(cookies)} 条)")

            # 7. 显示 TTL 信息
            for c in cookies:
                if c["name"] in critical:
                    expires = c.get("expires")
                    if expires and expires > 0:
                        import datetime
                        expiry_dt = datetime.datetime.fromtimestamp(expires)
                        print(f"   {c['name']}: 过期时间 {expiry_dt}")

            await browser.close()

    async def check_all(self) -> None:
        """检查所有平台 Cookie 状态。"""
        print("\n=== Cookie 状态检查 ===\n")
        for platform in self._LOGIN_URLS:
            manager = CookieManager(
                platform=platform,
                storage_dir=self._storage_dir,
            )
            try:
                cookies = await manager.load()
                # 计算 TTL
                critical = self._CRITICAL_COOKIES.get(platform, [])
                for c in cookies:
                    if c["name"] in critical:
                        expires = c.get("expires")
                        if expires and expires > 0:
                            hours_left = max(0, (expires - time.time()) / 3600)
                            status = "✅" if hours_left > 24 else "⚠️"
                            print(f"  {status} {platform}/{c['name']}: "
                                  f"{hours_left:.0f}h remaining")
                        else:
                            print(f"  ℹ️  {platform}/{c['name']}: session cookie (no expiry)")
            except CookieExpiredError:
                print(f"  ❌ {platform}: Cookie 过期或不存在")
        print()
```

---

## 3. CLI 入口

```python
# search_service/scripts/cookie_helper.py

def main():
    parser = argparse.ArgumentParser(description="Cookie 刷新工具")
    parser.add_argument("--platform", choices=["zhihu", "weibo"])
    parser.add_argument("--check-all", action="store_true")
    parser.add_argument("--storage-dir", default="./data/cookies")
    args = parser.parse_args()

    helper = CookieHelper(
        platform=args.platform,
        storage_dir=Path(args.storage_dir),
    )

    if args.check_all:
        asyncio.run(helper.check_all())
    elif args.platform:
        asyncio.run(helper.refresh())
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

---

## 4. 定时检测 (Crontab)

```bash
# 每天上午 9 点检查 Cookie 状态
0 9 * * * cd /home/tianwei/workspace/deep_research_agent && \
  .venv/bin/python -m search_service.scripts.cookie_helper --check-all \
  >> data/logs/cookie_check.log 2>&1
```

---

## 5. 行业价值分析与主流实现方式

### 5.1 功能实际价值评估

| 维度 | 评估 | 说明 |
|:---|:---|:---|
| **运维效率** | ⭐⭐⭐⭐ | 半自动脚本将 Cookie 刷新从 10 分钟手动操作降到 2 分钟 |
| **可靠性** | ⭐⭐⭐ | 减少人为错误（如忘记刷新 Cookie 导致服务降级） |
| **优先级** | ⭐⭐⭐ | Phase 3 优先级合理——Phase 2 手动刷新足以支撑低频使用 |

### 5.2 业界主流实现方式

| 方案 | 采用度 | 说明 |
|:---|:---|:---|
| **半自动浏览器脚本** | ⭐⭐⭐⭐ | 我们采用的方案，行业中小团队标准做法 |
| **全自动登录 (高风险)** | ⭐⭐ | 自动化手机号+验证码登录，平台封号风险极高 |
| **Cookie Pool + 轮换** | ⭐⭐⭐ | 多账号 Cookie 池，过期的自动下线 |
| **托管服务** | ⭐⭐⭐⭐ 上升 | 付费 API 服务完全外包 session 管理 |

> [!TIP]
> 半自动方案与行业中小团队实践一致。行业共识是不尝试全自动登录（封号风险不可接受）。定时 crontab 检查 + 预警通知是标准运维模式。

---

## 6. 待确认事项

1. **通知渠道**: Cookie 即将过期时，是否需要发送企业微信/Slack 通知？
2. **自动化程度**: 是否尝试 Playwright 自动登录（手机号+验证码流程）？风险较高且平台可能封号。
3. **Cookie 加密**: 保存到磁盘的 Cookie 是否需要加密？
