# S04a: Page Scraper (通用 URL → Markdown)

> **Phase**: 2 | **预估工时**: 1 天  
> **产出文件**: `search_service/backends/page_scraper.py`  
> **依赖**: S01 (models, exceptions), S03 (BrowserPool)  
> **下游**: S05 (MCP Server 暴露 `scrape_url` 工具)

---

## 1. 目标

实现通用的 URL 内容抓取工具：
- 输入 URL → 输出 Markdown 格式正文
- Playwright 渲染（支持 SPA/动态加载页面）
- 自动移除导航栏、侧边栏、广告等非正文内容
- HTML → Markdown 转换（保留标题层级、列表、代码块）
- 超时和内容长度保护

---

## 2. 核心接口

```python
class PageScraper:
    """通用页面抓取器：URL → Markdown。

    使用 Playwright 渲染页面后提取正文。
    依赖 BrowserPool 获取 BrowserContext。

    Args:
        browser_pool: BrowserPool 实例。
    """

    def __init__(self, browser_pool: BrowserPool):
        self._pool = browser_pool
        self._logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "page_scraper"

    async def scrape(
        self,
        url: str,
        timeout_seconds: float = 15.0,
        max_content_length: int = 50000,
        wait_for_selector: str | None = None,
    ) -> ScrapeResponse:
        """抓取页面内容。

        Args:
            url: 目标 URL。
            timeout_seconds: 页面加载超时。
            max_content_length: 内容最大字符数。
            wait_for_selector: 等待指定 DOM 元素出现后再提取。

        Returns:
            ScrapeResponse

        Raises:
            ContentExtractionError: 提取失败。
        """
        async with self._pool.acquire() as context:
            page = await context.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded",
                              timeout=timeout_seconds * 1000)

                if wait_for_selector:
                    await page.wait_for_selector(
                        wait_for_selector, timeout=5000
                    )

                # 移除非正文元素
                await self._remove_noise(page)

                # 提取正文 HTML
                main_html = await self._extract_main_content(page)

                # HTML → Markdown
                content = self._html_to_markdown(main_html)
                content = content[:max_content_length]

                title = await page.title() or ""

                return ScrapeResponse(
                    url=url,
                    title=title,
                    content=content,
                    content_length=len(content),
                    metadata={"source": "playwright"},
                )
            except Exception as e:
                raise ContentExtractionError(url, str(e))
            finally:
                await page.close()

    async def _remove_noise(self, page) -> None:
        """移除导航栏、侧边栏、广告等。"""
        noise_selectors = [
            "nav", "header", "footer",
            "[role='navigation']", "[role='banner']",
            ".sidebar", ".ad", ".advertisement",
            ".comment", ".comments",
            "script", "style", "iframe",
        ]
        for selector in noise_selectors:
            await page.evaluate(f"""
                document.querySelectorAll('{selector}')
                    .forEach(el => el.remove());
            """)

    async def _extract_main_content(self, page) -> str:
        """提取正文 HTML。优先使用 article/main 标签。"""
        # 1. 尝试 <article> 标签
        article = await page.query_selector("article")
        if article:
            return await article.inner_html()

        # 2. 尝试 <main> 标签
        main = await page.query_selector("main")
        if main:
            return await main.inner_html()

        # 3. 尝试最大文本密度的 div (Readability 启发式)
        result = await page.evaluate("""
            () => {
                const divs = document.querySelectorAll('div, section');
                let best = null, bestScore = 0;
                for (const div of divs) {
                    const text = div.innerText || '';
                    const score = text.length - (div.querySelectorAll('a').length * 30);
                    if (score > bestScore) {
                        bestScore = score;
                        best = div;
                    }
                }
                return best ? best.innerHTML : document.body.innerHTML;
            }
        """)
        return result

    @staticmethod
    def _html_to_markdown(html: str) -> str:
        """HTML → Markdown 转换。"""
        try:
            import markdownify
            return markdownify.markdownify(
                html,
                heading_style="ATX",
                strip=["img", "video", "audio"],
                convert=["p", "h1", "h2", "h3", "h4", "h5", "h6",
                         "ul", "ol", "li", "pre", "code",
                         "blockquote", "table", "tr", "td", "th",
                         "strong", "em", "a"],
            ).strip()
        except ImportError:
            # Fallback: 简单文本提取
            import re
            text = re.sub(r"<[^>]+>", "", html)
            return re.sub(r"\s+", " ", text).strip()
```

---

## 3. 依赖库

| 库 | 用途 | 是否新增 |
|:---|:---|:---|
| `playwright` | 页面渲染 | 已有 (S03) |
| `markdownify` | HTML → Markdown | **新增** |

```bash
pip install markdownify
```

---

## 4. 验收标准

```bash
python -m pytest tests/test_search_service/test_page_scraper.py -v
```

| 测试用例 | 说明 |
|:---|:---|
| `test_scrape_static_page` | 抓取静态 HTML 页面 |
| `test_scrape_spa_page` | 抓取 SPA 动态加载页面 |
| `test_content_length_limit` | 超过 max_content_length 截断 |
| `test_noise_removal` | 移除 nav/footer/sidebar |
| `test_html_to_markdown` | h1/p/ul/code 正确转换 |
| `test_timeout_handling` | 超时抛 ContentExtractionError |
| `test_invalid_url` | 不可达 URL 的错误处理 |
