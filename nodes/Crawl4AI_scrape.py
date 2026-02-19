"""
Node 4 (Temp): Crawl4AI Scrape
Scrapes pages with Crawl4AI and returns markdown for each URL.
Handles PDF URLs with the PDF crawler strategy.
"""

from typing import TYPE_CHECKING, List, Dict

from nodes.base import BaseNode

if TYPE_CHECKING:
    from algorithm import ResearchState, ProgressTracker


class Crawl4AIScrapeNode(BaseNode):
    """
    Scrapes markdown from a list of URLs.

    Input State Keys:
        - validated_urls: List[str]

    Output State Keys:
        - scraped_content: List[Dict] with keys: url, text
    """

    def __init__(self, timeout_ms: int = 15000):
        super().__init__()
        self.timeout_ms = timeout_ms

    async def execute(self, state: "ResearchState", progress: "ProgressTracker") -> "ResearchState":
        from crawl4ai import AsyncWebCrawler
        from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, CacheMode
        from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy

        urls = state.get("validated_urls") or state.get("urls") or []
        if not urls:
            state["scraped_content"] = []
            progress.update("📄 Scraping Complete", "No URLs to scrape")
            return state

        progress.update("📄 Scraping Starting", f"Scraping {len(urls)} URLs with Crawl4AI...")

        pdf_urls = [u for u in urls if ".pdf" in u.lower()]
        html_urls = [u for u in urls if u not in pdf_urls]

        scraped: List[Dict] = []

        if html_urls:
            browser_config = BrowserConfig(verbose=True, headless=True)
            run_config = CrawlerRunConfig(
                process_iframes=True,
                remove_overlay_elements=True,
                cache_mode=CacheMode.BYPASS
            )
            async with AsyncWebCrawler(config=browser_config) as crawler:
                results = await crawler.arun_many(html_urls, run_config=run_config)
                for i, result in enumerate(results):
                    url = html_urls[i]
                    if result.success:
                        markdown = result.markdown or ""
                        scraped.append({"url": url, "text": markdown})
                    else:
                        print(f"⚠️  Crawl failed for {url}: {result.error_message}")

        if pdf_urls:
            pdf_crawler_strategy = PDFCrawlerStrategy()
            pdf_scraping_strategy = PDFContentScrapingStrategy()
            run_config = CrawlerRunConfig(scraping_strategy=pdf_scraping_strategy)
            async with AsyncWebCrawler(crawler_strategy=pdf_crawler_strategy) as crawler:
                for url in pdf_urls:
                    result = await crawler.arun(url=url, config=run_config)
                    if result.success:
                        if result.markdown and hasattr(result.markdown, "raw_markdown"):
                            markdown = result.markdown.raw_markdown
                        else:
                            markdown = str(result.markdown or "")
                        scraped.append({"url": url, "text": markdown})
                    else:
                        print(f"⚠️  PDF crawl failed for {url}: {result.error_message}")

        state["scraped_content"] = scraped
        progress.update(
            "📄 Scraping Complete",
            f"Scraped {len(scraped)} documents from {len(urls)} URLs",
            {"source_documents": len(urls), "scraped_documents": len(scraped)}
        )

        return state
