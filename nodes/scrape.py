"""
Node 4: Scrape
Scrapes validated URLs using Playwright browser automation.
"""

from datetime import datetime
from typing import TYPE_CHECKING

from nodes.base import BaseNode

if TYPE_CHECKING:
    from algorithm import ResearchState, ProgressTracker


class ScrapeNode(BaseNode):
    """
    Scrapes validated URLs to extract HTML and text content.
    
    Input State Keys:
        - validated_urls: List of URLs to scrape
    
    Output State Keys:
        - scraped_content: List of {"url", "html", "text", "timestamp"}
    """
    
    def __init__(self, timeout_ms: int = 15000):
        super().__init__()
        self.timeout_ms = timeout_ms
    
    async def execute(self, state: "ResearchState", progress: "ProgressTracker") -> "ResearchState":
        """Scrape validated URLs"""
        
        urls = state['validated_urls']
        
        progress.update(
            "📄 Scraping Starting",
            f"Scraping {len(urls)} validated URLs..."
        )
        
        scraped = []
        
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            print("❌ Playwright not installed. Install with: pip install playwright")
            state['error'] = "Playwright not available"
            return state
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            
            for idx, url in enumerate(urls):
                if (idx + 1) % 5 == 0:
                    progress.update(
                        "📄 Scraping",
                        f"Scraped {idx + 1}/{len(urls)} URLs"
                    )
                
                try:
                    page = await browser.new_page()
                    await page.goto(url, wait_until="networkidle", timeout=self.timeout_ms)
                    
                    html = await page.content()
                    text = await page.evaluate('() => document.body.innerText')
                    
                    await page.close()
                    
                    scraped.append({
                        "url": url,
                        "html": html,
                        "text": text,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    print(f"⚠️  Failed to scrape {url}: {e}")
            
            await browser.close()
        
        state['scraped_content'] = scraped
        
        progress.update(
            "📄 Scraping Complete",
            f"Successfully scraped {len(scraped)}/{len(urls)} URLs",
            {"scraped": len(scraped), "failed": len(urls) - len(scraped)}
        )
        
        return state
