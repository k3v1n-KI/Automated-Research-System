"""
Real-world scrape failure analysis using actual URLs from the research algorithm
Tests against real websites to understand failure modes
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nodes.scrape import ScrapeNode


class ProgressTracker:
    """Track scrape progress"""
    def __init__(self):
        self.updates = []
    
    def update(self, title, message, data=None):
        self.updates.append({
            "title": title,
            "message": message,
            "data": data
        })


async def test_real_urls():
    """Test scraping against real URLs found by algorithm"""
    
    # Real URLs found by the algorithm searching for "hospitals in toronto"
    test_urls = [
        "https://www.torontohealth.ca/",
        "https://www.healthyalberta.com/glossary/hospitals",
        "https://www.sjwh.ca/",
        "https://www.smh.ca/",
        "https://www.sinaihealth.ca/",
        "https://www.michaelgarron.ca/",
        "https://www.hhs.ca/",
        "https://www.shothamilton.ca/",
        "https://www.caredtoronto.ca/",
        "https://www.stmichaelshospital.com/",
        "https://toronto.jamsports.com/pdfs/hospital_list.pdf",  # Known failure: PDF
        "https://socialwork.utoronto.ca/wp-content/uploads/2015/01/Public-Hospitals-List.pdf",  # Known failure: PDF
    ]
    
    print("\n" + "="*80)
    print("REAL-WORLD SCRAPE FAILURE ANALYSIS")
    print("="*80)
    print(f"\nTesting {len(test_urls)} URLs from actual algorithm execution")
    print("Testing against: hospitals in Toronto search results\n")
    
    scraper = ScrapeNode(timeout_ms=15000)
    progress = ProgressTracker()
    
    state = {
        'validated_urls': test_urls,
        'scraped_content': [],
        'error': None
    }
    
    # Run the scraper
    result_state = await scraper.execute(state, progress)
    
    # Analyze results
    successful = result_state['scraped_content']
    failed_count = len(test_urls) - len(successful)
    
    print(f"\n📊 RESULTS:")
    print(f"  Total URLs: {len(test_urls)}")
    print(f"  ✓ Successful: {len(successful)}")
    print(f"  ✗ Failed: {failed_count}")
    print(f"  Success Rate: {(len(successful)/len(test_urls)*100):.1f}%")
    
    # Identify failures
    successful_urls = {item['url'] for item in successful}
    failed_urls = [url for url in test_urls if url not in successful_urls]
    
    failure_analysis = {
        "timestamp": datetime.now().isoformat(),
        "total_urls": len(test_urls),
        "successful": len(successful),
        "failed": len(failed_urls),
        "success_rate": f"{(len(successful)/len(test_urls)*100):.1f}%",
        "failures": []
    }
    
    print(f"\n❌ FAILED URLS ({len(failed_urls)}):")
    for url in failed_urls:
        print(f"\n   • {url}")
        
        # Categorize the failure
        failure_reason = "Unknown"
        failure_category = "OTHER"
        
        if ".pdf" in url.lower():
            failure_reason = "PDF file - Playwright cannot process downloads"
            failure_category = "PDF_DOWNLOAD"
        elif "jamsports" in url:
            failure_reason = "PDF file - Playwright cannot process downloads"
            failure_category = "PDF_DOWNLOAD"
        elif "socialwork.utoronto" in url:
            failure_reason = "PDF file - Playwright cannot process downloads"
            failure_category = "PDF_DOWNLOAD"
        
        print(f"     Reason: {failure_reason}")
        print(f"     Category: {failure_category}")
        
        failure_analysis["failures"].append({
            "url": url,
            "reason": failure_reason,
            "category": failure_category
        })
    
    # Analyze successful scrapes
    print(f"\n✓ SUCCESSFUL SCRAPES ({len(successful)}):")
    content_stats = {
        "total_html_size": 0,
        "total_text_length": 0,
        "min_text": float('inf'),
        "max_text": 0,
        "empty_pages": 0,
        "urls_scraped": []
    }
    
    for item in successful:
        url = item['url']
        html_size = len(item.get('html', ''))
        text_length = len(item.get('text', ''))
        
        content_stats["total_html_size"] += html_size
        content_stats["total_text_length"] += text_length
        content_stats["min_text"] = min(content_stats["min_text"], text_length)
        content_stats["max_text"] = max(content_stats["max_text"], text_length)
        
        if text_length < 50:
            content_stats["empty_pages"] += 1
        
        content_stats["urls_scraped"].append({
            "url": url,
            "html_size": html_size,
            "text_length": text_length,
            "is_minimal": text_length < 100
        })
        
        print(f"\n   • {url}")
        print(f"     HTML: {html_size:,} bytes | Text: {text_length:,} chars")
        if text_length < 100:
            print(f"     ⚠ WARNING: Minimal content ({text_length} chars)")
    
    # Summary statistics
    print(f"\n" + "="*80)
    print("CONTENT QUALITY ANALYSIS")
    print("="*80)
    print(f"\nHTML Size Statistics:")
    print(f"  Total: {content_stats['total_html_size']:,} bytes")
    print(f"  Average per page: {content_stats['total_html_size']/max(1, len(successful)):,.0f} bytes")
    
    print(f"\nText Content Statistics:")
    print(f"  Total: {content_stats['total_text_length']:,} characters")
    if len(successful) > 0:
        print(f"  Average per page: {content_stats['total_text_length']/len(successful):,.0f} chars")
    print(f"  Min per page: {content_stats['min_text']:,} chars")
    print(f"  Max per page: {content_stats['max_text']:,} chars")
    
    print(f"\nData Quality Issues:")
    print(f"  Minimal content pages: {content_stats['empty_pages']} ({content_stats['empty_pages']/max(1, len(successful))*100:.1f}%)")
    
    # Save failure report
    report_file = Path(__file__).parent / "scrape_failure_report.json"
    failure_analysis["content_stats"] = {
        "total_urls_scraped": len(successful),
        "total_html_size": content_stats['total_html_size'],
        "total_text_length": content_stats['total_text_length'],
        "avg_html_size": content_stats['total_html_size'] / max(1, len(successful)),
        "avg_text_length": content_stats['total_text_length'] / max(1, len(successful)),
        "minimal_content_pages": content_stats['empty_pages']
    }
    
    with open(report_file, 'w') as f:
        json.dump(failure_analysis, f, indent=2)
    
    print(f"\n💾 Report saved to: {report_file}")
    
    return failure_analysis


if __name__ == "__main__":
    result = asyncio.run(test_real_urls())
    print("\n✅ Real-world scrape failure analysis complete")
