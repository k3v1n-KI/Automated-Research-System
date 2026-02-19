"""
Comparison test: Original vs Improved Scraper
Tests both against the same URLs and generates a detailed comparison report
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from nodes.scrape import ScrapeNode
from improved_scraper import ImprovedScrapeNode


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


async def test_both_scrapers():
    """Test original and improved scrapers against same URLs"""
    
    # Real URLs found by the algorithm
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
        "https://toronto.jamsports.com/pdfs/hospital_list.pdf",
        "https://socialwork.utoronto.ca/wp-content/uploads/2015/01/Public-Hospitals-List.pdf",
    ]
    
    print("\n" + "="*90)
    print("SCRAPER COMPARISON TEST: Original vs Improved")
    print("="*90)
    print(f"\nTesting {len(test_urls)} URLs against both scrapers\n")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_urls": len(test_urls),
        "original": None,
        "improved": None,
        "comparison": None
    }
    
    # Test Original Scraper
    print("🔄 Testing ORIGINAL Scraper (15s timeout, no validation)...")
    print("-" * 90)
    
    progress_orig = ProgressTracker()
    state_orig = {
        'validated_urls': test_urls,
        'scraped_content': [],
        'error': None
    }
    
    original = ScrapeNode(timeout_ms=15000)
    result_orig = await original.execute(state_orig, progress_orig)
    
    orig_success = result_orig['scraped_content']
    orig_failed = len(test_urls) - len(orig_success)
    
    print(f"\n✓ Original Scraper Results:")
    print(f"  Successful: {len(orig_success)}")
    print(f"  Failed: {orig_failed}")
    print(f"  Success Rate: {(len(orig_success)/len(test_urls)*100):.1f}%")
    
    results["original"] = {
        "successful": len(orig_success),
        "failed": orig_failed,
        "success_rate": f"{(len(orig_success)/len(test_urls)*100):.1f}%",
        "urls": [
            {
                "url": item['url'],
                "html_size": len(item.get('html', '')),
                "text_length": len(item.get('text', ''))
            }
            for item in orig_success
        ]
    }
    
    # Test Improved Scraper
    print("\n" + "="*90)
    print("🚀 Testing IMPROVED Scraper (30s timeout, DNS validation, retries, PDF detection)...")
    print("-" * 90)
    
    progress_improved = ProgressTracker()
    state_improved = {
        'validated_urls': test_urls,
        'scraped_content': [],
        'pdf_urls': [],
        'error': None
    }
    
    improved = ImprovedScrapeNode(timeout_ms=30000, max_retries=3)
    result_improved = await improved.execute(state_improved, progress_improved)
    
    improved_success = result_improved['scraped_content']
    improved_pdf = result_improved.get('pdf_urls', [])
    improved_failed = len(test_urls) - len(improved_success) - len(improved_pdf)
    
    print(f"\n✓ Improved Scraper Results:")
    print(f"  Successfully Scraped: {len(improved_success)}")
    print(f"  PDF URLs Detected: {len(improved_pdf)}")
    print(f"  Failed After Retries: {improved_failed}")
    print(f"  Success Rate (including PDFs): {((len(improved_success) + len(improved_pdf))/len(test_urls)*100):.1f}%")
    print(f"  Net Improvement: +{len(improved_success) - len(orig_success)} URLs")
    
    results["improved"] = {
        "successfully_scraped": len(improved_success),
        "pdf_detected": len(improved_pdf),
        "failed": improved_failed,
        "success_rate_including_pdf": f"{((len(improved_success) + len(improved_pdf))/len(test_urls)*100):.1f}%",
        "success_rate_html_only": f"{(len(improved_success)/len(test_urls)*100):.1f}%",
        "improvement_over_original": len(improved_success) - len(orig_success),
        "scraper_stats": {
            "dns_skipped": improved.stats['dns_skipped'],
            "pdf_detected": improved.stats['pdf_skipped'],
            "successful": improved.stats['successful'],
            "failed_after_retries": improved.stats['failed_after_retries'],
            "total_retries_used": improved.stats['retries_used']
        },
        "urls": [
            {
                "url": item['url'],
                "html_size": len(item.get('html', '')),
                "text_length": len(item.get('text', '')),
                "attempts": item.get('attempts', 1)
            }
            for item in improved_success
        ]
    }
    
    # Comparison Analysis
    print("\n" + "="*90)
    print("COMPARISON ANALYSIS")
    print("="*90)
    
    comparison = {
        "metric": "Original vs Improved",
        "rows": [
            {
                "metric": "HTML Pages Scraped",
                "original": len(orig_success),
                "improved": len(improved_success),
                "improvement": f"+{len(improved_success) - len(orig_success)}"
            },
            {
                "metric": "PDF URLs Detected",
                "original": "N/A",
                "improved": len(improved_pdf),
                "improvement": f"New feature"
            },
            {
                "metric": "Failed URLs",
                "original": orig_failed,
                "improved": improved_failed,
                "improvement": f"-{orig_failed - improved_failed}"
            },
            {
                "metric": "Success Rate (HTML)",
                "original": f"{(len(orig_success)/len(test_urls)*100):.1f}%",
                "improved": f"{(len(improved_success)/len(test_urls)*100):.1f}%",
                "improvement": f"+{(len(improved_success) - len(orig_success))/len(test_urls)*100:.1f}%"
            },
            {
                "metric": "With PDF Detection",
                "original": f"{(len(orig_success)/len(test_urls)*100):.1f}%",
                "improved": f"{((len(improved_success) + len(improved_pdf))/len(test_urls)*100):.1f}%",
                "improvement": f"+{((len(improved_success) + len(improved_pdf)) - len(orig_success))/len(test_urls)*100:.1f}%"
            },
            {
                "metric": "Timeout Duration",
                "original": "15 seconds",
                "improved": "30 seconds",
                "improvement": "2x longer"
            },
            {
                "metric": "Retry Logic",
                "original": "None",
                "improved": "3 attempts",
                "improvement": "New feature"
            },
            {
                "metric": "DNS Validation",
                "original": "None",
                "improved": "Yes",
                "improvement": "Eliminates dead links"
            }
        ]
    }
    
    results["comparison"] = comparison
    
    # Print comparison table
    print("\n")
    print(f"{'Metric':<30} {'Original':<25} {'Improved':<25} {'Change':<15}")
    print("-" * 95)
    for row in comparison["rows"]:
        metric = row["metric"]
        orig = str(row["original"])
        impr = str(row["improved"])
        change = row["improvement"]
        print(f"{metric:<30} {orig:<25} {impr:<25} {change:<15}")
    
    # Identify differences
    print("\n" + "="*90)
    print("DETAILED URL ANALYSIS")
    print("="*90)
    
    orig_urls = {item['url'] for item in orig_success}
    improved_urls = {item['url'] for item in improved_success}
    
    newly_scraped = improved_urls - orig_urls
    still_failed = set(test_urls) - orig_urls - improved_urls - set(improved_pdf)
    
    if newly_scraped:
        print(f"\n✨ NEWLY SCRAPED BY IMPROVED SCRAPER ({len(newly_scraped)}):")
        for url in newly_scraped:
            print(f"   • {url}")
    
    if improved_pdf:
        print(f"\n📄 PDF URLs DETECTED & ROUTED ({len(improved_pdf)}):")
        for url in improved_pdf:
            print(f"   • {url}")
    
    if still_failed:
        print(f"\n❌ STILL FAILING ({len(still_failed)}):")
        for url in still_failed:
            print(f"   • {url}")
    
    # Save results
    report_file = Path(__file__).parent / "test" / "scraper_comparison_report.json"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed comparison saved to: {report_file}")
    
    return results


if __name__ == "__main__":
    results = asyncio.run(test_both_scrapers())
    print("\n✅ Scraper comparison complete")
