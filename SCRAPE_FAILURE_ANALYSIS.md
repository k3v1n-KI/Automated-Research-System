# Scrape Failure Analysis Report

**Date:** February 2, 2026  
**Test Dataset:** Hospitals in Toronto (Real URLs from Algorithm)  
**Total URLs Tested:** 12  
**Success Rate:** 25% (3/12 successful)

---

## Executive Summary

The scraper achieved only a **25% success rate** when tested against real hospital websites. Out of 12 URLs found through the search algorithm, 9 failed to scrape for various reasons. This indicates a **critical need for scraper improvements** before production use.

**Key Finding:** Even with validated URLs from the search node, the scraper fails on 75% of requests due to diverse failure modes that require different handling strategies.

---

## Failed URLs (9 Total)

### 1. **DNS Resolution Failures** (4 URLs - 44% of failures)
These URLs fail because their domain names cannot be resolved to IP addresses.

#### URLs:
- `https://www.smh.ca/`
- `https://www.michaelgarron.ca/`
- `https://www.shothamilton.ca/`
- `https://www.caredtoronto.ca/`

#### Error:
```
net::ERR_NAME_NOT_RESOLVED
```

#### Why It Failed:
- The domain names do not exist in DNS or are not currently registered
- Could indicate: outdated search results, deleted sites, or domain registration issues

#### Impact on Dataset:
🔴 **CRITICAL** - Dead links add nothing to the dataset but waste computation and create confusion during data extraction. These should be filtered out immediately.

#### Recommendation:
- Add DNS pre-validation before scraping
- Use `socket.getaddrinfo()` to check domain resolution
- Skip URLs that fail DNS resolution in the validate node

---

### 2. **Connection Failures** (1 URL - 11% of failures)
Server actively refused the connection.

#### URL:
- `https://www.healthyalberta.com/glossary/hospitals`

#### Error:
```
net::ERR_CONNECTION_RESET
```

#### Why It Failed:
- Server rejected the connection (TCP RST packet)
- Could indicate: server overload, rate limiting, IP blocking, or active DDoS protection

#### Impact on Dataset:
🔴 **HIGH** - The site exists but rejected our request. Might be temporary (rate limit) or permanent (IP blocked). Retrying could succeed, but we can't distinguish without retry logic.

#### Recommendation:
- Implement exponential backoff with retry logic
- Add delay between requests to avoid rate limiting
- Consider rotating user agents
- Respect `Retry-After` headers if present

---

### 3. **SSL/TLS Certificate Error** (1 URL - 11% of failures)
HTTPS certificate validation failed.

#### URL:
- `https://www.sjwh.ca/`

#### Error:
```
net::ERR_CERT_COMMON_NAME_INVALID
```

#### Why It Failed:
- SSL certificate's common name doesn't match the domain
- Could indicate: misconfigured SSL, expired certificate, or domain mismatch

#### Impact on Dataset:
🟡 **MEDIUM** - The content might be valuable but we're blocking it for security. Can be bypassed cautiously.

#### Recommendation:
- Add flag to ignore SSL verification (cautiously, for testing)
- Or improve error handling to skip only truly invalid certs
- Consider adding to whitelist for known safe misconfigurations

---

### 4. **Timeout** (1 URL - 11% of failures)
Server didn't respond within the timeout window.

#### URL:
- `https://www.hhs.ca/`

#### Error:
```
Timeout 15000ms exceeded
```

#### Why It Failed:
- Server is extremely slow or unresponsive
- Current timeout: 15 seconds (15,000ms)
- Could indicate: server overload, network issues, or very heavy page

#### Impact on Dataset:
🟡 **MEDIUM** - Site exists and may have valuable data, but is too slow. Could be temporary.

#### Recommendation:
- Increase timeout to 30 seconds for initial attempt
- Add retry with longer timeout (30-60s)
- Consider adaptive timeout based on domain speed history
- Skip URL if timeout occurs twice

---

### 5. **PDF Downloads** (2 URLs - 22% of failures)
URLs point to PDF files that Playwright cannot process (triggers browser download).

#### URLs:
- `https://toronto.jamsports.com/pdfs/hospital_list.pdf`
- `https://socialwork.utoronto.ca/wp-content/uploads/2015/01/Public-Hospitals-List.pdf`

#### Error:
```
Download is starting
```

#### Why It Failed:
- Playwright expects HTML pages but received a download prompt
- Browser doesn't load PDF content, just triggers download
- PDFs need separate processing pipeline

#### Impact on Dataset:
🟡 **MEDIUM** - PDFs may contain valuable structured data (lists, tables), but require different parsing. Currently lost entirely.

#### Recommendation:
- Detect PDF URLs before attempting HTML scrape
- Route PDFs to separate pipeline using pdf2image or PyPDF2
- Extract text from PDFs using specialized library
- Add to URL validation to flag non-HTML content types

---

## Successful Scrapes (3 Total)

### URLs That Worked:
1. **https://www.torontohealth.ca/**
   - HTML: 27,978 bytes
   - Text: 476 characters
   
2. **https://www.sinaihealth.ca/**
   - HTML: 143,616 bytes
   - Text: 7,129 characters
   
3. **https://www.stmichaelshospital.com/**
   - HTML: 160,424 bytes
   - Text: 1,997 characters

### Content Quality:
- **Average page size:** 110.7 KB HTML
- **Average text content:** 3,200 characters
- **All pages:** Contain meaningful content (no empty pages)

---

## Failure Categories Summary

| Category | Count | % | Severity | Action |
|----------|-------|---|----------|--------|
| DNS Resolution Error | 4 | 44% | 🔴 Critical | Filter in validate node |
| PDF Download | 2 | 22% | 🟡 Medium | Route to PDF pipeline |
| Timeout | 1 | 11% | 🟡 Medium | Increase timeout, retry |
| Connection Reset | 1 | 11% | 🔴 High | Implement backoff retry |
| SSL Cert Error | 1 | 11% | 🟡 Medium | Handle gracefully |

---

## Impact Assessment

### Data Quality Impact:
- **75% failure rate** means only 1 in 4 URLs contribute to the dataset
- **Dead links (DNS failures)** waste computation on URLs that should never have been validated
- **PDF content** is completely lost (may contain valuable hospital lists/contacts)
- **Timeouts** are inconsistent (might work on retry)

### Computational Cost:
- Testing 12 URLs took ~3 minutes
- 9 failed attempts = 2.25 minutes wasted on failures
- **~75% of scraping time is wasted on non-productive failures**

---

## Recommendations (Priority Order)

### 🚨 CRITICAL - Do First:

1. **Add DNS pre-validation in Validate Node**
   ```python
   import socket
   try:
       socket.getaddrinfo(domain, 443)
   except socket.gaierror:
       skip_url()  # DNS won't resolve
   ```
   - Eliminates 44% of current failures immediately
   - Cost: ~10ms per URL

2. **Detect and route PDFs**
   ```python
   if url.lower().endswith('.pdf'):
       route_to_pdf_pipeline()
   ```
   - Eliminates 22% of current failures
   - Enables recovery of PDF content

### 📈 HIGH - Do Next:

3. **Implement retry logic with exponential backoff**
   - First attempt: immediate
   - Second attempt: 2 second delay
   - Third attempt: 5 second delay
   - Could recover ~10-15% of timeout/connection failures

4. **Increase timeout to 30 seconds**
   - Current: 15 seconds
   - Recommended: 30 seconds for first attempt
   - Would likely recover the 1 timeout failure

5. **Add HTTP status code checking**
   - Detect 404s, 403s, 500s before full page parse
   - Skip invalid responses early

### 🔧 MEDIUM - Do Later:

6. **SSL Certificate handling**
   - Option: Skip invalid certs (loses data)
   - Option: Bypass with warning (security risk)
   - Option: Whitelist known safe misconfigurations (best)

7. **Rate limiting and respectful scraping**
   - Add delay between requests (1-2 seconds)
   - Rotate user agents
   - Respect robots.txt
   - Implement caching

---

## Expected Improvements

| Current | With Recommendations | Improvement |
|---------|---------------------|------------|
| **Success Rate** | 25% | 75-80% | +50-55% |
| **Wasted Computation** | 75% | 20-25% | -50-55% |
| **Data Recovery** | 3/12 | 9-10/12 | +6-7 URLs |

### Implementation Impact:
- **Time to implement:** ~4-6 hours
- **Performance gain:** 3x more productive scraping
- **Data quality:** 3-4x more dataset coverage

---

## Conclusion

The current scraper has **fundamental limitations** that require addressing:

1. ✗ No DNS validation → wastes time on dead links
2. ✗ No PDF handling → loses valuable content
3. ✗ No retry logic → fails on transient errors
4. ✗ No adaptive timeouts → arbitrary cutoffs
5. ✗ No content-type filtering → tries to scrape everything

**Verdict:** ✗ **NOT PRODUCTION READY**

The 75% failure rate indicates the scraper needs improvements before wider deployment. The recommendations above are focused and achievable, with potential to increase success rate to 75-80%.

---

## Test Methodology

**Algorithm Used:** Full Automated Research System  
**Search Query:** "hospitals in Toronto"  
**Validation Method:** LangGraph validation node  
**Scraper:** Playwright browser automation with networkidle wait  
**Timeout:** 15 seconds per URL  
**Environment:** Python 3.11 + Playwright on Linux

