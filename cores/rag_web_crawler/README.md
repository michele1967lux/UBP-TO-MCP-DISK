# RAG Web Crawler Module v1.0

Web crawler for URL and sitemap ingestion into RAG knowledge bases in UBP Enterprise Hybrid.

## Architecture Overview

This module implements the **3-File Pattern Architecture**:

```
rag_web_crawler/
├── __init__.py          # Entry point with create_module() factory
├── adapter.py           # WebCrawlerAdapter (UBP integration layer)
├── providers.py         # WebCrawlerProvider (business logic)
├── config.json          # Configuration
├── manifest.json        # Module metadata and operations
└── README.md            # This file
```

## Features

### Crawling Modes
- **Single URL**: Crawl and optionally ingest a single page
- **Sitemap**: Parse sitemap.xml and crawl all URLs in background
- **Link Extraction**: Optionally extract links from crawled pages

### Content Processing
- **HTML Parsing**: Extract clean text from HTML using BeautifulSoup
- **Metadata Extraction**: Title, description, keywords, author
- **Auto-Ingestion**: Direct integration with rag_qdrant for indexing

### Production Features
- **Rate Limiting**: Configurable delay between requests
- **Background Tasks**: Sitemap crawling runs asynchronously
- **Job Tracking**: Monitor crawl progress and status
- **User Isolation**: Users see own jobs, admins see all

## Operations

| Operation | Description | Auth Required |
|-----------|-------------|---------------|
| `crawl_url` | Crawl a single URL | Yes |
| `crawl_sitemap` | Start sitemap crawl job | Yes |
| `get_crawl_status` | Check job progress | Yes |
| `list_crawl_jobs` | List crawl jobs | Yes |
| `cancel_crawl_job` | Cancel running job | Yes |
| `health_check` | Check module health | No |

## Configuration

### config.json

```json
{
  "enabled": true,
  "rate_limit_seconds": 1.0,
  "max_pages_per_sitemap": 500,
  "request_timeout_seconds": 30,
  "max_content_length": 1000000,
  "user_agent": "UBP-WebCrawler/1.0",
  "allowed_domains": [],
  "blocked_domains": []
}
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | true | Enable/disable module |
| `rate_limit_seconds` | float | 1.0 | Delay between requests |
| `max_pages_per_sitemap` | integer | 500 | Max pages per sitemap job |
| `request_timeout_seconds` | integer | 30 | HTTP request timeout |
| `max_content_length` | integer | 1000000 | Max content size (1MB) |
| `user_agent` | string | "UBP-WebCrawler/1.0" | HTTP User-Agent |
| `allowed_domains` | array | [] | Whitelist (empty = all) |
| `blocked_domains` | array | [] | Blacklist domains |

## API Usage Examples

### Crawl Single URL
```bash
curl -X POST "http://localhost:8000/api/modules/rag_web_crawler/crawl_url" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article",
    "target_collection": "web_docs",
    "auto_ingest": true,
    "extract_links": false
  }'
```

### Crawl Sitemap
```bash
curl -X POST "http://localhost:8000/api/modules/rag_web_crawler/crawl_sitemap" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "sitemap_url": "https://example.com/sitemap.xml",
    "target_collection": "web_docs",
    "max_pages": 100,
    "auto_ingest": true
  }'
```

### Check Job Status
```bash
curl -X POST "http://localhost:8000/api/modules/rag_web_crawler/get_crawl_status" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"job_id": "uuid-here"}'
```

### List Jobs
```bash
curl -X POST "http://localhost:8000/api/modules/rag_web_crawler/list_crawl_jobs" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"limit": 10, "status": "running"}'
```

## Response Examples

### Single URL Crawl
```json
{
  "job_id": "uuid-123",
  "status": "completed",
  "url": "https://example.com/article",
  "title": "Article Title",
  "content_length": 5420,
  "content_preview": "This article discusses...",
  "ingested": true,
  "ingest_result": {
    "doc_id": "doc-uuid",
    "collection": "web_docs"
  },
  "request_id": "req-uuid"
}
```

### Sitemap Job Status
```json
{
  "job_id": "uuid-456",
  "status": "running",
  "url": "https://example.com/sitemap.xml",
  "job_type": "sitemap",
  "total_pages": 50,
  "processed_pages": 23,
  "failed_pages": 2,
  "progress_percent": 46.0,
  "created_at": "2025-01-01T10:00:00Z",
  "started_at": "2025-01-01T10:00:01Z",
  "completed_at": null,
  "error_message": null,
  "request_id": "req-uuid"
}
```

## Job States

| Status | Description |
|--------|-------------|
| `pending` | Job created, waiting to start |
| `running` | Currently crawling |
| `completed` | Successfully finished |
| `failed` | Error occurred |
| `cancelled` | Manually cancelled |

## Events Published

- `crawler.url_crawled` - Single URL crawled
- `crawler.progress` - Sitemap progress update
- `crawler.job_completed` - Job finished successfully
- `crawler.job_failed` - Job failed with error

## UI Integration

The web crawler integrates with the RAG Admin Documents tab:

```javascript
// Start crawl from UI
async function crawlUrl(url, collection) {
    const response = await fetch('/api/modules/rag_web_crawler/crawl_url', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            url: url,
            target_collection: collection,
            auto_ingest: true
        })
    });
    return response.json();
}
```

## Security Considerations

- **Domain Filtering**: Use `allowed_domains` to restrict crawling
- **Rate Limiting**: Respect robots.txt and rate limits
- **Content Size**: `max_content_length` prevents memory issues
- **User Isolation**: Jobs are scoped to authenticated users

## Dependencies

- **Required**: rag_qdrant (for document ingestion)
- **Required**: aiohttp (for HTTP requests)
- **Required**: beautifulsoup4 (for HTML parsing)
- **Required**: lxml (for XML/HTML parsing)

## Performance Considerations

- **Rate Limiting**: Adjust based on target site's tolerance
- **Memory**: Large sitemaps spawn many background tasks
- **Network**: Consider proxy for high-volume crawling
- **Storage**: Monitor Qdrant storage for large crawls

## Testing

```bash
# Run module tests
cd ubp_enterprise_hybrid
pytest tests/integration/test_rag_v150_modules.py -k "web_crawler" -v
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-31 | Initial release with ROADMAP v1.5.0 |

---

**Module Type:** ingestion  
**Architecture:** 3-file-pattern  
**Production Ready:** Yes
