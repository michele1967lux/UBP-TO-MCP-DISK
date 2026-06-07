"""
RAG Web Crawler Module

Provides URL and sitemap crawling for automatic content ingestion into RAG collections.

Features:
- Single URL crawling with content extraction
- Sitemap.xml parsing with bulk crawling
- Background processing for large sitemaps
- Automatic ingestion to rag_qdrant collections
- Rate limiting and content validation

Usage:
    from ubp_enterprise_hybrid.modules.cores.rag_web_crawler import create_module, WebCrawlerAdapter

    module = create_module(module_path)
    await module.initialize()

    # Crawl single URL
    result = await module.crawl_url(
        url="https://example.com/page",
        target_collection="my_collection",
        ctx=ctx
    )

    # Crawl sitemap
    result = await module.crawl_sitemap(
        sitemap_url="https://example.com/sitemap.xml",
        target_collection="my_collection",
        max_pages=100,
        ctx=ctx
    )

Dependencies:
    - aiohttp: Async HTTP client
    - beautifulsoup4: HTML parsing
    - lxml: XML parsing for sitemaps
"""

from pathlib import Path
from .adapter import WebCrawlerAdapter

__version__ = "1.0.0"
__all__ = ["WebCrawlerAdapter", "create_module"]


def create_module(module_path: Path, **kwargs) -> WebCrawlerAdapter:
    """
    Factory function to create module instance.

    This is the standard entry point for the UBP module loader.

    Args:
        module_path: Path to the module directory
        **kwargs: Additional arguments passed to adapter

    Returns:
        WebCrawlerAdapter instance
    """
    return WebCrawlerAdapter(module_path, **kwargs)
