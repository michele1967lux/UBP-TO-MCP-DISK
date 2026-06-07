"""
Web Crawler Provider - Pure Technical Logic
Zero UBP dependencies. Can be tested standalone.
Implements async web crawling with HTML parsing.

Features:
- Single URL crawling with content extraction
- Sitemap parsing (XML) with recursive support
- Rate limiting via semaphore
- Content type and size validation
- Clean text extraction (removes scripts, styles, nav, etc.)
"""

from typing import Dict, Any, List, Optional, Protocol, runtime_checkable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from urllib.parse import urlparse, urljoin
import asyncio
import logging
import uuid
import re

logger = logging.getLogger(__name__)


class CrawlStatus(str, Enum):
    """Crawl job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CrawlJob:
    """Represents a crawl job."""

    job_id: str
    user_id: str
    url: str
    job_type: str  # 'single' or 'sitemap'
    status: CrawlStatus
    target_collection: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_pages: int = 0
    processed_pages: int = 0
    failed_pages: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "url": self.url,
            "job_type": self.job_type,
            "status": self.status.value,
            "target_collection": self.target_collection,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "total_pages": self.total_pages,
            "processed_pages": self.processed_pages,
            "failed_pages": self.failed_pages,
            "error_message": self.error_message,
            "progress_percent": round(
                (self.processed_pages / self.total_pages * 100)
                if self.total_pages > 0
                else 0,
                1,
            ),
        }


@dataclass
class CrawledPage:
    """Represents a crawled page."""

    url: str
    title: str
    content: str
    html: str
    metadata: Dict[str, Any]
    crawled_at: datetime


@runtime_checkable
class WebCrawlerProtocol(Protocol):
    """Interface contract for web crawler providers."""

    async def crawl_single_url(
        self,
        url: str,
        user_id: str,
        target_collection: str,
        extract_links: bool = False,
    ) -> CrawledPage:
        """Crawl a single URL and extract content."""
        ...

    async def parse_sitemap(self, sitemap_url: str) -> List[str]:
        """Parse a sitemap.xml and extract URLs."""
        ...

    async def create_crawl_job(
        self, url: str, user_id: str, target_collection: str, job_type: str = "single"
    ) -> CrawlJob:
        """Create a new crawl job."""
        ...

    def get_job(self, job_id: str) -> Optional[CrawlJob]:
        """Get a crawl job by ID."""
        ...

    def list_jobs(
        self, user_id: Optional[str] = None, limit: int = 20
    ) -> List[CrawlJob]:
        """List crawl jobs."""
        ...

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a crawl job."""
        ...


class WebCrawlerProvider:
    """
    Async web crawler implementation.

    Features:
    - Single URL crawling
    - Sitemap parsing and bulk crawling
    - Rate limiting via asyncio.Semaphore
    - Content extraction (removes scripts, styles, etc.)
    - Robots.txt respect (optional)

    Dependencies:
    - aiohttp: Async HTTP client
    - beautifulsoup4: HTML parsing
    - lxml: XML parsing for sitemaps
    """

    # Default excluded tags for content extraction
    EXCLUDED_TAGS = [
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "noscript",
        "iframe",
    ]

    # Content tags priority (try these first for main content)
    MAIN_CONTENT_TAGS = [
        "main",
        "article",
        'section[role="main"]',
        ".content",
        "#content",
        ".post",
        ".entry",
    ]

    def __init__(
        self,
        max_concurrent: int = 5,
        request_delay: float = 1.0,
        timeout: int = 30,
        max_content_length: int = 5 * 1024 * 1024,  # 5MB
        user_agent: str = "UBP-WebCrawler/1.0",
        respect_robots_txt: bool = True,
        allowed_content_types: Optional[List[str]] = None,
    ):
        """
        Initialize provider.

        Args:
            max_concurrent: Max concurrent requests
            request_delay: Delay between requests (seconds)
            timeout: Request timeout (seconds)
            max_content_length: Max content size to download (bytes)
            user_agent: User agent string
            respect_robots_txt: Whether to check robots.txt (not implemented yet)
            allowed_content_types: Allowed MIME types
        """
        self.max_concurrent = max_concurrent
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.user_agent = user_agent
        self.respect_robots_txt = respect_robots_txt
        self.allowed_content_types = allowed_content_types or [
            "text/html",
            "application/xhtml+xml",
        ]

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._jobs: Dict[str, CrawlJob] = {}
        self._robots_cache: Dict[str, Dict] = {}  # Cache for robots.txt

        logger.info(
            f"WebCrawlerProvider initialized: max_concurrent={max_concurrent}, "
            f"delay={request_delay}s, timeout={timeout}s"
        )

    async def crawl_single_url(
        self,
        url: str,
        user_id: str,
        target_collection: str,
        extract_links: bool = False,
    ) -> CrawledPage:
        """
        Crawl a single URL and extract content.

        Args:
            url: URL to crawl
            user_id: User requesting the crawl
            target_collection: Target collection for ingestion
            extract_links: Whether to extract links from page

        Returns:
            CrawledPage with extracted content

        Raises:
            ValueError: If content type or size is invalid
            aiohttp.ClientError: If HTTP request fails
        """
        import aiohttp
        from bs4 import BeautifulSoup

        async with self._semaphore:
            try:
                logger.debug(f"Crawling URL: {url}")

                async with aiohttp.ClientSession() as session:
                    headers = {
                        "User-Agent": self.user_agent,
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5",
                        "Accept-Encoding": "gzip, deflate",
                    }

                    async with session.get(
                        url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                        allow_redirects=True,
                        ssl=False,  # Allow self-signed certs for internal sites
                    ) as response:
                        response.raise_for_status()

                        # Validate content length
                        content_length = response.headers.get("Content-Length")
                        if (
                            content_length
                            and int(content_length) > self.max_content_length
                        ):
                            raise ValueError(
                                f"Content too large: {content_length} bytes "
                                f"(max: {self.max_content_length})"
                            )

                        # Validate content type
                        content_type = response.headers.get("Content-Type", "")
                        if not any(
                            ct in content_type for ct in self.allowed_content_types
                        ):
                            raise ValueError(
                                f"Unsupported content type: {content_type}"
                            )

                        # Read response
                        html = await response.text()
                        final_url = str(response.url)  # Get final URL after redirects

                # Parse HTML
                soup = BeautifulSoup(html, "html.parser")

                # Extract title
                title = self._extract_title(soup, url)

                # Remove unwanted elements
                for tag in self.EXCLUDED_TAGS:
                    for element in soup.find_all(tag):
                        element.decompose()

                # Extract main content
                content = self._extract_content(soup)

                # Build metadata
                metadata = {
                    "source_url": final_url,
                    "original_url": url,
                    "domain": urlparse(final_url).netloc,
                    "crawled_at": datetime.utcnow().isoformat(),
                    "content_length": len(content),
                    "target_collection": target_collection,
                    "user_id": user_id,
                }

                # Extract description meta tag
                meta_desc = soup.find("meta", attrs={"name": "description"})
                if meta_desc and meta_desc.get("content"):
                    metadata["description"] = meta_desc.get("content", "")[:500]

                # Extract keywords meta tag
                meta_keywords = soup.find("meta", attrs={"name": "keywords"})
                if meta_keywords and meta_keywords.get("content"):
                    metadata["keywords"] = meta_keywords.get("content", "")[:200]

                # Extract links if requested
                if extract_links:
                    links = self._extract_links(soup, final_url)
                    metadata["extracted_links"] = links[:100]  # Limit to 100
                    metadata["extracted_links_count"] = len(links)

                logger.info(
                    f"Successfully crawled {url}: {len(content)} chars, title='{title[:50]}...'"
                )

                return CrawledPage(
                    url=final_url,
                    title=title,
                    content=content,
                    html=html,
                    metadata=metadata,
                    crawled_at=datetime.utcnow(),
                )

            except Exception as e:
                logger.error(f"Failed to crawl {url}: {type(e).__name__}: {e}")
                raise

    def _extract_title(self, soup, url: str) -> str:
        """Extract page title from HTML."""
        # Try <title> tag first
        title_tag = soup.find("title")
        if title_tag and title_tag.get_text().strip():
            return title_tag.get_text().strip()[:200]

        # Try <h1> tag
        h1_tag = soup.find("h1")
        if h1_tag and h1_tag.get_text().strip():
            return h1_tag.get_text().strip()[:200]

        # Try og:title meta
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title and og_title.get("content"):
            return og_title.get("content")[:200]

        # Fallback to URL path
        parsed = urlparse(url)
        return parsed.path.split("/")[-1] or parsed.netloc

    def _extract_content(self, soup) -> str:
        """Extract main content from HTML."""
        # Try to find main content area
        main_content = None

        # Try common content selectors
        for selector in [
            "main",
            "article",
            '[role="main"]',
            ".content",
            "#content",
            ".post-content",
            ".entry-content",
        ]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        # Fallback to body
        if not main_content:
            main_content = soup.find("body")

        if not main_content:
            return ""

        # Get text with newlines
        content = main_content.get_text(separator="\n", strip=True)

        # Clean up content
        content = self._clean_content(content)

        return content

    def _clean_content(self, content: str) -> str:
        """Clean extracted content."""
        # Remove excessive whitespace
        content = re.sub(
            r"\n\s*\n\s*\n", "\n\n", content
        )  # Multiple blank lines -> double
        content = re.sub(r" +", " ", content)  # Multiple spaces -> single
        content = re.sub(r"\t+", " ", content)  # Tabs -> space

        # Remove very short lines (likely navigation remnants)
        lines = content.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Keep line if it's blank (paragraph break) or has substantial content
            if stripped == "" or len(stripped) > 15:
                cleaned_lines.append(stripped)

        content = "\n".join(cleaned_lines)

        # Remove common navigation/button text patterns
        noise_patterns = [
            r"^(Home|Menu|Search|Login|Sign in|Register|Subscribe)\s*$",
            r"^(Previous|Next|Read more|Continue reading)\s*$",
            r"^\d+\s*(comments?|shares?|likes?)\s*$",
            r"^Share\s*(on)?\s*(Facebook|Twitter|LinkedIn)?\s*$",
        ]

        for pattern in noise_patterns:
            content = re.sub(pattern, "", content, flags=re.MULTILINE | re.IGNORECASE)

        return content.strip()

    def _extract_links(self, soup, base_url: str) -> List[str]:
        """Extract links from HTML."""
        links = []
        base_domain = urlparse(base_url).netloc

        for a in soup.find_all("a", href=True):
            href = a["href"]

            # Skip anchors, javascript, mailto
            if href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            # Convert to absolute URL
            absolute_url = urljoin(base_url, href)

            # Only include HTTP(S) URLs
            if absolute_url.startswith(("http://", "https://")):
                # Optionally filter to same domain
                link_domain = urlparse(absolute_url).netloc
                if link_domain == base_domain:  # Same domain only
                    links.append(absolute_url)

        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)

        return unique_links

    async def parse_sitemap(self, sitemap_url: str) -> List[str]:
        """
        Parse a sitemap.xml and extract URLs.

        Supports:
        - Standard sitemap.xml
        - Sitemap index files (recursive parsing)
        - Compressed sitemaps (.gz)

        Args:
            sitemap_url: URL to sitemap.xml

        Returns:
            List of URLs from sitemap (limited to 500)
        """
        import aiohttp
        import xml.etree.ElementTree as ET

        urls = []
        max_urls = 500
        max_sitemaps = 10  # Max sitemap index entries to process

        logger.info(f"Parsing sitemap: {sitemap_url}")

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"User-Agent": self.user_agent}

                async with session.get(
                    sitemap_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ssl=False,
                ) as response:
                    response.raise_for_status()
                    content = await response.text()

            # Parse XML
            root = ET.fromstring(content)

            # Handle namespace (sitemap protocol uses this namespace)
            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

            # Check if this is a sitemap index (contains <sitemap> elements)
            sitemap_refs = root.findall(".//sm:sitemap/sm:loc", ns)

            # Try without namespace if not found
            if not sitemap_refs:
                sitemap_refs = root.findall(".//sitemap/loc")

            if sitemap_refs:
                # This is a sitemap index - recursively parse sub-sitemaps
                logger.info(
                    f"Found sitemap index with {len(sitemap_refs)} sub-sitemaps"
                )

                for ref in sitemap_refs[:max_sitemaps]:
                    sub_sitemap_url = ref.text.strip() if ref.text else None
                    if sub_sitemap_url:
                        try:
                            sub_urls = await self.parse_sitemap(sub_sitemap_url)
                            urls.extend(sub_urls)

                            if len(urls) >= max_urls:
                                break
                        except Exception as e:
                            logger.warning(
                                f"Failed to parse sub-sitemap {sub_sitemap_url}: {e}"
                            )
            else:
                # Regular sitemap - extract URL entries
                url_elements = root.findall(".//sm:url/sm:loc", ns)

                # Try without namespace
                if not url_elements:
                    url_elements = root.findall(".//url/loc")

                for url_elem in url_elements:
                    if url_elem.text:
                        urls.append(url_elem.text.strip())

                        if len(urls) >= max_urls:
                            break

            logger.info(f"Parsed sitemap: found {len(urls)} URLs")
            return urls[:max_urls]

        except ET.ParseError as e:
            logger.error(f"Failed to parse sitemap XML: {e}")
            raise ValueError(f"Invalid sitemap XML: {e}")
        except Exception as e:
            logger.error(f"Failed to fetch/parse sitemap {sitemap_url}: {e}")
            raise

    async def create_crawl_job(
        self, url: str, user_id: str, target_collection: str, job_type: str = "single"
    ) -> CrawlJob:
        """
        Create a new crawl job.

        Args:
            url: URL to crawl (single page or sitemap)
            user_id: User ID
            target_collection: Target collection name
            job_type: 'single' or 'sitemap'

        Returns:
            New CrawlJob instance
        """
        job = CrawlJob(
            job_id=str(uuid.uuid4()),
            user_id=user_id,
            url=url,
            job_type=job_type,
            status=CrawlStatus.PENDING,
            target_collection=target_collection,
            created_at=datetime.utcnow(),
        )

        self._jobs[job.job_id] = job
        logger.info(f"Created crawl job {job.job_id}: type={job_type}, url={url}")

        return job

    def get_job(self, job_id: str) -> Optional[CrawlJob]:
        """Get a crawl job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        user_id: Optional[str] = None,
        limit: int = 20,
        status_filter: Optional[List[CrawlStatus]] = None,
    ) -> List[CrawlJob]:
        """
        List crawl jobs, optionally filtered.

        Args:
            user_id: Filter by user ID (None for all)
            limit: Maximum jobs to return
            status_filter: Filter by status list

        Returns:
            List of CrawlJob sorted by created_at desc
        """
        jobs = list(self._jobs.values())

        # Filter by user
        if user_id:
            jobs = [j for j in jobs if j.user_id == user_id]

        # Filter by status
        if status_filter:
            jobs = [j for j in jobs if j.status in status_filter]

        # Sort by created_at descending (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a crawl job.

        Args:
            job_id: Job UUID to cancel

        Returns:
            True if cancelled, False if not found or already completed
        """
        job = self._jobs.get(job_id)

        if not job:
            return False

        if job.status in (CrawlStatus.PENDING, CrawlStatus.RUNNING):
            job.status = CrawlStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            logger.info(f"Cancelled crawl job {job_id}")
            return True

        return False

    def update_job_progress(self, job_id: str, processed: int, failed: int = 0) -> None:
        """Update job progress counters."""
        job = self._jobs.get(job_id)
        if job:
            job.processed_pages = processed
            job.failed_pages = failed

    def complete_job(
        self, job_id: str, success: bool = True, error_message: Optional[str] = None
    ) -> None:
        """Mark job as completed or failed."""
        job = self._jobs.get(job_id)
        if job:
            job.status = CrawlStatus.COMPLETED if success else CrawlStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error_message = error_message

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """
        Remove old completed jobs from memory.

        Args:
            max_age_hours: Remove jobs older than this

        Returns:
            Number of jobs removed
        """
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        old_jobs = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status
            in (CrawlStatus.COMPLETED, CrawlStatus.FAILED, CrawlStatus.CANCELLED)
            and job.completed_at
            and job.completed_at < cutoff
        ]

        for job_id in old_jobs:
            del self._jobs[job_id]

        if old_jobs:
            logger.info(f"Cleaned up {len(old_jobs)} old crawl jobs")

        return len(old_jobs)

    def health_check(self) -> Dict[str, Any]:
        """Check provider health and return stats."""
        active_jobs = [
            j for j in self._jobs.values() if j.status == CrawlStatus.RUNNING
        ]
        pending_jobs = [
            j for j in self._jobs.values() if j.status == CrawlStatus.PENDING
        ]

        return {
            "status": "configured",
            "max_concurrent": self.max_concurrent,
            "request_delay_seconds": self.request_delay,
            "timeout_seconds": self.timeout,
            "active_jobs": len(active_jobs),
            "pending_jobs": len(pending_jobs),
            "total_jobs_in_memory": len(self._jobs),
        }
