"""
UBP Framework Bridge for Web Crawler Module
Integrates WebCrawlerProvider with UBP module system.

Provides URL and sitemap crawling with automatic ingestion to RAG collections.

Operations:
- crawl_url: Crawl single URL and optionally ingest
- crawl_sitemap: Parse sitemap and crawl pages in background
- get_crawl_status: Get job status
- list_crawl_jobs: List user's crawl jobs
- cancel_crawl_job: Cancel running job
- health_check: Module health status
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import asyncio
import hashlib
import logging
import uuid

from ubp_enterprise_hybrid.modules.cores._shared import BaseHybridModule
from .providers import WebCrawlerProvider, CrawlStatus, CrawledPage

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        from ..._shared.operation_context import OperationContext

logger = logging.getLogger(__name__)


class WebCrawlerAdapter(BaseHybridModule):
    """
    UBP adapter for web crawling.

    Provides URL and sitemap crawling with automatic ingestion.
    Follows 3-file pattern: no business logic here, only UBP integration.

    Security:
    - User ID always from ctx, never from payload
    - Job ownership verified before operations
    - Admin can see all jobs
    """

    def __init__(self, module_path: Path, **kwargs):
        super().__init__(module_path, **kwargs)
        self.provider: Optional[WebCrawlerProvider] = None
        self._background_tasks: Dict[str, asyncio.Task] = {}
        self._total_pages_crawled = 0
        self._total_pages_ingested = 0

    # MCP-COMPAT: OperationContext helpers (ARCH-008)
    def _build_context_from_di(self) -> OperationContext:
        """Build OperationContext from DI — backward compatibility for REST path."""
        return OperationContext(
            client_id="default",
            user_id=None,
            session_id=None,
            source="rest",
        )

    def _normalize_ctx(self, ctx: Any) -> OperationContext:
        """Normalize any context format to OperationContext."""
        if ctx is None:
            return self._build_context_from_di()
        if isinstance(ctx, OperationContext):
            return ctx
        if hasattr(ctx, "user") and ctx.user:
            user_id = getattr(ctx.user, "user_id", None)
            roles = getattr(ctx.user, "roles", [])
            client_id = getattr(ctx.user, "client_id", "default")
            if not isinstance(roles, (list, tuple)):
                roles = []
            return OperationContext(
                client_id=str(client_id) if client_id else "default",
                user_id=str(user_id) if user_id else None,
                roles=list(roles),
                source="rest",
            )
        return self._build_context_from_di()

    async def initialize(self) -> None:
        """Initialize module and provider."""
        logger.info(f"Initializing {self.manifest.name}")

        # Create provider with config
        self.provider = WebCrawlerProvider(
            max_concurrent=self.config.get("max_concurrent_requests", 5),
            request_delay=self.config.get("request_delay_seconds", 1.0),
            timeout=self.config.get("request_timeout_seconds", 30),
            max_content_length=self.config.get(
                "max_content_length_bytes", 5 * 1024 * 1024
            ),
            user_agent=self.config.get("user_agent", "UBP-WebCrawler/1.0"),
            respect_robots_txt=self.config.get("respect_robots_txt", True),
            allowed_content_types=self.config.get("allowed_content_types", None),
        )

        # Resolve rag_qdrant for ingestion (same pattern as rag_orchestrator)
        self._qdrant = None
        if self.di_container:
            try:
                self._qdrant = await self.di_container.resolve("rag_qdrant")
                logger.info("✅ rag_qdrant resolved for web crawler ingestion")
            except Exception as e:
                logger.warning(f"rag_qdrant not available for ingestion: {e}")

        logger.info(f"✅ {self.manifest.name} initialized successfully")

    async def shutdown(self) -> None:
        """Shutdown module and cancel running tasks."""
        logger.info(f"Shutting down {self.manifest.name}")

        # Cancel all background tasks
        for task_id, task in list(self._background_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            logger.debug(f"Cancelled background task {task_id}")

        self._background_tasks.clear()
        self.provider = None

        logger.info(f"✅ {self.manifest.name} shutdown complete")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        provider_health = self.provider.health_check() if self.provider else None

        return {
            "module": self.manifest.name,
            "version": self.manifest.version,
            "status": "healthy" if self.provider else "unhealthy",
            "active_background_tasks": len(self._background_tasks),
            "total_pages_crawled": self._total_pages_crawled,
            "total_pages_ingested": self._total_pages_ingested,
            "provider": provider_health,
        }

    # === OPERATIONS ===

    async def crawl_url(
        self,
        url: str,
        target_collection: str,
        auto_ingest: bool = True,
        extract_links: bool = False,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Crawl a single URL and optionally ingest to collection.

        Args:
            url: URL to crawl (must start with http:// or https://)
            target_collection: Collection to ingest into
            auto_ingest: Whether to automatically ingest content (default True)
            extract_links: Whether to extract links from page
            ctx: Security context

        Returns:
            Dict with job_id, status, url, title, content_length, ingested flag
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        # Get user_id from context (SECURITY: never from payload)
        user_id = self._get_user_id_from_ctx(ctx)
        if not user_id:
            return {"error": "User not authenticated", "request_id": request_id}

        # Validate URL
        if not url or not isinstance(url, str):
            return {"error": "URL is required", "request_id": request_id}

        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return {
                "error": "Invalid URL. Must start with http:// or https://",
                "request_id": request_id,
            }

        # Validate collection
        if not target_collection or not isinstance(target_collection, str):
            return {"error": "target_collection is required", "request_id": request_id}

        try:
            # Create job for tracking
            job = await self.provider.create_crawl_job(
                url=url,
                user_id=user_id,
                target_collection=target_collection,
                job_type="single",
            )

            # Update job status
            job.status = CrawlStatus.RUNNING
            job.started_at = datetime.utcnow()
            job.total_pages = 1

            # Crawl the page
            page = await self.provider.crawl_single_url(
                url=url,
                user_id=user_id,
                target_collection=target_collection,
                extract_links=extract_links,
            )

            job.processed_pages = 1
            self._total_pages_crawled += 1

            # Auto-ingest if requested
            ingest_result = None
            if auto_ingest:
                ingest_result = await self._ingest_page(page, target_collection, ctx)
                if ingest_result and not ingest_result.get("error"):
                    self._total_pages_ingested += 1

            # Complete job
            job.status = CrawlStatus.COMPLETED
            job.completed_at = datetime.utcnow()

            # Publish event
            if self.publisher:
                await self.publisher.publish(
                    "crawler.url_crawled",
                    {
                        "job_id": job.job_id,
                        "url": url,
                        "title": page.title,
                        "ingested": ingest_result is not None,
                        "request_id": request_id,
                    },
                )

            return {
                "job_id": job.job_id,
                "status": "completed",
                "url": page.url,
                "title": page.title,
                "content_length": len(page.content),
                "content_preview": page.content[:500] + "..."
                if len(page.content) > 500
                else page.content,
                "ingested": ingest_result is not None
                and not ingest_result.get("error"),
                "ingest_result": ingest_result,
                "metadata": page.metadata,
                "request_id": request_id,
            }

        except ValueError as e:
            logger.warning(f"Validation error crawling {url}: {e}")
            return {"error": str(e), "url": url, "request_id": request_id}
        except Exception as e:
            logger.error(f"Failed to crawl {url}: {type(e).__name__}: {e}")
            return {
                "error": f"Crawl failed: {str(e)}",
                "url": url,
                "request_id": request_id,
            }

    async def crawl_sitemap(
        self,
        sitemap_url: str,
        target_collection: str,
        max_pages: int = 100,
        auto_ingest: bool = True,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Parse sitemap and crawl pages in background.

        Args:
            sitemap_url: URL to sitemap.xml
            target_collection: Collection to ingest into
            max_pages: Maximum pages to crawl (default 100, max 500)
            auto_ingest: Whether to automatically ingest content
            ctx: Security context

        Returns:
            Dict with job_id, status, total_urls (crawling starts in background)
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        user_id = self._get_user_id_from_ctx(ctx)
        if not user_id:
            return {"error": "User not authenticated", "request_id": request_id}

        # Validate URL
        if not sitemap_url or not isinstance(sitemap_url, str):
            return {"error": "sitemap_url is required", "request_id": request_id}

        sitemap_url = sitemap_url.strip()
        if not sitemap_url.startswith(("http://", "https://")):
            return {
                "error": "Invalid sitemap URL. Must start with http:// or https://",
                "request_id": request_id,
            }

        # Cap max_pages
        max_pages_limit = self.config.get("max_pages_per_sitemap", 500)
        max_pages = min(max(1, max_pages), max_pages_limit)

        try:
            # Parse sitemap first to get URLs
            logger.info(f"Parsing sitemap: {sitemap_url}")
            urls = await self.provider.parse_sitemap(sitemap_url)
            urls = urls[:max_pages]

            if not urls:
                return {
                    "error": "No URLs found in sitemap",
                    "sitemap_url": sitemap_url,
                    "request_id": request_id,
                }

            # Create job
            job = await self.provider.create_crawl_job(
                url=sitemap_url,
                user_id=user_id,
                target_collection=target_collection,
                job_type="sitemap",
            )
            job.total_pages = len(urls)

            # Start background crawl task
            task = asyncio.create_task(
                self._background_crawl(job, urls, target_collection, auto_ingest, ctx)
            )
            self._background_tasks[job.job_id] = task

            # Clean up task reference when done
            task.add_done_callback(
                lambda t: self._background_tasks.pop(job.job_id, None)
            )

            logger.info(f"Started sitemap crawl job {job.job_id}: {len(urls)} URLs")

            return {
                "job_id": job.job_id,
                "status": "started",
                "sitemap_url": sitemap_url,
                "total_urls": len(urls),
                "target_collection": target_collection,
                "auto_ingest": auto_ingest,
                "request_id": request_id,
            }

        except ValueError as e:
            logger.warning(f"Sitemap parse error: {e}")
            return {
                "error": str(e),
                "sitemap_url": sitemap_url,
                "request_id": request_id,
            }
        except Exception as e:
            logger.error(f"Sitemap crawl failed: {type(e).__name__}: {e}")
            return {
                "error": f"Sitemap crawl failed: {str(e)}",
                "request_id": request_id,
            }

    async def _background_crawl(
        self, job, urls: List[str], target_collection: str, auto_ingest: bool, ctx
    ):
        """
        Background task for bulk crawling from sitemap.

        Updates job progress and publishes events during crawl.
        """
        job.status = CrawlStatus.RUNNING
        job.started_at = datetime.utcnow()

        try:
            for i, url in enumerate(urls):
                # Check if job was cancelled
                if job.status == CrawlStatus.CANCELLED:
                    logger.info(f"Job {job.job_id} was cancelled, stopping crawl")
                    break

                try:
                    # Crawl the page
                    page = await self.provider.crawl_single_url(
                        url=url,
                        user_id=job.user_id,
                        target_collection=target_collection,
                    )

                    self._total_pages_crawled += 1

                    # Ingest if requested
                    if auto_ingest:
                        ingest_result = await self._ingest_page(
                            page, target_collection, ctx
                        )
                        if ingest_result and not ingest_result.get("error"):
                            self._total_pages_ingested += 1

                    job.processed_pages += 1

                    # Publish progress event every 5 pages or on completion
                    if self.publisher and (
                        job.processed_pages % 5 == 0
                        or job.processed_pages == job.total_pages
                    ):
                        await self.publisher.publish(
                            "crawler.progress",
                            {
                                "job_id": job.job_id,
                                "processed": job.processed_pages,
                                "total": job.total_pages,
                                "failed": job.failed_pages,
                                "progress_percent": job.to_dict()["progress_percent"],
                            },
                        )

                except Exception as e:
                    logger.warning(f"Failed to crawl {url} in job {job.job_id}: {e}")
                    job.failed_pages += 1

                # Rate limiting delay
                await asyncio.sleep(self.provider.request_delay)

            # Mark job completed
            job.status = CrawlStatus.COMPLETED
            job.completed_at = datetime.utcnow()

            logger.info(
                f"Completed sitemap crawl job {job.job_id}: "
                f"{job.processed_pages}/{job.total_pages} pages, {job.failed_pages} failed"
            )

            # Publish completion event
            if self.publisher:
                await self.publisher.publish(
                    "crawler.job_completed",
                    {
                        "job_id": job.job_id,
                        "processed": job.processed_pages,
                        "failed": job.failed_pages,
                        "total": job.total_pages,
                    },
                )

        except asyncio.CancelledError:
            job.status = CrawlStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            logger.info(f"Job {job.job_id} cancelled via task cancellation")
            raise

        except Exception as e:
            job.status = CrawlStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            logger.error(f"Background crawl job {job.job_id} failed: {e}")

            if self.publisher:
                await self.publisher.publish(
                    "crawler.job_failed", {"job_id": job.job_id, "error": str(e)}
                )

    async def _ingest_page(
        self, page: CrawledPage, target_collection: str, ctx
    ) -> Optional[Dict[str, Any]]:
        """
        Ingest crawled page to rag_qdrant collection.

        Args:
            page: CrawledPage to ingest
            target_collection: Collection name
            ctx: Security context

        Returns:
            Ingest result or None if qdrant unavailable
        """
        try:
            # Use rag_qdrant resolved at initialize()
            qdrant_module = self._qdrant
            if not qdrant_module:
                logger.warning("rag_qdrant module not available for ingestion")
                return {
                    "warning": "rag_qdrant module not available, content not ingested"
                }

            # Build metadata for document
            doc_metadata = {
                "source": "web_crawler",
                "source_type": "webpage",
                "source_url": page.url,
                "title": page.title,
                "domain": page.metadata.get("domain", ""),
                "crawled_at": page.crawled_at.isoformat(),
                "description": page.metadata.get("description", ""),
            }

            # Inter-module call — add_document_internal bypasses admin check
            # (crawler has already validated permissions in crawl_url)
            doc_id = hashlib.sha256(page.url.encode()).hexdigest()[:16]
            result = await qdrant_module.add_document_internal(
                doc_id=doc_id,
                text=page.content,
                metadata=doc_metadata,
                collection=target_collection,
            )

            logger.debug(f"Ingested page {page.url} to collection {target_collection}")
            return result

        except Exception as e:
            logger.error(f"Failed to ingest page {page.url}: {type(e).__name__}: {e}")
            return {"error": f"Ingest failed: {str(e)}"}

    async def get_crawl_status(
        self, job_id: str, request_id: Optional[str] = None, ctx=None, **kwargs
    ) -> Dict[str, Any]:
        """
        Get status of a crawl job.

        Args:
            job_id: UUID of the crawl job
            ctx: Security context

        Returns:
            Job details including status, progress, etc.
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        if not job_id:
            return {"error": "job_id is required", "request_id": request_id}

        user_id = self._get_user_id_from_ctx(ctx)
        job = self.provider.get_job(job_id)

        if not job:
            return {
                "error": "Job not found",
                "job_id": job_id,
                "request_id": request_id,
            }

        # Verify ownership (unless admin)
        if job.user_id != user_id and not self._is_admin(ctx):
            return {"error": "Access denied", "request_id": request_id}

        return {**job.to_dict(), "request_id": request_id}

    async def list_crawl_jobs(
        self,
        limit: int = 20,
        status: Optional[str] = None,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        List crawl jobs for current user (or all if admin).

        Args:
            limit: Maximum jobs to return (max 100)
            status: Filter by status (pending, running, completed, failed, cancelled)
            ctx: Security context

        Returns:
            List of jobs
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        user_id = self._get_user_id_from_ctx(ctx)

        # Admins see all jobs, users see only their own
        filter_user = None if self._is_admin(ctx) else user_id

        # Parse status filter
        status_filter = None
        if status:
            try:
                status_filter = [CrawlStatus(status)]
            except ValueError:
                pass  # Invalid status, ignore filter

        jobs = self.provider.list_jobs(
            user_id=filter_user,
            limit=min(max(1, limit), 100),
            status_filter=status_filter,
        )

        return {
            "jobs": [j.to_dict() for j in jobs],
            "count": len(jobs),
            "request_id": request_id,
        }

    async def cancel_crawl_job(
        self, job_id: str, request_id: Optional[str] = None, ctx=None, **kwargs
    ) -> Dict[str, Any]:
        """
        Cancel a running crawl job.

        Args:
            job_id: UUID of the crawl job to cancel
            ctx: Security context

        Returns:
            Cancellation result
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        if not job_id:
            return {"error": "job_id is required", "request_id": request_id}

        user_id = self._get_user_id_from_ctx(ctx)
        job = self.provider.get_job(job_id)

        if not job:
            return {
                "error": "Job not found",
                "job_id": job_id,
                "request_id": request_id,
            }

        # Verify ownership (unless admin)
        if job.user_id != user_id and not self._is_admin(ctx):
            return {"error": "Access denied", "request_id": request_id}

        # Cancel the job in provider
        success = self.provider.cancel_job(job_id)

        # Also cancel the background task if running
        if job_id in self._background_tasks:
            task = self._background_tasks[job_id]
            if not task.done():
                task.cancel()

        return {
            "job_id": job_id,
            "cancelled": success,
            "status": job.status.value,
            "message": "Job cancelled"
            if success
            else "Job could not be cancelled (may be already completed)",
            "request_id": request_id,
        }

    # === HELPER METHODS ===

    def _get_user_id_from_ctx(self, ctx) -> Optional[str]:
        """Extract user_id from security context."""
        if ctx and hasattr(ctx, "user") and ctx.user:
            return getattr(ctx.user, "user_id", None) or getattr(ctx.user, "id", None)
        return None

    def _is_admin(self, ctx) -> bool:
        """Check if user is admin."""
        if ctx and hasattr(ctx, "user") and ctx.user:
            return getattr(ctx.user, "is_admin", False)
        return False

    def _get_module(self, module_name: str):
        """Get adapter instance of another module via module_manager.

        module_manager.get_module() returns ModuleMetadata — we need .adapter_instance.
        Pattern consistent with dependencies.py and dynamic_router.py.
        """
        if hasattr(self, "module_manager") and self.module_manager:
            metadata = self.module_manager.get_module(module_name)
            if metadata and getattr(metadata, "adapter_instance", None):
                return metadata.adapter_instance
        return None
