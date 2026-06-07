"""
Admin Clients Module

Enterprise-grade OAuth/API client management for UBP Enterprise Hybrid Edition.

Features:
- Secure secret generation (cryptographically secure random)
- Secret hashing with bcrypt
- O(1) client_name lookup
- OAuth2, API Key, and Service Account support
- Multi-tenancy support
- Event bus integration
- Secret rotation
- Soft revocation (is_active flag)
- Audit logging

Architecture:
- __init__.py: Entry point (this file)
- adapter.py: UBP framework bridge (AdminClientsAdapter)
- providers.py: Pure technical logic (ZERO UBP dependencies)
- config.json: Module configuration
- manifest.json: UBP standard manifest
"""

from pathlib import Path
from .adapter import AdminClientsAdapter

# Module version
__version__ = "1.0.0"

# Public API
__all__ = ["AdminClientsAdapter", "create_module"]


def create_module(module_path: Path, **kwargs) -> AdminClientsAdapter:
    """
    Factory function to create admin_clients module instance.

    Args:
        module_path: Path to module directory
        **kwargs: Additional arguments (app, config, etc.)

    Returns:
        Initialized AdminClientsAdapter instance
    """
    return AdminClientsAdapter(module_path, **kwargs)
