"""
Admin Users Module

Enterprise-grade user management for UBP Enterprise Hybrid Edition.

Features:
- Secure password hashing with bcrypt
- O(1) username lookup
- Role-based access control
- Multi-tenancy support
- Event bus integration
- Audit logging

Architecture:
- __init__.py: Entry point (this file)
- adapter.py: UBP framework bridge (AdminUsersAdapter)
- config.json: Module configuration
- manifest.json: UBP standard manifest
"""

from pathlib import Path
from .adapter import AdminUsersAdapter

# Module version
__version__ = "1.0.0"

# Public API
__all__ = ["AdminUsersAdapter", "create_module"]


def create_module(module_path: Path, **kwargs) -> AdminUsersAdapter:
    """
    Factory function to create admin_users module instance.

    Args:
        module_path: Path to module directory
        **kwargs: Additional arguments (app, config, etc.)

    Returns:
        Initialized AdminUsersAdapter instance
    """
    return AdminUsersAdapter(module_path, **kwargs)
