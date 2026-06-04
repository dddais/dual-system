"""Executor adapters for standalone Dual-System."""

from dualsystem.executor.base import ExecutorClient
from dualsystem.executor.http_executor import HTTPExecutorClient

__all__ = ["ExecutorClient", "HTTPExecutorClient"]
