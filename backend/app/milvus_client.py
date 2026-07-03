"""Compatibility layer - imports moved to app/repositories/milvus_repo.py."""
from app.repositories.milvus_repo import milvus_repo, MilvusRepository

# Legacy aliases for backwards compatibility
milvus_client = milvus_repo
MilvusClient = MilvusRepository

__all__ = ["milvus_repo", "MilvusRepository", "milvus_client", "MilvusClient"]