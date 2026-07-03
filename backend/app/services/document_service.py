"""Service: Document upload and processing."""
import os
import uuid
from pathlib import Path

import structlog
from fastapi import HTTPException, UploadFile

from app.config import settings
from app.tasks import enqueue_parse

logger = structlog.get_logger()


class DocumentService:
    """文档上传与处理服务。"""

    def upload(
        self,
        file: UploadFile,
        kb_id: str = "",
        source_type: str = "local_file",
        source_url: str = "",
        config: dict | None = None,
        creator: str = "",
    ) -> dict:
        """
        上传文件并启动异步解析。

        Args:
            config: 文档切分配置（可选），包含 chunk_strategy 等参数

        返回: {"task_id": ..., "doc_id": ..., "doc_type": ..., "doc_name": ..., "error": None}
        """
        # 验证文件类型
        try:
            self._validate_file(file)
        except HTTPException as e:
            return {"task_id": "", "doc_id": "", "doc_type": "", "doc_name": file.filename or "", "error": e.detail}

        # 生成文档 ID
        doc_id = str(uuid.uuid4())
        file_path = self._safe_file_path(settings.UPLOAD_DIR, doc_id, file.filename or "upload")

        # 流式读取文件，边读边检查大小限制
        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        file_size_mb = 0.0
        file_size = 0
        with open(file_path, "wb") as f:
            while True:
                chunk = file.file.read(1024 * 1024)  # 每次读 1MB
                if not chunk:
                    break
                file_size += len(chunk)
                file_size_mb = file_size / (1024 * 1024)
                if file_size_mb > settings.MAX_FILE_SIZE_MB:
                    f.close()
                    os.unlink(file_path)
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large ({file_size_mb:.1f}MB > {settings.MAX_FILE_SIZE_MB}MB)",
                    )
                f.write(chunk)

        # 检测文档类型（通过扩展名）
        ext = Path(file.filename or "").suffix.lower().lstrip(".")
        doc_type = ext if ext in settings.ALLOWED_EXTENSIONS else "unknown"
        doc_name = file.filename or ""

        # 从配置中提取切分策略（默认 smart）
        chunk_strategy = config.get("chunk_strategy", "smart") if config else "smart"

        # 提交 Celery 任务（传递所有需要的信息）
        task_id = enqueue_parse(
            file_path, doc_id,
            kb_id=kb_id, source_type=source_type,
            source_url=source_url, doc_name=doc_name,
            file_size=file_size, doc_type=doc_type,
            chunk_strategy=chunk_strategy,
        )

        logger.info("document_uploaded", doc_id=doc_id, task_id=task_id, doc_type=doc_type, kb_id=kb_id, chunk_strategy=chunk_strategy)
        return {"task_id": task_id, "doc_id": doc_id, "doc_type": doc_type, "doc_name": doc_name, "error": None}

    def upload_batch(
        self,
        files: list[UploadFile],
        kb_id: str = "",
        source_type: str = "local_file",
        config: dict | None = None,
        creator: str = "",
    ) -> dict:
        """
        批量上传文件并启动异步解析。

        Args:
            config: 文档切分配置（可选，应用于所有文件）

        返回: {"total": ..., "success": ..., "failed": ..., "results": [...]}
        """
        results = []
        for file in files:
            try:
                result = self.upload(file, kb_id=kb_id, source_type=source_type, config=config, creator=creator)
                results.append(result)
            except Exception as exc:
                results.append({
                    "task_id": "",
                    "doc_id": "",
                    "doc_type": "",
                    "doc_name": file.filename or "",
                    "error": str(exc),
                })

        success_count = sum(1 for r in results if r.get("error") is None)
        failed_count = len(results) - success_count

        logger.info("batch_uploaded", total=len(results), success=success_count, failed=failed_count, kb_id=kb_id)
        return {
            "total": len(results),
            "success": success_count,
            "failed": failed_count,
            "results": results,
        }

    @staticmethod
    def _validate_file(file: UploadFile) -> None:
        ext = Path(file.filename or "").suffix.lower().lstrip(".")
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}",
            )

    @staticmethod
    def _safe_file_path(upload_dir: str, doc_id: str, original_filename: str) -> str:
        ext = Path(original_filename).suffix
        safe_name = f"{doc_id}{ext}"
        resolved_dir = os.path.realpath(upload_dir)
        file_path = os.path.join(resolved_dir, safe_name)
        resolved_path = os.path.realpath(file_path)

        if not resolved_path.startswith(resolved_dir + os.sep):
            raise HTTPException(status_code=400, detail="Invalid file path")

        return resolved_path


# 单例
document_service = DocumentService()
