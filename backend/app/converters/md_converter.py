"""Markdown 文件转换器。

Markdown 文件已经是目标格式，只需要：
1. 原样返回内容
2. 提取图片路径（如果有）

设计原则：
- 健壮性：文件读取失败时抛出 ConversionError
- 可维护性：复用 table_utils 通用工具
"""
import os
import re
import structlog

from app.config import settings
from app.converters import MarkdownConverter
from app.converters.models import ConverterOutput, ConversionError, ImageData, CodeBlockData
from app.converters.table_utils import extract_markdown_tables, extract_code_blocks

logger = structlog.get_logger()


class MdConverter(MarkdownConverter):
    """Markdown 文件转换器（原样返回 + 结构化提取）。"""

    def convert(self, file_path: str, doc_id: str) -> str:
        """读取 Markdown 文件内容（兼容旧接口）。"""
        result = self.convert_full(file_path, doc_id)
        return result.markdown

    def convert_full(self, file_path: str, doc_id: str) -> ConverterOutput:
        """读取 Markdown 文件并提取结构化数据。

        Args:
            file_path: Markdown 文件路径
            doc_id: 文档 ID

        Returns:
            ConverterOutput 模型

        Raises:
            ConversionError: 文件读取失败
        """
        try:
            # 读取文件
            markdown_text = self._read_file(file_path)
        except Exception as e:
            logger.exception("md_file_read_failed", file_path=file_path, error=str(e))
            raise ConversionError(f"无法读取文件: {e}", doc_type="md", file_path=file_path)

        try:
            # 提取图片
            image_paths = self._extract_images_internal(file_path, doc_id)
            images_data = [ImageData(path=p, index=i) for i, p in enumerate(image_paths)]
        except Exception as e:
            # 图片提取失败不影响转换，降级为空列表
            logger.warning("md_image_extract_failed", file_path=file_path, error=str(e))
            images_data = []

        try:
            # 使用通用工具提取表格和代码块
            tables_data, markdown_text = extract_markdown_tables(markdown_text, exclude_code_blocks=True)
            code_blocks_raw, _ = extract_code_blocks(markdown_text)
            code_blocks_data = [
                CodeBlockData(index=cb.get("index", i), content=cb.get("content", ""), language=cb.get("language", ""))
                for i, cb in enumerate(code_blocks_raw)
            ]
        except Exception as e:
            logger.warning("md_content_extract_failed", file_path=file_path, error=str(e))
            # 降级：不提取表格和代码块，保持原文本
            tables_data = []
            code_blocks_data = []

        logger.info(
            "md_file_converted",
            file_path=file_path,
            doc_id=doc_id,
            tables=len(tables_data),
            images=len(images_data),
            code_blocks=len(code_blocks_data),
        )

        return ConverterOutput(
            markdown=markdown_text,
            tables=tables_data,
            images=images_data,
            code_blocks=code_blocks_data,
            doc_type="markdown",
            page_count=1,
        )

    def _read_file(self, file_path: str) -> str:
        """读取 Markdown 文件并清理。"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # 简单清理：移除多余空行
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content.strip()

    def _extract_images_internal(self, file_path: str, doc_id: str) -> list[str]:
        """内部图片提取方法。"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 匹配本地图片路径
        pattern = r"!\[.*?\]\(([^http].*?\.(png|jpg|jpeg|gif|webp))\)"
        matches = re.findall(pattern, content, re.IGNORECASE)

        image_paths: list[str] = []
        for match in matches:
            img_path = match[0]
            # 处理相对路径
            if img_path.startswith("./"):
                img_path = img_path[2:]
            if not os.path.isabs(img_path):
                img_path = os.path.join(os.path.dirname(file_path), img_path)

            if os.path.exists(img_path):
                # 复制到输出目录
                output_dir = settings.IMAGE_OUTPUT_DIR
                os.makedirs(output_dir, exist_ok=True)
                ext = os.path.splitext(img_path)[1]
                dest_name = f"{doc_id}_md_img_{len(image_paths)}{ext}"
                dest_path = os.path.join(output_dir, dest_name)

                import shutil
                shutil.copy2(img_path, dest_path)
                image_paths.append(dest_path)

        return image_paths

    def extract_images(self, file_path: str, doc_id: str) -> list[str]:
        """从 Markdown 文件中提取图片引用路径（公开接口）。"""
        try:
            return self._extract_images_internal(file_path, doc_id)
        except Exception as e:
            logger.warning("md_image_extract_failed", file_path=file_path, error=str(e))
            return []