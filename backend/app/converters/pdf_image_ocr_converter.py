"""PDF 图片化 + OCR 转换器。

将 PDF 每一页转换为图片（base64），调用 OCR 服务识别内容，输出 Markdown。

适用场景：
- 扫描版 PDF（无法直接提取文本）
- 复杂排版 PDF（多栏、嵌套表格）
- 图片型 PDF（CAD 图、手写文档）

与现有 PdfConverter 的区别：
- PdfConverter: 使用 unstructured 直接解析 PDF 文本结构
- PdfImageOcrConverter: 图片化后 OCR，适合无法直接解析的 PDF

实现特点：
- 使用 PyMuPDF (fitz) 转换 PDF（已安装，无需额外依赖）
- 图片以 base64 传输，无需临时文件
- 支持 ThreadPoolExecutor 并发处理
- OCR 结果保存到 uploads/{文件名}/ 目录（Markdown + 图片）
"""
import base64
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import fitz  # PyMuPDF
import structlog

from app.config import settings
from app.clients.ocr_client import ocr_client
from app.converters import MarkdownConverter
from app.converters.models import ConverterOutput, ImageData

logger = structlog.get_logger()


class PdfImageOcrConverter(MarkdownConverter):
    """PDF 图片化 + OCR 转换器。"""

    def convert(self, file_path: str, doc_id: str) -> str:
        """将 PDF 文件转换为 Markdown（兼容旧接口）。"""
        result = self.convert_full(file_path, doc_id)
        return result["markdown"]

    def convert_full(self, file_path: str, doc_id: str) -> dict:
        """
        将 PDF 文件转换为结构化数据（图片化 + OCR）。

        流程：
        1. 使用 PyMuPDF 将 PDF 每页转为 base64 图片
        2. 并发调用 OCR 服务识别每页
        3. 汇总所有页面的 Markdown 输出
        4. 保存 Markdown 和图片到 uploads/{文件名}/ 目录

        参数：
            file_path: PDF 文件路径
            doc_id: 文档 ID

        返回：
            {
                "markdown": str,           # Markdown 文本
                "tables": list[dict],      # 表格结构化数据（暂无）
                "images": list[dict],      # 图片信息（每页图片）
                "code_blocks": list[dict], # 代码块信息（暂无）
                "page_count": int,         # 页数
                "output_dir": str,         # 输出目录路径
            }
        """
        # 获取原始文件名（不含扩展名）
        file_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(file_name)[0]

        # 创建输出目录：uploads/{文件名}/
        output_dir = os.path.join(settings.UPLOAD_DIR, file_name_without_ext)
        os.makedirs(output_dir, exist_ok=True)

        # 步骤1：PDF 转图片（base64）
        images_base64 = self._pdf_to_images_base64(file_path)
        if not images_base64:
            logger.warning("pdf_no_pages", doc_id=doc_id)
            return {
                "markdown": "",
                "tables": [],
                "images": [],
                "code_blocks": [],
                "page_count": 0,
                "output_dir": output_dir,
            }

        total_pages = len(images_base64)
        logger.info("pdf_pages_converted", doc_id=doc_id, pages=total_pages, dpi=settings.PDF_IMAGE_DPI)

        # 步骤2：并发 OCR 识别
        results = ocr_client.recognize_images_parallel(images_base64, desc=f"OCR {doc_id}")

        # 步骤3：汇总输出，使用分隔符而不是 Markdown 标题
        # 避免被 MarkdownChunker 的标题识别切分
        markdown_with_images = []
        for idx, page_md in enumerate(results):
            page_num = idx + 1
            # 图片引用
            img_ref = f"\n![Page {page_num}](page_{page_num}.png)\n\n"
            # 使用 --- 分隔符，不使用 ## 标题
            markdown_with_images.append(f"\n--- PAGE {page_num} ---\n\n{img_ref}{page_md}")

        full_markdown = "\n".join(markdown_with_images)

        # 步骤4：保存 Markdown 到文件
        md_file_path = os.path.join(output_dir, f"{file_name_without_ext}.md")
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write(full_markdown)
        logger.info("ocr_markdown_saved", path=md_file_path, length=len(full_markdown))

        # 步骤5：保存图片到同一目录
        images_data = self._save_images_to_dir(images_base64, output_dir, file_name_without_ext)

        logger.info(
            "pdf_ocr_conversion_complete",
            doc_id=doc_id,
            pages=total_pages,
            markdown_len=len(full_markdown),
            output_dir=output_dir,
        )

        return {
            "markdown": full_markdown,
            "tables": [],  # OCR 识别表格后可填充
            "images": images_data,
            "code_blocks": [],  # OCR 识别代码块后可填充
            "page_count": total_pages,
            "output_dir": output_dir,
        }

    def _pdf_to_images_base64(self, file_path: str) -> list[str]:
        """
        将 PDF 每页转换为 base64 编码的 PNG 图片。

        使用 PyMuPDF (fitz)，无需额外依赖。

        参数：
            file_path: PDF 文件路径

        返回：
            base64 图片列表
        """
        try:
            doc = fitz.open(file_path)
            images_base64 = []
            dpi = settings.PDF_IMAGE_DPI

            for page_num in range(len(doc)):
                page = doc[page_num]
                # DPI 转换：fitz Matrix 使用 72 DPI 基准
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                images_base64.append(img_base64)

            doc.close()
            return images_base64

        except Exception as e:
            logger.exception("pdf_to_image_failed", path=file_path, error=str(e))
            raise

    def _save_images_to_dir(self, images_base64: list[str], output_dir: str, file_name: str) -> list[dict]:
        """
        将 base64 图片保存到指定目录。

        参数：
            images_base64: base64 图片列表
            output_dir: 输出目录（uploads/{文件名}/）
            file_name: 文件名（不含扩展名）

        返回：
            图片数据列表，包含 path（本地路径）和 url（访问地址）
        """
        images_data = []
        for idx, img_base64 in enumerate(images_base64):
            page_num = idx + 1
            # 图片命名：page_1.png, page_2.png, ...
            image_name = f"page_{page_num}.png"
            image_path = os.path.join(output_dir, image_name)

            # 保存图片
            img_bytes = base64.b64decode(img_base64)
            with open(image_path, "wb") as f:
                f.write(img_bytes)

            # 计算访问 URL
            # URL 格式：/uploads/{文件名}/page_1.png
            image_url = f"/uploads/{file_name}/{image_name}"

            images_data.append({
                "path": image_path,
                "url": image_url,  # 前端访问地址
                "index": idx,
                "page_num": page_num,
            })

        logger.info("ocr_images_saved", dir=output_dir, count=len(images_data))
        return images_data

    def _save_images_to_disk(self, images_base64: list[str], doc_id: str) -> list[dict]:
        """
        将 base64 图片保存到磁盘（兼容旧接口）。

        参数：
            images_base64: base64 图片列表
            doc_id: 文档 ID

        返回：
            图片数据列表
        """
        output_dir = settings.IMAGE_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        images_data = []
        for idx, img_base64 in enumerate(images_base64):
            page_num = idx + 1
            image_name = f"{doc_id}_ocr_page_{page_num}.png"
            image_path = os.path.join(output_dir, image_name)

            # 保存图片
            img_bytes = base64.b64decode(img_base64)
            with open(image_path, "wb") as f:
                f.write(img_bytes)

            # 计算访问 URL
            image_url = f"/uploads/images/{image_name}"

            images_data.append({
                "path": image_path,
                "url": image_url,
                "index": idx,
                "page_num": page_num,
            })

        logger.info("ocr_images_saved", doc_id=doc_id, count=len(images_data))
        return images_data

    def extract_images(self, file_path: str, doc_id: str) -> list[str]:
        """
        提取图片（图片化模式下，返回保存的图片路径）。

        参数：
            file_path: PDF 文件路径
            doc_id: 文档 ID

        返回：
            图片路径列表
        """
        images_base64 = self._pdf_to_images_base64(file_path)
        images_data = self._save_images_to_disk(images_base64, doc_id)
        return [img["path"] for img in images_data]

    def get_doc_type(self) -> str:
        """获取文档类型标识。"""
        return "pdf_ocr"