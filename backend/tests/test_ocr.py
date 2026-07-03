"""OCR 功能测试。

测试内容：
1. OCRClient - 单图识别、并发识别
2. PdfImageOcrConverter - PDF 转 base64 图片、转换流程

运行方式：
- 单元测试：pytest tests/test_ocr.py -v
- 集成测试（需要 OCR 服务）：pytest tests/test_ocr.py -v -m ocr_real
"""
import base64
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# 确保 backend/ 在 sys.path
_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_pdf_path():
    """创建一个简单的测试 PDF。"""
    import fitz  # PyMuPDF

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((100, 100), "测试文档 - 第一页")
    page.insert_text((100, 150), "This is a test document for OCR.")
    page2 = doc.new_page()
    page2.insert_text((100, 100), "测试文档 - 第二页")
    page2.insert_text((100, 150), "Page 2 content here.")
    doc.save(pdf_path)
    doc.close()

    yield pdf_path

    # 清理
    if os.path.exists(pdf_path):
        os.unlink(pdf_path)


@pytest.fixture
def sample_image_base64():
    """生成一个简单的测试图片 base64。"""
    minimal_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    return base64.b64encode(minimal_png).decode("utf-8")


@pytest.fixture
def mock_ocr_response():
    """Mock OCR API 返回的 Markdown 内容。"""
    return """# 测试文档

这是第一页的内容。

## 标题一

- 列表项 1
- 列表项 2

| 列1 | 列2 |
|-----|-----|
| A   | B   |

"""


# ============================================================
# OCRClient 单元测试
# ============================================================

class TestOCRClient:
    """OCR 客户端测试。"""

    def test_client_initialization(self):
        """测试客户端初始化。"""
        from app.clients.ocr_client import OCRClient
        from app.config import settings

        client = OCRClient()
        assert client._api_url == settings.OCR_API_URL
        assert client._timeout == settings.OCR_TIMEOUT
        assert client._max_workers == settings.OCR_MAX_WORKERS

    def test_recognize_image_file_not_found(self):
        """测试图片不存在时返回空字符串。"""
        from app.clients.ocr_client import OCRClient

        client = OCRClient()
        result = client.recognize_image("/nonexistent/path.png")
        assert result == ""

    def test_recognize_image_success_with_mock(self, tmp_path, mock_ocr_response):
        """测试单图识别成功（mock API）。"""
        from app.clients.ocr_client import OCRClient

        # 创建测试图片文件
        image_path = tmp_path / "test.png"
        image_path.write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        ))

        client = OCRClient()

        # Mock _call_ocr_api 方法
        with patch.object(client, "_call_ocr_api", return_value=mock_ocr_response):
            result = client.recognize_image(str(image_path))
            assert result == mock_ocr_response

    def test_recognize_image_base64_with_mock(self, sample_image_base64, mock_ocr_response):
        """测试 base64 图片识别（mock API）。"""
        from app.clients.ocr_client import OCRClient

        client = OCRClient()

        with patch.object(client, "_call_ocr_api", return_value=mock_ocr_response):
            result = client.recognize_image_base64(sample_image_base64)
            assert result == mock_ocr_response

    def test_recognize_images_parallel_order_preserved(self):
        """测试并发识别时顺序保持正确。"""
        from app.clients.ocr_client import OCRClient

        client = OCRClient()
        # 设置一个有效的 URL 以绕过 "未配置" 检查
        client._api_url = "http://test-server/v1"

        images = ["img1", "img2", "img3"]

        # Mock 单图识别方法，返回不同内容
        def mock_recognize(img):
            idx = images.index(img)
            return f"Page {idx + 1} result"

        with patch.object(client, "recognize_image_base64", side_effect=mock_recognize):
            results = client.recognize_images_parallel(images)

            assert len(results) == 3
            assert results[0] == "Page 1 result"
            assert results[1] == "Page 2 result"
            assert results[2] == "Page 3 result"


# ============================================================
# PdfImageOcrConverter 单元测试
# ============================================================

class TestPdfImageOcrConverter:
    """PDF OCR 转换器测试。"""

    def test_converter_initialization(self):
        """测试转换器初始化。"""
        from app.converters.pdf_image_ocr_converter import PdfImageOcrConverter

        converter = PdfImageOcrConverter()
        assert converter.get_doc_type() == "pdf_ocr"

    def test_pdf_to_images_base64(self, sample_pdf_path):
        """测试 PDF 转 base64 图片。"""
        from app.converters.pdf_image_ocr_converter import PdfImageOcrConverter

        converter = PdfImageOcrConverter()
        images = converter._pdf_to_images_base64(sample_pdf_path)

        assert len(images) == 2  # 两页 PDF
        assert all(isinstance(img, str) for img in images)
        assert all(len(img) > 0 for img in images)

        # 验证是有效的 base64
        for img in images:
            decoded = base64.b64decode(img)
            assert len(decoded) > 0

    def test_convert_full_with_mock_ocr(self, sample_pdf_path):
        """测试完整转换流程（mock OCR）。"""
        from app.converters.pdf_image_ocr_converter import PdfImageOcrConverter
        from app.clients.ocr_client import ocr_client

        converter = PdfImageOcrConverter()

        # Mock OCR 返回
        mock_results = [
            "# Page 1\n\nContent of page 1.",
            "# Page 2\n\nContent of page 2.",
        ]

        with patch.object(ocr_client, "recognize_images_parallel", return_value=mock_results):
            result = converter.convert_full(sample_pdf_path, "test_doc_001")

            assert "markdown" in result
            assert "Page 1" in result["markdown"]
            assert "Page 2" in result["markdown"]
            assert result["page_count"] == 2
            assert "---" in result["markdown"]  # 页面分隔符

    def test_save_images_to_disk(self, sample_pdf_path, tmp_path):
        """测试图片保存到磁盘。"""
        from app.converters.pdf_image_ocr_converter import PdfImageOcrConverter
        from app.config import settings

        converter = PdfImageOcrConverter()
        images = converter._pdf_to_images_base64(sample_pdf_path)

        # Mock settings
        with patch.object(settings, "IMAGE_OUTPUT_DIR", str(tmp_path)):
            with patch.object(settings, "EXTRACT_IMAGES", True):
                images_data = converter._save_images_to_disk(images, "test_doc")

                assert len(images_data) == 2
                assert all("path" in img for img in images_data)
                assert all(os.path.exists(img["path"]) for img in images_data)


# ============================================================
# 集成测试（需要真实 OCR 服务）
# ============================================================

@pytest.fixture
def ocr_service_available():
    """检查 OCR 服务是否可用。"""
    import urllib.request
    from app.config import settings

    try:
        # 尝试连接 OCR 服务
        req = urllib.request.Request(
            settings.OCR_API_URL,
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


@pytest.mark.ocr_real
class TestOCRIntegration:
    """OCR 集成测试（需要真实 OCR 服务）。"""

    def test_real_ocr_pdf(self, sample_pdf_path, ocr_service_available):
        """测试真实 OCR 服务处理 PDF。"""
        if not ocr_service_available:
            pytest.skip("OCR 服务不可用")

        from app.converters.pdf_image_ocr_converter import PdfImageOcrConverter

        converter = PdfImageOcrConverter()
        result = converter.convert_full(sample_pdf_path, "real_test_doc")

        assert result["markdown"]
        assert result["page_count"] == 2
        assert len(result["markdown"]) > 50

    def test_real_ocr_api_direct(self, sample_image_base64, ocr_service_available):
        """直接测试 OCR API。"""
        if not ocr_service_available:
            pytest.skip("OCR 服务不可用")

        from app.clients.ocr_client import OCRClient

        client = OCRClient()
        result = client.recognize_image_base64(sample_image_base64)

        assert isinstance(result, str)


# ============================================================
# 手动测试脚本
# ============================================================

def manual_test_ocr(
    pdf_path: str,
    output_path: str = None,
    separate_pages: bool = False,
    save_images: bool = False,
):
    """
    手动 OCR 测试脚本。

    用法：
        python tests/test_ocr.py --pdf /path/to/your.pdf
        python tests/test_ocr.py --pdf /path/to/your.pdf --output result.md
        python tests/test_ocr.py --pdf /path/to/your.pdf --output ./pages --separate
        python tests/test_ocr.py --pdf /path/to/your.pdf --output ./output --save-images

    参数：
        pdf_path: PDF 文件路径
        output_path: 输出路径（可选）
                     - 不指定：只打印预览
                     - 指定文件：保存为单个 Markdown 文件
                     - 指定目录：保存 markdown 和可选图片
        separate_pages: 是否每页单独保存 markdown 文件
        save_images: 是否同时保存每页的图片（PNG 格式）
    """
    from app.converters.pdf_image_ocr_converter import PdfImageOcrConverter
    import os
    import base64

    converter = PdfImageOcrConverter()
    print(f"📄 处理 PDF: {pdf_path}")

    # 获取 PDF 页面图片（base64）
    images_base64 = converter._pdf_to_images_base64(pdf_path)
    total_pages = len(images_base64)
    print(f"📊 PDF 共 {total_pages} 页")

    # OCR 识别
    result = converter.convert_full(pdf_path, "manual_test")
    print(f"✅ OCR 完成，Markdown 长度: {len(result['markdown'])} 字符")

    # 打印预览
    print("\n--- Markdown 内容预览 ---\n")
    preview_len = 500
    print(result["markdown"][:preview_len])
    if len(result["markdown"]) > preview_len:
        print("\n...")

    # 保存到文件
    if output_path:
        os.makedirs(output_path, exist_ok=True)

        # 拆分每页 markdown
        pages_md = result["markdown"].split("\n\n---\n\n")

        for idx in range(1, total_pages + 1):
            # 保存 markdown
            if separate_pages:
                page_md = pages_md[idx - 1] if idx <= len(pages_md) else ""
                page_file = os.path.join(output_path, f"page_{idx}.md")
                with open(page_file, "w", encoding="utf-8") as f:
                    f.write(page_md.strip())
                print(f"  ✅ 第 {idx} 页 markdown: {page_file}")

            # 保存图片
            if save_images:
                img_path = os.path.join(output_path, f"page_{idx}.png")
                img_bytes = base64.b64decode(images_base64[idx - 1])
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                print(f"  ✅ 第 {idx} 页图片: {img_path}")

        # 保存完整 markdown（单文件模式）
        if not separate_pages:
            all_md_path = os.path.join(output_path, "full_document.md")
            with open(all_md_path, "w", encoding="utf-8") as f:
                f.write(result["markdown"])
            print(f"\n✅ 完整 Markdown: {all_md_path}")

        print(f"\n📁 输出目录: {output_path}")
        return output_path

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OCR 测试")
    parser.add_argument("--pdf", required=True, help="PDF 文件路径")
    parser.add_argument("--output", help="输出目录路径")
    parser.add_argument("--separate", action="store_true", help="每页单独保存 markdown 文件")
    parser.add_argument("--save-images", action="store_true", help="同时保存每页图片（PNG）")
    parser.add_argument("--unit", action="store_true", help="运行单元测试")

    args = parser.parse_args()

    if args.unit:
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        manual_test_ocr(args.pdf, args.output, args.separate, args.save_images)


# ============================================================
# 使用示例
# ============================================================
"""
常用命令：

# 1. 只打印预览（不保存文件）
python tests/test_ocr.py --pdf /path/to/your.pdf

# 2. 保存完整 Markdown + 每页图片
python tests/test_ocr.py --pdf /path/to/your.pdf --output ./ocr_output --save-images

# 3. 每页单独 Markdown + 每页图片
python tests/test_ocr.py --pdf /path/to/your.pdf --output ./ocr_output --separate --save-images

# 4. 只保存 Markdown（不保存图片）
python tests/test_ocr.py --pdf /path/to/your.pdf --output ./ocr_output

# 5. 每页单独 Markdown（不保存图片）
python tests/test_ocr.py --pdf /path/to/your.pdf --output ./ocr_output --separate

# 6. 运行单元测试
python tests/test_ocr.py --pdf dummy.pdf --unit

# 7. 运行 pytest 单元测试
pytest tests/test_ocr.py -v

# 8. 运行集成测试（需要 OCR 服务）
pytest tests/test_ocr.py -v -m ocr_real
"""