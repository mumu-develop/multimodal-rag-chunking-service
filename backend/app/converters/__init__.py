"""统一 Markdown 转换层。

将各种文件类型（DOCX、PDF、PPTX、MD）转换为 Markdown 格式，
然后统一使用 MarkdownChunker 进行分块。

设计原则：
1. **可维护性**：统一接口，单一职责，清晰的抽象
2. **可扩展性**：注册机制，易于添加新转换器
3. **健壮性**：错误处理，降级策略，日志记录

优点：
- 统一分块逻辑，不需要为每种文件类型写不同的 chunker
- Markdown 标记明确传递层级信息，不会丢失
- 代码简化，利用 unstructured 成熟的分块能力
"""
from abc import ABC, abstractmethod
from typing import Optional, Type
import structlog

from app.converters.models import ConverterOutput, ConversionError

logger = structlog.get_logger()


class MarkdownConverter(ABC):
    """Markdown 转换器抽象基类。

    所有转换器必须实现：
    - convert(): 转换为 Markdown 文本（兼容旧接口）
    - convert_full(): 转换为完整结构化数据（新接口，推荐）
    - extract_images(): 提取图片

    输出格式统一为 ConverterOutput 模型。
    """

    @abstractmethod
    def convert(self, file_path: str, doc_id: str) -> str:
        """将文件转换为 Markdown 格式（兼容旧接口）。

        Args:
            file_path: 文件路径
            doc_id: 文档 ID

        Returns:
            Markdown 格式的文本

        Raises:
            ConversionError: 转换失败
        """
        pass

    @abstractmethod
    def convert_full(self, file_path: str, doc_id: str) -> ConverterOutput:
        """将文件转换为完整结构化数据（推荐使用）。

        Args:
            file_path: 文件路径
            doc_id: 文档 ID

        Returns:
            ConverterOutput 模型，包含 Markdown、表格、图片、代码块

        Raises:
            ConversionError: 转换失败
        """
        pass

    @abstractmethod
    def extract_images(self, file_path: str, doc_id: str) -> list[str]:
        """提取文件中的图片。

        Args:
            file_path: 文件路径
            doc_id: 文档 ID

        Returns:
            图片路径列表

        Raises:
            ImageExtractionError: 图片提取失败（通常忽略，不影响转换）
        """
        pass

    def get_doc_type(self) -> str:
        """获取文档类型标识。"""
        return self.__class__.__name__.replace("Converter", "").lower()


# 转换器注册表（可扩展）
_CONVERTER_REGISTRY: dict[str, Type[MarkdownConverter]] = {}


def register_converter(doc_type: str, converter_class: Type[MarkdownConverter]) -> None:
    """注册转换器（扩展点）。

    Args:
        doc_type: 文档类型标识（如 'docx', 'pdf'）
        converter_class: 转换器类

    Example:
        register_converter('xlsx', XlsxConverter)
    """
    _CONVERTER_REGISTRY[doc_type.lower()] = converter_class
    logger.info("converter_registered", doc_type=doc_type)


def get_converter(doc_type: str) -> Optional[MarkdownConverter]:
    """根据文档类型获取转换器实例。

    Args:
        doc_type: 文档类型（docx, pdf, pptx, md, xlsx, html 等）

    Returns:
        对应的转换器实例，如果不支持返回 None
    """
    doc_type_lower = doc_type.lower()

    # 如果注册表中已有，直接返回
    if doc_type_lower in _CONVERTER_REGISTRY:
        return _CONVERTER_REGISTRY[doc_type_lower]()

    # 延迟加载内置转换器（避免循环导入）
    builtin_converters = {
        "docx": "app.converters.docx_converter:DocxConverter",
        "pdf": "app.converters.pdf_image_ocr_converter:PdfImageOcrConverter",
        "pptx": "app.converters.pptx_converter:PptxConverter",
        "md": "app.converters.md_converter:MdConverter",
        "markdown": "app.converters.md_converter:MdConverter",
        "xlsx": "app.converters.xlsx_converter:XlsxConverter",
        "xls": "app.converters.xlsx_converter:XlsxConverter",
    }

    if doc_type_lower in builtin_converters:
        import importlib
        module_path, class_name = builtin_converters[doc_type_lower].split(":")
        module = importlib.import_module(module_path)
        converter_class = getattr(module, class_name)
        # 注册到缓存
        _CONVERTER_REGISTRY[doc_type_lower] = converter_class
        return converter_class()

    logger.warning("converter_not_found", doc_type=doc_type)
    return None


def list_supported_types() -> list[str]:
    """列出所有支持的文档类型。"""
    return list(_CONVERTER_REGISTRY.keys()) or ["docx", "pdf", "pptx", "md"]