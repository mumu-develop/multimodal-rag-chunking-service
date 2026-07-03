"""统一转换器输出数据模型。

使用 Pydantic 强类型模型，确保：
- 类型安全
- 自动验证
- 文档化
- 易于扩展
"""
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class TableData(BaseModel):
    """表格结构化数据模型。"""

    index: int = Field(..., ge=0, description="表格索引")
    headers: list[str] = Field(default_factory=list, description="表头列表")
    rows: list[list[str]] = Field(default_factory=list, description="数据行列表")
    merges: list[dict[str, int]] = Field(
        default_factory=list,
        description="合并单元格信息 [{row, col, rowspan, colspan}]"
    )
    html: str = Field(..., min_length=0, description="HTML 表格字符串（前端渲染必需）")
    row_count: int = Field(default=0, ge=0, description="总行数")
    col_count: int = Field(default=0, ge=0, description="总列数")

    # 嵌套表格支持（可选）
    nested_tables: Optional[list[dict[str, Any]]] = Field(default=None, description="嵌套表格列表")
    has_nested: bool = Field(default=False, description="是否包含嵌套表格")

    # 大表格切分支持（可选）
    table_part: Optional[int] = Field(default=None, description="切分后的部分编号")
    table_total_parts: Optional[int] = Field(default=None, description="切分后的总部分数")

    @field_validator("html")
    @classmethod
    def validate_html(cls, v: str) -> str:
        """确保 HTML 不为空（除非表格本身为空）。"""
        # 如果有 headers 或 rows，html 应不为空
        # 这里只做基本检查，不做严格验证
        return v

    def is_valid(self) -> bool:
        """检查表格是否有效（至少有表头）。"""
        return len(self.headers) > 0 or len(self.rows) > 0


class CodeBlockData(BaseModel):
    """代码块数据模型。"""

    index: int = Field(default=0, ge=0, description="代码块索引")
    content: str = Field(..., min_length=1, description="代码内容")
    language: str = Field(default="", description="代码语言")

    def is_valid(self) -> bool:
        """检查代码块是否有效。"""
        return len(self.content.strip()) > 0


class ImageData(BaseModel):
    """图片数据模型。"""

    path: str = Field(..., description="图片存储路径")
    index: int = Field(default=0, ge=0, description="图片索引")
    alt_text: Optional[str] = Field(default=None, description="图片替代文本")
    caption: Optional[str] = Field(default=None, description="图片说明")


class ConverterOutput(BaseModel):
    """转换器统一输出模型。"""

    markdown: str = Field(..., description="Markdown 文本（表格用 [TABLE_idx] 标记）")
    tables: list[TableData] = Field(default_factory=list, description="表格数据列表")
    images: list[ImageData] = Field(default_factory=list, description="图片数据列表")
    code_blocks: list[CodeBlockData] = Field(default_factory=list, description="代码块数据列表")

    # 元信息
    doc_type: str = Field(default="", description="文档类型")
    page_count: int = Field(default=1, ge=1, description="页数/幻灯片数")

    def get_table_by_index(self, idx: int) -> Optional[TableData]:
        """根据索引获取表格数据。"""
        for table in self.tables:
            if table.index == idx:
                return table
        return None

    def validate_consistency(self) -> list[str]:
        """验证 Markdown 标记与数据一致性。"""
        import re
        errors = []

        # 检查所有 [TABLE_idx] 标记都有对应数据
        table_markers = re.findall(r"\[TABLE_(\d+)\]", self.markdown)
        for idx_str in table_markers:
            idx = int(idx_str)
            if not self.get_table_by_index(idx):
                errors.append(f"[TABLE_{idx}] 标记无对应表格数据")

        # 检查所有表格数据都有标记（反向）
        for table in self.tables:
            if f"[TABLE_{table.index}]" not in self.markdown:
                errors.append(f"表格 {table.index} 无对应 Markdown 标记")

        return errors

    model_config = {
        "extra": "forbid",  # 禁止额外字段，防止意外数据
    }


class ConversionError(Exception):
    """转换错误基类。"""

    def __init__(self, message: str, doc_type: str = "", file_path: str = "", details: dict = None):
        super().__init__(message)
        self.doc_type = doc_type
        self.file_path = file_path
        self.details = details or {}


class TableParsingError(ConversionError):
    """表格解析错误。"""
    pass


class ImageExtractionError(ConversionError):
    """图片提取错误。"""
    pass