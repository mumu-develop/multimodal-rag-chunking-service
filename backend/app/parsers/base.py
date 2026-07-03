from abc import ABC, abstractmethod

from pydantic import BaseModel


class ParsedPage(BaseModel):
    """从文档中提取的单页/幻灯片数据。

    字段说明：
    - text: 页面文本内容（Markdown 格式）
    - headings: 标题列表 [{"level": int, "text": str}, ...]
    - tables: HTML 格式表格列表（用于搜索）
    - tables_data: 结构化表格数据列表（用于前端渲染）
        [{index, headers, rows, merges, html, nested_tables, has_nested, row_count, col_count}, ...]
        - nested_tables: 嵌套表格列表（独立结构化数据）
        - has_nested: 是否包含嵌套表格
    - images: 图片路径列表
    - images_data: 结构化图片数据列表（用于前端渲染）
        [{path, index, alt_text}, ...]
    - code_blocks: 代码块内容列表
    - code_blocks_data: 结构化代码块数据列表
        [{content, language, paragraph_index}, ...]
    """
    doc_id: str
    page_num: int
    text: str = ""
    headings: list[dict] = []  # [{"level": 1, "text": "..."}, ...]
    tables: list[str] = []     # HTML 格式表格
    tables_data: list[dict] = []  # 结构化表格数据（用于前端渲染）
    images: list[str] = []     # 提取的图片路径
    images_data: list[dict] = []  # 结构化图片数据（用于前端渲染）
    code_blocks: list[str] = []
    code_blocks_data: list[dict] = []  # 结构化代码块数据
    doc_type: str = ""


class DocumentParser(ABC):
    """文档解析器抽象基类。"""

    @abstractmethod
    def parse(self, file_path: str, doc_id: str) -> list[ParsedPage]:
        """解析文档文件并返回 ParsedPage 对象列表。"""
        ...
