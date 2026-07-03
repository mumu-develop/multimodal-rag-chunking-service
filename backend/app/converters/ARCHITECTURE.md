# 转换器架构设计

## 设计原则

### 1. 可维护性 (Maintainability)
- **统一接口**：所有转换器继承 `MarkdownConverter` 抽象基类
- **单一职责**：每个模块职责明确（models: 数据定义, table_utils: 通用工具）
- **复用代码**：`table_utils.py` 提供通用表格处理，避免重复
- **强类型**：Pydantic 模型确保类型安全，IDE 支持更好

### 2. 可扩展性 (Extensibility)
- **注册机制**：`register_converter()` 可动态注册新转换器
- **插件架构**：新文件类型只需实现 `convert_full()` 方法
- **配置驱动**：转换器可配置（如 `PARSING_STRATEGY`）

### 3. 健壮性 (Robustness)
- **降级策略**：表格解析失败 → 保留原文本，不中断流程
- **错误处理**：明确的异常类型 `ConversionError`, `TableParsingError`
- **边界检查**：合并单元格验证、空数据处理
- **日志记录**：关键操作有日志，便于调试

## 模块结构

```
app/converters/
├── __init__.py          # 抽象基类 + 注册机制
├── models.py            # Pydantic 数据模型
├── table_utils.py       # 通用表格处理工具
├── docx_converter.py    # DOCX 转换器
├── pdf_converter.py     # PDF 转换器
├── pptx_converter.py    # PPTX 转换器
└── md_converter.py      # Markdown 转换器
```

## 核心类图

```
MarkdownConverter (ABC)
    │
    ├── convert(file_path, doc_id) -> str       # 兼容旧接口
    ├── convert_full(file_path, doc_id) -> ConverterOutput  # 推荐接口
    └── extract_images(file_path, doc_id) -> list[str]
    │
    └── 子类:
        ├── DocxConverter
        ├── PdfConverter
        ├── PptxConverter
        └── MdConverter
```

## 数据模型

```python
ConverterOutput:
    markdown: str              # Markdown 文本
    tables: list[TableData]    # 表格列表（含 html 字段）
    images: list[ImageData]    # 图片列表
    code_blocks: list[CodeBlockData]  # 代码块列表
    doc_type: str              # 文档类型
    page_count: int            # 页数

TableData:
    index: int
    headers: list[str]
    rows: list[list[str]]
    merges: list[dict]         # 合并单元格信息
    html: str                  # HTML 表格（前端必需）
```

## 如何添加新转换器

```python
from app.converters import MarkdownConverter, register_converter
from app.converters.models import ConverterOutput
from app.converters.table_utils import build_table_html

class XlsxConverter(MarkdownConverter):
    def convert(self, file_path, doc_id) -> str:
        return self.convert_full(file_path, doc_id).markdown

    def convert_full(self, file_path, doc_id) -> ConverterOutput:
        # 1. 解析文件
        # 2. 生成 Markdown
        # 3. 提取表格（必须含 html 字段）
        # 4. 返回 ConverterOutput
        return ConverterOutput(...)

    def extract_images(self, file_path, doc_id) -> list[str]:
        return []

# 注册
register_converter('xlsx', XlsxConverter)
```

## 降级策略

| 场景 | 降级策略 |
|:---|:---|
| 文件读取失败 | 抛出 `ConversionError`，任务失败 |
| 图片提取失败 | 返回空列表，继续处理 |
| 表格解析失败 | 返回空 `TableData`，保留原文本 |
| 合并单元格越界 | 忽略该合并，继续生成 HTML |
| 列数不一致 | 补齐空值或截断 |

## 前端兼容

所有表格切片 `content` 字段统一为 JSON 格式：

```json
{
  "type": "table",
  "headers": ["列1", "列2"],
  "rows": [["a", "b"]],
  "html": "<table class='chunk-table'><thead>...</table>"
}
```

前端只需：
1. 解析 `content` JSON
2. 提取 `html` 字段
3. 安全渲染（使用 `dangerouslySetInnerHTML` 但需配合 CSS）