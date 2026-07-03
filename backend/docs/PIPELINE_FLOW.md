# 文档处理完整流程

## 总览

从文件上传到切片结束的完整流水线：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              文件上传入口                                    │
│  POST /api/v1/upload → upload.py → document_service.upload()                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Celery 任务队列                                 │
│  enqueue_parse() → parse_and_chunk_task (异步执行)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          阶段 1: 解析 (10%)                                  │
│  Converter.convert_full() → Markdown + 结构化数据                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          阶段 2: 分块 (20%)                                  │
│  markdown_chunker.chunk() → 切片列表 + 树状结构                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          阶段 3: VLM Caption (30%)                          │
│  llm_client.summarize_image() → 图片描述生成                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          阶段 4: 嵌入 (50%)                                  │
│  unified_embedder.encode_texts/images() → 2560维向量                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          阶段 5: Milvus 存储 (70%)                          │
│  milvus_repo.batch_upsert() → 向量数据库                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          阶段 6: MySQL 存储 (85%)                           │
│  chunk_repo.insert_chunks() → 元数据持久化                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          阶段 7: 完成 (100%)                                 │
│  doc_repo.create() → Document 记录创建 + 任务状态更新                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 阶段 1: 文件上传 (API 入口)

### 入口路由 (`app/api/routers/upload.py`)

```python
@router.post("", response_model=UploadResponse)
async def upload_file(file, kb_id, source_type, source_url):
    # 1. 验证 KB 存在
    collection_name = _validate_kb(kb_id)
    # 2. 调用 document_service.upload()
    data = document_service.upload(file, kb_id=kb_id, ...)
    # 3. 返回 task_id + doc_id
    return _ok(trace_id, data=UploadResponseData(**data))
```

### 上传服务 (`app/services/document_service.py`)

```python
class DocumentService:
    def upload(self, file, kb_id, source_type, source_url):
        # 1. 验证文件类型（allowed extensions）
        self._validate_file(file)  # 只允许 pdf/docx/pptx/md

        # 2. 生成文档 ID
        doc_id = str(uuid.uuid4())

        # 3. 流式保存文件（边读边检查大小限制）
        file_path = self._safe_file_path(settings.UPLOAD_DIR, doc_id, file.filename)
        # 流式读取，每次 1MB，超过 MAX_FILE_SIZE_MB(50MB) 则拒绝

        # 4. 检测文档类型（通过扩展名）
        ext = Path(file.filename).suffix.lower().lstrip(".")
        doc_type = ext if ext in settings.ALLOWED_EXTENSIONS else "unknown"

        # 5. 提交 Celery 任务
        task_id = enqueue_parse(file_path, doc_id, kb_id, doc_type, ...)

        return {"task_id": task_id, "doc_id": doc_id, "doc_type": doc_type}
```

---

## 阶段 2: 解析 (10%)

### Celery 任务入口 (`app/tasks.py`)

```python
@celery_app.task(bind=True, name="tasks.parse_and_chunk")
def parse_and_chunk_task(self, file_path, doc_id, kb_id, collection_name, ...):
    task_id = self.request.id

    # 检查缓存（重试场景）
    cached = _restore_pipeline_state(task_id)
    if cached:
        chunks = cached["chunks"]
        entities = cached["entities"]
    else:
        # ========== 阶段 1: 解析 ==========
        # 获取 Converter
        converter = get_converter(doc_type)
        if not converter:
            raise ConversionError(f"不支持的文档类型: {doc_type}")

        # 调用 convert_full 获取完整结构化数据
        result = converter.convert_full(file_path, doc_id)
        markdown_text = result.markdown
        tables_data = [t.model_dump() for t in result.tables]
        images_data = [img.model_dump() for img in result.images]
        code_blocks_data = [cb.model_dump() for cb in result.code_blocks]
        image_paths = [img["path"] for img in images_data]
```

### Converter 工作原理 (`app/converters/`)

#### ConverterOutput 数据模型

```python
class ConverterOutput(BaseModel):
    markdown: str              # Markdown 文本（含 [TABLE_idx] 标记）
    tables: list[TableData]    # 表格结构化数据
    images: list[ImageData]    # 图片数据
    code_blocks: list[CodeBlockData]  # 代码块数据
    doc_type: str              # 文档类型
    page_count: int            # 页数

class TableData(BaseModel):
    index: int                 # 表格索引（对应 [TABLE_idx]）
    headers: list[str]         # 表头
    rows: list[list[str]]      # 数据行
    merges: list[dict]         # 合并单元格信息
    html: str                  # HTML 表格（前端渲染必需）
    row_count: int
    col_count: int
```

#### 各类型 Converter 处理逻辑

| 文件类型 | Converter | 解析方式 | 表格处理 |
|:---|:---|:---|:---|
| **PDF** | `PdfConverter` | unstructured + pdfplumber | Table 元素 + 文本表格 |
| **DOCX** | `DocxConverter` | python-docx + unstructured | 嵌套表格 + 合并单元格 |
| **PPTX** | `PptxConverter` | python-pptx | 表格对象 → HTML |
| **Markdown** | `MdConverter` | 原样读取 + 提取 | `|` 语法表格 → HTML |
| **Excel** | `XlsxConverter` | openpyxl | 每个 Sheet → HTML 表格 |

---

## 阶段 3: 分块 (20%) - 详细流程

### MarkdownChunker 核心流程 (`app/converters/markdown_chunker.py`)

```
Markdown 文本
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│  第一层：按标题切分 (_split_markdown_by_headers)            │
│  - 根据 # ## ### 标题层级识别内容边界                       │
│  - 维护标题栈，保留完整标题路径                              │
│  - 每个 section 包含：heading_path + content               │
└────────────────────────────────────────────────────────────┘
     │ sections = [{heading_path, level, content}]
     ▼
┌────────────────────────────────────────────────────────────┐
│  第二层：内容片段识别 (_split_by_markers_and_code)          │
│  - 识别 [TABLE_idx] 标记 → 注入表格结构化数据               │
│  - 识别 Markdown `|` 表格 → 直接解析                       │
│  - 识别 ``` 代码块 → 保持完整                              │
│  - 其他内容 → 普通文本                                      │
└────────────────────────────────────────────────────────────┘
     │ fragments = [{type: table_marker/markdown_table/code_block/text}]
     ▼
┌────────────────────────────────────────────────────────────┐
│  第三层：生成切片 (_make_chunk/_make_table_chunk/etc)       │
│  - table_marker → 创建表格切片（JSON 格式）                 │
│  - markdown_table → 解析并创建表格切片                      │
│  - code_block → 创建代码切片（保持完整）                    │
│  - text → 检查长度，超长则二次切分                          │
└────────────────────────────────────────────────────────────┘
     │ chunks_list + tree_data
     ▼
  最终输出
```

### 第一层：按标题切分详解

**方法**: `_split_markdown_by_headers(markdown_text)`

**原理**: 维护标题栈，遇到标题时更新栈，正文归属到完整路径下

```markdown
输入示例：
# 一级标题 A
内容 A1

## 二级标题 B
内容 B1

### 三级标题 C
内容 C1

# 一级标题 D
内容 D1
```

输出结果：
```python
[
  {
    "heading_path": ["一级标题 A"],
    "level": 1,
    "content": "内容 A1"
  },
  {
    "heading_path": ["一级标题 A", "二级标题 B"],
    "level": 2,
    "content": "内容 B1"
  },
  {
    "heading_path": ["一级标题 A", "二级标题 B", "三级标题 C"],
    "level": 3,
    "content": "内容 C1"
  },
  {
    "heading_path": ["一级标题 D"],
    "level": 1,
    "content": "内容 D1"
  },
]
```

**关键逻辑**:
- 检测 `#` 开头的行
- 统计 `#` 数量确定层级 (1-6)
- 栈操作：遇到同级或更深层级时，弹出栈中 >= 当前 level 的标题
- 代码块保护：`in_code_block` 状态跟踪，避免代码中 `#` 被误识别

### 第二层：内容片段识别详解

**方法**: `_split_by_markers_and_code(content, tables_map)`

**识别四种片段类型**:

| 片段类型 | 识别方式 | 处理逻辑 |
|:---|:---|:---|
| `table_marker` | `[TABLE_idx]` 正则匹配 | 从 `tables_map` 注入结构化数据 |
| `markdown_table` | `|` 语法表格行识别 | 直接解析为 TableData |
| `code_block` | ` ``` ``` 正则匹配 | 保持完整，language 字段 |
| `text` | 其他内容 | 后续可能二次切分 |

**排除代码块内的表格**:
```python
# 先找出代码块的范围
code_block_ranges = [(block["start"], block["end"]) for block in code_blocks]
# Markdown 表格识别时排除这些范围
markdown_tables = self._find_markdown_tables(content, exclude_ranges=code_block_ranges)
```

### 第三层：切片生成详解

#### 表格切片 (`_make_table_chunk`)

```python
def _make_table_chunk(self, chunk_id, doc_id, paragraph_id, table_data, ...):
    # table_data 包含：headers, rows, merges, html

    # content 存储完整结构化数据（JSON 格式）
    content_data = {
        "type": "table",
        "headers": headers,
        "rows": rows,
        "merges": merges,
        "row_count": row_count,
        "col_count": col_count,
        "html": html,  # HTML 用于前端渲染
    }

    return {
        "chunk_id": chunk_id,
        "chunk_type": "table",
        "content": json.dumps(content_data),
        "metadata": {
            "row_count": row_count,
            "col_count": col_count,
        },
    }
```

#### 代码切片 (`_make_code_chunk`)

```python
def _make_code_chunk(self, chunk_id, doc_id, paragraph_id, code_content, language, ...):
    # 代码块保持完整，不切分

    return {
        "chunk_id": chunk_id,
        "chunk_type": "code",
        "content": code_content,  # 原始代码
        "metadata": {
            "language": language,
            "is_code_block": True,
        },
    }
```

#### 图片切片

```python
# 图片切片在 chunk() 方法末尾单独处理
for img_idx, img_path in enumerate(image_paths):
    img_chunk = {
        "chunk_id": f"{doc_id}_img_{uuid}",
        "chunk_type": "image",
        "content": json.dumps({"type": "image", "path": img_path}),
        "metadata": {
            "image_path": img_path,
            "vlm_caption": "",  # 后续 VLM 填充
        },
    }
```

---

## 三阶段切分策略：标题 → 语义 → 递归

### 可配置切分方案

系统支持用户在界面上选择切分策略，通过 `chunk_strategy` 参数控制：

| 策略 | 名称 | 流程 | 适用场景 | 特点 |
|:---|:---|:---|:---|:---|
| **smart** | 智能切分 | 标题 → 语义 → 递归 | 默认方案，最高质量检索 | 三阶段组合，兼顾结构与语义 |
| **heading** | 标题切分 | 只按标题切分 | 标题层级明确的文档 | 保持原始文档结构，便于导航 |
| **semantic** | 语义切分 | 只按语义切分 | 论文、技术文档 | 保持语义连贯性，无标题依赖 |
| **recursive** | 递归切分 | 只按递归切分 | 快速处理场景 | 最简单快速，段落→句子→截断 |
| **fixed** | 固定长度 | 固定字符数切分 | 测试用途 | 最基础方案，不考虑任何边界 |

### API 使用示例

```bash
# 上传时指定切分策略
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@document.pdf" \
  -F "kb_id=1" \
  -F "chunk_strategy=smart"  # 可选: smart/heading/semantic/recursive/fixed
```

### 核心流程（smart 策略）

```
Markdown 文本
     │
     ▼ 统一清洗入口
┌────────────────────────────────────────────────────────────┐
│  _clean_markdown()                                          │
│  - 基础清洗：控制字符、多余空白                              │
│  - 噪音过滤：页码、页眉页脚、水印关键词                       │
└────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│  第一阶段：按标题切分 (_split_markdown_by_headers)           │
│  - 根据 # ## ### 标题层级识别内容边界                       │
│  - 维护标题栈，保留完整标题路径                              │
│  - 每个 section 包含：heading_path + content               │
└────────────────────────────────────────────────────────────┘
     │ sections = [{heading_path, level, content}]
     ▼
┌────────────────────────────────────────────────────────────┐
│  第二阶段：内容片段识别 (_split_by_markers_and_code)         │
│  - 识别 [TABLE_idx] 标记 → 注入表格结构化数据               │
│  - 识别 Markdown `|` 表格 → 直接解析                       │
│  - 识别 ``` 代码块 → 保持完整                              │
│  - 其他内容 → 普通文本                                      │
└────────────────────────────────────────────────────────────┘
     │ fragments = [{type: table_marker/markdown_table/code_block/text}]
     ▼
┌────────────────────────────────────────────────────────────┐
│  第三阶段：语义 + 递归切分 (_semantic_then_recursive_split)  │
│  普通文本片段的处理流程：                                    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Step 1: 语义切分                                    │   │
│  │  - 拆分为句子（支持中英文）                          │   │
│  │  - 计算句子嵌入向量                                  │   │
│  │  - 计算相邻句子语义相似度                            │   │
│  │  - 在低相似度位置（语义边界）切分                    │   │
│  └─────────────────────────────────────────────────────┘   │
│       │ semantic_segments                                  │
│       ▼                                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Step 2: 长度检查 + 递归切分                         │   │
│  │  - 如果语义片段 ≤ max_size → 直接输出               │   │
│  │  - 如果语义片段 > max_size → 递归切分兜底           │   │
│  │    (段落 → 句子 → 强制截断)                          │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
     │ chunks_list + tree_data
     ▼
  最终输出
```

### 语义切分原理

**目标**: 保持语义连贯性，避免在不相关位置切分

```
句子序列:  [S1] → [S2] → [S3] → [S4] → [S5] → [S6] → [S7]
相似度:      0.85   0.72   0.35   0.88   0.45   0.90
                                         ↓
                              相似度 < 0.5 = 语义边界
                                         ↓
切分结果:  [S1+S2+S3] | [S4] | [S5+S6+S7]
```

**关键参数**:
- `similarity_threshold = 0.5`: 相似度阈值，低于此值视为语义边界
- `buffer_sentences = 3`: 边界缓冲，防止过短 chunk

### 语义切分方法详解

#### `_split_sentences_semantic(text)`
拆分文本为句子，支持中英文：
- 中文边界：`。！？；\n`
- 英文边界：`.!?;` + 空格

#### `_compute_embeddings(sentences)`
批量计算句子嵌入：
- 使用 `unified_embedder.encode_texts()`
- 批量处理，batch_size=32

#### `_detect_semantic_boundaries(embeddings, threshold, buffer)`
检测语义边界：
1. 计算相邻句子余弦相似度
2. 找到低相似度位置 (< threshold)
3. 应用缓冲规则，避免连续边界

#### `_merge_sentences_to_segments(sentences, boundaries)`
合并句子为语义片段

### 递归切分兜底详解

**触发条件**: 语义片段长度 > `settings.CHUNK_SIZE_TEXT` (默认 800 字符)

**切分优先级**:

```
超长语义片段 (>800字符)
    │
    ▼ 尝试按段落边界切分 (\n\n)
┌─────────────────────────────────────┐
│  段落 1 (长度 OK) → 成为 chunk       │
│  段落 2 (超长) → 进入下一层          │
└─────────────────────────────────────┘
    │
    ▼ 尝试按句子边界切分 (。！？.!?)
┌─────────────────────────────────────┐
│  句子 1 (长度 OK) → 成为 chunk       │
│  句子 2 (超长) → 进入下一层          │
└─────────────────────────────────────┘
    │
    ▼ 强制截断（兜底）
┌─────────────────────────────────────┐
│  每 800 字符截断为一个 chunk         │
└─────────────────────────────────────┘
```

### 三阶段切分优势

| 策略 | 优点 | 适用场景 |
|:---|:---|:---|
| **标题切分** | 保持文档结构完整性 | 有明确标题层级的技术文档 |
| **语义切分** | 保持内容语义连贯性 | 论文、技术文档、长篇幅文章 |
| **递归切分** | 确保长度限制 | 所有超长文本的兜底方案 |

### 为什么采用三阶段策略？

1. **第一阶段（标题）**: 尊重文档原始结构，用户导航更清晰
2. **第二阶段（语义）**: 保持语义连贯，提升检索 Context Recall
3. **第三阶段（递归）**: 确保 chunk 不超过向量模型限制

最终实现：**最高质量的检索效果**

---

## 表格切分详解

### 表格来源

| 来源 | 处理方式 | 结果 |
|:---|:---|:---|
| **Converter 提供** | `[TABLE_idx]` 标记 | 从 `tables_data` 注入 |
| **Markdown `|` 语法** | 直接识别 | `_parse_markdown_table()` 解析 |

### 表格切片格式

```json
{
  "chunk_id": "doc_id_table_uuid",
  "chunk_type": "table",
  "content": "{\"type\":\"table\",\"headers\":[\"列1\",\"列2\"],\"rows\":[[\"a\",\"b\"]],\"html\":\"<table class='chunk-table'>...</table>\"}",
  "heading_level": 2,
  "heading_text": "一级标题 > 二级标题",
  "metadata": {
    "row_count": 3,
    "col_count": 2
  }
}
```

### 超长表格处理

**触发条件**: 表格 HTML 长度 > `settings.CHUNK_SIZE_TABLE` (默认 1600 字符)

**处理方法**: `_split_table(table_content)` - 按行切分，保持表头

```python
def _split_table(self, table_content: str) -> list[str]:
    lines = table_content.split("\n")

    # 表头 + 分隔行（保持完整）
    header = lines[0]
    separator = lines[1]
    header_block = f"{header}\n{separator}"

    # 数据行按长度贪心聚合
    data_lines = lines[2:]
    max_data_len = CHUNK_SIZE_TABLE - len(header_block)

    result: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    for line in data_lines:
        if current_len + len(line) > max_data_len and current_lines:
            # 输出当前 chunk
            result.append(header_block + "\n" + "\n".join(current_lines))
            current_lines = [line]
            current_len = len(line)
        else:
            current_lines.append(line)
            current_len += len(line)

    # 最后一个 chunk
    if current_lines:
        result.append(header_block + "\n" + "\n".join(current_lines))

    return result
```

### 表格 HTML 生成

**方法**: `build_table_html(headers, rows, merges, class_name)` (table_utils.py)

```python
def build_table_html(headers, rows, merges=None, class_name="chunk-table"):
    html_lines = [f'<table class="{class_name}">']

    # 表头
    html_lines.append("<thead><tr>")
    for header_text in headers:
        html_lines.append(f"<th>{escape(header_text)}</th>")
    html_lines.append("</tr></thead>")

    # 数据行
    html_lines.append("<tbody>")
    for row_data in rows:
        html_lines.append("<tr>")
        for cell_text in row_data:
            html_lines.append(f"<td>{escape(cell_text)}</td>")
        html_lines.append("</tr>")
    html_lines.append("</tbody></table>")

    return "\n".join(html_lines)
```

**合并单元格支持**:
```python
# merges = [{"row": 0, "col": 0, "rowspan": 2, "colspan": 1}]
# 生成：<th rowspan="2">...</th>
```

---

## 代码块切分详解

### 代码块识别

**正则**: `r"```(\w*)?\s*(.*?)\s*```"` (DOTALL 模式)

支持格式：
- ` ```python\ncode\n``` `
- ` ```\ncode\n``` `
- ` ```code``` ` (紧凑格式)

### 代码块处理原则

**不切分**：代码块保持完整，避免语法破坏

```python
# 代码块切片
{
    "chunk_id": "doc_id_code_uuid",
    "chunk_type": "code",
    "content": "def hello():\n    print('world')",  # 完整代码
    "metadata": {
        "language": "python",
        "is_code_block": True,
    }
}
```

### 超长代码块处理

如果代码块长度 > `CHUNK_SIZE_TEXT`：

1. **不切分**：保持完整性，可能影响检索精度
2. **未来优化**：可按函数/类定义切分（待实现）

---

## 切片类型汇总

| 切片类型 | chunk_type | content 格式 | 特殊处理 |
|:---|:---|:---|:---|
| **text** | `"text"` | 纯文本字符串 | 超长时递归切分 |
| **table** | `"table"` | JSON 结构化数据 | HTML 用于前端 |
| **code** | `"code"` | 原始代码字符串 | 保持完整 |
| **image** | `"image"` | JSON 路径信息 | VLM caption 后续填充 |

---

## 阶段 3.5: 切分质量评估（闭环）(22%)

### 质量评估流程

切分完成后自动评估质量，生成报告和优化建议：

```
切分完成
    │
    ▼ chunk_quality_evaluator.evaluate()
┌────────────────────────────────────────────────────────────┐
│  质量指标计算                                               │
│  - size_consistency: chunk 大小一致性                       │
│  - content_coverage: 内容覆盖率                             │
│  - heading_inheritance: 标题继承率                          │
│  - token_distribution: token 分布合理性                     │
│  - quality_score: 综合质量评分                              │
└────────────────────────────────────────────────────────────┘
    │
    ▼ 生成优化建议
┌────────────────────────────────────────────────────────────┐
│  自动诊断问题                                               │
│  - 超长 chunk → 建议减小 chunk_size                         │
│  - 过短 chunk → 建议增大 chunk_size                         │
│  - 低一致性 → 建议使用 semantic 策略                        │
│  - 低标题继承 → 建议使用 heading 策略                        │
└────────────────────────────────────────────────────────────┘
    │
    ▼ 保存质量报告
┌────────────────────────────────────────────────────────────┐
│  quality_report = {                                         │
│    "quality_score": 0.85,                                   │
│    "is_qualified": true,                                    │
│    "recommendations": ["切分质量达标"],                      │
│  }                                                          │
└────────────────────────────────────────────────────────────┘
    │
    ▼ 任务完成时返回质量报告
```

### 质量指标详解

| 指标 | 计算方式 | 目标值 | 说明 |
|:---|:---|:---|:---|
| **size_consistency** | `1 - variance / avg²` | ≥ 0.7 | 切片大小是否均匀 |
| **content_coverage** | `chunk_chars / original_chars` | ≥ 0.9 | 内容是否完整覆盖 |
| **heading_inheritance** | `chunks_with_heading / total` | ≥ 0.8 | 标题信息是否保留 |
| **token_distribution** | `1 - oversized_ratio * 0.5` | ≥ 0.7 | token 数是否合理 |
| **quality_score** | 加权综合 | ≥ 0.85 | 综合质量评分 |

### 综合质量评分公式

```python
quality_score = (
    token_distribution_score * 0.4 +  # token 分布最重要
    size_consistency * 0.2 +           # 大小一致性
    heading_inheritance * 0.2 +        # 标题继承
    content_coverage * 0.2             # 内容覆盖
)
```

### 质量报告示例

```json
{
  "summary": {
    "quality_score": 0.87,
    "is_qualified": true,
    "chunk_count": 45
  },
  "metrics": {
    "avg_chunk_size": 650,
    "avg_chunk_tokens": 180,
    "size_consistency": 0.82,
    "heading_inheritance": 0.91
  },
  "recommendations": [
    "切分质量达标，当前策略 smart 效果良好"
  ]
}
```

### 项目已有模块

项目已有完整的质量评估体系：

| 模块 | 文件 | 功能 |
|:---|:---|:---|
| **质量评估器** | `chunk_quality_evaluator.py` | 切分质量评估 |
| **质量闭环** | `quality_loop.py` | 自动迭代优化 |
| **策略选择器** | `auto_selector.py` | 自动策略推荐 |
| **Token 计数** | `token_utils.py` | Token 数计算 |

---

## 阶段 4: VLM Caption (30%)

### 图片描述生成

```python
# 阶段 3: VLM Caption
image_chunks = [c for c in chunks if c["chunk_type"] == "image"]

for c in image_chunks:
    image_path = c.get("metadata", {}).get("image_path", "")
    context = c.get("metadata", {}).get("context_text", "")
    heading = c.get("metadata", {}).get("heading_text", "")

    if image_path and len(context) >= 20:
        summary = llm_client.summarize_image(context, heading)
        vlm_caption = summary or f"图片: {img_name}"

    # 存储到 metadata.vlm_caption
    c["metadata"]["vlm_caption"] = vlm_caption
```

---

## 阶段 5: 嵌入 (50%)

### Unified Embedder

使用 DashScope `qwen3-vl-embedding` 模型，输出维度 2560。

```python
# 文本类切片（text/table/code）
text_embeddable = [c for c in chunks if c["chunk_type"] in ("text", "table", "code")]
texts = [c["content"] for c in text_embeddable]
dense_vectors = unified_embedder.encode_texts(texts)

# 图片切片（使用 vlm_caption + image_path）
for c in image_embeddable:
    image_path = c.get("metadata", {}).get("image_path", "")
    vlm_caption = c.get("metadata", {}).get("vlm_caption", "")

    if image_path:
        vec = unified_embedder.encode_multimodal(vlm_caption, image_path)
    else:
        vec = unified_embedder.encode_query(vlm_caption)

    vector_map[c["chunk_id"]] = vec
```

---

## 阶段 6: Milvus 存储 (70%)

### Milvus Entity 构建

```python
for c in chunks:
    milvus_content = c["content"]

    # 图片：content 用 vlm_caption
    if c["chunk_type"] == "image":
        vlm_caption = c.get("metadata", {}).get("vlm_caption", "")
        if vlm_caption:
            milvus_content = vlm_caption

    # 表格：content 用 HTML（便于 BM25 检索）
    if c["chunk_type"] in ("table", "nested_table"):
        table_data = json.loads(c["content"])
        if table_data.get("html"):
            milvus_content = table_data["html"]

    entity = {
        "chunk_id": cid,
        "vector": dense_vec,
        "content": milvus_content,
        "metadata": metadata,
    }
```

---

## 阶段 7: MySQL 存储 (85%)

### Chunk 存储

```python
mysql_ok = _retry_with_backoff(
    lambda: chunk_repo.insert_chunks(chunks),
    max_retries=5,
)
```

---

## 重试机制

### Redis 中间状态缓存

```python
# parse/chunk/embed 完成后缓存
_cache_pipeline_state(task_id, {"chunks": chunks, "entities": entities})

# 重试时恢复
cached = _restore_pipeline_state(task_id)
if cached:
    chunks = cached["chunks"]
    entities = cached["entities"]
```

### 独立重试

```python
# Milvus 独立重试（失败不影响 MySQL）
milvus_ok = _retry_with_backoff(
    lambda: milvus_repo.batch_upsert(entities),
    max_retries=5,
)

# MySQL 独立重试
mysql_ok = _retry_with_backoff(
    lambda: chunk_repo.insert_chunks(chunks),
    max_retries=5,
)
```

---

## 支持的文件类型

| 文件类型 | doc_type | Converter | 分块策略 |
|:---|:---|:---|:---|
| **PDF** | `pdf` | `PdfConverter` | markdown_chunker（标题切分） |
| **DOCX** | `docx` | `DocxConverter` | markdown_chunker（标题切分） |
| **PPTX** | `pptx` | `PptxConverter` | markdown_chunker（标题切分） |
| **Markdown** | `md` | `MdConverter` | markdown_chunker（标题切分） |
| **Excel** | `xlsx/xls` | `XlsxConverter` | markdown_chunker（直接表格切片） |

---

## 统一入口架构

所有文档类型都走统一的处理流程：

```
文件上传
    │
    ▼ document_service.upload() [统一上传入口]
    │
    ▼ enqueue_parse() [统一任务提交]
    │
    ▼ parse_and_chunk_task [Celery 统一任务入口]
    │
    ▼ get_converter(doc_type) [统一 Converter 获取]
    │
    ▼ Converter.convert_full() [统一解析入口]
    │   → ConverterOutput (markdown, tables, images, code_blocks)
    │
    ▼ markdown_chunker.chunk() [统一分块入口]
    │   │
    │   ▼ _clean_markdown() [统一清洗入口]
    │   │   - 基础清洗：控制字符、多余空白
    │   │   - 噪音过滤：页码、页眉页脚、水印关键词
    │   │   - 垃圾模式：正则匹配过滤
    │   │
    │   ├── 纯表格文档 (xlsx/xls)
    │   │   └── _make_pure_table_chunks() 直接生成切片
    │   │
    │   └── 其他文档 (pdf/docx/pptx/md)
    │       └── 按标题切分 + 内容片段识别
    │
    ▼ VLM Caption → Embedding → Milvus → MySQL
```

---

## 统一清洗流程

所有文档在分块前都会经过统一清洗：

### 清洗步骤

```
Markdown 文本
    │
    ▼ 1. 基础清洗 (clean_text)
┌─────────────────────────────────────┐
│  - 去除控制字符                     │
│  - 去除多余空白/制表符              │
│  - 去除多余换行                     │
└─────────────────────────────────────┘
    │
    ▼ 2. 按行噪音过滤 (is_noise_text)
┌─────────────────────────────────────┐
│  过滤内容：                         │
│  - 纯数字行（页码）                 │
│  - "第 X 页"、"Page X"              │
│  - 分隔线（---、===、***）          │
│  - 页脚关键词（机密、版权、内部）   │
│  - PDF 损坏标记 (cid:123)           │
│                                     │
│  保留内容：                         │
│  - Markdown 标题（# 开头）          │
│  - Markdown 表格（| 开头）          │
│  - 代码块标记（``` 开头）           │
└─────────────────────────────────────┘
    │
    ▼ 3. 多余空行处理
┌─────────────────────────────────────┐
│  连续超过3个空行 → 变为2个空行     │
└─────────────────────────────────────┘
    │
    ▼ 清洗后的 Markdown
```

---

## 数据流总览

```
文件上传
    │ file_path + doc_id + doc_type
    ▼
Converter.convert_full()
    │ ConverterOutput:
    │   - markdown (含 [TABLE_idx])
    │   - tables_data [{index, headers, rows, html}]
    │   - images_data [{path}]
    │   - code_blocks_data [{content, language}]
    ▼
MarkdownChunker.chunk()
    │ 三层切分:
    │   1. 按标题切分 → sections
    │   2. 片段识别 → fragments
    │   3. 切片生成 → chunks
    │   + 超长文本递归切分
    │   + 表格保持/按行切分
    │   + 代码块保持完整
    ▼
chunks_list + tree_data
    │ chunk_type: text/table/code/image
    ▼
VLM Caption (图片)
    │ metadata.vlm_caption
    ▼
Unified Embedder
    │ vector_map {chunk_id: vector}
    ▼
Milvus + MySQL 存储
    │ 完成
    ▼
Document 记录创建
```