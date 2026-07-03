"""统一 Markdown 分块器。

直接从 Markdown 的 # 标记识别标题层级，按标题切分内容，
然后构建 Paragraph + Chunks 双层树状结构。

核心流程：
1. 解析 Markdown 文本，识别 # ## ### 标题
2. 按标题切分内容（标题及其下的正文）
3. **识别代码块，保持代码完整性（不切断）**
4. **识别表格标记，注入结构化表格数据**
5. 对超长内容进行二次切分（跳过代码块）
6. 构建 Paragraph + Chunks 树

输出切片格式：
- text: 普通文本
- table: 表格（包含 table_data 结构化数据）
- image: 图片（包含 image_url）
- code: 代码块（包含 language）
"""
import json
import re
import uuid
from typing import Any

import numpy as np
import structlog

from app.config import settings
from app.parsers.noise_filter import clean_text, is_noise_text, should_filter_text

logger = structlog.get_logger()

# 段落边界正则
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
# 句子边界正则
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？.!?])\s*(?=\S)")
# Markdown 表格正则：匹配 | 开头的行或代码块中的表格
_TABLE_BLOCK_RE = re.compile(
    r"(```\s*\n)?([|\s]*\n)*"
    r"(\|[^\n]+\|\n)"  # 表头行
    r"(\|[:\-]+\|\n)"  # 分隔行
    r"(\|[^\n]+\|\n)*"  # 数据行
    r"(```)?",
    re.MULTILINE
)
# 代码块正则：匹配 ``` 包围的内容（宽松匹配，支持多行代码）
# 支持: ```python\ncode\n``` 或 ```\ncode\n``` 或 ```code```
_CODE_BLOCK_RE = re.compile(r"```(\w*)?\s*(.*?)\s*```", re.DOTALL)
# 表格标记正则：匹配 [TABLE_idx]
_TABLE_MARKER_RE = re.compile(r"\[TABLE_(\d+)\]")


class MarkdownChunker:
    """统一 Markdown 分块器。

    支持多种切分策略（通过 chunk_strategy 参数选择）：
    - smart: 智能切分（标题 + 语义 + 递归）- 最高质量，默认
    - heading: 只按标题切分 - 保持文档结构
    - semantic: 只按语义切分 - 保持语义连贯
    - recursive: 只按递归切分 - 最简单快速
    - fixed: 固定长度切分 - 最基础方案
    """

    def chunk(
        self,
        markdown_text: str,
        doc_id: str,
        doc_type: str,
        image_paths: list[str] = None,
        creator: str = "",
        kb_id: str = "",
        doc_name: str = "",
        tables_data: list[dict] = None,
        code_blocks_data: list[dict] = None,
        images_data: list[dict] = None,
        chunk_strategy: str = "smart",
    ) -> tuple[list[dict], dict]:
        """
        将 Markdown 文本分块并构建树状结构。

        支持结构化数据注入：
        - tables_data: 表格结构化数据（从 DocxConverter 获取）
        - code_blocks_data: 代码块数据
        - images_data: 图片结构化数据 [{path, index}]
        - chunk_strategy: 切分策略选择

        Args:
            markdown_text: Markdown 格式的文本
            doc_id: 文档 ID
            doc_type: 文档类型
            image_paths: 图片路径列表（兼容旧接口）
            creator: 创建者
            kb_id: 知识库 ID
            doc_name: 文档名称
            tables_data: 表格结构化数据 [{index, headers, rows, merges, html}]
            code_blocks_data: 代码块数据 [{content, language}]
            images_data: 图片结构化数据 [{path, index}]
            chunk_strategy: 切分策略（smart/heading/semantic/recursive/fixed）

        Returns:
            (chunks_list, tree_data): 切片列表和树状结构数据
        """
        image_paths = image_paths or []
        tables_data = tables_data or []
        code_blocks_data = code_blocks_data or []
        images_data = images_data or []

        # 验证策略参数
        valid_strategies = {"smart", "heading", "semantic", "recursive", "fixed"}
        if chunk_strategy not in valid_strategies:
            logger.warning(
                "invalid_chunk_strategy",
                strategy=chunk_strategy,
                fallback="smart",
            )
            chunk_strategy = "smart"

        logger.info(
            "chunk_strategy_selected",
            doc_id=doc_id,
            strategy=chunk_strategy,
        )

        # 如果提供了 images_data，优先使用它
        if images_data and not image_paths:
            image_paths = [img["path"] for img in images_data]

        # ========== 统一清洗入口 ==========
        # 对 Markdown 文本进行噪音清洗（页码、页眉页脚等）
        if markdown_text:
            markdown_text = self._clean_markdown(markdown_text, doc_type)

        # ========== 纯表格文档处理（Excel 等）==========
        # Excel 等纯表格文档直接从 tables_data 生成切片，无需 Markdown 分块
        pure_table_types = {"xlsx", "xls"}
        if doc_type in pure_table_types and tables_data:
            return self._make_pure_table_chunks(
                tables_data, doc_id, doc_type, creator, kb_id, doc_name
            )

        # ========== 根据策略选择切分流程 ==========
        # heading 策略：跳过标题切分，直接处理整个文本
        if chunk_strategy == "heading":
            # heading 策略：只按标题切分，不做二次切分
            return self._chunk_with_heading_only(
                markdown_text, doc_id, doc_type, image_paths,
                creator, kb_id, doc_name, tables_data, code_blocks_data, images_data
            )

        # semantic 策略：不按标题切分，直接对全文语义切分
        if chunk_strategy == "semantic":
            return self._chunk_with_semantic_only(
                markdown_text, doc_id, doc_type, image_paths,
                creator, kb_id, doc_name, tables_data, code_blocks_data, images_data
            )

        # recursive 策略：不按标题切分，直接递归切分
        if chunk_strategy == "recursive":
            return self._chunk_with_recursive_only(
                markdown_text, doc_id, doc_type, image_paths,
                creator, kb_id, doc_name, tables_data, code_blocks_data, images_data
            )

        # fixed 策略：固定长度切分
        if chunk_strategy == "fixed":
            return self._chunk_with_fixed_length(
                markdown_text, doc_id, doc_type, image_paths,
                creator, kb_id, doc_name, tables_data, code_blocks_data, images_data
            )

        # smart 策略（默认）：标题 + 语义 + 递归
        # ========== 普通文档处理（PDF/DOCX/PPTX/MD）==========
        # 构建表格索引映射
        tables_map = {t["index"]: t for t in tables_data}

        # 1. 按标题切分 Markdown 内容
        sections = self._split_markdown_by_headers(markdown_text)

        logger.info(
            "markdown_sections",
            doc_id=doc_id,
            section_count=len(sections),
        )

        # 2. 构建切片列表和树状结构
        chunks_list: list[dict] = []
        sort_order = 0

        # 树构建状态
        paragraphs: list[dict] = []  # 根节点列表
        stack: list[tuple[int, dict]] = []  # (level, paragraph_node)
        last_chunk_id: str | None = None

        for section in sections:
            heading_path = section["heading_path"]
            heading_levels = section["heading_levels"]
            level = section["level"]
            content = section["content"]

            if not content or len(content.strip()) < 10:
                continue

            # 从 heading_path 构建树状结构（确保父节点存在）
            paragraph_id = ""

            for i, path_title in enumerate(heading_path):
                path_level = heading_levels[i]

                # 弹出栈中 >= 当前 level 的节点
                while stack and stack[-1][0] >= path_level:
                    stack.pop()

                # 查找已存在的节点
                # 如果是第一个元素（根级），从 paragraphs 根列表查找
                # 否则从栈顶节点的 childContentParagraph 中查找
                existing_child = None
                if i == 0:
                    # 从根列表查找
                    for p in paragraphs:
                        if p["title"] == path_title and p["level"] == path_level:
                            existing_child = p
                            break
                else:
                    # 从栈顶节点的子节点查找
                    if stack:
                        parent_node = stack[-1][1]
                        for child in parent_node["childContentParagraph"]:
                            if child["title"] == path_title and child["level"] == path_level:
                                existing_child = child
                                break

                if existing_child:
                    # 使用已存在的节点
                    paragraph_node = existing_child
                else:
                    # 创建新节点
                    paragraph_node = {
                        "id": f"p-{path_level}-{uuid.uuid4().hex[:6]}",
                        "title": path_title,
                        "level": path_level,
                        "content": path_title,
                        "childContentParagraph": [],
                        "chunks": [],
                    }
                    if i == 0:
                        # 添加到根列表
                        paragraphs.append(paragraph_node)
                    elif stack:
                        # 添加到栈顶节点的子节点
                        stack[-1][1]["childContentParagraph"].append(paragraph_node)

                stack.append((path_level, paragraph_node))
                paragraph_id = paragraph_node["id"]

            # 构建完整路径文本（用于检索定位）
            full_heading_path = " > ".join(heading_path)

            # 对内容进行切分处理
            content_fragments = self._split_by_markers_and_code(content, tables_map)

            for frag_idx, fragment in enumerate(content_fragments):
                if fragment["type"] == "table_marker":
                    # 表格标记：创建表格切片
                    table_idx = fragment["table_idx"]
                    table_data = tables_map.get(table_idx)
                    if table_data:
                        chunk_id = f"{doc_id}_table_{uuid.uuid4().hex[:8]}_{sort_order}"
                        chunk_item = self._make_table_chunk(
                            chunk_id, doc_id, paragraph_id, table_data,
                            level, full_heading_path, sort_order, creator, kb_id, doc_type,
                            last_chunk_id, doc_name,
                        )
                        chunks_list.append(chunk_item)
                        # 添加到栈顶 paragraph
                        if stack:
                            stack[-1][1]["chunks"].append(self._chunk_to_tree_item(chunk_item, paragraph_id))
                        last_chunk_id = chunk_id
                        sort_order += 1
                elif fragment["type"] == "markdown_table":
                    # Markdown 表格：直接创建表格切片
                    table_content = fragment["content"]
                    chunk_id = f"{doc_id}_table_{uuid.uuid4().hex[:8]}_{sort_order}"
                    # 将 Markdown 表格转换为结构化数据
                    table_data = self._parse_markdown_table(table_content)
                    chunk_item = self._make_table_chunk(
                        chunk_id, doc_id, paragraph_id, table_data,
                        level, full_heading_path, sort_order, creator, kb_id, doc_type,
                        last_chunk_id, doc_name,
                    )
                    chunks_list.append(chunk_item)
                    if stack:
                        stack[-1][1]["chunks"].append(self._chunk_to_tree_item(chunk_item, paragraph_id))
                    last_chunk_id = chunk_id
                    sort_order += 1
                elif fragment["type"] == "code_block":
                    # 代码块：保持完整，不切分
                    chunk_id = f"{doc_id}_code_{uuid.uuid4().hex[:8]}_{sort_order}"
                    chunk_item = self._make_code_chunk(
                        chunk_id, doc_id, paragraph_id, fragment["content"],
                        fragment["language"], level, full_heading_path, sort_order, creator,
                        kb_id, doc_type, last_chunk_id, doc_name,
                    )
                    chunks_list.append(chunk_item)
                    if stack:
                        stack[-1][1]["chunks"].append(self._chunk_to_tree_item(chunk_item, paragraph_id))
                    last_chunk_id = chunk_id
                    sort_order += 1
                else:
                    # 普通文本：先语义切分，再长度检查
                    # 第一优先级：语义切分（保持语义连贯）
                    # 第二优先级：递归切分（长度控制）
                    sub_chunks = self._semantic_then_recursive_split(
                        fragment["content"],
                        doc_id,
                        settings.CHUNK_SIZE_TEXT,
                        doc_type,
                    )
                    for sub_idx, sub_content in enumerate(sub_chunks):
                        if not sub_content.strip() or len(sub_content.strip()) < 10:
                            continue
                        chunk_id = f"{doc_id}_text_{uuid.uuid4().hex[:8]}_{sort_order}"
                        chunk_item = self._make_chunk(
                            chunk_id, doc_id, paragraph_id, sub_content.strip(),
                            level, full_heading_path, sort_order, creator, kb_id, doc_type,
                            last_chunk_id, doc_name,
                        )
                        chunks_list.append(chunk_item)
                        if stack:
                            stack[-1][1]["chunks"].append(self._chunk_to_tree_item(chunk_item, paragraph_id))
                        last_chunk_id = chunk_id
                        sort_order += 1

        # 3. 处理图片
        # 图片 chunk：
        # - MySQL content 存储 JSON 结构化数据（便于前端渲染）
        # - Milvus content 存储 VLM caption（便于 BM25 检索）
        for img_idx, img_path in enumerate(image_paths):
            img_chunk_id = f"{doc_id}_img_{uuid.uuid4().hex[:8]}"
            # 构建结构化图片数据
            img_data = {
                "type": "image",
                "path": img_path,
                "index": img_idx,
            }
            # 如果有 images_data，合并更多信息
            if images_data and img_idx < len(images_data):
                img_data.update(images_data[img_idx])

            img_chunk = {
                "chunk_id": img_chunk_id,
                "doc_id": doc_id,
                "chunk_type": "image",
                "content": json.dumps(img_data, ensure_ascii=False),  # MySQL 存 JSON
                "caption": None,
                "page_num": 0,
                "heading_level": 0,
                "heading_text": "",
                "sort_order": sort_order,
                "metadata": {  # 使用 metadata（与数据库字段名一致）
                    "doc_type": doc_type,
                    "vector_status": "pending",
                    "image_path": img_path,
                    "vlm_caption": "",  # 后续 VLM 填充（Milvus 用）
                    "context_text": "",  # 后续填充
                },
                "creator": creator,
                "kb_id": kb_id,
                "doc_name": doc_name,
            }
            chunks_list.append(img_chunk)
            sort_order += 1

        # 4. 构建树数据
        tree_data = {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "paragraphs": paragraphs,
        }

        return chunks_list, tree_data

    def _split_markdown_by_headers(self, markdown_text: str) -> list[dict]:
        """递归按 Markdown 标题切分内容，每个 chunk 包含完整标题路径。

        核心逻辑：
        1. 维护标题栈，遇到标题时更新栈
        2. 遇到正文时，当前栈就是完整的父级标题路径
        3. 内容归属到完整路径下，不再丢失中间层级信息

        Returns:
            [
                {
                    'heading_path': ['一级标题', '二级标题', '三级标题'],
                    'level': 3,
                    'content': '实际内容...'
                },
                ...
            ]
        """
        lines = markdown_text.split("\n")
        sections: list[dict] = []

        # 标题栈：存储当前层级路径 [(level, title), ...]
        heading_stack: list[tuple[int, str]] = []
        # 当前正在收集的内容
        current_content: list[str] = []

        # 代码块状态跟踪
        in_code_block = False

        def _save_section():
            """保存当前收集的内容为一个 section。"""
            if current_content:
                content_str = "\n".join(current_content).strip()
                if content_str:
                    # 构建完整标题路径和级别列表
                    heading_path = [h[1] for h in heading_stack]
                    heading_levels = [h[0] for h in heading_stack]
                    current_level = heading_stack[-1][0] if heading_stack else 0
                    sections.append({
                        "heading_path": heading_path,
                        "heading_levels": heading_levels,
                        "level": current_level,
                        "content": content_str,
                    })
                current_content.clear()

        for line in lines:
            # 检测代码块边界
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                current_content.append(line)
                continue

            # 如果在代码块内，不检查标题（避免代码内的 # 被误识别）
            if in_code_block:
                current_content.append(line)
                continue

            # 检查是否是标题行
            header_match = re.match(r"^#{1,6}\s+(.+)", line)

            if header_match:
                # 先保存当前收集的内容
                _save_section()

                # 解析新标题
                new_level = len(line.split()[0])  # # 的数量
                new_title = header_match.group(1).strip()

                # 更新标题栈：
                # 弹出栈中 >= 新级别的标题（同级或更深层级）
                while heading_stack and heading_stack[-1][0] >= new_level:
                    heading_stack.pop()

                # 新标题入栈
                heading_stack.append((new_level, new_title))
            else:
                # 正文行
                if line.strip():
                    current_content.append(line)

        # 保存最后一个 section
        _save_section()

        return sections

    def _split_by_markers_and_code(
        self,
        content: str,
        tables_map: dict[int, dict],
    ) -> list[dict]:
        """
        按表格标记、Markdown 表格和代码块分割内容。

        流程步骤：
        1. 扫描内容，识别 [TABLE_idx] 标记（结构化表格注入）
        2. 扫描内容，识别 Markdown 表格（| 语法）
        3. 扫描内容，识别 ``` 包围的代码块
        4. 将内容分割为有序片段列表：
           - table_marker: 表格标记片段
           - markdown_table: Markdown 表格片段
           - code_block: 代码块片段
           - text: 普通文本片段

        Args:
            content: 待分割的内容
            tables_map: 表格索引映射 {idx: table_data}

        Returns:
            [
                {"type": "text", "content": "..."},
                {"type": "table_marker", "table_idx": 0},
                {"type": "markdown_table", "content": "| a | b |\\n|---|---|\\n| 1 | 2 |"},
                {"type": "code_block", "content": "...", "language": "python"},
                ...
            ]
        """
        fragments: list[dict] = []

        # 步骤1：先找出所有表格标记的位置
        table_markers = []
        for match in _TABLE_MARKER_RE.finditer(content):
            table_markers.append({
                "start": match.start(),
                "end": match.end(),
                "table_idx": int(match.group(1)),
            })

        # 步骤2：找出所有代码块的位置
        code_blocks = []
        for match in _CODE_BLOCK_RE.finditer(content):
            code_blocks.append({
                "start": match.start(),
                "end": match.end(),
                "content": match.group(2),
                "language": match.group(1) or "",
            })

        # 步骤3：找出所有 Markdown 表格的位置（排除代码块内的内容）
        # 先找出代码块的范围
        code_block_ranges = [(block["start"], block["end"]) for block in code_blocks]
        markdown_tables = self._find_markdown_tables(content, exclude_ranges=code_block_ranges)

        # 步骤4：合并所有特殊区域的边界点
        boundaries = set([0, len(content)])
        for marker in table_markers:
            boundaries.add(marker["start"])
            boundaries.add(marker["end"])
        for block in code_blocks:
            boundaries.add(block["start"])
            boundaries.add(block["end"])
        for table in markdown_tables:
            boundaries.add(table["start"])
            boundaries.add(table["end"])

        # 步骤5：按边界点分割内容
        sorted_boundaries = sorted(boundaries)
        for i in range(len(sorted_boundaries) - 1):
            start = sorted_boundaries[i]
            end = sorted_boundaries[i + 1]
            segment = content[start:end]

            if not segment.strip():
                continue

            # 判断片段类型
            is_table_marker = False
            for marker in table_markers:
                if start == marker["start"] and end == marker["end"]:
                    fragments.append({
                        "type": "table_marker",
                        "table_idx": marker["table_idx"],
                    })
                    is_table_marker = True
                    break

            if is_table_marker:
                continue

            is_code_block = False
            for block in code_blocks:
                if start == block["start"] and end == block["end"]:
                    fragments.append({
                        "type": "code_block",
                        "content": block["content"],
                        "language": block["language"],
                    })
                    is_code_block = True
                    break

            if is_code_block:
                continue

            # Markdown 表格
            is_markdown_table = False
            for table in markdown_tables:
                if start == table["start"] and end == table["end"]:
                    fragments.append({
                        "type": "markdown_table",
                        "content": table["content"],
                    })
                    is_markdown_table = True
                    break

            if is_markdown_table:
                continue

            # 普通文本
            if segment.strip():
                fragments.append({
                    "type": "text",
                    "content": segment.strip(),
                })

        return fragments

    def _find_markdown_tables(self, content: str, exclude_ranges: list[tuple[int, int]] = None) -> list[dict]:
        """找出所有 Markdown 表格的位置（排除指定范围）。

        Markdown 表格特征：
        - 以 | 开头和结尾的行组成
        - 至少有表头行和分隔行（包含 | 和 -）
        - 多个连续的表格行组成一个完整表格

        Args:
            content: 文本内容
            exclude_ranges: 要排除的范围列表 [(start, end), ...]，例如代码块范围

        Returns:
            [{"start": 0, "end": 100, "content": "|...|"}, ...]
        """
        exclude_ranges = exclude_ranges or []
        lines = content.split("\n")
        tables: list[dict] = []

        current_table_start: int | None = None
        current_table_lines: list[str] = []

        for i, line in enumerate(lines):
            # 计算行的起始位置
            line_start = sum(len(lines[j]) + 1 for j in range(i))  # +1 for \n
            line_end = line_start + len(line)

            # 检查是否在排除范围内（如代码块内）
            in_excluded = False
            for ex_start, ex_end in exclude_ranges:
                if line_start >= ex_start and line_end <= ex_end:
                    in_excluded = True
                    break

            if in_excluded:
                # 在排除范围内，如果有正在收集的表格，先保存
                if current_table_lines and self._is_valid_table_lines(current_table_lines):
                    table_content = "\n".join(current_table_lines)
                    table_end = current_table_start + len(table_content)
                    tables.append({
                        "start": current_table_start,
                        "end": table_end,
                        "content": table_content,
                    })
                current_table_start = None
                current_table_lines = []
                continue

            is_table_line = self._is_table_line(line)

            if is_table_line:
                if current_table_start is None:
                    current_table_start = line_start
                current_table_lines.append(line)
            else:
                # 非表格行：如果之前有表格，检查是否有效并保存
                if current_table_lines and self._is_valid_table_lines(current_table_lines):
                    table_content = "\n".join(current_table_lines)
                    table_end = current_table_start + len(table_content)
                    tables.append({
                        "start": current_table_start,
                        "end": table_end,
                        "content": table_content,
                    })
                current_table_start = None
                current_table_lines = []

        # 处理文档末尾的表格
        if current_table_lines and self._is_valid_table_lines(current_table_lines):
            table_content = "\n".join(current_table_lines)
            table_end = current_table_start + len(table_content)
            tables.append({
                "start": current_table_start,
                "end": table_end,
                "content": table_content,
            })

        return tables

    def _is_valid_table_lines(self, lines: list[str]) -> bool:
        """判断行列表是否构成有效的 Markdown 表格。

        有效表格条件：
        - 至少 2 行（表头 + 分隔行）
        - 第一行以 | 开头
        - 第二行包含 | 和 -（分隔行）
        """
        if len(lines) < 2:
            return False

        # 检查表头行
        if not lines[0].strip().startswith("|"):
            return False

        # 检查分隔行
        separator = lines[1].strip()
        if not all(c in "|-: " for c in separator):
            return False
        if separator.count("|") < 2:
            return False

        return True

    def _parse_markdown_table(self, table_content: str) -> dict:
        """将 Markdown 表格转换为结构化数据。

        输入格式：
        | 列1 | 列2 | 列3 |
        |-----|-----|-----|
        | a   | b   | c   |
        | d   | e   | f   |

        输出格式（与 _make_table_chunk 兼容）：
        {
            "headers": ["列1", "列2", "列3"],
            "rows": [["a", "b", "c"], ["d", "e", "f"]],
            "merges": [],
            "html": "<table>...</table>",
            "row_count": 3,
            "col_count": 3,
        }
        """
        lines = [l.strip() for l in table_content.split("\n") if l.strip()]

        if len(lines) < 2:
            return {"headers": [], "rows": [], "merges": [], "html": "", "row_count": 0, "col_count": 0}

        # 解析表头行
        header_line = lines[0]
        headers = [cell.strip() for cell in header_line.split("|") if cell.strip()]

        # 跳过分隔行（第2行）
        data_lines = lines[2:]

        # 解析数据行
        rows = []
        for line in data_lines:
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            if cells:
                rows.append(cells)

        # 构建 HTML 表格（用于渲染和搜索）
        html_parts = ["<table class='markdown-table'>"]
        html_parts.append("<thead><tr>")
        for h in headers:
            html_parts.append(f"<th>{h}</th>")
        html_parts.append("</tr></thead>")
        html_parts.append("<tbody>")
        for row in rows:
            html_parts.append("<tr>")
            for cell in row:
                html_parts.append(f"<td>{cell}</td>")
            html_parts.append("</tr>")
        html_parts.append("</tbody></table>")
        html = "".join(html_parts)

        return {
            "headers": headers,
            "rows": rows,
            "merges": [],  # Markdown 表格不支持单元格合并
            "html": html,
            "row_count": len(rows) + 1,  # 包括表头
            "col_count": len(headers),
        }

    def _make_table_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        paragraph_id: str,
        table_data: dict,
        heading_level: int,
        heading_text: str,
        sort_order: int,
        creator: str,
        kb_id: str,
        doc_type: str,
        previous_id: str | None,
        doc_name: str = "",
    ) -> dict:
        """
        创建表格切片。

        流程步骤：
        1. 从 table_data 提取结构化信息（headers, rows, merges）
        2. 使用 html 字段作为 content（用于搜索）
        3. 构建 metadata 包含 table_data 完整结构
        4. 返回切片字典

        Args:
            chunk_id: 切片 ID
            doc_id: 文档 ID
            paragraph_id: 段落 ID
            table_data: 表格结构化数据 {index, headers, rows, merges, html}
            heading_level: 标题层级
            heading_text: 标题文本
            sort_order: 排序号
            creator: 创建者
            kb_id: 知识库 ID
            doc_type: 文档类型
            previous_id: 前一个切片 ID
            doc_name: 文档名称

        Returns:
            表格切片字典
        """
        # 步骤1-2：提取结构化数据
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])
        merges = table_data.get("merges", [])
        html = table_data.get("html", "")
        row_count = table_data.get("row_count", len(rows) + 1)
        col_count = table_data.get("col_count", len(headers))

        # content 存储完整结构化数据（JSON 格式）
        # 前端直接解析 content 获取表格数据
        content_data = {
            "type": "table",
            "headers": headers,
            "rows": rows,
            "merges": merges,
            "row_count": row_count,
            "col_count": col_count,
            "html": html,  # HTML 格式用于渲染和搜索展示
        }

        # 步骤3：构建 metadata（精简版，用于过滤）
        metadata = {
            "doc_type": doc_type,
            "vector_status": "pending",
            "paragraph_id": paragraph_id,
            "previous_id": previous_id,
            "row_count": row_count,
            "col_count": col_count,
        }

        # 步骤4：返回切片
        return {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "chunk_type": "table",
            "content": json.dumps(content_data, ensure_ascii=False),  # JSON 格式存储
            "caption": None,
            "page_num": 0,
            "heading_level": heading_level,
            "heading_text": heading_text,
            "sort_order": sort_order,
            "metadata": metadata,
            "creator": creator,
            "kb_id": kb_id,
            "doc_name": doc_name,
        }

    def _make_code_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        paragraph_id: str,
        code_content: str,
        language: str,
        heading_level: int,
        heading_text: str,
        sort_order: int,
        creator: str,
        kb_id: str,
        doc_type: str,
        previous_id: str | None,
        doc_name: str = "",
    ) -> dict:
        """
        创建代码块切片。

        流程步骤：
        1. 使用完整代码内容作为 content
        2. 构建 metadata 包含 language 字段
        3. 返回切片字典

        Args:
            chunk_id: 切片 ID
            doc_id: 文档 ID
            paragraph_id: 段落 ID
            code_content: 代码内容
            language: 代码语言
            heading_level: 标题层级
            heading_text: 标题文本
            sort_order: 排序号
            creator: 创建者
            kb_id: 知识库 ID
            doc_type: 文档类型
            previous_id: 前一个切片 ID
            doc_name: 文档名称

        Returns:
            代码块切片字典
        """
        # 步骤2：构建 metadata
        metadata = {
            "doc_type": doc_type,
            "vector_status": "pending",
            "paragraph_id": paragraph_id,
            "previous_id": previous_id,
            "language": language,
            "is_code_block": True,
        }

        # 步骤3：返回切片
        return {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "chunk_type": "code",
            "content": code_content,
            "caption": None,
            "page_num": 0,
            "heading_level": heading_level,
            "heading_text": heading_text,
            "sort_order": sort_order,
            "metadata": metadata,
            "creator": creator,
            "kb_id": kb_id,
            "doc_name": doc_name,
        }

    def _make_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        paragraph_id: str,
        content: str,
        heading_level: int,
        heading_text: str,
        sort_order: int,
        creator: str,
        kb_id: str,
        doc_type: str,
        previous_id: str | None,
        doc_name: str = "",
    ) -> dict:
        """创建切片字典。"""
        return {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "chunk_type": "text",
            "content": content,
            "caption": None,
            "page_num": 0,
            "heading_level": heading_level,
            "heading_text": heading_text,
            "sort_order": sort_order,
            "metadata": {  # 使用 metadata（与数据库字段名一致）
                "doc_type": doc_type,
                "vector_status": "pending",
                "paragraph_id": paragraph_id,
                "previous_id": previous_id,
            },
            "creator": creator,
            "kb_id": kb_id,
            "doc_name": doc_name,
        }

    def _chunk_to_tree_item(self, chunk: dict, paragraph_id: str) -> dict:
        """将切片转换为树状结构中的 chunk item。"""
        return {
            "id": chunk["chunk_id"],
            "paragraphId": paragraph_id,
            "content": chunk["content"],
            "status": 1,
            "auditStatus": 0,
            "previousId": chunk["metadata"].get("previous_id"),  # 使用 metadata
            "matchCount": 0,
        }

    def _split_long_content(
        self,
        content: str,
        doc_id: str,
        sort_order: int,
    ) -> list[str]:
        """对超长内容进行二次切分。

        **特殊处理 Markdown 表格**：
        - 识别 | 开头的行组成的表格结构
        - 表格不按段落切分，保持完整性
        - 超长表格由 TableChunker 处理

        普通文本：
        - 按段落边界切分，如果单个段落仍超长，按句子边界切分
        - 确保每个切片不超过 CHUNK_SIZE_TEXT (800字符)

        Args:
            content: 待切分的内容
            doc_id: 文档 ID（用于日志）
            sort_order: 当前排序号（用于日志）

        Returns:
            切分后的内容列表
        """
        max_size = settings.CHUNK_SIZE_TEXT

        # 如果内容不超长，直接返回
        if len(content) <= max_size:
            return [content]

        logger.debug(
            "splitting_long_content",
            doc_id=doc_id,
            sort_order=sort_order,
            content_len=len(content),
            max_size=max_size,
        )

        # 先识别并提取表格，避免切分时破坏表格结构
        tables_and_text = self._extract_tables_from_content(content)

        # 如果没有表格，使用原来的切分逻辑
        if not tables_and_text["tables"]:
            return self._split_text_only(content, max_size, doc_id)

        # 有表格：混合处理
        result: list[str] = []
        for item in tables_and_text["ordered_items"]:
            if item["type"] == "table":
                # 表格：保持完整，或由 TableChunker 处理
                table_content = item["content"]
                if len(table_content) <= settings.CHUNK_SIZE_TABLE:
                    # 小表格：直接添加
                    result.append(table_content)
                else:
                    # 大表格：按行切分（保持表头）
                    table_chunks = self._split_table(table_content)
                    result.extend(table_chunks)
            else:
                # 文本：按段落切分
                text_chunks = self._split_text_only(item["content"], max_size, doc_id)
                result.extend(text_chunks)

        return result if result else [content]

    def _extract_tables_from_content(self, content: str) -> dict:
        """从内容中提取 Markdown 表格，返回表格和文本的有序列表。

        Returns:
            {
                "tables": [table1, table2, ...],  # 表格内容列表
                "texts": [text1, text2, ...],     # 非表格文本列表
                "ordered_items": [{"type": "table/text", "content": "..."}, ...]
            }
        """
        lines = content.split("\n")
        tables: list[str] = []
        texts: list[str] = []
        ordered_items: list[dict] = []

        current_table_lines: list[str] = []
        current_text_lines: list[str] = []
        in_table = False

        for line in lines:
            # 判断是否是表格行：以 | 开头或结尾，或全是分隔符
            is_table_line = self._is_table_line(line)

            if is_table_line:
                # 如果之前有文本，先保存
                if current_text_lines:
                    text_content = "\n".join(current_text_lines).strip()
                    if text_content:
                        texts.append(text_content)
                        ordered_items.append({"type": "text", "content": text_content})
                    current_text_lines = []

                # 添加到当前表格
                current_table_lines.append(line)
                in_table = True
            else:
                # 如果之前有表格，先保存
                if current_table_lines:
                    table_content = "\n".join(current_table_lines).strip()
                    if table_content and self._is_valid_table(table_content):
                        tables.append(table_content)
                        ordered_items.append({"type": "table", "content": table_content})
                    elif table_content:
                        # 不是有效表格，当作文本
                        texts.append(table_content)
                        ordered_items.append({"type": "text", "content": table_content})
                    current_table_lines = []

                # 添加到当前文本
                if line.strip():
                    current_text_lines.append(line)
                in_table = False

        # 处理剩余内容
        if current_table_lines:
            table_content = "\n".join(current_table_lines).strip()
            if table_content and self._is_valid_table(table_content):
                tables.append(table_content)
                ordered_items.append({"type": "table", "content": table_content})
            elif table_content:
                texts.append(table_content)
                ordered_items.append({"type": "text", "content": table_content})

        if current_text_lines:
            text_content = "\n".join(current_text_lines).strip()
            if text_content:
                texts.append(text_content)
                ordered_items.append({"type": "text", "content": text_content})

        return {
            "tables": tables,
            "texts": texts,
            "ordered_items": ordered_items,
        }

    def _is_table_line(self, line: str) -> bool:
        """判断一行是否是 Markdown 表格行。"""
        stripped = line.strip()

        # 空行不是表格
        if not stripped:
            return False

        # 代码块标记不是表格
        if stripped in ("```", "``"):
            return False

        # 以 | 开头和结尾
        if stripped.startswith("|") and stripped.endswith("|"):
            return True

        # 表格分隔行：全是 | - : 空格
        if all(c in "|-: " for c in stripped):
            return stripped.count("|") >= 2

        return False

    def _is_valid_table(self, table_content: str) -> bool:
        """判断内容是否是有效的 Markdown 表格（至少有表头和分隔行）。"""
        lines = [l.strip() for l in table_content.split("\n") if l.strip()]

        # 至少需要 2 行（表头 + 分隔行）
        if len(lines) < 2:
            return False

        # 第一行是表头（以 | 开头）
        if not lines[0].startswith("|"):
            return False

        # 第二行是分隔行（包含 - 和 |）
        if not all(c in "|-: " for c in lines[1]):
            return False

        return True

    def _split_table(self, table_content: str) -> list[str]:
        """切分超长表格，保持表头完整性。"""
        lines = table_content.split("\n")

        # 表头 + 分隔行
        header = lines[0]
        separator = lines[1] if len(lines) > 1 else ""
        header_block = f"{header}\n{separator}" if separator else header

        # 数据行
        data_lines = [l for l in lines[2:] if l.strip() and not self._is_table_separator_only(l)]

        if not data_lines:
            return [table_content]

        # 按数据行贪心聚合
        max_data_len = settings.CHUNK_SIZE_TABLE - len(header_block)
        result: list[str] = []
        current_lines: list[str] = []
        current_len = 0

        for line in data_lines:
            line_len = len(line)

            if current_len + line_len > max_data_len and current_lines:
                # 输出当前 chunk
                result.append(header_block + "\n" + "\n".join(current_lines))
                current_lines = [line]
                current_len = line_len
            else:
                current_lines.append(line)
                current_len += line_len

        # 最后一个 chunk
        if current_lines:
            result.append(header_block + "\n" + "\n".join(current_lines))

        return result if result else [table_content]

    def _is_table_separator_only(self, line: str) -> bool:
        """判断是否只是表格分隔行（只有 | - : 空格）。"""
        stripped = line.strip()
        return all(c in "|-: " for c in stripped) and stripped.count("|") >= 2

    def _split_text_only(
        self,
        content: str,
        max_size: int,
        doc_id: str,
    ) -> list[str]:
        """切分纯文本内容（不含表格）。"""
        # 如果内容不超长，直接返回
        if len(content) <= max_size:
            return [content]

        result: list[str] = []

        # 1. 先按段落边界切分
        paragraphs = _PARAGRAPH_RE.split(content)
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果当前段落能加入当前切片
            if len(current_chunk) + len(para) + 2 <= max_size:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                # 保存当前切片
                if current_chunk:
                    result.append(current_chunk)

                # 如果段落本身超长，需要按句子切分
                if len(para) > max_size:
                    sentences = _SENTENCE_BOUNDARY_RE.split(para)
                    current_chunk = ""
                    for sent in sentences:
                        sent = sent.strip()
                        if not sent:
                            continue
                        if len(current_chunk) + len(sent) + 1 <= max_size:
                            current_chunk += (" " if current_chunk else "") + sent
                        else:
                            if current_chunk:
                                result.append(current_chunk)
                            # 如果单个句子超长，强制截断
                            if len(sent) > max_size:
                                for i in range(0, len(sent), max_size):
                                    result.append(sent[i : i + max_size])
                                current_chunk = ""
                            else:
                                current_chunk = sent
                else:
                    current_chunk = para

        # 保存最后一个切片
        if current_chunk:
            result.append(current_chunk)

        # 最终检查：确保没有切片超长
        final_result = []
        for chunk in result:
            if len(chunk) > max_size:
                # 强制截断（兜底）
                for i in range(0, len(chunk), max_size):
                    final_result.append(chunk[i : i + max_size])
            else:
                final_result.append(chunk)

        logger.debug(
            "content_split_complete",
            doc_id=doc_id,
            original_len=len(content),
            split_count=len(final_result),
        )

        return final_result if final_result else [content]

    # ============================================================
    # 三阶段切分策略：标题 → 语义 → 递归
    # ============================================================

    def _semantic_then_recursive_split(
        self,
        content: str,
        doc_id: str,
        max_size: int,
        doc_type: str,
    ) -> list[str]:
        """两阶段切分：语义切分 + 递归切分（兜底）。

        流程：
        1. 语义切分：计算句子嵌入相似度，在语义边界切分
        2. 长度检查：超长片段使用递归切分作为兜底

        优势：
        - 保持语义连贯性，避免在不相关位置切分
        - 自适应切分，不同文档自动调整边界
        - 确保 chunk 不超过长度限制

        Args:
            content: 待切分的文本内容
            doc_id: 文档 ID（用于日志）
            max_size: 最大切片长度（字符数）
            doc_type: 文档类型

        Returns:
            切分后的内容列表
        """
        # 如果内容不超长，直接返回
        if len(content) <= max_size:
            return [content]

        # 语义切分参数
        similarity_threshold = 0.5  # 相似度阈值，低于此值视为语义边界
        buffer_sentences = 3  # 边界缓冲，防止过短 chunk

        # 1. 拆分为句子（支持中英文）
        sentences = self._split_sentences_semantic(content)
        if not sentences:
            return [content]

        # 过滤噪音句子
        filtered_sentences = []
        for s in sentences:
            should_filter, reason = should_filter_text(s)
            if not should_filter:
                filtered_sentences.append(s)

        if not filtered_sentences:
            return [content]

        # 如果句子数量太少，直接用递归切分
        if len(filtered_sentences) < 4:
            return self._split_text_only(content, max_size, doc_id)

        # 2. 计算句子嵌入（批量）
        embeddings = self._compute_embeddings(filtered_sentences)
        if embeddings is None or len(embeddings) < 2:
            # 嵌入失败，回退到递归切分
            logger.warning("embedding_failed_fallback_recursive", doc_id=doc_id)
            return self._split_text_only(content, max_size, doc_id)

        # 3. 检测语义边界
        boundaries = self._detect_semantic_boundaries(
            embeddings, similarity_threshold, buffer_sentences
        )

        logger.debug(
            "semantic_boundaries_detected",
            doc_id=doc_id,
            total_sentences=len(filtered_sentences),
            boundaries_count=len(boundaries),
        )

        # 4. 合并句子为语义片段
        semantic_segments = self._merge_sentences_to_segments(
            filtered_sentences, boundaries
        )

        # 5. 对每个语义片段进行长度检查
        result: list[str] = []
        for segment in semantic_segments:
            if len(segment) <= max_size:
                result.append(segment)
            else:
                # 超长片段：使用递归切分兜底
                sub_chunks = self._split_text_only(segment, max_size, doc_id)
                result.extend(sub_chunks)

        logger.info(
            "semantic_then_recursive_split_complete",
            doc_id=doc_id,
            original_len=len(content),
            segment_count=len(result),
        )

        return result if result else [content]

    def _split_sentences_semantic(self, text: str) -> list[str]:
        """拆分文本为句子（支持中英文）。

        用于语义切分的句子分割，保留句子完整性。
        """
        # 中文句子边界：句号、问号、感叹号、分号、换行
        zh_pattern = r'[。！？；\n]+'
        # 英文句子边界：句号/问号/感叹号 + 空格
        en_pattern = r'[.!?;]\s+'

        # 合并两种边界
        pattern = f'({zh_pattern}|{en_pattern})'

        sentences = re.split(pattern, text)
        # 过滤空句子和过短句子（少于5字符通常不是完整句子）
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

    def _compute_embeddings(self, sentences: list[str]) -> np.ndarray | None:
        """批量计算句子嵌入。

        使用 unified_embedder 进行文本嵌入。

        Args:
            sentences: 句子列表

        Returns:
            嵌入矩阵 (n_sentences, embedding_dim)，失败返回 None
        """
        try:
            from app.embedders.unified_embedder import unified_embedder

            embeddings = []
            batch_size = 32  # 批量处理，避免 API 限流

            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                vectors = unified_embedder.encode_texts(batch)
                if vectors:
                    embeddings.extend(vectors)

            if not embeddings:
                return None

            return np.array(embeddings)

        except Exception as e:
            logger.error("embedding_compute_error", error=str(e))
            return None

    def _detect_semantic_boundaries(
        self,
        embeddings: np.ndarray,
        threshold: float,
        buffer: int,
    ) -> list[int]:
        """检测语义边界。

        通过计算相邻句子相似度，在低相似度位置识别语义边界。

        Args:
            embeddings: 嵌入矩阵
            threshold: 相似度阈值
            buffer: 边界缓冲（避免连续边界）

        Returns:
            边界索引列表
        """
        if len(embeddings) < 2:
            return []

        # 计算相邻句子的余弦相似度
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # 找到低相似度位置（语义边界）
        boundaries = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                # 添加缓冲，避免过短 chunk
                if not boundaries or i - boundaries[-1] >= buffer:
                    boundaries.append(i + 1)  # 边界在 i+1 处切分

        return boundaries

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算余弦相似度。"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _merge_sentences_to_segments(
        self,
        sentences: list[str],
        boundaries: list[int],
    ) -> list[str]:
        """将句子合并为语义片段。

        Args:
            sentences: 句子列表
            boundaries: 边界索引列表

        Returns:
            合并后的片段列表
        """
        # 添加最后边界（句子总数）
        all_boundaries = sorted(set(boundaries + [len(sentences)]))

        segments: list[str] = []
        start = 0

        for end in all_boundaries:
            if start < end and end <= len(sentences):
                segment_sentences = sentences[start:end]
                content = " ".join(segment_sentences)
                if content.strip():
                    segments.append(content.strip())
            start = end

        return segments

    # ============================================================
    # 纯表格文档处理（Excel 等）
    # ============================================================

    def _make_pure_table_chunks(
        self,
        tables_data: list[dict],
        doc_id: str,
        doc_type: str,
        creator: str = "",
        kb_id: str = "",
        doc_name: str = "",
    ) -> tuple[list[dict], dict]:
        """直接从表格数据生成切片（用于 Excel 等纯表格文档）。

        Excel 等文档本质是纯表格数据，无需 Markdown 分块流程，
        直接从 Converter 提供的 tables_data 生成切片。

        Args:
            tables_data: 表格结构化数据列表
            doc_id: 文档 ID
            doc_type: 文档类型
            creator: 创建者
            kb_id: 知识库 ID
            doc_name: 文档名称

        Returns:
            (chunks_list, tree_data): 切片列表和树状结构数据
        """
        import uuid
        chunks_list: list[dict] = []

        for table_idx, table in enumerate(tables_data):
            chunk_id = f"{doc_id}_table_{uuid.uuid4().hex[:8]}"

            # Sheet 名称作为标题
            heading_text = table.get("sheet_name", f"表格{table_idx + 1}")

            # 表格切片内容（JSON 格式）
            content_data = {
                "type": "table",
                "headers": table.get("headers", []),
                "rows": table.get("rows", []),
                "merges": table.get("merges", []),
                "row_count": table.get("row_count", 0),
                "col_count": table.get("col_count", 0),
                "html": table.get("html", ""),
                "sheet_name": table.get("sheet_name", f"表格{table_idx + 1}"),
            }

            chunk = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_type": "table",
                "content": json.dumps(content_data),
                "page_num": table_idx + 1,
                "heading_level": 2,
                "heading_text": heading_text,
                "sort_order": table_idx,
                "creator": creator,
                "kb_id": kb_id,
                "doc_name": doc_name,
                "metadata": {
                    "doc_type": doc_type,
                    "row_count": table.get("row_count", 0),
                    "col_count": table.get("col_count", 0),
                    "sheet_name": table.get("sheet_name", f"表格{table_idx + 1}"),
                },
            }
            chunks_list.append(chunk)

        # Excel 无树状结构，返回空树
        tree_data = {"nodes": [], "flat": []}

        logger.info(
            "pure_table_chunks_created",
            doc_id=doc_id,
            doc_type=doc_type,
            table_count=len(tables_data),
            chunk_count=len(chunks_list),
        )

        return chunks_list, tree_data

    # ============================================================
    # 统一清洗方法
    # ============================================================

    def _clean_markdown(self, markdown_text: str, doc_type: str) -> str:
        """清洗 Markdown 文本中的噪音内容。

        使用 noise_filter 模块进行统一清洗：
        1. 基础清洗：去除控制字符、多余空白
        2. 噪音过滤：页码、页眉页脚、水印关键词
        3. 按行过滤：垃圾模式匹配

        Args:
            markdown_text: 原始 Markdown 文本
            doc_type: 文档类型

        Returns:
            清洗后的 Markdown 文本
        """
        # 1. 基础清洗：控制字符、多余空白
        cleaned = clean_text(markdown_text)

        # 2. 按行过滤噪音
        lines = cleaned.split("\n")
        filtered_lines: list[str] = []

        for line in lines:
            line_stripped = line.strip()

            # 空行保留（Markdown 格式需要）
            if not line_stripped:
                filtered_lines.append("")
                continue

            # 保留 Markdown 标题（以 # 开头）
            if line_stripped.startswith("#"):
                filtered_lines.append(line)
                continue

            # 保留 Markdown 表格行（以 | 开头）
            if line_stripped.startswith("|"):
                filtered_lines.append(line)
                continue

            # 保留代码块标记
            if line_stripped.startswith("```"):
                filtered_lines.append(line)
                continue

            # 检查是否为噪音文本
            if is_noise_text(line_stripped):
                logger.debug("noise_filtered", line=line_stripped[:50], doc_type=doc_type)
                continue

            filtered_lines.append(line)

        result = "\n".join(filtered_lines)

        # 3. 去除多余空行（连续超过3个变为2个）
        result = re.sub(r"\n{4,}", "\n\n\n", result)

        logger.info(
            "markdown_cleaned",
            doc_type=doc_type,
            original_lines=len(lines),
            filtered_lines=len(filtered_lines),
            removed_lines=len(lines) - len(filtered_lines),
        )

        return result

    # ============================================================
    # 策略切分方法
    # ============================================================

    def _chunk_with_heading_only(
        self,
        markdown_text: str,
        doc_id: str,
        doc_type: str,
        image_paths: list[str],
        creator: str,
        kb_id: str,
        doc_name: str,
        tables_data: list[dict],
        code_blocks_data: list[dict],
        images_data: list[dict],
    ) -> tuple[list[dict], dict]:
        """heading 策略：只按标题切分，不做二次切分。

        适用场景：
        - 需要保持文档原始结构的场景
        - 标题层级明确的文档
        - 用户希望通过标题导航的场景

        特点：
        - 每个 section 直接成为一个 chunk
        - 保留完整的标题路径
        - 不检查长度，可能产生超长 chunk
        """
        tables_map = {t["index"]: t for t in tables_data}
        sections = self._split_markdown_by_headers(markdown_text)

        chunks_list: list[dict] = []
        paragraphs: list[dict] = []
        stack: list[tuple[int, dict]] = []
        sort_order = 0

        for section in sections:
            heading_path = section["heading_path"]
            heading_levels = section["heading_levels"]
            level = section["level"]
            content = section["content"]

            if not content or len(content.strip()) < 10:
                continue

            # 构建树状结构
            paragraph_id = ""
            for i, path_title in enumerate(heading_path):
                path_level = heading_levels[i]
                while stack and stack[-1][0] >= path_level:
                    stack.pop()

                existing_child = None
                if i == 0:
                    for p in paragraphs:
                        if p["title"] == path_title and p["level"] == path_level:
                            existing_child = p
                            break
                else:
                    if stack:
                        parent_node = stack[-1][1]
                        for child in parent_node["childContentParagraph"]:
                            if child["title"] == path_title and child["level"] == path_level:
                                existing_child = child
                                break

                if existing_child:
                    paragraph_node = existing_child
                else:
                    paragraph_node = {
                        "id": f"p-{path_level}-{uuid.uuid4().hex[:6]}",
                        "title": path_title,
                        "level": path_level,
                        "content": path_title,
                        "childContentParagraph": [],
                        "chunks": [],
                    }
                    if i == 0:
                        paragraphs.append(paragraph_node)
                    elif stack:
                        stack[-1][1]["childContentParagraph"].append(paragraph_node)

                stack.append((path_level, paragraph_node))
                paragraph_id = paragraph_node["id"]

            full_heading_path = " > ".join(heading_path)

            # 处理内容片段（表格/代码块单独处理，文本保持完整）
            content_fragments = self._split_by_markers_and_code(content, tables_map)

            for fragment in content_fragments:
                if fragment["type"] == "table_marker":
                    table_idx = fragment["table_idx"]
                    table_data = tables_map.get(table_idx)
                    if table_data:
                        chunk_id = f"{doc_id}_table_{uuid.uuid4().hex[:8]}_{sort_order}"
                        chunk_item = self._make_table_chunk(
                            chunk_id, doc_id, paragraph_id, table_data,
                            level, full_heading_path, sort_order, creator, kb_id, doc_type,
                            None, doc_name,
                        )
                        chunks_list.append(chunk_item)
                        if stack:
                            stack[-1][1]["chunks"].append(self._chunk_to_tree_item(chunk_item, paragraph_id))
                        sort_order += 1
                elif fragment["type"] == "markdown_table":
                    table_data = self._parse_markdown_table(fragment["content"])
                    chunk_id = f"{doc_id}_table_{uuid.uuid4().hex[:8]}_{sort_order}"
                    chunk_item = self._make_table_chunk(
                        chunk_id, doc_id, paragraph_id, table_data,
                        level, full_heading_path, sort_order, creator, kb_id, doc_type,
                        None, doc_name,
                    )
                    chunks_list.append(chunk_item)
                    if stack:
                        stack[-1][1]["chunks"].append(self._chunk_to_tree_item(chunk_item, paragraph_id))
                    sort_order += 1
                elif fragment["type"] == "code_block":
                    chunk_id = f"{doc_id}_code_{uuid.uuid4().hex[:8]}_{sort_order}"
                    chunk_item = self._make_code_chunk(
                        chunk_id, doc_id, paragraph_id, fragment["content"],
                        fragment["language"], level, full_heading_path, sort_order, creator,
                        kb_id, doc_type, None, doc_name,
                    )
                    chunks_list.append(chunk_item)
                    if stack:
                        stack[-1][1]["chunks"].append(self._chunk_to_tree_item(chunk_item, paragraph_id))
                    sort_order += 1
                else:
                    # 文本：直接成为 chunk，不做二次切分
                    chunk_id = f"{doc_id}_text_{uuid.uuid4().hex[:8]}_{sort_order}"
                    chunk_item = self._make_chunk(
                        chunk_id, doc_id, paragraph_id, fragment["content"].strip(),
                        level, full_heading_path, sort_order, creator, kb_id, doc_type,
                        None, doc_name,
                    )
                    chunks_list.append(chunk_item)
                    if stack:
                        stack[-1][1]["chunks"].append(self._chunk_to_tree_item(chunk_item, paragraph_id))
                    sort_order += 1

        # 处理图片
        chunks_list.extend(self._make_image_chunks(image_paths, images_data, doc_id, creator, kb_id, doc_name, sort_order))

        tree_data = {"doc_id": doc_id, "doc_type": doc_type, "paragraphs": paragraphs}
        logger.info("heading_only_chunks_created", doc_id=doc_id, chunk_count=len(chunks_list))
        return chunks_list, tree_data

    def _chunk_with_semantic_only(
        self,
        markdown_text: str,
        doc_id: str,
        doc_type: str,
        image_paths: list[str],
        creator: str,
        kb_id: str,
        doc_name: str,
        tables_data: list[dict],
        code_blocks_data: list[dict],
        images_data: list[dict],
    ) -> tuple[list[dict], dict]:
        """semantic 策略：不按标题切分，直接对全文语义切分。

        适用场景：
        - 需要保持语义连贯性的场景
        - 无明确标题层级的长文档
        - 论文、技术文档

        特点：
        - 跳过标题切分
        - 使用嵌入相似度检测语义边界
        - 保持语义完整性
        """
        tables_map = {t["index"]: t for t in tables_data}
        max_size = settings.CHUNK_SIZE_TEXT

        # 先提取表格和代码块
        content_fragments = self._split_by_markers_and_code(markdown_text, tables_map)

        chunks_list: list[dict] = []
        sort_order = 0

        for fragment in content_fragments:
            if fragment["type"] == "table_marker":
                table_idx = fragment["table_idx"]
                table_data = tables_map.get(table_idx)
                if table_data:
                    chunk_id = f"{doc_id}_table_{uuid.uuid4().hex[:8]}_{sort_order}"
                    chunk_item = self._make_table_chunk(
                        chunk_id, doc_id, "", table_data,
                        0, "", sort_order, creator, kb_id, doc_type,
                        None, doc_name,
                    )
                    chunks_list.append(chunk_item)
                    sort_order += 1
            elif fragment["type"] == "markdown_table":
                table_data = self._parse_markdown_table(fragment["content"])
                chunk_id = f"{doc_id}_table_{uuid.uuid4().hex[:8]}_{sort_order}"
                chunk_item = self._make_table_chunk(
                    chunk_id, doc_id, "", table_data,
                    0, "", sort_order, creator, kb_id, doc_type,
                    None, doc_name,
                )
                chunks_list.append(chunk_item)
                sort_order += 1
            elif fragment["type"] == "code_block":
                chunk_id = f"{doc_id}_code_{uuid.uuid4().hex[:8]}_{sort_order}"
                chunk_item = self._make_code_chunk(
                    chunk_id, doc_id, "", fragment["content"],
                    fragment["language"], 0, "", sort_order, creator,
                    kb_id, doc_type, None, doc_name,
                )
                chunks_list.append(chunk_item)
                sort_order += 1
            else:
                # 文本：语义切分
                sub_chunks = self._semantic_then_recursive_split(
                    fragment["content"], doc_id, max_size, doc_type
                )
                for sub_content in sub_chunks:
                    if sub_content.strip() and len(sub_content.strip()) >= 10:
                        chunk_id = f"{doc_id}_text_{uuid.uuid4().hex[:8]}_{sort_order}"
                        chunk_item = self._make_chunk(
                            chunk_id, doc_id, "", sub_content.strip(),
                            0, "", sort_order, creator, kb_id, doc_type,
                            None, doc_name,
                        )
                        chunks_list.append(chunk_item)
                        sort_order += 1

        # 处理图片
        chunks_list.extend(self._make_image_chunks(image_paths, images_data, doc_id, creator, kb_id, doc_name, sort_order))

        # semantic 策略无树状结构
        tree_data = {"doc_id": doc_id, "doc_type": doc_type, "paragraphs": []}
        logger.info("semantic_only_chunks_created", doc_id=doc_id, chunk_count=len(chunks_list))
        return chunks_list, tree_data

    def _chunk_with_recursive_only(
        self,
        markdown_text: str,
        doc_id: str,
        doc_type: str,
        image_paths: list[str],
        creator: str,
        kb_id: str,
        doc_name: str,
        tables_data: list[dict],
        code_blocks_data: list[dict],
        images_data: list[dict],
    ) -> tuple[list[dict], dict]:
        """recursive 策略：不按标题切分，直接递归切分。

        适用场景：
        - 快速处理，不需要高质量切分
        - 对文档结构不敏感的场景
        - 测试或调试用途

        特点：
        - 跳过标题切分
        - 按段落 → 句子 → 截断递归切分
        - 最简单快速
        """
        tables_map = {t["index"]: t for t in tables_data}
        max_size = settings.CHUNK_SIZE_TEXT

        content_fragments = self._split_by_markers_and_code(markdown_text, tables_map)

        chunks_list: list[dict] = []
        sort_order = 0

        for fragment in content_fragments:
            if fragment["type"] == "table_marker":
                table_idx = fragment["table_idx"]
                table_data = tables_map.get(table_idx)
                if table_data:
                    chunk_id = f"{doc_id}_table_{uuid.uuid4().hex[:8]}_{sort_order}"
                    chunk_item = self._make_table_chunk(
                        chunk_id, doc_id, "", table_data,
                        0, "", sort_order, creator, kb_id, doc_type,
                        None, doc_name,
                    )
                    chunks_list.append(chunk_item)
                    sort_order += 1
            elif fragment["type"] == "markdown_table":
                table_data = self._parse_markdown_table(fragment["content"])
                chunk_id = f"{doc_id}_table_{uuid.uuid4().hex[:8]}_{sort_order}"
                chunk_item = self._make_table_chunk(
                    chunk_id, doc_id, "", table_data,
                    0, "", sort_order, creator, kb_id, doc_type,
                    None, doc_name,
                )
                chunks_list.append(chunk_item)
                sort_order += 1
            elif fragment["type"] == "code_block":
                chunk_id = f"{doc_id}_code_{uuid.uuid4().hex[:8]}_{sort_order}"
                chunk_item = self._make_code_chunk(
                    chunk_id, doc_id, "", fragment["content"],
                    fragment["language"], 0, "", sort_order, creator,
                    kb_id, doc_type, None, doc_name,
                )
                chunks_list.append(chunk_item)
                sort_order += 1
            else:
                # 文本：直接递归切分
                sub_chunks = self._split_text_only(fragment["content"], max_size, doc_id)
                for sub_content in sub_chunks:
                    if sub_content.strip() and len(sub_content.strip()) >= 10:
                        chunk_id = f"{doc_id}_text_{uuid.uuid4().hex[:8]}_{sort_order}"
                        chunk_item = self._make_chunk(
                            chunk_id, doc_id, "", sub_content.strip(),
                            0, "", sort_order, creator, kb_id, doc_type,
                            None, doc_name,
                        )
                        chunks_list.append(chunk_item)
                        sort_order += 1

        # 处理图片
        chunks_list.extend(self._make_image_chunks(image_paths, images_data, doc_id, creator, kb_id, doc_name, sort_order))

        tree_data = {"doc_id": doc_id, "doc_type": doc_type, "paragraphs": []}
        logger.info("recursive_only_chunks_created", doc_id=doc_id, chunk_count=len(chunks_list))
        return chunks_list, tree_data

    def _chunk_with_fixed_length(
        self,
        markdown_text: str,
        doc_id: str,
        doc_type: str,
        image_paths: list[str],
        creator: str,
        kb_id: str,
        doc_name: str,
        tables_data: list[dict],
        code_blocks_data: list[dict],
        images_data: list[dict],
    ) -> tuple[list[dict], dict]:
        """fixed 策略：固定长度切分。

        适用场景：
        - 最基础方案
        - 对切分质量要求不高
        - 测试用途

        特点：
        - 按固定字符数切分
        - 不考虑语义边界
        - 最简单实现
        """
        max_size = settings.CHUNK_SIZE_TEXT
        tables_map = {t["index"]: t for t in tables_data}

        # 先提取表格和代码块
        content_fragments = self._split_by_markers_and_code(markdown_text, tables_map)

        chunks_list: list[dict] = []
        sort_order = 0

        for fragment in content_fragments:
            if fragment["type"] == "table_marker":
                table_idx = fragment["table_idx"]
                table_data = tables_map.get(table_idx)
                if table_data:
                    chunk_id = f"{doc_id}_table_{uuid.uuid4().hex[:8]}_{sort_order}"
                    chunk_item = self._make_table_chunk(
                        chunk_id, doc_id, "", table_data,
                        0, "", sort_order, creator, kb_id, doc_type,
                        None, doc_name,
                    )
                    chunks_list.append(chunk_item)
                    sort_order += 1
            elif fragment["type"] == "markdown_table":
                table_data = self._parse_markdown_table(fragment["content"])
                chunk_id = f"{doc_id}_table_{uuid.uuid4().hex[:8]}_{sort_order}"
                chunk_item = self._make_table_chunk(
                    chunk_id, doc_id, "", table_data,
                    0, "", sort_order, creator, kb_id, doc_type,
                    None, doc_name,
                )
                chunks_list.append(chunk_item)
                sort_order += 1
            elif fragment["type"] == "code_block":
                chunk_id = f"{doc_id}_code_{uuid.uuid4().hex[:8]}_{sort_order}"
                chunk_item = self._make_code_chunk(
                    chunk_id, doc_id, "", fragment["content"],
                    fragment["language"], 0, "", sort_order, creator,
                    kb_id, doc_type, None, doc_name,
                )
                chunks_list.append(chunk_item)
                sort_order += 1
            else:
                # 文本：固定长度切分
                content = fragment["content"]
                for i in range(0, len(content), max_size):
                    chunk_content = content[i:i + max_size].strip()
                    if chunk_content and len(chunk_content) >= 10:
                        chunk_id = f"{doc_id}_text_{uuid.uuid4().hex[:8]}_{sort_order}"
                        chunk_item = self._make_chunk(
                            chunk_id, doc_id, "", chunk_content,
                            0, "", sort_order, creator, kb_id, doc_type,
                            None, doc_name,
                        )
                        chunks_list.append(chunk_item)
                        sort_order += 1

        # 处理图片
        chunks_list.extend(self._make_image_chunks(image_paths, images_data, doc_id, creator, kb_id, doc_name, sort_order))

        tree_data = {"doc_id": doc_id, "doc_type": doc_type, "paragraphs": []}
        logger.info("fixed_length_chunks_created", doc_id=doc_id, chunk_count=len(chunks_list))
        return chunks_list, tree_data

    def _make_image_chunks(
        self,
        image_paths: list[str],
        images_data: list[dict],
        doc_id: str,
        creator: str,
        kb_id: str,
        doc_name: str,
        start_sort_order: int,
    ) -> list[dict]:
        """生成图片切片列表。

        注意：content 字段存储图片的说明文本（用于语义搜索），不存储 JSON 元数据。
        图片路径和 URL 等信息存储在 metadata 中。
        """
        chunks: list[dict] = []
        sort_order = start_sort_order

        for img_idx, img_path in enumerate(image_paths):
            img_chunk_id = f"{doc_id}_img_{uuid.uuid4().hex[:8]}"
            img_url = None
            img_page_num = 0
            if images_data and img_idx < len(images_data):
                img_url = images_data[img_idx].get("url")
                img_page_num = images_data[img_idx].get("page_num", 0)

            # content 存储描述性文本，用于语义搜索和展示
            # 不存储 JSON 元数据，避免被错误使用为导航参数
            img_content = f"图片内容 (第 {img_page_num} 页)"

            img_chunk = {
                "chunk_id": img_chunk_id,
                "doc_id": doc_id,
                "chunk_type": "image",
                "content": img_content,  # 描述性文本，不是 JSON
                "caption": None,
                "page_num": img_page_num,  # 图片页码
                "heading_level": 0,
                "heading_text": "",
                "sort_order": sort_order,
                "metadata": {
                    "doc_type": "",
                    "vector_status": "pending",
                    "image_path": img_path,
                    "url": img_url or img_path,  # 前端访问地址
                    "image_index": img_idx,  # 图片索引
                    "vlm_caption": "",
                    "context_text": "",
                },
                "creator": creator,
                "kb_id": kb_id,
                "doc_name": doc_name,
            }
            chunks.append(img_chunk)
            sort_order += 1

        return chunks


# 单例
markdown_chunker = MarkdownChunker()