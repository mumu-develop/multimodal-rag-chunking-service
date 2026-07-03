import os
import re
import zipfile
from typing import Optional

import structlog

from app.config import settings
from app.converters import MarkdownConverter

logger = structlog.get_logger()

# Word 标准样式到 Markdown 标题的映射
_HEADING_STYLE_TO_MARKDOWN = {
    "Heading 1": "#",
    "Heading 2": "##",
    "Heading 3": "###",
    "Heading 4": "####",
    "Heading 5": "#####",
    "Heading 6": "######",
    "Heading 7": "######",
    "Heading 8": "######",
    "Heading 9": "######",
    "Title": "#",
    "Subtitle": "##",
}

# 中文样式映射
_CHINESE_STYLE_TO_MARKDOWN = {
    "标题 1": "#",
    "标题 2": "##",
    "标题 3": "###",
    "标题 4": "####",
    "标题 5": "#####",
    "标题 6": "######",
}


class DocxConverter(MarkdownConverter):
    """DOCX 文件转换器，提取结构化数据供前端渲染。"""

    def convert(self, file_path: str, doc_id: str) -> str:
        """将 DOCX 文件转换为 Markdown（兼容旧接口）。"""
        result = self.convert_full(file_path, doc_id)
        return result["markdown"]

    def convert_full(self, file_path: str, doc_id: str) -> dict:
        """
        将 DOCX 文件转换为结构化数据。

        流程步骤：
        1. 使用 python-docx 打开文档
        2. 提取段落文本，转换为 Markdown 格式
        3. 提取表格，生成 HTML + 结构化数据（headers, rows, merges）
        4. 提取图片路径
        5. 识别代码块（通过样式或特殊标记）
        6. 返回结构化输出

        参数：
            file_path: DOCX 文件路径
            doc_id: 文档 ID

        返回：
            {
                "markdown": str,               # Markdown 文本（用于分块和搜索）
                "tables": list[dict],          # 表格结构化数据（用于前端渲染）
                "images": list[dict],          # 图片信息
                "code_blocks": list[dict],     # 代码块信息
            }
        """
        from docx import Document

        # 步骤1：打开文档
        doc = Document(file_path)

        lines: list[str] = []
        tables_data: list[dict] = []
        code_blocks: list[dict] = []
        table_idx = 0

        # 步骤2：处理段落
        for para_idx, para in enumerate(doc.paragraphs):
            text = para.text.strip()
            if not text:
                continue

            # 过滤页眉页脚/页码
            if self._is_header_footer(text, para.style.name if para.style else None):
                continue

            style_name = para.style.name if para.style else "Normal"

            # 检查是否是代码块（特殊样式或标记）
            is_code = self._is_code_paragraph(style_name, text)
            if is_code:
                # 代码块：添加标记，不切断
                code_blocks.append({
                    "content": text,
                    "language": self._detect_language(text),
                    "paragraph_index": para_idx,
                })
                lines.append(f"\n```{self._detect_language(text)}\n{text}\n```\n")
                continue

            # 映射样式到 Markdown 标题
            md_heading = self._style_to_markdown_heading(style_name, text)
            if md_heading:
                lines.append(f"\n{md_heading} {text}\n")
            else:
                # 检查是否是列表项
                if style_name.startswith("List"):
                    lines.append(f"- {text}")
                else:
                    lines.append(text)

        # 步骤3：处理表格
        for table in doc.tables:
            table_info = self._extract_table_structure(table, table_idx)
            tables_data.append(table_info)
            # Markdown 中用特殊标记表示表格位置
            lines.append(f"\n[TABLE_{table_idx}]\n")
            table_idx += 1

        # 步骤4：提取图片
        images = self.extract_images(file_path, doc_id)
        images_data = [{"path": p, "index": i} for i, p in enumerate(images)]

        # 步骤5：构建输出
        return {
            "markdown": "\n".join(lines),
            "tables": tables_data,
            "images": images_data,
            "code_blocks": code_blocks,
        }

    def _extract_table_structure(self, table, table_idx: int, nested_idx_offset: int = 0) -> dict:
        """
        提取表格结构化数据（用于前端渲染），支持嵌套表格。

        流程步骤：
        1. 获取表格所有行
        2. 检测每个单元格是否有嵌套表格
        3. 检测合并单元格（rowspan/colspan）
        4. 生成 HTML 格式（嵌套表格用占位符标记）
        5. 嵌套表格作为独立数据存储在 nested_tables 字段

        参数：
            table: python-docx Table 对象
            table_idx: 表格索引
            nested_idx_offset: 嵌套表格索引偏移（用于生成唯一索引）

        返回：
            {
                "index": int,
                "headers": list[str],
                "rows": list[list[str | dict]],  # 嵌套表格位置为 {"nested_index": int}
                "merges": list[dict],
                "html": str,                      # 嵌套位置用占位符
                "nested_tables": list[dict],      # 嵌套表格列表（独立数据）
                "has_nested": bool,
            }
        """
        from docx.table import Table

        # 步骤1：获取所有行，检测嵌套表格
        rows_data: list[list[str | dict]] = []
        nested_tables: list[dict] = []
        nested_positions: list[dict] = []  # 记录嵌套表格位置
        current_nested_idx = nested_idx_offset

        for row_idx, row in enumerate(table.rows):
            row_cells = []
            for col_idx, cell in enumerate(row.cells):
                # 检测嵌套表格
                if cell.tables:
                    # 有嵌套表格，提取嵌套表格数据
                    nested_table_idx = current_nested_idx
                    for nested_table in cell.tables:
                        nested_data = self._extract_table_structure(
                            nested_table, nested_table_idx, nested_idx_offset=current_nested_idx + 1
                        )
                        nested_data["parent_row"] = row_idx
                        nested_data["parent_col"] = col_idx
                        nested_data["nested_index"] = nested_table_idx
                        nested_tables.append(nested_data)
                        current_nested_idx += 1

                    # 单元格内容标记为嵌套表格引用
                    row_cells.append({"nested_index": nested_table_idx})
                    nested_positions.append({
                        "row": row_idx,
                        "col": col_idx,
                        "nested_index": nested_table_idx,
                    })
                    nested_table_idx += 1
                else:
                    # 普通单元格，提取文本
                    cell_text = cell.text.strip().replace("\n", " ")
                    row_cells.append(cell_text)
            rows_data.append(row_cells)

        if not rows_data:
            return {"index": table_idx, "headers": [], "rows": [], "merges": [], "html": "", "nested_tables": [], "has_nested": False}

        # 步骤2：检测合并单元格
        merges = self._detect_merges(table)

        # 步骤3：生成 HTML 格式（嵌套表格用占位符）
        html = self._table_to_html_with_nested(table, merges, nested_positions)

        # 步骤4：第一行作为表头
        headers = rows_data[0] if rows_data else []
        rows = rows_data[1:] if len(rows_data) > 1 else []

        return {
            "index": table_idx,
            "headers": headers,
            "rows": rows,
            "merges": merges,
            "html": html,
            "nested_tables": nested_tables,
            "has_nested": len(nested_tables) > 0,
            "nested_positions": nested_positions,
            "row_count": len(rows_data),
            "col_count": len(headers),
        }

    def _detect_merges(self, table) -> list[dict]:
        """
        检测合并单元格。

        合并单元格的判断逻辑：
        使用 python-docx 的底层 XML 检测 vMerge（垂直合并）和 gridSpan（水平合并）

        返回：
            [{"row": int, "col": int, "rowspan": int, "colspan": int}, ...]
        """
        merges: list[dict] = []
        try:
            from docx.oxml.ns import qn

            # 获取表格的实际列数（考虑水平合并）
            actual_cols = len(table.columns)

            # 追踪每个单元格是否已被处理（作为合并的一部分）
            processed = set()

            for row_idx, row in enumerate(table.rows):
                col_idx = 0
                for cell in row.cells:
                    # 跳过已处理的单元格
                    while (row_idx, col_idx) in processed and col_idx < actual_cols:
                        col_idx += 1

                    if col_idx >= actual_cols:
                        break

                    tc = cell._tc
                    rowspan = 1
                    colspan = 1

                    # 检测水平合并（gridSpan）
                    grid_span = tc.find(qn('w:gridSpan'))
                    if grid_span is not None:
                        colspan = int(grid_span.get(qn('w:val'), 1))

                    # 检测垂直合并（vMerge）
                    v_merge = tc.find(qn('w:vMerge'))
                    if v_merge is not None:
                        merge_val = v_merge.get(qn('w:val'), 'continue')
                        if merge_val == 'restart':
                            # 这是合并的起始单元格，计算合并的行数
                            rowspan = 1
                            for next_row_idx in range(row_idx + 1, len(table.rows)):
                                next_cell = table.rows[next_row_idx].cells[col_idx]
                                next_tc = next_cell._tc
                                next_v_merge = next_tc.find(qn('w:vMerge'))
                                if next_v_merge is not None:
                                    next_val = next_v_merge.get(qn('w:val'), 'continue')
                                    if next_val == 'continue':
                                        rowspan += 1
                                    else:
                                        break
                                else:
                                    break

                    # 如果有合并，记录信息
                    if rowspan > 1 or colspan > 1:
                        merges.append({
                            "row": row_idx,
                            "col": col_idx,
                            "rowspan": rowspan,
                            "colspan": colspan,
                        })

                    # 标记已处理的单元格
                    for r in range(row_idx, row_idx + rowspan):
                        for c in range(col_idx, col_idx + colspan):
                            processed.add((r, c))

                    col_idx += colspan

        except Exception as e:
            logger.debug("table_merge_detection_failed", error=str(e))

        return merges

    def _table_to_html_with_nested(self, table, merges: list[dict], nested_positions: list[dict]) -> str:
        """
        将表格转换为 HTML 格式，嵌套表格用占位符标记。

        参数：
            table: python-docx Table 对象
            merges: 合并单元格信息列表
            nested_positions: 嵌套表格位置列表 [{row, col, nested_index}, ...]

        返回：
            HTML 表格字符串（嵌套位置用 data-nested 属性标记）
        """
        # 构建嵌套位置映射 {(row, col): nested_index}
        nested_map = {(p["row"], p["col"]): p["nested_index"] for p in nested_positions}

        # 构建合并单元格映射 {(row, col): (rowspan, colspan)}
        merge_map = {}
        for m in merges:
            merge_map[(m["row"], m["col"])] = (m["rowspan"], m["colspan"])

        # 跳过单元格集合（作为合并的一部分被跳过）
        skip_cells = set()
        for (row, col), (rowspan, colspan) in merge_map.items():
            for r in range(row, row + rowspan):
                for c in range(col, col + colspan):
                    if (r, c) != (row, col):
                        skip_cells.add((r, c))

        # 构建 HTML
        html_lines = ["<table>"]
        col_count = len(table.columns)

        for row_idx, row in enumerate(table.rows):
            html_lines.append("<tr>")
            col_idx = 0
            for cell in row.cells:
                # 跳过被合并的单元格
                while (row_idx, col_idx) in skip_cells and col_idx < col_count:
                    col_idx += 1

                if col_idx >= col_count:
                    break

                cell_text = cell.text.strip().replace("\n", "<br>").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

                # 检查是否是嵌套表格位置
                if (row_idx, col_idx) in nested_map:
                    nested_index = nested_map[(row_idx, col_idx)]
                    # 嵌套表格占位符：用 data-nested 属性标记，前端可识别并弹窗展示
                    placeholder = f'<span data-nested="{nested_index}" class="nested-table-placeholder">查看嵌套表格</span>'
                    tag = "th" if row_idx == 0 else "td"
                    html_lines.append(f"<{tag}>{placeholder}</{tag}>")
                    col_idx += 1
                elif (row_idx, col_idx) in merge_map:
                    rowspan, colspan = merge_map[(row_idx, col_idx)]
                    attrs = []
                    if rowspan > 1:
                        attrs.append(f'rowspan="{rowspan}"')
                    if colspan > 1:
                        attrs.append(f'colspan="{colspan}"')
                    attr_str = " " + " ".join(attrs) if attrs else ""
                    tag = "th" if row_idx == 0 else "td"
                    html_lines.append(f"<{tag}{attr_str}>{cell_text}</{tag}>")
                    col_idx += colspan
                else:
                    tag = "th" if row_idx == 0 else "td"
                    html_lines.append(f"<{tag}>{cell_text}</{tag}>")
                    col_idx += 1

            html_lines.append("</tr>")

        html_lines.append("</table>")
        return "\n".join(html_lines)

    def _table_to_html(self, table, merges: list[dict]) -> str:
        """
        将表格转换为 HTML 格式（支持合并单元格）。

        参数：
            table: python-docx Table 对象
            merges: 合并单元格信息列表

        返回：
            HTML 表格字符串
        """
        # 构建合并单元格映射 {(row, col): (rowspan, colspan)}
        merge_map = {}
        for m in merges:
            merge_map[(m["row"], m["col"])] = (m["rowspan"], m["colspan"])

        # 跳过单元格集合（作为合并的一部分被跳过）
        skip_cells = set()
        for (row, col), (rowspan, colspan) in merge_map.items():
            for r in range(row, row + rowspan):
                for c in range(col, col + colspan):
                    if (r, c) != (row, col):
                        skip_cells.add((r, c))

        # 构建 HTML
        html_lines = ["<table>"]

        for row_idx, row in enumerate(table.rows):
            html_lines.append("<tr>")
            col_idx = 0
            for cell in row.cells:
                # 跳过被合并的单元格
                while (row_idx, col_idx) in skip_cells and col_idx < len(table.columns):
                    col_idx += 1

                if col_idx >= len(table.columns):
                    break

                cell_text = cell.text.strip().replace("\n", "<br>").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

                # 检查是否是合并起始单元格
                if (row_idx, col_idx) in merge_map:
                    rowspan, colspan = merge_map[(row_idx, col_idx)]
                    attrs = []
                    if rowspan > 1:
                        attrs.append(f'rowspan="{rowspan}"')
                    if colspan > 1:
                        attrs.append(f'colspan="{colspan}"')
                    attr_str = " " + " ".join(attrs) if attrs else ""
                    tag = "th" if row_idx == 0 else "td"
                    html_lines.append(f"<{tag}{attr_str}>{cell_text}</{tag}>")
                    col_idx += colspan
                else:
                    tag = "th" if row_idx == 0 else "td"
                    html_lines.append(f"<{tag}>{cell_text}</{tag}>")
                    col_idx += 1

            html_lines.append("</tr>")

        html_lines.append("</table>")
        return "\n".join(html_lines)

    def _is_code_paragraph(self, style_name: str, text: str) -> bool:
        """
        判断段落是否是代码块。

        判断逻辑：
        1. 样式名包含 "Code"、"Source"、"CodeBlock"
        2. 文本内容符合代码特征（关键字、缩进等）
        """
        code_styles = ["Code", "Source", "CodeBlock", "Code Text", "Preformatted"]

        # 样式判断
        for cs in code_styles:
            if cs.lower() in style_name.lower():
                return True

        # 内容特征判断（备用）
        # 如果文本有明显的代码特征，如 def/class/import 等
        code_keywords = ["def ", "class ", "import ", "function ", "const ", "let ", "var ", "public ", "private ", "#include"]
        if any(kw in text for kw in code_keywords):
            # 进一步检查：如果大部分内容是代码格式
            if text.count("\n") > 2 or text.count("    ") > 3:
                return True

        return False

    def _detect_language(self, text: str) -> str:
        """
        检测代码语言。

        检测逻辑：
        1. Python: def, import, class
        2. JavaScript: function, const, let, var
        3. Java: public, private, class
        4. C/C++: #include, int main
        5. SQL: SELECT, FROM, WHERE
        6. Shell: #!/bin/bash
        """
        if "def " in text or "import " in text and "from " in text:
            return "python"
        if "function " in text or "const " in text or "let " in text or "var " in text:
            return "javascript"
        if "public " in text or "private " in text and "class " in text:
            return "java"
        if "#include" in text or "int main" in text:
            return "cpp"
        if "SELECT " in text.upper() or "FROM " in text.upper():
            return "sql"
        if text.startswith("#!/bin/bash") or text.startswith("#!/bin/sh"):
            return "bash"
        if "{" in text and "}" in text and ":" in text:
            return "json"

        return ""  # 未知语言

    def extract_images(self, file_path: str, doc_id: str) -> list[str]:
        """从 DOCX 的 ZIP 结构中提取嵌入图片。"""
        output_dir = settings.IMAGE_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        image_paths: list[str] = []
        try:
            with zipfile.ZipFile(file_path) as zf:
                media_names = [
                    n for n in zf.namelist()
                    if n.startswith("word/media/") and not n.endswith("/")
                ]

                for idx, name in enumerate(sorted(media_names), start=1):
                    ext = os.path.splitext(name)[1] or ".png"
                    dest_name = f"{doc_id}_img_{idx - 1}{ext}"
                    dest_path = os.path.join(output_dir, dest_name)
                    with open(dest_path, "wb") as f:
                        f.write(zf.read(name))
                    image_paths.append(dest_path)
        except Exception:
            logger.exception("docx_image_extraction_failed", doc_id=doc_id)

        return image_paths

    def _style_to_markdown_heading(self, style_name: str, text: str) -> Optional[str]:
        """将 Word 样式映射到 Markdown 标题标记。"""
        # 短标签过滤
        short_labels = ["反例", "正例", "说明", "注意", "提示", "警告"]
        if any(text.startswith(label) for label in short_labels) and len(text) < 30:
            return None

        # 标准英文样式
        if style_name in _HEADING_STYLE_TO_MARKDOWN:
            return _HEADING_STYLE_TO_MARKDOWN[style_name]

        # 中文样式
        if style_name in _CHINESE_STYLE_TO_MARKDOWN:
            return _CHINESE_STYLE_TO_MARKDOWN[style_name]

        # 备用：从文本内容推断
        if style_name.lower() in ("normal", ""):
            return self._infer_heading_from_text(text)

        return None

    def _infer_heading_from_text(self, text: str) -> Optional[str]:
        """从文本内容推断标题层级。"""
        # 规则编号过滤
        if re.search(r"【(强制|推荐|参考)】", text):
            return None

        # 短标签过滤
        short_labels = ["反例", "正例", "说明", "注意", "提示", "警告"]
        if any(text.startswith(label) for label in short_labels) and len(text) < 20:
            return None

        # 中文章节
        if re.match(r"^第[一二三四五六七八九十百]+章", text):
            return "#"

        if re.match(r"^第[一二三四五六七八九十百]+节", text):
            return "##"

        if re.match(r"^([一二三四五六七八九十])[、\s：:]", text):
            return "#"

        if re.match(r"^[(（][一二三四五六七八九十]+[)）]", text):
            return "##"

        if re.match(r"^\d+\.\d+", text):
            return "##"

        return None

    def _is_header_footer(self, text: str, style_name: Optional[str]) -> bool:
        """判断是否是页眉页脚或页码。"""
        if style_name and style_name.lower() in ("header", "footer"):
            return True

        if re.match(r"^(\d+/\d+|第\s*\d+\s*页)", text):
            return True

        return False

    def split_table_data(self, table_data: dict, max_rows: int = 20) -> list[dict]:
        """
        智能切分大表格，保护合并单元格和嵌套表格行。

        流程步骤：
        1. 分析 merges，找出"禁止切分点"（合并单元格覆盖的行范围不能被切断）
        2. 加入嵌套表格行的禁止切分点
        3. 计算允许切分的边界点
        4. 在边界点切分
        5. 每个切分后的 chunk：
           - 保留表头（第一行）
           - 重新计算 rowspan（如果被截断）
           - 重新生成 HTML

        参数：
            table_data: 表格结构化数据 {headers, rows, merges, html, row_count, col_count, _extra_forbidden_rows}
            max_rows: 每个 chunk 最大行数（不含表头）

        返回：
            切分后的表格数据列表，每个元素格式与 table_data 相同
        """
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])
        merges = table_data.get("merges", [])
        row_count = table_data.get("row_count", 0)
        col_count = table_data.get("col_count", len(headers))
        index = table_data.get("index", 0)
        extra_forbidden_rows = table_data.get("_extra_forbidden_rows", set())  # 嵌套表格行

        # 总行数 = 表头(1行) + 数据行
        total_data_rows = len(rows)  # 数据行数（不含表头）

        # 表格不够大，不切分
        if total_data_rows <= max_rows:
            return [table_data]

        # 步骤1：分析合并单元格，找出禁止切分点
        forbidden_splits = self._compute_forbidden_splits(merges, row_count)

        # 步骤2：加入嵌套表格行的禁止切分点
        # 嵌套表格行号是绝对行号（表头=0），需要转换为数据行索引
        for forbidden_row in extra_forbidden_rows:
            # forbidden_row 是绝对行号，数据行索引 = forbidden_row - 1（如果 forbidden_row > 0）
            # 禁止在 forbidden_row 和 forbidden_row+1 之间切分
            if forbidden_row > 0:
                forbidden_splits.add(forbidden_row)

        # 步骤3：计算允许切分的边界点
        all_possible_splits = set(range(1, total_data_rows + 1))
        allowed_splits = all_possible_splits - forbidden_splits

        if not allowed_splits:
            # 没有允许的切分点，不切分
            return [table_data]

        # 步骤4：按 max_rows 贪心切分，但只在 allowed_splits 点切分
        split_points = self._compute_split_points(total_data_rows, max_rows, allowed_splits)

        if not split_points:
            return [table_data]

        # 步骤5：按切分点切分，生成多个 table_data
        result = []
        prev_point = 0

        for i, split_point in enumerate(split_points):
            # 切分范围：数据行 [prev_point : split_point]
            chunk_rows = rows[prev_point:split_point]
            chunk_merges = self._adjust_merges_for_chunk(merges, prev_point, split_point)
            chunk_html = self._generate_chunk_html(headers, chunk_rows, chunk_merges, is_first=(i == 0))

            chunk_data = {
                "index": index,
                "headers": headers,
                "rows": chunk_rows,
                "merges": chunk_merges,
                "html": chunk_html,
                "row_count": len(chunk_rows) + 1,  # +1 for header
                "col_count": col_count,
                "table_part": i + 1,
                "table_total_parts": len(split_points) + 1,
                "original_row_range": (prev_point, split_point),  # 原始数据行范围
            }
            result.append(chunk_data)
            prev_point = split_point

        # 最后一个 chunk（剩余数据行）
        if prev_point < total_data_rows:
            chunk_rows = rows[prev_point:]
            chunk_merges = self._adjust_merges_for_chunk(merges, prev_point, total_data_rows)
            chunk_html = self._generate_chunk_html(headers, chunk_rows, chunk_merges, is_first=False)

            chunk_data = {
                "index": index,
                "headers": headers,
                "rows": chunk_rows,
                "merges": chunk_merges,
                "html": chunk_html,
                "row_count": len(chunk_rows) + 1,
                "col_count": col_count,
                "table_part": len(split_points) + 1,
                "table_total_parts": len(split_points) + 1,
                "original_row_range": (prev_point, total_data_rows),
            }
            result.append(chunk_data)

        return result

    def _compute_forbidden_splits(self, merges: list[dict], row_count: int) -> set[int]:
        """
        计算禁止切分点。

        合并单元格覆盖的行范围不能被切断。
        例如：row=1, rowspan=3 的单元格覆盖第1-3行（数据行），
        则不能在第1行之后、第2行之后切分（即禁止切分点为 {1, 2}）。

        注意：merges 中的 row 是绝对行号（表头=0，数据行=1,2,3...）
        返回的禁止切分点也是绝对行号。

        参数：
            merges: [{"row": int, "col": int, "rowspan": int, "colspan": int}, ...]
            row_count: 表格总行数

        返回：
            禁止切分点集合 {行号}，表示不能在该行号之后切分
        """
        forbidden = set()

        for merge in merges:
            row = merge.get("row", 0)
            rowspan = merge.get("rowspan", 1)

            if rowspan > 1:
                # 合并单元格覆盖 row 到 row+rowspan-1 行
                # 不能在 row, row+1, ..., row+rowspan-2 之后切分
                for r in range(row, row + rowspan - 1):
                    forbidden.add(r)

        return forbidden

    def _compute_split_points(self, total_rows: int, max_rows: int, allowed_splits: set[int]) -> list[int]:
        """
        计算切分点。

        贪心策略：尽量每 max_rows 行切分一次，但必须在 allowed_splits 点切分。

        参数：
            total_rows: 数据行总数
            max_rows: 每个 chunk 最大数据行数
            allowed_splits: 允许的切分点集合

        返回：
            切分点列表 [row_index, ...]，表示在这些数据行索引处切分
        """
        split_points = []
        current = 0

        while current < total_rows:
            # 目标切分点：current + max_rows
            target = min(current + max_rows, total_rows)

            # 如果目标点是允许的，直接切分
            if target in allowed_splits:
                split_points.append(target)
                current = target
            else:
                # 找最近的允许切分点
                # 向前找（小于目标）
                candidates = [s for s in allowed_splits if s > current and s <= target]
                if candidates:
                    # 取最大的允许点（尽量接近目标）
                    best = max(candidates)
                    split_points.append(best)
                    current = best
                else:
                    # 向后找（大于目标）
                    candidates = [s for s in allowed_splits if s > target]
                    if candidates:
                        # 取最小的允许点
                        best = min(candidates)
                        split_points.append(best)
                        current = best
                    else:
                        # 没有更多允许切分点，停止
                        break

        # 过滤掉最后一个切分点（如果等于 total_rows，那是结束点不是切分点）
        if split_points and split_points[-1] == total_rows:
            split_points.pop()

        return split_points

    def _adjust_merges_for_chunk(self, merges: list[dict], start_row: int, end_row: int) -> list[dict]:
        """
        调整合并单元格信息到切分后的 chunk。

        参数：
            merges: 原始合并信息
            start_row: 数据行起始索引（相对原始数据行）
            end_row: 数据行结束索引

        返回：
            切分后的合并信息（相对于 chunk 的数据行）
        """
        chunk_merges = []

        for merge in merges:
            row = merge.get("row", 0)
            rowspan = merge.get("rowspan", 1)
            col = merge.get("col", 0)
            colspan = merge.get("colspan", 1)

            # row 是绝对行号（表头=0）
            # 数据行从 1 开始，所以数据行索引 = row - 1
            data_row_idx = row - 1 if row > 0 else -1

            # 检查合并单元格是否在当前 chunk 范围内
            # 合并单元格覆盖数据行 data_row_idx 到 data_row_idx + rowspan - 1
            merge_start = data_row_idx
            merge_end = data_row_idx + rowspan

            if merge_start < 0:
                # 表头行的合并单元格，特殊处理
                # 表头行（row=0）的合并单元格应该在每个 chunk 都保留
                if row == 0 and rowspan > 1:
                    # 表头合并：检查是否完全在当前 chunk 内
                    # chunk 包含表头 + 数据行[start_row:end_row]
                    chunk_rowspan = min(rowspan, end_row - start_row + 1)
                    if chunk_rowspan > 1:
                        chunk_merges.append({
                            "row": row,  # 表头行号不变
                            "col": col,
                            "rowspan": chunk_rowspan,
                            "colspan": colspan,
                        })
                continue

            # 检查合并是否跨越当前 chunk
            if merge_start >= end_row:
                # 合并单元格在 chunk 之后，跳过
                continue
            if merge_end <= start_row:
                # 合并单元格在 chunk 之前，跳过
                continue

            # 合并单元格至少部分在当前 chunk 内
            # 计算在 chunk 内的有效范围
            effective_start = max(merge_start, start_row)
            effective_end = min(merge_end, end_row)

            # 新的 rowspan
            new_rowspan = effective_end - effective_start

            # 新的 row（相对于 chunk，数据行从 1 开始）
            # chunk 内：表头=第0行，数据行从第1行开始
            # 数据行索引 effective_start - start_row 是 chunk 内的相对索引
            # 转换为绝对行号：表头是第0行，第一个数据行是第1行
            new_row = (effective_start - start_row) + 1

            if new_rowspan > 1:
                chunk_merges.append({
                    "row": new_row,
                    "col": col,
                    "rowspan": new_rowspan,
                    "colspan": colspan,
                })

        return chunk_merges

    def _generate_chunk_html(self, headers: list[str], rows: list[list[str]], merges: list[dict], is_first: bool) -> str:
        """
        为切分后的 chunk 生成 HTML。

        参数：
            headers: 表头列表
            rows: 数据行列表
            merges: 合并单元格信息（已调整）
            is_first: 是否是第一个 chunk（第一个 chunk 用 <th>，其他用 <td>）

        返回：
            HTML 表格字符串
        """
        # 构建合并单元格映射 {(row, col): (rowspan, colspan)}
        merge_map = {}
        for m in merges:
            merge_map[(m["row"], m["col"])] = (m["rowspan"], m["colspan"])

        # 跳过单元格集合
        skip_cells = set()
        for (row, col), (rowspan, colspan) in merge_map.items():
            for r in range(row, row + rowspan):
                for c in range(col, col + colspan):
                    if (r, c) != (row, col):
                        skip_cells.add((r, c))

        # 构建 HTML
        html_lines = ["<table>"]

        # 表头行（row=0）
        col_count = len(headers)
        html_lines.append("<tr>")
        col_idx = 0
        for header_text in headers:
            while (0, col_idx) in skip_cells and col_idx < col_count:
                col_idx += 1
            if col_idx >= col_count:
                break

            header_text = str(header_text).replace("\n", "<br>").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            if (0, col_idx) in merge_map:
                rowspan, colspan = merge_map[(0, col_idx)]
                attrs = []
                if rowspan > 1:
                    attrs.append(f'rowspan="{rowspan}"')
                if colspan > 1:
                    attrs.append(f'colspan="{colspan}"')
                attr_str = " " + " ".join(attrs) if attrs else ""
                html_lines.append(f"<th{attr_str}>{header_text}</th>")
                col_idx += colspan
            else:
                html_lines.append(f"<th>{header_text}</th>")
                col_idx += 1
        html_lines.append("</tr>")

        # 数据行（row=1, 2, 3...）
        for row_idx, row_data in enumerate(rows):
            html_row_idx = row_idx + 1  # HTML 行号（表头=0，数据行从1开始）
            html_lines.append("<tr>")
            col_idx = 0
            for cell_text in row_data:
                while (html_row_idx, col_idx) in skip_cells and col_idx < col_count:
                    col_idx += 1
                if col_idx >= col_count:
                    break

                cell_text = str(cell_text).replace("\n", "<br>").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

                if (html_row_idx, col_idx) in merge_map:
                    rowspan, colspan = merge_map[(html_row_idx, col_idx)]
                    attrs = []
                    if rowspan > 1:
                        attrs.append(f'rowspan="{rowspan}"')
                    if colspan > 1:
                        attrs.append(f'colspan="{colspan}"')
                    attr_str = " " + " ".join(attrs) if attrs else ""
                    html_lines.append(f"<td{attr_str}>{cell_text}</td>")
                    col_idx += colspan
                else:
                    html_lines.append(f"<td>{cell_text}</td>")
                    col_idx += 1
            html_lines.append("</tr>")

        html_lines.append("</table>")
        return "\n".join(html_lines)