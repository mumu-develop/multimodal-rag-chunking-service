"""通用转换工具，为所有文件类型提供统一的输出格式。

设计原则：
- **健壮性**：降级策略，失败时返回空数据而非异常
- **可维护性**：单一职责，清晰的函数边界
- **可扩展性**：支持自定义表格解析器

统一输出结构：
{
    "markdown": str,           # Markdown 文本
    "tables": list[dict],      # 表格数据（必须含 html 字段）
    "images": list[dict],      # 图片数据
    "code_blocks": list[dict], # 代码块数据
}

所有表格必须有 html 字段，前端统一渲染。
"""
import re
import html
import structlog
from typing import Any, Callable, Optional

from app.converters.models import TableData, TableParsingError

logger = structlog.get_logger()


def build_table_html(
    headers: list[str],
    rows: list[list[str]],
    merges: list[dict] = None,
    class_name: str = "chunk-table",
    escape_content: bool = True,
) -> str:
    """
    通用 HTML 表格生成器（健壮版）。

    降级策略：
    - 空 headers/rows → 返回空字符串
    - 单元格内容为 None → 转为空字符串
    - 合并单元格超出范围 → 忽略该合并

    Args:
        headers: 表头列表
        rows: 数据行列表
        merges: 合并单元格信息 [{"row": int, "col": int, "rowspan": int, "colspan": int}, ...]
        class_name: CSS 类名
        escape_content: 是否转义 HTML 特殊字符（安全默认 True）

    Returns:
        HTML 表格字符串（空表格返回空字符串）
    """
    # 降级：空表格
    if not headers and not rows:
        logger.debug("empty_table_skipped")
        return ""

    merges = merges or []

    # 构建合并单元格映射，验证边界
    merge_map = {}
    for m in merges:
        row = m.get("row", 0)
        col = m.get("col", 0)
        rowspan = m.get("rowspan", 1)
        colspan = m.get("colspan", 1)

        # 验证合并不超出表格范围
        max_row = len(rows) if rows else 0
        max_col = max(len(headers), max(len(r) for r in rows) if rows else 0)

        if row < 0 or col < 0 or row + rowspan > max_row + 1 or col + colspan > max_col:
            logger.warning("merge_out_of_bounds", merge=m, max_row=max_row, max_col=max_col)
            continue

        merge_map[(row, col)] = (rowspan, colspan)

    # 跳过单元格集合（作为合并的一部分被跳过）
    skip_cells = set()
    for (row, col), (rowspan, colspan) in merge_map.items():
        for r in range(row, row + rowspan):
            for c in range(col, col + colspan):
                if (r, c) != (row, col):
                    skip_cells.add((r, c))

    # 构建 HTML
    html_lines = [f'<table class="{class_name}">']

    # 表头行（row=0）
    if headers:
        html_lines.append("<thead><tr>")
        col_idx = 0
        col_count = len(headers)
        for header_text in headers:
            # 跳过被合并的单元格
            while (0, col_idx) in skip_cells and col_idx < col_count:
                col_idx += 1
            if col_idx >= col_count:
                break

            header_text = _safe_cell_text(header_text, escape_content)

            if (0, col_idx) in merge_map:
                rowspan, colspan = merge_map[(0, col_idx)]
                attrs = _build_cell_attrs(rowspan, colspan)
                html_lines.append(f"<th{attrs}>{header_text}</th>")
                col_idx += colspan
            else:
                html_lines.append(f"<th>{header_text}</th>")
                col_idx += 1
        html_lines.append("</tr></thead>")

    # 数据行（row=1, 2, 3...）
    if rows:
        html_lines.append("<tbody>")
        for row_idx, row_data in enumerate(rows):
            html_row_idx = row_idx + 1  # HTML 行号（表头=0，数据行从1开始）
            html_lines.append("<tr>")
            col_idx = 0
            col_count = len(row_data) if row_data else 0
            for cell_text in row_data:
                # 跳过被合并的单元格
                while (html_row_idx, col_idx) in skip_cells and col_idx < col_count:
                    col_idx += 1
                if col_idx >= col_count:
                    break

                cell_text = _safe_cell_text(cell_text, escape_content)

                if (html_row_idx, col_idx) in merge_map:
                    rowspan, colspan = merge_map[(html_row_idx, col_idx)]
                    attrs = _build_cell_attrs(rowspan, colspan)
                    html_lines.append(f"<td{attrs}>{cell_text}</td>")
                    col_idx += colspan
                else:
                    html_lines.append(f"<td>{cell_text}</td>")
                    col_idx += 1
            html_lines.append("</tr>")
        html_lines.append("</tbody>")

    html_lines.append("</table>")
    return "\n".join(html_lines)


def parse_markdown_table(table_text: str, index: int = 0) -> TableData:
    """
    解析 Markdown 表格文本，转换为结构化数据（健壮版）。

    降级策略：
    - 无效表格 → 返回空 TableData（而非异常）
    - 列数不一致 → 按第一行列数填充空值

    Args:
        table_text: Markdown 表格文本（| 语法）
        index: 表格索引

    Returns:
        TableData 模型（无效表格返回空数据）
    """
    try:
        lines = [l.strip() for l in table_text.split("\n") if l.strip()]

        if len(lines) < 2:
            logger.debug("markdown_table_invalid", reason="less_than_2_lines", lines=len(lines))
            return TableData(index=index, headers=[], rows=[], html="", row_count=0, col_count=0)

        # 解析表头
        headers = [cell.strip() for cell in lines[0].split("|") if cell.strip()]
        if not headers:
            logger.debug("markdown_table_invalid", reason="empty_headers")
            return TableData(index=index, headers=[], rows=[], html="", row_count=0, col_count=0)

        # 验证分隔行
        separator = lines[1]
        if not all(c in "|-: " for c in separator) or separator.count("|") < 2:
            logger.debug("markdown_table_invalid", reason="invalid_separator", separator=separator[:20])
            return TableData(index=index, headers=[], rows=[], html="", row_count=0, col_count=0)

        # 跳过分隔行，解析数据行
        rows = []
        col_count = len(headers)
        for line in lines[2:]:
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            if cells:
                # 列数不一致时，补齐或截断
                if len(cells) < col_count:
                    cells.extend([""] * (col_count - len(cells)))
                elif len(cells) > col_count:
                    cells = cells[:col_count]
                rows.append(cells)

        # 生成 HTML
        html_str = build_table_html(headers, rows)

        logger.debug("markdown_table_parsed", index=index, headers=len(headers), rows=len(rows))

        return TableData(
            index=index,
            headers=headers,
            rows=rows,
            merges=[],  # Markdown 表格不支持单元格合并
            html=html_str,
            row_count=len(rows) + 1,
            col_count=len(headers),
        )
    except Exception as e:
        logger.warning("markdown_table_parse_failed", error=str(e), index=index)
        return TableData(index=index, headers=[], rows=[], html="", row_count=0, col_count=0)


def extract_markdown_tables(
    markdown_text: str,
    exclude_code_blocks: bool = True,
    on_table_parsed: Optional[Callable[[TableData], None]] = None,
) -> tuple[list[TableData], str]:
    """
    从 Markdown 文本中提取所有表格，替换为 [TABLE_idx] 标记（健壮版）。

    降级策略：
    - 代码块内的表格 → 不解析（避免误识别）
    - 无效表格 → 不记录，保留原文本

    Args:
        markdown_text: Markdown 文本
        exclude_code_blocks: 是否排除代码块内的表格
        on_table_parsed: 回调函数（用于自定义处理）

    Returns:
        (tables_data, modified_markdown): 表格数据列表和替换后的 Markdown 文本
    """
    tables_data: list[TableData] = []
    lines = markdown_text.split("\n")

    # 找出代码块范围（如果需要排除）
    code_block_ranges: list[tuple[int, int]] = []
    if exclude_code_blocks:
        in_code_block = False
        code_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                if in_code_block:
                    code_block_ranges.append((code_start, i))
                    in_code_block = False
                else:
                    code_start = i
                    in_code_block = True
        # 未闭合的代码块（文档末尾）
        if in_code_block:
            code_block_ranges.append((code_start, len(lines)))

    # 找出所有表格的位置
    table_ranges: list[tuple[int, int, list[str]]] = []

    current_table_start: int | None = None
    current_table_lines: list[str] = []

    for i, line in enumerate(lines):
        # 检查是否在代码块内
        in_code = any(start <= i <= end for start, end in code_block_ranges)
        if in_code:
            if current_table_lines:
                # 保存当前收集的表格
                if _is_valid_markdown_table(current_table_lines):
                    table_ranges.append((current_table_start, i, current_table_lines.copy()))
            current_table_start = None
            current_table_lines = []
            continue

        is_table_line = _is_markdown_table_line(line)

        if is_table_line:
            if current_table_start is None:
                current_table_start = i
            current_table_lines.append(line)
        else:
            if current_table_lines and _is_valid_markdown_table(current_table_lines):
                table_ranges.append((current_table_start, i, current_table_lines.copy()))
            current_table_start = None
            current_table_lines = []

    # 处理文档末尾的表格
    if current_table_lines and _is_valid_markdown_table(current_table_lines):
        table_ranges.append((current_table_start, len(lines), current_table_lines.copy()))

    # 构建表格数据并替换原文本
    for idx, (start, end, table_lines) in enumerate(table_ranges):
        table_content = "\n".join(table_lines)
        table_data = parse_markdown_table(table_content, index=idx)

        # 只保存有效表格（有内容）
        if table_data.is_valid():
            tables_data.append(table_data)
            if on_table_parsed:
                on_table_parsed(table_data)

            # 替换表格行为标记
            for i in range(start, end):
                if i == start:
                    lines[i] = f"\n[TABLE_{idx}]\n"
                else:
                    lines[i] = ""  # 清空其他表格行

    modified_markdown = "\n".join(lines)
    logger.debug("markdown_tables_extracted", count=len(tables_data))

    return tables_data, modified_markdown


def extract_code_blocks(markdown_text: str) -> tuple[list[dict], str]:
    """
    从 Markdown 文本中提取所有代码块（健壮版）。

    Args:
        markdown_text: Markdown 文本

    Returns:
        (code_blocks_data, markdown_text): 代码块数据列表和原文本
    """
    code_blocks: list[dict] = []
    # 宽松匹配：```language\ncode\n``` 或 ```\ncode\n```
    pattern = r"```(\w*)?\s*\n(.*?)\n```"

    try:
        for idx, match in enumerate(re.finditer(pattern, markdown_text, re.DOTALL)):
            language = match.group(1) or ""
            content = match.group(2) or ""
            if content.strip():
                code_blocks.append({
                    "index": idx,
                    "content": content,
                    "language": language,
                })
    except Exception as e:
        logger.warning("code_blocks_extract_failed", error=str(e))

    return code_blocks, markdown_text


# === 内部辅助函数 ===


def _safe_cell_text(text: Any, escape: bool = True) -> str:
    """安全处理单元格文本。

    - None → ""
    - 非 str → str()
    - 转义 HTML 特殊字符
    """
    if text is None:
        return ""
    text_str = str(text)
    if escape:
        text_str = html.escape(text_str).replace("\n", "<br>")
    return text_str


def _build_cell_attrs(rowspan: int, colspan: int) -> str:
    """构建单元格属性字符串。"""
    attrs = []
    if rowspan > 1:
        attrs.append(f'rowspan="{rowspan}"')
    if colspan > 1:
        attrs.append(f'colspan="{colspan}"')
    return " " + " ".join(attrs) if attrs else ""


def _is_markdown_table_line(line: str) -> bool:
    """判断一行是否是 Markdown 表格行。"""
    stripped = line.strip()
    if not stripped:
        return False
    # 以 | 开头和结尾
    if stripped.startswith("|") and stripped.endswith("|"):
        return True
    # 表格分隔行：全是 | - : 空格
    if all(c in "|-: " for c in stripped):
        return stripped.count("|") >= 2
    return False


def _is_valid_markdown_table(lines: list[str]) -> bool:
    """判断行列表是否构成有效的 Markdown 表格。"""
    if len(lines) < 2:
        return False
    if not lines[0].strip().startswith("|"):
        return False
    separator = lines[1].strip()
    if not all(c in "|-: " for c in separator):
        return False
    return separator.count("|") >= 2