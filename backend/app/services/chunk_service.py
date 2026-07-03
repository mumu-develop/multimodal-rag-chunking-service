"""Service: Chunk listing and document tree."""
from typing import Optional

import structlog

from app.repositories.chunk_repo import chunk_repo

logger = structlog.get_logger()


class ChunkService:
    """Chunk 查询与文档树服务。"""

    def list_chunks(
        self,
        doc_id: str,
        page: int = 1,
        size: int = 10,
        chunk_type: Optional[str] = None,
    ) -> tuple[int, list[dict]]:
        """分页查询 chunk。"""
        total, items = chunk_repo.get_by_doc(doc_id, page, size, chunk_type)

        result = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "chunk_type": c.chunk_type,
                "content": c.content,
                "caption": c.caption,
                "page_num": c.page_num,
                "heading_level": c.heading_level or 0,
                "heading_text": c.heading_text or "",
                "sort_order": c.sort_order or 0,
                "metadata": c.meta_data,
                "status": c.status or 1,
                "audit_status": c.audit_status or 0,
                "creator": c.creator,
                "doc_type": c.meta_data.get("doc_type", "") if c.meta_data else "",
            }
            for c in items
        ]
        return total, result

    def get_doc_tree(self, doc_id: str) -> Optional[dict]:
        """获取文档的层级树结构（原有 TreeNode 格式）。"""
        chunks = chunk_repo.get_all_by_doc(doc_id)
        if not chunks:
            return None

        doc_type = chunks[0].meta_data.get("doc_type", "") if chunks[0].meta_data else ""
        tree = self._build_tree(chunks)

        return {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "tree": tree,
        }

    def get_production_doc_tree(self, doc_id: str) -> Optional[dict]:
        """获取生产系统格式的文档树（Paragraph + Chunks 双层结构）。"""
        chunks = chunk_repo.get_all_by_doc(doc_id)
        if not chunks:
            return None

        doc_type = chunks[0].meta_data.get("doc_type", "") if chunks[0].meta_data else ""
        paragraphs = self._build_production_tree(chunks)

        return {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "paragraphs": paragraphs,
        }

    @staticmethod
    def _build_tree(chunks) -> list[dict]:
        """从 chunk 列表构建嵌套树结构。

        原则：
        - heading_level > 0 的 chunk 代表标题节点，是结构节点
        - heading_level == 0 的 chunk 是正文内容，作为最近标题的子内容
        """
        root: list[dict] = []
        stack: list[tuple[int, dict]] = []  # (level, node) 栈只存储标题节点

        for c in chunks:
            level = c.heading_level or 0
            node = _chunk_to_dict(c)
            node["children"] = []  # 预初始化 children

            if level == 0:
                # 正文内容：添加到栈顶标题的 children
                if stack:
                    stack[-1][1]["children"].append(node)
                else:
                    root.append(node)
            else:
                # 标题节点：弹出栈中同级或更高级的标题
                while stack and stack[-1][0] >= level:
                    stack.pop()

                # 标题节点添加到父节点或 root
                if stack:
                    stack[-1][1]["children"].append(node)
                else:
                    root.append(node)

                # 标题节点入栈（作为后续正文的父节点）
                stack.append((level, node))

        # 清理空 children
        def _clean_empty_children(nodes):
            for n in nodes:
                if "children" in n:
                    _clean_empty_children(n["children"])
                    if not n["children"]:
                        del n["children"]
            return nodes

        _clean_empty_children(root)
        return root

    @staticmethod
    def _build_production_tree(chunks) -> list[dict]:
        """构建生产系统的 Paragraph + Chunks 双层结构。

        算法：
        1. 遍历所有 chunks
        2. heading_level > 0 的 chunk 创建 Paragraph 节点，同时将 chunk 内容添加到 chunks[]
        3. heading_level == 0 的 chunk 归入最近 Paragraph 的 chunks[]
        4. 使用栈维护 Paragraph 的层级关系
        5. 为每个 chunk 生成 previousId（基于 sort_order）
        """
        root: list[dict] = []
        stack: list[tuple[int, dict]] = []  # (level, paragraph_node)

        # 用于生成 paragraphId
        def _make_paragraph_id(level: int, text: str, idx: int) -> str:
            return f"p-{level}-{idx}-{hash(text) % 10000}"

        paragraph_counter = 0

        for c in chunks:
            level = c.heading_level or 0

            if level > 0:
                # 创建 Paragraph 节点
                paragraph_counter += 1
                paragraph_id = _make_paragraph_id(level, c.heading_text or "", paragraph_counter)

                paragraph = {
                    "id": paragraph_id,
                    "title": c.heading_text or "",
                    "level": level,
                    "content": c.heading_text or "",
                    "childContentParagraph": [],
                    "chunks": [],
                }

                # 弹出栈中 >= 当前 level 的节点
                while stack and stack[-1][0] >= level:
                    stack.pop()

                # 添加到父节点或 root
                if stack:
                    stack[-1][1]["childContentParagraph"].append(paragraph)
                else:
                    root.append(paragraph)

                # 入栈作为后续 chunk 的父节点
                stack.append((level, paragraph))

                # 重要：标题 chunk 本身的内容也要添加到 chunks[] 中
                # 这样 Paragraph.chunks 就不会为空
                if c.content and len(c.content.strip()) >= 10:
                    chunk_item = {
                        "id": c.chunk_id,
                        "paragraphId": paragraph_id,
                        "content": c.content,
                        "status": c.status or 1,
                        "auditStatus": c.audit_status or 0,
                        "previousId": None,
                        "matchCount": 0,
                    }
                    paragraph["chunks"].append(chunk_item)

            else:
                # 正文 chunk：添加到栈顶 Paragraph 的 chunks[]
                chunk_item = {
                    "id": c.chunk_id,
                    "paragraphId": stack[-1][1]["id"] if stack else "",
                    "content": c.content,
                    "status": c.status or 1,
                    "auditStatus": c.audit_status or 0,
                    "previousId": None,  # 后续填充
                    "matchCount": 0,
                }

                if stack:
                    # 添加 previousId：取该 Paragraph chunks 列表中最后一个的 id
                    existing_chunks = stack[-1][1]["chunks"]
                    if existing_chunks:
                        chunk_item["previousId"] = existing_chunks[-1]["id"]
                    stack[-1][1]["chunks"].append(chunk_item)
                else:
                    # 无父 Paragraph，直接加入 root（特殊情况）
                    root.append(chunk_item)

        return root


def _chunk_to_dict(c) -> dict:
    """将 ORM 对象转为 dict（用于树节点）。"""
    return {
        "heading_level": c.heading_level or 0,
        "heading_text": c.heading_text or "",
        "chunk_id": c.chunk_id,
        "chunk_type": c.chunk_type,
        "content": c.content,
        "caption": c.caption,
        "page_num": c.page_num,
        "metadata": c.meta_data,
        "status": c.status or 1,
        "audit_status": c.audit_status or 0,
        "creator": c.creator,
        "doc_type": c.meta_data.get("doc_type", "") if c.meta_data else "",
    }


# 单例
chunk_service = ChunkService()
