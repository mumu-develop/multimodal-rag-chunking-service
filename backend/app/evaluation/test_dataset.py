"""测试集构建工具：支持人工标注导入、反向标注、LLM 辅助标注。"""

import json
import os

import structlog

from app.evaluation.rag_evaluator import TestCase
from app.services.search_service import search_service

logger = structlog.get_logger()


class TestDatasetBuilder:
    """RAG 检索评估测试集构建器。

    支持三种数据来源：
    1. 人工标注 — 从 JSON/CSV 文件加载
    2. 反向标注 — 从已有高质量搜索结果反推
    3. LLM 辅助标注 — 调用 LLM 生成候选片段，人工校验

    用法:
        builder = TestDatasetBuilder()

        # 方式一：从 JSON 文件加载人工标注
        cases = builder.load_from_json("tests/eval/manual_annotations.json")

        # 方式二：反向标注（从已知好结果反推）
        cases = builder.reverse_annotate(doc_id="xxx", sample_queries=[...])

        # 方式三：LLM 辅助标注
        cases = builder.llm_assist(doc_id="xxx", num_queries=50)

        # 合并 + 去重 + 导出
        builder.merge(cases)
        builder.export("tests/eval/test_cases.json")
    """

    def __init__(self):
        self.cases: list[TestCase] = []

    def load_from_json(self, file_path: str) -> list[TestCase]:
        """
        从 JSON 文件加载人工标注。

        JSON 格式:
        [
            {
                "query": "公司 2024 年营收多少？",
                "relevant_chunk_ids": ["chunk_001", "chunk_042"],
                "doc_id": "doc_abc123",
                "tags": ["财务", "数字查询"]
            },
            ...
        ]
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cases = []
        for item in data:
            cases.append(TestCase(
                query=item["query"],
                relevant_chunk_ids=item.get("relevant_chunk_ids", []),
                doc_id=item.get("doc_id", ""),
                tags=item.get("tags", []),
            ))

        logger.info("test_dataset_loaded_from_json", path=file_path, count=len(cases))
        self.cases.extend(cases)
        return cases

    def load_from_csv(self, file_path: str) -> list[TestCase]:
        """
        从 CSV 文件加载标注（无表头或带表头均可）。

        CSV 列顺序: query, relevant_chunk_ids, doc_id, tags
        - relevant_chunk_ids 用分号分隔
        - tags 用逗号分隔
        """
        import csv

        cases = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                # 跳过表头
                if row[0].strip().lower() == "query":
                    continue
                relevant = [x.strip() for x in row[1].split(";") if x.strip()]
                doc_id = row[2].strip() if len(row) > 2 else ""
                tags = [t.strip() for t in row[3].split(",") if t.strip()] if len(row) > 3 else []
                cases.append(TestCase(
                    query=row[0].strip(),
                    relevant_chunk_ids=relevant,
                    doc_id=doc_id,
                    tags=tags,
                ))

        logger.info("test_dataset_loaded_from_csv", path=file_path, count=len(cases))
        self.cases.extend(cases)
        return cases

    def reverse_annotate(
        self,
        doc_id: str,
        sample_queries: list[str],
        top_k: int = 5,
    ) -> list[TestCase]:
        """
        反向标注：对一组 query 执行搜索，将 top_k 结果标记为"相关"。

        适用场景：已有已知高质量的搜索结果，反推相关 chunk_id。

        Args:
            doc_id: 目标文档 ID
            sample_queries: 查询列表
            top_k: 前 K 个结果视为相关

        Returns:
            标注好的 TestCase 列表
        """
        cases = []
        for query in sample_queries:
            hits = search_service.search(query=query, top_k=top_k, doc_ids=[doc_id])
            chunk_ids = [h["chunk"]["chunk_id"] for h in hits]
            cases.append(TestCase(
                query=query,
                relevant_chunk_ids=chunk_ids,
                doc_id=doc_id,
                tags=["reverse_annotated"],
            ))

        logger.info(
            "reverse_annotation_done",
            doc_id=doc_id,
            query_count=len(cases),
            avg_relevant=sum(len(c.relevant_chunk_ids) for c in cases) / max(len(cases), 1),
        )
        return cases

    def llm_assist(
        self,
        doc_id: str,
        query_file: str,
        top_k: int = 10,
        output_file: str | None = None,
    ) -> list[TestCase]:
        """
        LLM 辅助标注：对一组 query 检索候选 chunk，输出 JSON 供人工校验。

        工作流程：
        1. 对每个 query 执行搜索，取 top_k 结果
        2. 输出 JSON 文件，包含 query + 候选 chunk 内容
        3. 人工审阅 JSON，将相关 chunk_id 填入 relevant_chunk_ids
        4. 重新 load_from_json 导入

        Args:
            doc_id: 目标文档 ID
            query_file: 包含查询列表的 JSON 文件路径，格式: [{"query": "...", "tags": [...]}, ...]
            top_k: 每个 query 取 top_k 个候选
            output_file: 输出的待标注 JSON 文件路径

        Returns:
            候选 TestCase 列表（relevant_chunk_ids 为空，需人工填写）
        """
        with open(query_file, "r", encoding="utf-8") as f:
            queries_data = json.load(f)

        candidates = []
        for item in queries_data:
            query = item["query"]
            hits = search_service.search(query=query, top_k=top_k, doc_ids=[doc_id])

            candidates.append(TestCase(
                query=query,
                relevant_chunk_ids=[],  # 待人工填写
                doc_id=doc_id,
                tags=item.get("tags", ["llm_assist_candidate"]),
            ))

        # 输出待标注文件
        annotate_data = []
        for case in candidates:
            hits = search_service.search(query=case.query, top_k=top_k, doc_ids=[doc_id])
            annotate_data.append({
                "query": case.query,
                "tags": case.tags,
                "doc_id": doc_id,
                "candidates": [
                    {
                        "chunk_id": h["chunk"]["chunk_id"],
                        "content": h["chunk"]["content"][:200],
                        "page_num": h["chunk"]["page_num"],
                        "score": h["score"],
                    }
                    for h in hits
                ],
                "relevant_chunk_ids": [],  # <-- 人工填写这里
            })

        out_path = output_file or f"/tmp/llm_annotate_{doc_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(annotate_data, f, ensure_ascii=False, indent=2)

        logger.info(
            "llm_assist_candidates_generated",
            query_count=len(candidates),
            output_path=out_path,
        )
        return candidates

    # ============================================================
    # 合并、去重、导出
    # ============================================================

    def merge(self, new_cases: list[TestCase]) -> None:
        """
        合并新用例到已有数据集。

        去重规则：如果 query 文本完全相同，用新的替换旧的。
        """
        existing_queries = {tc.query: i for i, tc in enumerate(self.cases)}
        for nc in new_cases:
            if nc.query in existing_queries:
                self.cases[existing_queries[nc.query]] = nc
            else:
                self.cases.append(nc)

    def deduplicate(self) -> int:
        """
        基于 query 文本去重。

        Returns:
            删除的重复条目数
        """
        seen: set[str] = set()
        deduped: list[TestCase] = []
        removed = 0
        for tc in self.cases:
            if tc.query in seen:
                removed += 1
            else:
                seen.add(tc.query)
                deduped.append(tc)
        self.cases = deduped
        return removed

    def split(self, ratio: float = 0.8, seed: int = 42) -> tuple[list[TestCase], list[TestCase]]:
        """
        将测试集划分为训练集和验证集。

        Args:
            ratio: 训练集占比
            seed: 随机种子

        Returns:
            (train_cases, val_cases)
        """
        import random
        random.seed(seed)
        shuffled = list(self.cases)
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * ratio)
        return shuffled[:split_idx], shuffled[split_idx:]

    def export(self, output_path: str) -> int:
        """
        将测试集导出为 JSON 文件。

        Returns:
            导出的用例数量
        """
        data = []
        for tc in self.cases:
            data.append({
                "query": tc.query,
                "relevant_chunk_ids": tc.relevant_chunk_ids,
                "doc_id": tc.doc_id,
                "tags": tc.tags,
            })

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("test_dataset_exported", path=output_path, count=len(data))
        return len(data)

    def stats(self) -> dict:
        """返回测试集统计信息。"""
        if not self.cases:
            return {"total": 0}

        tag_counts: dict[str, int] = {}
        doc_counts: dict[str, int] = {}
        avg_relevant = 0.0

        for tc in self.cases:
            for tag in (tc.tags if tc.tags else ["_untagged_"]):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            if tc.doc_id:
                doc_counts[tc.doc_id] = doc_counts.get(tc.doc_id, 0) + 1
            avg_relevant += len(tc.relevant_chunk_ids)

        avg_relevant /= len(self.cases)

        return {
            "total": len(self.cases),
            "avg_relevant_chunks": round(avg_relevant, 2),
            "by_tag": tag_counts,
            "by_doc": doc_counts,
        }
