"""RAGAS 合成测试集生成器：从知识库自动生成 query + ground_truth 测试集。

利用 ragas 的 TestsetGenerator，将文档内容转换为评估测试用例，
无需人工编写问题。支持按 query 类型（简单/复杂/推理/摘要等）生成多样化问题。
"""

import time

import structlog

logger = structlog.get_logger()


class DashScopeEmbedding:
    """基于 DashScope 原生 SDK 的 Embedding 封装，供 ragas TestsetGenerator 使用。

    支持通过 DashScope SDK 调用 qwen3-vl-embedding 等模型，
    绕过 OpenAI 兼容端点不兼容的问题。
    """

    def __init__(self, config: dict):
        self._model = config.get("model_name", "qwen3-vl-embedding")
        self._api_key = config.get("api_key", "")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """嵌入多个文本，返回向量列表。"""
        import dashscope
        vectors = []
        for text in texts:
            resp = dashscope.MultiModalEmbedding.call(
                api_key=self._api_key,
                model=self._model,
                input=[{"text": text}],
            )
            if resp.status_code != 200 or resp.output is None:
                raise RuntimeError(
                    f"DashScope embedding failed: code={resp.status_code}, message={resp.message}"
                )
            vectors.append(resp.output["embeddings"][0]["embedding"])
        return vectors

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """异步嵌入多个文本。DashScope SDK 无原生异步，用线程执行。"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> list[float]:
        result = await self.aembed_documents([text])
        return result[0]


class RagasTestsetGenerator:
    """基于 ragas 的合成测试集生成器。

    工作流程:
    1. 从 MySQL/Milvus 获取目标文档的全部 chunk
    2. 将 chunk 转换为 langchain Document 对象
    3. 调用 ragas TestsetGenerator 生成 query + ground_truth
    4. 导出为项目标准的 RagasTestCase 格式

    用法:
        generator = RagasTestsetGenerator(llm_config={...})
        test_cases = generator.generate_from_doc(
            doc_id="doc_abc123",
            testset_size=20,
        )
    """

    def __init__(self, llm_config: dict | None = None, embed_config: dict | None = None, use_dashscope: bool = True):
        """
        Args:
            llm_config: LLM 配置
                - model_name: 模型名
                - api_key: API Key
                - base_url: API 端点（OpenAI 兼容格式）
            embed_config: Embedding 模型配置
                - model_name: embedding 模型名
                - api_key: API Key
                - base_url: API 端点
            use_dashscope: 是否使用 DashScope 原生 SDK 调用 embedding（支持 qwen3-vl-embedding）
        """
        from app.config import settings
        self._llm_config = llm_config or self._default_llm_config()
        self._embed_config = embed_config or self._default_embed_config()
        self._use_dashscope = use_dashscope
        self._generator = None  # lazy init

    def generate_from_doc(
        self,
        doc_id: str,
        testset_size: int = 20,
        chunk_type_filter: str | None = None,
    ) -> list[dict]:
        """从指定文档生成测试集。

        Args:
            doc_id: 目标文档 ID
            testset_size: 生成测试用例数量
            chunk_type_filter: 可选，按 chunk_type 过滤（text/table/code/image）

        Returns:
            [
                {
                    "query": "公司2024年营收多少？",
                    "ground_truth": "公司2024年营收100亿元",
                    "doc_id": "doc_abc123",
                    "tags": ["simple_query"],
                },
                ...
            ]
        """
        generator = self._get_generator()

        # 获取文档 chunk（返回的是 DocChunk ORM 对象）
        from app.repositories.chunk_repo import chunk_repo
        chunks = chunk_repo.get_all_by_doc(doc_id)

        if not chunks:
            return []

        # 过滤 chunk 类型
        if chunk_type_filter:
            chunks = [c for c in chunks if getattr(c, "chunk_type", None) == chunk_type_filter]

        # 仅保留 text/table/code 类型（纯文本适合生成问题）
        chunks = [c for c in chunks if getattr(c, "chunk_type", None) in ("text", "table", "code")]

        if not chunks:
            return []

        # 转换为 langchain Document
        from langchain_core.documents import Document as LCDocument

        lc_docs = []
        for c in chunks:
            content = getattr(c, "content", "") or ""
            if len(content.strip()) < 50:
                continue  # 跳过太短的 chunk
            lc_docs.append(
                LCDocument(
                    page_content=content,
                    metadata={
                        "chunk_id": getattr(c, "chunk_id", ""),
                        "doc_id": doc_id,
                        "chunk_type": getattr(c, "chunk_type", ""),
                        "page_num": getattr(c, "page_num", 0),
                    },
                )
            )

        if len(lc_docs) < 3:
            logger.warning("ragas_testset_too_few_docs", doc_id=doc_id, count=len(lc_docs))
            return []

        logger.info(
            "ragas_testset_generating",
            doc_id=doc_id,
            doc_count=len(lc_docs),
            testset_size=testset_size,
        )

        t0 = time.monotonic()

        # 生成测试集
        from ragas.run_config import RunConfig
        from app.config import settings
        run_config = RunConfig(max_workers=settings.EVAL_MAX_WORKERS, timeout=180)  # 统一使用配置的并发数，180s 超时

        testset = generator.generate_with_langchain_docs(
            documents=lc_docs,
            testset_size=testset_size,
            raise_exceptions=True,
            run_config=run_config,
        )

        latency_ms = (time.monotonic() - t0) * 1000
        logger.info("ragas_testset_generated", latency_ms=round(latency_ms, 2), count=len(testset.samples))

        # 转换为项目格式
        test_cases = []
        for sample in testset.samples:
            tc = {
                "query": getattr(sample, "user_input", "") or "",
                "ground_truth": getattr(sample, "reference", "") or "",
                "doc_id": doc_id,
                "tags": [],
            }
            # 添加 persona/query_style 作为 tag
            persona = getattr(sample, "persona_name", None)
            if persona:
                tc["tags"].append(persona)
            query_style = getattr(sample, "query_style", None)
            if query_style:
                tc["tags"].append(query_style)

            # 跳过没有有效内容的
            if tc["query"] and tc["ground_truth"]:
                test_cases.append(tc)

        return test_cases

    def generate_from_multiple_docs(
        self,
        doc_ids: list[str],
        testset_size_per_doc: int = 10,
        chunk_type_filter: str | None = None,
    ) -> list[dict]:
        """从多个文档生成测试集。

        对每个 doc_id 单独调用 ragas 生成器，然后合并结果。

        Args:
            doc_ids: 目标文档 ID 列表
            testset_size_per_doc: 每个文档生成测试用例数量
            chunk_type_filter: 可选，按 chunk_type 过滤

        Returns:
            合并后的测试用例列表
        """
        all_cases = []
        for doc_id in doc_ids:
            cases = self.generate_from_doc(
                doc_id=doc_id,
                testset_size=testset_size_per_doc,
                chunk_type_filter=chunk_type_filter,
            )
            all_cases.extend(cases)

        logger.info(
            "ragas_testset_multi_doc_generated",
            doc_count=len(doc_ids),
            total_cases=len(all_cases),
        )
        return all_cases

    # ============================================================
    # 内部方法
    # ============================================================

    def _default_llm_config(self) -> dict:
        """从 settings 获取默认 LLM 配置。"""
        from app.config import settings
        return {
            "model_name": getattr(settings, "EVAL_MODEL_NAME", "gpt-4o-mini"),
            "api_key": getattr(settings, "EVAL_API_KEY", "") or getattr(settings, "DASHSCOPE_API_KEY", ""),
            "base_url": getattr(settings, "EVAL_BASE_URL", "") or getattr(settings, "VLM_API_URL", ""),
        }

    def _default_embed_config(self) -> dict:
        """从 settings 获取默认 Embedding 配置。"""
        from app.config import settings
        base_url = getattr(settings, "EVAL_BASE_URL", "") or getattr(settings, "VLM_API_URL", "")
        return {
            "model_name": getattr(settings, "EMBED_MODEL_NAME", "text-embedding-v3"),
            "api_key": getattr(settings, "DASHSCOPE_API_KEY", "") or getattr(settings, "EVAL_API_KEY", ""),
            "base_url": base_url,
        }

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        """将 OpenAI 兼容端点标准化为 base URL（不含路径后缀）。

        langchain_openai 会自动追加 /chat/completions 或 /embeddings，
        所以 base_url 不应包含这些后缀。
        """
        base_url = base_url.rstrip("/")
        # 移除常见的 OpenAI 兼容端点后缀
        for suffix in (
            "/v1/chat/completions",
            "/v1/embeddings",
            "/chat/completions",
            "/compatible-mode/v1",
            "/v1",
        ):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break
        return base_url + "/v1"

    def _get_generator(self):
        """获取或创建 TestsetGenerator（lazy init + 缓存）。"""
        if self._generator is not None:
            return self._generator

        from langchain_openai import ChatOpenAI

        from ragas.embeddings import BaseRagasEmbeddings
        from ragas.testset import TestsetGenerator

        # LLM
        llm_cfg = self._llm_config
        llm_api_key = llm_cfg.get("api_key", "")
        llm_base_url = self._normalize_base_url(llm_cfg.get("base_url", ""))

        llm = ChatOpenAI(model=llm_cfg["model_name"], api_key=llm_api_key, base_url=llm_base_url)

        # Embedding
        if self._use_dashscope:
            embeddings = DashScopeEmbedding(self._embed_config)
        else:
            from langchain_openai import OpenAIEmbeddings

            emb_cfg = self._embed_config
            emb_api_key = emb_cfg.get("api_key", llm_api_key)
            emb_base_url = self._normalize_base_url(emb_cfg.get("base_url", llm_cfg.get("base_url", "")))
            embeddings = OpenAIEmbeddings(model=emb_cfg["model_name"], api_key=emb_api_key, base_url=emb_base_url)

        self._generator = TestsetGenerator.from_langchain(
            llm=llm,
            embedding_model=embeddings,
        )
        logger.info(
            "ragas_testset_generator_initialized",
            llm_model=llm_cfg["model_name"],
            embed_model=self._embed_config["model_name"],
            embed_backend="dashscope" if self._use_dashscope else "openai_compat",
            base_url=llm_base_url,
        )
        return self._generator
