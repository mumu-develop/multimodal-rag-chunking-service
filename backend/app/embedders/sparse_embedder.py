"""Query-side sparse embedding for BM25 search.

Uses Milvus's run_analyzer to tokenize queries with the same analyzer
that the BM25 Function uses on the corpus side.
"""
import math

import structlog

logger = structlog.get_logger()


class SparseEmbedder:
    """
    稀疏向量编码器，用于 BM25 搜索的查询侧。

    调用 Milvus 的 run_analyzer 进行分词，确保与 BM25 Function
    生成的 sparse_vector 使用相同的 term ID 映射。
    """

    def __init__(self):
        self._milvus_client = None

    def _get_milvus_client(self):
        """惰性获取 Milvus 客户端。"""
        if self._milvus_client is not None:
            return self._milvus_client
        try:
            from app.repositories.milvus_repo import milvus_repo
            self._milvus_client = milvus_repo
            return self._milvus_client
        except Exception:
            return None

    def encode_sparse(self, text: str) -> dict:
        """
        将查询文本转为稀疏向量格式，用于 Milvus BM25 搜索。

        使用 Milvus 的 run_analyzer 分词，确保 term ID 与 corpus 一致。
        返回 {term_hash: tf_weight} 字典。

        如果 Milvus 不可用，fallback 到本地 jieba 分词。
        """
        if not text or not text.strip():
            return {}

        # 尝试使用 Milvus analyzer
        client_wrapper = self._get_milvus_client()
        if client_wrapper:
            try:
                client = client_wrapper._get_client()
                tokens_result = client.run_analyzer(
                    text,
                    analyzer_params={"type": "chinese"},
                    with_hash=True,
                    with_detail=False,
                )
                # tokens_result is a list of token dicts with 'token' and optionally 'hash'
                if tokens_result:
                    from collections import Counter
                    # tokens_result might be list[str] or list[dict] depending on with_detail
                    if isinstance(tokens_result[0], dict):
                        token_counts = Counter()
                        for item in tokens_result:
                            tok = item.get("token", "")
                            h = item.get("hash", None)
                            if tok:
                                token_counts[h if h is not None else hash(tok) & 0xFFFFFFFF] += 1
                    else:
                        token_counts = Counter(tokens_result)

                    # Compute TF weights: count / (count + 1.0) (BM25-style normalization)
                    sparse = {}
                    for idx, count in token_counts.items():
                        tf_norm = count / (count + 1.0)
                        sparse[int(idx)] = tf_norm

                    if sparse:
                        return sparse
            except Exception:
                logger.debug("milvus_analyzer_failed_fallback")

        # Fallback: local jieba tokenization
        return self._encode_local(text)

    def _encode_local(self, text: str) -> dict:
        """本地 jieba 分词 fallback。"""
        try:
            import jieba
            tokens = list(jieba.cut(text))
        except ImportError:
            tokens = list(text)

        if not tokens:
            return {}

        from collections import Counter
        tf = Counter(tokens)

        sparse = {}
        for token, count in tf.items():
            # Use same hash approach as Milvus's analyzer would
            h = hash(token) & 0xFFFFFFFF
            tf_norm = count / (count + 1.0)
            sparse[int(h)] = tf_norm

        return sparse


# 单例
sparse_embedder = SparseEmbedder()
