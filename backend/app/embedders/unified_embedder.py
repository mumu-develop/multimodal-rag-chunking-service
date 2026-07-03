"""
统一多模态嵌入器模块

提供文本和图片的向量编码能力，使用 DashScope 多模态 API。

模型：qwen3-vl-embedding
输出维度：2560（统一向量）

特点：
- 无需本地加载模型，通过 API 调用
- 支持纯文本、纯图片、文本+图片联合编码
- 支持批量编码（并发调用）
- 内置重试机制（应对 API 瞬时错误）
"""
import base64
from pathlib import Path

import structlog

from app.config import settings

logger = structlog.get_logger()


def image_to_base64_data_uri(image_path: str) -> str | None:
    """
    将图片转为 data URI 格式（用于 API 传输）。

    业务场景：
    - 本地图片文件需要转为 base64 才能通过 HTTP 传输
    - DashScope API 支持 data URI 格式的图片输入

    流程步骤：
    1. 判断输入类型（HTTP URL 或本地路径）
    2. HTTP URL：直接读取网络图片数据
    3. 本地路径：读取文件数据
    4. 根据扩展名确定 MIME 类型
    5. 构造 data URI：data:{mime};base64,{base64_data}
    6. 返回 data URI 字符串（失败返回 None）

    参数：
        image_path: 图片路径（本地文件或 HTTP URL）

    返回：
        str: data URI 格式字符串，如 "data:image/png;base64,iVBORw0KGgo..."
        失败时返回 None
    """
    import urllib.request

    try:
        # 步骤1-2：根据路径类型读取数据
        if image_path.startswith(("http://", "https://")):
            # HTTP URL：通过 urllib 读取网络图片
            # - timeout=30 防止长时间阻塞
            with urllib.request.urlopen(image_path, timeout=30) as resp:
                data = resp.read()
        else:
            # 本地路径：直接读取文件
            data = Path(image_path).read_bytes()

        # 步骤3-4：确定 MIME 类型
        # - 获取文件扩展名（如 .png, .jpg）
        # - 映射到标准 MIME 类型
        ext = Path(image_path).suffix.lower().lstrip(".")
        mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp", "bmp": "bmp"}
        mime = f"image/{mime_map.get(ext, 'jpeg')}"

        # 步骤5：构造 data URI
        # - 格式：data:{mime};base64,{base64_encoded_data}
        # - base64.b64encode 返回 bytes，decode() 转为字符串
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        # 步骤6：失败时记录异常并返回 None
        logger.exception("image_to_base64_failed", path=image_path)
        return None


class UnifiedEmbedder:
    """
    统一多模态嵌入器。

    使用 DashScope MultiModalEmbedding API（qwen3-vl-embedding 模型）。
    输出维度：2560（统一向量，支持文本+图片融合）。

    流程概述：
    1. 惰性初始化 DashScope API Key
    2. 根据输入类型构造 API 请求
    3. 调用 API 获取向量
    4. 返回向量结果

    支持的编码方式：
    - encode_query: 单文本查询编码
    - encode_image_query: 单图片查询编码
    - encode_multimodal: 文本+图片联合编码
    - encode_texts: 批量文本编码
    - encode_images: 批量图片编码
    - encode_multimodal_batch: 批量联合编码
    """

    def __init__(self):
        """
        初始化嵌入器。

        步骤：
        1. 设置加载标记为 False（惰性初始化）
        """
        # 步骤1：惰性初始化标记
        # - _loaded 在首次调用 _ensure_loaded() 时才设为 True
        # - 避免启动时就初始化 API Key（可能未配置）
        self._loaded = False

    def _ensure_loaded(self):
        """
        惰性初始化 DashScope（内部方法）。

        流程步骤：
        1. 检查是否已加载，已加载则直接返回
        2. 导入 dashscope 模块
        3. 设置 API Key（从 settings 读取）
        4. 设置加载标记
        5. 记录初始化日志

        注意：
        - API Key 在环境变量 DASHSCOPE_API_KEY 中配置
        - 阿里云 DashScope 服务需要有效 API Key
        """
        # 步骤1：检查是否已加载
        if self._loaded:
            return

        # 步骤2-3：导入并配置 API Key
        import dashscope
        dashscope.api_key = settings.DASHSCOPE_API_KEY

        # 步骤4：设置加载标记
        self._loaded = True

        # 步骤5：记录初始化日志
        logger.info(
            "unified_embedder_loaded",
            model=settings.EMBED_MODEL_NAME,
            dim=settings.UNIFIED_VECTOR_DIM,
        )

    def _prepare_image_input(self, image_ref: str) -> str:
        """
        将图片引用转为 DashScope 可接受格式（内部方法）。

        DashScope 支持的图片格式：
        - data URI（base64）：data:image/png;base64,...
        - HTTP URL：https://example.com/image.jpg

        流程步骤：
        1. 检查是否为 data URI（已编码）
        2. 检查是否为 HTTP URL（可直接使用）
        3. 本地文件：调用 image_to_base64_data_uri 转换
        4. 返回可用的图片引用

        参数：
            image_ref: 图片引用（路径/URL/data URI）

        返回：
            str: DashScope 可接受的图片格式

        抛出：
            RuntimeError: 图片文件无法读取
        """
        # 步骤1：data URI 直接返回（已编码）
        if image_ref.startswith("data:"):
            return image_ref

        # 步骤2：HTTP URL 直接返回（DashScope 支持）
        if image_ref.startswith(("http://", "https://")):
            return image_ref

        # 步骤3：本地文件转为 data URI
        # - image_to_base64_data_uri 失败时返回 None
        data_uri = image_to_base64_data_uri(image_ref)
        if data_uri is None:
            raise RuntimeError(f"无法读取图片文件: {image_ref}")

        return data_uri

    def _call_api(self, input_list: list[dict]) -> list[list[float]]:
        """
        调用 DashScope MultiModalEmbedding API（内部方法）。

        业务场景：
        - 批量编码切片（Celery 任务调用）
        - 编码查询向量（搜索请求调用）

        流程步骤：
        1. 确保 DashScope 已初始化
        2. 确定并发线程数（最多10路）
        3. 定义单次编码函数 _embed_one：
           a. 处理图片引用（转 data URI）
           b. 文本截断保护（>20000 字符截断）
           c. 调用 dashscope.MultiModalEmbedding.call
           d. 重试机制（最多3次，指数等待）
           e. 解析响应获取 embedding
        4. 使用 ThreadPoolExecutor 并发编码
        5. 返回向量列表

        参数：
            input_list: 输入列表，每个元素为 dict：
                - {"text": "文本内容"}
                - {"image": "图片路径/URL"}
                - {"text": "文本", "image": "图片"}

        返回：
            list[list[float]]: 向量列表，每个向量2560维
        """
        import concurrent.futures
        import time

        import dashscope

        # 步骤1：确保初始化
        self._ensure_loaded()

        # 步骤2：确定并发线程数
        # - 最多10路并发（DashScope QPS 限制）
        # - 输入数量少于10时按实际数量
        max_workers = min(10, len(input_list)) if input_list else 1

        def _embed_one(inp):
            """
            编码单个输入（内部函数）。

            流程：
            1. 处理图片引用（转 data URI）
            2. 文本截断（保护超长文本）
            3. 调用 DashScope API
            4. 重试机制（应对瞬时错误）
            5. 返回 embedding 向量
            """
            # 步骤1：处理图片引用
            # - 本地路径/URL → data URI（如果需要）
            if "image" in inp:
                inp = {**inp, "image": self._prepare_image_input(inp["image"])}

            # 步骤2：文本截断保护
            # - DashScope 文本限制约 20000 字符
            # - 超长文本截断并记录警告
            if "text" in inp and len(inp["text"]) > 20000:
                inp = {**inp, "text": inp["text"][:20000]}
                logger.warning("text_truncated_for_embedding", length=len(inp["text"]))

            # 步骤3-4：调用 API 并重试
            max_retries = 3
            for attempt in range(max_retries):
                # 调用 DashScope MultiModalEmbedding API
                # - api_key: 认证
                # - model: qwen3-vl-embedding
                # - input: 单个输入（list格式）
                resp = dashscope.MultiModalEmbedding.call(
                    api_key=settings.DASHSCOPE_API_KEY,
                    model=settings.EMBED_MODEL_NAME,
                    input=[inp],
                )

                # 检查响应状态
                if resp.status_code == 200 and resp.output is not None:
                    break  # 成功则跳出重试循环

                # 记录失败日志
                logger.error(
                    "dashscope_embedding_failed",
                    code=resp.status_code,
                    message=resp.message,
                    attempt=attempt + 1,
                    text_len=len(inp.get("text", "")),
                    has_image="image" in inp,
                )

                # 重试等待（指数退避：1s, 2s, 4s）
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)

            # 重试耗尽仍失败则抛出异常
            if resp.status_code != 200 or resp.output is None:
                raise RuntimeError(
                    f"DashScope API failed after {max_retries} attempts: "
                    f"code={resp.status_code}, message={resp.message}"
                )

            # 步骤5：解析响应获取 embedding
            # - resp.output["embeddings"] 是列表
            # - 取第一个（因为 input 是单个）
            embeddings = resp.output.get("embeddings")
            if not embeddings:
                logger.error("dashscope_no_embedding", output=resp.output)
                raise RuntimeError("DashScope API returned no embeddings")

            return embeddings[0]["embedding"]

        # 步骤4：并发编码
        # - ThreadPoolExecutor 并发调用 _embed_one
        # - pool.map 返回结果列表
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            vectors = list(pool.map(_embed_one, input_list))

        # 步骤5：返回向量列表
        return vectors

    def _encode_inputs(
        self,
        inputs: list[dict],
        is_single: bool = False,
    ) -> list[list[float]] | list[float]:
        """
        编码输入列表（内部方法）。

        流程步骤：
        1. 调用 _call_api 获取向量列表
        2. 根据 is_single 决定返回格式：
           - True：返回单个向量（查询编码）
           - False：返回向量列表（批量编码）

        参数：
            inputs: 格式化的输入列表
            is_single: 是否单输入模式

        返回：
            单向量或向量列表
        """
        # 步骤1：调用 API 编码
        vectors = self._call_api(inputs)

        # 步骤2：根据模式返回
        if is_single:
            return vectors[0]  # 单个向量
        return vectors  # 向量列表

    def encode_texts(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """
        批量编码文本列表。

        业务场景：
        - Celery 任务批量编码 text/table 切片

        流程步骤：
        1. 检查输入是否为空
        2. 构造输入列表 [{"text": t1}, {"text": t2}, ...]
        3. 调用 _encode_inputs 获取向量

        参数：
            texts: 文本字符串列表
            batch_size: 批次大小（暂未使用，API 自动并发）

        返回：
            list[list[float]]: 向量列表
        """
        # 步骤1：空输入返回空列表
        if not texts:
            return []

        # 步骤2：构造输入格式
        inputs = [{"text": t} for t in texts]

        # 步骤3：编码并返回
        return self._encode_inputs(inputs)

    def encode_images(self, image_paths: list[str], batch_size: int | None = None) -> list[list[float]]:
        """
        批量编码图片列表。

        业务场景：
        - Celery 任务批量编码 image 切片

        流程步骤：
        1. 检查输入是否为空
        2. 构造输入列表 [{"image": path1}, {"image": path2}, ...]
        3. 调用 _encode_inputs 获取向量

        参数：
            image_paths: 图片路径列表（本地路径或 HTTP URL）
            batch_size: 批次大小（暂未使用）

        返回：
            list[list[float]]: 向量列表
        """
        # 步骤1：空输入返回空列表
        if not image_paths:
            return []

        # 步骤2：构造输入格式
        inputs = [{"image": str(p)} for p in image_paths]

        # 步骤3：编码并返回
        return self._encode_inputs(inputs)

    def encode_query(self, query: str) -> list[float]:
        """
        编码单个文本查询。

        业务场景：
        - 搜索请求时编码用户输入的文本查询

        流程步骤：
        1. 构造单输入 [{"text": query}]
        2. 调用 _encode_inputs，is_single=True
        3. 返回单个向量

        参数：
            query: 用户查询文本

        返回：
            list[float]: 2560维向量
        """
        return self._encode_inputs([{"text": query}], is_single=True)

    def encode_image_query(self, image_path: str) -> list[float]:
        """
        编码单张图片作为查询向量。

        业务场景：
        - 用户上传图片进行图片检索

        流程步骤：
        1. 构造单输入 [{"image": image_path}]
        2. 调用 _encode_inputs，is_single=True
        3. 返回单个向量

        参数：
            image_path: 图片路径（本地路径或 HTTP URL）

        返回：
            list[float]: 2560维向量
        """
        return self._encode_inputs([{"image": str(image_path)}], is_single=True)

    def encode_multimodal(self, text: str, image_path: str) -> list[float]:
        """
        联合编码文本和图片为单个多模态向量。

        业务场景：
        - 用户同时输入文本和图片进行联合检索
        - 文本和图片共同决定向量表示

        流程步骤：
        1. 构造联合输入 [{"text": text, "image": image_path}]
        2. 调用 _encode_inputs，is_single=True
        3. 返回单个融合向量

        参数：
            text: 文本内容
            image_path: 图片路径（本地路径/HTTP URL/data URI）

        返回：
            list[float]: 2560维融合向量
        """
        return self._encode_inputs(
            [{"text": text, "image": str(image_path)}],
            is_single=True,
        )

    def encode_multimodal_batch(
        self,
        items: list[tuple[str, str | None]],
    ) -> list[list[float]]:
        """
        批量联合编码文本+图片。

        业务场景：
        - Celery 任务批量编码切片
        - text 切片：只有文本
        - image 切片：有文本（VLM caption）+图片路径

        流程步骤：
        1. 检查输入是否为空
        2. 遍历 items，根据是否有图片构造输入：
           - 有图片：{"text": text, "image": path}
           - 无图片：{"text": text}
        3. 调用 _encode_inputs 获取向量列表

        参数：
            items: 元组列表，每个元素为 (text, image_path)
                   - image_path 为 None 表示纯文本
                   - image_path 有值表示文本+图片

        返回：
            list[list[float]]: 向量列表
        """
        # 步骤1：空输入返回空列表
        if not items:
            return []

        # 步骤2：构造输入格式
        inputs = []
        for text, image_path in items:
            if image_path:
                # 文本+图片：联合编码
                inputs.append({"text": text, "image": str(image_path)})
            else:
                # 纯文本：仅编码文本
                inputs.append({"text": text})

        # 步骤3：编码并返回
        return self._encode_inputs(inputs)


# 单例模式：全局共享一个 UnifiedEmbedder 实例
# - 避免重复初始化 DashScope
# - API Key 只需设置一次
unified_embedder = UnifiedEmbedder()